"""Stage 5 -- transition stitching (video compose).

Stitches the shot clips with ffmpeg according to each decision's transition_out:
  - hard cut / wipe-by (already joined at the first frame during generation) -> plain concat
  - dissolve / flash-to-white / flash-to-black / wipe -> xfade
  - J-cut / L-cut -> the picture and audio timelines are separated and offset by timing_offset_seconds
  - a clip without an audio track gets a silent one, so the filters stay aligned

Strategy: normalise every clip to one resolution/frame rate/sample rate/codec; compose the picture
left to right by net duration and optical effect; remix the audio from the headroom-preserving full clips along the J/L/straight timeline; finally force the target duration.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.frame_utils import has_audio_stream, probe_duration

logger = logging.getLogger("compose")

# Resolution tier + aspect ratio -> pixel size (matching WAN output)
_RES_WH = {
    ("720P", "16:9"): (1280, 720),
    ("720P", "9:16"): (720, 1280),
    ("720P", "1:1"): (960, 960),
    ("720P", "4:3"): (1104, 832),
    ("720P", "3:4"): (832, 1104),
    ("1080P", "16:9"): (1920, 1080),
    ("1080P", "9:16"): (1080, 1920),
    ("1080P", "1:1"): (1440, 1440),
    ("1080P", "4:3"): (1648, 1248),
    ("1080P", "3:4"): (1248, 1648),
}

_SAMPLE_RATE = 48000
# One render duration for optical effects (dissolve/flash black/flash white/wipe), matching
# lib.transition_map.TRANSITION_EFFECT_SECONDS; used as a fallback when transition_duration_seconds
# is missing or 0.
_DEFAULT_XFADE_DUR = 0.40

_XFADE_AVAILABLE: Optional[bool] = None  # runtime detection cache


def resolution_to_wh(resolution: str, ratio: str) -> tuple:
    return _RES_WH.get((resolution, ratio), (1280, 720))


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("ffmpeg failed:\n%s", r.stderr[-800:])
    return r


def _duration_arg(duration: Optional[float]) -> List[str]:
    if duration is None or duration <= 0:
        return []
    return ["-t", f"{float(duration):.3f}"]


def _has_xfade() -> bool:
    """Detect whether the current ffmpeg supports the xfade filter (>=4.3); the result is cached."""
    global _XFADE_AVAILABLE
    if _XFADE_AVAILABLE is None:
        try:
            r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                               capture_output=True, text=True)
            _XFADE_AVAILABLE = (" xfade " in r.stdout)
        except Exception:  # noqa: BLE001
            _XFADE_AVAILABLE = False
        logger.info("xfade availability: %s", _XFADE_AVAILABLE)
    return _XFADE_AVAILABLE


def _audio_cleanup_filter(audio_cleanup_cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Build a conservative per-shot audio cleanup filter."""
    cfg = audio_cleanup_cfg or {}
    if not cfg.get("enabled", False):
        return None
    filters = []
    highpass_hz = float(cfg.get("highpass_hz", 80) or 0)
    if highpass_hz > 0:
        filters.append(f"highpass=f={highpass_hz:g}")
    if str(cfg.get("denoise", "off")).lower() in ("light", "true", "yes", "afftdn"):
        filters.append("afftdn=nf=-25")
    target_lufs = float(cfg.get("target_lufs", -18))
    filters.append(f"loudnorm=I={target_lufs:g}:LRA=11:TP=-1.5")
    filters.append("alimiter=limit=0.95")
    return ",".join(filters)


def normalize_clip(inp: str, out: str, w: int, h: int, fps: int,
                   target_duration: Optional[float] = None,
                   audio_cleanup_cfg: Optional[Dict[str, Any]] = None) -> str:
    """Normalise a clip to the shared parameters, optionally trimming it to an exact target duration."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p"
    )
    duration_args = _duration_arg(target_duration)
    af = _audio_cleanup_filter(audio_cleanup_cfg)
    if has_audio_stream(inp):
        cmd = [
            "ffmpeg", "-y", "-i", inp,
            "-vf", vf,
        ]
        if af:
            cmd.extend(["-af", af])
        cmd.extend([
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-ar", str(_SAMPLE_RATE), "-ac", "2",
            *duration_args,
            out,
        ])
    else:
        # add a silent audio track
        cmd = [
            "ffmpeg", "-y", "-i", inp,
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={_SAMPLE_RATE}",
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-ar", str(_SAMPLE_RATE), "-ac", "2",
            *duration_args,
            "-shortest",
            out,
        ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"failed to normalize clip: {inp}\n{r.stderr[-500:]}")
    return out


def _concat_pair(a: str, b: str, out: str) -> str:
    """Hard-cut two segments together (filter concat, safe under shared parameters)."""
    cmd = [
        "ffmpeg", "-y", "-i", a, "-i", b,
        "-filter_complex",
        "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", str(_SAMPLE_RATE),
        out,
    ]
    _run(cmd)
    return out


def _xfade_pair(a: str, b: str, out: str, transition: str, dur: float) -> str:
    """Cross-transition two segments. Prefers xfade (ffmpeg>=4.3), otherwise a manual
    blend/fade implementation (compatible with ffmpeg 4.2); falls back to a hard cut if both fail."""
    dur_a = probe_duration(a) or 0
    dur_b = probe_duration(b) or 0
    # the transition cannot be longer than either segment
    dur = max(0.1, min(dur, dur_a - 0.05, dur_b - 0.05))
    offset = max(0.0, dur_a - dur)

    if _has_xfade():
        cmd = [
            "ffmpeg", "-y", "-i", a, "-i", b,
            "-filter_complex",
            f"[0:v][1:v]xfade=transition={transition}:duration={dur:.3f}:offset={offset:.3f}[v];"
            f"[0:a][1:a]acrossfade=d={dur:.3f}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-ar", str(_SAMPLE_RATE),
            out,
        ]
        if _run(cmd).returncode == 0:
            return out
        logger.warning("xfade failed, trying the manual transition")

    # ---- manual transitions compatible with ffmpeg 4.2 ----
    if transition in ("fadewhite", "fadeblack"):
        ok = _flash_pair(a, b, out, dur, dur_a, color="white" if transition == "fadewhite" else "black")
    else:
        # fade (dissolve) and wipe are both approximated with a cross-dissolve
        ok = _dissolve_pair(a, b, out, dur, dur_a, dur_b)
    if ok:
        return out
    logger.warning("manual transition failed, falling back to a hard cut")
    return _concat_pair(a, b, out)


def _dissolve_pair(a: str, b: str, out: str, dur: float, dur_a: float, dur_b: float) -> bool:
    """Cross-dissolve (blend): A[0:dur_a-D] + blended overlap(D) + B[D:dur_b], total = dur_a+dur_b-D.
    The audio uses acrossfade (same overlap D) to match the total length."""
    a_keep = max(0.0, dur_a - dur)
    fc = (
        f"[0:v]trim=0:{a_keep:.3f},setpts=PTS-STARTPTS[a0];"
        f"[0:v]trim={a_keep:.3f}:{dur_a:.3f},setpts=PTS-STARTPTS[a1];"
        f"[1:v]trim=0:{dur:.3f},setpts=PTS-STARTPTS[b0];"
        f"[1:v]trim={dur:.3f}:{dur_b:.3f},setpts=PTS-STARTPTS[b1];"
        f"[a1][b0]blend=all_expr='A*(1-T/{dur:.3f})+B*(T/{dur:.3f})'[tr];"
        f"[a0][tr][b1]concat=n=3:v=1:a=0[v];"
        f"[0:a][1:a]acrossfade=d={dur:.3f}[a]"
    )
    cmd = [
        "ffmpeg", "-y", "-i", a, "-i", b, "-filter_complex", fc,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", str(_SAMPLE_RATE), out,
    ]
    return _run(cmd).returncode == 0


def _flash_pair(a: str, b: str, out: str, dur: float, dur_a: float, color: str = "white") -> bool:
    """Flash white/black: A fades to the colour over its last D, B fades in from it over its first D, then plain concat.
    Total = dur_a + dur_b (no overlap); the audio is concatenated to match."""
    st = max(0.0, dur_a - dur)
    fc = (
        f"[0:v]fade=t=out:st={st:.3f}:d={dur:.3f}:color={color}[a0];"
        f"[1:v]fade=t=in:st=0:d={dur:.3f}:color={color}[b0];"
        f"[a0][b0]concat=n=2:v=1:a=0[v];"
        f"[0:a][1:a]concat=n=2:v=0:a=1[a]"
    )
    cmd = [
        "ffmpeg", "-y", "-i", a, "-i", b, "-filter_complex", fc,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", str(_SAMPLE_RATE), out,
    ]
    return _run(cmd).returncode == 0


def add_global_bgm(
    video_path: str,
    bgm_path: str,
    output_path: str,
    volume: float = 0.18,
    fade_in: float = 1.0,
    fade_out: float = 1.5,
    loudnorm: bool = False,
) -> str:
    """Mix one global BGM into the final cut."""
    total_dur = probe_duration(video_path) or 0
    if total_dur <= 0:
        raise RuntimeError(f"cannot read the cut duration: {video_path}")
    if not Path(bgm_path).exists():
        raise FileNotFoundError(f"BGM file does not exist: {bgm_path}")

    fade_in = max(0.0, min(float(fade_in), total_dur / 2))
    fade_out = max(0.0, min(float(fade_out), total_dur / 2))
    fade_out_start = max(0.0, total_dur - fade_out)
    music_filters = [
        f"atrim=0:{total_dur:.3f}",
        "asetpts=PTS-STARTPTS",
        f"aformat=sample_fmts=fltp:sample_rates={_SAMPLE_RATE}:channel_layouts=stereo",
        f"volume={volume}",
    ]
    if fade_in > 0:
        music_filters.append(f"afade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        music_filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}")

    premix_label = "premix"
    fc = (
        f"[0:a]aformat=sample_fmts=fltp:sample_rates={_SAMPLE_RATE}:channel_layouts=stereo[base];"
        f"[1:a]{','.join(music_filters)}[music];"
        f"[base][music]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[{premix_label}]"
    )
    out_label = premix_label
    if loudnorm:
        fc += f";[{premix_label}]loudnorm=I=-16:LRA=11:TP=-1.5[aout]"
        out_label = "aout"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-stream_loop", "-1", "-i", bgm_path,
        "-filter_complex", fc,
        "-map", "0:v:0", "-map", f"[{out_label}]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(_SAMPLE_RATE),
        "-shortest",
        output_path,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"global BGM mixing failed: {r.stderr[-500:]}")
    return output_path


def _shot_timeline(decisions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Estimate each shot's time span in the final cut from the net durations."""
    cursor = 0.0
    timeline: Dict[str, Dict[str, float]] = {}
    for dec in decisions:
        shot_id = str(dec.get("shot_id"))
        dur = float(dec.get("net_duration", dec.get("duration", 0)) or 0)
        if dur <= 0:
            continue
        timeline[shot_id] = {"start": cursor, "end": cursor + dur}
        cursor += dur
    return timeline


def _merge_segments(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not segments:
        return []
    merged: List[Dict[str, float]] = []
    for seg in sorted(segments, key=lambda s: s["start"]):
        if not merged or seg["start"] > merged[-1]["end"] + 0.05:
            merged.append(dict(seg))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
    return merged


def _segments_for_bed(bed: Dict[str, Any], decisions: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    timeline = _shot_timeline(decisions)
    applies = {str(x) for x in bed.get("applies_to_shots", [])}
    segments = [timeline[sid] for sid in applies if sid in timeline]
    return _merge_segments(segments)


def _mix_one_scene_audio_bed(
    video_path: str,
    bed: Dict[str, Any],
    decisions: List[Dict[str, Any]],
    output_path: str,
) -> Dict[str, Any]:
    """Mix one scene-level diegetic audio bed into the cut over its shot spans."""
    source_path = bed.get("source_path")
    segments = _segments_for_bed(bed, decisions)
    report = {
        "id": bed.get("id"),
        "type": bed.get("type"),
        "source_path": source_path,
        "segments": segments,
        "mixed": False,
    }
    if not source_path or not Path(source_path).exists():
        report["reason"] = "source_path missing or not found"
        return report
    if not segments:
        report["reason"] = "no matching shot segments"
        return report

    total_dur = probe_duration(video_path) or 0
    if total_dur <= 0:
        raise RuntimeError(f"cannot read the cut duration: {video_path}")
    mix_rule = bed.get("mix_rule") or {}
    volume = float(mix_rule.get("volume", 0.35))
    fade_in = float(mix_rule.get("fade_in_seconds", 0.2))
    fade_out = float(mix_rule.get("fade_out_seconds", 0.4))
    parts = []
    for seg in segments:
        s = max(0.0, float(seg["start"]))
        e = min(total_dur, float(seg["end"]))
        if e <= s:
            continue
        fin = min(fade_in, max((e - s) / 2, 0.0))
        fout = min(fade_out, max((e - s) / 2, 0.0))
        fade_in_end = s + fin
        fade_out_start = e - fout
        if fin <= 0 and fout <= 0:
            parts.append(f"if(between(t,{s:.3f},{e:.3f}),{volume},0)")
        else:
            fin = max(fin, 0.001)
            fout = max(fout, 0.001)
            parts.append(
                f"if(lt(t,{s:.3f}),0,"
                f"if(lt(t,{fade_in_end:.3f}),{volume}*(t-{s:.3f})/{fin:.3f},"
                f"if(lt(t,{fade_out_start:.3f}),{volume},"
                f"if(lt(t,{e:.3f}),{volume}*({e:.3f}-t)/{fout:.3f},0))))"
            )
    if not parts:
        report["reason"] = "segments collapsed after duration clamp"
        return report

    vol_expr = "+".join(f"({p})" for p in parts)
    fc = (
        f"[0:a]aformat=sample_fmts=fltp:sample_rates={_SAMPLE_RATE}:channel_layouts=stereo[base];"
        f"[1:a]atrim=0:{total_dur:.3f},asetpts=PTS-STARTPTS,"
        f"aformat=sample_fmts=fltp:sample_rates={_SAMPLE_RATE}:channel_layouts=stereo,"
        f"volume='{vol_expr}':eval=frame[bed];"
        f"[base][bed]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-stream_loop", "-1", "-i", source_path,
        "-filter_complex", fc,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(_SAMPLE_RATE),
        "-shortest", output_path,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"scene diegetic audio bed mixing failed: {r.stderr[-500:]}")
    report["mixed"] = True
    report["output_path"] = output_path
    return report


def add_scene_audio_beds(
    video_path: str,
    scene_audio_beds: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    output_path: str,
    work_dir: str,
) -> Dict[str, Any]:
    """Mix the continuous diegetic sound sources described by scene_audio_beds into the cut."""
    reports = []
    current = video_path
    mix_dir = Path(work_dir) / "scene_audio_beds"
    mix_dir.mkdir(parents=True, exist_ok=True)
    active_beds = [bed for bed in scene_audio_beds or [] if bed.get("type") == "diegetic_music"]
    for idx, bed in enumerate(active_beds):
        is_last = idx == len(active_beds) - 1
        out = output_path if is_last else str(mix_dir / f"scene_bed_{idx:02d}.mp4")
        rep = _mix_one_scene_audio_bed(current, bed, decisions, out)
        reports.append(rep)
        if rep.get("mixed"):
            current = out
    if current != output_path:
        r = _run(["ffmpeg", "-y", "-i", current, "-c", "copy", output_path])
        if r.returncode != 0:
            raise RuntimeError(f"failed to write the scene audio bed cut: {r.stderr[-500:]}")
    return {"enabled": bool(active_beds), "input_path": video_path, "output_path": output_path, "beds": reports}


def _parse_volumedetect(stderr: str) -> Dict[str, Optional[float]]:
    mean_match = re.search(r"mean_volume:\s*([-\d.]+) dB", stderr or "")
    max_match = re.search(r"max_volume:\s*([-\d.]+) dB", stderr or "")
    return {
        "mean_volume_db": float(mean_match.group(1)) if mean_match else None,
        "max_volume_db": float(max_match.group(1)) if max_match else None,
    }


def measure_audio_volume(
    video_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Roughly measure the loudness of a whole file or a span with volumedetect, for silence / low-loudness compensation."""
    if start is not None and end is not None and end > start:
        af = f"atrim=start={float(start):.3f}:end={float(end):.3f},asetpts=PTS-STARTPTS,volumedetect"
    else:
        af = "volumedetect"
    cmd = ["ffmpeg", "-hide_banner", "-i", video_path, "-af", af, "-f", "null", "-"]
    r = _run(cmd)
    values = _parse_volumedetect(r.stderr)
    values["start"] = start
    values["end"] = end
    return values


def _segments_from_decisions(decisions: List[Dict[str, Any]], total_dur: float) -> List[Dict[str, float]]:
    cursor = 0.0
    segments: List[Dict[str, float]] = []
    for dec in decisions:
        dur = float(dec.get("net_duration", dec.get("duration", 0)) or 0)
        if dur <= 0:
            continue
        start = cursor
        end = min(total_dur, cursor + dur)
        if end > start:
            segments.append({"shot_id": dec.get("shot_id"), "start": start, "end": end})
        cursor += dur
    return segments


def add_room_tone_floor(
    video_path: str,
    decisions: List[Dict[str, Any]],
    output_path: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Mix continuous diegetic room tone into low-loudness / near-silent shots, avoiding audio cliffs between shots."""
    tone_cfg = cfg or {}
    report: Dict[str, Any] = {
        "enabled": bool(tone_cfg.get("enabled", False)),
        "input_path": video_path,
        "output_path": output_path,
        "segments": [],
        "mixed": False,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not report["enabled"]:
        r = _run(["ffmpeg", "-y", "-i", video_path, "-c", "copy", output_path])
        if r.returncode != 0:
            raise RuntimeError(f"failed to write the room-tone-skipped cut: {r.stderr[-500:]}")
        report["reason"] = "disabled"
        return report

    total_dur = probe_duration(video_path) or 0
    if total_dur <= 0:
        raise RuntimeError(f"cannot read the cut duration: {video_path}")

    threshold_db = float(tone_cfg.get("low_mean_threshold_db", -45))
    max_threshold_db = float(tone_cfg.get("low_max_threshold_db", -30))
    floor_volume = float(tone_cfg.get("floor_volume", 0.008))
    low_segment_volume = float(tone_cfg.get("low_segment_volume", 0.045))
    fade_seconds = float(tone_cfg.get("fade_seconds", 0.15))
    noise_color = str(tone_cfg.get("noise_color", "brown"))
    noise_amplitude = float(tone_cfg.get("noise_amplitude", 0.18))
    always_floor = bool(tone_cfg.get("always_floor", True))

    measured_segments = []
    low_segments = []
    for seg in _segments_from_decisions(decisions, total_dur):
        values = measure_audio_volume(video_path, seg["start"], seg["end"])
        values["shot_id"] = seg.get("shot_id")
        mean_db = values.get("mean_volume_db")
        max_db = values.get("max_volume_db")
        is_low = mean_db is None or mean_db <= threshold_db or (max_db is not None and max_db <= max_threshold_db)
        values["low_loudness"] = is_low
        measured_segments.append(values)
        if is_low:
            low_segments.append(seg)
    report["segments"] = measured_segments
    report["low_segment_count"] = len(low_segments)

    if not low_segments and not always_floor:
        r = _run(["ffmpeg", "-y", "-i", video_path, "-c", "copy", output_path])
        if r.returncode != 0:
            raise RuntimeError(f"failed to write the room-tone-unmixed cut: {r.stderr[-500:]}")
        report["reason"] = "no low-loudness segments"
        return report

    expr_parts = []
    if always_floor and floor_volume > 0:
        expr_parts.append(f"{floor_volume:g}")
    for seg in low_segments:
        s = max(0.0, float(seg["start"]))
        e = min(total_dur, float(seg["end"]))
        if e <= s:
            continue
        fade = max(0.001, min(fade_seconds, (e - s) / 2))
        expr_parts.append(
            f"if(lt(t,{s:.3f}),0,"
            f"if(lt(t,{s + fade:.3f}),{low_segment_volume:g}*(t-{s:.3f})/{fade:.3f},"
            f"if(lt(t,{e - fade:.3f}),{low_segment_volume:g},"
            f"if(lt(t,{e:.3f}),{low_segment_volume:g}*({e:.3f}-t)/{fade:.3f},0))))"
        )
    if not expr_parts:
        expr_parts.append("0")
    volume_expr = "+".join(f"({p})" for p in expr_parts)
    fc = (
        f"[0:a]aformat=sample_fmts=fltp:sample_rates={_SAMPLE_RATE}:channel_layouts=stereo[base];"
        f"[1:a]aformat=sample_fmts=fltp:sample_rates={_SAMPLE_RATE}:channel_layouts=stereo,"
        "highpass=f=70,lowpass=f=6500,"
        f"volume='{volume_expr}':eval=frame[tone];"
        "[base][tone]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
    )
    # Do not add -t on top of a -c:v copy video track (frame-boundary quantisation would drop frames);
    # -shortest alone lets the audio track end with the video track.
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-f", "lavfi", "-i", f"anoisesrc=color={noise_color}:amplitude={noise_amplitude:g}:sample_rate={_SAMPLE_RATE}:duration={total_dur:.3f}",
        "-filter_complex", fc,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(_SAMPLE_RATE),
        "-shortest", "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"room tone low-loudness compensation failed: {r.stderr[-500:]}")
    report["mixed"] = True
    report["settings"] = {
        "low_mean_threshold_db": threshold_db,
        "low_max_threshold_db": max_threshold_db,
        "floor_volume": floor_volume,
        "low_segment_volume": low_segment_volume,
        "always_floor": always_floor,
        "noise_color": noise_color,
    }
    return report


def final_audio_master(
    video_path: str,
    output_path: str,
    target_lufs: float = -16,
    enabled: bool = True,
) -> str:
    """Final loudness mastering: unify the overall loudness and limit the peak."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not enabled:
        r = _run(["ffmpeg", "-y", "-i", video_path, "-c", "copy", output_path])
        if r.returncode != 0:
            raise RuntimeError(f"failed to write the final cut: {r.stderr[-500:]}")
        return output_path
    # Note: never add -t on top of an already -c:v copy video track (keyframe / frame-boundary
    # quantisation would drop a frame and break the strictly aligned 15.000s length upstream).
    # -shortest alone lets the audio end with the video, leaving the video length untouched.
    fc = f"[0:a]loudnorm=I={float(target_lufs):g}:LRA=11:TP=-1.5,alimiter=limit=0.95[aout]"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter_complex", fc,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(_SAMPLE_RATE),
        "-shortest", "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"final audio mastering failed: {r.stderr[-500:]}")
    return output_path


def _fit_to_duration(inp: str, out: str, target_duration: Optional[float]) -> str:
    """Force the final output to the target duration: trim when long, clone the last frame and pad silence when short."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if target_duration is None or target_duration <= 0:
        r = _run(["ffmpeg", "-y", "-i", inp, "-c", "copy", out])
        if r.returncode != 0:
            raise RuntimeError(f"failed to write the cut: {r.stderr[-500:]}")
        return out

    actual = probe_duration(inp) or 0
    pad = max(0.0, float(target_duration) - actual)
    if pad > 0.05:
        fc = (
            f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.3f},trim=0:{target_duration:.3f},setpts=PTS-STARTPTS[v];"
            f"[0:a]apad,atrim=0:{target_duration:.3f},asetpts=PTS-STARTPTS[a]"
        )
        cmd = [
            "ffmpeg", "-y", "-i", inp, "-filter_complex", fc,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-ar", str(_SAMPLE_RATE),
            out,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", inp,
            "-t", f"{float(target_duration):.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-ar", str(_SAMPLE_RATE),
            out,
        ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"failed to force the cut duration: {r.stderr[-500:]}")
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _audio_relation(trans: Optional[Dict[str, Any]]) -> str:
    if not trans:
        return "straight"
    relation = str(trans.get("audio_relation") or trans.get("audio_visual_relation") or "").lower()
    if "j-cut" in relation:
        return "j-cut"
    if "l-cut" in relation:
        return "l-cut"
    return "straight"


def _timing_offset(trans: Optional[Dict[str, Any]]) -> float:
    return max(0.0, _safe_float((trans or {}).get("timing_offset_seconds"), 0.0))


def _compose_jl_audio_track(
    video_body: str,
    audio_sources: List[str],
    decisions: List[Dict[str, Any]],
    output_path: str,
    target_duration: Optional[float],
) -> str:
    """Replace video_body's audio with an independent audio timeline, giving real J-cut / L-cut / straight."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not audio_sources:
        return _fit_to_duration(video_body, output_path, target_duration)

    total_dur = float(target_duration or sum(_safe_float(d.get("net_duration", d.get("duration", 0))) for d in decisions))
    if total_dur <= 0:
        total_dur = probe_duration(video_body) or 0
    if total_dur <= 0:
        raise RuntimeError(f"cannot determine the J/L-cut output duration: {video_body}")

    starts: List[float] = []
    cursor = 0.0
    for dec in decisions:
        starts.append(cursor)
        cursor += max(0.0, _safe_float(dec.get("net_duration", dec.get("duration", 0)), 0.0))

    filters: List[str] = []
    labels: List[str] = []
    for idx, source in enumerate(audio_sources):
        if idx >= len(decisions):
            break
        dec = decisions[idx]
        net = max(0.0, _safe_float(dec.get("net_duration", dec.get("duration", 0)), 0.0))
        if net <= 0:
            continue
        video_start = starts[idx]
        prev_trans = decisions[idx - 1].get("transition_out") if idx > 0 else None
        this_trans = dec.get("transition_out")
        incoming = _timing_offset(prev_trans) if _audio_relation(prev_trans) == "j-cut" else 0.0
        outgoing = _timing_offset(this_trans) if _audio_relation(this_trans) == "l-cut" else 0.0

        placed_start = max(0.0, video_start - incoming)
        source_start = max(0.0, incoming - video_start)
        desired_end = min(total_dur, video_start + net + outgoing)
        duration = max(0.0, desired_end - placed_start)
        if duration <= 0.05:
            continue

        delay_ms = int(round(placed_start * 1000))
        fin = min(0.12, max(duration / 4, 0.0)) if incoming > 0 else 0.0
        fout = min(0.18, outgoing, max(duration / 4, 0.0)) if outgoing > 0 else 0.0
        chain = [
            f"atrim=start={source_start:.3f}:duration={duration:.3f}",
            "asetpts=PTS-STARTPTS",
            f"aformat=sample_fmts=fltp:sample_rates={_SAMPLE_RATE}:channel_layouts=stereo",
        ]
        if fin > 0.001:
            chain.append(f"afade=t=in:st=0:d={fin:.3f}")
        if fout > 0.001:
            chain.append(f"afade=t=out:st={max(duration - fout, 0):.3f}:d={fout:.3f}")
        if delay_ms > 0:
            chain.append(f"adelay={delay_ms}|{delay_ms}")
        label = f"a{idx}"
        filters.append(f"[{idx + 1}:a]{','.join(chain)}[{label}]")
        labels.append(f"[{label}]")

    if not labels:
        r = _run(["ffmpeg", "-y", "-i", video_body, "-t", f"{total_dur:.3f}", "-c", "copy", output_path])
        if r.returncode != 0:
            raise RuntimeError(f"failed to write the cut without J/L audio: {r.stderr[-500:]}")
        return output_path

    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:dropout_transition=0:normalize=0,"
        # apad=whole_dur: if any clip is shorter than planned (the provider returned less than requested),
        # the mixed audio would end before target_duration and -shortest would then truncate the picture
        # track that was already aligned to target_duration, silently shortening the cut. Padding with
        # silence up to total_dur keeps the audio length equal to total_dur. whole_dur is required:
        # a bare apad generates silence forever and ffmpeg never exits (observed to hang).
        f"apad=whole_dur={total_dur:.3f},atrim=0:{total_dur:.3f},asetpts=PTS-STARTPTS[aout]"
    )
    cmd = ["ffmpeg", "-y", "-i", video_body]
    for source in audio_sources:
        cmd.extend(["-i", source])
    cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", "0:v:0", "-map", "[aout]",
        "-t", f"{total_dur:.3f}",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(_SAMPLE_RATE),
        "-shortest", "-avoid_negative_ts", "make_zero",
        output_path,
    ])
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"J/L-cut audio timeline composition failed: {r.stderr[-500:]}")
    return output_path


def compose(
    clip_paths: List[str],
    decisions: List[Dict[str, Any]],
    output_path: str,
    work_dir: str,
    resolution: str = "720P",
    ratio: str = "16:9",
    fps: int = 30,
    target_duration: Optional[float] = None,
    audio_cleanup_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Stitch every clip according to the transition decisions and write the final cut.

    clip_paths[i] matches decisions[i]; decisions[i].transition_out describes the transition after
    segment i (entering segment i+1).
    """
    assert clip_paths, "no clip available to stitch"
    w, h = resolution_to_wh(resolution, ratio)
    norm_dir = Path(work_dir) / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normalise every clip: full keeps the J/L audio headroom, visual keeps only the picture net duration + visual overlap.
    normalized_full: List[str] = []
    normalized_visual: List[str] = []
    for i, cp in enumerate(clip_paths):
        dec = decisions[i] if i < len(decisions) else {}
        net = float(dec.get("net_duration", dec.get("duration", 0)) or 0) if dec else 0.0
        overlap = float(dec.get("overlap_out_seconds", 0) or 0) if dec else 0.0
        gen_duration = float(dec.get("gen_duration", 0) or 0) if dec else 0.0
        audio_ext = float(dec.get("audio_extension_seconds", 0) or 0) if dec else 0.0
        visual_target = (net + overlap) if net > 0 else None
        full_target = max(net + overlap, net + audio_ext, gen_duration) if net > 0 else None

        full_out = str(norm_dir / f"norm_full_{i:02d}.mp4")
        visual_out = str(norm_dir / f"norm_visual_{i:02d}.mp4")
        normalize_clip(cp, full_out, w, h, fps, target_duration=full_target, audio_cleanup_cfg=audio_cleanup_cfg)
        normalize_clip(cp, visual_out, w, h, fps, target_duration=visual_target, audio_cleanup_cfg=audio_cleanup_cfg)
        normalized_full.append(full_out)
        normalized_visual.append(visual_out)

    body_out = str(Path(work_dir) / "body_video_timeline.mp4")
    if len(normalized_visual) == 1:
        _fit_to_duration(normalized_visual[0], body_out, target_duration)
        return _compose_jl_audio_track(body_out, normalized_full, decisions, output_path, target_duration)

    # 2. Compose the picture timeline left to right; the old audio is only a filter placeholder and is replaced by the J/L timeline.
    tmp_dir = Path(work_dir) / "compose_steps"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    current = normalized_visual[0]

    for i in range(1, len(normalized_visual)):
        nxt = normalized_visual[i]
        # transition entering segment i = decisions[i-1].transition_out
        trans = decisions[i - 1].get("transition_out") if i - 1 < len(decisions) else None
        step_out = str(tmp_dir / f"step_{i:02d}.mp4")

        use_cut = True
        transition_name = "fade"
        dur = _DEFAULT_XFADE_DUR
        if trans:
            compose_info = trans.get("compose", {}) or {}
            occlusion_handled = trans.get("occlusion_first_frame_handled", False)
            if compose_info.get("mode") == "xfade" and not occlusion_handled:
                use_cut = False
                transition_name = compose_info.get("xfade", "fade")
                td = float(trans.get("transition_duration_seconds") or 0)
                overlap = float(trans.get("overlap_seconds") or 0)
                if transition_name in ("fadewhite", "fadeblack"):
                    # flash white/black: no overlap (concat), dur is only the visual fade duration
                    dur = td if td > 0 else _DEFAULT_XFADE_DUR
                else:
                    # dissolve/wipe: the overlap equals the effect duration exactly (overlap_seconds ==
                    # the extra visual headroom of the previous shot), so it eats precisely that extra material.
                    dur = td if td > 0 else (overlap if overlap > 0 else _DEFAULT_XFADE_DUR)

        if use_cut:
            current = _concat_pair(current, nxt, step_out)
        else:
            current = _xfade_pair(current, nxt, step_out, transition_name, dur)

    # 3. Force the picture timeline to the target duration first, then replace it with the strict J/L-cut audio timeline.
    _fit_to_duration(current, body_out, target_duration)
    return _compose_jl_audio_track(body_out, normalized_full, decisions, output_path, target_duration)
