"""Reusable Stage4 / Stage5 executor (shot_runner).

Extracts the shot generation and the transition stitching + audio post-processing that used to be
inlined in pipeline.py into reusable units, so the agent's rolling evaluate-repair loop can:
  - regenerate only the failing shots (ShotGenerator.generate_shot is re-entrant per shot);
  - restitch without regenerating anything (compose_case, for pure effect-duration changes).

pipeline.py (the original baseline behaviour) and agent_loop.py (with evaluation and repair) share
one implementation, so the two paths have identical generation / stitching semantics.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Stitching / frame extraction / mixing rely on ffmpeg: the J/L-cut audio timeline uses amix's
# normalize option, xfade and other 4.3+ features, and the system's older ffmpeg 4.2 reports
# "Option not found". The ffmpeg on PATH is used by default; to pin a dedicated environment,
# export AV_PROCESS_BIN=/path/to/env/bin before running and it is prepended to PATH, so importing this module directly (without run_*.sh) never falls back to the old ffmpeg.
_AV_PROCESS_BIN = os.environ.get("AV_PROCESS_BIN", "")
if (_AV_PROCESS_BIN and os.path.isdir(_AV_PROCESS_BIN)
        and _AV_PROCESS_BIN not in os.environ.get("PATH", "").split(os.pathsep)):
    os.environ["PATH"] = _AV_PROCESS_BIN + os.pathsep + os.environ.get("PATH", "")

from .audio_policy import decide_global_bgm
from .audio_reference import extract_reference_audio, public_audio_url
from .frame_utils import extract_frame_at, extract_last_frame


def resolve_provider_modules(provider: str):
    """Pick the Stage4 generation adapter modules from the provider in config.yaml.

    Only the wan provider is supported: DashScope wan2.7 series (t2v/i2v/r2v).
    Imported lazily so the module can be loaded without the generation dependencies installed.
    """
    provider = (provider or "wan").lower()
    if provider != "wan":
        raise ValueError(f"unsupported provider: {provider!r} (only 'wan' is supported)")
    from tools import wan_t2v, wan_i2v  # noqa: PLC0415
    return wan_t2v, wan_i2v


def safe_title(title: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip().replace(" ", "_")


def write_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class ShotGenerator:
    """Stage4 shot generator.

    Holds the cross-shot state (same-label audio/visual references, previous clip path), so it can
    generate a whole case in order or regenerate one shot during repair (reusing the first round's references).
    """

    def __init__(self, cfg: Dict[str, Any], case_id: Any, artifacts_dir: Path,
                 report: Dict[str, Any]):
        self.cfg = cfg
        self.case_id = case_id
        self.artifacts_dir = Path(artifacts_dir)
        self.report = report

        self.clips_dir = self.artifacts_dir / "clips"
        self.frames_dir = self.artifacts_dir / "frames"
        self.audio_refs_dir = self.artifacts_dir / "audio_refs"
        self.visual_refs_dir = self.artifacts_dir / "visual_refs"
        for d in (self.clips_dir, self.frames_dir, self.audio_refs_dir, self.visual_refs_dir):
            d.mkdir(parents=True, exist_ok=True)

        provider = str(cfg.get("provider", "wan")).lower()
        self.provider = provider
        self.t2v_mod, self.i2v_mod = resolve_provider_modules(provider)

        visual_cfg = cfg.get("visual_reference") or {}
        self.use_same_label_visual_ref = bool(
            visual_cfg.get("same_label_reference", visual_cfg.get("same_label_first_frame", True)))
        same_label_visual_mode = str(visual_cfg.get("same_label_mode", "r2v_reference")).lower()
        self.use_same_label_r2v = same_label_visual_mode in ("r2v", "r2v_reference", "reference_image")
        self.visual_ref_frame_position = float(visual_cfg.get("reference_frame_position", 0.5))
        self.r2v_create_url = (cfg.get("r2v_create_url") or cfg.get("video_create_url")
                               or cfg.get("wan_create_url"))
        self.resolution = cfg.get("resolution", "720P")
        self.ratio = cfg.get("ratio", "16:9")
        self.poll = cfg.get("poll_interval", 15)
        self.timeout = cfg.get("timeout", 1800)

        # cross-shot state
        self.prev_clip: Optional[str] = None
        self.audio_reference_urls: Dict[Any, str] = {}
        self.audio_reference_files: Dict[Any, str] = {}
        self.visual_reference_frames: Dict[Any, str] = {}

    # ---- internal helpers ----

    def clip_path_for(self, shot_id: Any) -> str:
        return str(self.clips_dir / f"shot_{shot_id}.mp4")

    def _t2v(self, dec: Dict[str, Any], clip_out: str, audio_url: Optional[str]) -> None:
        self.t2v_mod.generate_t2v(
            prompt=dec["prompt"], save_path=clip_out,
            duration=dec.get("gen_duration", dec["duration"]),
            ratio=self.ratio, resolution=self.resolution,
            model_name=self.cfg.get("t2v_model", "wan2.7-t2v"),
            negative_prompt=dec.get("negative_prompt"),
            audio_url=audio_url,
            poll_interval=self.poll, timeout=self.timeout,
        )

    def _i2v(self, dec: Dict[str, Any], clip_out: str, first_frame_image: str,
             audio_url: Optional[str]) -> None:
        self.i2v_mod.generate_i2v(
            prompt=dec["prompt"], first_frame_image=first_frame_image, save_path=clip_out,
            duration=dec.get("gen_duration", dec["duration"]), resolution=self.resolution,
            model_name=self.cfg.get("i2v_model", "wan2.7-i2v-2026-04-25"),
            negative_prompt=dec.get("negative_prompt"),
            audio_url=audio_url,
            duration_supported=bool(self.cfg.get("i2v_duration_supported", True)),
            poll_interval=self.poll, timeout=self.timeout,
        )

    def _r2v(self, dec: Dict[str, Any], clip_out: str, reference_image: str) -> None:
        self.i2v_mod.generate_r2v(
            prompt=dec["prompt"],
            reference_media=[{"type": "reference_image", "url": reference_image}],
            save_path=clip_out,
            duration=dec.get("gen_duration", dec["duration"]),
            resolution=self.resolution,
            model_name=self.cfg.get("r2v_model", "wan2.7-r2v"),
            negative_prompt=dec.get("negative_prompt"),
            poll_interval=self.poll, timeout=self.timeout,
            create_url=self.r2v_create_url,
        )

    # ---- single-shot generation (re-entrant, used to regenerate failed shots during repair) ----

    def generate_shot(self, dec: Dict[str, Any], allow_occlusion_i2v: bool = True) -> Dict[str, Any]:
        """Generate one shot and return its shot_report (including the ok flag)."""
        shot_id = dec["shot_id"]
        clip_out = self.clip_path_for(shot_id)
        shot_report: Dict[str, Any] = {
            "shot_id": shot_id, "gen_mode": dec["gen_mode"],
            "net_duration": dec.get("net_duration", dec["duration"]),
            "gen_duration": dec.get("gen_duration", dec["duration"]),
            "audio_extension_seconds": dec.get("audio_extension_seconds", 0),
            "incoming_j_offset_seconds": dec.get("incoming_j_offset_seconds", 0),
            "outgoing_l_offset_seconds": dec.get("outgoing_l_offset_seconds", 0),
            "overlap_out_seconds": dec.get("overlap_out_seconds", 0),
            "visual_reference": dec.get("visual_reference"),
            "audio_reference": dec.get("audio_reference"),
            "scene_audio_beds": dec.get("scene_audio_beds", []),
            "clip": clip_out, "ok": False,
        }
        audio_ref = dec.get("audio_reference") or {}
        visual_ref = dec.get("visual_reference") or {}
        audio_url = None
        visual_ref_img = None

        scene_audio_source = next(
            (bed.get("source_path") for bed in dec.get("scene_audio_beds", [])
             if bed.get("source_path") and Path(str(bed.get("source_path"))).exists()),
            None,
        )
        if scene_audio_source:
            audio_url = public_audio_url(str(scene_audio_source), use_data_uri=True)
            if audio_url:
                shot_report["reference_audio_file"] = str(scene_audio_source)
                shot_report["reference_audio_type"] = "scene_audio_bed_source"

        if self.use_same_label_visual_ref and visual_ref.get("use_as_reference"):
            visual_ref_img = self.visual_reference_frames.get(visual_ref.get("source_shot_id"))
            if visual_ref_img:
                shot_report["reference_frame_from_shot"] = visual_ref.get("source_shot_id")
                shot_report["reference_frame_file"] = visual_ref_img
            else:
                msg = (f"shot {shot_id} is missing its same-label visual reference source shot "
                       f"{visual_ref.get('source_shot_id')}, generating this shot without a reference image")
                print(f"  [VisualRef] {msg}")
                self.report["errors"].append(msg)

        if audio_ref.get("use_as_reference") and not audio_url:
            audio_url = self.audio_reference_urls.get(audio_ref.get("source_shot_id"))
            if audio_url:
                shot_report["reference_audio_from_shot"] = audio_ref.get("source_shot_id")
                shot_report["reference_audio_file"] = self.audio_reference_files.get(
                    audio_ref.get("source_shot_id"))
            else:
                msg = (f"shot {shot_id} is missing its audio reference source shot "
                       f"{audio_ref.get('source_shot_id')}, generating this shot without audio_url")
                print(f"  [AudioRef] {msg}")
                self.report["errors"].append(msg)

        print(f"  [Stage4] case {self.case_id} shot {shot_id} generation started "
              f"({dec['gen_mode']}, {shot_report['gen_duration']}s)", flush=True)
        try:
            first_frame_image = None
            if allow_occlusion_i2v and dec.get("gen_mode") == "i2v" and self.prev_clip:
                frame_img = str(self.frames_dir / f"shot_{shot_id}_firstframe.png")
                first_frame_image = extract_last_frame(self.prev_clip, frame_img)

            if first_frame_image:
                try:
                    self._i2v(dec, clip_out, first_frame_image, audio_url)
                    shot_report["gen_mode"] = "i2v"
                    shot_report["first_frame_image"] = first_frame_image
                    shot_report["first_frame_reason"] = "occlusion_tail_frame"
                except Exception as e:  # noqa: BLE001
                    print(f"  [Stage4] shot {shot_id} i2v failed({e}), falling back to t2v")
                    self.report["errors"].append(f"shot {shot_id} i2v->t2v: {e}")
                    shot_report["gen_mode"] = "t2v(fallback)"
                    self._t2v(dec, clip_out, audio_url)
            elif visual_ref_img and self.use_same_label_r2v:
                try:
                    self._r2v(dec, clip_out, visual_ref_img)
                    shot_report["gen_mode"] = "r2v(visual_ref)"
                    shot_report["reference_image"] = visual_ref_img
                    shot_report["reference_reason"] = "same_label_reference_image"
                    shot_report["reference_model"] = self.cfg.get("r2v_model", "wan2.7-r2v")
                except Exception as e:  # noqa: BLE001
                    print(f"  [Stage4] shot {shot_id} r2v failed({e}), falling back to t2v")
                    self.report["errors"].append(f"shot {shot_id} r2v->t2v: {e}")
                    shot_report["gen_mode"] = "t2v(fallback)"
                    self._t2v(dec, clip_out, audio_url)
            elif visual_ref_img:
                try:
                    self._i2v(dec, clip_out, visual_ref_img, audio_url)
                    shot_report["gen_mode"] = "i2v(visual_ref)"
                    shot_report["first_frame_image"] = visual_ref_img
                    shot_report["first_frame_reason"] = "same_label_first_frame"
                except Exception as e:  # noqa: BLE001
                    print(f"  [Stage4] shot {shot_id} i2v with a visual reference failed({e}), falling back to t2v")
                    self.report["errors"].append(f"shot {shot_id} i2v visual_ref->t2v: {e}")
                    shot_report["gen_mode"] = "t2v(fallback)"
                    self._t2v(dec, clip_out, audio_url)
            else:
                try:
                    self._t2v(dec, clip_out, audio_url)
                except Exception as e:  # noqa: BLE001
                    if audio_url:
                        print(f"  [Stage4] shot {shot_id} generation with audio_url failed({e}), retrying without the audio reference")
                        self.report["errors"].append(f"shot {shot_id} audio_url fallback: {e}")
                        shot_report["audio_reference_fallback"] = str(e)
                        self._t2v(dec, clip_out, None)
                    else:
                        raise

            shot_report["ok"] = True
            self.prev_clip = clip_out
            self._harvest_references(dec, shot_id, clip_out, shot_report,
                                    visual_ref, audio_ref)
            print(f"  [Stage4] shot {shot_id} generation done ({shot_report['gen_mode']})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [Stage4] shot {shot_id} generation failed: {e}", flush=True)
            self.report["errors"].append(f"shot {shot_id} generation failed: {e}")
        return shot_report

    def _harvest_references(self, dec: Dict[str, Any], shot_id: Any, clip_out: str,
                            shot_report: Dict[str, Any], visual_ref: Dict[str, Any],
                            audio_ref: Dict[str, Any]) -> None:
        """When this shot is a reference source, extract the appearance reference frame and audio for later same-label shots."""
        if self.use_same_label_visual_ref and visual_ref.get("role") == "source":
            ref_frame_path = str(
                self.visual_refs_dir / f"label_{visual_ref.get('label')}_shot_{shot_id}_reference.png")
            got_frame = extract_frame_at(clip_out, ref_frame_path,
                                         ratio=self.visual_ref_frame_position)
            if got_frame:
                self.visual_reference_frames[shot_id] = got_frame
                self.report.setdefault("visual_references", {})[str(shot_id)] = {
                    "label": visual_ref.get("label"),
                    "source_shot_id": shot_id,
                    "frame_file": got_frame,
                    "frame_position": self.visual_ref_frame_position,
                    "reference_type": ("r2v_reference_image" if self.use_same_label_r2v
                                       else visual_ref.get("reference_type", "same_label_first_frame")),
                }
                shot_report["reference_frame_source_file"] = got_frame
                shot_report["reference_frame_position"] = self.visual_ref_frame_position
                print(f"  [VisualRef] label={visual_ref.get('label')} appearance reference frame taken from shot "
                      f"{shot_id} @ {self.visual_ref_frame_position:.2f}")
            else:
                self.report["errors"].append(f"shot {shot_id} same-label visual reference frame extraction failed")

        if audio_ref.get("role") == "source":
            ref_path = str(self.audio_refs_dir / f"label_{audio_ref.get('label')}_shot_{shot_id}.mp3")
            got_audio = extract_reference_audio(
                clip_out, ref_path,
                speech_only=(audio_ref.get("reference_type") == "speech_voice"),
            )
            if got_audio:
                audio_ref_url = public_audio_url(got_audio, use_data_uri=True)
                if audio_ref_url:
                    self.audio_reference_urls[shot_id] = audio_ref_url
                    self.audio_reference_files[shot_id] = got_audio
                    self.report.setdefault("audio_references", {})[str(shot_id)] = {
                        "label": audio_ref.get("label"),
                        "source_shot_id": shot_id,
                        "audio_file": got_audio,
                        "url_type": "data_uri",
                        "reference_type": audio_ref.get("reference_type", "speech_voice"),
                    }
                    shot_report["reference_audio_source_file"] = got_audio
                    print(f"  [AudioRef] label={audio_ref.get('label')} "
                          f"{audio_ref.get('reference_type', 'speech_voice')} reference taken from shot {shot_id}")
            else:
                self.report["errors"].append(f"shot {shot_id} audio reference source extraction failed")

    # ---- full sequential generation ----

    def generate_all(self, decisions: List[Dict[str, Any]]) -> List[str]:
        """Generate every shot in order and return the list of successful clip paths (also written into report['shots'])."""
        clip_paths: List[str] = []
        for dec in decisions:
            shot_report = self.generate_shot(dec)
            self.report.setdefault("shots", []).append(shot_report)
            if shot_report["ok"]:
                clip_paths.append(shot_report["clip"])
        return clip_paths


def compose_case(cfg: Dict[str, Any], case_id: Any, title: str,
                 plan: Dict[str, Any], edit_decisions: Dict[str, Any],
                 used_decisions: List[Dict[str, Any]], clip_paths: List[str],
                 report: Dict[str, Any], artifacts_dir: Path,
                 final_path: Optional[str] = None,
                 work_subdir: Optional[str] = None) -> str:
    """Stage5 -- transition stitching + room tone / scene beds / BGM / mastering; returns the final cut path.

    When work_subdir is set, all intermediate artifacts go under artifacts_dir/work_subdir, so a repair
    round does not overwrite the previous round's files and the two can be compared.
    """
    from tools import video_compose  # noqa: PLC0415

    artifacts_dir = Path(artifacts_dir)
    work_dir = artifacts_dir / work_subdir if work_subdir else artifacts_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_cfg = cfg.get("audio") or {}
    resolution = cfg.get("resolution", "720P")
    ratio = cfg.get("ratio", "16:9")
    target_duration = cfg.get("target_duration", 15)

    if final_path is None:
        output_dir = Path(cfg["paths"]["output_dir"])
        final_path = str(output_dir / f"{case_id}_{safe_title(title)}.mp4")

    # net_duration is allocated by _allocate_net_durations() as float seconds proportional to the GT
    # timestamps and sums exactly to target_duration, so the cut is always trimmed to it. This is just a check.
    actual_total_duration = sum(
        float(d.get("net_duration", d.get("duration", 0)) or 0) for d in used_decisions)
    if abs(actual_total_duration - float(target_duration)) > 0.2:
        print(f"  [Stage5] warning: the sum of shot net durations({actual_total_duration:.2f}s) does not match the configured "
              f"target_duration({target_duration}s) (some shots may have failed to generate and been dropped), "
              f"the cut will still be trimmed strictly to target_duration={target_duration}s")

    bgm_plan = decide_global_bgm(plan, audio_cfg)
    report["bgm_plan"] = bgm_plan
    scene_beds = edit_decisions.get("scene_audio_beds", [])
    scene_beds_enabled = bool((audio_cfg.get("scene_audio_beds") or {}).get("enabled", True))
    room_tone_cfg = audio_cfg.get("room_tone_floor") or {}
    room_tone_enabled = bool(room_tone_cfg.get("enabled", False))
    master_cfg = audio_cfg.get("final_master") or {}
    master_enabled = bool(master_cfg.get("enabled", True))
    needs_scene_mix = bool(scene_beds and scene_beds_enabled)
    needs_intermediate = bool(needs_scene_mix or bgm_plan.get("enabled")
                              or room_tone_enabled or master_enabled)
    compose_out = final_path if not needs_intermediate else str(work_dir / "body_composed.mp4")

    video_compose.compose(
        clip_paths=clip_paths, decisions=used_decisions,
        output_path=compose_out, work_dir=str(work_dir),
        resolution=resolution, ratio=ratio, fps=cfg.get("fps", 30),
        target_duration=target_duration,
        audio_cleanup_cfg=audio_cfg.get("clip_audio_cleanup") or {},
    )
    current_audio_video = compose_out
    report.setdefault("audio_processing", {})["clip_audio_cleanup"] = \
        audio_cfg.get("clip_audio_cleanup") or {}

    if room_tone_enabled:
        room_tone_out = (final_path
                         if not needs_scene_mix and not bgm_plan.get("enabled") and not master_enabled
                         else str(work_dir / "with_room_tone_floor.mp4"))
        room_tone_report = video_compose.add_room_tone_floor(
            video_path=current_audio_video, decisions=used_decisions,
            output_path=room_tone_out, cfg=room_tone_cfg,
        )
        report["audio_processing"]["room_tone_floor"] = room_tone_report
        current_audio_video = room_tone_out
        print(f"  [Stage5] room tone low-loudness compensation done: "
              f"low_segments={room_tone_report.get('low_segment_count', 0)}")

    if needs_scene_mix:
        scene_out = (final_path if not bgm_plan.get("enabled") and not master_enabled
                     else str(work_dir / "with_scene_audio_beds.mp4"))
        scene_report = video_compose.add_scene_audio_beds(
            video_path=current_audio_video, scene_audio_beds=scene_beds,
            decisions=used_decisions, output_path=scene_out, work_dir=str(work_dir),
        )
        report["scene_audio_bed_mix"] = scene_report
        current_audio_video = scene_out
        mixed = sum(1 for bed in scene_report.get("beds", []) if bed.get("mixed"))
        print(f"  [Stage5] scene-level diegetic audio bed processing done: mixed={mixed}/{len(scene_report.get('beds', []))}")

    if bgm_plan.get("enabled"):
        bgm_cfg = audio_cfg.get("global_bgm") or {}
        bgm_out = final_path if not master_enabled else str(work_dir / "with_global_bgm.mp4")
        video_compose.add_global_bgm(
            video_path=current_audio_video, bgm_path=bgm_plan["source_path"],
            output_path=bgm_out,
            volume=float(bgm_cfg.get("volume", 0.18)),
            fade_in=float(bgm_cfg.get("fade_in_seconds", 1.0)),
            fade_out=float(bgm_cfg.get("fade_out_seconds", 1.5)),
            loudnorm=bool(bgm_cfg.get("loudnorm", False)),
        )
        current_audio_video = bgm_out
        print(f"  [Stage5] global BGM mixed in: {bgm_plan['source_path']}")
    else:
        print(f"  [Stage5] no global BGM added: {bgm_plan.get('reason')}")

    if master_enabled:
        video_compose.final_audio_master(
            video_path=current_audio_video, output_path=final_path,
            target_lufs=float(master_cfg.get("target_lufs", -16)), enabled=True,
        )
        report["audio_processing"]["final_master"] = {
            "enabled": True, "target_lufs": float(master_cfg.get("target_lufs", -16)),
        }

    report["final_video"] = final_path
    print(f"  [Stage5] cut written to: {final_path}")
    return final_path


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
