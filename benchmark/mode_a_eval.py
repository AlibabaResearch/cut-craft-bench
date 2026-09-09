#!/usr/bin/env python3
"""
Mode A evaluation engine -- direct computation from expert-model outputs.
Every dimension becomes a 0-1 score from expert-model outputs plus a deterministic formula.

Dimensions:
  A1: shot count accuracy    B1: shot duration accuracy   B2: beat synchronisation
  A4: style similarity       D1: transition quality (routed by GT type)   D2: optical transition type
  D3: transition audio-visual relation   E1: camera movement type   E2: image quality
  E3: style consistency      F1: intra-shot audio-visual sync   F2: audio quality
"""

import os
import sys
import json
import time
import argparse
import datetime
import re
import base64
import subprocess
import tempfile
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

# Directory holding ffmpeg / ffprobe. The versions on PATH are used by default; to pin a dedicated
# environment (e.g. a conda ffmpeg 8.x), export AV_PROCESS_BIN=/path/to/env/bin before running.
AV_PROCESS_BIN = os.environ.get("AV_PROCESS_BIN", "")
if AV_PROCESS_BIN and AV_PROCESS_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = AV_PROCESS_BIN + os.pathsep + os.environ.get("PATH", "")

# Add current dir to path for service_client import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from service_client import (
    call_transnetv2, call_raft, call_dinov2, call_whisper,
    call_demucs, call_e2quality, call_panns, call_dnsmos,
    call_clip_style_match,
    check_all_services
)
from vlm_shot_alignment import (
    align_shots_with_vlm, get_aligned_transitions, get_aligned_shot_clip_range
)

# ============================================================
# Skill config loading (Plan B: prompts.yaml is the single source of truth)
# ============================================================
# E3's DINOv2 sampling params used to be hardcoded function defaults
# (sample_fps=4.0, max_frames=80). They now live in
# benchmark/skills/e3-event-coherence-skill/prompts.yaml. Loading failures
# fall back to the historical defaults with a warning instead of raising,
# because mode_a_eval.py evaluates many unrelated dimensions and a broken
# E3 skill config should not break the whole Mode A run.
try:
    from skill_loader import load_skill
    _e3_scoring_cfg = load_skill("e3-event-coherence-skill").get("scoring", {})
    _E3_SAMPLE_FPS = float(_e3_scoring_cfg.get("sample_fps", 4.0))
    _E3_MAX_FRAMES = int(_e3_scoring_cfg.get("max_frames", 80))
except Exception as _e3_cfg_err:
    print(f"[WARN] Failed to load e3-event-coherence-skill scoring config, "
          f"using defaults sample_fps=4.0/max_frames=80: {_e3_cfg_err}")
    _E3_SAMPLE_FPS = 4.0
    _E3_MAX_FRAMES = 80


# ============================================================
#  VLM call infrastructure (used by the F2 audio quality evaluation)
# ============================================================
_DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
_DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
_VLM_MODEL = os.getenv("VLM_MODEL")
_vlm_client = OpenAI(api_key=_DASHSCOPE_API_KEY, base_url=_DASHSCOPE_BASE_URL)


class _VLMRetryExhausted(RuntimeError):
    """A Mode A internal VLM call exhausted its retries."""


def _get_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _is_retryable_vlm_error(exc: Exception) -> bool:
    err = str(exc).lower()
    retry_keywords = [
        "429", "insufficient_quota", "rate limit", "too many requests",
        "token-limit", "timeout", "timed out", "connection", "temporarily",
        "server error", "internal error", "internal_error", "500", "502", "503", "504",
    ]
    return any(keyword in err for keyword in retry_keywords)


def _get_video_duration(video_path: str) -> float:
    """Video duration in seconds, via ffprobe."""
    import json as _json
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, timeout=10
        )
        data = _json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def _ensure_min_duration(video_path: str, min_dur: float = 2.0) -> str:
    """Ensure the video meets the minimum duration; loop-concatenate it when too short."""
    dur = _get_video_duration(video_path)
    if dur <= 0 or dur >= min_dur:
        return video_path
    loop_count = int(min_dur // dur) + 1
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-stream_loop", str(loop_count),
        "-i", video_path, "-t", str(min_dur),
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "ultrafast", "-q:v", "23",
        tmp.name
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)
    return tmp.name


def _compress_video_for_vlm(video_path: str, max_size_mb: float = 4.0) -> str:
    """Compress the video down to a size the VLM accepts."""
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    if file_size <= max_size_mb:
        return video_path
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "scale=-2:360",
        "-b:v", "1500k", "-b:a", "48k", "-ac", "1", "-r", "15",
        tmp.name
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)
    return tmp.name


def _call_vlm(system_prompt: str, user_text: str, video_path: str = None,
              max_tokens: int = 500, temperature: float = 0.1) -> str:
    """Call the Qwen3.5-omni-plus VLM."""
    combined_text = f"[ROLE]\n{system_prompt}\n\n[TASK]\n{user_text}"

    if video_path:
        # Ensure video meets VLM minimum duration requirement
        video_path = _ensure_min_duration(video_path)
        compressed = _compress_video_for_vlm(video_path)
        with open(compressed, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content = [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
            {"type": "text", "text": combined_text},
        ]
        if compressed != video_path:
            os.unlink(compressed)
    else:
        content = [{"type": "text", "text": combined_text}]

    messages = [{"role": "user", "content": content}]

    max_retries = _get_env_int("VLM_MAX_RETRIES", 4)
    base_wait = _get_env_float("VLM_RETRY_BASE_SECONDS", 30.0)
    max_wait = _get_env_float("VLM_RETRY_MAX_SECONDS", 240.0)

    _eval_seed = os.environ.get("EVAL_SEED")
    _seed_kwargs = {"seed": int(_eval_seed)} if _eval_seed else {}

    for attempt in range(1, max_retries + 1):
        try:
            completion = _vlm_client.chat.completions.create(
                model=_VLM_MODEL,
                messages=messages,
                modalities=["text"],
                stream=True,
                stream_options={"include_usage": True},
                temperature=temperature,
                max_tokens=max_tokens,
                **_seed_kwargs,
            )
            response_text = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            return response_text.strip()
        except Exception as e:
            if attempt >= max_retries or not _is_retryable_vlm_error(e):
                print(f"  [VLM ERROR] attempt {attempt}/{max_retries} failed: {e}")
                raise _VLMRetryExhausted(f"VLM call failed after {attempt}/{max_retries} attempts: {e}") from e
            wait = min(max_wait, base_wait * (2 ** (attempt - 1)))
            print(f"  [VLM RETRY] attempt {attempt}/{max_retries} failed: {e}")
            print(f"  [VLM RETRY] waiting {wait:.1f}s before retrying...")
            time.sleep(wait)


# ============================================================
#  Utilities
# ============================================================

def clip01(x: float) -> float:
    """Clip value to [0, 1]"""
    return max(0.0, min(1.0, x))


def cosine_similarity(a, b) -> float:
    """Cosine similarity."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def parse_time_range(description: str) -> Tuple[float, float]:
    """Parse the time range [start-end] out of a shot description."""
    match = re.search(r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)s?\]', description)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


def extract_shot_time_ranges(prompt: dict) -> List[Tuple[float, float]]:
    """Extract each shot's time range."""
    ranges = []
    for shot in prompt.get("shots", []):
        desc = shot.get("description_prompt", "")
        start, end = parse_time_range(desc)
        ranges.append((start, end))
    return ranges


class TestLogger:
    """Live test-log writer for markdown output (file writing is disabled)."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        # self._init_log()  # md log files are no longer created
    
    def _init_log(self):
        # with open(self.log_path, "w", encoding="utf-8") as f:
        #     f.write(f"# Mode A evaluation log\n\n")
        #     f.write(f"> Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        #     f.write(f"---\n\n")
        pass
    
    def log(self, text: str):
        # with open(self.log_path, "a", encoding="utf-8") as f:
        #     f.write(text + "\n")
        pass
    
    def log_section(self, title: str):
        self.log(f"\n## {title}\n")
    
    def log_subsection(self, title: str):
        self.log(f"\n### {title}\n")
    
    def log_metric(self, dim: str, score: float, details: str = ""):
        line = f"- **{dim}**: `{score:.4f}`"
        if details:
            line += f"  ({details})"
        self.log(line)
    
    def log_error(self, dim: str, error: str):
        self.log(f"- **{dim}**: ❌ ERROR: {error}")
    
    def log_info(self, text: str):
        self.log(f"  - {text}")


# ============================================================
#  Dimension evaluation functions
# ============================================================

# ---- A1: shot count accuracy ----
def eval_a1_shot_count(transnetv2_result: dict, prompt: dict,
                       alignment_result: dict = None) -> dict:
    """
    A1: shot count accuracy (how faithfully the model follows the shot-count instruction)
    Always computed from the raw TransNetV2 shot count: 1 - |N_pred - N_gt| / max(N_gt, 1), clipped to [0,1].
    alignment_result is recorded as supplementary information only and does not affect the score.
    """
    n_gt = prompt.get("number_of_shots", 1)
    n_pred = transnetv2_result.get("num_shots", 0)
    score = clip01(1.0 - abs(n_pred - n_gt) / max(n_gt, 1))
    
    result = {
        "dimension": "A1",
        "metric": "shot count accuracy (raw TransNetV2)",
        "score": score,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "method": "transnetv2_raw",
    }
    
    # Record the alignment info (does not affect the score; consumed downstream by A2)
    if alignment_result is not None:
        n_matched = alignment_result.get("n_matched", 0)
        n_merged = alignment_result.get("n_merged", 0)
        n_missing = alignment_result.get("n_missing", 0)
        result["alignment_info"] = {
            "n_matched": n_matched,
            "n_merged": n_merged,
            "n_missing": n_missing,
            "shot_accuracy": clip01((n_matched + n_merged) / max(n_gt, 1)),
        }
    
    return result


# ---- A4: style match ----
# 9 fixed style classes (the third field of source_seed in prompt.json)
STYLE_CATEGORIES = [
    "Documentary Style", "Vintage / Retro Style", "Naturalistic Realism",
    "Social Media/Vlog Style", "Music Video Style", "Cinematic Narrative Style",
    "Commercial Advertising Style", "Sci-fi Futuristic Style", "Experimental Style",
]


def extract_target_style(prompt: dict) -> str:
    """Extract the target style label from the third field of source_seed (e.g. 'Documentary Style')."""
    ss = prompt.get("source_seed", "") or ""
    parts = ss.split("_")
    if len(parts) >= 3:
        return parts[2].strip()
    return ""


def eval_a4_style(clip_style_result: dict, prompt: dict) -> dict:
    """
    A4: style match (the video's picture style vs the target style given in the prompt)
    CLIP zero-shot style classification: frame-vs-text similarity against the 9 style prompts, softmax per
    frame, and the mean probability of the target style is the A4 score (CLIP's confidence in that style).
    """
    score = clip_style_result.get("a4_score", 0.0)
    return {
        "dimension": "A4",
        "metric": "style matching",
        "score": clip01(score),
        "target_style": clip_style_result.get("target_style", ""),
        "predicted_style": clip_style_result.get("predicted_style", ""),
        "per_style_mean_prob": clip_style_result.get("per_style_mean_prob", {}),
    }


# ---- B1: shot duration accuracy ----
def eval_b1_shot_duration(transnetv2_result: dict, prompt: dict,
                          alignment_result: dict = None) -> dict:
    """
    B1: shot duration accuracy
    With an alignment_result, only shots whose status is 'matched' or 'merged' contribute a duration error;
    otherwise fall back to the original formula: 1 - mean(|d_i - d_i_gt| / d_i_gt), clipped to [0,1].
    """
    gt_ranges = extract_shot_time_ranges(prompt)

    if not gt_ranges:
        return {"dimension": "B1", "metric": "shot duration accuracy", "score": 0.0, "error": "no GT data"}

    gt_durations = [end - start for start, end in gt_ranges if end > start]

    if alignment_result is not None:
        # Use the alignment result: only valid shots are evaluated
        aligned_shots = alignment_result.get("aligned_shots", [])
        errors = []
        gt_durs_used = []
        pred_durs_used = []
        for ashot in aligned_shots:
            if ashot["status"] == "missing" or ashot["time_range"] is None:
                continue
            gt_idx = ashot["gt_shot_idx"]
            if gt_idx >= len(gt_durations):
                continue
            d_gt = gt_durations[gt_idx]
            d_pred = ashot["time_range"][1] - ashot["time_range"][0]
            if d_gt > 0:
                errors.append(abs(d_pred - d_gt) / d_gt)
                gt_durs_used.append(round(d_gt, 3))
                pred_durs_used.append(round(d_pred, 3))

        if not errors:
            return {"dimension": "B1", "metric": "shot duration accuracy", "score": 0.0,
                    "error": "no valid aligned shots", "method": "vlm_aligned"}

        score = clip01(1.0 - float(np.mean(errors)))
        return {
            "dimension": "B1",
            "metric": "shot duration accuracy",
            "score": score,
            "gt_durations": gt_durs_used,
            "pred_durations": pred_durs_used,
            "mean_relative_error": round(float(np.mean(errors)), 4),
            "n_evaluated": len(errors),
            "method": "vlm_aligned",
        }
    else:
        # original logic
        shots_pred = transnetv2_result.get("shots", [])
        if not shots_pred:
            return {"dimension": "B1", "metric": "shot duration accuracy", "score": 0.0, "error": "no data"}

        pred_durations = [s.get("duration", 0.0) for s in shots_pred]
        n = min(len(gt_durations), len(pred_durations))
        if n == 0:
            return {"dimension": "B1", "metric": "shot duration accuracy", "score": 0.0, "error": "empty"}

        errors = []
        for i in range(n):
            d_gt = gt_durations[i]
            d_pred = pred_durations[i]
            if d_gt > 0:
                errors.append(abs(d_pred - d_gt) / d_gt)

        if not errors:
            return {"dimension": "B1", "metric": "shot duration accuracy", "score": 0.0}

        score = clip01(1.0 - float(np.mean(errors)))
        return {
            "dimension": "B1",
            "metric": "shot duration accuracy",
            "score": score,
            "gt_durations": [round(d, 3) for d in gt_durations[:n]],
            "pred_durations": [round(d, 3) for d in pred_durations[:n]],
            "mean_relative_error": round(float(np.mean(errors)), 4),
            "method": "transnetv2_only",
        }


# ---- B2: beat synchronisation ----
def eval_b2_beat_sync(transnetv2_result: dict, demucs_result: dict,
                      video_path: str, raft_result: dict = None,
                      prompt: dict = None,
                      alignment_result: dict = None) -> dict:
    """
    B2: beat synchronisation (adjusted formula, two fused sub-metrics)
    Sub2: correlation between music energy and video motion energy
    Sub3: transition sound match (transition type vs audio impact)
    Final score = 0.65 * sub2 + 0.35 * sub3
    """
    import librosa
    import subprocess
    import tempfile
    from scipy import stats as scipy_stats

    # Cut times
    if alignment_result is not None:
        aligned_transitions = get_aligned_transitions(alignment_result)
        cut_times = [t["cut_time"] for t in aligned_transitions
                     if t["evaluable"] and t["cut_time"] is not None]
    else:
        cuts = transnetv2_result.get("transitions", [])
        cut_times = [(t["start_time"] + t["end_time"]) / 2 for t in cuts] if cuts else []

    # ===== Load audio =====
    # Try the music stem separated by Demucs first
    music_y = None
    mix_y = None
    SR = 22050

    source_files = demucs_result.get("source_files", {})
    demucs_sr = demucs_result.get("sample_rate", 44100)

    if source_files:
        try:
            # Load drums + bass + other as the music stem
            tracks = []
            for track_name in ["drums", "bass", "other"]:
                path = source_files.get(track_name, "")
                if path and os.path.exists(path):
                    y_track, _ = librosa.load(path, sr=SR, mono=True)
                    tracks.append(y_track)
            if tracks:
                # Align lengths and sum
                min_len = min(len(t) for t in tracks)
                music_y = np.sum([t[:min_len] for t in tracks], axis=0)
        except Exception:
            music_y = None

    # Extract the mixed audio (for sub3 and as a fallback)
    audio_path = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-vn", "-ar", str(SR),
            "-ac", "1", audio_path
        ], capture_output=True, timeout=30)
        if os.path.exists(audio_path):
            mix_y, _ = librosa.load(audio_path, sr=SR)
            os.unlink(audio_path)
    except Exception:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

    # Fall back to the mixed audio when there is no separated music stem
    if music_y is None:
        music_y = mix_y

    # ===== BGM presence check =====
    # Use the energy_ratio Demucs returns to help decide whether real background music exists
    has_bgm = True
    bgm_detail = {}
    demucs_sources = demucs_result.get("sources", {})
    if demucs_sources:
        # Energy ratio of the music stem (drums+bass+other) against the vocals stem
        music_ratio = sum(
            demucs_sources.get(t, {}).get("energy_ratio", 0.0)
            for t in ["drums", "bass", "other"]
        )
        vocals_ratio = demucs_sources.get("vocals", {}).get("energy_ratio", 0.0)
        bgm_detail["music_energy_ratio"] = round(music_ratio, 4)
        bgm_detail["vocals_energy_ratio"] = round(vocals_ratio, 4)

        # Rule: when vocals dominate (vocals_ratio > music_ratio), the source audio is mostly
        # speech / ambience and the "music" Demucs separated out is leakage
        if vocals_ratio > music_ratio * 1.5:
            has_bgm = False
            bgm_detail["reason"] = "vocals_dominant"
        # Extra check: even with a plausible energy ratio, a very low music-stem RMS means no BGM
        elif music_y is not None and len(music_y) > SR:
            music_rms_global = float(np.sqrt(np.mean(music_y ** 2)))
            bgm_detail["music_rms"] = round(music_rms_global, 6)
            if music_rms_global < 0.015:
                has_bgm = False
                bgm_detail["reason"] = "low_music_rms"

    bgm_detail["has_bgm"] = has_bgm

    # ===== Sub-metric 2: music energy vs video motion energy correlation =====
    sub2_score = 0.5
    sub2_detail = {}
    if not has_bgm:
        # No real background music (the model generated no BGM) -> score 0 as a penalty
        sub2_score = 0.0
        sub2_detail["note"] = "no_bgm_detected"
        sub2_detail["bgm_detection"] = bgm_detail
    elif music_y is not None and raft_result:
        motion_energy = raft_result.get("motion_energy", [])
        sample_fps = raft_result.get("sample_fps", 8.0)

        if len(motion_energy) >= 4:
            # Music RMS energy envelope (aligned to the RAFT frame rate)
            hop_samples = int(SR / sample_fps)
            music_rms_envelope = []
            for start in range(0, len(music_y) - hop_samples, hop_samples):
                seg = music_y[start:start + hop_samples]
                music_rms_envelope.append(float(np.sqrt(np.mean(seg ** 2))))

            # Align lengths
            n = min(len(music_rms_envelope), len(motion_energy))
            if n >= 4:
                music_env = np.array(music_rms_envelope[:n])
                motion_env = np.array(motion_energy[:n], dtype=np.float64)

                # Pearson correlation
                if np.std(music_env) > 1e-8 and np.std(motion_env) > 1e-8:
                    r, p_value = scipy_stats.pearsonr(music_env, motion_env)
                    sub2_score = clip01((r + 1.0) / 2.0)
                    sub2_detail["pearson_r"] = round(float(r), 4)
                    sub2_detail["p_value"] = round(float(p_value), 4)
                else:
                    sub2_detail["note"] = "constant_signal"

            sub2_detail["music_frames"] = len(music_rms_envelope)
            sub2_detail["motion_frames"] = len(motion_energy)

    # ===== Sub-metric 3: transition sound match =====
    sub3_score = 0.5
    sub3_detail = {}
    if mix_y is not None and prompt and cut_times:
        shots = prompt.get("shots", [])
        # Build the alignment info
        aligned_trans = None
        if alignment_result is not None:
            aligned_trans = get_aligned_transitions(alignment_result)

        sfx_scores = []
        for i, shot in enumerate(shots[:-1]):
            # Alignment check
            if aligned_trans is not None and i < len(aligned_trans):
                if not aligned_trans[i]["evaluable"]:
                    continue

            gt_effect = shot.get("transition_to_next", {}).get("optical_effect", "")
            if not gt_effect:
                continue

            # Cut time
            if aligned_trans is not None and i < len(aligned_trans):
                at = aligned_trans[i]
                ct = at["cut_time"] if at["cut_time"] is not None else parse_time_range(shot["description_prompt"])[1]
            else:
                ct = parse_time_range(shot["description_prompt"])[1]

            # Audio energy around the cut
            ct_sample = int(ct * SR)
            wide_window = int(0.5 * SR)   # ±0.5s
            narrow_window = int(0.2 * SR)  # ±0.2s

            wide_start = max(0, ct_sample - wide_window)
            wide_end = min(len(mix_y), ct_sample + wide_window)
            narrow_start = max(0, ct_sample - narrow_window)
            narrow_end = min(len(mix_y), ct_sample + narrow_window)

            if wide_end - wide_start < SR // 4:
                continue

            wide_rms = float(np.sqrt(np.mean(mix_y[wide_start:wide_end] ** 2)))
            narrow_rms = float(np.sqrt(np.mean(mix_y[narrow_start:narrow_end] ** 2)))

            # Energy impact ratio
            impact_ratio = narrow_rms / (wide_rms + 1e-8)

            # Score by transition type
            effect_lower = gt_effect.lower()
            if any(kw in effect_lower for kw in ["flash white", "flash-white", "white flash",
                                                   "flash black", "flash-black", "black flash"]):
                # flash types: an impact sound is expected
                if impact_ratio > 1.5:
                    sfx_s = 1.0
                elif impact_ratio > 1.0:
                    sfx_s = 0.6
                else:
                    sfx_s = 0.3
            elif any(kw in effect_lower for kw in ["dissolve", "cross-dissolve", "crossfade",
                                                     "fade", "blend", "morph"]):
                # dissolve types: expected to be smooth
                if impact_ratio < 1.3:
                    sfx_s = 1.0
                elif impact_ratio < 2.0:
                    sfx_s = 0.5
                else:
                    sfx_s = 0.2
            elif "wipe" in effect_lower or "iris" in effect_lower:
                sfx_s = 0.7
            else:
                # hard_cut or anything else
                sfx_s = 0.7

            sfx_scores.append(sfx_s)

        if sfx_scores:
            sub3_score = clip01(float(np.mean(sfx_scores)))
            sub3_detail["num_transitions_scored"] = len(sfx_scores)
            sub3_detail["per_transition"] = [round(s, 2) for s in sfx_scores]

    # ===== Final B2 score =====
    # Adjusted formula: weights the music-vs-motion energy correlation (sub2, from pearson_r)
    # and the transition sound match (sub3)
    final_score = 0.65 * sub2_score + 0.35 * sub3_score

    return {
        "dimension": "B2",
        "metric": "beat synchronization (energy correlation + sfx matching)",
        "score": clip01(final_score),
        "sub2_energy_corr": round(sub2_score, 4),
        "sub3_sfx_match": round(sub3_score, 4),
        "sub2_detail": sub2_detail,
        "sub3_detail": sub3_detail,
        "bgm_detection": bgm_detail,
        "num_cuts": len(cut_times),
    }


# ---- D1 sub-dimensions ----

# D1 transition type -> Mode A evaluation method routing
# Categories 1/2/3/4/7 belong to Mode A; the rest (5/6/8/9/10/11/12) are left to Mode B
D1_TYPE_ROUTER = {
    # Category 1: Visual Match -> DINOv2
    "match cut on visual": "similar_visual",
    "graphic match": "similar_visual",
    "visual match": "similar_visual",
    "similar visual": "similar_visual",
    "similar shot": "similar_visual",

    # Category 2: Match on Action -> RAFT
    "match on action": "similar_action",
    "action match": "similar_action",
    "similar action": "similar_action",

    # Category 3: Sound Match -> PANNs
    "match on sound": "similar_audio",
    "sound match": "similar_audio",
    "audio match": "similar_audio",
    "similar audio": "similar_audio",

    # Category 4: Contrast Cut -> DINOv2
    "contrast cut": "contrast",
    "contrasting cut": "contrast",

    # Category 7: Camera Move Cut -> migrated to Mode B (VLM + RAFT evidence)
    "camera move cut": None,
    "whip pan transition": None,
    "swish pan transition": None,
    "movement cut": None,

    # The following belong to Mode B (not evaluated in Mode A, placeholder None)
    "two extremes": None,                    # category 5: two extremes
    "polar shot cut": None,
    "extreme scale cut": None,               # category 5: two extremes (alias)
    "polar-scale cut": None,
    "smash cut": None,                       # category 6: smash cut
    "pov shot": None,                        # category 8: POV shot
    "eye-line match": None,
    "exit and entry cut": None,              # category 9: exit and entry
    "exit/entry cut": None,
    "occlusion cut": None,                   # category 10: occlusion cut
    "empty shot cut": None,                  # category 11: empty shot cut
    "logical cut": None,                     # category 12: logical cut
    "causal transition": None,
    "logical cut (causal transition)": None,
}


def route_d1_method(ctype: str) -> Optional[str]:
    """Route a cinematographic_type string to a Mode A method name; None when it is not a Mode A category."""
    if not ctype:
        return None
    key = ctype.lower().strip()
    # Exact match first
    if key in D1_TYPE_ROUTER:
        return D1_TYPE_ROUTER[key]
    # Then keyword containment
    for k, v in D1_TYPE_ROUTER.items():
        if k in key:
            return v
    return None


# ---------- per-cut sub-methods ----------

def _find_frame_idx_at(frame_times: list, cut_time: float) -> Tuple[int, int]:
    """Return the nearest frame index before the cut and the nearest one after it."""
    before_idx, after_idx = -1, -1
    for i, t in enumerate(frame_times):
        if t <= cut_time:
            before_idx = i
        elif after_idx == -1:
            after_idx = i
            break
    return before_idx, after_idx


def _eval_d1_visual_at(dinov2_result: dict, cut_time: float) -> Optional[float]:
    """Category 1: DINOv2 visual match -- cosine between the last frame before and the first frame after the cut."""
    embeddings = dinov2_result.get("embeddings", [])
    frame_times = dinov2_result.get("frame_times", [])
    if not embeddings or not frame_times:
        return None
    b, a = _find_frame_idx_at(frame_times, cut_time)
    if b < 0 or a < 0 or a >= len(embeddings):
        return None
    sim = cosine_similarity(embeddings[b], embeddings[a])
    return clip01(1.0 if sim >= 0.7 else sim / 0.7)


def _eval_d1_action_at(raft_result: dict, cut_time: float) -> Optional[float]:
    """Category 2: RAFT match on action -- cosine between the optical-flow patterns either side of the cut."""
    magnitudes = raft_result.get("flow_magnitudes", [])
    directions = raft_result.get("flow_directions", [])
    sample_fps = raft_result.get("sample_fps", 8.0)
    if len(magnitudes) < 4:
        return None
    flow_idx = int(cut_time * sample_fps)
    window = 2
    bs, be = max(0, flow_idx - window), flow_idx
    as_, ae = flow_idx, min(len(magnitudes), flow_idx + window)
    if be <= bs or ae <= as_:
        return None
    feat_b = magnitudes[bs:be] + directions[bs:be]
    feat_a = magnitudes[as_:ae] + directions[as_:ae]
    n = min(len(feat_b), len(feat_a))
    if n == 0:
        return None
    return clip01(cosine_similarity(feat_b[:n], feat_a[:n]))


def _eval_d1_audio_at(panns_result: dict, cut_time: float, total_duration: float,
                      seg_duration: float = 1.0) -> Optional[float]:
    """Category 3: PANNs sound match -- cosine between the segments either side of the cut."""
    seg_embeddings = panns_result.get("segment_embeddings", [])
    if len(seg_embeddings) < 2:
        return None
    # Locate the two segments around the cut
    seg_idx_before = int(cut_time / seg_duration) - 1
    seg_idx_after = int(cut_time / seg_duration)
    seg_idx_before = max(0, min(seg_idx_before, len(seg_embeddings) - 1))
    seg_idx_after = max(0, min(seg_idx_after, len(seg_embeddings) - 1))
    if seg_idx_before == seg_idx_after:
        seg_idx_after = min(seg_idx_before + 1, len(seg_embeddings) - 1)
    if seg_idx_before == seg_idx_after:
        return None
    return clip01(cosine_similarity(seg_embeddings[seg_idx_before],
                                    seg_embeddings[seg_idx_after]))


def _eval_d1_contrast_at(dinov2_result: dict, cut_time: float) -> Optional[float]:
    """Category 4: DINOv2 contrast cut -- 1 - cos_sim across the cut."""
    embeddings = dinov2_result.get("embeddings", [])
    frame_times = dinov2_result.get("frame_times", [])
    if not embeddings or not frame_times:
        return None
    b, a = _find_frame_idx_at(frame_times, cut_time)
    if b < 0 or a < 0 or a >= len(embeddings):
        return None
    sim = cosine_similarity(embeddings[b], embeddings[a])
    return clip01(1.0 - sim)


def _eval_d1_camera_at(raft_result: dict, cut_time: float) -> Optional[float]:
    """Category 7: RAFT camera move cut -- optical-flow magnitude around the cut."""
    magnitudes = raft_result.get("flow_magnitudes", [])
    sample_fps = raft_result.get("sample_fps", 8.0)
    if not magnitudes:
        return None
    flow_idx = int(cut_time * sample_fps)
    window = 3
    s, e = max(0, flow_idx - window), min(len(magnitudes), flow_idx + window)
    if e <= s:
        return None
    threshold = 5.0
    local_mag = float(np.mean(magnitudes[s:e]))
    return clip01(1.0 if local_mag > threshold else local_mag / threshold)


# ---------- D1 main routing ----------

def eval_d1_routed(prompt: dict,
                   dinov2_result: dict,
                   raft_result: dict,
                   panns_result: dict,
                   transnetv2_result: dict,
                   alignment_result: dict = None) -> dict:
    """
    D1: route to the matching Mode A method by the cinematographic_type annotated in the prompt.
    
    With an alignment_result, only transitions whose two neighbouring aligned shots both exist are evaluated,
    and the cut time comes from the aligned shot boundaries.
    Each transition is evaluated once, by the method its annotated type maps to, instead of running all 6 sub-dimensions.
    Mode B categories (logical / two extremes / smash / POV / exit-entry / occlusion / empty shot) are marked skipped and excluded from the D1 aggregate.
    """
    shots = prompt.get("shots", [])
    if len(shots) < 2:
        return {"dimension": "D1", "score": None, "note": "< 2 shots"}
    
    total_duration = transnetv2_result.get("duration", 15.0)
    
    # Build the aligned transition info (to decide which transitions are evaluable)
    aligned_transitions = None
    if alignment_result is not None:
        aligned_transitions = get_aligned_transitions(alignment_result)
    
    per_transition = []
    mode_a_scores = []  # only Mode A types contribute to the mean
    
    for i, shot in enumerate(shots[:-1]):
        trans = shot.get("transition_to_next", {})
        ctype = trans.get("cinematographic_type", "")
        method = route_d1_method(ctype)
        
        # Determine the cut time: prefer the alignment result
        if aligned_transitions is not None and i < len(aligned_transitions):
            at = aligned_transitions[i]
            if not at["evaluable"]:
                # A neighbouring shot is missing -> the transition scores 0 (only Mode A types count towards the mean)
                per_transition.append({
                    "shot_id": shot.get("shot_id"),
                    "cinematographic_type": ctype,
                    "cut_time": None,
                    "method": method,
                    "score": 0.0,
                    "evaluable_in_mode_a": False,
                    "note": "Adjacent shot missing → score 0",
                })
                if method is not None:
                    mode_a_scores.append(0.0)
                continue
            cut_time = at["cut_time"] if at["cut_time"] is not None else parse_time_range(shot["description_prompt"])[1]
        else:
            cut_time = parse_time_range(shot["description_prompt"])[1]
        
        entry = {
            "shot_id": shot.get("shot_id"),
            "cinematographic_type": ctype,
            "cut_time": round(cut_time, 3),
            "method": method,
            "score": None,
            "evaluable_in_mode_a": method is not None,
        }
        
        if method is None:
            entry["note"] = "Mode B category, not evaluated in the Mode A stage"
        else:
            try:
                if method == "similar_visual":
                    score = _eval_d1_visual_at(dinov2_result, cut_time)
                elif method == "similar_action":
                    score = _eval_d1_action_at(raft_result, cut_time)
                elif method == "similar_audio":
                    score = _eval_d1_audio_at(panns_result, cut_time, total_duration)
                elif method == "contrast":
                    score = _eval_d1_contrast_at(dinov2_result, cut_time)
                elif method == "camera_motion":
                    score = _eval_d1_camera_at(raft_result, cut_time)
                else:
                    score = None
                
                if score is not None:
                    entry["score"] = round(float(score), 4)
                    mode_a_scores.append(float(score))
                else:
                    entry["note"] = "data unavailable"
            except Exception as e:
                entry["error"] = str(e)
        
        per_transition.append(entry)
    
    # The aggregate averages only the transitions Mode A actually evaluated
    if mode_a_scores:
        d1_score = clip01(float(np.mean(mode_a_scores)))
    else:
        d1_score = None  # all Mode B types: D1 cannot be evaluated for this video in the Mode A stage
    
    return {
        "dimension": "D1",
        "metric": "transition quality (routed by GT type)",
        "score": d1_score,
        "num_transitions": len(per_transition),
        "num_mode_a_evaluated": len(mode_a_scores),
        "per_transition": per_transition,
    }



# ---- D2: optical transition type ----
def eval_d2_transition_type(transnetv2_result: dict, prompt: dict,
                            alignment_result: dict = None) -> dict:
    """
    D2: optical transition type detection (5-way)
    With an alignment_result, only transitions whose two neighbouring shots both exist are evaluated.
    
    The five supported types:
      - hard_cut
      - dissolve
      - wipe
      - flash_white
      - flash_black
    
    Scoring:
      - exact match: 1.0
      - dissolve <-> wipe confusion: 0.5
      - no transition matched: 0.0 (including a missed hard cut)
      - wrong type: 0.0
    """
    transitions_pred = transnetv2_result.get("transitions", [])
    shots = prompt.get("shots", [])
    
    if not shots:
        return {"dimension": "D2", "metric": "effect transition type", "score": 0.5}
    
    # Map the GT optical_effect onto the canonical 5-way labels
    def normalize_gt_type(optical_effect: str) -> str:
        """Normalise the prompt's optical_effect label onto the 5-way set."""
        effect_lower = optical_effect.lower().strip()
        
        # hard cut
        if any(kw in effect_lower for kw in ["hard cut", "straight cut", "hard-cut"]):
            return "hard_cut"
        # flash white
        if any(kw in effect_lower for kw in ["flash white", "flash-white", "white flash"]):
            return "flash_white"
        # flash black
        if any(kw in effect_lower for kw in ["flash black", "flash-black", "black flash",
                                              "fade to black", "fade-to-black"]):
            return "flash_black"
        # wipe
        if any(kw in effect_lower for kw in ["wipe", "iris"]):
            return "wipe"
        # dissolve (including the dissolve/fade/blend variants)
        if any(kw in effect_lower for kw in ["dissolve", "cross-dissolve", "fade",
                                              "blend", "morph", "crossfade"]):
            return "dissolve"
        
        # Default: an unrecognised label counts as a hard cut
        return "hard_cut"
    
    # Build the aligned transition info
    aligned_transitions = None
    if alignment_result is not None:
        aligned_transitions = get_aligned_transitions(alignment_result)

    scores = []
    details = []
    for i, shot in enumerate(shots[:-1]):  # the last shot has no transition
        # Alignment check: a missing neighbouring shot means this transition scores 0
        if aligned_transitions is not None and i < len(aligned_transitions):
            if not aligned_transitions[i]["evaluable"]:
                gt_effect = shot.get("transition_to_next", {}).get("optical_effect", "")
                if gt_effect:
                    scores.append(0.0)
                    details.append({"shot_idx": i, "score": 0.0,
                                    "note": "Adjacent shot missing → score 0"})
                continue

        gt_effect = shot.get("transition_to_next", {}).get("optical_effect", "")
        if not gt_effect:
            continue
        
        gt_type = normalize_gt_type(gt_effect)
        
        # Find the matching predicted transition (matched by time)
        # Prefer the aligned cut time
        if aligned_transitions is not None and i < len(aligned_transitions):
            at = aligned_transitions[i]
            shot_end_time = at["cut_time"] if at["cut_time"] is not None else parse_time_range(shot["description_prompt"])[1]
        else:
            shot_end_time = parse_time_range(shot["description_prompt"])[1]
        matched_pred = None
        for t in transitions_pred:
            if abs(t["start_time"] - shot_end_time) < 1.0:
                matched_pred = t
                break
        
        if matched_pred:
            pred_type = matched_pred.get("type", "")
            
            if pred_type == gt_type:
                # exact match
                scores.append(1.0)
                details.append(f"match: gt={gt_type}({gt_effect}), pred={pred_type}")
            elif {pred_type, gt_type} == {"dissolve", "wipe"}:
                # dissolve vs wipe confusion scores 0.5 (both are gradual, so they are hard to tell apart)
                scores.append(0.5)
                details.append(f"partial: gt={gt_type}({gt_effect}), pred={pred_type} (dissolve/wipe cross)")
            else:
                # wrong type
                scores.append(0.0)
                details.append(f"mismatch: gt={gt_type}({gt_effect}), pred={pred_type}")
        else:
            # No matching transition detected: always 0 (including a missed hard cut)
            scores.append(0.0)
            details.append(f"no_pred: gt={gt_type}({gt_effect})")
    
    if not scores:
        return {"dimension": "D2", "metric": "effect transition type", "score": 0.5}
    
    return {
        "dimension": "D2",
        "metric": "effect transition type",
        "score": clip01(float(np.mean(scores))),
        "per_transition": [round(s, 4) for s in scores],
        "details": details,
    }


# ---- D3: transition audio-visual relation ----
def _extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """MFCC feature vector (averaged over time, giving an n_mfcc-dim vector)."""
    import librosa
    if len(y) < sr * 0.1:  # less than 0.1s of data
        return np.zeros(n_mfcc)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


def _audio_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two feature vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _analyze_env_audio_continuity(
    env_y: np.ndarray, sr: int, cut_time: float,
    window: float = 0.3, gap: float = 0.3) -> dict:
    """
    Analyse the waveform continuity of the ambience track either side of the cut with librosa.

    MFCC features are taken from three windows:
      earlier: cut-gap-window ~ cut-gap  (the earlier span, for J-cut detection)
      before:  cut-window ~ cut          (before the cut)
      after:   cut ~ cut+window          (after the cut)

    Returns:
      sim_before_after: similarity of before vs after (high -> the sound continues across the cut)
      sim_earlier_before: similarity of earlier vs before (low -> a new sound may already appear in the before window)
      sim_earlier_after: similarity of earlier vs after (supplementary evidence)
      energy_ratio: RMS energy ratio of before vs after (low -> a large loudness change -> sync cut)
    """
    n_earlier_start = max(0, int((cut_time - gap - window) * sr))
    n_earlier_end = max(0, int((cut_time - gap) * sr))
    n_before_start = max(0, int((cut_time - window) * sr))
    n_before_end = max(0, int(cut_time * sr))
    n_after_start = min(len(env_y), int(cut_time * sr))
    n_after_end = min(len(env_y), int((cut_time + window) * sr))

    result = {
        "sim_before_after": 0.0,
        "sim_earlier_before": 0.0,
        "sim_earlier_after": 0.0,
        "energy_ratio": 0.0,
    }

    # Check there is enough data
    if n_before_end - n_before_start < int(sr * 0.05):
        return result
    if n_after_end - n_after_start < int(sr * 0.05):
        return result

    y_before = env_y[n_before_start:n_before_end]
    y_after = env_y[n_after_start:n_after_end]
    mfcc_before = _extract_mfcc_features(y_before, sr)
    mfcc_after = _extract_mfcc_features(y_after, sr)
    result["sim_before_after"] = _audio_cosine_similarity(mfcc_before, mfcc_after)

    # RMS energy ratio: min/max, the lower the value the bigger the loudness change
    rms_before = float(np.sqrt(np.mean(y_before ** 2)))
    rms_after = float(np.sqrt(np.mean(y_after ** 2)))
    if rms_before > 1e-6 and rms_after > 1e-6:
        result["energy_ratio"] = min(rms_before, rms_after) / max(rms_before, rms_after)
    else:
        result["energy_ratio"] = 0.0

    # The earlier window (when there is enough data)
    if n_earlier_end - n_earlier_start >= int(sr * 0.05):
        y_earlier = env_y[n_earlier_start:n_earlier_end]
        mfcc_earlier = _extract_mfcc_features(y_earlier, sr)
        result["sim_earlier_before"] = _audio_cosine_similarity(mfcc_earlier, mfcc_before)
        result["sim_earlier_after"] = _audio_cosine_similarity(mfcc_earlier, mfcc_after)

    return result


def _extract_d3_transition_clip(video_path: str, cut_time: float,
                                tmp_dir: str, transition_idx: int,
                                pre_sec: float = 1.2,
                                post_sec: float = 1.2) -> str:
    """Cut out a transition clip with audio, for the VLM to attribute the sound crossing the cut."""
    video_dur = _get_video_duration(video_path)
    start_sec = max(0.0, float(cut_time) - pre_sec)
    end_sec = float(cut_time) + post_sec
    if video_dur > 0:
        end_sec = min(video_dur, end_sec)
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0.2:
        raise ValueError(f"invalid D3 transition clip duration: {start_sec}-{end_sec}")

    out_path = os.path.join(tmp_dir, f"d3_transition_{transition_idx:03d}.mp4")
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-ac", "1",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=90)
    if result.returncode != 0 or not os.path.exists(out_path):
        stderr = result.stderr.decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"ffmpeg D3 transition extraction failed: {stderr}")
    return out_path


def _extract_json_from_vlm_response(response: str) -> Optional[dict]:
    """Extract a JSON object from a VLM response."""
    if not response or response.upper().strip().startswith("ERROR:"):
        return None
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    first_brace = response.find("{")
    last_brace = response.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(response[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass
    return None


_SYSTEM_D3_AUDIO_VISUAL_RELATION = """You are a strict film-sound editor.
Classify the ACTUAL perceived audio-visual relation at a visual cut from the provided video clip.
Do not assume the written prompt is correct. Prioritize what is audible and visible in the clip."""


def _normalize_d3_relation(value: str) -> str:
    text = (value or "").strip().lower()
    if "j" in text and "cut" in text:
        return "j-cut"
    if "l" in text and "cut" in text:
        return "l-cut"
    if "straight" in text or "sync" in text or "hard" in text:
        return "straight"
    if "overlap" in text and "unclear" in text:
        return "overlap-unclear"
    if "next" in text or "incoming" in text or "后一" in text or "下一" in text:
        return "j-cut"
    if "previous" in text or "outgoing" in text or "前一" in text or "上一" in text:
        return "l-cut"
    return "unclear"


def _slice_audio_window(y: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    if y is None or sr <= 0:
        return np.array([], dtype=np.float32)
    start = max(0, int(start_sec * sr))
    end = min(len(y), int(end_sec * sr))
    if end <= start:
        return np.array([], dtype=np.float32)
    return y[start:end]


def _rms_energy(y: np.ndarray) -> float:
    if y is None or len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))


def _dbfs_from_rms(rms: float) -> float:
    return 20.0 * float(np.log10(max(float(rms), 1e-9)))


D3_OBJECT_SEGMENT_DURATION = 0.5
D3_OBJECT_MIN_DBFS = -42.0
D3_OBJECT_WEAK_NOISE_DBFS = -34.0
D3_OBJECT_MIN_TAG_CONFIDENCE = 0.12
D3_OBJECT_RUN_GAP = 0.6
D3_OBJECT_SYNC_ONSET_THRESHOLD = 0.25
D3_OBJECT_SYNC_TAIL_THRESHOLD = _get_env_float("D3_OBJECT_SYNC_TAIL_THRESHOLD", 0.25)

_D3_OBJECT_SALIENT_KEYWORDS = (
    "guitar", "string", "strum", "pluck", "piano", "keyboard", "drum", "percussion",
    "cymbal", "bell", "chime", "click", "clack", "knock", "tap", "thump", "bang",
    "crash", "slam", "impact", "squeak", "creak", "engine", "motor", "machine",
    "vehicle", "door", "footstep", "skate", "ball", "whistle",
)
_D3_OBJECT_DEFAULT_STRAIGHT_KEYWORDS = (
    "silence", "wind", "water", "rain", "stream", "ocean", "sea", "waves", "noise",
    "hum", "static", "rustle", "background", "ambience", "environment", "room tone",
)
_D3_OBJECT_FAMILY_KEYWORDS = (
    ("guitar", ("guitar", "string", "strum", "pluck")),
    ("percussion", ("drum", "percussion", "cymbal", "tap", "knock", "click", "clack")),
    ("impact", ("impact", "thump", "bang", "crash", "slam", "clack")),
    ("machine", ("engine", "motor", "machine", "vehicle")),
)


def _tag_label_text(tags: List[dict]) -> str:
    return " ".join(str(t.get("label", "")).lower() for t in tags or [])


def _object_sound_family(tags: List[dict]) -> str:
    label_text = _tag_label_text(tags)
    for family, keywords in _D3_OBJECT_FAMILY_KEYWORDS:
        if any(keyword in label_text for keyword in keywords):
            return family
    if label_text.strip():
        return "object"
    return "unknown"


def _is_default_straight_noise(tags: List[dict], dbfs: float) -> bool:
    label_text = _tag_label_text(tags)
    if dbfs < D3_OBJECT_MIN_DBFS:
        return True
    has_noise = any(keyword in label_text for keyword in _D3_OBJECT_DEFAULT_STRAIGHT_KEYWORDS)
    has_salient = any(keyword in label_text for keyword in _D3_OBJECT_SALIENT_KEYWORDS)
    return bool(has_noise and (dbfs < D3_OBJECT_WEAK_NOISE_DBFS or not has_salient))


def _is_salient_object_sound(tags: List[dict], dbfs: float) -> bool:
    if not tags or dbfs < D3_OBJECT_MIN_DBFS or _is_default_straight_noise(tags, dbfs):
        return False
    for tag in tags:
        label = str(tag.get("label", "")).lower()
        try:
            confidence = float(tag.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence >= D3_OBJECT_MIN_TAG_CONFIDENCE and any(
            keyword in label for keyword in _D3_OBJECT_SALIENT_KEYWORDS
        ):
            return True
    return False


def _build_object_sound_windows(panns_by_track: dict, track_audio: dict, track_sr: dict) -> List[dict]:
    windows = []
    for track_name, panns_result in (panns_by_track or {}).items():
        segment_tags = panns_result.get("segment_tags", []) or []
        segment_times = panns_result.get("segment_times", []) or []
        for idx, tags in enumerate(segment_tags):
            time_info = segment_times[idx] if idx < len(segment_times) else {}
            try:
                start = float(time_info.get("start", idx * D3_OBJECT_SEGMENT_DURATION))
                end = float(time_info.get("end", start + D3_OBJECT_SEGMENT_DURATION))
            except (TypeError, ValueError):
                continue
            y = track_audio.get(track_name)
            sr = track_sr.get(track_name, 0)
            rms = _rms_energy(_slice_audio_window(y, sr, start, end)) if y is not None and sr else 0.0
            dbfs = _dbfs_from_rms(rms)
            salient = _is_salient_object_sound(tags, dbfs)
            windows.append({
                "track": track_name,
                "index": idx,
                "start": round(start, 3),
                "end": round(end, 3),
                "dbfs": round(dbfs, 2),
                "tags": tags[:5],
                "family": _object_sound_family(tags),
                "salient": salient,
                "default_straight_noise": _is_default_straight_noise(tags, dbfs),
            })
    windows.sort(key=lambda item: (item["start"], item["end"], item["track"]))
    return windows


def _merge_object_sound_runs(windows: List[dict]) -> List[dict]:
    runs = []
    for win in [w for w in windows if w.get("salient")]:
        family = win.get("family", "object")
        track = win.get("track", "")
        current = runs[-1] if runs else None
        if (current and current.get("family") == family and current.get("track") == track and
                float(win["start"]) - float(current["end"]) <= D3_OBJECT_RUN_GAP):
            current["end"] = max(float(current["end"]), float(win["end"]))
            current["max_dbfs"] = max(float(current["max_dbfs"]), float(win["dbfs"]))
            current["windows"].append(win)
            current["top_tags"] = current["top_tags"] or win.get("tags", [])[:3]
        else:
            runs.append({
                "track": track,
                "family": family,
                "start": float(win["start"]),
                "end": float(win["end"]),
                "max_dbfs": float(win["dbfs"]),
                "top_tags": win.get("tags", [])[:3],
                "windows": [win],
            })
    for run in runs:
        run["duration"] = round(float(run["end"]) - float(run["start"]), 4)
        run["start"] = round(float(run["start"]), 4)
        run["end"] = round(float(run["end"]), 4)
        run["max_dbfs"] = round(float(run["max_dbfs"]), 2)
        run["windows"] = run["windows"][:8]
    return runs


def _analyze_object_sound_projection(object_runs: List[dict], object_windows: List[dict], cut_time: float,
                                     prev_shot_range: Optional[Tuple[float, float]],
                                     next_shot_range: Optional[Tuple[float, float]]) -> dict:
    base = {
        "object_sound_crosses_cut": False,
        "object_sound_projection_mode": "none",
        "object_sound_recommended_owner": "none",
        "object_sound_recommended_relation": "straight",
        "object_sound_reason": "no_salient_object_sound_crossing_cut",
        "object_sound_projection": None,
        "object_sound_windows_near_cut": [w for w in object_windows if abs(float(w.get("start", 0.0)) - cut_time) <= 1.0][:12],
    }
    if not prev_shot_range or not next_shot_range:
        base["object_sound_reason"] = "missing_shot_range"
        return base
    prev_start, prev_end = map(float, prev_shot_range)
    next_start, next_end = map(float, next_shot_range)
    best = None
    best_score = -1.0
    for run in object_runs or []:
        run_start = float(run.get("start", 0.0))
        run_end = float(run.get("end", 0.0))
        prev_overlap = _overlap_duration(run_start, run_end, prev_start, prev_end)
        next_overlap = _overlap_duration(run_start, run_end, next_start, next_end)
        crosses_cut = run_start < cut_time < run_end or (prev_overlap > 0 and next_overlap > 0)
        if not crosses_cut:
            continue
        if abs(run_start - cut_time) <= D3_OBJECT_SYNC_ONSET_THRESHOLD:
            recommended_owner = "none"
            recommended_relation = "straight"
            reason = "salient_object_onset_sync_with_cut"
        elif prev_overlap > 1.0 and next_overlap > 1.0 and min(prev_overlap, next_overlap) / max(prev_overlap, next_overlap) >= 0.5:
            recommended_owner = "none"
            recommended_relation = "straight"
            reason = "continuous_object_bed_spans_both_shots_without_clear_ownership"
        elif prev_overlap > next_overlap:
            recommended_owner = "previous"
            recommended_relation = "l-cut"
            reason = "salient_object_previous_sound_continues_after_cut"
        elif next_overlap > prev_overlap:
            recommended_owner = "next"
            recommended_relation = "j-cut"
            reason = "salient_object_next_sound_heard_before_or_through_cut"
        else:
            recommended_owner = "unclear"
            recommended_relation = "overlap-unclear"
            reason = "balanced_object_sound_overlap"
        score = min(prev_overlap, next_overlap) + 0.2 * max(prev_overlap, next_overlap) + max(0.0, float(run.get("max_dbfs", -80)) + 80) / 100.0
        candidate = {
            "track": run.get("track"),
            "family": run.get("family"),
            "run_start": round(run_start, 4),
            "run_end": round(run_end, 4),
            "onset_offset": round(run_start - cut_time, 4),
            "prev_overlap_duration": round(prev_overlap, 4),
            "next_overlap_duration": round(next_overlap, 4),
            "recommended_owner": recommended_owner,
            "recommended_relation": recommended_relation,
            "max_dbfs": run.get("max_dbfs"),
            "top_tags": run.get("top_tags", []),
            "reason": reason,
        }
        if score > best_score:
            best_score = score
            best = candidate
    if not best:
        return base
    return {
        "object_sound_crosses_cut": True,
        "object_sound_projection_mode": "cross_salient_object",
        "object_sound_recommended_owner": best["recommended_owner"],
        "object_sound_recommended_relation": best["recommended_relation"],
        "object_sound_reason": best["reason"],
        "object_sound_projection": best,
        "object_sound_windows_near_cut": base["object_sound_windows_near_cut"],
    }


D3_SYNC_STRAIGHT_THRESHOLD = 0.15
D3_WORD_GAP_SPLIT_THRESHOLD = 0.1


def _detect_onset_near_cut(y: np.ndarray, sr: int, cut_time: float,
                           radius: float = 0.45) -> dict:
    """Detect whether there is a clear onset / transient event near the cut."""
    result = {
        "onset_near_cut": False,
        "onset_offset": None,
        "onset_strength": 0.0,
    }
    if y is None or sr <= 0 or len(y) < int(sr * 0.1):
        return result
    try:
        import librosa
        start_sec = max(0.0, cut_time - radius)
        end_sec = min(len(y) / sr, cut_time + radius)
        window = _slice_audio_window(y, sr, start_sec, end_sec)
        if len(window) < int(sr * 0.1):
            return result
        onset_env = librosa.onset.onset_strength(y=window, sr=sr)
        if onset_env.size == 0:
            return result
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr) + start_sec
        local_mean = float(np.mean(onset_env))
        local_std = float(np.std(onset_env))
        idx = int(np.argmax(onset_env))
        peak_time = float(times[idx])
        peak_strength = float(onset_env[idx])
        strength_norm = peak_strength / max(local_mean + local_std, 1e-6)
        offset = peak_time - cut_time
        result.update({
            "onset_near_cut": bool(abs(offset) <= radius and strength_norm >= 1.8),
            "onset_offset": round(offset, 4),
            "onset_strength": round(strength_norm, 4),
        })
    except Exception:
        pass
    return result


def _find_word_gap_near_cut(words: List[dict], cut_time: float,
                            sync_threshold: float = D3_SYNC_STRAIGHT_THRESHOLD,
                            gap_threshold: float = D3_WORD_GAP_SPLIT_THRESHOLD) -> dict:
    valid_words = []
    for word in words or []:
        try:
            start = float(word.get("start", 0.0))
            end = float(word.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end >= start:
            valid_words.append({
                "word": str(word.get("word", "")).strip(),
                "start": start,
                "end": end,
            })
    valid_words.sort(key=lambda item: (item["start"], item["end"]))

    best_gap = None
    best_distance = None
    for prev_word, next_word in zip(valid_words, valid_words[1:]):
        gap_start = prev_word["end"]
        gap_end = next_word["start"]
        gap = gap_end - gap_start
        if gap < gap_threshold:
            continue
        sentence_end = gap_start
        distance = abs(sentence_end - cut_time)
        if distance > sync_threshold:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_gap = {
                "speech_word_gap_split": True,
                "speech_word_gap": round(gap, 4),
                "speech_word_gap_start": round(gap_start, 4),
                "speech_word_gap_end": round(gap_end, 4),
                "speech_sentence_end": round(sentence_end, 4),
                "speech_sentence_end_offset": round(sentence_end - cut_time, 4),
                "speech_sentence_end_cut_distance": round(distance, 4),
                "speech_word_gap_prev_word": prev_word["word"],
                "speech_word_gap_next_word": next_word["word"],
            }

    if best_gap:
        return best_gap
    return {
        "speech_word_gap_split": False,
        "speech_word_gap": 0.0,
        "speech_word_gap_start": None,
        "speech_word_gap_end": None,
        "speech_sentence_end": None,
        "speech_sentence_end_offset": None,
        "speech_sentence_end_cut_distance": None,
        "speech_word_gap_prev_word": "",
        "speech_word_gap_next_word": "",
    }


def _overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _analyze_speech_projection(segments: List[dict], cut_time: float,
                               prev_shot_range: Optional[Tuple[float, float]],
                               next_shot_range: Optional[Tuple[float, float]]) -> dict:
    base = {
        "speech_projection_crosses_cut": False,
        "speech_projection_mode": "none",
        "speech_projection_recommended_owner": "none",
        "speech_projection_reason": "no_cross_shot_speech",
        "speech_projection": None,
    }
    if not prev_shot_range or not next_shot_range:
        base["speech_projection_reason"] = "missing_shot_range"
        return base

    prev_start, prev_end = map(float, prev_shot_range)
    next_start, next_end = map(float, next_shot_range)
    prev_duration = max(prev_end - prev_start, 1e-6)
    next_duration = max(next_end - next_start, 1e-6)
    best = None
    best_score = -1.0

    for seg in segments or []:
        try:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if seg_end <= seg_start:
            continue

        prev_overlap = _overlap_duration(seg_start, seg_end, prev_start, prev_end)
        next_overlap = _overlap_duration(seg_start, seg_end, next_start, next_end)
        if prev_overlap <= 0.0 and next_overlap <= 0.0:
            continue
        if prev_overlap > 0.0 and next_overlap <= 0.0:
            continue
        if next_overlap > 0.0 and prev_overlap <= 0.0:
            continue

        prev_ratio = clip01(prev_overlap / prev_duration)
        next_ratio = clip01(next_overlap / next_duration)
        both_full = bool(prev_ratio >= 0.95 and next_ratio >= 0.95)
        if both_full:
            mode = "cross_both_full"
            recommended_owner = "unclear"
            reason = "same_speech_covers_both_complete_shots"
            bias_strength = 0.0
        else:
            mode = "cross_partial"
            if prev_overlap > next_overlap:
                recommended_owner = "previous"
                reason = "previous_shot_has_longer_speech_coverage"
            elif next_overlap > prev_overlap:
                recommended_owner = "next"
                reason = "next_shot_has_longer_speech_coverage"
            else:
                recommended_owner = "unclear"
                reason = "speech_coverage_is_balanced"
            bias_strength = abs(prev_overlap - next_overlap) / max(prev_overlap + next_overlap, 1e-6)

        score = min(prev_overlap, next_overlap) + 0.25 * (prev_overlap + next_overlap)
        candidate = {
            "segment_id": seg.get("id"),
            "segment_start": round(seg_start, 4),
            "segment_end": round(seg_end, 4),
            "segment_text": str(seg.get("text", ""))[:300],
            "prev_shot_range": [round(prev_start, 4), round(prev_end, 4)],
            "next_shot_range": [round(next_start, 4), round(next_end, 4)],
            "prev_overlap_duration": round(prev_overlap, 4),
            "next_overlap_duration": round(next_overlap, 4),
            "prev_coverage_ratio": round(prev_ratio, 4),
            "next_coverage_ratio": round(next_ratio, 4),
            "both_shots_fully_covered": both_full,
            "mode": mode,
            "recommended_owner": recommended_owner,
            "bias_strength": round(clip01(bias_strength), 4),
            "reason": reason,
        }
        if score > best_score:
            best_score = score
            best = candidate

    if not best:
        return base

    return {
        "speech_projection_crosses_cut": True,
        "speech_projection_mode": best["mode"],
        "speech_projection_recommended_owner": best["recommended_owner"],
        "speech_projection_reason": best["reason"],
        "speech_projection": best,
    }


def _analyze_speech_overlap(segments: List[dict], words: List[dict], vocals_y: np.ndarray, vocals_sr: int,
                            cut_time: float, window: float = 0.35,
                            prev_shot_range: Optional[Tuple[float, float]] = None,
                            next_shot_range: Optional[Tuple[float, float]] = None) -> dict:
    """Decide from Whisper timestamps and local vocals energy whether speech really crosses the cut."""
    crossing_segments = []
    nearest_boundary_offset = None
    max_before = 0.0
    max_after = 0.0
    for seg in segments or []:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        for boundary in (start, end):
            offset = boundary - cut_time
            if abs(offset) <= window:
                if nearest_boundary_offset is None or abs(offset) < abs(nearest_boundary_offset):
                    nearest_boundary_offset = offset
        if start < cut_time < end:
            before_dur = cut_time - start
            after_dur = end - cut_time
            max_before = max(max_before, before_dur)
            max_after = max(max_after, after_dur)
            crossing_segments.append(seg)

    word_gap = _find_word_gap_near_cut(words, cut_time)
    speech_projection = _analyze_speech_projection(segments, cut_time, prev_shot_range, next_shot_range)
    if word_gap.get("speech_word_gap_split"):
        sentence_end_offset = word_gap.get("speech_sentence_end_offset")
        if sentence_end_offset is not None:
            nearest_boundary_offset = sentence_end_offset

    before_y = _slice_audio_window(vocals_y, vocals_sr, cut_time - window, cut_time)
    after_y = _slice_audio_window(vocals_y, vocals_sr, cut_time, cut_time + window)
    before_energy = _rms_energy(before_y)
    after_energy = _rms_energy(after_y)
    energy_ratio = 0.0
    if before_energy > 1e-7 and after_energy > 1e-7:
        energy_ratio = min(before_energy, after_energy) / max(before_energy, after_energy)

    has_local_voice_energy = before_energy > 1e-4 and after_energy > 1e-4 and energy_ratio > 0.25
    has_balanced_segment = max_before >= 0.15 and max_after >= 0.15
    speech_overlap = bool(crossing_segments and not word_gap.get("speech_word_gap_split") and has_balanced_segment and has_local_voice_energy)
    confidence = 0.0
    if crossing_segments:
        duration_score = clip01(min(max_before, max_after) / 0.5)
        energy_score = clip01(energy_ratio)
        confidence = 0.5 * duration_score + 0.5 * energy_score

    return {
        "speech_overlap": speech_overlap,
        "speech_overlap_confidence": round(confidence, 4),
        "speech_before_energy": round(before_energy, 6),
        "speech_after_energy": round(after_energy, 6),
        "speech_energy_ratio": round(energy_ratio, 4),
        "speech_crossing_segments": crossing_segments,
        **word_gap,
        **speech_projection,
        "speech_nearest_boundary_offset": round(nearest_boundary_offset, 4) if nearest_boundary_offset is not None else None,
        "speech_boundary_sync_within_0p15": bool(
            word_gap.get("speech_word_gap_split") and
            word_gap.get("speech_sentence_end_offset") is not None and
            abs(float(word_gap.get("speech_sentence_end_offset"))) <= D3_SYNC_STRAIGHT_THRESHOLD
        ),
        "speech_max_before": round(max_before, 4),
        "speech_max_after": round(max_after, 4),
    }


# ============================================================
#  D3 ambience channel (ambient projection) -- kept consistent with the edit_baseline agent self-eval
#
#  Anchored on the picture cut, take four blocks A/B/C/D (each of width delta) and locate the "audio cut" by mel spectral-shape distance:
#     A=[cut-2δ,cut-δ]  B=[cut-δ,cut]  C=[cut,cut+δ]  D=[cut+δ,cut+2δ]
#  A is always the previous shot alone and D always the next shot alone, so:
#     straight: B=P,   C=N     -> the strongest jump is at B|C, of the same magnitude as d(A,D)
#     J-cut:    B=P+N, C=N     -> B leans clearly towards next
#     L-cut:    B=P,   C=P+N   -> C leans clearly towards prev
#  Enabled when at least one shot lacks trusted speech; arbitration order: speech projection > ambience > object sound > sync change.
# ------------------------------------------------------------
# delta: must match JL_CUT_OFFSET_SECONDS in the edit_baseline renderer (0.8s), otherwise the four blocks
# miss the real overlap spans. benchmark does not depend on edit_baseline, so the same constant is inlined here.
# Only the global constant is used; no individual transition's GT is read (a per-transition timing_offset maps 1:1 to its relation, so reading it would be reading GT).
D3_AMBIENT_OFFSET_SECONDS = _get_env_float("D3_AMBIENT_OFFSET_SECONDS", 0.8)
# How many times stronger d(B,C) must be than the two candidate boundaries to conclude that the audio cut sits on the picture cut
D3_AMBIENT_DOMINANCE = _get_env_float("D3_AMBIENT_DOMINANCE", 1.5)
# A straight cut is one complete replacement, so d(B,C) should be of the same magnitude as the two shots' own ambience difference d(A,D) (0.85 leaves room for overlap-estimation noise);
# J/L only add or remove one source, so d(B,C) is clearly smaller than d(A,D).
D3_AMBIENT_FULL_REPLACE_RATIO = _get_env_float("D3_AMBIENT_FULL_REPLACE_RATIO", 0.85)
# When the two shots' ambience is virtually identical (digital silence / one noise floor throughout), draw no conclusion and hand back to VLM / fallback
D3_AMBIENT_MIN_CONTRAST = _get_env_float("D3_AMBIENT_MIN_CONTRAST", 0.01)
# Minimum speech duration (seconds) to accept "this shot has a voice"; Whisper hallucinations are usually very short
D3_AMBIENT_SPEECH_MIN_DURATION = _get_env_float("D3_AMBIENT_SPEECH_MIN_DURATION", 0.35)
# Whisper hallucinates reliably on pure ambience: word-level cases (word probability 0.0~0.02) and whole-file gibberish segments
# (confidence 0.54, word probability 0.27~0.32). Such fake speech masks the ambience channel and drives a wrong J/L attribution.
# So D3 keeps only TRUSTED speech: segment confidence above threshold AND mean word probability above threshold (real dialogue is usually 0.6~0.9).
D3_SPEECH_MIN_WORD_PROB = _get_env_float("D3_SPEECH_MIN_WORD_PROB", 0.5)
D3_SPEECH_MIN_SEGMENT_CONFIDENCE = _get_env_float("D3_SPEECH_MIN_SEGMENT_CONFIDENCE", 0.5)
_AMBIENT_SR = 22050
_AMBIENT_HOP = 256
_AMBIENT_N_MELS = 48


def _load_audio_mono(path: str, sr: int = 22050):
    """Load mono audio.

    librosa/soundfile cannot read an mp4 container and falls back to the very slow audioread. Decoding to a
    temporary wav with ffmpeg first is one or two orders of magnitude faster and raises no deprecation warning.
    """
    import librosa

    if path.lower().endswith((".wav", ".flac", ".ogg")):
        return librosa.load(path, sr=sr, mono=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
             "-f", "wav", tmp.name],
            capture_output=True, timeout=180,
        )
        if r.returncode != 0 or os.path.getsize(tmp.name) == 0:
            return np.array([], dtype=np.float32), sr
        return librosa.load(tmp.name, sr=sr, mono=True)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _ambient_features(y: np.ndarray, sr: int):
    """Loudness curve and spectral shape curve in the mel domain.

    level uses the dB of the total energy (not the mean of per-band dB), otherwise quiet bands would flatten a
    real several-dB step; shape drops overall loudness and keeps only the frequency distribution, to judge "is this the same source".
    """
    import librosa

    if y is None or len(y) < int(sr * 0.2):
        return None, None, None
    S = np.maximum(librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=_AMBIENT_HOP, n_mels=_AMBIENT_N_MELS), 1e-12)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=_AMBIENT_HOP)
    level = 10.0 * np.log10(S.sum(axis=0))
    log_s = librosa.power_to_db(S, ref=1.0)
    shape = log_s - log_s.mean(axis=0, keepdims=True)
    return times, level, shape


def _ambient_block(times, level, shape, start: float, end: float) -> Optional[dict]:
    mask = (times >= start) & (times < end)
    if int(mask.sum()) < 3:
        return None
    return {"shape": shape[:, mask].mean(axis=1), "level": float(np.mean(level[mask])),
            "n": int(mask.sum())}


def _ambient_dissim(a: Optional[dict], b: Optional[dict]) -> Optional[float]:
    """Spectral shape dissimilarity (1-cosine) between two ambience blocks.

    Returns None when either block is truly silent (its loudness-free shape vector is near zero): the cosine
    is undefined there and forcing it would yield 1.0, which looks like a huge jump and turns digital silence into strong evidence.
    """
    if a is None or b is None:
        return None
    sa, sb = a["shape"], b["shape"]
    na, nb = float(np.linalg.norm(sa)), float(np.linalg.norm(sb))
    if na < 1e-6 or nb < 1e-6:
        return None
    return 1.0 - float(np.dot(sa, sb) / max(na * nb, 1e-9))


def _filter_reliable_speech(segments: List[dict],
                            words: List[dict]) -> Tuple[List[dict], List[dict], dict]:
    """Drop Whisper hallucination segments; returns (trusted segments, trusted words, statistics).

    Criteria: segment confidence >= D3_SPEECH_MIN_SEGMENT_CONFIDENCE and the mean word-level probability
    >= D3_SPEECH_MIN_WORD_PROB (confidence alone when there is no word-level data). The mean, not the max:
    a gibberish segment can have one word sneak above 0.3 while its mean stays far below real dialogue.
    """
    kept_segments: List[dict] = []
    dropped: List[dict] = []
    kept_ids = set()
    for seg in segments or []:
        text = str(seg.get("text", "") or "").strip()
        try:
            confidence = float(seg.get("confidence", 1.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        seg_words = seg.get("words") or []
        probs = []
        for w in seg_words:
            try:
                probs.append(float(w.get("probability", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        word_ok = (float(np.mean(probs)) >= D3_SPEECH_MIN_WORD_PROB) if probs else True
        if text and confidence >= D3_SPEECH_MIN_SEGMENT_CONFIDENCE and word_ok:
            kept_segments.append(seg)
            if seg.get("id") is not None:
                kept_ids.add(seg.get("id"))
        else:
            dropped.append({"start": seg.get("start"), "end": seg.get("end"),
                            "text": text[:60], "confidence": round(confidence, 4),
                            "mean_word_prob": round(float(np.mean(probs)), 4) if probs else None})

    kept_words: List[dict] = []
    for w in words or []:
        seg_id = w.get("segment_id")
        if seg_id is not None and kept_ids:
            if seg_id in kept_ids:
                kept_words.append(w)
            continue
        try:
            prob = float(w.get("probability", 0.0) or 0.0)
        except (TypeError, ValueError):
            prob = 0.0
        if prob >= D3_SPEECH_MIN_WORD_PROB:
            kept_words.append(w)

    stats = {"n_segments_in": len(segments or []), "n_segments_kept": len(kept_segments),
             "n_words_in": len(words or []), "n_words_kept": len(kept_words),
             "dropped_segments": dropped[:10],
             "min_segment_confidence": D3_SPEECH_MIN_SEGMENT_CONFIDENCE,
             "min_word_probability": D3_SPEECH_MIN_WORD_PROB}
    return kept_segments, kept_words, stats


def _shot_has_speech(segments: List[dict], shot_range: Optional[Tuple[float, float]]) -> bool:
    """Whether this shot span carries usable speech (cumulative speech duration reaches the threshold)."""
    if not shot_range:
        return False
    start, end = float(shot_range[0]), float(shot_range[1])
    total = 0.0
    for seg in segments or []:
        try:
            s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        total += _overlap_duration(s, e, start, end)
    return total >= D3_AMBIENT_SPEECH_MIN_DURATION


def _analyze_ambient_projection(times, level, shape, cut_time: float,
                                anchor_time: Optional[float] = None,
                                delta: float = D3_AMBIENT_OFFSET_SECONDS) -> dict:
    """Locate the audio cut from ambience and attribute it, returning a suggested relation, or unclear.

    anchor_time: the blocking anchor, by default the detected picture cut cut_time (the perceived J/L offset is
    really the audio boundary relative to the cut the viewer sees, so anchoring on the detected cut is more accurate).
    """
    anchor = float(cut_time if anchor_time is None else anchor_time)
    base = {
        "ambient_available": False,
        "ambient_recommended_relation": "unclear",
        "ambient_boundary_offset": None,
        "ambient_reason": "ambient_track_unavailable",
        "ambient_offset_seconds": round(float(delta), 4),
        "ambient_anchor_time": round(anchor, 4),
    }
    if times is None or shape is None:
        return base

    blocks = {
        "A": _ambient_block(times, level, shape, anchor - 2 * delta, anchor - delta),
        "B": _ambient_block(times, level, shape, anchor - delta, anchor),
        "C": _ambient_block(times, level, shape, anchor, anchor + delta),
        "D": _ambient_block(times, level, shape, anchor + delta, anchor + 2 * delta),
    }
    d_ab = _ambient_dissim(blocks["A"], blocks["B"])
    d_bc = _ambient_dissim(blocks["B"], blocks["C"])
    d_cd = _ambient_dissim(blocks["C"], blocks["D"])
    d_ad = _ambient_dissim(blocks["A"], blocks["D"])
    detail = {
        "ambient_block_dissim": {
            "cut_minus_delta": None if d_ab is None else round(d_ab, 5),
            "cut": None if d_bc is None else round(d_bc, 5),
            "cut_plus_delta": None if d_cd is None else round(d_cd, 5),
            "prev_vs_next": None if d_ad is None else round(d_ad, 5),
        },
        "ambient_block_levels": {
            k: (None if v is None else round(v["level"], 2)) for k, v in blocks.items()},
        "ambient_offset_seconds": round(float(delta), 4),
        "ambient_anchor_time": round(anchor, 4),
    }
    if d_bc is None or d_ab is None or d_cd is None or d_ad is None:
        # a transition next to the first/last shot has no complete A/D reference blocks
        return {**base, **detail, "ambient_reason": "insufficient_reference_blocks"}
    if max(d_ad, d_bc) < D3_AMBIENT_MIN_CONTRAST:
        # the two shots' ambience is indistinguishable (common when generation returns near digital silence), so any attribution is noise
        return {**base, **detail, "ambient_reason": "ambient_flat_no_contrast"}

    side = max(d_ab, d_cd)
    dominance = d_bc / max(side, 1e-9)
    full_ratio = d_bc / max(d_ad, 1e-9)
    detail.update({
        "ambient_cut_dominance": round(dominance, 4),
        "ambient_full_replace_ratio": round(full_ratio, 4),
    })
    if dominance >= D3_AMBIENT_DOMINANCE and full_ratio >= D3_AMBIENT_FULL_REPLACE_RATIO:
        return {
            **detail,
            "ambient_available": True,
            "ambient_recommended_relation": "straight",
            "ambient_boundary_offset": round(anchor - float(cut_time), 4),
            "ambient_reason": "ambient_boundary_on_cut_full_replacement",
        }

    # not a complete replacement: see which of B / C is the overlaid block (attribution leans to the opposite shot)
    d_bd = _ambient_dissim(blocks["B"], blocks["D"])
    d_ca = _ambient_dissim(blocks["C"], blocks["A"])
    if d_bd is None or d_ca is None:
        return {**base, **detail, "ambient_reason": "insufficient_reference_blocks"}
    j_score = (d_ab - d_bd) / max(d_ab + d_bd, 1e-9)
    l_score = (d_cd - d_ca) / max(d_cd + d_ca, 1e-9)
    detail.update({"ambient_j_score": round(j_score, 4), "ambient_l_score": round(l_score, 4)})
    if j_score >= l_score:
        return {**detail, "ambient_available": True,
                "ambient_recommended_relation": "j-cut",
                "ambient_boundary_offset": round(anchor - float(delta) - float(cut_time), 4),
                "ambient_reason": "ambient_next_shot_sound_present_before_cut"}
    return {**detail, "ambient_available": True,
            "ambient_recommended_relation": "l-cut",
            "ambient_boundary_offset": round(anchor + float(delta) - float(cut_time), 4),
            "ambient_reason": "ambient_prev_shot_sound_continues_after_cut"}


def _build_d3_signal_gate(env_y: np.ndarray, env_sr: int,
                          vocals_y: np.ndarray, vocals_sr: int,
                          segments: List[dict], words: List[dict], cut_time: float,
                          prev_shot_range: Optional[Tuple[float, float]] = None,
                          next_shot_range: Optional[Tuple[float, float]] = None,
                          object_sound_runs: List[dict] = None,
                          object_sound_windows: List[dict] = None,
                          ambient_curves=None, ambient_anchor_time=None) -> dict:
    """First-stage signal diagnosis: separate synchronous change / continuous background / candidate J-L."""
    cont = _analyze_env_audio_continuity(env_y, env_sr, cut_time, window=0.3, gap=0.3) if env_y is not None else {}
    sim_ba = float(cont.get("sim_before_after", 0.0) or 0.0)
    sim_eb = float(cont.get("sim_earlier_before", 0.0) or 0.0)
    sim_ea = float(cont.get("sim_earlier_after", 0.0) or 0.0)
    energy_ratio = float(cont.get("energy_ratio", 0.0) or 0.0)
    mfcc_sync = clip01(1.0 - sim_ba) if sim_ba > 0 else 0.0
    energy_sync = clip01(1.0 - energy_ratio) if energy_ratio > 0 else 0.0
    sync_change_score = max(mfcc_sync, energy_sync)
    ambient_continuity = bool(sim_ba > 0.85 and energy_ratio > 0.45)
    onset = _detect_onset_near_cut(env_y, env_sr, cut_time) if env_y is not None else {
        "onset_near_cut": False,
        "onset_offset": None,
        "onset_strength": 0.0,
    }
    speech = _analyze_speech_overlap(
        segments, words, vocals_y, vocals_sr, cut_time,
        prev_shot_range=prev_shot_range,
        next_shot_range=next_shot_range,
    )
    object_sound = _analyze_object_sound_projection(
        object_sound_runs or [], object_sound_windows or [], cut_time,
        prev_shot_range=prev_shot_range,
        next_shot_range=next_shot_range,
    )

    # Ambience channel: needed as soon as either shot lacks trusted speech. It is computed unconditionally
    # (four block spectral distances, negligible cost) and the arbitration above decides whether to use it:
    # speech projection wins when it yields an attribution, ambience otherwise. "Both shots have speech" is not a hard mask, because Whisper may still let one hallucination through.
    prev_has_speech = _shot_has_speech(segments, prev_shot_range)
    next_has_speech = _shot_has_speech(segments, next_shot_range)
    a_times, a_level, a_shape = (ambient_curves or (None, None, None))
    ambient = _analyze_ambient_projection(a_times, a_level, a_shape, cut_time,
                                          anchor_time=ambient_anchor_time)
    ambient["ambient_prev_shot_has_speech"] = prev_has_speech
    ambient["ambient_next_shot_has_speech"] = next_has_speech
    ambient["ambient_speech_gated"] = bool(prev_has_speech and next_has_speech)

    onset_offset = onset.get("onset_offset")
    onset_sync_within_0p15 = bool(
        onset.get("onset_near_cut") and
        onset_offset is not None and
        abs(float(onset_offset)) <= D3_SYNC_STRAIGHT_THRESHOLD
    )
    speech_sync_within_0p15 = bool(speech.get("speech_boundary_sync_within_0p15"))
    av_sync_within_0p15 = speech_sync_within_0p15
    owned_event_candidate = bool(
        onset.get("onset_near_cut") or speech.get("speech_overlap") or
        object_sound.get("object_sound_crosses_cut")
    )
    clear_sync_change = bool(av_sync_within_0p15 or (sync_change_score >= 0.45 and not speech.get("speech_overlap")))
    ambient_continuity_only = bool(ambient_continuity and not owned_event_candidate and sync_change_score < 0.35)

    if clear_sync_change:
        overlap_type = "none_or_sync"
        overlap_detected = False
        overlap_confidence = sync_change_score
    elif owned_event_candidate:
        overlap_type = "candidate_jl"
        overlap_detected = True
        overlap_confidence = max(float(speech.get("speech_overlap_confidence", 0.0)),
                                 float(onset.get("onset_strength", 0.0)) / 3.0,
                                 0.75 if object_sound.get("object_sound_crosses_cut") else 0.0)
    elif ambient_continuity_only:
        overlap_type = "ambient_continuity_only"
        overlap_detected = False
        overlap_confidence = sim_ba
    else:
        overlap_type = "unclear"
        overlap_detected = False
        overlap_confidence = max(sync_change_score, sim_ba * 0.3)

    return {
        "overlap_type": overlap_type,
        "overlap_detected": bool(overlap_detected),
        "overlap_confidence": round(clip01(overlap_confidence), 4),
        "clear_sync_change": clear_sync_change,
        "av_sync_within_0p15": av_sync_within_0p15,
        "onset_sync_within_0p15": onset_sync_within_0p15,
        "onset_sync_used_for_straight": False,
        "speech_sync_within_0p15": speech_sync_within_0p15,
        "ambient_continuity": ambient_continuity,
        "ambient_continuity_only": ambient_continuity_only,
        "env_sim_before_after": round(sim_ba, 4),
        "env_sim_earlier_before": round(sim_eb, 4),
        "env_sim_earlier_after": round(sim_ea, 4),
        "env_energy_ratio": round(energy_ratio, 4),
        "mfcc_sync_score": round(mfcc_sync, 4),
        "energy_sync_score": round(energy_sync, 4),
        "sync_change_score": round(sync_change_score, 4),
        **speech,
        **object_sound,
        **ambient,
        **onset,
    }


def _build_d3_signal_text_for_vlm(signal_gate: dict) -> str:
    """Build the structured signal JSON shown to VLM.

    Experimental policy: only unclear overlap cues are normalized to straight
    before VLM so dissolve-like audio overlap is not over-penalized as J/L.
    The original signal_gate remains unchanged in evaluation details.
    """
    signal_for_vlm = {
        "overlap_type": signal_gate.get("overlap_type"),
        "sync_change_score": signal_gate.get("sync_change_score"),
        "av_sync_within_0p15": signal_gate.get("av_sync_within_0p15"),
        "speech_boundary_sync_within_0p15": signal_gate.get("speech_boundary_sync_within_0p15"),
        "speech_word_gap_split": signal_gate.get("speech_word_gap_split"),
        "speech_word_gap": signal_gate.get("speech_word_gap"),
        "speech_sentence_end_offset": signal_gate.get("speech_sentence_end_offset"),
        "speech_projection_crosses_cut": signal_gate.get("speech_projection_crosses_cut"),
        "speech_projection_mode": signal_gate.get("speech_projection_mode"),
        "speech_projection_recommended_owner": signal_gate.get("speech_projection_recommended_owner"),
        "speech_projection": signal_gate.get("speech_projection"),
        "object_sound_crosses_cut": signal_gate.get("object_sound_crosses_cut"),
        "object_sound_projection_mode": signal_gate.get("object_sound_projection_mode"),
        "object_sound_recommended_owner": signal_gate.get("object_sound_recommended_owner"),
        "object_sound_recommended_relation": signal_gate.get("object_sound_recommended_relation"),
        "object_sound_projection": signal_gate.get("object_sound_projection"),
        "object_sound_windows_near_cut": signal_gate.get("object_sound_windows_near_cut"),
        "ambient_available": signal_gate.get("ambient_available"),
        "ambient_recommended_relation": signal_gate.get("ambient_recommended_relation"),
        "ambient_boundary_offset": signal_gate.get("ambient_boundary_offset"),
        "ambient_reason": signal_gate.get("ambient_reason"),
        "ambient_cut_dominance": signal_gate.get("ambient_cut_dominance"),
        "ambient_full_replace_ratio": signal_gate.get("ambient_full_replace_ratio"),
        "ambient_prev_shot_has_speech": signal_gate.get("ambient_prev_shot_has_speech"),
        "ambient_next_shot_has_speech": signal_gate.get("ambient_next_shot_has_speech"),
        "ambient_speech_gated": signal_gate.get("ambient_speech_gated"),
        "speech_overlap": signal_gate.get("speech_overlap"),
        "ambient_continuity_only": signal_gate.get("ambient_continuity_only"),
        "onset_near_cut": signal_gate.get("onset_near_cut"),
        "onset_offset": signal_gate.get("onset_offset"),
    }

    overlap_policy_applied = False
    overlap_policy_reasons = []
    original_overlap_type = signal_gate.get("overlap_type")

    if original_overlap_type in {"overlap-unclear", "unclear"}:
        overlap_policy_applied = True
        overlap_policy_reasons.append(f"overlap_type={original_overlap_type}")
        signal_for_vlm["overlap_type"] = "straight"

    if signal_gate.get("speech_projection_mode") == "cross_both_full":
        overlap_policy_applied = True
        overlap_policy_reasons.append("speech_projection_mode=cross_both_full")
        signal_for_vlm["speech_projection_mode"] = "treated_as_straight_overlap"
        signal_for_vlm["speech_projection_recommended_owner"] = "none"

    object_relation = _normalize_d3_relation(signal_gate.get("object_sound_recommended_relation", ""))
    if object_relation == "overlap-unclear":
        overlap_policy_applied = True
        overlap_policy_reasons.append("object_sound_recommended_relation=overlap-unclear")
        signal_for_vlm["object_sound_recommended_relation"] = "straight"
        signal_for_vlm["object_sound_recommended_owner"] = "none"

    signal_for_vlm["overlap_policy_applied"] = overlap_policy_applied
    signal_for_vlm["overlap_policy_relation"] = "straight" if overlap_policy_applied else None
    signal_for_vlm["overlap_policy_reason"] = "; ".join(overlap_policy_reasons)

    return json.dumps(signal_for_vlm, ensure_ascii=False)


def _vlm_predict_d3_relation(video_path: str, cut_time: float,
                             prev_shot: dict, next_shot: dict,
                             signal_gate: dict, transition_idx: int) -> dict:
    """Let the VLM predict the actual audio-visual relation directly, without giving it the GT label."""
    with tempfile.TemporaryDirectory(prefix="d3_vlm_") as tmp_dir:
        clip_path = _extract_d3_transition_clip(video_path, cut_time, tmp_dir, transition_idx)
        prev_desc = prev_shot.get("description_prompt", "")
        next_desc = next_shot.get("description_prompt", "")
        signal_text = _build_d3_signal_text_for_vlm(signal_gate)
        user_text = f"""
The clip is centered on one visual cut. The visual cut occurs at approximately 1.20 seconds after this clip starts.

Previous/outgoing shot description:
{prev_desc}

Next/incoming shot description:
{next_desc}

Signal pre-check, for reference only:
{signal_text}

Classify the ACTUAL perceived audio-visual relation at the cut into exactly one of:
- straight: image and dominant sound transition together at the cut, with no shot-owned sound from the outgoing shot continuing into the incoming shot and no shot-owned sound from the incoming shot heard before its image.
- j-cut: a sound clearly belonging to the incoming/next shot is heard before its image appears, including continuous speech/music/ambience whose credible source is in the next shot.
- l-cut: a sound clearly belonging to the outgoing/previous shot continues after its image disappears, including continuous speech/music/ambience whose credible source is in the previous shot.
- overlap-unclear: sound overlaps the cut, but ownership/timing is ambiguous.
- unclear: evidence is insufficient.

Important rules:
- For this experiment, only unclear overlap cues in the signal pre-check are normalized before you see them: if overlap_policy_applied is true and overlap_policy_relation is straight, treat the unclear overlap as straight because dissolve/crossfade-style visual effects may also dissolve the audio. Do not apply this policy to explicit J-cut/L-cut evidence or candidate_jl cues.
- If a speech segment split by word gaps ends within 0.15 seconds of the visual cut, humans may perceive it as synchronized; consider straight unless stronger shot-owned continuation evidence exists.
- If speech_projection_crosses_cut is true, use speech_projection as direct evidence about the same speech spanning both adjacent shots.
- If speech_projection_mode is cross_both_full, both complete shots are covered by the same speech; decide ownership from the visible speaking subject, shot scale/distance, lip/face presence, and perceived loudness/source credibility.
- If speech_projection_mode is cross_partial and speech_projection_recommended_owner is previous, the default interpretation is L-cut: the previous shot's sound continues into the next shot. Override only if you are highly confident the previous shot has no plausible source.
- If speech_projection_mode is cross_partial and speech_projection_recommended_owner is next, the default interpretation is J-cut: the next shot's sound is heard before/through the cut. Override only if you are highly confident the next shot has no plausible source.
- A continuous background, music bed, room tone, wind, water, or speech bed should NOT be dismissed as straight merely because it is continuous.
- However, classify continuous sound as J-cut/L-cut ONLY when it is a salient subject sound or a clearly attributable source sound, such as speech, singing, instrument performance, action sound, or a visible/credible object-machine sound that dominates the cut.
- Pure mechanical bed noise, low-level scene tone, wind, water, room tone, or generic ambience without a salient subject/source ownership can still be straight, even if it continues across the cut.
- If a clearly attributable previous-shot sound continues over the next image, classify L-cut; if a clearly attributable next-shot sound is heard before/through the cut, classify J-cut.
- If neither shot has speech (ambient_prev_shot_has_speech / ambient_next_shot_has_speech are false) and ambient_available is true, the ambient-track boundary is measured evidence: ambient_boundary_offset < 0 means the incoming shot's ambience is already audible before the cut (J-cut), > 0 means the outgoing shot's ambience continues past the cut (L-cut), and 0 means the ambience is fully replaced exactly at the cut (straight). Override it only with clearly audible contrary evidence.
- For non-speech object sounds, use object_sound_projection as primary structured evidence. PANN tags were computed on Demucs non-vocal tracks (drums/bass/other) every 0.5 seconds, with quiet/generic noise filtered out.
- If object_sound_recommended_relation is l-cut or j-cut, follow it when the object_sound_projection describes a salient guitar/pluck/impact/action/source sound crossing the cut. Natural decay of a guitar/pluck/chord that remains audible after its visual source disappears still counts as L-cut; do not require the continuation to be artificial.
- If a shot-owned sound source begins in the previous shot and only its natural tail/decay continues into the next shot, classify it as L-cut, even when there is no new onset after the cut.
- If continuous music, sound, or speech has an obvious pause, rest, beat hit, drum hit, or chord attack exactly at/near the visual cut, and the next shot still shows the same sound source or performance context, classify it as straight because the cut is synchronized to the musical/sound punctuation.
- If object_sound_recommended_relation is straight because onset is synchronized with the visual cut, or because a continuous music bed spans both adjacent shots without clear ownership, prefer straight unless there is a stronger off-sync source cue.
- Low-level mechanical bed noise, weak wind/water/room tone, or generic ambience should not trigger J/L by itself; it can remain straight.
- Environment or image changes at the cut do not by themselves make the relation straight when the same salient shot-owned sound continues across the cut.
- Do NOT infer from the written intended relation; judge the actual clip.
- Choose straight only when the dominant sound is synchronized with the image change OR the continuing sound has no credible ownership to either adjacent shot.

Return ONLY JSON:
{{
  "predicted_relation": "straight|j-cut|l-cut|overlap-unclear|unclear",
  "overlap_detected": true,
  "sound_owner": "previous|next|both|none|unclear",
  "dominant_sound_event": "brief event description",
  "timing_observation": "brief timing observation",
  "confidence": 0.0,
  "reason": "brief reason"
}}
""".strip()
        response = _call_vlm(_SYSTEM_D3_AUDIO_VISUAL_RELATION, user_text, clip_path,
                             max_tokens=900, temperature=0.0)

    parsed = _extract_json_from_vlm_response(response)
    if not parsed:
        return {
            "used": True,
            "predicted_relation": "unclear",
            "overlap_detected": signal_gate.get("overlap_detected", False),
            "sound_owner": "unclear",
            "dominant_sound_event": "",
            "timing_observation": "",
            "confidence": 0.0,
            "reason": "parse_failed",
            "raw_response": response[:500] if response else "",
        }

    relation = _normalize_d3_relation(parsed.get("predicted_relation", ""))
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence <= 0.0 and relation != "unclear" and parsed.get("reason"):
        confidence = 0.6
    owner = (parsed.get("sound_owner") or "unclear").strip().lower()
    if owner not in {"previous", "next", "both", "none", "unclear"}:
        owner = "unclear"

    return {
        "used": True,
        "predicted_relation": relation,
        "overlap_detected": bool(parsed.get("overlap_detected", relation in {"j-cut", "l-cut", "overlap-unclear"})),
        "sound_owner": owner,
        "dominant_sound_event": str(parsed.get("dominant_sound_event", ""))[:300],
        "timing_observation": str(parsed.get("timing_observation", ""))[:300],
        "confidence": round(clip01(confidence), 4),
        "reason": str(parsed.get("reason", ""))[:300],
    }


def _fallback_d3_relation_from_signal(signal_gate: dict) -> str:
    if signal_gate.get("clear_sync_change"):
        return "straight"
    if signal_gate.get("overlap_type") == "ambient_continuity_only":
        return "straight"
    if signal_gate.get("overlap_type") == "candidate_jl":
        return "overlap-unclear"
    return "unclear"


def _pre_vlm_d3_relation_from_signal(signal_gate: dict) -> dict:
    if signal_gate.get("av_sync_within_0p15"):
        return {
            "relation": "straight",
            "reason": "speech_segment_ends_within_0p15_of_cut",
        }

    if signal_gate.get("speech_projection_crosses_cut"):
        mode = signal_gate.get("speech_projection_mode")
        owner = signal_gate.get("speech_projection_recommended_owner")
        projection = signal_gate.get("speech_projection") or {}
        if mode == "cross_partial" and owner == "previous":
            return {
                "relation": "l-cut",
                "reason": "pre_vlm_speech_projection_previous_longer",
                "prev_overlap_duration": projection.get("prev_overlap_duration"),
                "next_overlap_duration": projection.get("next_overlap_duration"),
            }
        if mode == "cross_partial" and owner == "next":
            return {
                "relation": "j-cut",
                "reason": "pre_vlm_speech_projection_next_longer",
                "prev_overlap_duration": projection.get("prev_overlap_duration"),
                "next_overlap_duration": projection.get("next_overlap_duration"),
            }
        if mode == "cross_both_full":
            return {
                "relation": "overlap-unclear",
                "reason": "pre_vlm_speech_projection_both_full_needs_visual_owner",
            }

    if signal_gate.get("object_sound_crosses_cut"):
        object_relation = _normalize_d3_relation(signal_gate.get("object_sound_recommended_relation", ""))
        if object_relation != "unclear":
            return {
                "relation": object_relation,
                "reason": "pre_vlm_object_sound_projection",
                "object_sound_recommended_owner": signal_gate.get("object_sound_recommended_owner"),
                "object_sound_projection": signal_gate.get("object_sound_projection"),
            }

    return {
        "relation": _fallback_d3_relation_from_signal(signal_gate),
        "reason": "signal_fallback",
    }


def eval_d3_audio_visual_relation(transnetv2_result: dict, whisper_result: dict,
                                  prompt: dict, video_path: str,
                                  alignment_result: dict = None,
                                  demucs_result: dict = None) -> dict:
    """
    D3: transition audio-visual relation.

    Logic: predict the actual relation from the video itself, and keep GT only as a comparison field.
    Predicted classes: straight / j-cut / l-cut / overlap-unclear / unclear.
    """
    import librosa

    shots = prompt.get("shots", [])

    if not shots or len(shots) < 2:
        return {"dimension": "D3", "metric": "transition audio-visual relation", "score": 0.5}

    segments = whisper_result.get("segments", [])
    words = whisper_result.get("words", [])
    # Drop Whisper hallucination segments first: fake speech masks the ambience channel and drives a wrong J/L attribution.
    segments, words, speech_filter_stats = _filter_reliable_speech(segments, words)
    if speech_filter_stats["n_segments_in"] != speech_filter_stats["n_segments_kept"]:
        print(f"  [D3] filtered unreliable speech segments "
              f"{speech_filter_stats['n_segments_in']} -> {speech_filter_stats['n_segments_kept']}"
              f" (hallucinated segments: "
              f"{[d['text'] for d in speech_filter_stats['dropped_segments']]})")

    # ===== Load the Demucs-separated stems =====
    env_y = None
    env_sr = 22050
    vocals_y = None
    vocals_sr = 22050
    source_files = {}
    if demucs_result:
        source_files = demucs_result.get("source_files", {})
        env_sr = demucs_result.get("sample_rate", 44100)
        vocals_sr = demucs_result.get("sample_rate", 44100)

    env_tracks = []
    object_track_audio = {}
    object_track_sr = {}
    object_panns_by_track = {}
    for track_name in ["drums", "bass", "other"]:
        track_path = source_files.get(track_name, "")
        if track_path and os.path.exists(track_path):
            try:
                y_track, env_sr = librosa.load(track_path, sr=env_sr, mono=True)
                env_tracks.append(y_track)
                object_track_audio[track_name] = y_track
                object_track_sr[track_name] = env_sr
                try:
                    object_panns_by_track[track_name] = call_panns(track_path, segment_duration=D3_OBJECT_SEGMENT_DURATION)
                except Exception as e:
                    object_panns_by_track[track_name] = {"success": False, "error": str(e)[:200]}
            except Exception:
                pass

    if env_tracks:
        min_len = min(len(t) for t in env_tracks)
        env_y = np.sum([t[:min_len] for t in env_tracks], axis=0)

    vocals_path = source_files.get("vocals", "")
    if vocals_path and os.path.exists(vocals_path):
        try:
            vocals_y, vocals_sr = librosa.load(vocals_path, sr=vocals_sr, mono=True)
        except Exception:
            vocals_y = None

    object_sound_windows = _build_object_sound_windows(object_panns_by_track, object_track_audio, object_track_sr)
    object_sound_runs = _merge_object_sound_runs(object_sound_windows)

    # The ambience curve uses the final cut's full mix directly: demucs stem separation redistributes each
    # clip's noise floor across stems and measurably weakens the only usable structural feature.
    ambient_curves = (None, None, None)
    if video_path:
        try:
            mix_y, mix_sr = _load_audio_mono(video_path, sr=_AMBIENT_SR)
            ambient_curves = _ambient_features(mix_y, mix_sr)
        except Exception as e:
            print(f"  [D3] failed to extract the ambience curve, this channel is disabled: {str(e)[:200]}")

    aligned_transitions = None
    if alignment_result is not None:
        aligned_transitions = get_aligned_transitions(alignment_result)

    scores = []
    details = []
    for i, shot in enumerate(shots[:-1]):
        gt_relation = shot.get("transition_to_next", {}).get("audio_visual_relation", "")
        gt_relation_norm = _normalize_d3_relation(gt_relation)

        if aligned_transitions is not None and i < len(aligned_transitions):
            if not aligned_transitions[i]["evaluable"]:
                scores.append(0.0)
                details.append({
                    "transition": i,
                    "gt_relation": gt_relation,
                    "gt_relation_normalized": gt_relation_norm,
                    "predicted_relation": "unclear",
                    "matches_gt": False,
                    "score": 0.0,
                    "reason": "missing_shot",
                })
                continue
            at = aligned_transitions[i]
            shot_end = at["cut_time"] if at["cut_time"] is not None else parse_time_range(shot["description_prompt"])[1]
        else:
            shot_end = parse_time_range(shot["description_prompt"])[1]

        if alignment_result is not None:
            prev_shot_range = get_aligned_shot_clip_range(alignment_result, i)
            next_shot_range = get_aligned_shot_clip_range(alignment_result, i + 1)
        else:
            prev_shot_range = parse_time_range(shot["description_prompt"])
            next_shot_range = parse_time_range(shots[i + 1]["description_prompt"])

        signal_gate = _build_d3_signal_gate(
            env_y=env_y,
            env_sr=env_sr,
            vocals_y=vocals_y,
            vocals_sr=vocals_sr,
            segments=segments,
            words=words,
            cut_time=shot_end,
            prev_shot_range=prev_shot_range,
            next_shot_range=next_shot_range,
            object_sound_runs=object_sound_runs,
            object_sound_windows=object_sound_windows,
            ambient_curves=ambient_curves,
        )

        vlm_relation = None
        if video_path and i + 1 < len(shots):
            try:
                vlm_relation = _vlm_predict_d3_relation(
                    video_path=video_path,
                    cut_time=shot_end,
                    prev_shot=shot,
                    next_shot=shots[i + 1],
                    signal_gate=signal_gate,
                    transition_idx=i,
                )
                print(f"  [D3] actual relation transition {i}: "
                      f"{vlm_relation.get('predicted_relation')} "
                      f"conf={vlm_relation.get('confidence')} "
                      f"gate={signal_gate.get('overlap_type')}")
            except Exception as e:
                vlm_relation = {
                    "used": True,
                    "predicted_relation": "unclear",
                    "overlap_detected": signal_gate.get("overlap_detected", False),
                    "sound_owner": "unclear",
                    "dominant_sound_event": "",
                    "timing_observation": "",
                    "confidence": 0.0,
                    "reason": f"vlm_failed: {str(e)[:200]}",
                }

        pre_vlm_result = _pre_vlm_d3_relation_from_signal(signal_gate)
        fallback_relation = _fallback_d3_relation_from_signal(signal_gate)

        # Generic signal arbitration (never reads GT), from strongest to weakest source attribution:
        #   1) word-level speech projection (cross_partial) -> the attributed side decides J/L
        #   2) ambience audio cut (usable only when at least one shot has no speech) -> its position decides J/L/straight
        #   3) a prominent object sound continuing long -> follow the object sound projection
        #   4) synchronous change + a very short / absent natural tail -> straight
        # The ambience channel must come before "sync change -> straight": the latter is almost always true
        # on speechless material, and hitting it first would call every transition straight.
        arbitration_relation = None
        arbitration_reason = None
        object_projection = signal_gate.get("object_sound_projection") or {}
        object_next_overlap = float(object_projection.get("next_overlap_duration", 0.0) or 0.0)
        object_relation = _normalize_d3_relation(
            signal_gate.get("object_sound_recommended_relation", "")
        )
        ambient_relation = _normalize_d3_relation(
            signal_gate.get("ambient_recommended_relation", "")
        )
        if (signal_gate.get("speech_projection_crosses_cut") and
                signal_gate.get("speech_projection_mode") == "cross_partial"):
            speech_owner = signal_gate.get("speech_projection_recommended_owner")
            if speech_owner == "next":
                arbitration_relation = "j-cut"
                arbitration_reason = "word_timestamp_speech_projection_next"
            elif speech_owner == "previous":
                arbitration_relation = "l-cut"
                arbitration_reason = "word_timestamp_speech_projection_previous"
        if arbitration_relation is None and signal_gate.get("ambient_available") and \
                ambient_relation in {"j-cut", "l-cut", "straight"}:
            arbitration_relation = ambient_relation
            arbitration_reason = signal_gate.get("ambient_reason") or "ambient_boundary_projection"
        if arbitration_relation is None:
            if (signal_gate.get("object_sound_crosses_cut") and
                    object_relation in {"j-cut", "l-cut"} and
                    object_next_overlap > D3_OBJECT_SYNC_TAIL_THRESHOLD):
                arbitration_relation = object_relation
                arbitration_reason = "salient_object_sound_long_cross_cut"
            elif (signal_gate.get("clear_sync_change") and
                  (not signal_gate.get("object_sound_crosses_cut") or
                   object_next_overlap <= D3_OBJECT_SYNC_TAIL_THRESHOLD)):
                arbitration_relation = "straight"
                arbitration_reason = "sync_change_with_short_or_no_object_tail"

        predicted_relation = fallback_relation
        if arbitration_relation is not None:
            predicted_relation = arbitration_relation
        elif vlm_relation:
            vlm_pred = _normalize_d3_relation(vlm_relation.get("predicted_relation", ""))
            if vlm_pred != "unclear":
                predicted_relation = vlm_pred
            elif fallback_relation != "unclear":
                predicted_relation = fallback_relation
            else:
                predicted_relation = vlm_pred

        matches_gt = bool(gt_relation_norm != "unclear" and predicted_relation == gt_relation_norm)
        final_score = 1.0 if matches_gt else 0.0
        scores.append(final_score)

        detail = {
            "transition": i,
            "cut_time": round(float(shot_end), 3),
            "gt_relation": gt_relation,
            "gt_relation_normalized": gt_relation_norm,
            "predicted_relation": predicted_relation,
            "pre_vlm_relation": pre_vlm_result.get("relation"),
            "pre_vlm_reason": pre_vlm_result.get("reason"),
            "pre_vlm_result": pre_vlm_result,
            "post_vlm_relation": predicted_relation,
            "relation_arbitration": {
                "applied": arbitration_relation is not None,
                "relation": arbitration_relation,
                "reason": arbitration_reason,
                "object_next_overlap_duration": round(object_next_overlap, 4),
                "object_sync_tail_threshold": D3_OBJECT_SYNC_TAIL_THRESHOLD,
                "ambient_relation": ambient_relation,
                "ambient_available": bool(signal_gate.get("ambient_available")),
            },
            "matches_gt": matches_gt,
            "score": round(final_score, 4),
            "overlap_detected": bool(signal_gate.get("overlap_detected", False)),
            "overlap_confidence": signal_gate.get("overlap_confidence"),
            "sound_owner": (vlm_relation or {}).get("sound_owner", "unclear"),
            "object_sound_summary": {
                "num_windows": len(object_sound_windows),
                "num_runs": len(object_sound_runs),
                "panns_tracks": {name: {
                    "success": result.get("success", False),
                    "num_segments": result.get("num_segments", 0),
                    "audio_tags": result.get("audio_tags", [])[:5],
                } for name, result in object_panns_by_track.items()},
            },
            "signal_gate": signal_gate,
        }
        if vlm_relation:
            detail["vlm_relation"] = vlm_relation
        details.append(detail)

    if not scores:
        return {"dimension": "D3", "metric": "transition audio-visual relation", "score": 0.5}

    return {
        "dimension": "D3",
        "metric": "transition audio-visual relation",
        "score": clip01(float(np.mean(scores))),
        "prediction_mode": "actual_video_first",
        "speech_filter": speech_filter_stats,
        "per_transition": details,
    }


# ---- E2: image quality ----
# A dedicated model scores every frame and the results are averaged (replacing DOVER):
#   - Aesthetic Quality: CLIP ViT-L/14 + the LAION aesthetic linear head (raw/10 -> [0,1])
# The service also returns a MUSIQ imaging score, but the adjusted formula does not use it.


def eval_e2_video_quality(e2_result: dict) -> dict:
    """
    E2: image quality
    Adjusted formula: only the aesthetic score is used
         aesthetic = CLIP ViT-L/14 + LAION aesthetic head (per-frame mean)
    """
    aesthetic = e2_result.get("aesthetic_quality", 0.0)
    return {
        "dimension": "E2",
        "metric": "visual quality",
        "score": clip01(aesthetic),
        "aesthetic": aesthetic,
        "method": "aesthetic_quality_only(CLIP+LAION)",
        "details": e2_result.get("details", {}),
    }


# ---- E3: style consistency ----
def _eval_e3_legacy_temporal(dinov2_result: dict, reason: str = "fallback") -> dict:
    """Legacy E3: mean DINOv2 adjacent-frame temporal similarity."""
    temporal_sim = dinov2_result.get("temporal_consistency", [])
    mean_sim = dinov2_result.get("mean_similarity", 0.0)

    if temporal_sim:
        score = float(np.mean(temporal_sim))
    else:
        score = mean_sim

    return {
        "dimension": "E3",
        "metric": "style consistency",
        "score": clip01(score),
        "mean_temporal_sim": round(score, 4),
        "num_frame_pairs": len(temporal_sim),
        "method": "legacy_temporal",
        "used_alignment": False,
        "fallback_reason": reason,
        "per_label_scores": {},
        "num_event_labels": 0,
    }


def _extract_e3_labels(prompt: dict) -> List[int]:
    labels = []
    for shot in prompt.get("shots", []):
        label = shot.get("event_coherence_label", 0)
        try:
            label = int(label)
        except (TypeError, ValueError):
            label = 0
        labels.append(max(0, label))
    return labels


def _score_dinov2_temporal(dinov2_result: dict) -> Tuple[float, int]:
    temporal_sim = dinov2_result.get("temporal_consistency", [])
    if temporal_sim:
        return float(np.mean(temporal_sim)), len(temporal_sim)
    return float(dinov2_result.get("mean_similarity", 0.0)), 0


def _extract_e3_segment(video_path: str, start_sec: float, end_sec: float,
                        tmp_dir: str, segment_idx: int) -> str:
    duration = max(0.0, float(end_sec) - float(start_sec))
    if duration <= 0.05:
        raise ValueError(f"invalid segment duration: {start_sec}-{end_sec}")

    out_path = os.path.join(tmp_dir, f"segment_{segment_idx:03d}.mp4")
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", video_path,
        "-t", f"{duration:.3f}",
        "-an", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=90)
    if result.returncode != 0 or not os.path.exists(out_path):
        stderr = result.stderr.decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"ffmpeg segment extraction failed: {stderr}")
    return out_path


def _concat_e3_segments(segment_paths: List[str], tmp_dir: str, label: int) -> str:
    if len(segment_paths) == 1:
        return segment_paths[0]

    list_path = os.path.join(tmp_dir, f"concat_label_{label}.txt")
    out_path = os.path.join(tmp_dir, f"label_{label}_reconstructed.mp4")
    with open(list_path, "w", encoding="utf-8") as f:
        for path in segment_paths:
            safe_path = path.replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-an", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=180)
    if result.returncode != 0 or not os.path.exists(out_path):
        stderr = result.stderr.decode("utf-8", errors="ignore")[:300]
        raise RuntimeError(f"ffmpeg concat failed: {stderr}")
    return out_path


def eval_e3_style_consistency(dinov2_result: dict,
                              prompt: dict = None,
                              alignment_result: dict = None,
                              video_path: str = None,
                              sample_fps: float = _E3_SAMPLE_FPS,
                              max_frames: int = _E3_MAX_FRAMES) -> dict:
    """
    E3: inter-frame style stability.

    Current logic:
    - group shots by prompt.shots[].event_coherence_label;
    - use the actual time_range of the VLM aligned_shots to cut the shots of one label out of the source video;
    - hard-cut the same-label shots together, in their original order, into one temporary recomposed video;
    - call DINOv2 on each recomposed video separately and compute its temporal consistency;
    - finally take a weighted mean by that label's share of the prompt shot sequence.
    """
    if not prompt:
        return _eval_e3_legacy_temporal(dinov2_result, "missing prompt")
    if not alignment_result:
        return _eval_e3_legacy_temporal(dinov2_result, "missing alignment_result")
    if not video_path:
        return _eval_e3_legacy_temporal(dinov2_result, "missing video_path")

    shots = prompt.get("shots", [])
    labels = _extract_e3_labels(prompt)
    if not shots or len(labels) != len(shots):
        return _eval_e3_legacy_temporal(dinov2_result, "missing prompt shots")

    aligned_by_gt = {}
    for ashot in alignment_result.get("aligned_shots", []):
        gt_idx = ashot.get("gt_shot_idx")
        if isinstance(gt_idx, int):
            aligned_by_gt[gt_idx] = ashot
    if not aligned_by_gt:
        return _eval_e3_legacy_temporal(dinov2_result, "empty aligned_shots")

    label_counts: Dict[int, int] = {}
    label_to_ranges: Dict[int, List[Tuple[int, float, float]]] = {}
    for gt_idx, label in enumerate(labels):
        label_counts[label] = label_counts.get(label, 0) + 1
        ashot = aligned_by_gt.get(gt_idx)
        if not ashot or ashot.get("status") == "missing" or not ashot.get("time_range"):
            continue
        start_sec, end_sec = ashot["time_range"]
        if end_sec <= start_sec:
            continue
        label_to_ranges.setdefault(label, []).append((gt_idx + 1, float(start_sec), float(end_sec)))

    total_shots = max(1, len(labels))
    weighted_score = 0.0
    total_pairs = 0
    per_label_scores = {}

    with tempfile.TemporaryDirectory(prefix="e3_recompose_") as tmp_dir:
        for label in sorted(label_counts.keys()):
            weight = label_counts[label] / total_shots
            ranges = label_to_ranges.get(label, [])
            if not ranges:
                per_label_scores[str(label)] = {
                    "score": 0.0,
                    "weight": round(weight, 4),
                    "shot_count": label_counts[label],
                    "valid_shot_count": 0,
                    "shot_ids": [],
                    "num_frame_pairs": 0,
                    "note": "no valid aligned shots for this label",
                }
                continue

            try:
                segment_paths = []
                for segment_idx, (_, start_sec, end_sec) in enumerate(ranges):
                    segment_paths.append(_extract_e3_segment(
                        video_path, start_sec, end_sec, tmp_dir,
                        segment_idx + label * 1000,
                    ))
                reconstructed_path = _concat_e3_segments(segment_paths, tmp_dir, label)
                label_dinov2 = call_dinov2(reconstructed_path, sample_fps=sample_fps,
                                           max_frames=max_frames)
                label_score_raw, num_pairs = _score_dinov2_temporal(label_dinov2)
                label_score = clip01(label_score_raw)
            except Exception as exc:
                label_score = 0.0
                num_pairs = 0
                per_label_scores[str(label)] = {
                    "score": 0.0,
                    "weight": round(weight, 4),
                    "shot_count": label_counts[label],
                    "valid_shot_count": len(ranges),
                    "shot_ids": [shot_id for shot_id, _, _ in ranges],
                    "num_frame_pairs": 0,
                    "error": str(exc)[:300],
                }
                continue

            weighted_score += label_score * weight
            total_pairs += num_pairs
            per_label_scores[str(label)] = {
                "score": round(label_score, 4),
                "weight": round(weight, 4),
                "weighted_score": round(label_score * weight, 4),
                "shot_count": label_counts[label],
                "valid_shot_count": len(ranges),
                "shot_ids": [shot_id for shot_id, _, _ in ranges],
                "num_frame_pairs": num_pairs,
                "reconstructed": len(ranges) > 1,
            }

    return {
        "dimension": "E3",
        "metric": "style consistency",
        "score": clip01(weighted_score),
        "mean_temporal_sim": round(weighted_score, 4),
        "num_frame_pairs": total_pairs,
        "method": "event_coherence_reconstructed_video",
        "used_alignment": True,
        "per_label_scores": per_label_scores,
        "num_event_labels": len(per_label_scores),
        "weighting": "prompt_label_shot_ratio",
    }


# ---- F1: intra-shot audio-visual sync ----
def eval_f1_av_sync(raft_result: dict, video_path: str) -> dict:
    """
    F1: intra-shot audio-visual sync
    Formula: linear calibration max(0, 1 - mean_offset/OMAX), OMAX=1.0s
    Time lag between RAFT motion energy peaks and librosa onset peaks
    (the original exp(-|offset|/0.2) over-penalised long-tail offsets, giving low, poorly separated scores)
    """
    motion_energy = raft_result.get("motion_energy", [])
    sample_fps = raft_result.get("sample_fps", 8.0)
    
    if not motion_energy:
        return {"dimension": "F1", "metric": "in-shot audio-visual sync", "score": 0.5,
                "note": "no motion energy"}
    
    try:
        import librosa
        import subprocess
        import tempfile
        
        # Extract the audio
        audio_path = tempfile.mktemp(suffix=".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "22050",
            "-ac", "1", audio_path
        ], capture_output=True, timeout=30)
        
        if not os.path.exists(audio_path):
            return {"dimension": "F1", "metric": "in-shot audio-visual sync", "score": 0.5,
                    "note": "audio extraction failed"}
        
        y, sr = librosa.load(audio_path, sr=22050)
        os.unlink(audio_path)
        
        # Audio onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Visual motion peak detection
        me = np.array(motion_energy)
        me_threshold = np.mean(me) + np.std(me)
        motion_peak_indices = np.where(me > me_threshold)[0]
        motion_peak_times = motion_peak_indices / sample_fps
        
        if len(onset_times) == 0 or len(motion_peak_times) == 0:
            return {"dimension": "F1", "metric": "in-shot audio-visual sync", "score": 0.5,
                    "note": "no peaks detected"}
        
        # Time lag of the nearest onset-motion peak pairs
        offsets = []
        for mt in motion_peak_times:
            if len(onset_times) > 0:
                closest_onset = min(onset_times, key=lambda x: abs(x - mt))
                offsets.append(abs(mt - closest_onset))
        
        if not offsets:
            return {"dimension": "F1", "metric": "in-shot audio-visual sync", "score": 0.5}
        
        mean_offset = float(np.mean(offsets))
        # Linear calibration: linear credit within a 1s tolerance, clearly higher and better separated than exp(-off/0.2)
        # (the exponential over-penalises long-tail offsets so scores stay low, while clipping outliers would flatten model differences)
        OMAX = 1.0
        score = max(0.0, 1.0 - mean_offset / OMAX)
        
        return {
            "dimension": "F1",
            "metric": "in-shot audio-visual sync",
            "score": clip01(score),
            "mean_offset": round(mean_offset, 4),
            "num_motion_peaks": len(motion_peak_times),
            "num_audio_onsets": len(onset_times),
        }
    
    except Exception as e:
        return {"dimension": "F1", "metric": "in-shot audio-visual sync", "score": 0.5,
                "note": f"error: {str(e)}"}


# ---- F2: overall perceptual audio quality (signal processing) ----

def eval_f2_audio_quality(audio_quality_result: dict) -> dict:
    """
    F2: overall perceptual audio quality

    Adjusted formula: the score is the DNSMOS signal-processing quality only
    (SNR + dynamic range + spectral richness - distortion penalty), reported by
    the dnsmos service as objective_quality_score.
    """
    signal_score = audio_quality_result.get("objective_quality_score", 0.5)

    return {
        "dimension": "F2",
        "metric": "overall audio perceptual quality",
        "score": clip01(signal_score),
        "signal_score": round(signal_score, 4),
        "snr_db": audio_quality_result.get("snr_db"),
        "spectral_richness": audio_quality_result.get("spectral_richness"),
    }


# ============================================================
#  Main evaluation pipeline
# ============================================================

def evaluate_single_video(video_path: str, prompt: dict, logger: TestLogger,
                          alignment_result: dict = None) -> dict:
    """
    Run every Mode A evaluation on one video.
    
    Args:
        alignment_result: VLM-assisted shot alignment result (optional; used by every dimension when given)
    
    Returns:
        A dict holding every dimension score
    """
    video_name = os.path.basename(video_path)
    logger.log_section(f"video: {video_name} (ID={prompt.get('id', '?')})")
    logger.log(f"- title: {prompt.get('title', 'N/A')}")
    logger.log(f"- GT shot count: {prompt.get('number_of_shots', '?')}")
    logger.log(f"- file: `{video_path}`")
    
    results = {}
    raw_outputs = {}
    
    # ---- Step 1: call every microservice ----
    logger.log_subsection("Step 1: expert model calls")
    
    n_gt_shots = len(prompt.get("shots", []))
    target_style = extract_target_style(prompt)
    services_to_call = [
        ("transnetv2", lambda: call_transnetv2(video_path, expected_shots=n_gt_shots)),
        ("raft", lambda: call_raft(video_path, sample_fps=8.0)),
        ("dinov2", lambda: call_dinov2(video_path, sample_fps=4.0, max_frames=80)),
        ("whisper", lambda: call_whisper(video_path, word_timestamps=True)),
        ("demucs", lambda: call_demucs(video_path)),
        ("e2quality", lambda: call_e2quality(video_path)),
        ("clip_style", lambda: call_clip_style_match(video_path, target_style, STYLE_CATEGORIES)),
        ("panns", lambda: call_panns(video_path, segment_duration=1.0)),
        ("dnsmos", lambda: call_dnsmos(video_path)),
    ]
    
    for service_name, call_fn in services_to_call:
        t0 = time.time()
        try:
            raw_outputs[service_name] = call_fn()
            elapsed = time.time() - t0
            logger.log_info(f"✅ {service_name}: {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            logger.log_info(f"❌ {service_name}: FAILED ({elapsed:.1f}s) - {str(e)[:100]}")
            raw_outputs[service_name] = {}
    
    # ---- Step 2: after Demucs separation, send the vocals stem back to Whisper ----
    demucs_raw = raw_outputs.get("demucs", {})
    source_files = demucs_raw.get("source_files", {})
    vocals_path = source_files.get("vocals", "")
    if vocals_path and os.path.exists(vocals_path):
        try:
            t0 = time.time()
            raw_outputs["whisper"] = call_whisper(vocals_path, word_timestamps=True)
            elapsed = time.time() - t0
            logger.log_info(f"✅ whisper (vocals track from Demucs): {elapsed:.1f}s")
        except Exception as e:
            logger.log_info(f"⚠ whisper on vocals failed, using original video whisper: {str(e)[:100]}")
    else:
        logger.log_info("⚠ Demucs vocals track not found, using original video whisper")

    # ---- Step 3: VLM shot alignment (when not provided) ----
    if alignment_result is None:
        try:
            alignment_result = align_shots_with_vlm(
                video_path, raw_outputs.get("transnetv2", {}), prompt)
            logger.log_info(f"VLM alignment: matched={alignment_result.get('n_matched', 0)}, "
                           f"merged={alignment_result.get('n_merged', 0)}, "
                           f"missing={alignment_result.get('n_missing', 0)}, "
                           f"method={alignment_result.get('method', '?')}")
        except Exception as e:
            logger.log_info(f"VLM alignment failed, falling back to TransNetV2: {e}")
            alignment_result = None

    # ---- Step 4: compute each dimension score ----
    logger.log_subsection("Step 4: dimension score computation")
    
    # A1
    try:
        r = eval_a1_shot_count(raw_outputs.get("transnetv2", {}), prompt,
                               alignment_result=alignment_result)
        results["A1"] = r
        logger.log_metric("A1 shot count accuracy", r["score"],
                         f"pred={r.get('n_pred')}, gt={r.get('n_gt')}, method={r.get('method')}")
    except Exception as e:
        logger.log_error("A1", str(e))
        results["A1"] = {"dimension": "A1", "score": 0.0, "error": str(e)}
    
    # A4
    try:
        r = eval_a4_style(raw_outputs.get("clip_style", {}), prompt)
        results["A4"] = r
        logger.log_metric("A4 style matching", r["score"],
                         f"target={r.get('target_style')}, pred={r.get('predicted_style')}")
    except Exception as e:
        logger.log_error("A4", str(e))
        results["A4"] = {"dimension": "A4", "score": 0.0, "error": str(e)}
    
    # B1
    try:
        r = eval_b1_shot_duration(raw_outputs.get("transnetv2", {}), prompt,
                                   alignment_result=alignment_result)
        results["B1"] = r
        logger.log_metric("B1 shot duration accuracy", r["score"],
                         f"mean_err={r.get('mean_relative_error', 'N/A')}")
    except Exception as e:
        logger.log_error("B1", str(e))
        results["B1"] = {"dimension": "B1", "score": 0.0, "error": str(e)}
    
    # B2
    try:
        r = eval_b2_beat_sync(raw_outputs.get("transnetv2", {}),
                              raw_outputs.get("demucs", {}), video_path,
                              raft_result=raw_outputs.get("raft", {}),
                              prompt=prompt,
                              alignment_result=alignment_result)
        results["B2"] = r
        logger.log_metric("B2 beat synchronization", r["score"],
                         f"sub2={r.get('sub2_energy_corr')}, sub3={r.get('sub3_sfx_match')}")
    except Exception as e:
        logger.log_error("B2", str(e))
        results["B2"] = {"dimension": "B2", "score": 0.0, "error": str(e)}
    
    # D1: route to the matching Mode A method by the GT cinematographic_type
    try:
        d1_result = eval_d1_routed(
            prompt=prompt,
            dinov2_result=raw_outputs.get("dinov2", {}),
            raft_result=raw_outputs.get("raft", {}),
            panns_result=raw_outputs.get("panns", {}),
            transnetv2_result=raw_outputs.get("transnetv2", {}),
            alignment_result=alignment_result,
        )
        results["D1"] = d1_result
        # Log every transition
        for entry in d1_result.get("per_transition", []):
            sid = entry.get("shot_id", "?")
            ctype = entry.get("cinematographic_type", "")
            method = entry.get("method")
            sc = entry.get("score")
            if method is None:
                logger.log_info(f"  D1 Shot{sid}→{ctype} → Mode B (skipped)")
            elif sc is not None:
                logger.log_metric(f"  D1 Shot{sid}→{ctype}[{method}]", sc)
            else:
                logger.log_info(f"  D1 Shot{sid}→{ctype}[{method}] → not enough data")
        d1_score = d1_result.get("score")
        if d1_score is not None:
            logger.log_metric("**D1 overall** (routed by GT type)", d1_score)
        else:
            logger.log_info("D1: every transition is a Mode B type, nothing evaluated in the Mode A stage")
    except Exception as e:
        logger.log_error("D1", str(e))
        results["D1"] = {"dimension": "D1", "score": None, "error": str(e)}
    
    # D2
    try:
        r = eval_d2_transition_type(raw_outputs.get("transnetv2", {}), prompt,
                                     alignment_result=alignment_result)
        results["D2"] = r
        logger.log_metric("D2 effect transition type", r["score"])
    except Exception as e:
        logger.log_error("D2", str(e))
        results["D2"] = {"dimension": "D2", "score": 0.0, "error": str(e)}
    
    # D3
    try:
        r = eval_d3_audio_visual_relation(raw_outputs.get("transnetv2", {}),
                                          raw_outputs.get("whisper", {}),
                                          prompt, video_path,
                                          alignment_result=alignment_result,
                                          demucs_result=raw_outputs.get("demucs", {}))
        results["D3"] = r
        logger.log_metric("D3 transition audio-visual relation", r["score"])
    except Exception as e:
        logger.log_error("D3", str(e))
        results["D3"] = {"dimension": "D3", "score": 0.0, "error": str(e)}
    
    # E1 - migrated to Mode B (MonST3R + VLM)
    # results["E1"] is computed in mode_b_eval.py
    logger.log_info("E1 camera movement type → Mode B (MonST3R + VLM)")
    
    # E2
    try:
        r = eval_e2_video_quality(raw_outputs.get("e2quality", {}))
        results["E2"] = r
        logger.log_metric("E2 visual quality", r["score"],
                         f"aes={r.get('aesthetic', 0):.3f}")
    except Exception as e:
        logger.log_error("E2", str(e))
        results["E2"] = {"dimension": "E2", "score": 0.0, "error": str(e)}
    
    # E3
    try:
        r = eval_e3_style_consistency(raw_outputs.get("dinov2", {}),
                                      prompt=prompt,
                                      alignment_result=alignment_result,
                                      video_path=video_path)
        results["E3"] = r
        logger.log_metric("E3 style consistency", r["score"])
    except Exception as e:
        logger.log_error("E3", str(e))
        results["E3"] = {"dimension": "E3", "score": 0.0, "error": str(e)}
    
    # F1
    try:
        r = eval_f1_av_sync(raw_outputs.get("raft", {}), video_path)
        results["F1"] = r
        logger.log_metric("F1 in-shot audio-visual sync", r["score"],
                         r.get("note", f"offset={r.get('mean_offset', 'N/A')}"))
    except Exception as e:
        logger.log_error("F1", str(e))
        results["F1"] = {"dimension": "F1", "score": 0.0, "error": str(e)}
    
    # F2 (DNSMOS signal-processing quality)
    try:
        r = eval_f2_audio_quality(raw_outputs.get("dnsmos", {}))
        results["F2"] = r
        logger.log_metric("F2 overall audio perceptual quality", r["score"],
                         f"signal={r.get('signal_score', 'N/A')}, "
                         f"snr={r.get('snr_db', 'N/A')}")
    except Exception as e:
        logger.log_error("F2", str(e))
        results["F2"] = {"dimension": "F2", "score": 0.0, "error": str(e)}
    
    # ---- Step 5: aggregate ----
    logger.log_subsection("Step 5: dimension summary")
    
    # Aggregation: dimensions sharing a letter+digit are averaged
    dimension_scores = {}
    for key, val in results.items():
        if key.startswith("D1-"):
            continue  # D1 sub-dimensions are already aggregated
        dim = key
        score = val.get("score", 0.0)
        # Skip when score is None (e.g. every D1 transition was a Mode B type)
        if score is None:
            continue
        dimension_scores[dim] = float(score)
    
    # Print the summary table
    logger.log("\n| dimension | score |")
    logger.log("|------|------|")
    for dim in sorted(dimension_scores.keys()):
        logger.log(f"| {dim} | {dimension_scores[dim]:.4f} |")
    
    overall = float(np.mean(list(dimension_scores.values()))) if dimension_scores else 0.0
    logger.log(f"\n**overall mean**: `{overall:.4f}`\n")
    
    return {
        "video_id": prompt.get("id"),
        "video_title": prompt.get("title"),
        "video_path": video_path,
        "dimension_scores": dimension_scores,
        "detailed_results": results,
        "overall_score": overall,
        "alignment_result": alignment_result,
        "raw_outputs": raw_outputs,
    }


# ============================================================
#  Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Mode A Evaluation Engine")
    parser.add_argument("--video-dir", required=True,
                        help="Directory containing test videos")
    parser.add_argument("--prompt", required=True,
                        help="Path to prompt_init.json")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: benchmark/mode_a_results.json)")
    parser.add_argument("--log", default=None,
                        help="Test log markdown path (default: benchmark/test_log.md)")
    parser.add_argument("--video-ids", type=str, default=None,
                        help="Comma-separated video IDs to evaluate (default: all)")
    args = parser.parse_args()
    
    # Paths
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(benchmark_dir, "mode_a_results.json")
    log_path = args.log or os.path.join(benchmark_dir, "test_log.md")
    
    # Init logger
    logger = TestLogger(log_path)
    logger.log("## Configuration\n")
    logger.log(f"- video directory: `{args.video_dir}`")
    logger.log(f"- prompt file: `{args.prompt}`")
    logger.log(f"- output file: `{output_path}`")
    logger.log(f"- start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load prompts
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    logger.log(f"- total videos: {len(prompts)}")
    
    # Filter video IDs if specified
    if args.video_ids:
        target_ids = [int(x.strip()) for x in args.video_ids.split(",")]
        prompts = [p for p in prompts if p["id"] in target_ids]
        logger.log(f"- filtered videos: {target_ids}")
    
    # Check services
    logger.log_section("service health check")
    health = check_all_services()
    for name, status in health.items():
        icon = "✅" if status.get("status") == "ok" else "❌"
        logger.log(f"- {icon} {name} (port {status.get('port', '?')}): {status.get('status', 'unknown')}")
    
    # Map video files
    video_files = {}
    for f in sorted(os.listdir(args.video_dir)):
        if f.endswith((".mp4", ".avi", ".mov", ".mkv")):
            # Try to extract video ID from filename (e.g., "1_Rain_..." → 1)
            match = re.match(r'^(\d+)_', f)
            if match:
                vid = int(match.group(1))
                video_files[vid] = os.path.join(args.video_dir, f)
    
    logger.log(f"\nvideo files found: {list(video_files.keys())}")
    
    # Evaluate each video
    all_results = []
    logger.log_section("evaluation run")
    
    for prompt in prompts:
        vid = prompt["id"]
        if vid not in video_files:
            logger.log(f"\n⚠️ Video ID={vid} has no matching video file, skipping")
            continue
        
        video_path = video_files[vid]
        t0 = time.time()
        result = evaluate_single_video(video_path, prompt, logger)
        elapsed = time.time() - t0
        logger.log(f"\n⏱️ elapsed: {elapsed:.1f}s")
        all_results.append(result)
    
    # Final summary
    logger.log_section("final summary")
    
    if all_results:
    # Per-dimension mean across videos
        all_dims = set()
        for r in all_results:
            all_dims.update(r["dimension_scores"].keys())
        
        logger.log("\n| dimension | " + " | ".join(f"V{r['video_id']}" for r in all_results) + " | mean |")
        logger.log("|------|" + "|".join(["------"] * (len(all_results) + 1)) + "|")
        
        dim_averages = {}
        for dim in sorted(all_dims):
            scores = [r["dimension_scores"].get(dim, 0.0) for r in all_results]
            avg = float(np.mean(scores))
            dim_averages[dim] = avg
            row = f"| {dim} | " + " | ".join(f"{s:.4f}" for s in scores) + f" | **{avg:.4f}** |"
            logger.log(row)
        
        overall_avg = float(np.mean(list(dim_averages.values())))
        logger.log(f"\n**overall mean: {overall_avg:.4f}**")
    
    # Save results
    output_data = {
        "evaluation_mode": "Mode A",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "video_dir": args.video_dir,
            "prompt_file": args.prompt,
        },
        "results": all_results,
        "dimension_averages": dim_averages if all_results else {},
        "overall_score": overall_avg if all_results else 0.0,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.log(f"\n✅ results saved: `{output_path}`")
    logger.log(f"\n---\n> finish time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{'='*60}")
    print(f"  Mode A evaluation finished")
    print(f"  results: {output_path}")
    print(f"  log: {log_path}")
    print(f"  overall mean: {overall_avg if all_results else 0.0:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
