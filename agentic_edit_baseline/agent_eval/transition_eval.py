#!/usr/bin/env python3
"""Self-evaluation module for the editing agent -- B1 transition timing / D2 transition effect / D3 transition audio-visual relation.

Design constraints (fully isolated from benchmark):
  1. This file is SELF-CONTAINED: it imports nothing from benchmark/, and both the evaluation logic
     and the expert-model calls are inlined here, so the agent needs only this file for all three dimensions.
  2. The expert-model microservice ports are OFFSET from the formal evaluation (default +100):
        TransNetV2 8101 / Whisper 8104 / Demucs 8105 / PANNs 8107
     the 8001-8015 range used by the formal evaluation is never touched on any code path,
     and AGENT_EVAL_PORT_OFFSET or AGENT_EVAL_PORT_<SERVICE> can override it.
  3. GT comes from the agent's own generation plan (edit_decisions/shot_plan), so no VLM shot-alignment
     service is needed: the agent knows each shot's planned net duration and planned cut, and only has
     to align the cuts TransNetV2 detects back onto the planned ones.

The three dimensions follow the same definitions as benchmark Mode A:
  B1 = clip01(1 - mean(|d_pred - d_gt| / d_gt))            threshold 0.90
  D2 = mean(5-way hit; dissolve<->wipe confusion 0.5)      threshold 0.70
  D3 = mean(predicted_relation == gt_relation)             threshold 0.50
       prediction goes through signal arbitration > VLM > signal fallback, never reading GT.

Usage:
  python transition_eval.py --video FINAL.mp4 --decisions edit_decisions.json
  python transition_eval.py --video FINAL.mp4 --decisions edit_decisions.json \
         --dimensions B1,D2 --output eval.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# Directory holding ffmpeg / ffprobe. The versions on PATH are used by default; to share the
# environment used by the expert services, export AV_PROCESS_BIN=/path/to/env/bin before running.
AV_PROCESS_BIN = os.environ.get("AV_PROCESS_BIN", "")
if AV_PROCESS_BIN and AV_PROCESS_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = AV_PROCESS_BIN + os.pathsep + os.environ.get("PATH", "")


# ============================================================
#  1. Expert microservice client (offset ports, independent of benchmark's service_client)
# ============================================================

BASE_URL = os.environ.get("AGENT_EVAL_SERVICE_BASE_URL", "http://localhost")
# Port offset from the formal evaluation. Formal evaluation: 8001/8004/8005/8007
PORT_OFFSET = int(os.environ.get("AGENT_EVAL_PORT_OFFSET", "100"))
# Ports held by the formal evaluation, treated as a hard deny list
_BENCHMARK_RESERVED_PORTS = set(range(8001, 8016))

SERVICE_PORTS = {
    "transnetv2": int(os.environ.get("AGENT_EVAL_PORT_TRANSNETV2", 8001 + PORT_OFFSET)),
    "whisper": int(os.environ.get("AGENT_EVAL_PORT_WHISPER", 8004 + PORT_OFFSET)),
    "demucs": int(os.environ.get("AGENT_EVAL_PORT_DEMUCS", 8005 + PORT_OFFSET)),
    "panns": int(os.environ.get("AGENT_EVAL_PORT_PANNS", 8007 + PORT_OFFSET)),
}

for _name, _port in SERVICE_PORTS.items():
    if _port in _BENCHMARK_RESERVED_PORTS:
        raise RuntimeError(
            f"agent self-eval service {_name} port {_port} falls inside the 8001-8015 range reserved for the formal evaluation, "
            f"which would interfere with a running evaluation. Adjust AGENT_EVAL_PORT_OFFSET."
        )

TIMEOUT = int(os.environ.get("AGENT_EVAL_SERVICE_TIMEOUT", "300"))


class ServiceUnavailable(RuntimeError):
    """An expert microservice is unavailable (not started / health check failed)."""


def get_service_url(service_name: str) -> str:
    return f"{BASE_URL}:{SERVICE_PORTS[service_name]}"


def check_health(service_name: str) -> dict:
    try:
        resp = requests.get(f"{get_service_url(service_name)}/health", timeout=10)
        return resp.json()
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


def check_services(required: Optional[List[str]] = None) -> Dict[str, dict]:
    """Check the services the agent self-evaluation needs; returns {service: health}."""
    names = required or list(SERVICE_PORTS)
    return {name: check_health(name) for name in names}


def require_services(required: List[str]) -> None:
    health = check_services(required)
    down = [n for n, h in health.items() if h.get("status") == "error"]
    if down:
        ports = ", ".join(f"{n}:{SERVICE_PORTS[n]}" for n in down)
        raise ServiceUnavailable(
            f"the following agent self-eval services are not ready: {ports}. "
            f"Run bash edit_baseline/agent_eval/start_agent_services.sh start first."
        )


def call_service(service_name: str, file_path: str,
                 extra_params: Optional[dict] = None,
                 max_retries: int = 2) -> dict:
    """Call a microservice /predict. Retries quickly on failure, with no watchdog recovery wait (the agent-side services have no watchdog)."""
    url = f"{get_service_url(service_name)}/predict"
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "video/mp4")}
                resp = requests.post(url, files=files, data=extra_params or {}, timeout=TIMEOUT)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            result = resp.json()
            if not result.get("success", True):
                raise RuntimeError(f"{service_name} error: {result.get('error')}")
            return result
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [Retry] {service_name} attempt {attempt + 1} failed: {e}, retry in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"service {service_name} failed after {max_retries + 1} attempts: {last_err}")


def call_transnetv2(video_path: str, threshold: float = 0.35, expected_shots: int = 0) -> dict:
    params = {"threshold": str(threshold)}
    if expected_shots > 0:
        params["expected_shots"] = str(expected_shots)
    return call_service("transnetv2", video_path, params)


def call_whisper(video_path: str, language: Optional[str] = None,
                 word_timestamps: bool = True,
                 condition_on_previous_text: bool = True,
                 word_gap_threshold: float = 0.1) -> dict:
    """Word-level timestamps are on by default -- D3's speech projection arbitration relies entirely on the words field."""
    params = {
        "word_timestamps": str(word_timestamps).lower(),
        "condition_on_previous_text": str(condition_on_previous_text).lower(),
        "word_gap_threshold": str(word_gap_threshold),
    }
    if language:
        params["language"] = language
    return call_service("whisper", video_path, params)


def call_demucs(video_path: str) -> dict:
    return call_service("demucs", video_path)


def call_panns(audio_path: str, segment_duration: float = 0.0) -> dict:
    return call_service("panns", audio_path, {"segment_duration": str(segment_duration)})


# ============================================================
#  2. Shared utilities (ffprobe / ffmpeg / VLM)
# ============================================================

def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def get_video_duration(video_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, timeout=15,
        )
        data = json.loads(r.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:  # noqa: BLE001
        return 0.0


def _ensure_min_duration(video_path: str, min_dur: float = 2.0) -> str:
    dur = get_video_duration(video_path)
    if dur <= 0 or dur >= min_dur:
        return video_path
    loop_count = int(min_dur // dur) + 1
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-stream_loop", str(loop_count), "-i", video_path,
         "-t", str(min_dur), "-c:v", "libx264", "-c:a", "aac",
         "-preset", "ultrafast", "-q:v", "23", tmp.name],
        capture_output=True, timeout=90,
    )
    return tmp.name


def _compress_video_for_vlm(video_path: str, max_size_mb: float = 4.0) -> str:
    if os.path.getsize(video_path) / (1024 * 1024) <= max_size_mb:
        return video_path
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", "scale=-2:360",
         "-b:v", "1500k", "-b:a", "48k", "-ac", "1", "-r", "15", tmp.name],
        capture_output=True, timeout=120,
    )
    return tmp.name


_VLM_MODEL = os.environ.get("AGENT_EVAL_VLM_MODEL", "qwen3.5-omni-plus")
_DASHSCOPE_BASE_URL = os.environ.get(
    "AGENT_EVAL_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def _dashscope_keys() -> List[str]:
    raw = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    return [k.strip() for k in raw.split(",") if k.strip()]


# "Quota exhausted" errors: exponential backoff is pointless, the quota has to reset. D3 is called
# repeatedly inside the repair loop and one raised error degrades the whole round's D3 to the signal
# fallback, so sleep a fixed 2 minutes and retry without consuming the max_retries budget.
_VLM_QUOTA_KEYWORDS = [
    "insufficient_quota", "exceeded your current quota", "allocated quota exceeded",
    "allocationquotaexceeded", "budget", "arrearage", "预算", "额度已用尽",
    "额度不足", "余额不足", "欠费",
]
VLM_QUOTA_WAIT_SECONDS = float(os.environ.get("AGENT_EVAL_VLM_QUOTA_WAIT_SECONDS", "120"))
VLM_QUOTA_MAX_WAITS = int(os.environ.get("AGENT_EVAL_VLM_QUOTA_MAX_WAITS", "10"))


def _is_vlm_quota_exhausted(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _VLM_QUOTA_KEYWORDS)


def call_vlm(system_prompt: str, user_text: str, video_path: Optional[str] = None,
             max_tokens: int = 900, temperature: float = 0.0,
             max_retries: int = 3) -> str:
    """Call the DashScope omni VLM (OpenAI-compatible endpoint, plain requests, no openai package).

    Quota exhaustion is not an ordinary failure: rotate every key first, and if all are exhausted sleep
    VLM_QUOTA_WAIT_SECONDS (2 minutes by default) and continue, for at most VLM_QUOTA_MAX_WAITS rounds.
    """
    keys = _dashscope_keys()
    if not keys:
        raise RuntimeError("DASHSCOPE_API_KEY is not set, the D3 VLM judgement cannot run")

    combined = f"[ROLE]\n{system_prompt}\n\n[TASK]\n{user_text}"
    if video_path:
        prepared = _ensure_min_duration(video_path)
        compressed = _compress_video_for_vlm(prepared)
        with open(compressed, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        for path in {compressed, prepared} - {video_path}:
            try:
                os.unlink(path)
            except OSError:
                pass
        content: Any = [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
            {"type": "text", "text": combined},
        ]
    else:
        content = [{"type": "text", "text": combined}]

    body = {
        "model": _VLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["text"],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = f"{_DASHSCOPE_BASE_URL.rstrip('/')}/chat/completions"
    last_err: Optional[Exception] = None
    attempt = 0
    quota_waits = 0
    quota_keys_tried = 0
    key_idx = 0
    while attempt < max_retries:
        key = keys[key_idx % len(keys)]
        try:
            resp = requests.post(
                url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body, timeout=300,
            )
            data = resp.json()
            if resp.status_code != 200 or data.get("error"):
                raise RuntimeError(f"{resp.status_code} {data.get('error', data)}")
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_vlm_quota_exhausted(e):
                if len(keys) > 1 and quota_keys_tried < len(keys) - 1:
                    quota_keys_tried += 1
                    key_idx += 1
                    print(f"  [VLM QUOTA] quota exhausted, switching key -> {keys[key_idx % len(keys)][:8]}...")
                    continue
                if quota_waits < VLM_QUOTA_MAX_WAITS:
                    quota_waits += 1
                    quota_keys_tried = 0
                    print(f"  [VLM QUOTA] budget/quota exhausted ({str(e)[:120]}), "
                          f"continuing after sleep {VLM_QUOTA_WAIT_SECONDS:.0f}s "
                          f"({quota_waits}/{VLM_QUOTA_MAX_WAITS})")
                    time.sleep(VLM_QUOTA_WAIT_SECONDS)
                    continue
                break
            attempt += 1
            key_idx += 1
            if attempt < max_retries:
                wait = min(120.0, 20.0 * (2 ** (attempt - 1)))
                print(f"  [VLM RETRY] attempt {attempt}/{max_retries} failed: {e}, wait {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"VLM call failed after {max_retries} attempts: {last_err}")


def _extract_json_from_response(response: str) -> Optional[dict]:
    if not response or response.upper().strip().startswith("ERROR:"):
        return None
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    first, last = response.find("{"), response.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(response[first:last + 1])
        except json.JSONDecodeError:
            pass
    return None


# ============================================================
#  3. Planned -> detected cut alignment (replaces benchmark's VLM shot alignment)
# ============================================================

def build_plan_from_decisions(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the evaluation plan structure (GT) from edit_decisions.decisions.

    Returns:
      {
        "shots": [{shot_id, net_duration, planned_start, planned_end,
                   prompt, optical_effect, audio_visual_relation,
                   transition_duration_seconds, timing_offset_seconds}],
        "planned_cuts": [c1, c2, ...],   # n-1 planned cuts
        "total_duration": float,
      }
    """
    shots: List[Dict[str, Any]] = []
    cursor = 0.0
    for dec in decisions:
        net = float(dec.get("net_duration", dec.get("duration", 0)) or 0)
        trans = dec.get("transition_out") or {}
        shots.append({
            "shot_id": dec.get("shot_id"),
            "net_duration": round(net, 4),
            "planned_start": round(cursor, 4),
            "planned_end": round(cursor + net, 4),
            "prompt": dec.get("prompt", ""),
            "optical_effect": trans.get("optical_effect") or "",
            "audio_visual_relation": trans.get("audio_visual_relation") or "",
            "transition_duration_seconds": trans.get("transition_duration_seconds"),
            "timing_offset_seconds": trans.get("timing_offset_seconds"),
            "gt_transition_duration_seconds": trans.get("gt_transition_duration_seconds"),
            "gt_timing_offset_seconds": trans.get("gt_timing_offset_seconds"),
            "has_transition": bool(trans),
        })
        cursor += net
    return {
        "shots": shots,
        "planned_cuts": [s["planned_end"] for s in shots[:-1]],
        "total_duration": round(cursor, 4),
    }


def align_cuts_to_plan(transnetv2_result: dict, plan: Dict[str, Any],
                       video_duration: Optional[float] = None) -> Dict[str, Any]:
    """Monotonically align the cuts TransNetV2 detected back onto the planned cuts.

    For each planned cut, take the nearest unclaimed detected cut inside a tolerance window of
    max(0.6s, 0.3 * the shorter net duration of the two neighbouring shots), forcing the matches to increase monotonically.

    Returns:
      {
        "video_duration": float,
        "detected_cuts": [...],              # times of the detected internal cuts
        "n_detected_shots": int,
        "transitions": [{index, planned_cut, cut_time, matched, offset, detected_transition}],
        "aligned_shots": [{shot_idx, shot_id, status, time_range, planned_duration, pred_duration}],
      }
    """
    detected_shots = transnetv2_result.get("shots", []) or []
    detected_transitions = transnetv2_result.get("transitions", []) or []
    total_dur = float(video_duration or transnetv2_result.get("duration")
                      or plan.get("total_duration") or 0)

    # Detected cut times: prefer the midpoint of transitions (steadier for gradual ones), fall back to shot boundaries
    detected_cuts: List[Dict[str, Any]] = []
    for t in detected_transitions:
        try:
            start = float(t.get("start_time", 0))
            end = float(t.get("end_time", start))
        except (TypeError, ValueError):
            continue
        detected_cuts.append({
            "time": round((start + end) / 2.0, 4),
            "start_time": start,
            "end_time": end,
            "type": t.get("type", ""),
            "confidence": t.get("confidence"),
        })
    if not detected_cuts:
        for s in detected_shots[1:]:
            try:
                start = float(s.get("start_time", 0))
            except (TypeError, ValueError):
                continue
            detected_cuts.append({"time": round(start, 4), "start_time": start,
                                  "end_time": start, "type": "", "confidence": None})
    detected_cuts.sort(key=lambda c: c["time"])

    shots = plan["shots"]
    planned_cuts = plan["planned_cuts"]
    used = set()
    transitions: List[Dict[str, Any]] = []
    last_matched_time = -1.0
    for i, planned_cut in enumerate(planned_cuts):
        left_net = float(shots[i]["net_duration"])
        right_net = float(shots[i + 1]["net_duration"]) if i + 1 < len(shots) else left_net
        tol = max(0.6, 0.3 * min(left_net, right_net))
        best_idx, best_dist = None, None
        for j, cut in enumerate(detected_cuts):
            if j in used or cut["time"] <= last_matched_time:
                continue
            dist = abs(cut["time"] - planned_cut)
            if dist > tol:
                continue
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = j, dist
        if best_idx is None:
            transitions.append({
                "index": i, "planned_cut": planned_cut, "cut_time": None,
                "matched": False, "offset": None, "tolerance": round(tol, 3),
                "detected_transition": None,
            })
            continue
        used.add(best_idx)
        cut = detected_cuts[best_idx]
        last_matched_time = cut["time"]
        transitions.append({
            "index": i, "planned_cut": planned_cut, "cut_time": cut["time"],
            "matched": True, "offset": round(cut["time"] - planned_cut, 4),
            "tolerance": round(tol, 3), "detected_transition": cut,
        })

    # known boundaries (including the first and last)
    boundaries: List[Optional[float]] = [0.0]
    boundaries.extend(t["cut_time"] for t in transitions)
    boundaries.append(total_dur if total_dur > 0 else plan["total_duration"])

    aligned_shots: List[Dict[str, Any]] = []
    for idx, shot in enumerate(shots):
        left, right = boundaries[idx], boundaries[idx + 1]
        status = "matched"
        if left is None or right is None:
            status = "merged"  # a missed neighbouring cut -> this shot was merged with its neighbour
            # Scan both ways to the nearest known boundary; pred duration is that of the merged span (which honestly penalises B1)
            li = idx
            while li > 0 and boundaries[li] is None:
                li -= 1
            ri = idx + 1
            while ri < len(boundaries) - 1 and boundaries[ri] is None:
                ri += 1
            left = boundaries[li] if boundaries[li] is not None else 0.0
            right = boundaries[ri] if boundaries[ri] is not None else boundaries[-1]
        pred_dur = max(0.0, float(right) - float(left))
        aligned_shots.append({
            "shot_idx": idx,
            "shot_id": shot["shot_id"],
            "status": status,
            "time_range": [round(float(left), 4), round(float(right), 4)],
            "planned_duration": shot["net_duration"],
            "pred_duration": round(pred_dur, 4),
        })

    return {
        "video_duration": round(total_dur, 4),
        "detected_cuts": detected_cuts,
        "n_detected_shots": int(transnetv2_result.get("num_shots", len(detected_shots))),
        "n_planned_shots": len(shots),
        "transitions": transitions,
        "aligned_shots": aligned_shots,
    }


# ============================================================
#  4. B1: shot / transition timing accuracy
# ============================================================

def eval_b1_shot_duration(plan: Dict[str, Any], alignment: Dict[str, Any]) -> dict:
    """B1 = clip01(1 - mean(|d_pred - d_gt| / d_gt)), the per-shot relative duration error.

    Also reports each cut's time drift (cut_offset), so the central planner can find the cuts to nudge.
    """
    aligned_shots = alignment.get("aligned_shots", [])
    errors: List[float] = []
    per_shot: List[Dict[str, Any]] = []
    for ashot in aligned_shots:
        d_gt = float(ashot["planned_duration"] or 0)
        d_pred = float(ashot["pred_duration"] or 0)
        if d_gt <= 0:
            continue
        rel_err = abs(d_pred - d_gt) / d_gt
        errors.append(rel_err)
        per_shot.append({
            "shot_idx": ashot["shot_idx"],
            "shot_id": ashot["shot_id"],
            "status": ashot["status"],
            "gt_duration": round(d_gt, 3),
            "pred_duration": round(d_pred, 3),
            "relative_error": round(rel_err, 4),
            "delta_seconds": round(d_pred - d_gt, 3),
        })

    per_transition = [{
        "index": t["index"],
        "planned_cut": t["planned_cut"],
        "cut_time": t["cut_time"],
        "matched": t["matched"],
        "cut_offset_seconds": t["offset"],
    } for t in alignment.get("transitions", [])]

    if not errors:
        return {"dimension": "B1", "metric": "shot duration accuracy", "score": 0.0,
                "error": "no valid shot duration", "per_shot": per_shot,
                "per_transition": per_transition, "method": "plan_aligned"}

    mean_err = float(np.mean(errors))
    return {
        "dimension": "B1",
        "metric": "shot duration accuracy",
        "score": clip01(1.0 - mean_err),
        "mean_relative_error": round(mean_err, 4),
        "n_evaluated": len(errors),
        "per_shot": per_shot,
        "per_transition": per_transition,
        "method": "plan_aligned",
    }


# ============================================================
#  5. D2: transition effect type (5-way)
# ============================================================

def normalize_optical_effect(optical_effect: str) -> str:
    """Normalise the prompt's optical_effect onto the 5-way label set."""
    s = (optical_effect or "").lower().strip()
    if any(kw in s for kw in ["hard cut", "straight cut", "hard-cut"]):
        return "hard_cut"
    if any(kw in s for kw in ["flash white", "flash-white", "white flash", "flash-to-white"]):
        return "flash_white"
    if any(kw in s for kw in ["flash black", "flash-black", "black flash",
                              "fade to black", "fade-to-black", "flash-to-black"]):
        return "flash_black"
    if any(kw in s for kw in ["wipe", "iris"]):
        return "wipe"
    if any(kw in s for kw in ["dissolve", "cross-dissolve", "fade", "blend", "morph", "crossfade"]):
        return "dissolve"
    return "hard_cut"


def eval_d2_transition_type(plan: Dict[str, Any], transnetv2_result: dict,
                            alignment: Dict[str, Any]) -> dict:
    """D2: 5-way hit rate. Exact match 1.0; dissolve<->wipe confusion 0.5; missed / wrong 0.0."""
    transitions_pred = transnetv2_result.get("transitions", []) or []
    shots = plan["shots"]
    aligned_transitions = alignment.get("transitions", [])

    scores: List[float] = []
    details: List[Dict[str, Any]] = []
    for i, shot in enumerate(shots[:-1]):
        gt_effect = shot.get("optical_effect") or ""
        if not gt_effect:
            continue
        gt_type = normalize_optical_effect(gt_effect)
        at = aligned_transitions[i] if i < len(aligned_transitions) else {}
        ref_time = at.get("cut_time") if at.get("cut_time") is not None else shot["planned_end"]

        matched_pred = None
        for t in transitions_pred:
            try:
                if abs(float(t.get("start_time", 0)) - float(ref_time)) < 1.0:
                    matched_pred = t
                    break
            except (TypeError, ValueError):
                continue

        if matched_pred is None:
            scores.append(0.0)
            details.append({
                "index": i, "score": 0.0, "gt_effect": gt_effect, "gt_type": gt_type,
                "pred_type": None, "cut_time": ref_time,
                "rendered_effect_seconds": shot.get("transition_duration_seconds"),
                "note": "no_pred: this cut point was not detected as a transition (an effect that is too short reads as a hard cut, too long reads as a blurry dissolve)",
            })
            continue

        pred_type = matched_pred.get("type", "")
        pred_span = round(float(matched_pred.get("end_time", 0)) -
                          float(matched_pred.get("start_time", 0)), 4)
        if pred_type == gt_type:
            score, note = 1.0, "match"
        elif {pred_type, gt_type} == {"dissolve", "wipe"}:
            score, note = 0.5, "partial: dissolve/wipe cross"
        else:
            score, note = 0.0, "mismatch"
        scores.append(score)
        details.append({
            "index": i, "score": score, "gt_effect": gt_effect, "gt_type": gt_type,
            "pred_type": pred_type, "cut_time": ref_time,
            "pred_detected_span_seconds": pred_span,
            "rendered_effect_seconds": shot.get("transition_duration_seconds"),
            "confidence": matched_pred.get("confidence"),
            "note": note,
        })

    if not scores:
        return {"dimension": "D2", "metric": "effect transition type", "score": 0.5,
                "note": "no GT optical_effect", "per_transition": []}

    return {
        "dimension": "D2",
        "metric": "effect transition type",
        "score": clip01(float(np.mean(scores))),
        "per_transition": details,
    }


# ============================================================
#  6. D3: transition audio-visual relation (signal arbitration + VLM)
# ============================================================

D3_OBJECT_SEGMENT_DURATION = 0.5
D3_OBJECT_MIN_DBFS = -42.0
D3_OBJECT_WEAK_NOISE_DBFS = -34.0
D3_OBJECT_MIN_TAG_CONFIDENCE = 0.12
D3_OBJECT_RUN_GAP = 0.6
D3_OBJECT_SYNC_ONSET_THRESHOLD = 0.25
D3_OBJECT_SYNC_TAIL_THRESHOLD = float(os.environ.get("D3_OBJECT_SYNC_TAIL_THRESHOLD", "0.25"))
D3_SYNC_STRAIGHT_THRESHOLD = 0.15
D3_WORD_GAP_SPLIT_THRESHOLD = 0.1

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


def normalize_d3_relation(value: str) -> str:
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


def _extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    import librosa
    if len(y) < sr * 0.1:
        return np.zeros(n_mfcc)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).mean(axis=1)


def _audio_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    na, nb = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (na * nb))


def _analyze_env_audio_continuity(env_y: np.ndarray, sr: int, cut_time: float,
                                  window: float = 0.3, gap: float = 0.3) -> dict:
    """MFCC continuity and energy ratio of the ambience track either side of the cut."""
    n_earlier_start = max(0, int((cut_time - gap - window) * sr))
    n_earlier_end = max(0, int((cut_time - gap) * sr))
    n_before_start = max(0, int((cut_time - window) * sr))
    n_before_end = max(0, int(cut_time * sr))
    n_after_start = min(len(env_y), int(cut_time * sr))
    n_after_end = min(len(env_y), int((cut_time + window) * sr))

    result = {"sim_before_after": 0.0, "sim_earlier_before": 0.0,
              "sim_earlier_after": 0.0, "energy_ratio": 0.0}
    if n_before_end - n_before_start < int(sr * 0.05):
        return result
    if n_after_end - n_after_start < int(sr * 0.05):
        return result

    y_before = env_y[n_before_start:n_before_end]
    y_after = env_y[n_after_start:n_after_end]
    result["sim_before_after"] = _audio_cosine_similarity(
        _extract_mfcc_features(y_before, sr), _extract_mfcc_features(y_after, sr))

    rms_before = _rms_energy(y_before)
    rms_after = _rms_energy(y_after)
    if rms_before > 1e-6 and rms_after > 1e-6:
        result["energy_ratio"] = min(rms_before, rms_after) / max(rms_before, rms_after)

    if n_earlier_end - n_earlier_start >= int(sr * 0.05):
        mfcc_earlier = _extract_mfcc_features(env_y[n_earlier_start:n_earlier_end], sr)
        result["sim_earlier_before"] = _audio_cosine_similarity(
            mfcc_earlier, _extract_mfcc_features(y_before, sr))
        result["sim_earlier_after"] = _audio_cosine_similarity(
            mfcc_earlier, _extract_mfcc_features(y_after, sr))
    return result


def _detect_onset_near_cut(y: np.ndarray, sr: int, cut_time: float,
                           radius: float = 0.45) -> dict:
    result = {"onset_near_cut": False, "onset_offset": None, "onset_strength": 0.0}
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
        local_mean, local_std = float(np.mean(onset_env)), float(np.std(onset_env))
        idx = int(np.argmax(onset_env))
        strength_norm = float(onset_env[idx]) / max(local_mean + local_std, 1e-6)
        offset = float(times[idx]) - cut_time
        result.update({
            "onset_near_cut": bool(abs(offset) <= radius and strength_norm >= 1.8),
            "onset_offset": round(offset, 4),
            "onset_strength": round(strength_norm, 4),
        })
    except Exception:  # noqa: BLE001
        pass
    return result


def _overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _tag_label_text(tags: List[dict]) -> str:
    return " ".join(str(t.get("label", "")).lower() for t in tags or [])


def _object_sound_family(tags: List[dict]) -> str:
    label_text = _tag_label_text(tags)
    for family, keywords in _D3_OBJECT_FAMILY_KEYWORDS:
        if any(kw in label_text for kw in keywords):
            return family
    return "object" if label_text.strip() else "unknown"


def _is_default_straight_noise(tags: List[dict], dbfs: float) -> bool:
    label_text = _tag_label_text(tags)
    if dbfs < D3_OBJECT_MIN_DBFS:
        return True
    has_noise = any(kw in label_text for kw in _D3_OBJECT_DEFAULT_STRAIGHT_KEYWORDS)
    has_salient = any(kw in label_text for kw in _D3_OBJECT_SALIENT_KEYWORDS)
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
                kw in label for kw in _D3_OBJECT_SALIENT_KEYWORDS):
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
            windows.append({
                "track": track_name, "index": idx,
                "start": round(start, 3), "end": round(end, 3),
                "dbfs": round(dbfs, 2), "tags": tags[:5],
                "family": _object_sound_family(tags),
                "salient": _is_salient_object_sound(tags, dbfs),
                "default_straight_noise": _is_default_straight_noise(tags, dbfs),
            })
    windows.sort(key=lambda item: (item["start"], item["end"], item["track"]))
    return windows


def _merge_object_sound_runs(windows: List[dict]) -> List[dict]:
    runs: List[dict] = []
    for win in [w for w in windows if w.get("salient")]:
        family, track = win.get("family", "object"), win.get("track", "")
        current = runs[-1] if runs else None
        if (current and current.get("family") == family and current.get("track") == track and
                float(win["start"]) - float(current["end"]) <= D3_OBJECT_RUN_GAP):
            current["end"] = max(float(current["end"]), float(win["end"]))
            current["max_dbfs"] = max(float(current["max_dbfs"]), float(win["dbfs"]))
            current["windows"].append(win)
            current["top_tags"] = current["top_tags"] or win.get("tags", [])[:3]
        else:
            runs.append({
                "track": track, "family": family,
                "start": float(win["start"]), "end": float(win["end"]),
                "max_dbfs": float(win["dbfs"]), "top_tags": win.get("tags", [])[:3],
                "windows": [win],
            })
    for run in runs:
        run["duration"] = round(float(run["end"]) - float(run["start"]), 4)
        run["start"] = round(float(run["start"]), 4)
        run["end"] = round(float(run["end"]), 4)
        run["max_dbfs"] = round(float(run["max_dbfs"]), 2)
        run["windows"] = run["windows"][:8]
    return runs


def _analyze_object_sound_projection(object_runs: List[dict], object_windows: List[dict],
                                     cut_time: float,
                                     prev_shot_range: Optional[Tuple[float, float]],
                                     next_shot_range: Optional[Tuple[float, float]]) -> dict:
    base = {
        "object_sound_crosses_cut": False,
        "object_sound_projection_mode": "none",
        "object_sound_recommended_owner": "none",
        "object_sound_recommended_relation": "straight",
        "object_sound_reason": "no_salient_object_sound_crossing_cut",
        "object_sound_projection": None,
        "object_sound_windows_near_cut": [
            w for w in object_windows if abs(float(w.get("start", 0.0)) - cut_time) <= 1.0][:12],
    }
    if not prev_shot_range or not next_shot_range:
        base["object_sound_reason"] = "missing_shot_range"
        return base
    prev_start, prev_end = map(float, prev_shot_range)
    next_start, next_end = map(float, next_shot_range)
    best, best_score = None, -1.0
    for run in object_runs or []:
        run_start, run_end = float(run.get("start", 0.0)), float(run.get("end", 0.0))
        prev_overlap = _overlap_duration(run_start, run_end, prev_start, prev_end)
        next_overlap = _overlap_duration(run_start, run_end, next_start, next_end)
        crosses = run_start < cut_time < run_end or (prev_overlap > 0 and next_overlap > 0)
        if not crosses:
            continue
        if abs(run_start - cut_time) <= D3_OBJECT_SYNC_ONSET_THRESHOLD:
            owner, relation = "none", "straight"
            reason = "salient_object_onset_sync_with_cut"
        elif (prev_overlap > 1.0 and next_overlap > 1.0 and
              min(prev_overlap, next_overlap) / max(prev_overlap, next_overlap) >= 0.5):
            owner, relation = "none", "straight"
            reason = "continuous_object_bed_spans_both_shots_without_clear_ownership"
        elif prev_overlap > next_overlap:
            owner, relation = "previous", "l-cut"
            reason = "salient_object_previous_sound_continues_after_cut"
        elif next_overlap > prev_overlap:
            owner, relation = "next", "j-cut"
            reason = "salient_object_next_sound_heard_before_or_through_cut"
        else:
            owner, relation = "unclear", "overlap-unclear"
            reason = "balanced_object_sound_overlap"
        score = (min(prev_overlap, next_overlap) + 0.2 * max(prev_overlap, next_overlap) +
                 max(0.0, float(run.get("max_dbfs", -80)) + 80) / 100.0)
        candidate = {
            "track": run.get("track"), "family": run.get("family"),
            "run_start": round(run_start, 4), "run_end": round(run_end, 4),
            "onset_offset": round(run_start - cut_time, 4),
            "prev_overlap_duration": round(prev_overlap, 4),
            "next_overlap_duration": round(next_overlap, 4),
            "recommended_owner": owner, "recommended_relation": relation,
            "max_dbfs": run.get("max_dbfs"), "top_tags": run.get("top_tags", []),
            "reason": reason,
        }
        if score > best_score:
            best, best_score = candidate, score
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


def _find_word_gap_near_cut(words: List[dict], cut_time: float,
                            sync_threshold: float = D3_SYNC_STRAIGHT_THRESHOLD,
                            gap_threshold: float = D3_WORD_GAP_SPLIT_THRESHOLD) -> dict:
    valid_words = []
    for word in words or []:
        try:
            start, end = float(word.get("start", 0.0)), float(word.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end >= start:
            valid_words.append({"word": str(word.get("word", "")).strip(),
                                "start": start, "end": end})
    valid_words.sort(key=lambda item: (item["start"], item["end"]))

    best_gap, best_distance = None, None
    for prev_word, next_word in zip(valid_words, valid_words[1:]):
        gap_start, gap_end = prev_word["end"], next_word["start"]
        gap = gap_end - gap_start
        if gap < gap_threshold:
            continue
        distance = abs(gap_start - cut_time)
        if distance > sync_threshold:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_gap = {
                "speech_word_gap_split": True,
                "speech_word_gap": round(gap, 4),
                "speech_word_gap_start": round(gap_start, 4),
                "speech_word_gap_end": round(gap_end, 4),
                "speech_sentence_end": round(gap_start, 4),
                "speech_sentence_end_offset": round(gap_start - cut_time, 4),
                "speech_sentence_end_cut_distance": round(distance, 4),
                "speech_word_gap_prev_word": prev_word["word"],
                "speech_word_gap_next_word": next_word["word"],
            }
    if best_gap:
        return best_gap
    return {
        "speech_word_gap_split": False, "speech_word_gap": 0.0,
        "speech_word_gap_start": None, "speech_word_gap_end": None,
        "speech_sentence_end": None, "speech_sentence_end_offset": None,
        "speech_sentence_end_cut_distance": None,
        "speech_word_gap_prev_word": "", "speech_word_gap_next_word": "",
    }


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
    best, best_score = None, -1.0

    for seg in segments or []:
        try:
            seg_start, seg_end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if seg_end <= seg_start:
            continue
        prev_overlap = _overlap_duration(seg_start, seg_end, prev_start, prev_end)
        next_overlap = _overlap_duration(seg_start, seg_end, next_start, next_end)
        # keep only the speech segments that really cross the cut (overlapping on both sides)
        if prev_overlap <= 0.0 or next_overlap <= 0.0:
            continue

        prev_ratio = clip01(prev_overlap / prev_duration)
        next_ratio = clip01(next_overlap / next_duration)
        both_full = bool(prev_ratio >= 0.95 and next_ratio >= 0.95)
        if both_full:
            mode, owner = "cross_both_full", "unclear"
            reason, bias = "same_speech_covers_both_complete_shots", 0.0
        else:
            mode = "cross_partial"
            if prev_overlap > next_overlap:
                owner, reason = "previous", "previous_shot_has_longer_speech_coverage"
            elif next_overlap > prev_overlap:
                owner, reason = "next", "next_shot_has_longer_speech_coverage"
            else:
                owner, reason = "unclear", "speech_coverage_is_balanced"
            bias = abs(prev_overlap - next_overlap) / max(prev_overlap + next_overlap, 1e-6)

        score = min(prev_overlap, next_overlap) + 0.25 * (prev_overlap + next_overlap)
        candidate = {
            "segment_id": seg.get("id"),
            "segment_start": round(seg_start, 4), "segment_end": round(seg_end, 4),
            "segment_text": str(seg.get("text", ""))[:300],
            "prev_shot_range": [round(prev_start, 4), round(prev_end, 4)],
            "next_shot_range": [round(next_start, 4), round(next_end, 4)],
            "prev_overlap_duration": round(prev_overlap, 4),
            "next_overlap_duration": round(next_overlap, 4),
            "prev_coverage_ratio": round(prev_ratio, 4),
            "next_coverage_ratio": round(next_ratio, 4),
            "both_shots_fully_covered": both_full,
            "mode": mode, "recommended_owner": owner,
            "bias_strength": round(clip01(bias), 4), "reason": reason,
        }
        if score > best_score:
            best, best_score = candidate, score

    if not best:
        return base
    return {
        "speech_projection_crosses_cut": True,
        "speech_projection_mode": best["mode"],
        "speech_projection_recommended_owner": best["recommended_owner"],
        "speech_projection_reason": best["reason"],
        "speech_projection": best,
    }


def _analyze_speech_overlap(segments: List[dict], words: List[dict],
                            vocals_y: np.ndarray, vocals_sr: int, cut_time: float,
                            window: float = 0.35,
                            prev_shot_range: Optional[Tuple[float, float]] = None,
                            next_shot_range: Optional[Tuple[float, float]] = None) -> dict:
    crossing_segments = []
    nearest_boundary_offset = None
    max_before = max_after = 0.0
    for seg in segments or []:
        try:
            start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        for boundary in (start, end):
            offset = boundary - cut_time
            if abs(offset) <= window:
                if nearest_boundary_offset is None or abs(offset) < abs(nearest_boundary_offset):
                    nearest_boundary_offset = offset
        if start < cut_time < end:
            max_before = max(max_before, cut_time - start)
            max_after = max(max_after, end - cut_time)
            crossing_segments.append(seg)

    word_gap = _find_word_gap_near_cut(words, cut_time)
    speech_projection = _analyze_speech_projection(segments, cut_time, prev_shot_range, next_shot_range)
    if word_gap.get("speech_word_gap_split") and word_gap.get("speech_sentence_end_offset") is not None:
        nearest_boundary_offset = word_gap["speech_sentence_end_offset"]

    before_energy = _rms_energy(_slice_audio_window(vocals_y, vocals_sr, cut_time - window, cut_time))
    after_energy = _rms_energy(_slice_audio_window(vocals_y, vocals_sr, cut_time, cut_time + window))
    energy_ratio = 0.0
    if before_energy > 1e-7 and after_energy > 1e-7:
        energy_ratio = min(before_energy, after_energy) / max(before_energy, after_energy)

    has_local_voice_energy = before_energy > 1e-4 and after_energy > 1e-4 and energy_ratio > 0.25
    has_balanced_segment = max_before >= 0.15 and max_after >= 0.15
    speech_overlap = bool(crossing_segments and not word_gap.get("speech_word_gap_split")
                          and has_balanced_segment and has_local_voice_energy)
    confidence = 0.0
    if crossing_segments:
        confidence = 0.5 * clip01(min(max_before, max_after) / 0.5) + 0.5 * clip01(energy_ratio)

    return {
        "speech_overlap": speech_overlap,
        "speech_overlap_confidence": round(confidence, 4),
        "speech_before_energy": round(before_energy, 6),
        "speech_after_energy": round(after_energy, 6),
        "speech_energy_ratio": round(energy_ratio, 4),
        "speech_crossing_segments": crossing_segments,
        **word_gap,
        **speech_projection,
        "speech_nearest_boundary_offset": (round(nearest_boundary_offset, 4)
                                           if nearest_boundary_offset is not None else None),
        "speech_boundary_sync_within_0p15": bool(
            word_gap.get("speech_word_gap_split") and
            word_gap.get("speech_sentence_end_offset") is not None and
            abs(float(word_gap["speech_sentence_end_offset"])) <= D3_SYNC_STRAIGHT_THRESHOLD
        ),
        "speech_max_before": round(max_before, 4),
        "speech_max_after": round(max_after, 4),
    }


# ------------------------------------------------------------
#  D3 ambience channel: with no speech, locate the "audio cut" from ambience loudness / frequency
#  jumps, then decide J/L/straight with the same attribution logic as speech projection.
#
#  The stitcher only puts the audio cut in three places (delta = the J/L audio offset, a global render constant):
#     J-cut:    the next shot's audio enters at cut-delta -> [cut-delta, cut] has both shots overlaid
#     straight: the two tracks meet at cut               -> cut is one complete replacement
#     L-cut:    the previous track continues to cut+delta -> [cut, cut+delta] has both shots overlaid
#  Take four blocks of length delta: A=[cut-2d,cut-d] B=[cut-d,cut] C=[cut,cut+d] D=[cut+d,cut+2d];
#  A is always the previous shot alone and D always the next shot alone, so:
#     straight: B=P,   C=N     -> the strongest jump is at B|C, of the same magnitude as the two shots' own ambience difference d(A,D)
#     J-cut:    B=P+N, C=N     -> B leans clearly towards next
#     L-cut:    B=P,   C=P+N   -> C leans clearly towards prev
# ------------------------------------------------------------

# delta must match the renderer's lib.transition_map.JL_CUT_OFFSET_SECONDS, otherwise the four blocks
# miss the real overlap spans (0.1s off measurably drops accuracy). Only the global constant is used;
# no individual transition's GT is read (a per-transition timing_offset maps 1:1 to its relation, so reading it would be reading GT).
try:  # pragma: no cover - requires edit_baseline on sys.path
    _EDIT_BASELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _EDIT_BASELINE_ROOT not in sys.path:
        sys.path.insert(0, _EDIT_BASELINE_ROOT)
    from lib.transition_map import JL_CUT_OFFSET_SECONDS as _RENDER_JL_OFFSET
except Exception:  # noqa: BLE001
    _RENDER_JL_OFFSET = 0.8
D3_AMBIENT_OFFSET_SECONDS = float(
    os.environ.get("D3_AMBIENT_OFFSET_SECONDS", _RENDER_JL_OFFSET))
# How many times stronger d(B,C) must be than the two candidate boundaries to conclude that the audio cut sits on the picture cut
D3_AMBIENT_DOMINANCE = float(os.environ.get("D3_AMBIENT_DOMINANCE", "1.5"))
# A straight cut is one complete replacement, so d(B,C) should be of the same magnitude as the two
# shots' own ambience difference d(A,D) (ideally equal; 0.85 leaves room for overlap-block estimation
# noise). J/L only add or remove one source, so d(B,C) is clearly smaller than d(A,D) (0.26~0.70 measured).
D3_AMBIENT_FULL_REPLACE_RATIO = float(os.environ.get("D3_AMBIENT_FULL_REPLACE_RATIO", "0.85"))
# When the two shots' ambience is virtually identical (digital silence / one noise floor throughout), draw no conclusion and hand back to VLM / fallback
D3_AMBIENT_MIN_CONTRAST = float(os.environ.get("D3_AMBIENT_MIN_CONTRAST", "0.01"))
# Minimum speech duration (seconds) to accept "this shot has a voice"; Whisper hallucinations are usually very short
D3_AMBIENT_SPEECH_MIN_DURATION = float(
    os.environ.get("D3_AMBIENT_SPEECH_MIN_DURATION", "0.35"))
# Whisper hallucinates reliably on pure ambience: word-level cases ('home'/'glowing' observed, word
# probability 0.0~0.02) and whole-file gibberish segments (confidence 0.54, word probability 0.27~0.32).
# Such fake speech does two bad things at once: it masks the ambience channel and it drives a wrong
# J/L attribution. So D3 keeps only TRUSTED speech: segment confidence above threshold AND mean word probability above threshold (real dialogue is usually 0.6~0.9).
D3_SPEECH_MIN_WORD_PROB = float(os.environ.get("D3_SPEECH_MIN_WORD_PROB", "0.5"))
D3_SPEECH_MIN_SEGMENT_CONFIDENCE = float(
    os.environ.get("D3_SPEECH_MIN_SEGMENT_CONFIDENCE", "0.5"))
_AMBIENT_SR = 22050
_AMBIENT_HOP = 256
_AMBIENT_N_MELS = 48


def _ambient_features(y: np.ndarray, sr: int):
    """Loudness curve and spectral shape curve in the mel domain.

    level uses the dB of the total energy (not the mean of per-band dB), otherwise quiet bands would
    flatten a real several-dB step; shape drops overall loudness and keeps only the frequency distribution, to judge "is this the same source".
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

    Returns None when either block is truly silent (its loudness-free shape vector is near zero): the
    cosine is undefined there and forcing it would yield 1.0, which looks like a huge jump and turns digital silence into strong evidence.
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

    anchor_time: the blocking anchor, by default the detected picture cut cut_time. The stitcher anchors
    the audio on the planned cumulative net duration, but blocking on the planned cut measured worse
    (8/10 vs 10/10 on three pilot cases): the perceived J/L offset is really the audio boundary
    relative to the cut the viewer sees. The parameter is kept so the anchor can be switched while debugging.
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
        # the two shots' ambience is indistinguishable (common when the generator returns near digital silence), so any attribution is noise
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


def _build_d3_signal_gate(env_y, env_sr, vocals_y, vocals_sr,
                          segments: List[dict], words: List[dict], cut_time: float,
                          prev_shot_range=None, next_shot_range=None,
                          object_sound_runs=None, object_sound_windows=None,
                          ambient_curves=None, ambient_anchor_time=None) -> dict:
    """First-stage signal diagnosis: separate synchronous change / continuous background / candidate J-L."""
    cont = (_analyze_env_audio_continuity(env_y, env_sr, cut_time, window=0.3, gap=0.3)
            if env_y is not None else {})
    sim_ba = float(cont.get("sim_before_after", 0.0) or 0.0)
    sim_eb = float(cont.get("sim_earlier_before", 0.0) or 0.0)
    sim_ea = float(cont.get("sim_earlier_after", 0.0) or 0.0)
    energy_ratio = float(cont.get("energy_ratio", 0.0) or 0.0)
    mfcc_sync = clip01(1.0 - sim_ba) if sim_ba > 0 else 0.0
    energy_sync = clip01(1.0 - energy_ratio) if energy_ratio > 0 else 0.0
    sync_change_score = max(mfcc_sync, energy_sync)
    ambient_continuity = bool(sim_ba > 0.85 and energy_ratio > 0.45)

    onset = (_detect_onset_near_cut(env_y, env_sr, cut_time) if env_y is not None
             else {"onset_near_cut": False, "onset_offset": None, "onset_strength": 0.0})
    speech = _analyze_speech_overlap(segments, words, vocals_y, vocals_sr, cut_time,
                                     prev_shot_range=prev_shot_range,
                                     next_shot_range=next_shot_range)
    object_sound = _analyze_object_sound_projection(
        object_sound_runs or [], object_sound_windows or [], cut_time,
        prev_shot_range=prev_shot_range, next_shot_range=next_shot_range)

    # Ambience channel: needed as soon as either shot lacks trusted speech. It is computed
    # unconditionally (four block spectral distances, negligible cost) and the arbitration above decides
    # whether to use it: speech projection wins when it yields an attribution, ambience otherwise. "Both
    # shots have speech" is not a hard mask, because Whisper may still let one hallucination through and a hard mask would kill the whole channel.
    prev_has_speech = _shot_has_speech(segments, prev_shot_range)
    next_has_speech = _shot_has_speech(segments, next_shot_range)
    a_times, a_level, a_shape = (ambient_curves or (None, None, None))
    ambient = _analyze_ambient_projection(a_times, a_level, a_shape, cut_time,
                                          anchor_time=ambient_anchor_time)
    ambient["ambient_prev_shot_has_speech"] = prev_has_speech
    ambient["ambient_next_shot_has_speech"] = next_has_speech
    ambient["ambient_speech_gated"] = bool(prev_has_speech and next_has_speech)

    onset_offset = onset.get("onset_offset")
    onset_sync_within_0p15 = bool(onset.get("onset_near_cut") and onset_offset is not None
                                  and abs(float(onset_offset)) <= D3_SYNC_STRAIGHT_THRESHOLD)
    speech_sync_within_0p15 = bool(speech.get("speech_boundary_sync_within_0p15"))
    av_sync_within_0p15 = speech_sync_within_0p15
    owned_event_candidate = bool(onset.get("onset_near_cut") or speech.get("speech_overlap")
                                 or object_sound.get("object_sound_crosses_cut"))
    clear_sync_change = bool(av_sync_within_0p15 or
                             (sync_change_score >= 0.45 and not speech.get("speech_overlap")))
    ambient_continuity_only = bool(ambient_continuity and not owned_event_candidate
                                   and sync_change_score < 0.35)

    if clear_sync_change:
        overlap_type, overlap_detected = "none_or_sync", False
        overlap_confidence = sync_change_score
    elif owned_event_candidate:
        overlap_type, overlap_detected = "candidate_jl", True
        overlap_confidence = max(float(speech.get("speech_overlap_confidence", 0.0)),
                                 float(onset.get("onset_strength", 0.0)) / 3.0,
                                 0.75 if object_sound.get("object_sound_crosses_cut") else 0.0)
    elif ambient_continuity_only:
        overlap_type, overlap_detected = "ambient_continuity_only", False
        overlap_confidence = sim_ba
    else:
        overlap_type, overlap_detected = "unclear", False
        overlap_confidence = max(sync_change_score, sim_ba * 0.3)

    return {
        "overlap_type": overlap_type,
        "overlap_detected": bool(overlap_detected),
        "overlap_confidence": round(clip01(overlap_confidence), 4),
        "clear_sync_change": clear_sync_change,
        "av_sync_within_0p15": av_sync_within_0p15,
        "onset_sync_within_0p15": onset_sync_within_0p15,
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


_SYSTEM_D3 = """You are a strict film-sound editor.
Classify the ACTUAL perceived audio-visual relation at a visual cut from the provided video clip.
Do not assume the written prompt is correct. Prioritize what is audible and visible in the clip."""


def _build_d3_signal_text_for_vlm(signal_gate: dict) -> str:
    """Structured signals fed to the VLM; only unclear-class overlaps are normalised to straight."""
    signal_for_vlm = {
        k: signal_gate.get(k) for k in (
            "overlap_type", "sync_change_score", "av_sync_within_0p15",
            "speech_boundary_sync_within_0p15", "speech_word_gap_split", "speech_word_gap",
            "speech_sentence_end_offset", "speech_projection_crosses_cut",
            "speech_projection_mode", "speech_projection_recommended_owner", "speech_projection",
            "object_sound_crosses_cut", "object_sound_projection_mode",
            "object_sound_recommended_owner", "object_sound_recommended_relation",
            "object_sound_projection", "object_sound_windows_near_cut",
            "ambient_available", "ambient_recommended_relation", "ambient_boundary_offset",
            "ambient_reason", "ambient_cut_dominance", "ambient_full_replace_ratio",
            "ambient_prev_shot_has_speech", "ambient_next_shot_has_speech",
            "ambient_speech_gated",
            "speech_overlap", "ambient_continuity_only", "onset_near_cut", "onset_offset",
        )
    }
    applied, reasons = False, []
    if signal_gate.get("overlap_type") in {"overlap-unclear", "unclear"}:
        applied = True
        reasons.append(f"overlap_type={signal_gate.get('overlap_type')}")
        signal_for_vlm["overlap_type"] = "straight"
    if signal_gate.get("speech_projection_mode") == "cross_both_full":
        applied = True
        reasons.append("speech_projection_mode=cross_both_full")
        signal_for_vlm["speech_projection_mode"] = "treated_as_straight_overlap"
        signal_for_vlm["speech_projection_recommended_owner"] = "none"
    if normalize_d3_relation(signal_gate.get("object_sound_recommended_relation", "")) == "overlap-unclear":
        applied = True
        reasons.append("object_sound_recommended_relation=overlap-unclear")
        signal_for_vlm["object_sound_recommended_relation"] = "straight"
        signal_for_vlm["object_sound_recommended_owner"] = "none"
    signal_for_vlm["overlap_policy_applied"] = applied
    signal_for_vlm["overlap_policy_relation"] = "straight" if applied else None
    signal_for_vlm["overlap_policy_reason"] = "; ".join(reasons)
    return json.dumps(signal_for_vlm, ensure_ascii=False)


def _extract_d3_transition_clip(video_path: str, cut_time: float, tmp_dir: str,
                                transition_idx: int, pre_sec: float = 1.2,
                                post_sec: float = 1.2) -> str:
    video_dur = get_video_duration(video_path)
    start_sec = max(0.0, float(cut_time) - pre_sec)
    end_sec = float(cut_time) + post_sec
    if video_dur > 0:
        end_sec = min(video_dur, end_sec)
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0.2:
        raise ValueError(f"invalid D3 transition clip duration: {start_sec}-{end_sec}")
    out_path = os.path.join(tmp_dir, f"d3_transition_{transition_idx:03d}.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", video_path, "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "96k", "-ac", "1", out_path],
        capture_output=True, timeout=120)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg D3 clip extraction failed: "
                           f"{r.stderr.decode('utf-8', errors='ignore')[:300]}")
    return out_path


def _vlm_predict_d3_relation(video_path: str, cut_time: float, prev_prompt: str,
                             next_prompt: str, signal_gate: dict, transition_idx: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="agent_d3_vlm_") as tmp_dir:
        clip_path = _extract_d3_transition_clip(video_path, cut_time, tmp_dir, transition_idx)
        signal_text = _build_d3_signal_text_for_vlm(signal_gate)
        user_text = f"""
The clip is centered on one visual cut. The visual cut occurs at approximately 1.20 seconds after this clip starts.

Previous/outgoing shot description:
{prev_prompt[:1200]}

Next/incoming shot description:
{next_prompt[:1200]}

Signal pre-check, for reference only:
{signal_text}

Classify the ACTUAL perceived audio-visual relation at the cut into exactly one of:
- straight: image and dominant sound transition together at the cut, with no shot-owned sound from the outgoing shot continuing into the incoming shot and no shot-owned sound from the incoming shot heard before its image.
- j-cut: a sound clearly belonging to the incoming/next shot is heard before its image appears.
- l-cut: a sound clearly belonging to the outgoing/previous shot continues after its image disappears.
- overlap-unclear: sound overlaps the cut, but ownership/timing is ambiguous.
- unclear: evidence is insufficient.

Important rules:
- If overlap_policy_applied is true and overlap_policy_relation is straight, treat that unclear overlap as straight (dissolve/crossfade may also dissolve the audio). Do not apply this policy to explicit J/L evidence or candidate_jl cues.
- If a speech segment split by word gaps ends within 0.15s of the cut, prefer straight unless stronger shot-owned continuation evidence exists.
- If speech_projection_mode is cross_partial and speech_projection_recommended_owner is previous, default to L-cut; if it is next, default to J-cut. Override only when highly confident the owner shot has no plausible source.
- If speech_projection_mode is cross_both_full, decide ownership from the visible speaking subject, shot scale, lip/face presence and perceived loudness.
- If neither shot has speech (ambient_prev_shot_has_speech / ambient_next_shot_has_speech are false) and ambient_available is true, the ambient-track boundary is measured evidence: ambient_boundary_offset < 0 means the incoming shot's ambience is already audible before the cut (J-cut), > 0 means the outgoing shot's ambience continues past the cut (L-cut), and 0 means the ambience is fully replaced exactly at the cut (straight). Override it only with clearly audible contrary evidence.
- Classify continuous sound as J/L only when it is a salient subject or clearly attributable source sound (speech, singing, instrument, action sound, visible object/machine). Generic ambience, room tone, wind, water or low-level bed noise can remain straight even when continuous.
- Natural decay/tail of a shot-owned sound continuing into the next shot still counts as L-cut.
- If continuous sound has an obvious pause/beat/chord attack exactly at the cut, classify straight.
- Do NOT infer from the written intended relation; judge the actual clip.

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
        response = call_vlm(_SYSTEM_D3, user_text, clip_path, max_tokens=900, temperature=0.0)

    parsed = _extract_json_from_response(response)
    if not parsed:
        return {"used": True, "predicted_relation": "unclear",
                "overlap_detected": signal_gate.get("overlap_detected", False),
                "sound_owner": "unclear", "dominant_sound_event": "",
                "timing_observation": "", "confidence": 0.0, "reason": "parse_failed",
                "raw_response": (response or "")[:500]}

    relation = normalize_d3_relation(parsed.get("predicted_relation", ""))
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
        "used": True, "predicted_relation": relation,
        "overlap_detected": bool(parsed.get("overlap_detected",
                                            relation in {"j-cut", "l-cut", "overlap-unclear"})),
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


def eval_d3_audio_visual_relation(plan: Dict[str, Any], video_path: str,
                                  alignment: Dict[str, Any],
                                  whisper_result: dict, demucs_result: dict,
                                  use_vlm: bool = True) -> dict:
    """D3: transition audio-visual relation. Prediction goes signal arbitration > VLM > signal fallback, never reading GT."""
    import librosa

    shots = plan["shots"]
    if len(shots) < 2:
        return {"dimension": "D3", "metric": "transition audio-visual relation", "score": 0.5,
                "note": "less than 2 shots", "per_transition": []}

    segments = (whisper_result or {}).get("segments", []) or []
    words = (whisper_result or {}).get("words", []) or []
    segments, words, speech_filter_stats = _filter_reliable_speech(segments, words)
    if speech_filter_stats["n_segments_in"] != speech_filter_stats["n_segments_kept"]:
        print(f"  [D3] filtered unreliable speech segments "
              f"{speech_filter_stats['n_segments_in']} -> {speech_filter_stats['n_segments_kept']}"
              f" (hallucinated segments: "
              f"{[d['text'] for d in speech_filter_stats['dropped_segments']]})")

    env_y = vocals_y = None
    env_sr = vocals_sr = int((demucs_result or {}).get("sample_rate", 44100) or 44100)
    source_files = (demucs_result or {}).get("source_files", {}) or {}

    env_tracks: List[np.ndarray] = []
    object_track_audio: Dict[str, np.ndarray] = {}
    object_track_sr: Dict[str, int] = {}
    object_panns_by_track: Dict[str, dict] = {}
    for track_name in ("drums", "bass", "other"):
        track_path = source_files.get(track_name, "")
        if not track_path or not os.path.exists(track_path):
            continue
        try:
            y_track, env_sr = librosa.load(track_path, sr=env_sr, mono=True)
        except Exception:  # noqa: BLE001
            continue
        env_tracks.append(y_track)
        object_track_audio[track_name] = y_track
        object_track_sr[track_name] = env_sr
        try:
            object_panns_by_track[track_name] = call_panns(
                track_path, segment_duration=D3_OBJECT_SEGMENT_DURATION)
        except Exception as e:  # noqa: BLE001
            object_panns_by_track[track_name] = {"success": False, "error": str(e)[:200]}

    if env_tracks:
        min_len = min(len(t) for t in env_tracks)
        env_y = np.sum([t[:min_len] for t in env_tracks], axis=0)

    vocals_path = source_files.get("vocals", "")
    if vocals_path and os.path.exists(vocals_path):
        try:
            vocals_y, vocals_sr = librosa.load(vocals_path, sr=vocals_sr, mono=True)
        except Exception:  # noqa: BLE001
            vocals_y = None

    object_sound_windows = _build_object_sound_windows(
        object_panns_by_track, object_track_audio, object_track_sr)
    object_sound_runs = _merge_object_sound_runs(object_sound_windows)

    # The ambience channel uses the final cut's full mix directly: demucs stem separation redistributes
    # each clip's noise floor across stems and measurably weakens the only usable structural feature, the two shots' ambience overlap (9/10 vs 10/10 at equal settings).
    ambient_curves = (None, None, None)
    if video_path:
        try:
            mix_y, mix_sr = _load_audio_mono(video_path, sr=_AMBIENT_SR)
            ambient_curves = _ambient_features(mix_y, mix_sr)
        except Exception as e:  # noqa: BLE001
            print(f"  [D3] failed to extract the ambience curve, this channel is disabled: {str(e)[:200]}")

    aligned_transitions = alignment.get("transitions", [])
    aligned_shots = alignment.get("aligned_shots", [])

    scores: List[float] = []
    details: List[Dict[str, Any]] = []
    for i, shot in enumerate(shots[:-1]):
        gt_relation = shot.get("audio_visual_relation") or ""
        gt_relation_norm = normalize_d3_relation(gt_relation)
        at = aligned_transitions[i] if i < len(aligned_transitions) else {}

        if not at.get("matched"):
            # the cut was not detected -> this transition cannot be evaluated, scored 0 (same rule as benchmark's missing shots)
            scores.append(0.0)
            details.append({
                "index": i, "cut_time": None, "gt_relation": gt_relation,
                "gt_relation_normalized": gt_relation_norm,
                "predicted_relation": "unclear", "matches_gt": False, "score": 0.0,
                "reason": "cut_not_detected",
                "rendered_timing_offset_seconds": shot.get("timing_offset_seconds"),
            })
            continue

        cut_time = float(at["cut_time"])
        prev_range = tuple(aligned_shots[i]["time_range"]) if i < len(aligned_shots) else None
        next_range = tuple(aligned_shots[i + 1]["time_range"]) if i + 1 < len(aligned_shots) else None

        signal_gate = _build_d3_signal_gate(
            env_y=env_y, env_sr=env_sr, vocals_y=vocals_y, vocals_sr=vocals_sr,
            segments=segments, words=words, cut_time=cut_time,
            prev_shot_range=prev_range, next_shot_range=next_range,
            object_sound_runs=object_sound_runs, object_sound_windows=object_sound_windows,
            ambient_curves=ambient_curves)

        vlm_relation = None
        if use_vlm and video_path:
            try:
                vlm_relation = _vlm_predict_d3_relation(
                    video_path, cut_time, shot.get("prompt", ""),
                    shots[i + 1].get("prompt", ""), signal_gate, i)
                print(f"  [D3] transition {i}: {vlm_relation.get('predicted_relation')} "
                      f"conf={vlm_relation.get('confidence')} gate={signal_gate.get('overlap_type')}")
            except Exception as e:  # noqa: BLE001
                vlm_relation = {"used": True, "predicted_relation": "unclear",
                                "overlap_detected": signal_gate.get("overlap_detected", False),
                                "sound_owner": "unclear", "dominant_sound_event": "",
                                "timing_observation": "", "confidence": 0.0,
                                "reason": f"vlm_failed: {str(e)[:200]}"}

        fallback_relation = _fallback_d3_relation_from_signal(signal_gate)

        # Generic signal arbitration (never reads GT), from strongest to weakest source attribution:
        #   1) word-level speech projection (cross_partial) -> the attributed side decides J/L
        #   2) ambience audio cut (usable only when at least one shot has no speech) -> its position decides J/L/straight
        #   3) a prominent object sound continuing long -> follow the object sound projection
        #   4) synchronous change + very short / no natural tail -> straight
        # The ambience channel must come before "synchronous change -> straight": the latter is almost
        # always true on speechless material and would judge every transition straight (exactly why D3 used to score so low).
        arbitration_relation = arbitration_reason = None
        object_projection = signal_gate.get("object_sound_projection") or {}
        object_next_overlap = float(object_projection.get("next_overlap_duration", 0.0) or 0.0)
        object_relation = normalize_d3_relation(
            signal_gate.get("object_sound_recommended_relation", ""))
        ambient_relation = normalize_d3_relation(
            signal_gate.get("ambient_recommended_relation", ""))
        if (signal_gate.get("speech_projection_crosses_cut") and
                signal_gate.get("speech_projection_mode") == "cross_partial"):
            owner = signal_gate.get("speech_projection_recommended_owner")
            if owner == "next":
                arbitration_relation, arbitration_reason = "j-cut", "word_timestamp_speech_projection_next"
            elif owner == "previous":
                arbitration_relation, arbitration_reason = "l-cut", "word_timestamp_speech_projection_previous"
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
            vlm_pred = normalize_d3_relation(vlm_relation.get("predicted_relation", ""))
            if vlm_pred != "unclear":
                predicted_relation = vlm_pred
            elif fallback_relation != "unclear":
                predicted_relation = fallback_relation
            else:
                predicted_relation = vlm_pred

        matches_gt = bool(gt_relation_norm != "unclear" and predicted_relation == gt_relation_norm)
        score = 1.0 if matches_gt else 0.0
        scores.append(score)

        detail = {
            "index": i,
            "cut_time": round(cut_time, 3),
            "gt_relation": gt_relation,
            "gt_relation_normalized": gt_relation_norm,
            "predicted_relation": predicted_relation,
            "matches_gt": matches_gt,
            "score": score,
            "rendered_timing_offset_seconds": shot.get("timing_offset_seconds"),
            "relation_arbitration": {
                "applied": arbitration_relation is not None,
                "relation": arbitration_relation,
                "reason": arbitration_reason,
                "object_next_overlap_duration": round(object_next_overlap, 4),
                "object_sync_tail_threshold": D3_OBJECT_SYNC_TAIL_THRESHOLD,
                "ambient_relation": ambient_relation,
                "ambient_available": bool(signal_gate.get("ambient_available")),
            },
            "signal_summary": {
                "overlap_type": signal_gate.get("overlap_type"),
                "clear_sync_change": signal_gate.get("clear_sync_change"),
                "sync_change_score": signal_gate.get("sync_change_score"),
                "speech_projection_mode": signal_gate.get("speech_projection_mode"),
                "speech_projection_recommended_owner": signal_gate.get("speech_projection_recommended_owner"),
                "ambient_available": signal_gate.get("ambient_available"),
                "ambient_recommended_relation": signal_gate.get("ambient_recommended_relation"),
                "ambient_boundary_offset": signal_gate.get("ambient_boundary_offset"),
                "ambient_reason": signal_gate.get("ambient_reason"),
                "ambient_speech_gated": signal_gate.get("ambient_speech_gated"),
                "ambient_cut_dominance": signal_gate.get("ambient_cut_dominance"),
                "ambient_full_replace_ratio": signal_gate.get("ambient_full_replace_ratio"),
                "object_sound_crosses_cut": signal_gate.get("object_sound_crosses_cut"),
                "object_sound_recommended_relation": signal_gate.get("object_sound_recommended_relation"),
                "onset_offset": signal_gate.get("onset_offset"),
                "env_sim_before_after": signal_gate.get("env_sim_before_after"),
                "env_energy_ratio": signal_gate.get("env_energy_ratio"),
            },
            "signal_gate": signal_gate,
        }
        if vlm_relation:
            detail["vlm_relation"] = vlm_relation
        details.append(detail)

    if not scores:
        return {"dimension": "D3", "metric": "transition audio-visual relation", "score": 0.5, "per_transition": []}

    return {
        "dimension": "D3",
        "metric": "transition audio-visual relation",
        "score": clip01(float(np.mean(scores))),
        "prediction_mode": "actual_video_first",
        "speech_filter": speech_filter_stats,
        "per_transition": details,
    }


# ============================================================
#  7. Diagnostics: head/tail sound check of a shot clip (for the central planner's D3 repairs)
# ============================================================

def _load_audio_mono(path: str, sr: int = 22050):
    """Load mono audio.

    librosa/soundfile cannot read an mp4 container and fall back to the very slow audioread (tens of
    seconds per clip). Decoding to a temporary wav with ffmpeg first is one or two orders faster and free of deprecation warnings.
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


def inspect_clip_audio(clip_path: str, head_window: float = 1.0,
                       tail_window: float = 1.0,
                       silence_dbfs: float = -45.0,
                       run_whisper: bool = False) -> dict:
    """Check a single shot clip's head/tail sound, to locate the \"no sound at the start or end\" case.

    A J-cut needs sound at the START of the incoming clip (so it can be heard early),
    an L-cut needs sound at the END of the outgoing clip (so it can continue into the next shot).
    Edge silence is the most common physical cause of a D3 failure and must be measured before deciding between a prompt change and an offset change.
    """
    report: Dict[str, Any] = {"clip": clip_path, "exists": os.path.exists(clip_path)}
    if not report["exists"]:
        report["error"] = "clip not found"
        return report

    duration = get_video_duration(clip_path)
    report["duration"] = round(duration, 3)
    try:
        y, sr = _load_audio_mono(clip_path, sr=22050)
    except Exception as e:  # noqa: BLE001
        report["error"] = f"audio load failed: {str(e)[:200]}"
        return report

    if y is None or len(y) == 0:
        report.update({"has_audio": False, "note": "no audio stream / empty audio"})
        return report

    total_sec = len(y) / sr
    head = _slice_audio_window(y, sr, 0.0, min(head_window, total_sec))
    tail = _slice_audio_window(y, sr, max(0.0, total_sec - tail_window), total_sec)
    head_dbfs = _dbfs_from_rms(_rms_energy(head))
    tail_dbfs = _dbfs_from_rms(_rms_energy(tail))
    full_dbfs = _dbfs_from_rms(_rms_energy(y))

    report.update({
        "has_audio": True,
        "audio_duration": round(total_sec, 3),
        "full_dbfs": round(full_dbfs, 2),
        "head_window_seconds": round(min(head_window, total_sec), 3),
        "tail_window_seconds": round(min(tail_window, total_sec), 3),
        "head_dbfs": round(head_dbfs, 2),
        "tail_dbfs": round(tail_dbfs, 2),
        "head_silent": bool(head_dbfs <= silence_dbfs),
        "tail_silent": bool(tail_dbfs <= silence_dbfs),
        "silence_threshold_dbfs": silence_dbfs,
    })

    # Energy curve every 0.25s, which shows the moment the sound actually comes up
    step = 0.25
    envelope = []
    t = 0.0
    while t < total_sec - 1e-6:
        seg = _slice_audio_window(y, sr, t, min(t + step, total_sec))
        envelope.append({"start": round(t, 3), "dbfs": round(_dbfs_from_rms(_rms_energy(seg)), 2)})
        t += step
    report["energy_envelope_0p25s"] = envelope

    first_voiced = next((e["start"] for e in envelope if e["dbfs"] > silence_dbfs), None)
    last_voiced = next((e["start"] for e in reversed(envelope) if e["dbfs"] > silence_dbfs), None)
    report["first_voiced_time"] = first_voiced
    report["last_voiced_time"] = last_voiced

    if run_whisper:
        try:
            w = call_whisper(clip_path, word_timestamps=True)
            report["speech"] = {
                "text": (w.get("text") or "")[:400],
                "n_segments": len(w.get("segments", []) or []),
                "segments": [{"start": s.get("start"), "end": s.get("end"),
                              "text": str(s.get("text", ""))[:120]}
                             for s in (w.get("segments") or [])[:8]],
            }
        except Exception as e:  # noqa: BLE001
            report["speech"] = {"error": str(e)[:200]}

    return report


def inspect_transition_audio(prev_clip: str, next_clip: str,
                             timing_offset_seconds: float,
                             audio_relation: str) -> dict:
    """For one transition, check both the outgoing and incoming clips and return an actionable conclusion."""
    relation = normalize_d3_relation(audio_relation)
    prev_report = inspect_clip_audio(prev_clip)
    next_report = inspect_clip_audio(next_clip)

    findings: List[str] = []
    if relation == "l-cut":
        if prev_report.get("tail_silent"):
            findings.append("outgoing clip tail is silent: the L-cut has no sound to carry over, the prompt must "
                            "require the tail of that shot to keep a sustained on-screen sound (trailing speech / object resonance / footsteps)")
        if next_report.get("head_dbfs") is not None and prev_report.get("tail_dbfs") is not None:
            if float(next_report["head_dbfs"]) - float(prev_report["tail_dbfs"]) > 12:
                findings.append("incoming clip starts noticeably louder and will mask the L-cut tail sound, "
                                "consider increasing timing_offset or lowering the described sound intensity at the incoming head")
    elif relation == "j-cut":
        if next_report.get("head_silent"):
            findings.append("incoming clip head is silent: the J-cut has no sound to bring forward, the prompt must "
                            "require an explicit on-screen sound right at the start of that shot (opening line of speech / an object trigger sound)")
        first_voiced = next_report.get("first_voiced_time")
        if first_voiced is not None and float(first_voiced) >= float(timing_offset_seconds or 0):
            findings.append(f"incoming clip sound only starts at {first_voiced}s, "
                            f"the current offset {timing_offset_seconds}s is not enough for it to cross the cut point, "
                            f"consider raising the offset to at least {round(float(first_voiced) + 0.4, 2)}s")
    else:
        if prev_report.get("tail_silent") and next_report.get("head_silent"):
            findings.append("both sides of the cut point are silent: there is not enough synchronisation evidence for straight, "
                            "consider scheduling an explicit sound event near the cut point in the prompt")

    return {
        "audio_relation": relation,
        "timing_offset_seconds": timing_offset_seconds,
        "outgoing_clip": prev_report,
        "incoming_clip": next_report,
        "findings": findings,
    }


# ============================================================
#  8. Evaluation orchestration entry point
# ============================================================

DEFAULT_THRESHOLDS = {"B1": 0.90, "D2": 0.70, "D3": 0.50}

DIMENSION_SERVICES = {
    "B1": ["transnetv2"],
    "D2": ["transnetv2"],
    "D3": ["transnetv2", "whisper", "demucs", "panns"],
}


def evaluate_video(video_path: str, decisions: List[Dict[str, Any]],
                   dimensions: Optional[List[str]] = None,
                   use_vlm: bool = True,
                   thresholds: Optional[Dict[str, float]] = None,
                   transnetv2_threshold: float = 0.35) -> Dict[str, Any]:
    """Run B1 / D2 / D3 on one final cut; returns the scores + per-transition detail + failure list."""
    dims = [d.upper() for d in (dimensions or ["B1", "D2", "D3"])]
    thr = dict(DEFAULT_THRESHOLDS)
    thr.update(thresholds or {})

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"the composed video does not exist: {video_path}")

    needed: List[str] = []
    for d in dims:
        for svc in DIMENSION_SERVICES.get(d, []):
            if svc not in needed:
                needed.append(svc)
    require_services(needed)

    plan = build_plan_from_decisions(decisions)
    video_duration = get_video_duration(video_path)

    print(f"  [Eval] TransNetV2 cut point detection (port {SERVICE_PORTS['transnetv2']}) ...")
    transnetv2_result = call_transnetv2(video_path, threshold=transnetv2_threshold,
                                       expected_shots=len(plan["shots"]))
    alignment = align_cuts_to_plan(transnetv2_result, plan, video_duration)
    n_unmatched = sum(1 for t in alignment["transitions"] if not t["matched"])
    print(f"  [Eval] planned {len(plan['shots'])} shots / detected {alignment['n_detected_shots']} shots, "
          f"unmatched cut points {n_unmatched}/{len(alignment['transitions'])}")

    results: Dict[str, Any] = {
        "video": video_path,
        "video_duration": round(video_duration, 3),
        "n_planned_shots": len(plan["shots"]),
        "n_detected_shots": alignment["n_detected_shots"],
        "thresholds": thr,
        "alignment": alignment,
        "dimensions": {},
    }

    if "B1" in dims:
        results["dimensions"]["B1"] = eval_b1_shot_duration(plan, alignment)
    if "D2" in dims:
        results["dimensions"]["D2"] = eval_d2_transition_type(plan, transnetv2_result, alignment)
    if "D3" in dims:
        print(f"  [Eval] Whisper (port {SERVICE_PORTS['whisper']}) / "
              f"Demucs (port {SERVICE_PORTS['demucs']}) / PANNs (port {SERVICE_PORTS['panns']}) ...")
        whisper_result = call_whisper(video_path, word_timestamps=True)
        demucs_result = call_demucs(video_path)
        results["dimensions"]["D3"] = eval_d3_audio_visual_relation(
            plan, video_path, alignment, whisper_result, demucs_result, use_vlm=use_vlm)

    scores = {d: round(float(r.get("score", 0.0)), 4)
              for d, r in results["dimensions"].items()}
    results["scores"] = scores
    results["passed"] = {d: bool(scores.get(d, 0.0) >= thr.get(d, 0.0)) for d in scores}
    results["failed_dimensions"] = [d for d, ok in results["passed"].items() if not ok]
    results["failed_transitions"] = collect_failed_transitions(results)
    return results


def collect_failed_transitions(eval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge the per-transition failures into a list the central planner can act on directly."""
    dims = eval_result.get("dimensions", {})
    by_index: Dict[int, Dict[str, Any]] = {}

    def slot(idx: int) -> Dict[str, Any]:
        return by_index.setdefault(int(idx), {"transition_index": int(idx), "issues": []})

    b1 = dims.get("B1") or {}
    if b1 and float(b1.get("score", 1.0)) < eval_result["thresholds"].get("B1", 0.9):
        # B1 attribution must land on CUTS, not shots: a shot's duration error is decided jointly by the
        # drift of the two cuts around it, and fixing shots independently would correct a shared cut twice and overshoot.
        # So one issue is emitted per cut whose drift exceeds the threshold, with both neighbouring shot durations as context.
        per_shot = {int(s["shot_idx"]): s for s in b1.get("per_shot", [])}
        for t in b1.get("per_transition", []):
            idx = int(t["index"])
            offset = t.get("cut_offset_seconds")
            if not t.get("matched") or offset is None:
                # cut not detected: the timing cannot be nudged, so leave it to D2 / the generation side and only record it
                slot(idx)["issues"].append({
                    "dimension": "B1",
                    "cut_matched": False,
                    "planned_cut": t.get("planned_cut"),
                    "note": "this planned cut point was not detected, it cannot be fixed by tuning the cut timing",
                })
                continue
            if abs(float(offset)) < 0.08:
                continue
            outgoing = per_shot.get(idx, {})
            incoming = per_shot.get(idx + 1, {})
            slot(idx)["issues"].append({
                "dimension": "B1",
                "cut_matched": True,
                "planned_cut": t.get("planned_cut"),
                "cut_time": t.get("cut_time"),
                # positive = cut too late (should move earlier), negative = cut too early
                "cut_offset_seconds": round(float(offset), 4),
                "suggested_delta_seconds": round(-float(offset), 4),
                "outgoing_shot_id": outgoing.get("shot_id"),
                "outgoing_gt_duration": outgoing.get("gt_duration"),
                "outgoing_pred_duration": outgoing.get("pred_duration"),
                "incoming_shot_id": incoming.get("shot_id"),
                "incoming_gt_duration": incoming.get("gt_duration"),
                "incoming_pred_duration": incoming.get("pred_duration"),
            })

    d2 = dims.get("D2") or {}
    if d2 and float(d2.get("score", 1.0)) < eval_result["thresholds"].get("D2", 0.7):
        for t in d2.get("per_transition", []):
            if float(t.get("score", 1.0)) >= 1.0:
                continue
            slot(t["index"])["issues"].append({
                "dimension": "D2",
                "gt_type": t.get("gt_type"),
                "pred_type": t.get("pred_type"),
                "score": t.get("score"),
                "rendered_effect_seconds": t.get("rendered_effect_seconds"),
                "pred_detected_span_seconds": t.get("pred_detected_span_seconds"),
                "note": t.get("note"),
            })

    d3 = dims.get("D3") or {}
    if d3 and float(d3.get("score", 1.0)) < eval_result["thresholds"].get("D3", 0.5):
        for t in d3.get("per_transition", []):
            if t.get("matches_gt"):
                continue
            slot(t["index"])["issues"].append({
                "dimension": "D3",
                "gt_relation": t.get("gt_relation_normalized"),
                "predicted_relation": t.get("predicted_relation"),
                "rendered_timing_offset_seconds": t.get("rendered_timing_offset_seconds"),
                "reason": t.get("reason") or (t.get("relation_arbitration") or {}).get("reason"),
                "signal_summary": t.get("signal_summary"),
            })

    return [by_index[k] for k in sorted(by_index) if by_index[k]["issues"]]


def load_decisions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("decisions", [])
    return data


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Edit agent self-evaluation: B1 transition timing / D2 transition effect / D3 transition audio-visual relation")
    ap.add_argument("--video", required=True, help="path to the composed video")
    ap.add_argument("--decisions", required=True, help="path to edit_decisions.json")
    ap.add_argument("--dimensions", default="B1,D2,D3", help="comma separated, e.g. B1,D2")
    ap.add_argument("--output", default=None, help="output path for the evaluation result json")
    ap.add_argument("--no-vlm", action="store_true", help="skip the VLM for D3, use signal arbitration only")
    ap.add_argument("--check-services", action="store_true", help="only check the service health and exit")
    args = ap.parse_args()

    if args.check_services:
        print(f"agent self-eval service ports (offset={PORT_OFFSET}):")
        for name, health in check_services().items():
            status = health.get("status", "?")
            print(f"  {name:12s} :{SERVICE_PORTS[name]}  {status}  {health}")
        return 0

    decisions = load_decisions(args.decisions)
    if not decisions:
        print("edit_decisions has no decisions")
        return 1

    result = evaluate_video(
        video_path=args.video,
        decisions=decisions,
        dimensions=[d.strip() for d in args.dimensions.split(",") if d.strip()],
        use_vlm=not args.no_vlm,
    )

    print("\n" + "=" * 60)
    for dim, score in result["scores"].items():
        flag = "PASS" if result["passed"][dim] else "FAIL"
        print(f"  {dim}: {score:.4f}  (threshold {result['thresholds'][dim]:.2f})  [{flag}]")
    if result["failed_transitions"]:
        print("  failed transitions:")
        for ft in result["failed_transitions"]:
            dims_hit = ",".join(sorted({i["dimension"] for i in ft["issues"]}))
            print(f"    transition {ft['transition_index']}: {dims_hit}")
    print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"evaluation result written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
