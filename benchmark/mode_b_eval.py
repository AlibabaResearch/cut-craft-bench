"""
Mode B Evaluation Engine — VLM-Assisted Assessment
Expert model outputs + video clips → VLM (qwen3.5-omni-plus) judgment

Each dimension is designed as a True/False or Multiple-Choice question for accuracy statistics.
E1 dimensions are evaluated PER-SHOT using TransNetV2 segmentation.

All model-facing prompts are in English for optimal VLM performance.
"""
import os
import sys
import json
import re
import time
import base64
import subprocess
import tempfile
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

# Directory holding ffmpeg / ffprobe. The versions on PATH are used by default; to pin a dedicated
# environment, export AV_PROCESS_BIN=/path/to/env/bin before running.
AV_PROCESS_BIN = os.environ.get("AV_PROCESS_BIN", "")
if AV_PROCESS_BIN and AV_PROCESS_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = AV_PROCESS_BIN + os.pathsep + os.environ.get("PATH", "")

# Ensure benchmark/ (this file's own dir) is importable for skill_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Configuration
# ============================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
VLM_MODEL = os.getenv("VLM_MODEL")
# Expert model microservice URLs
SERVICE_URLS = {
    "transnetv2": "http://localhost:8001",
    "raft": "http://localhost:8002",
    "dinov2": "http://localhost:8003",
    "whisper": "http://localhost:8004",
    "demucs": "http://localhost:8005",
    "panns": "http://localhost:8007",
    "yolov8": "http://localhost:8009",
    "clip": "http://localhost:8010",
    "saliency": "http://localhost:8011",
    "movieshots": "http://localhost:8013",
    "sixdrepnet": "http://localhost:8014",
    "monst3r": "http://localhost:8015",
}

# ============================================================
# VLM Client
# ============================================================
client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


class VLMRetryExhausted(RuntimeError):
    """A VLM call exhausted its retries; abort this sample so a low score does not pollute the cache."""


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
    """Whether a VLM error is worth waiting for and retrying."""
    err = str(exc).lower()
    retry_keywords = [
        "429", "insufficient_quota", "rate limit", "too many requests",
        "token-limit", "timeout", "timed out", "connection", "temporarily",
        "server error", "internal error", "internal_error", "500", "502", "503", "504",
    ]
    return any(keyword in err for keyword in retry_keywords)


def _raise_if_vlm_retry_exhausted(exc: Exception) -> None:
    """When a Mode B sub-item swallows exceptions, a fatal VLM error must still propagate."""
    if isinstance(exc, VLMRetryExhausted):
        raise exc


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe"""
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
    """Ensure video file is at least min_dur seconds; loop if too short"""
    dur = _get_video_duration(video_path)
    if dur <= 0 or dur >= min_dur:
        return video_path
    # Video too short: loop it to reach min_dur
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


def compress_video_for_vlm(video_path: str, max_size_mb: float = 4.0) -> str:
    """Compress video to VLM-friendly size (base64 < 10MB)"""
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    if file_size <= max_size_mb:
        return video_path

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "scale=-2:360",
        "-b:v", "1500k",
        "-b:a", "48k",
        "-ac", "1",
        "-r", "15",
        tmp.name
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)
    return tmp.name


def video_to_base64(video_path: str) -> str:
    """Convert video to base64 string"""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(system_prompt: str, user_text: str, video_path: str = None,
             max_tokens: int = 500, temperature: float = 0.1) -> str:
    """Call Qwen3.5-omni-plus VLM
    
    Note: Qwen-Omni API does not support system role combined with video_url,
    so system_prompt is merged into user text.
    Note: Qwen-Omni does not accept a system prompt together with video, so system_prompt is merged into the user text.
    """
    combined_text = f"[ROLE]\n{system_prompt}\n\n[TASK]\n{user_text}"

    if video_path:
        # Ensure video meets VLM minimum duration requirement
        video_path = _ensure_min_duration(video_path)
        compressed = compress_video_for_vlm(video_path)
        b64 = video_to_base64(compressed)
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
            completion = client.chat.completions.create(
                model=VLM_MODEL,
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
                raise VLMRetryExhausted(f"VLM call failed after {attempt}/{max_retries} attempts: {e}") from e
            wait = min(max_wait, base_wait * (2 ** (attempt - 1)))
            print(f"  [VLM RETRY] attempt {attempt}/{max_retries} failed: {e}")
            print(f"  [VLM RETRY] waiting {wait:.1f}s before retrying...")
            time.sleep(wait)


def parse_choice(response: str, valid_choices: List[str]) -> Optional[str]:
    """Parse VLM response to extract choice"""
    # Guard: if VLM returned an error string, do not attempt to parse it
    if response.upper().strip().startswith("ERROR:"):
        return None
    response_upper = response.upper().strip()
    for choice in valid_choices:
        if response_upper.startswith(choice.upper()):
            return choice
        if f"ANSWER IS {choice.upper()}" in response_upper or f"CHOOSE {choice.upper()}" in response_upper:
            return choice
        if f"答案是{choice.upper()}" in response_upper or f"选择{choice.upper()}" in response_upper:
            return choice
    # True/False for Yes/No questions
    if "Yes" in valid_choices or "No" in valid_choices:
        if "YES" in response_upper[:20] and "NOT" not in response_upper[:20]:
            return "Yes"
        if "NO" in response_upper[:20]:
            return "No"
    if "是" in valid_choices or "否" in valid_choices:
        if "是" in response_upper[:10] and "不是" not in response_upper[:10]:
            return "是"
        if "否" in response_upper[:10] or "不是" in response_upper[:10]:
            return "否"
    return None


# ============================================================
# Expert Service Helpers
# ============================================================

# Service-down recovery wait configuration
_SERVICE_DOWN_MAX_WAIT = 90    # max seconds to wait for a service to come back after it dies
_SERVICE_DOWN_POLL_INTERVAL = 5  # polling interval while waiting for recovery (seconds)


def _is_service_down_error(e: Exception) -> bool:
    """Whether an exception means the service is completely down (process dead / port unreachable)."""
    if isinstance(e, requests.exceptions.ConnectionError):
        return True
    if isinstance(e, requests.exceptions.ConnectTimeout):
        return True
    err_str = str(e).lower()
    if any(kw in err_str for kw in ["connection refused", "connectionerror",
                                     "connect timeout", "no route to host",
                                     "remotedisconnected", "connection aborted"]):
        return True
    return False


def _wait_for_service_recovery(service: str, max_wait: int = _SERVICE_DOWN_MAX_WAIT) -> bool:
    """Wait for a service to recover (restarted by the watchdog) by polling its /health endpoint."""
    health_url = f"{SERVICE_URLS[service]}/health"
    start = time.time()
    poll_count = 0
    print(f"  [Recovery] {service} appears DOWN. "
          f"Waiting up to {max_wait}s for watchdog to restart...")
    while (time.time() - start) < max_wait:
        time.sleep(_SERVICE_DOWN_POLL_INTERVAL)
        poll_count += 1
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                elapsed = time.time() - start
                print(f"  [Recovery] {service} is BACK after {elapsed:.1f}s. Resuming...")
                time.sleep(2)  # extra settling time
                return True
        except Exception:
            pass
        if poll_count % 4 == 0:
            elapsed = time.time() - start
            print(f"  [Recovery] Still waiting for {service}... ({elapsed:.0f}s/{max_wait}s)")
    print(f"  [Recovery] {service} did NOT recover within {max_wait}s. Giving up.")
    return False


def call_service(service: str, video_path: str, endpoint: str = None,
                 extra_data: dict = None, max_retries: int = 2) -> dict:
    """Call a microservice with correct endpoint/field per service.
    Call a microservice, picking the right endpoint and field name from the service type.
    
    Retry strategy:
      Tier 1: fast retries (3 attempts, 1s / 2s apart)
      Tier 2: on a detected service outage, wait for the watchdog restart and retry
    
    Endpoint/field mapping:
      monst3r: POST /analyze, field="video"
      All others: POST /predict, field="file"
    """
    # Determine endpoint
    if endpoint is None:
        endpoint = "/analyze" if service == "monst3r" else "/predict"
    # Determine upload field name
    field_name = "video" if service == "monst3r" else "file"
    
    url = f"{SERVICE_URLS[service]}{endpoint}"
    service_was_down = False
    
    for attempt in range(max_retries + 1):
        try:
            with open(video_path, "rb") as f:
                files = {field_name: (os.path.basename(video_path), f, "video/mp4")}
                data = extra_data or {}
                resp = requests.post(url, files=files, data=data, timeout=300)
            if resp.status_code == 200:
                return resp.json()
            else:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                # Tier 2: wait for the service to recover
                if _is_service_down_error(e) and not service_was_down:
                    service_was_down = True
                    recovered = _wait_for_service_recovery(service)
                    if recovered:
                        # Retry once after the service is back
                        try:
                            with open(video_path, "rb") as f:
                                files = {field_name: (os.path.basename(video_path), f, "video/mp4")}
                                data = extra_data or {}
                                resp = requests.post(url, files=files, data=data, timeout=300)
                            if resp.status_code == 200:
                                return resp.json()
                            else:
                                raise RuntimeError(f"HTTP {resp.status_code} after recovery: {resp.text[:200]}")
                        except Exception as e2:
                            raise RuntimeError(f"Service {service} failed after recovery: {e2}")
                    else:
                        raise RuntimeError(
                            f"Service {service} is DOWN and did not recover "
                            f"within {_SERVICE_DOWN_MAX_WAIT}s: {e}")
                else:
                    raise


def call_transnetv2(video_path: str) -> dict:
    """Call TransNetV2 with correct endpoint/field (a special interface)"""
    url = f"{SERVICE_URLS['transnetv2']}/predict"
    with open(video_path, "rb") as f:
        files = {"file": (os.path.basename(video_path), f, "video/mp4")}
        resp = requests.post(url, files=files, timeout=300)
    return resp.json()


def extract_shot_clip(video_path: str, start_sec: float, end_sec: float,
                      min_duration: float = 2.0) -> str:
    """Extract a clip from video using ffmpeg

    If the clip is shorter than min_duration, the end time is extended forward.
    If the video ends before reaching min_duration, the start time is shifted back
    so the clip ends at the video's end with length = min_duration.
    If the entire video is shorter than min_duration, the full video is used.
    """
    # Ensure clip meets minimum duration
    clip_dur = end_sec - start_sec
    if clip_dur < min_duration:
        total_dur = _get_video_duration(video_path)
        if total_dur > 0:
            # Try extending forward
            new_end = start_sec + min_duration
            if new_end <= total_dur:
                end_sec = new_end
            else:
                # Shift start back to keep min_dur ending at video end
                new_start = total_dur - min_duration
                if new_start < 0:
                    # Video itself is shorter than min_duration
                    start_sec = 0.0
                    end_sec = total_dur
                else:
                    start_sec = new_start
                    end_sec = total_dur

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-i", video_path,
        "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
        "-preset", "ultrafast", "-q:v", "23",
        tmp.name
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)
    return tmp.name


def get_shot_boundaries(video_path: str, transnetv2_result: dict = None) -> List[Tuple[float, float]]:
    """Get shot boundaries from TransNetV2
    Returns list of (start_sec, end_sec) tuples.
    TransNetV2 returns: {shots: [{start_time, end_time, ...}, ...]}
    """
    if transnetv2_result is None:
        transnetv2_result = call_transnetv2(video_path)

    # TransNetV2 returns a "shots" array
    shots = transnetv2_result.get("shots", [])
    if not shots:
        # Fallback: try "scenes" key
        shots = transnetv2_result.get("scenes", [])

    if not shots:
        duration = transnetv2_result.get("duration", 15.0)
        return [(0.0, duration)]

    boundaries = []
    for shot in shots:
        start = shot.get("start_time", shot.get("start", 0))
        end = shot.get("end_time", shot.get("end", 0))
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            boundaries.append((float(start), float(end)))

    if not boundaries:
        duration = transnetv2_result.get("duration", 15.0)
        return [(0.0, duration)]

    return sorted(boundaries, key=lambda x: x[0])


# ============================================================
# Normalization Maps
# (Aligned with shot_details.txt terminology)
# ============================================================

# Camera Motion - from shot_details.txt line 44-45, 47-48
MOTION_NORMALIZE = {
    # Primary camera movements
    "static": "Static", "locked-off": "Static", "locked": "Static",
    "pan": "Pan",
    "tilt": "Tilt",
    "roll": "Roll",
    "dolly-in": "Dolly-In", "push-in": "Dolly-In", "dolly in": "Dolly-In",
    "dolly-out": "Dolly-Out", "pull-out": "Dolly-Out", "dolly out": "Dolly-Out",
    "tracking": "Tracking", "follow": "Tracking",
    "truck": "Truck", "lateral": "Truck",
    "arc": "Arc", "orbit": "Arc",
    "crane": "Crane", "boom": "Crane", "pedestal": "Crane",
    "handheld": "Handheld",
    "steadicam": "Steadicam", "gimbal": "Steadicam",
    # Optical / Hybrid
    "zoom": "Zoom", "zoom-in": "Zoom", "zoom-out": "Zoom",
    "dolly zoom": "Dolly Zoom", "vertigo": "Dolly Zoom",
}

# Shot Scale - from shot_details.txt line 39
SCALE_NORMALIZE = {
    "els": "ELS", "extreme long": "ELS",
    "ls": "LS", "long shot": "LS", "wide": "LS",
    "fs": "FS", "full shot": "FS", "full body": "FS",
    "ms": "MS", "medium shot": "MS", "mid": "MS",
    "mcu": "MCU", "medium close": "MCU",
    "close shot": "MCU", "bust": "MCU",
    "cu": "CU", "close-up": "CU", "close up": "CU",
    "ecu": "ECU", "extreme close": "ECU",
}

# Camera Angle - from shot_details.txt line 42
ANGLE_NORMALIZE = {
    "eye-level": "Eye-Level", "eye level": "Eye-Level",
    "high-angle": "High-Angle", "high angle": "High-Angle",
    "low-angle": "Low-Angle", "low angle": "Low-Angle",
    "bird's-eye": "Bird's-Eye", "overhead": "Bird's-Eye", "bird": "Bird's-Eye",
    "worm's-eye": "Worm's-Eye", "ground-level": "Worm's-Eye", "worm": "Worm's-Eye",
    "dutch": "Dutch Angle", "canted": "Dutch Angle", "tilt": "Dutch Angle",
    "over-the-shoulder": "OTS", "ots": "OTS",
    "pov": "POV", "first-person": "POV", "point-of-view": "POV",
}

# Depth of Field - from shot_details.txt line 50-51
DOF_NORMALIZE = {
    "shallow": "Shallow DoF", "shallow dof": "Shallow DoF",
    "deep": "Deep Focus", "deep focus": "Deep Focus",
    "wide-angle": "Wide-Angle Lens", "wide angle": "Wide-Angle Lens",
    "standard": "Standard Lens", "normal": "Standard Lens",
    "telephoto": "Telephoto Lens", "tele": "Telephoto Lens",
    "macro": "Macro Lens",
    "tilt-shift": "Tilt-Shift Lens",
}


def normalize_motion(raw: str) -> str:
    """Normalize camera motion string to standard form"""
    raw_lower = raw.lower().strip().replace("(", "").replace(")", "").replace(" shot", "")
    for key, val in MOTION_NORMALIZE.items():
        if key in raw_lower:
            return val
    return raw.strip()


def normalize_scale(raw: str) -> str:
    """Normalize shot scale string to standard form"""
    raw_lower = raw.lower().strip()
    for key, val in SCALE_NORMALIZE.items():
        if key in raw_lower:
            return val
    return raw.upper().strip()


def normalize_angle(raw: str) -> str:
    """Normalize camera angle string to standard form"""
    raw_lower = raw.lower().strip().replace(" shot", "")
    for key, val in ANGLE_NORMALIZE.items():
        if key in raw_lower:
            return val
    return raw.strip()


def normalize_dof(raw: str) -> str:
    """Normalize depth of field string to standard form"""
    raw_lower = raw.lower().strip().replace("(", "").replace(")", "")
    for key, val in DOF_NORMALIZE.items():
        if key in raw_lower:
            return val
    return raw.strip()


# ============================================================
# Per-Shot E1 Evaluation Functions
# ============================================================

# System prompts in English

# E1-Camera Motion
SYSTEM_E1_MOTION = (
    "You are an expert cinematographer and camera movement analyst. "
    "Based on the provided 6DoF camera trajectory data and the video clip, "
    "identify the primary camera movement technique used in this specific shot. "
    "Answer with ONLY the option letter."
)

# E1-Shot Scale
SYSTEM_E1_SCALE = (
    "You are a professional director of photography. "
    "Based on the expert model analysis and the video clip, "
    "identify the shot scale (framing size) used in this specific shot. "
    "Answer with ONLY the option letter."
)

# E1-Camera Angle
SYSTEM_E1_ANGLE = (
    "You are a professional cinematographer specializing in camera placement. "
    "Based on the head pose estimation data and the video clip, "
    "identify the camera angle used in this specific shot. "
    "Answer with ONLY the option letter."
)

# E1-Depth of Field
SYSTEM_E1_DOF = (
    "You are a professional lens and focus specialist. "
    "Based on the depth-of-field analysis and the video clip, "
    "identify the depth of field / focal length characteristic of this shot. "
    "Answer with ONLY the option letter."
)

# C1-Montage Type
# ============================================================
# Skill config loading (Plan B: prompts.yaml is the single source of truth)
# ============================================================
# SYSTEM_C1_MONTAGE is now loaded from
# benchmark/skills/montage-classification-skill/prompts.yaml (mode_b.system_prompt).
# Falls back to the historical hardcoded text with a printed warning.
try:
    from skill_loader import load_skill
    _c1_mode_b_cfg = load_skill("montage-classification-skill").get("mode_b", {})
    SYSTEM_C1_MONTAGE = _c1_mode_b_cfg["system_prompt"]
except Exception as _c1_mode_b_err:
    print(f"[WARN] Failed to load montage-classification-skill mode_b config: {_c1_mode_b_err}")
    SYSTEM_C1_MONTAGE = (
        "You are an expert in film editing theory and montage classification. "
        "Based on the cross-shot semantic analysis and the video, "
        "identify the montage structure used in this video. "
        "Answer with ONLY the option letter."
    )

# D1-Transition
# ============================================================
# Skill config loading (Plan B: prompts.yaml is the single source of truth)
# ============================================================
# SYSTEM_D1_TRANSITION and the 8 per-type user prompt templates are now loaded
# from benchmark/skills/transition-semantics-skill/prompts.yaml (mode_b.*).
# Falls back to the historical hardcoded text with a printed warning.
try:
    from skill_loader import load_skill
    _d1_mode_b_cfg = load_skill("transition-semantics-skill").get("mode_b", {})
    SYSTEM_D1_TRANSITION = _d1_mode_b_cfg["system_prompt"]
    D1_TEMPLATES = _d1_mode_b_cfg["templates"]
except Exception as _d1_mode_b_err:
    print(f"[WARN] Failed to load transition-semantics-skill mode_b config, "
          f"falling back to hardcoded D1 prompts: {_d1_mode_b_err}")
    SYSTEM_D1_TRANSITION = (
        "You are a professional film editor specializing in transition analysis. "
        "Based on the expert model evidence and the video, "
        "determine whether the described transition technique was correctly executed. "
        "Answer with ONLY 'Yes' or 'No'."
    )
    D1_TEMPLATES = {
        "logical": (
            "At the transition from Shot {shot_from} to Shot {shot_to}:\n"
            "{evidence}\n\n"
            "Question: Does this transition establish a clear causal/logical relationship "
            "(i.e., the action or event in the preceding shot naturally motivates or leads to "
            "the content of the following shot)?\n"
            "Answer: Yes or No"
        ),
        "pov": (
            "At the transition from Shot {shot_from} to Shot {shot_to}:\n"
            "{evidence}\n\n"
            "Question: Does this transition correctly implement a Point-of-View cut "
            "(one shot shows a character looking, the next shot shows what they see "
            "from their visual perspective)?\n"
            "Answer: Yes or No"
        ),
        "exit_enter": (
            "At the transition from Shot {shot_from} to Shot {shot_to}:\n"
            "Object detection edge analysis: {edge_analysis}\n\n"
            "Question: Does the principal subject exit the frame in one shot and "
            "a subject enters the frame in the following shot (Exit-and-Entry transition)?\n"
            "Answer: Yes or No"
        ),
        "obstruction": (
            "At the transition from Shot {shot_from} to Shot {shot_to}:\n"
            "Saliency mask coverage: {mask_ratio}\n\n"
            "Question: Is the transition masked by a foreground element passing through "
            "and momentarily occluding the frame (Wipe-By / Foreground-Occlusion transition)?\n"
            "Answer: Yes or No"
        ),
        "cutaway": (
            "At the transition from Shot {shot_from} to Shot {shot_to}:\n"
            "{evidence}\n\n"
            "Question: Does this transition use a cutaway to scenery / empty shot "
            "(a landscape or object-only shot devoid of characters) as a bridging element?\n"
            "Answer: Yes or No"
        ),
        "flag": (
            "At the transition from Shot {shot_from} to Shot {shot_to}:\n"
            "Cross-shot semantic similarity: {sim_val}\n\n"
            "Question: Does this transition implement a Gilligan Cut "
            "(a character makes a confident declaration in one shot, "
            "immediately followed by a shot that contradicts or undermines it)?\n"
            "Answer: Yes or No"
        ),
        "whip_pan": (
            "At the transition from Shot {shot_from} to Shot {shot_to}:\n"
            "{evidence}\n\n"
            "Question: Does this transition use rapid camera motion (whip pan, swish pan, "
            "or fast camera movement) to bridge between two disparate scenes?\n"
            "Answer: Yes or No"
        ),
        "extreme_scale": (
            "At the transition from Shot {shot_from} to Shot {shot_to}:\n"
            "{evidence}\n\n"
            "Question: Does this transition cut from one extreme shot scale to the opposite extreme "
            "(e.g., Extreme Close-Up to Extreme Long Shot, or vice versa) to create a dramatic scale contrast?\n"
            "Answer: Yes or No"
        ),
        "generic": (
            "At the transition from Shot {shot_from} to Shot {shot_to}:\n"
            "Intended transition type: {ctype}\n"
            "Design intent: {desc}\n\n"
            "Question: Does this transition successfully convey the intended "
            "cinematographic effect described above?\n"
            "Answer: Yes or No"
        ),
    }


# ---- E1-Camera Motion per shot ----
def eval_e1_camera_motion_per_shot(shot_clip_path: str, shot_idx: int,
                                    gt_motion: str, monst3r_result: dict) -> dict:
    """Evaluate camera motion for a single shot"""
    # MonST3R evidence
    evidence = monst3r_result.get("summary", {})
    primary = monst3r_result.get("primary_motion", "static")

    evidence_text = (
        f"6DoF Camera Trajectory Analysis for Shot {shot_idx + 1}:\n"
        f"- Detected primary motion: {primary}\n"
        f"- Horizontal rotation (Pan): {evidence.get('pan_degrees', 0):.1f}°\n"
        f"- Vertical rotation (Tilt): {evidence.get('tilt_degrees', 0):.1f}°\n"
        f"- Forward/backward displacement (Dolly): {evidence.get('dolly_z', 0):.4f}\n"
        f"- Lateral displacement (Truck): {evidence.get('track_x', 0):.4f}\n"
        f"- Vertical displacement (Crane): {evidence.get('crane_y', 0):.4f}\n"
        f"- Focal length change (Zoom): {evidence.get('focal_change_ratio', 0)*100:.1f}%\n"
    )

    # Options aligned with shot_details.txt
    options = (
        "A. Pan (horizontal rotation in place)\n"
        "B. Tilt (vertical rotation in place)\n"
        "C. Dolly-In / Push-In (camera moves forward toward subject)\n"
        "D. Dolly-Out / Pull-Out (camera moves backward away from subject)\n"
        "E. Tracking Shot / Follow Shot (camera moves alongside subject)\n"
        "F. Truck / Lateral Tracking (camera moves sideways)\n"
        "G. Arc Shot / Orbit (camera circles around subject)\n"
        "H. Crane Shot / Boom / Pedestal (camera rises or descends vertically)\n"
        "I. Handheld Shot (visible hand-held camera shake)\n"
        "J. Steadicam Shot (smooth stabilized movement)\n"
        "K. Zoom (focal length change, camera stays still)\n"
        "L. Dolly Zoom / Vertigo Effect (simultaneous dolly + zoom)\n"
        "M. Static Shot / Locked-Off (camera completely still)\n"
    )

    user_text = (
        f"{evidence_text}\n"
        f"Watch this shot and determine the primary camera movement technique.\n\n"
        f"{options}\n"
        f"Answer with the option letter (A-M):"
    )

    response = call_vlm(SYSTEM_E1_MOTION, user_text, shot_clip_path)

    option_map = {
        "A": "Pan", "B": "Tilt", "C": "Dolly-In", "D": "Dolly-Out",
        "E": "Tracking", "F": "Truck", "G": "Arc", "H": "Crane",
        "I": "Handheld", "J": "Steadicam", "K": "Zoom", "L": "Dolly Zoom",
        "M": "Static",
    }
    pred_letter = parse_choice(response, list(option_map.keys()))
    pred_motion = option_map.get(pred_letter, "Unknown")

    # Compare with GT
    gt_norm = normalize_motion(gt_motion)
    pred_norm = pred_motion  # Already normalized

    # Scoring: exact match or close match
    hit = (gt_norm == pred_norm)
    # Allow partial matches for related movements
    close_pairs = [
        ("Dolly-In", "Tracking"), ("Dolly-Out", "Tracking"),
        ("Steadicam", "Tracking"), ("Steadicam", "Handheld"),
        ("Crane", "Tilt"), ("Truck", "Tracking"),
    ]
    partial = any((gt_norm == a and pred_norm == b) or (gt_norm == b and pred_norm == a)
                  for a, b in close_pairs)
    score = 1.0 if hit else (0.5 if partial else 0.0)

    return {
        "dimension": "E1-camera_motion",
        "shot_idx": shot_idx,
        "score": score,
        "pred_motion": pred_motion,
        "gt_motion": gt_motion,
        "gt_normalized": gt_norm,
        "monst3r_primary": primary,
        "vlm_choice": pred_letter,
        "vlm_response": response[:200],
    }


# ---- E1-Shot Scale per shot ----
def eval_e1_shot_scale_per_shot(shot_clip_path: str, shot_idx: int,
                                 gt_scale: str, movieshots_result: dict,
                                 yolov8_result: dict) -> dict:
    """Evaluate shot scale for a single shot"""
    dominant = movieshots_result.get("dominant_scale", "MS")
    face_ratio = yolov8_result.get("avg_face_ratio", 0)
    body_ratio = yolov8_result.get("avg_body_ratio", 0)

    evidence_text = (
        f"Shot Scale Analysis for Shot {shot_idx + 1}:\n"
        f"- Model-predicted dominant scale: {dominant}\n"
        f"- Face-to-frame ratio: {face_ratio:.3f}\n"
        f"- Body-to-frame ratio: {body_ratio:.3f}\n"
    )

    # Options aligned with shot_details.txt line 39
    options = (
        "A. Extreme Long Shot (ELS) - subject is very small, vast landscape\n"
        "B. Long Shot (LS) / Wide Shot - full environment with small figures\n"
        "C. Full Shot (FS) - complete human body head to toe\n"
        "D. Medium Shot (MS) - waist up\n"
        "E. Medium Close-Up (MCU) / Close Shot / Bust Shot - chest up\n"
        "F. Close-Up (CU) - face fills the frame\n"
        "G. Extreme Close-Up (ECU) - only a part of face or small detail\n"
    )

    user_text = (
        f"{evidence_text}\n"
        f"Watch this shot and identify the shot scale (framing size).\n\n"
        f"{options}\n"
        f"Answer with the option letter (A-G):"
    )

    response = call_vlm(SYSTEM_E1_SCALE, user_text, shot_clip_path)

    option_map = {
        "A": "ELS", "B": "LS", "C": "FS", "D": "MS",
        "E": "MCU", "F": "CU", "G": "ECU",
    }
    pred_letter = parse_choice(response, list(option_map.keys()))
    pred_scale = option_map.get(pred_letter, "MS")

    # Compare with GT
    gt_norm = normalize_scale(gt_scale)

    # Scoring with adjacency tolerance
    scale_order = ["ECU", "CU", "MCU", "MS", "FS", "LS", "ELS"]
    gt_idx = scale_order.index(gt_norm) if gt_norm in scale_order else 3
    pred_idx = scale_order.index(pred_scale) if pred_scale in scale_order else 3
    distance = abs(gt_idx - pred_idx)

    if distance == 0:
        score = 1.0
    elif distance == 1:
        score = 0.7  # Adjacent scale
    elif distance == 2:
        score = 0.3
    else:
        score = 0.0

    return {
        "dimension": "E1-shot_scale",
        "shot_idx": shot_idx,
        "score": score,
        "pred_scale": pred_scale,
        "gt_scale": gt_scale,
        "gt_normalized": gt_norm,
        "vlm_choice": pred_letter,
        "vlm_response": response[:200],
    }


# ---- E1-Camera Angle per shot ----
def eval_e1_angle_per_shot(shot_clip_path: str, shot_idx: int,
                            gt_angle: str, sixdrepnet_result: dict) -> dict:
    """Evaluate camera angle for a single shot"""
    avg_pitch = sixdrepnet_result.get("avg_pitch", 0)

    evidence_text = (
        f"Camera Angle Analysis for Shot {shot_idx + 1}:\n"
        f"- Head pose estimation avg pitch: {avg_pitch:.1f}° "
        f"(negative=looking down→camera is high, positive=looking up→camera is low)\n"
    )

    # Options aligned with shot_details.txt line 42
    options = (
        "A. Eye-Level Shot (camera at subject's eye height)\n"
        "B. High-Angle Shot (camera looks down at subject)\n"
        "C. Low-Angle Shot (camera looks up at subject)\n"
        "D. Bird's-Eye View / Overhead Shot (camera directly above)\n"
        "E. Worm's-Eye View / Ground-Level Shot (camera at ground level looking up)\n"
        "F. Dutch Angle / Canted Angle (tilted frame)\n"
        "G. Over-the-Shoulder Shot (OTS)\n"
        "H. Point-of-View Shot (POV / First-Person)\n"
    )

    user_text = (
        f"{evidence_text}\n"
        f"Watch this shot and identify the camera angle.\n\n"
        f"{options}\n"
        f"Answer with the option letter (A-H):"
    )

    response = call_vlm(SYSTEM_E1_ANGLE, user_text, shot_clip_path)

    option_map = {
        "A": "Eye-Level", "B": "High-Angle", "C": "Low-Angle",
        "D": "Bird's-Eye", "E": "Worm's-Eye", "F": "Dutch Angle",
        "G": "OTS", "H": "POV",
    }
    pred_letter = parse_choice(response, list(option_map.keys()))
    pred_angle = option_map.get(pred_letter, "Eye-Level")

    # Compare with GT
    gt_norm = normalize_angle(gt_angle)
    hit = (gt_norm == pred_angle)

    return {
        "dimension": "E1-angle",
        "shot_idx": shot_idx,
        "score": 1.0 if hit else 0.0,
        "pred_angle": pred_angle,
        "gt_angle": gt_angle,
        "gt_normalized": gt_norm,
        "vlm_choice": pred_letter,
        "vlm_response": response[:200],
    }


# ---- E1-DoF per shot ----
def eval_e1_dof_per_shot(shot_clip_path: str, shot_idx: int,
                          gt_dof: str, saliency_result: dict) -> dict:
    """Evaluate depth of field for a single shot"""
    dof_score = saliency_result.get("dof_score", 0.5)

    evidence_text = (
        f"Depth of Field Analysis for Shot {shot_idx + 1}:\n"
        f"- DoF score: {dof_score:.3f} (higher = shallower depth of field)\n"
    )

    # Options aligned with shot_details.txt line 50-51
    options = (
        "A. Shallow Depth of Field (background significantly blurred, subject sharp)\n"
        "B. Deep Focus / Deep Depth of Field (both foreground and background sharp)\n"
        "C. Wide-Angle Lens (broad field of view, some distortion at edges)\n"
        "D. Standard Lens / Normal Lens (natural perspective, moderate DoF)\n"
        "E. Telephoto Lens (compressed perspective, shallow DoF)\n"
        "F. Macro Lens (extreme close-up with very thin focal plane)\n"
        "G. Tilt-Shift Lens (selective focus with miniature effect)\n"
    )

    user_text = (
        f"{evidence_text}\n"
        f"Watch this shot and identify the depth of field / focal length characteristic.\n\n"
        f"{options}\n"
        f"Answer with the option letter (A-G):"
    )

    response = call_vlm(SYSTEM_E1_DOF, user_text, shot_clip_path)

    option_map = {
        "A": "Shallow DoF", "B": "Deep Focus", "C": "Wide-Angle Lens",
        "D": "Standard Lens", "E": "Telephoto Lens", "F": "Macro Lens",
        "G": "Tilt-Shift Lens",
    }
    pred_letter = parse_choice(response, list(option_map.keys()))
    pred_dof = option_map.get(pred_letter, "Standard Lens")

    # Compare with GT
    gt_norm = normalize_dof(gt_dof)
    hit = (gt_norm == pred_dof)
    # Allow partial match for related DoF types
    shallow_group = {"Shallow DoF", "Telephoto Lens", "Macro Lens"}
    deep_group = {"Deep Focus", "Wide-Angle Lens"}
    partial = (gt_norm in shallow_group and pred_dof in shallow_group) or \
              (gt_norm in deep_group and pred_dof in deep_group)

    score = 1.0 if hit else (0.5 if partial else 0.0)

    return {
        "dimension": "E1-dof",
        "shot_idx": shot_idx,
        "score": score,
        "pred_dof": pred_dof,
        "gt_dof": gt_dof,
        "gt_normalized": gt_norm,
        "vlm_choice": pred_letter,
        "vlm_response": response[:200],
    }


# ============================================================
# C1 Montage type evaluation
# ============================================================
def _extract_prompt_shot_time_range(shot: dict) -> Optional[Tuple[float, float]]:
    """Extract [start-end] seconds from shot description text if present."""
    desc = shot.get("description_prompt", "")
    match = re.search(r"\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)s\]", desc)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _mean_cross_shot_frame_similarity(sim_matrix: list,
                                      frame_indices_a: list,
                                      frame_indices_b: list) -> Optional[float]:
    """Mean CLIP similarity between frames from two different shot groups."""
    values = []
    for i in frame_indices_a:
        if i >= len(sim_matrix):
            continue
        row = sim_matrix[i]
        for j in frame_indices_b:
            if j >= len(row):
                continue
            values.append(float(row[j]))
    return float(np.mean(values)) if values else None


def _build_c1_event_label_clip_evidence(prompt: dict, clip_result: dict) -> tuple:
    """Build C1 Mode B evidence grouped by prompt.shots[].event_coherence_label.

    This does not call CLIP again. It reuses the existing frame similarity matrix
    and aggregates similarities between shots sharing the same event label.
    """
    shots = prompt.get("shots", [])
    labels = [shot.get("event_coherence_label", "unlabeled") for shot in shots]
    label_sequence = " -> ".join(str(label) for label in labels) if labels else "N/A"

    groups = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(idx)

    transitions = sum(1 for i in range(len(labels) - 1) if labels[i] != labels[i + 1])
    same_label_adjacent = max(len(labels) - 1, 0) - transitions

    frame_times = clip_result.get("frame_times", []) or []
    sim_matrix = clip_result.get("frame_sim_matrix", []) or []
    shot_ranges = [_extract_prompt_shot_time_range(shot) for shot in shots]

    shot_frame_indices = [[] for _ in shots]
    for frame_idx, frame_time in enumerate(frame_times):
        for shot_idx, shot_range in enumerate(shot_ranges):
            if shot_range is None:
                continue
            start_sec, end_sec = shot_range
            is_last_shot = shot_idx == len(shot_ranges) - 1
            if start_sec <= float(frame_time) < end_sec or (is_last_shot and float(frame_time) <= end_sec):
                shot_frame_indices[shot_idx].append(frame_idx)
                break

    within_label_summaries = []
    between_label_summaries = []
    raw_stats = {
        "label_sequence": labels,
        "label_groups": {str(label): [idx + 1 for idx in indices] for label, indices in groups.items()},
        "label_transitions": transitions,
        "same_label_adjacent_cuts": same_label_adjacent,
        "same_label_similarity": {},
        "same_label_pair_counts": {},
        "between_label_similarity": {},
        "between_label_pair_counts": {},
    }

    sorted_groups = sorted(groups.items(), key=lambda item: str(item[0]))

    for label, shot_indices in sorted_groups:
        pair_means = []
        for pos, shot_i in enumerate(shot_indices):
            for shot_j in shot_indices[pos + 1:]:
                mean_sim = _mean_cross_shot_frame_similarity(
                    sim_matrix, shot_frame_indices[shot_i], shot_frame_indices[shot_j]
                )
                if mean_sim is not None:
                    pair_means.append(mean_sim)

        label_key = str(label)
        raw_stats["same_label_pair_counts"][label_key] = len(pair_means)
        if pair_means:
            label_mean = float(np.mean(pair_means))
            raw_stats["same_label_similarity"][label_key] = round(label_mean, 4)
            sim_text = f"mean_within_label_clip={label_mean:.3f}, n_shot_pairs={len(pair_means)}"
        else:
            raw_stats["same_label_similarity"][label_key] = None
            sim_text = "mean_within_label_clip=N/A"

        within_label_summaries.append(
            f"  - within label {label_key}: shots {[idx + 1 for idx in shot_indices]}, {sim_text}"
        )

    for pos, (label_a, shot_indices_a) in enumerate(sorted_groups):
        for label_b, shot_indices_b in sorted_groups[pos + 1:]:
            pair_means = []
            for shot_i in shot_indices_a:
                for shot_j in shot_indices_b:
                    mean_sim = _mean_cross_shot_frame_similarity(
                        sim_matrix, shot_frame_indices[shot_i], shot_frame_indices[shot_j]
                    )
                    if mean_sim is not None:
                        pair_means.append(mean_sim)

            pair_key = f"{label_a}-vs-{label_b}"
            raw_stats["between_label_pair_counts"][pair_key] = len(pair_means)
            if pair_means:
                pair_mean = float(np.mean(pair_means))
                raw_stats["between_label_similarity"][pair_key] = round(pair_mean, 4)
                sim_text = f"mean_between_label_clip={pair_mean:.3f}, n_shot_pairs={len(pair_means)}"
            else:
                raw_stats["between_label_similarity"][pair_key] = None
                sim_text = "mean_between_label_clip=N/A"

            between_label_summaries.append(
                f"  - between label {label_a} and {label_b}: "
                f"shots {[idx + 1 for idx in shot_indices_a]} vs {[idx + 1 for idx in shot_indices_b]}, {sim_text}"
            )

    # ---- Plan B evidence: expose ONLY the pooled CLIP similarity numbers. ----
    # The grouping itself comes from the ground-truth prompt.shots[].event_coherence_label
    # (derived from global_editing_style, i.e. the C1 answer), so the label ids, the
    # label sequence, the per-group shot lists and the switch counts are NEVER shown to
    # the VLM. Only two pooled scalars are exposed, pair-count weighted so that they are
    # true pooled means rather than means-of-means.
    def _pooled(sim_map: dict, count_map: dict):
        num = 0.0
        den = 0
        for key, value in sim_map.items():
            if value is None:
                continue
            weight = int(count_map.get(key, 0) or 0)
            if weight <= 0:
                continue
            num += float(value) * weight
            den += weight
        return (num / den) if den else None

    within_pooled = _pooled(raw_stats["same_label_similarity"], raw_stats["same_label_pair_counts"])
    between_pooled = _pooled(raw_stats["between_label_similarity"], raw_stats["between_label_pair_counts"])
    raw_stats["within_pooled_similarity"] = round(within_pooled, 4) if within_pooled is not None else None
    raw_stats["between_pooled_similarity"] = round(between_pooled, 4) if between_pooled is not None else None

    evidence_text = (
        "Shot-Group CLIP Similarity Statistics:\n"
        f"- Mean CLIP similarity between shots belonging to the same coherent event group: "
        f"{f'{within_pooled:.3f}' if within_pooled is not None else 'N/A'}\n"
        f"- Mean CLIP similarity between shots belonging to different coherent event groups: "
        f"{f'{between_pooled:.3f}' if between_pooled is not None else 'N/A'}\n"
    )
    return evidence_text, raw_stats


def eval_c1_montage(video_path: str, prompt: dict, clip_result: dict,
                    transnetv2_result: dict = None) -> dict:
    """C1: Montage type identification
    Aligned with the seed.txt montage taxonomy
    """
    gt_style = prompt.get("global_editing_style", "")

    cross_sims_raw = clip_result.get("cross_shot_similarities", [])
    # CLIP returns list of dicts: [{pair:[t1,t2], similarity:float}, ...]
    cross_sims = [s["similarity"] if isinstance(s, dict) else float(s) for s in cross_sims_raw]
    # Number of shots now comes from the TransNetV2 detection on the actual video,
    # not from the ground-truth prompt.number_of_shots.
    detected_num_shots = (transnetv2_result or {}).get("num_shots")
    if detected_num_shots is None:
        detected_shots = (transnetv2_result or {}).get("shots") or []
        detected_num_shots = len(detected_shots) if detected_shots else "?"
    evidence_text = (
        f"Cross-shot Semantic Analysis:\n"
        f"- Cross-shot CLIP similarities: {[f'{s:.3f}' for s in cross_sims[:5]]}\n"
        f"- Average cross-shot CLIP similarity: {clip_result.get('avg_cross_shot_sim', 'N/A')}\n"
        f"- Number of shots: {detected_num_shots}\n"
    )
    # Plan B: inject ONLY the pooled CLIP similarity scalars (no label ids / sequence /
    # per-group shot lists / switch counts). Full per-label stats stay in the result JSON
    # under event_label_clip_stats for auditing only.
    grouped_clip_evidence, grouped_label_stats = _build_c1_event_label_clip_evidence(prompt, clip_result)

    # Options aligned with the seed.txt montage taxonomy (13 types)
    options = (
        "A. Sequential Montage (events in strict chronological order)\n"
        "B. Parallel Montage (multiple storylines intercut, converging later)\n"
        "C. Crosscut Montage (rapid alternation of simultaneous events for tension)\n"
        "D. Repetition Montage (meaningful shot repeated at key moments)\n"
        "E. Dialogue Montage (a conversation split across different scenes/time)\n"
        "F. Lyrical Montage (insert scenic/poetic shots to evoke emotion)\n"
        "G. Psychological Montage (visualize dreams, memories, hallucinations, imagination)\n"
        "H. Metaphorical Montage (visual analogy to imply deeper meaning)\n"
        "I. Contrast Montage (juxtapose opposites for dramatic conflict)\n"
        "J. Accumulative Montage (rapid succession of similar shots to build intensity)\n"
        "K. Montage of Attractions (insert unrelated shots to provoke emotion/idea)\n"
        "L. Reflexive Montage (metaphor drawn from objects already in the scene)\n"
        "M. Ideological Montage (re-edit existing footage to argue a thesis/ideology)\n"
    )

    user_text = (
        f"{evidence_text}\n"
        f"{grouped_clip_evidence}\n"
        f"Watch this video and identify which montage structure best describes its editing. "
        f"Base your decision on the full-video visual and narrative structure that you actually "
        f"observe. The CLIP similarity numbers above are auxiliary signals only; when they "
        f"conflict with what you see in the video, trust the video.\n\n"
        f"{options}\n"
        f"Answer with the option letter (A-M):"
    )

    response = call_vlm(SYSTEM_C1_MONTAGE, user_text, video_path)
    pred = parse_choice(response, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"])

    montage_map = {
        "A": "Sequential", "B": "Parallel", "C": "Crosscut",
        "D": "Repetition", "E": "Dialogue", "F": "Lyrical",
        "G": "Psychological", "H": "Metaphorical", "I": "Contrast",
        "J": "Accumulative", "K": "Attractions", "L": "Reflexive",
        "M": "Ideological",
    }
    pred_type = montage_map.get(pred, "Unknown")
    hit = pred_type.lower() in gt_style.lower() if gt_style else False
    score = 1.0 if hit else 0.0

    return {
        "dimension": "C1",
        "metric": "montage_type",
        "score": score,
        "pred_type": pred_type,
        "gt_style": gt_style[:100],
        "vlm_choice": pred,
        "vlm_response": response[:200],
        "event_label_clip_stats": grouped_label_stats,
    }


# ============================================================
# D1 Transition Evaluation
# ============================================================
def eval_d1_transition(video_path: str, prompt: dict, shot_idx: int,
                       expert_results: dict) -> dict:
    """Evaluate a single D1 transition (Mode B types)"""
    shots = prompt.get("shots", [])
    if shot_idx >= len(shots) or "transition_to_next" not in shots[shot_idx]:
        return None

    transition = shots[shot_idx]["transition_to_next"]
    ctype = transition.get("cinematographic_type", "")
    description = transition.get("description", "")

    # Route by transition type (case-insensitive)
    ctype_lower = ctype.lower()
    if "logical" in ctype_lower or "causal" in ctype_lower:
        return _eval_d1_logical(video_path, shot_idx, transition, expert_results)
    elif "pov" in ctype_lower or "point-of-view" in ctype_lower:
        return _eval_d1_pov(video_path, shot_idx, transition, expert_results)
    elif "exit" in ctype_lower or "entry" in ctype_lower or "walk" in ctype_lower:
        return _eval_d1_exit_enter(video_path, shot_idx, transition, expert_results)
    elif "occlusion" in ctype_lower or "wipe-by" in ctype_lower or "foreground" in ctype_lower:
        return _eval_d1_obstruction(video_path, shot_idx, transition, expert_results)
    elif "cutaway" in ctype_lower or "empty" in ctype_lower or "insert" in ctype_lower:
        return _eval_d1_cutaway(video_path, shot_idx, transition, expert_results)
    elif "gilligan" in ctype_lower or "flag" in ctype_lower:
        return _eval_d1_flag(video_path, shot_idx, transition, expert_results)
    elif "whip" in ctype_lower or "camera-movement" in ctype_lower or "camera movement" in ctype_lower:
        return _eval_d1_whip_pan(video_path, shot_idx, transition, expert_results, prompt)
    elif "extreme" in ctype_lower or "polar" in ctype_lower:
        return _eval_d1_extreme_scale(video_path, shot_idx, transition, expert_results, prompt)
    else:
        return _eval_d1_generic(video_path, shot_idx, transition, expert_results)


def _eval_d1_logical(video_path, shot_idx, transition, expert_results):
    """D1-Logical Cut / Causal Transition"""
    clip_data = expert_results.get("clip", {})
    cross_sims_raw = clip_data.get("cross_shot_similarities", [])
    # Extract float from dict if needed
    if shot_idx < len(cross_sims_raw):
        s = cross_sims_raw[shot_idx]
        sim_val = s["similarity"] if isinstance(s, dict) else float(s)
    else:
        sim_val = 0.5

    desc = transition.get("description", "")
    evidence = (
        f"Cross-shot semantic similarity at cut point: {sim_val:.3f}\n"
        f"Intended transition design: {desc[:150]}"
    )

    user_text = D1_TEMPLATES["logical"].format(
        shot_from=shot_idx + 1, shot_to=shot_idx + 2, evidence=evidence
    )

    response = call_vlm(SYSTEM_D1_TRANSITION, user_text, video_path)
    pred = parse_choice(response, ["Yes", "No"])
    if pred is None:
        pred = parse_choice(response, ["是", "否"])
        pred = "Yes" if pred == "是" else ("No" if pred == "否" else None)
    score = 1.0 if pred == "Yes" else 0.0

    return {
        "transition_idx": shot_idx, "type": "Logical Cut",
        "score": score, "vlm_answer": pred, "vlm_response": response[:150],
    }


def _eval_d1_pov(video_path, shot_idx, transition, expert_results):
    """D1-Point-of-View Cut"""
    sixdrepnet_data = expert_results.get("sixdrepnet", {})
    yolov8_data = expert_results.get("yolov8", {})

    evidence = (
        f"Head pose data: {json.dumps(sixdrepnet_data.get('summary', {}), ensure_ascii=False)[:150]}\n"
        f"Person detection: avg_count={yolov8_data.get('avg_person_count', 0)}"
    )

    user_text = D1_TEMPLATES["pov"].format(
        shot_from=shot_idx + 1, shot_to=shot_idx + 2, evidence=evidence
    )

    response = call_vlm(SYSTEM_D1_TRANSITION, user_text, video_path)
    pred = parse_choice(response, ["Yes", "No"])
    if pred is None:
        pred = parse_choice(response, ["是", "否"])
        pred = "Yes" if pred == "是" else ("No" if pred == "否" else None)
    score = 1.0 if pred == "Yes" else 0.0

    return {"transition_idx": shot_idx, "type": "POV Cut", "score": score,
            "vlm_answer": pred, "vlm_response": response[:150]}


def _eval_d1_exit_enter(video_path, shot_idx, transition, expert_results):
    """D1-Exit-and-Entry Transition"""
    yolov8_data = expert_results.get("yolov8", {})

    user_text = D1_TEMPLATES["exit_enter"].format(
        shot_from=shot_idx + 1, shot_to=shot_idx + 2,
        edge_analysis=json.dumps(yolov8_data.get('edge_analysis', {}), ensure_ascii=False)[:150],
    )

    response = call_vlm(SYSTEM_D1_TRANSITION, user_text, video_path)
    pred = parse_choice(response, ["Yes", "No"])
    if pred is None:
        pred = parse_choice(response, ["是", "否"])
        pred = "Yes" if pred == "是" else ("No" if pred == "否" else None)
    score = 1.0 if pred == "Yes" else 0.0

    return {"transition_idx": shot_idx, "type": "Exit-Entry", "score": score,
            "vlm_answer": pred, "vlm_response": response[:150]}


def _eval_d1_obstruction(video_path, shot_idx, transition, expert_results):
    """D1-Foreground-Occlusion Transition"""
    saliency_data = expert_results.get("saliency", {})

    user_text = D1_TEMPLATES["obstruction"].format(
        shot_from=shot_idx + 1, shot_to=shot_idx + 2,
        mask_ratio=f"{saliency_data.get('max_mask_ratio', 0):.2f}",
    )

    response = call_vlm(SYSTEM_D1_TRANSITION, user_text, video_path)
    pred = parse_choice(response, ["Yes", "No"])
    if pred is None:
        pred = parse_choice(response, ["是", "否"])
        pred = "Yes" if pred == "是" else ("No" if pred == "否" else None)
    score = 1.0 if pred == "Yes" else 0.0

    return {"transition_idx": shot_idx, "type": "Occlusion", "score": score,
            "vlm_answer": pred, "vlm_response": response[:150]}


def _eval_d1_cutaway(video_path, shot_idx, transition, expert_results):
    """D1-Cutaway to Scenery / Empty Shot Transition"""
    yolov8_data = expert_results.get("yolov8", {})
    # The places365 service is no longer started, so this evidence line always renders "N/A";
    # it is kept so the prompt text stays byte-identical to the reported runs.
    places365_data = expert_results.get("places365", {})

    evidence = (
        f"Person detection: {yolov8_data.get('avg_person_count', 'N/A')}\n"
        f"Scene classification: {places365_data.get('top_scene', 'N/A')}"
    )

    user_text = D1_TEMPLATES["cutaway"].format(
        shot_from=shot_idx + 1, shot_to=shot_idx + 2, evidence=evidence
    )

    response = call_vlm(SYSTEM_D1_TRANSITION, user_text, video_path)
    pred = parse_choice(response, ["Yes", "No"])
    if pred is None:
        pred = parse_choice(response, ["是", "否"])
        pred = "Yes" if pred == "是" else ("No" if pred == "否" else None)
    score = 1.0 if pred == "Yes" else 0.0

    return {"transition_idx": shot_idx, "type": "Cutaway", "score": score,
            "vlm_answer": pred, "vlm_response": response[:150]}


def _eval_d1_flag(video_path, shot_idx, transition, expert_results):
    """D1-Gilligan Cut"""
    clip_data = expert_results.get("clip", {})
    cross_sims_raw = clip_data.get("cross_shot_similarities", [])
    # Extract float from dict if needed
    if shot_idx < len(cross_sims_raw):
        s = cross_sims_raw[shot_idx]
        sim_val = s["similarity"] if isinstance(s, dict) else float(s)
    else:
        sim_val = 0.5

    user_text = D1_TEMPLATES["flag"].format(
        shot_from=shot_idx + 1, shot_to=shot_idx + 2, sim_val=f"{sim_val:.3f}",
    )

    response = call_vlm(SYSTEM_D1_TRANSITION, user_text, video_path)
    pred = parse_choice(response, ["Yes", "No"])
    if pred is None:
        pred = parse_choice(response, ["是", "否"])
        pred = "Yes" if pred == "是" else ("No" if pred == "否" else None)
    score = 1.0 if pred == "Yes" else 0.0

    return {"transition_idx": shot_idx, "type": "Gilligan Cut", "score": score,
            "vlm_answer": pred, "vlm_response": response[:150]}


def _eval_d1_whip_pan(video_path, shot_idx, transition, expert_results, prompt):
    """D1-Whip-Pan / Camera-Movement Transition -- VLM + RAFT evidence"""
    raft_data = expert_results.get("raft", {})
    magnitudes = raft_data.get("flow_magnitudes", [])
    sample_fps = raft_data.get("sample_fps", 8.0)

    # Parse cut time from shot description_prompt
    shots = prompt.get("shots", [])
    cut_time = 0.0
    if shot_idx < len(shots):
        desc = shots[shot_idx].get("description_prompt", "")
        match = re.search(r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)s?\]', desc)
        if match:
            cut_time = float(match.group(2))

    # Compute local optical flow magnitude around cut point
    threshold = 100.0
    if magnitudes:
        flow_idx = int(cut_time * sample_fps)
        window = 3
        s, e = max(0, flow_idx - window), min(len(magnitudes), flow_idx + window)
        if e > s:
            local_mag = float(np.mean(magnitudes[s:e]))
            peak_mag = float(np.max(magnitudes[s:e]))
            assessment = ("STRONG camera motion detected (above whip-pan threshold)"
                          if local_mag > threshold else
                          "weak/moderate camera motion (below whip-pan threshold)")
            evidence = (
                f"RAFT optical flow evidence at cut point (t={cut_time:.1f}s):\n"
                f"  - Average flow magnitude (±{window} frames): {local_mag:.2f}\n"
                f"  - Peak flow magnitude: {peak_mag:.2f}\n"
                f"  - Whip-pan threshold: {threshold:.1f}\n"
                f"  - Assessment: {assessment}"
            )
        else:
            local_mag = None
            evidence = "RAFT optical flow: insufficient frames around cut point"
    else:
        local_mag = None
        evidence = "RAFT optical flow: data unavailable"

    user_text = D1_TEMPLATES["whip_pan"].format(
        shot_from=shot_idx + 1, shot_to=shot_idx + 2, evidence=evidence,
    )

    response = call_vlm(SYSTEM_D1_TRANSITION, user_text, video_path)
    pred = parse_choice(response, ["Yes", "No"])
    if pred is None:
        pred = parse_choice(response, ["是", "否"])
        pred = "Yes" if pred == "是" else ("No" if pred == "否" else None)
    score = 1.0 if pred == "Yes" else 0.0

    return {"transition_idx": shot_idx, "type": "Whip-Pan", "score": score,
            "vlm_answer": pred, "vlm_response": response[:150],
            "raft_evidence": {"cut_time": cut_time, "local_mag": local_mag}}


# Shot scale ordering: index 0 (closest) → 7 (widest)
_SCALE_ORDER = ["ECU", "CU", "MCU", "MS", "MLS", "LS", "ELS", "XLS"]
_SCALE_LABELS = {
    "ECU": "Extreme Close-Up", "CU": "Close-Up", "MCU": "Medium Close-Up",
    "MS": "Medium Shot", "MLS": "Medium Long Shot", "LS": "Long Shot",
    "ELS": "Extreme Long Shot", "XLS": "Extra Long Shot (aerial/establishing)",
}


def _eval_d1_extreme_scale(video_path, shot_idx, transition, expert_results, prompt):
    """D1-Extreme Scale Cut / Polar-Scale Cut -- VLM + MovieShots evidence"""
    movieshots_data = expert_results.get("movieshots", {})
    per_frame = movieshots_data.get("per_frame", [])

    # Parse cut time from shot description_prompt
    shots = prompt.get("shots", [])
    cut_time = 0.0
    if shot_idx < len(shots):
        desc = shots[shot_idx].get("description_prompt", "")
        match = re.search(r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)s?\]', desc)
        if match:
            cut_time = float(match.group(2))

    # Find nearest MovieShots frames before and after the cut point
    before_frame = None
    after_frame = None
    for fr in per_frame:
        ft = fr.get("frame_time", 0.0)
        if ft <= cut_time:
            if before_frame is None or ft > before_frame.get("frame_time", -1):
                before_frame = fr
        else:
            if after_frame is None or ft < after_frame.get("frame_time", float('inf')):
                after_frame = fr

    # Compute scale distance
    if before_frame and after_frame:
        scale_before = before_frame.get("predicted_scale", "MS")
        scale_after = after_frame.get("predicted_scale", "MS")
        idx_before = before_frame.get("scale_index", 3)
        idx_after = after_frame.get("scale_index", 3)
        scale_distance = abs(idx_after - idx_before)
        extreme_threshold = 4  # e.g., ECU→MLS, CU→LS, MS→ELS, etc.
        assessment = ("EXTREME scale contrast detected"
                      if scale_distance >= extreme_threshold else
                      f"moderate scale difference (distance={scale_distance}, "
                      f"threshold for extreme={extreme_threshold})")
        evidence = (
            f"MovieShots scale evidence at cut point (t={cut_time:.1f}s):\n"
            f"  - Shot {shot_idx + 1} (before cut): {_SCALE_LABELS.get(scale_before, scale_before)} "
            f"(index {idx_before})\n"
            f"  - Shot {shot_idx + 2} (after cut):  {_SCALE_LABELS.get(scale_after, scale_after)} "
            f"(index {idx_after})\n"
            f"  - Scale distance: {scale_distance} (0=same, 7=maximum)\n"
            f"  - Extreme threshold: {extreme_threshold}\n"
            f"  - Assessment: {assessment}"
        )
    else:
        scale_before = scale_after = None
        scale_distance = None
        evidence = "MovieShots scale evidence: insufficient per-frame data around cut point"

    user_text = D1_TEMPLATES["extreme_scale"].format(
        shot_from=shot_idx + 1, shot_to=shot_idx + 2, evidence=evidence,
    )

    response = call_vlm(SYSTEM_D1_TRANSITION, user_text, video_path)
    pred = parse_choice(response, ["Yes", "No"])
    if pred is None:
        pred = parse_choice(response, ["是", "否"])
        pred = "Yes" if pred == "是" else ("No" if pred == "否" else None)
    score = 1.0 if pred == "Yes" else 0.0

    return {"transition_idx": shot_idx, "type": "Extreme Scale Cut", "score": score,
            "vlm_answer": pred, "vlm_response": response[:150],
            "movieshots_evidence": {"cut_time": cut_time,
                                     "scale_before": scale_before,
                                     "scale_after": scale_after,
                                     "scale_distance": scale_distance}}


def _eval_d1_generic(video_path, shot_idx, transition, expert_results):
    """D1-Generic transition evaluation"""
    ctype = transition.get("cinematographic_type", "Unknown")
    desc = transition.get("description", "")

    user_text = D1_TEMPLATES["generic"].format(
        shot_from=shot_idx + 1, shot_to=shot_idx + 2, ctype=ctype, desc=desc[:150],
    )

    response = call_vlm(SYSTEM_D1_TRANSITION, user_text, video_path)
    pred = parse_choice(response, ["Yes", "No"])
    if pred is None:
        pred = parse_choice(response, ["是", "否"])
        pred = "Yes" if pred == "是" else ("No" if pred == "否" else None)
    score = 1.0 if pred == "Yes" else 0.0

    return {"transition_idx": shot_idx, "type": ctype, "score": score,
            "vlm_answer": pred, "vlm_response": response[:150]}


# ============================================================
# B3 Rhythm-Mood Matching
# ============================================================
SYSTEM_B3_RHYTHM = (
    "You are a professional music supervisor and film editor specializing in "
    "evaluating the synchronization between audio rhythm and editing rhythm. "
    "Based on the provided audio source separation analysis and editing pace data, "
    "evaluate how well the overall audio rhythm (including sound effects, ambient sounds, "
    "and any music) matches the video's editing rhythm. "
    "Answer with ONLY the option letter."
)


def eval_b3_rhythm_mood(video_path: str, demucs_result: dict,
                        transnetv2_result: dict, panns_result: dict = None,
                        prompt: dict = None) -> dict:
    """B3: rhythm and mood matching (two sub-dimensions)
    
    Sub1: rhythm match between the overall sound design and the video (VLM, regardless of BGM presence)
    Sub2: match between background-music mood and the video scene (PANNs mood tags + mood inferred from the prompt)
    Final score = (sub1 + sub2) / 2
    """
    import numpy as np
    
    # --- Audio analysis from Demucs ---
    sources = demucs_result.get("sources", {})
    duration = demucs_result.get("duration", 15.0)
    
    # Music energy = drums + bass + other (exclude vocals)
    drums_energy = sources.get("drums", {}).get("energy_ratio", 0)
    bass_energy = sources.get("bass", {}).get("energy_ratio", 0)
    other_energy = sources.get("other", {}).get("energy_ratio", 0)
    vocals_energy = sources.get("vocals", {}).get("energy_ratio", 0)
    music_energy = drums_energy + bass_energy + other_energy
    
    drums_rms = sources.get("drums", {}).get("rms", 0)
    bass_rms = sources.get("bass", {}).get("rms", 0)
    other_rms = sources.get("other", {}).get("rms", 0)
    vocals_rms = sources.get("vocals", {}).get("rms", 0)
    
    # Drum dominance
    total_rms = drums_rms + bass_rms + other_rms + vocals_rms + 1e-8
    drum_dominance = drums_rms / total_rms
    bass_dominance = bass_rms / total_rms
    
    # --- Edit rate from TransNetV2 ---
    shots = transnetv2_result.get("shots", transnetv2_result.get("scenes", []))
    n_cuts = max(0, len(shots) - 1) if shots else 0
    edit_rate = n_cuts / duration if duration > 0 else 0
    avg_shot_duration = duration / len(shots) if shots else duration
    
    # ===== Sub1: overall sound-design rhythm match (VLM) =====
    evidence_text = (
        f"Audio Source Separation Analysis (Demucs):\n"
        f"- Music energy ratio (non-vocal): {music_energy:.3f}\n"
        f"- Drums energy ratio: {drums_energy:.3f} (RMS: {drums_rms:.4f})\n"
        f"- Bass energy ratio: {bass_energy:.3f} (RMS: {bass_rms:.4f})\n"
        f"- Other instruments energy: {other_energy:.3f} (RMS: {other_rms:.4f})\n"
        f"- Vocals energy ratio: {vocals_energy:.3f}\n"
        f"- Drum dominance: {drum_dominance:.2%}\n"
        f"- Bass dominance: {bass_dominance:.2%}\n\n"
        f"Editing Pace Analysis:\n"
        f"- Total duration: {duration:.1f}s\n"
        f"- Number of shots: {len(shots) if shots else 1}\n"
        f"- Number of cuts: {n_cuts}\n"
        f"- Edit rate: {edit_rate:.2f} cuts/second\n"
        f"- Average shot duration: {avg_shot_duration:.1f}s\n"
    )
    
    options = (
        "A. Excellent match (energetic audio with fast cutting, or calm audio with slow pacing - "
        "the overall audio rhythm perfectly complements the editing rhythm)\n"
        "B. Good match (audio and editing rhythm are generally aligned with minor mismatches)\n"
        "C. Moderate mismatch (audio energy level does not clearly correspond to editing pace, "
        "creating some dissonance)\n"
        "D. Poor match (high-energy audio with very slow editing or vice versa - "
        "the rhythm clearly contradicts)\n"
    )
    
    user_text = (
        f"{evidence_text}\n"
        f"Watch the video and listen to its audio. Based on the audio analysis data "
        f"and editing pace above, evaluate how well the overall audio rhythm (including "
        f"sound effects, ambient sounds, and any music) matches the video's editing rhythm "
        f"and visual pacing.\n\n"
        f"{options}\n"
        f"Answer with the option letter (A-D):"
    )
    
    response = call_vlm(SYSTEM_B3_RHYTHM, user_text, video_path)
    pred = parse_choice(response, ["A", "B", "C", "D"])
    
    score_map = {"A": 1.0, "B": 0.75, "C": 0.4, "D": 0.1}
    sub1_score = score_map.get(pred, 0.5)
    sub1_detail = {
        "vlm_choice": pred,
        "vlm_response": response[:200],
        "edit_rate": round(edit_rate, 3),
        "n_cuts": n_cuts,
    }
    
    # ===== Sub2: background-music mood vs scene match (PANNs mood tags) =====
    sub2_score = 0.0
    sub2_detail = {}
    
    # Decide whether there is BGM (using the Demucs energy_ratio)
    has_bgm = True
    if sources:
        vocals_ratio = sources.get("vocals", {}).get("energy_ratio", 0)
        music_ratio = drums_energy + bass_energy + other_energy
        if vocals_ratio > music_ratio * 1.5:
            has_bgm = False
            sub2_detail["reason"] = "no_bgm_vocals_dominant"
        elif (drums_rms + bass_rms + other_rms) < 0.015:
            has_bgm = False
            sub2_detail["reason"] = "no_bgm_low_music_rms"
    
    # PANNs music_presence as a cross-check
    if panns_result and has_bgm:
        music_presence = panns_result.get("music_presence", {})
        if music_presence.get("music", 0) < 0.1:
            has_bgm = False
            sub2_detail["reason"] = "no_bgm_panns_low_music"
    
    sub2_detail["has_bgm"] = has_bgm
    
    if has_bgm and panns_result:
        mood_tags = panns_result.get("mood_tags", {})
        if mood_tags:
            # Infer the expected mood from the prompt
            expected_moods = _infer_expected_mood_from_prompt(prompt)
            sub2_detail["expected_moods"] = expected_moods
            sub2_detail["detected_moods"] = mood_tags
            
            if expected_moods:
                # Match ratio: summed probability of the expected moods over the total mood probability
                mood_values = np.array(list(mood_tags.values()))
                total_mood_prob = float(np.sum(mood_values)) + 1e-8
                
                # Weighted score of the expected moods
                match_prob = sum(mood_tags.get(m, 0) for m in expected_moods)
                # Probability of the unwanted (conflicting) moods
                conflict_moods = _get_conflict_moods(expected_moods)
                conflict_prob = sum(mood_tags.get(m, 0) for m in conflict_moods)
                
                # Match = (expected - conflicting) / total probability, normalised to 0-1
                # When every mood probability is tiny, PANNs cannot tell: give a middling score
                if total_mood_prob < 0.001:
                    sub2_score = 0.5
                    sub2_detail["note"] = "mood_probs_too_low"
                else:
                    # Compute the match after a softmax normalisation
                    match_ratio = match_prob / total_mood_prob
                    conflict_ratio = conflict_prob / total_mood_prob
                    # score = match_ratio - 0.5*conflict_ratio, mapped to 0-1
                    raw_score = match_ratio - 0.5 * conflict_ratio
                    sub2_score = max(0.0, min(1.0, raw_score * 2.0 + 0.3))
                    sub2_detail["match_prob"] = round(match_prob, 4)
                    sub2_detail["conflict_prob"] = round(conflict_prob, 4)
                    sub2_detail["raw_score"] = round(raw_score, 4)
            else:
                # The mood cannot be inferred from the prompt: give a middling score
                sub2_score = 0.5
                sub2_detail["note"] = "cannot_infer_mood_from_prompt"
        else:
            sub2_score = 0.5
            sub2_detail["note"] = "no_mood_tags_available"
    elif not has_bgm:
        # Adjusted formula: no BGM no longer scores 0 outright, it falls back to 0.2
        sub2_score = 0.2
        sub2_detail["note"] = "no_bgm_penalty"
    else:
        sub2_score = 0.5
        sub2_detail["note"] = "panns_unavailable"
    
    # ===== Final B3 score =====
    final_score = (sub1_score + sub2_score) / 2.0
    
    return {
        "dimension": "B3",
        "metric": "rhythm_mood_matching(two sub-dimensions)",
        "score": max(0.0, min(1.0, final_score)),
        "sub1_rhythm": round(sub1_score, 4),
        "sub2_mood": round(sub2_score, 4),
        "sub1_detail": sub1_detail,
        "sub2_detail": sub2_detail,
        "music_energy": round(music_energy, 4),
        "drum_dominance": round(drum_dominance, 4),
        "avg_shot_duration": round(avg_shot_duration, 2),
    }


# Mood inference helpers
_SCENE_MOOD_MAP = {
    # Scene keyword -> list of expected moods
    "rain": ["sad", "tender"],
    "rainy": ["sad", "tender"],
    "storm": ["scary", "exciting"],
    "sunset": ["tender", "sad"],
    "sunrise": ["happy", "tender"],
    "wedding": ["happy", "tender"],
    "party": ["happy", "exciting", "funny"],
    "fight": ["angry", "exciting"],
    "battle": ["angry", "exciting", "scary"],
    "war": ["angry", "scary"],
    "horror": ["scary"],
    "chase": ["exciting", "scary"],
    "sport": ["exciting", "happy"],
    "dance": ["happy", "exciting"],
    "funeral": ["sad", "tender"],
    "romantic": ["tender", "happy"],
    "love": ["tender", "happy"],
    "peaceful": ["tender"],
    "calm": ["tender"],
    "quiet": ["tender", "sad"],
    "hushed": ["tender", "sad"],
    "energetic": ["exciting", "happy"],
    "fast": ["exciting"],
    "slow": ["tender", "sad"],
    "dark": ["scary", "sad"],
    "bright": ["happy"],
    "joy": ["happy", "funny"],
    "melanchol": ["sad", "tender"],
    "tense": ["scary", "exciting"],
    "suspense": ["scary", "exciting"],
    "comedy": ["funny", "happy"],
    "adventure": ["exciting", "happy"],
    "nature": ["tender"],
    "ocean": ["tender"],
    "forest": ["tender"],
    "city": ["exciting"],
    "night": ["scary", "tender"],
    "children": ["happy", "funny"],
    "book": ["tender"],
    "library": ["tender"],
    "bookstore": ["tender"],
}

# Conflicting mood pairs
_MOOD_CONFLICTS = {
    "happy": ["sad", "angry", "scary"],
    "funny": ["sad", "scary", "angry"],
    "sad": ["happy", "funny", "exciting"],
    "tender": ["angry", "exciting", "scary"],
    "exciting": ["sad", "tender"],
    "angry": ["happy", "funny", "tender"],
    "scary": ["happy", "funny"],
}


def _infer_expected_mood_from_prompt(prompt: dict) -> list:
    """Infer the expected music mood from the prompt's scene description."""
    import re
    if not prompt:
        return []
    
    # Concatenate all the text
    text_parts = []
    text_parts.append(prompt.get("title", ""))
    text_parts.append(prompt.get("overall_description_prompt", ""))
    for shot in prompt.get("shots", []):
        text_parts.append(shot.get("description_prompt", ""))
    
    full_text = " ".join(text_parts).lower()
    
    # Match scene keywords (on word boundaries, to avoid substring false positives)
    mood_scores = {}  # mood -> number of matches
    for keyword, moods in _SCENE_MOOD_MAP.items():
        # Use the \b word boundary for exact matching
        if re.search(r'\b' + re.escape(keyword) + r'\b', full_text):
            for m in moods:
                mood_scores[m] = mood_scores.get(m, 0) + 1
    
    if not mood_scores:
        return []
    
    # Return the top 2 moods by votes
    sorted_moods = sorted(mood_scores.items(), key=lambda x: -x[1])
    top_count = sorted_moods[0][1]
    result = [m for m, c in sorted_moods if c >= top_count * 0.5]
    return result[:3]  # at most 3


def _get_conflict_moods(expected_moods: list) -> list:
    """Moods that conflict with the expected ones."""
    conflicts = set()
    for m in expected_moods:
        for c in _MOOD_CONFLICTS.get(m, []):
            if c not in expected_moods:
                conflicts.add(c)
    return list(conflicts)


# ============================================================
# ============================================================
# Main Mode B Evaluation Function
# ============================================================
def evaluate_mode_b(video_path: str, prompt: dict, mode_a_raw_outputs: dict = None,
                    alignment_result: dict = None) -> dict:
    """
    Execute all Mode B dimension evaluations.
    E1 dimensions are evaluated PER-SHOT using VLM-aligned segmentation.

    Run every Mode B dimension. E1 is evaluated shot by shot, on the VLM-aligned shots.

    Args:
        video_path: path to video file
        prompt: prompt JSON configuration with GT data
        mode_a_raw_outputs: reuse Mode A expert model outputs
        alignment_result: VLM-assisted shot alignment result (optional)
    """
    results = {}
    raw = mode_a_raw_outputs or {}

    # ---- Step 1: Call expert services ----
    print("  [Mode B] Calling expert services...")

    # TransNetV2 (shot segmentation) - essential for per-shot E1
    try:
        if "transnetv2" not in raw:
            raw["transnetv2"] = call_transnetv2(video_path)
        n_shots = len(raw["transnetv2"].get("shots", []))
        print(f"    ✅ transnetv2: {n_shots} shots detected")
    except Exception as e:
        print(f"    ❌ transnetv2: {e}")
        raw["transnetv2"] = {}

    # MonST3R (E1-camera motion)
    try:
        if "monst3r" not in raw:
            raw["monst3r"] = call_service("monst3r", video_path,
                                          extra_data={"max_frames": "12", "target_fps": "3"})
        print(f"    ✅ monst3r: primary={raw['monst3r'].get('primary_motion', '?')}")
    except Exception as e:
        print(f"    ❌ monst3r: {e}")
        raw["monst3r"] = {}

    # RAFT (D1 whip-pan evidence)
    try:
        if "raft" not in raw:
            from service_client import call_raft
            raw["raft"] = call_raft(video_path, sample_fps=8.0)
        print(f"    ✅ raft")
    except Exception as e:
        print(f"    ❌ raft: {e}")
        raw["raft"] = {}

    # CLIP (C1, D1)
    try:
        if "clip" not in raw:
            raw["clip"] = call_service("clip", video_path)
        print(f"    ✅ clip")
    except Exception as e:
        print(f"    ❌ clip: {e}")
        raw["clip"] = {}

    # MovieShots (E1-shot scale)
    try:
        if "movieshots" not in raw:
            raw["movieshots"] = call_service("movieshots", video_path)
        print(f"    ✅ movieshots: {raw['movieshots'].get('dominant_scale', '?')}")
    except Exception as e:
        print(f"    ❌ movieshots: {e}")
        raw["movieshots"] = {}

    # YOLOv8
    try:
        if "yolov8" not in raw:
            raw["yolov8"] = call_service("yolov8", video_path)
        print(f"    ✅ yolov8")
    except Exception as e:
        print(f"    ❌ yolov8: {e}")
        raw["yolov8"] = {}

    # Saliency (E1-dof)
    try:
        if "saliency" not in raw:
            raw["saliency"] = call_service("saliency", video_path)
        print(f"    ✅ saliency")
    except Exception as e:
        print(f"    ❌ saliency: {e}")
        raw["saliency"] = {}

    # 6DRepNet (E1-angle)
    try:
        if "sixdrepnet" not in raw:
            raw["sixdrepnet"] = call_service("sixdrepnet", video_path)
        print(f"    ✅ sixdrepnet")
    except Exception as e:
        print(f"    ❌ sixdrepnet: {e}")
        raw["sixdrepnet"] = {}

    # Demucs (B3)
    try:
        if "demucs" not in raw:
            raw["demucs"] = call_service("demucs", video_path)
        print(f"    ✅ demucs: music_energy={sum(v.get('energy_ratio',0) for k,v in raw['demucs'].get('sources',{}).items() if k!='vocals'):.3f}")
    except Exception as e:
        print(f"    ❌ demucs: {e}")
        raw["demucs"] = {}

    # PANNs (B3) - supplies music_presence / mood_tags as rhythm-and-mood evidence
    try:
        if "panns" not in raw:
            raw["panns"] = call_service("panns", video_path,
                                        extra_data={"segment_duration": "2.0"})
        print(f"    ✅ panns (segments={len(raw['panns'].get('segment_embeddings', []))})")
    except Exception as e:
        print(f"    ❌ panns: {e}")
        raw["panns"] = {}

    # ---- Step 2: Get shot boundaries ----
    shot_boundaries = get_shot_boundaries(video_path, raw.get("transnetv2", {}))
    gt_shots = prompt.get("shots", [])
    n_detected = len(shot_boundaries)
    n_gt = len(gt_shots)
    print(f"  [Mode B] Shot alignment: detected={n_detected}, GT={n_gt}")

    # Use the aligned shot info when an alignment_result is available
    if alignment_result is not None:
        print(f"  [Mode B] Using VLM alignment: matched={alignment_result.get('n_matched', 0)}, "
              f"merged={alignment_result.get('n_merged', 0)}, "
              f"missing={alignment_result.get('n_missing', 0)}")

    # ---- Step 3: VLM evaluation ----
    print("  [Mode B] Calling VLM for each dimension...")

    # C1 Montage
    try:
        r = eval_c1_montage(video_path, prompt, raw.get("clip", {}),
                            raw.get("transnetv2", {}))
        results["C1"] = r
        print(f"    C1 montage: {r['score']:.1f} (pred={r['pred_type']})")
    except Exception as e:
        _raise_if_vlm_retry_exhausted(e)
        print(f"    C1 ERROR: {e}")
        results["C1"] = {"dimension": "C1", "score": 0.0, "error": str(e)}

    # D1 Transitions (Mode B types only)
    # Build the aligned transition info, to skip the transitions that cannot be evaluated
    aligned_transitions = None
    if alignment_result is not None:
        from vlm_shot_alignment import get_aligned_transitions
        aligned_transitions = get_aligned_transitions(alignment_result)

    d1_scores = []
    for i, shot in enumerate(gt_shots):
        if "transition_to_next" not in shot:
            continue

        transition = shot["transition_to_next"]
        ctype = transition.get("cinematographic_type", "")

        # Mode B transition types
        # Check the type first: non-Mode-B types are skipped (Mode A evaluates them with expert models)
        mode_b_types = [
            "Logical", "Causal", "POV", "Point-of-View",
            "Exit", "Entry", "Walk",
            "Occlusion", "Wipe-By", "Foreground",
            "Cutaway", "Empty", "Insert",
            "Gilligan", "Flag",
            "Whip", "Camera-Movement",
            "Extreme", "Polar",
        ]

        is_mode_b = any(t.lower() in ctype.lower() for t in mode_b_types)
        if not is_mode_b:
            continue

        # Alignment check: a missing neighbouring shot means this transition scores 0
        if aligned_transitions is not None and i < len(aligned_transitions):
            if not aligned_transitions[i]["evaluable"]:
                d1_scores.append(0.0)
                print(f"    D1 Shot{i+1}: 0.0 (adjacent shot missing)")
                continue

        try:
            r = eval_d1_transition(video_path, prompt, i, raw)
            if r:
                results[f"D1-{ctype[:25]}[{i}]"] = r
                d1_scores.append(r["score"])
                print(f"    D1 Shot{i+1}→{ctype[:30]}: {r['score']:.1f} ({r.get('vlm_answer', '?')})")
        except Exception as e:
            _raise_if_vlm_retry_exhausted(e)
            print(f"    D1 Shot{i+1} ERROR: {e}")

    if d1_scores:
        results["D1_mode_b_avg"] = {"score": float(np.mean(d1_scores)), "n_transitions": len(d1_scores)}

    # ---- E1 Per-Shot Evaluation ----
    e1_motion_scores = []
    e1_scale_scores = []
    e1_angle_scores = []
    e1_dof_scores = []
    e1_per_shot_details = []

    # Iterate over GT shots using alignment result for correct time ranges
    # Use the alignment result to get each GT shot's correct time range
    for shot_idx in range(n_gt):
        gt_shot = gt_shots[shot_idx]
        gt_camera = gt_shot.get("camera", {})

        # Determine the shot time range: prefer the alignment result
        if alignment_result is not None:
            from vlm_shot_alignment import get_first_segment_clip_range, _get_shot_boundaries_from_transnet
            # Get the TransNetV2 boundaries for get_first_segment_clip_range
            det_bounds = _get_shot_boundaries_from_transnet(raw.get("transnetv2", {}))
            clip_range = get_first_segment_clip_range(
                alignment_result, shot_idx, detected_boundaries=det_bounds)
            if clip_range is None:
                # This shot is missing -> every E1 sub-dimension scores 0
                e1_motion_scores.append(0.0)
                e1_scale_scores.append(0.0)
                e1_angle_scores.append(0.0)
                e1_dof_scores.append(0.0)
                zero_entry = {
                    "shot_idx": shot_idx, "status": "missing_zero",
                    "reason": f"Shot {shot_idx+1} missing → score 0"
                }
                e1_per_shot_details.append(zero_entry)
                print(f"    E1 Shot{shot_idx+1}: 0.0 (missing per alignment)")
                continue
            start_sec, end_sec = clip_range
        else:
            if shot_idx >= n_detected:
                # No corresponding detected shot → score 0
                e1_motion_scores.append(0.0)
                e1_scale_scores.append(0.0)
                e1_angle_scores.append(0.0)
                e1_dof_scores.append(0.0)
                zero_entry = {
                    "shot_idx": shot_idx, "status": "missing_zero",
                    "reason": f"Only {n_detected} shots detected, GT expects shot {shot_idx+1} → score 0"
                }
                e1_per_shot_details.append(zero_entry)
                print(f"    E1 Shot{shot_idx+1}: 0.0 (no detected shot)")
                continue
            start_sec, end_sec = shot_boundaries[shot_idx]

        # Extract shot clip
        shot_clip = extract_shot_clip(video_path, start_sec, end_sec)

        shot_detail = {"shot_idx": shot_idx, "status": "evaluated",
                       "time_range": f"{start_sec:.2f}-{end_sec:.2f}s"}

        # E1-Camera Motion
        try:
            gt_motion = gt_camera.get("camera_motion", "Static")
        # Also consider optical_motion
            optical = gt_camera.get("optical_motion", "None")
            if optical and optical != "None":
                gt_motion = optical  # Optical takes precedence if specified

            r = eval_e1_camera_motion_per_shot(shot_clip, shot_idx, gt_motion,
                                               raw.get("monst3r", {}))
            e1_motion_scores.append(r["score"])
            shot_detail["camera_motion"] = r
            print(f"    E1 Shot{shot_idx+1} motion: {r['score']:.1f} "
                  f"(pred={r['pred_motion']}, gt={r['gt_normalized']})")
        except Exception as e:
            _raise_if_vlm_retry_exhausted(e)
            print(f"    E1 Shot{shot_idx+1} motion ERROR: {e}")
            shot_detail["camera_motion"] = {"error": str(e)}

        # E1-Shot Scale
        try:
            gt_scale = gt_camera.get("shot_scale", "MS")
            r = eval_e1_shot_scale_per_shot(shot_clip, shot_idx, gt_scale,
                                             raw.get("movieshots", {}),
                                             raw.get("yolov8", {}))
            e1_scale_scores.append(r["score"])
            shot_detail["shot_scale"] = r
            print(f"    E1 Shot{shot_idx+1} scale: {r['score']:.1f} "
                  f"(pred={r['pred_scale']}, gt={r['gt_normalized']})")
        except Exception as e:
            _raise_if_vlm_retry_exhausted(e)
            print(f"    E1 Shot{shot_idx+1} scale ERROR: {e}")
            shot_detail["shot_scale"] = {"error": str(e)}

        # E1-Angle
        try:
            gt_angle = gt_camera.get("angle", "Eye-Level Shot")
            r = eval_e1_angle_per_shot(shot_clip, shot_idx, gt_angle,
                                        raw.get("sixdrepnet", {}))
            e1_angle_scores.append(r["score"])
            shot_detail["angle"] = r
            print(f"    E1 Shot{shot_idx+1} angle: {r['score']:.1f} "
                  f"(pred={r['pred_angle']}, gt={r['gt_normalized']})")
        except Exception as e:
            _raise_if_vlm_retry_exhausted(e)
            print(f"    E1 Shot{shot_idx+1} angle ERROR: {e}")
            shot_detail["angle"] = {"error": str(e)}

        # E1-DoF
        try:
            gt_dof = gt_camera.get("depth_of_field", "Standard Lens")
            r = eval_e1_dof_per_shot(shot_clip, shot_idx, gt_dof,
                                      raw.get("saliency", {}))
            e1_dof_scores.append(r["score"])
            shot_detail["dof"] = r
            print(f"    E1 Shot{shot_idx+1} dof: {r['score']:.1f} "
                  f"(pred={r['pred_dof']}, gt={r['gt_normalized']})")
        except Exception as e:
            _raise_if_vlm_retry_exhausted(e)
            print(f"    E1 Shot{shot_idx+1} dof ERROR: {e}")
            shot_detail["dof"] = {"error": str(e)}

        # Cleanup shot clip
        if os.path.exists(shot_clip) and shot_clip != video_path:
            os.unlink(shot_clip)

        e1_per_shot_details.append(shot_detail)

    # Aggregate E1 scores
    if e1_motion_scores:
        results["E1-camera_motion"] = {
            "dimension": "E1-camera_motion", "score": float(np.mean(e1_motion_scores)),
            "per_shot_scores": e1_motion_scores, "n_evaluated": len(e1_motion_scores),
        }
    if e1_scale_scores:
        results["E1-shot_scale"] = {
            "dimension": "E1-shot_scale", "score": float(np.mean(e1_scale_scores)),
            "per_shot_scores": e1_scale_scores, "n_evaluated": len(e1_scale_scores),
        }
    if e1_angle_scores:
        results["E1-angle"] = {
            "dimension": "E1-angle", "score": float(np.mean(e1_angle_scores)),
            "per_shot_scores": e1_angle_scores, "n_evaluated": len(e1_angle_scores),
        }
    if e1_dof_scores:
        results["E1-dof"] = {
            "dimension": "E1-dof", "score": float(np.mean(e1_dof_scores)),
            "per_shot_scores": e1_dof_scores, "n_evaluated": len(e1_dof_scores),
        }

    # B3 Rhythm-Mood Matching
    try:
        r = eval_b3_rhythm_mood(video_path, raw.get("demucs", {}),
                                raw.get("transnetv2", {}),
                                panns_result=raw.get("panns", {}),
                                prompt=prompt)
        results["B3"] = r
        print(f"    B3 rhythm-mood: {r['score']:.2f} (sub1={r.get('sub1_rhythm')}, sub2={r.get('sub2_mood')})")
    except Exception as e:
        _raise_if_vlm_retry_exhausted(e)
        print(f"    B3 ERROR: {e}")
        results["B3"] = {"dimension": "B3", "score": 0.5, "error": str(e)}

    # ---- Step 4: Aggregate scores ----
    scored_dims = {k: v.get("score", 0) for k, v in results.items()
                   if isinstance(v, dict) and "score" in v and v["score"] is not None}
    overall = float(np.mean(list(scored_dims.values()))) if scored_dims else 0.0

    return {
        "mode": "B",
        "video_path": video_path,
        "shot_alignment": {"detected": n_detected, "gt": n_gt},
        "e1_per_shot_details": e1_per_shot_details,
        "dimension_results": results,
        "dimension_scores": scored_dims,
        "overall_score": round(overall, 4),
        "expert_services_used": list(raw.keys()),
    }


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        video = _sys.argv[1]
    else:
        raise SystemExit("usage: python mode_b_eval.py <video.mp4>")
    # Single-file debug entry only: the prompt path comes solely from the injected EVAL_PROMPT_FILE
    # (see the config block at the top of benchmark/run_model_sequence_with_watchdog.sh).
    prompt_file = os.environ.get("EVAL_PROMPT_FILE")
    if not prompt_file:
        raise SystemExit("[CONFIG ERROR] export EVAL_PROMPT_FILE=/path/to/prompt.json first")

    with open(prompt_file) as f:
        prompts = json.load(f)

    prompt = prompts[0]
    result = evaluate_mode_b(video, prompt)
    print(json.dumps(result["dimension_scores"], indent=2, ensure_ascii=False))
