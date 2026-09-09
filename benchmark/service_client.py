"""
Unified microservice client -- calls the 8 expert model services.
Port mapping:
  TransNetV2: 8001  (shot boundary detection)
  RAFT:       8002  (optical flow)
  DINOv2:     8003  (visual embedding)
  Whisper:    8004  (speech recognition)
  Demucs:     8005  (source separation)
  E2Quality:  8006  (image quality: CLIP+LAION aesthetics & MUSIQ)
  PANNs:      8007  (audio embedding)
  DNSMOS:     8008  (audio signal quality analysis)
"""
import os
import time
import requests
from typing import Optional

BASE_URL = os.environ.get("SERVICE_BASE_URL", "http://localhost")

SERVICE_PORTS = {
    "transnetv2": 8001,
    "raft": 8002,
    "dinov2": 8003,
    "whisper": 8004,
    "demucs": 8005,
    "e2quality": 8006,
    "panns": 8007,
    "dnsmos": 8008,
    "clip": 8010,
}

# Request timeout (seconds)
TIMEOUT = 300


def get_service_url(service_name: str) -> str:
    port = SERVICE_PORTS[service_name]
    return f"{BASE_URL}:{port}"


def check_health(service_name: str) -> dict:
    """Check one service's health."""
    url = f"{get_service_url(service_name)}/health"
    try:
        resp = requests.get(url, timeout=10)
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_all_services() -> dict:
    """Check the health of every service."""
    results = {}
    for name in SERVICE_PORTS:
        results[name] = check_health(name)
    return results


# Service-down recovery wait configuration
SERVICE_DOWN_MAX_WAIT = 90    # max seconds to wait for a service to come back after it dies
SERVICE_DOWN_POLL_INTERVAL = 5  # polling interval while waiting for recovery (seconds)


def _is_service_down_error(e: Exception) -> bool:
    """Whether an exception means the service is completely down (process dead / port unreachable)."""
    # requests connection errors
    if isinstance(e, requests.exceptions.ConnectionError):
        return True
    # Connect timeout (as opposed to a read timeout)
    if isinstance(e, requests.exceptions.ConnectTimeout):
        return True
    # A RuntimeError carrying connection-related text
    err_str = str(e).lower()
    if any(kw in err_str for kw in ["connection refused", "connectionerror",
                                     "connect timeout", "no route to host",
                                     "name or service not known"]):
        return True
    return False


def _wait_for_service_recovery(service_name: str, max_wait: int = SERVICE_DOWN_MAX_WAIT) -> bool:
    """
    Wait for a service to recover (restarted by the watchdog).
    
    Polls the service's /health endpoint until it recovers or the wait times out.
    
    Returns:
        True = the service recovered, False = it timed out
    """
    url = f"{get_service_url(service_name)}/health"
    start = time.time()
    poll_count = 0
    
    print(f"  [Recovery] {service_name} appears DOWN. "
          f"Waiting up to {max_wait}s for watchdog to restart it...")
    
    while (time.time() - start) < max_wait:
        time.sleep(SERVICE_DOWN_POLL_INTERVAL)
        poll_count += 1
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                elapsed = time.time() - start
                print(f"  [Recovery] {service_name} is BACK after {elapsed:.1f}s "
                      f"({poll_count} polls). Resuming...")
                # Wait 2 more seconds for the service to settle
                time.sleep(2)
                return True
        except Exception:
            pass
        
        if poll_count % 4 == 0:  # print the status every 20 seconds
            elapsed = time.time() - start
            print(f"  [Recovery] Still waiting for {service_name}... "
                  f"({elapsed:.0f}s/{max_wait}s)")
    
    print(f"  [Recovery] {service_name} did NOT recover within {max_wait}s. Giving up.")
    return False


def call_service(service_name: str, video_path: str,
                 extra_params: Optional[dict] = None,
                 max_retries: int = 2) -> dict:
    """
    Call a microservice's /predict endpoint (with the service-down recovery wait).
    
    Retry strategy:
      Tier 1: fast retries for transient errors (3 attempts, 1s / 2s apart)
      Tier 2: on a detected full outage (ConnectionError), enter recovery-wait mode,
              polling /health until the watchdog restarts the service,
              then reissue the request.
    
    Args:
        service_name: service name
        video_path: path to the video / audio file
        extra_params: extra form parameters
        max_retries: maximum number of fast retries
    
    Returns:
        The JSON response the service returned
    """
    url = f"{get_service_url(service_name)}/predict"
    service_was_down = False
    
    for attempt in range(max_retries + 1):
        try:
            with open(video_path, "rb") as f:
                files = {"file": (os.path.basename(video_path), f, "video/mp4")}
                data = extra_params or {}
                resp = requests.post(url, files=files, data=data, timeout=TIMEOUT)
            
            if resp.status_code == 200:
                result = resp.json()
                if result.get("success", True):
                    return result
                else:
                    raise RuntimeError(f"Service {service_name} returned error: {result.get('error')}")
            else:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [Retry] {service_name} attempt {attempt+1} failed: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                # === Tier 2: wait for the service to recover ===
                # Every fast retry failing with a connection error means the service is down,
                # so wait for the watchdog restart and then try once more
                if _is_service_down_error(e) and not service_was_down:
                    service_was_down = True  # guard against infinite recursion
                    recovered = _wait_for_service_recovery(service_name)
                    if recovered:
                        # The service is back: reissue the request (last chance)
                        try:
                            with open(video_path, "rb") as f:
                                files = {"file": (os.path.basename(video_path), f, "video/mp4")}
                                data = extra_params or {}
                                resp = requests.post(url, files=files, data=data, timeout=TIMEOUT)
                            if resp.status_code == 200:
                                result = resp.json()
                                if result.get("success", True):
                                    return result
                                else:
                                    raise RuntimeError(
                                        f"Service {service_name} returned error after recovery: "
                                        f"{result.get('error')}")
                            else:
                                raise RuntimeError(
                                    f"HTTP {resp.status_code} after recovery: {resp.text[:200]}")
                        except Exception as e2:
                            raise RuntimeError(
                                f"Service {service_name} failed even after recovery: {e2}")
                    else:
                        raise RuntimeError(
                            f"Service {service_name} is DOWN and did not recover "
                            f"within {SERVICE_DOWN_MAX_WAIT}s: {e}")
                else:
                    raise RuntimeError(
                        f"Service {service_name} failed after {max_retries+1} attempts: {e}")


def call_transnetv2(video_path: str, threshold: float = 0.35, expected_shots: int = 0) -> dict:
    """TransNetV2 shot boundary detection (a lower threshold is more sensitive, default 0.35).
    
    Args:
        video_path: path to the video
        threshold: segmentation threshold
        expected_shots: GT shot count (default 0; > 0 triggers the semantic supplementary pass)
    """
    params = {"threshold": str(threshold)}
    if expected_shots > 0:
        params["expected_shots"] = str(expected_shots)
    return call_service("transnetv2", video_path, params)


def call_raft(video_path: str, sample_fps: float = 8.0, max_frames: int = 150) -> dict:
    """RAFT optical flow estimation.
    
    Args:
        video_path: path to the video
        sample_fps: sampling frame rate (default 8fps)
        max_frames: maximum frames to process (default 150, covering 15s@8fps=120 frames plus margin)
    """
    return call_service("raft", video_path, {
        "sample_fps": str(sample_fps),
        "max_frames": str(max_frames)
    })


def call_dinov2(video_path: str, sample_fps: float = 4.0, max_frames: int = 80) -> dict:
    """DINOv2 visual embedding.
    
    Args:
        video_path: path to the video
        sample_fps: sampling frame rate (default 4fps)
        max_frames: maximum frames to process (default 80, covering 15s@4fps=60 frames plus margin)
    """
    return call_service("dinov2", video_path, {
        "sample_fps": str(sample_fps),
        "max_frames": str(max_frames)
    })


def call_whisper(video_path: str, language: str = None,
                 word_timestamps: bool = False,
                 condition_on_previous_text: bool = True,
                 word_gap_threshold: float = 0.1) -> dict:
    """Whisper speech recognition; optionally returns word-level timestamps for the D3 sync judgement."""
    params = {
        "word_timestamps": str(word_timestamps).lower(),
        "condition_on_previous_text": str(condition_on_previous_text).lower(),
        "word_gap_threshold": str(word_gap_threshold),
    }
    if language:
        params["language"] = language
    return call_service("whisper", video_path, params)


def call_demucs(video_path: str) -> dict:
    """Demucs source separation."""
    return call_service("demucs", video_path)


def call_e2quality(video_path: str, num_frames: int = 32) -> dict:
    """E2 image quality (mean of CLIP+LAION aesthetics and MUSIQ)."""
    return call_service("e2quality", video_path, {"num_frames": str(num_frames)})


def call_clip_style_match(video_path: str, target_style: str,
                          candidate_styles: list, template: str = "a {} video",
                          max_retries: int = 2) -> dict:
    """A4 style match: calls the clip service /style_match (zero-shot style classification softmax)."""
    url = f"{get_service_url('clip')}/style_match"
    data = {
        "target_style": target_style,
        "candidate_styles": ",".join(candidate_styles),
        "template": template,
    }
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            with open(video_path, "rb") as f:
                files = {"file": (os.path.basename(video_path), f, "video/mp4")}
                resp = requests.post(url, files=files, data=data, timeout=TIMEOUT)
            if resp.status_code == 200:
                r = resp.json()
                if r.get("success", False):
                    return r
                raise RuntimeError(f"clip style_match error: {r.get('error')}")
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"clip style_match failed: {last_err}")


def call_panns(video_path: str, segment_duration: float = 0.0) -> dict:
    """PANNs audio embedding."""
    return call_service("panns", video_path, {"segment_duration": str(segment_duration)})


def call_dnsmos(video_path: str) -> dict:
    """Audio signal quality analysis (objective_quality_score, SNR, spectral richness)."""
    return call_service("dnsmos", video_path)
