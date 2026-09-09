"""Frame utilities -- ffmpeg/ffprobe frame extraction and duration probing.

- extract_last_frame: last frame of a video (wipe-by transitions: first frame of the next shot)
- extract_first_frame: first frame of a video (spare)
- extract_frame_at: representative frame at a time ratio (R2V appearance reference, avoids first-frame framing lock)
- probe_duration: video duration in seconds via ffprobe
- has_audio_stream: whether the video carries an audio track
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional


def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_duration(video_path: str) -> Optional[float]:
    """Return the video duration in seconds; None on failure."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        return None
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:  # noqa: BLE001
        return None


def has_audio_stream(video_path: str) -> bool:
    """Whether the video carries an audio stream."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "json", video_path,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        return False
    try:
        return len(json.loads(r.stdout).get("streams", [])) > 0
    except Exception:  # noqa: BLE001
        return False


def extract_last_frame(video_path: str, out_image: str) -> Optional[str]:
    """Extract the last frame of a video into out_image (png); returns the path, or None on failure.

    Uses sseof to seek near the end, which reliably lands on the final frame.
    """
    Path(out_image).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
        "-update", "1", "-q:v", "2", out_image,
    ]
    r = _run(cmd)
    if r.returncode == 0 and Path(out_image).exists():
        return out_image
    # fallback: seek by duration
    dur = probe_duration(video_path)
    if dur:
        cmd2 = [
            "ffmpeg", "-y", "-ss", f"{max(dur - 0.1, 0):.3f}", "-i", video_path,
            "-update", "1", "-q:v", "2", out_image,
        ]
        r2 = _run(cmd2)
        if r2.returncode == 0 and Path(out_image).exists():
            return out_image
    print(f"  [FRAME] failed to extract the last frame: {video_path}\n{r.stderr[-300:]}")
    return None


def extract_first_frame(video_path: str, out_image: str) -> Optional[str]:
    """Extract the first frame of a video into out_image."""
    Path(out_image).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "select=eq(n\\,0)", "-frames:v", "1", "-q:v", "2", out_image,
    ]
    r = _run(cmd)
    if r.returncode == 0 and Path(out_image).exists():
        return out_image
    print(f"  [FRAME] failed to extract the first frame: {video_path}\n{r.stderr[-300:]}")
    return None


def extract_frame_at(video_path: str, out_image: str, ratio: float = 0.5) -> Optional[str]:
    """Extract a representative frame at a ratio (0~1) of the duration; falls back to the first frame."""
    Path(out_image).parent.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(video_path)
    if not dur or dur <= 0:
        return extract_first_frame(video_path, out_image)
    ratio = min(max(float(ratio), 0.0), 1.0)
    ts = min(max(dur * ratio, 0.05), max(dur - 0.05, 0.05))
    cmd = [
        "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path,
        "-frames:v", "1", "-q:v", "2", out_image,
    ]
    r = _run(cmd)
    if r.returncode == 0 and Path(out_image).exists():
        return out_image
    print(f"  [FRAME] failed to extract the representative frame, falling back to the first frame: {video_path}\n{r.stderr[-300:]}")
    return extract_first_frame(video_path, out_image)
