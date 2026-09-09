"""Audio reference utilities.

Extract a short audio clip from the first shot of a group and turn it into a Data URI that can
be passed to WAN `audio_url`. Used by the "first same-label shot's audio as reference" strategy.
"""

from __future__ import annotations

import base64
import mimetypes
import subprocess
from pathlib import Path
from typing import Optional

from .frame_utils import has_audio_stream


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def extract_reference_audio(
    video_path: str,
    output_path: str,
    max_seconds: int = 30,
    speech_only: bool = True,
) -> Optional[str]:
    """Extract reference audio from a video; returns output_path on success, None on failure.

    The WAN docs constrain audio_url to roughly 2-30s, so at most 30 seconds are kept by default.
    """
    if not has_audio_stream(video_path):
        print(f"  [AudioRef] the source video has no audio track, cannot extract a reference audio: {video_path}")
        return None

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-t", str(max_seconds),
    ]
    if speech_only:
        cmd.extend(["-af", "highpass=f=120,lowpass=f=7600,dynaudnorm"])
    cmd.extend([
        "-ac", "1" if speech_only else "2", "-ar", "16000" if speech_only else "44100",
        "-b:a", "96k" if speech_only else "192k",
        str(out),
    ])
    result = _run(cmd)
    if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
        return str(out)
    print(f"  [AudioRef] failed to extract the reference audio: {video_path}\n{result.stderr[-300:]}")
    return None


def audio_to_data_uri(audio_path: str) -> str:
    """Convert a local audio file to a data URI, the locally passable form of WAN's audio_url."""
    path = Path(audio_path)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "audio/mpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def public_audio_url(audio_path: str, use_data_uri: bool = True) -> Optional[str]:
    """Return an audio_url that can be passed to WAN.

    There is no object-storage upload path in this environment, so a data URI is used by default.
    With OSS/CDN available later, swap this for a public HTTP(S) URL.
    """
    if not audio_path:
        return None
    if use_data_uri:
        return audio_to_data_uri(audio_path)
    return audio_path if audio_path.startswith(("http://", "https://")) else None
