"""Stage 4b -- WAN image-to-video / reference-to-video adapter.

Two kinds of visual reference:
  - i2v(first_frame): wipe-by transitions use the previous shot's last frame as first frame, so the picture is seamless across the wipe.
  - r2v(reference_image/reference_video): later shots of the same subject only reference the person/object appearance, without forcing the reference as first frame.

Important:
  - wan2.7-i2v uses the new image-to-video protocol: input.media=[{type:first_frame,url:...}].
  - wan2.7-r2v uses the reference-to-video protocol: input.media=[{type:reference_image,url:...}].
  - local reference images are passed through media.url as a base64 data URI.
  - if the account lacks i2v/r2v access or the call fails, the caller (pipeline) should fall back to t2v.

poll_task / download are reused from wan_t2v.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .wan_t2v import CREATE_URL, download, poll_task

logger = logging.getLogger("wan_i2v")


def _headers() -> dict:
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise EnvironmentError("environment variable DASHSCOPE_API_KEY is not set")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }


def _file_to_data_uri(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)
    mime = mime or "application/octet-stream"
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _image_to_data_uri(image_path: str) -> str:
    return _file_to_data_uri(image_path)


def _media_url(url_or_path: str) -> str:
    """HTTP(S) / data URIs pass through unchanged; local files become data URIs."""
    value = str(url_or_path or "")
    if value.startswith(("http://", "https://", "data:")):
        return value
    if not Path(value).exists():
        raise FileNotFoundError(f"reference media does not exist: {value}")
    return _file_to_data_uri(value)


def _resolve_create_url(create_url: Optional[str] = None) -> str:
    return create_url or os.getenv("DASHSCOPE_VIDEO_CREATE_URL") or CREATE_URL


def _uses_media_protocol(model_name: str) -> bool:
    """wan2.7-i2v uses the new media array protocol, the legacy i2v uses img_url."""
    return str(model_name or "").lower().startswith("wan2.7-i2v")


def create_i2v_task(
    prompt: str,
    first_frame_image: str,
    duration: int = 5,
    resolution: str = "720P",
    model_name: str = "wan2.7-i2v-2026-04-25",
    negative_prompt: Optional[str] = None,
    audio_url: Optional[str] = None,
    prompt_extend: bool = False,
    watermark: bool = False,
    seed: Optional[int] = None,
    duration_supported: bool = True,
) -> str:
    """Create an image-to-video task; first_frame_image is a local first-frame image path."""
    img_url = _image_to_data_uri(first_frame_image)
    if _uses_media_protocol(model_name):
        media = [{"type": "first_frame", "url": img_url}]
        if audio_url and audio_url.startswith(("http://", "https://")):
            media.append({"type": "driving_audio", "url": audio_url})
        input_data = {"prompt": prompt, "media": media}
    else:
        input_data = {"prompt": prompt, "img_url": img_url}
        if audio_url:
            input_data["audio_url"] = audio_url
    if negative_prompt:
        input_data["negative_prompt"] = negative_prompt
    params = {
        "resolution": resolution,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if duration_supported:
        params["duration"] = duration
    if seed is not None:
        params["seed"] = seed
    body = {"model": model_name, "input": input_data, "parameters": params}

    duration_tag = f"{duration}s" if duration_supported else "default-duration"
    message = f"[i2v] creating task | {model_name} | {duration_tag} | {resolution} | first_frame={Path(first_frame_image).name} | audio_ref={'yes' if audio_url else 'no'}"
    logger.info(message)
    print(f"  {message}", flush=True)
    resp = requests.post(_resolve_create_url(), headers=_headers(), json=body, timeout=120)
    result = resp.json()
    if result.get("code"):
        raise RuntimeError(f"WAN i2v creation failed: code={result['code']}, message={result.get('message')}")
    task_id = result["output"]["task_id"]
    print(f"  [i2v] task_id={task_id}", flush=True)
    return task_id


def generate_i2v(
    prompt: str,
    first_frame_image: str,
    save_path: str,
    duration: int = 5,
    resolution: str = "720P",
    model_name: str = "wan2.7-i2v-2026-04-25",
    negative_prompt: Optional[str] = None,
    audio_url: Optional[str] = None,
    prompt_extend: bool = False,
    seed: Optional[int] = None,
    duration_supported: bool = True,
    poll_interval: int = 15,
    timeout: int = 1800,
) -> str:
    """Full flow: create -> poll -> download; returns the local file path."""
    task_id = create_i2v_task(
        prompt=prompt, first_frame_image=first_frame_image, duration=duration,
        resolution=resolution, model_name=model_name, negative_prompt=negative_prompt,
        audio_url=audio_url, prompt_extend=prompt_extend, seed=seed,
        duration_supported=duration_supported,
    )
    output = poll_task(task_id, poll_interval=poll_interval, timeout=timeout)
    video_url = output.get("video_url")
    if not video_url:
        raise RuntimeError("WAN i2v task finished but returned no video_url")
    return download(video_url, save_path)


def create_r2v_task(
    prompt: str,
    reference_media: List[Dict[str, Any]],
    duration: int = 5,
    resolution: str = "720P",
    model_name: str = "wan2.7-r2v",
    negative_prompt: Optional[str] = None,
    prompt_extend: bool = False,
    watermark: bool = False,
    seed: Optional[int] = None,
    create_url: Optional[str] = None,
) -> str:
    """Create a reference-to-video task; reference_media uses reference_image/reference_video."""
    media = []
    for item in reference_media:
        media_type = str(item.get("type") or "reference_image")
        media_value = item.get("url") or item.get("path")
        if not media_value:
            raise ValueError(f"R2V reference media is missing url/path: {item}")
        media.append({"type": media_type, "url": _media_url(str(media_value))})
    if not media:
        raise ValueError("R2V requires at least one reference_image or reference_video")

    input_data: Dict[str, Any] = {"prompt": prompt, "media": media}
    if negative_prompt:
        input_data["negative_prompt"] = negative_prompt
    params = {
        "resolution": resolution,
        "duration": duration,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if seed is not None:
        params["seed"] = seed
    body = {"model": model_name, "input": input_data, "parameters": params}

    media_desc = ",".join(f"{m['type']}" for m in media)
    message = f"[r2v] creating task | {model_name} | {duration}s | {resolution} | refs={media_desc}"
    logger.info(message)
    print(f"  {message}", flush=True)
    resp = requests.post(_resolve_create_url(create_url), headers=_headers(), json=body, timeout=120)
    result = resp.json()
    if result.get("code"):
        raise RuntimeError(f"WAN r2v creation failed: code={result['code']}, message={result.get('message')}")
    task_id = result["output"]["task_id"]
    print(f"  [r2v] task_id={task_id}", flush=True)
    return task_id


def generate_r2v(
    prompt: str,
    reference_media: List[Dict[str, Any]],
    save_path: str,
    duration: int = 5,
    resolution: str = "720P",
    model_name: str = "wan2.7-r2v",
    negative_prompt: Optional[str] = None,
    prompt_extend: bool = False,
    seed: Optional[int] = None,
    poll_interval: int = 15,
    timeout: int = 1800,
    create_url: Optional[str] = None,
) -> str:
    """Full flow: create -> poll -> download; returns the local file path."""
    task_id = create_r2v_task(
        prompt=prompt, reference_media=reference_media, duration=duration,
        resolution=resolution, model_name=model_name, negative_prompt=negative_prompt,
        prompt_extend=prompt_extend, seed=seed, create_url=create_url,
    )
    output = poll_task(task_id, poll_interval=poll_interval, timeout=timeout)
    video_url = output.get("video_url")
    if not video_url:
        raise RuntimeError("WAN r2v task finished but returned no video_url")
    return download(video_url, save_path)
