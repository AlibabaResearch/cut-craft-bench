"""Stage 4a -- WAN text-to-video adapter (wan2.7-t2v).

Reuses the call flow of data_generator/wan2.7/generate_video.py:
  create an async task -> poll -> download.
Wrapped as generate_t2v(), callable straight from the pipeline.

Notes:
  - prompt_extend is off by default (Qwen3.7 owns the detailed prompt; a second rewrite would override it)
  - duration must be an integer in [2,15] seconds
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("wan_t2v")

CREATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
QUERY_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"


def _headers(async_mode: bool = True) -> dict:
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key:
        raise EnvironmentError("environment variable DASHSCOPE_API_KEY is not set")
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if async_mode:
        h["X-DashScope-Async"] = "enable"
    return h


def create_t2v_task(
    prompt: str,
    duration: int = 5,
    ratio: str = "16:9",
    resolution: str = "720P",
    model_name: str = "wan2.7-t2v",
    negative_prompt: Optional[str] = None,
    audio_url: Optional[str] = None,
    prompt_extend: bool = False,
    watermark: bool = False,
    seed: Optional[int] = None,
) -> str:
    input_data = {"prompt": prompt}
    if negative_prompt:
        input_data["negative_prompt"] = negative_prompt
    if audio_url:
        input_data["audio_url"] = audio_url
    params = {
        "resolution": resolution,
        "ratio": ratio,
        "duration": duration,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if seed is not None:
        params["seed"] = seed
    body = {"model": model_name, "input": input_data, "parameters": params}

    message = f"[t2v] creating task | {model_name} | {duration}s | {resolution} {ratio} | audio_ref={'yes' if audio_url else 'no'}"
    logger.info(message)
    print(f"  {message}", flush=True)
    resp = requests.post(CREATE_URL, headers=_headers(True), json=body, timeout=60)
    result = resp.json()
    if result.get("code"):
        raise RuntimeError(f"WAN t2v creation failed: code={result['code']}, message={result.get('message')}")
    task_id = result["output"]["task_id"]
    print(f"  [t2v] task_id={task_id}", flush=True)
    return task_id


def poll_task(task_id: str, poll_interval: int = 15, timeout: int = 1800) -> dict:
    url = f"{QUERY_URL}/{task_id}"
    start = time.time()
    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"WAN task timed out: {task_id}")
        resp = requests.get(url, headers=_headers(False), timeout=30)
        resp.raise_for_status()
        output = resp.json().get("output", {})
        status = output.get("task_status", "UNKNOWN")
        message = f"[{time.time()-start:.0f}s] {status}"
        logger.info(f"  {message}")
        print(f"  [WAN] task_id={task_id} {message}", flush=True)
        if status == "SUCCEEDED":
            return output
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            raise RuntimeError(f"WAN task failed: {status} | {output.get('message', '')}")
        time.sleep(poll_interval)


def download(video_url: str, save_path: str) -> str:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(video_url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    message = f"saved: {save_path} ({Path(save_path).stat().st_size/1024/1024:.2f} MB)"
    logger.info(f"  {message}")
    print(f"  [WAN] {message}", flush=True)
    return save_path


def generate_t2v(
    prompt: str,
    save_path: str,
    duration: int = 5,
    ratio: str = "16:9",
    resolution: str = "720P",
    model_name: str = "wan2.7-t2v",
    negative_prompt: Optional[str] = None,
    audio_url: Optional[str] = None,
    prompt_extend: bool = False,
    seed: Optional[int] = None,
    poll_interval: int = 15,
    timeout: int = 1800,
) -> str:
    """Full flow: create -> poll -> download; returns the local file path."""
    task_id = create_t2v_task(
        prompt=prompt, duration=duration, ratio=ratio, resolution=resolution,
        model_name=model_name, negative_prompt=negative_prompt, audio_url=audio_url,
        prompt_extend=prompt_extend, seed=seed,
    )
    output = poll_task(task_id, poll_interval=poll_interval, timeout=timeout)
    video_url = output.get("video_url")
    if not video_url:
        raise RuntimeError("WAN t2v task finished but returned no video_url")
    return download(video_url, save_path)
