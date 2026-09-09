"""Qwen3.7 LLM client -- DashScope OpenAI-compatible endpoint.

- endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1
- auth: DASHSCOPE_API_KEY (comma separated for multi-key rotation)
- rotates keys on rate limits; sleeps and retries when every key is throttled
Modelled on the KeyRotatingClient pattern in prompt_processing/generate_prompt.py.
"""

from __future__ import annotations

import os
import time
from typing import Callable, List, Optional

import requests  # call the DashScope OpenAI-compatible endpoint directly, no openai dependency

RATE_LIMIT_SLEEP_SECONDS = 300
_RATE_KEYWORDS = [
    "超过了", "rate limit", "Rate limit", "429", "Too Many Requests",
    "请下一个周期再试", "次/", "quota", "Throttling", "limit_requests",
]


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(kw in msg for kw in _RATE_KEYWORDS)


class QwenClient:
    """DashScope Qwen3-compatible client with multi-key rotation and rate-limit retries (requests based)."""

    def __init__(self, keys: List[str], base_url: str, model: str,
                 temperature: float = 0.5, max_tokens: int = 1024):
        if not keys:
            raise ValueError("at least one API key is required")
        self.keys = keys
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._current_idx = 0

    @property
    def current_key(self) -> str:
        return self.keys[self._current_idx]

    def _key_hint(self, idx: int) -> str:
        return self.keys[idx][:8] + "..."

    def _call(self, fn: Callable):
        tried = 0
        while True:
            try:
                return fn(self.current_key)
            except Exception as e:  # noqa: BLE001
                if _is_rate_limit_error(e):
                    tried += 1
                    self._current_idx = (self._current_idx + 1) % len(self.keys)
                    if tried >= len(self.keys):
                        print(f"  [RATE_LIMIT] all {len(self.keys)} keys are rate limited, "
                              f"sleeping {RATE_LIMIT_SLEEP_SECONDS // 60} minutes...")
                        time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                        tried = 0
                    else:
                        print(f"  [RATE_LIMIT] switching key -> {self._key_hint(self._current_idx)}")
                else:
                    raise

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: Optional[int] = None) -> str:
        """One-shot chat returning the text content. max_tokens can be overridden per call (consistency analysis needs more)."""
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        def _fn(key: str) -> str:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            data = resp.json()
            if resp.status_code != 200 or data.get("error"):
                err = data.get("error", {}) if isinstance(data, dict) else {}
                raise RuntimeError(f"{resp.status_code} {err.get('message', data)}")
            return data["choices"][0]["message"]["content"] or ""
        return self._call(_fn).strip()


def create_qwen_client(llm_cfg: dict) -> Optional[QwenClient]:
    """Create the client from config.llm; returns None when enable=False or a key/dependency is missing (fallback path)."""
    if not llm_cfg or not llm_cfg.get("enable", True):
        return None
    keys_env = llm_cfg.get("keys_env", "DASHSCOPE_API_KEY")
    raw = os.environ.get(keys_env, "").strip()
    if not raw:
        print(f"  [LLM] environment variable {keys_env} is not set, falling back to template assembly")
        return None
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    try:
        return QwenClient(
            keys=keys,
            base_url=llm_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=llm_cfg.get("model", "qwen3.7"),
            temperature=llm_cfg.get("temperature", 0.5),
            max_tokens=llm_cfg.get("max_tokens", 1024),
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [LLM] failed to create the client({e}), falling back to template assembly")
        return None
