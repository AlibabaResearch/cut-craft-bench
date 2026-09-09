"""Central planner -- transition repair decisions driven by rolling evaluation.

Role: after a cut is produced, run a rolling evaluation of B1 (transition timing) / D2 (transition
effect) / D3 (transition audio-visual relation); a below-threshold score triggers autonomous tool calls to locate the cause and emit structured repair actions.

Model: DashScope qwen3.7-plus (OpenAI-compatible endpoint + function calling).
When the LLM is unavailable or its output is invalid, a deterministic rule engine takes over, so the
agent still closes the repair loop without a key or under rate limits.

External tools the planner may call on its own:
  - run_transition_eval      : evaluate the current cut on a given dimension (direct to the offset-port expert services)
  - inspect_transition_audio : measure the head/tail sound of the clips on both sides of a transition (locates edge silence)
  - get_current_plan         : read the current per-shot net duration / effect duration / audio offset / sound description
  - propose_repairs          : terminal tool, submits the list of repair actions

Which repair actions require regenerating a shot:
  adjust_effect_duration  -> D2  only changes the effect duration; restitch only, no shot regenerated
  adjust_cut_timing       -> B1  nudges the cut time; only regenerates a shot when the material is too short
  adjust_audio_relation   -> D3  changes the audio offset and (optionally) the prompt sound description; a prompt change always regenerates
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from .transition_map import (
    JL_CUT_OFFSET_SECONDS,
    TRANSITION_EFFECT_SECONDS,
    map_audio_relation,
)

# ---- Repair parameter bounds (keep the LLM from proposing values that would break the cut) ----
MIN_NET_DURATION = 1.0
MAX_CUT_DELTA = 1.0          # max cut-time nudge per round (seconds)
MIN_EFFECT_DURATION = 0.25   # specified minimum 0.25s; anything shorter is inevitably judged a hard cut
MAX_EFFECT_DURATION = 0.40   # specified maximum 0.40s; anything longer is judged a blurry transition and eats into net duration
MIN_JL_OFFSET = 0.35         # below the D3 tail threshold (0.25s) plus margin, J/L cannot be detected
MAX_JL_OFFSET = 1.60

ACTION_TYPES = ("adjust_cut_timing", "adjust_effect_duration", "adjust_audio_relation")


# ============================================================
#  Planner LLM client (qwen3.7-plus + function calling)
# ============================================================

_RATE_KEYWORDS = [
    "超过了", "rate limit", "Rate limit", "429", "Too Many Requests",
    "请下一个周期再试", "次/", "quota", "Throttling", "limit_requests",
]

# "Quota exhausted" errors differ from plain rate limiting: rotating keys or exponential backoff
# cannot recover immediately, the quota has to reset. Skipping a repair round would lose the whole
# case's repair chance, so sleep a fixed 2 minutes and retry without consuming the normal retry budget.
_QUOTA_KEYWORDS = [
    "insufficient_quota", "exceeded your current quota", "Allocated quota exceeded",
    "AllocationQuotaExceeded", "budget", "Arrearage", "预算", "额度已用尽",
    "额度不足", "余额不足", "欠费",
]
# Wait time and max wait rounds after quota exhaustion (default at most 10 * 2min = 20min)
QUOTA_WAIT_SECONDS = float(os.environ.get("PLANNER_QUOTA_WAIT_SECONDS", "120"))
QUOTA_MAX_WAITS = int(os.environ.get("PLANNER_QUOTA_MAX_WAITS", "10"))


def _is_quota_exhausted_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw.lower() in msg for kw in _QUOTA_KEYWORDS)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(kw in msg for kw in _RATE_KEYWORDS)


class PlannerLLM:
    """DashScope client with tools (function calling) support, multi-key rotation and rate-limit retries."""

    def __init__(self, keys: List[str], base_url: str, model: str,
                 temperature: float = 0.2, max_tokens: int = 2048,
                 request_timeout: int = 600):
        if not keys:
            raise ValueError("at least one API key is required")
        self.keys = keys
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # qwen3.7-plus can take minutes on a long payload with many tool results, so the timeout must be generous
        self.request_timeout = request_timeout
        self._idx = 0

    def chat(self, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             max_retries: int = 6) -> Dict[str, Any]:
        """Return the assistant message (possibly with tool_calls).

        Quota exhaustion (insufficient_quota class) does not count as a retryable failure: rotate every
        key first, and if all are exhausted sleep QUOTA_WAIT_SECONDS (2 minutes by default) and continue,
        for at most QUOTA_MAX_WAITS rounds. Otherwise one quota error would drop the loop to the rule engine and lose a whole round of attribution.
        """
        url = f"{self.base_url}/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        tried_keys = 0
        quota_waits = 0
        quota_keys_tried = 0
        attempt = 0
        while attempt < max_retries:
            key = self.keys[self._idx]
            try:
                resp = requests.post(
                    url, headers={"Authorization": f"Bearer {key}",
                                  "Content-Type": "application/json"},
                    json=body, timeout=self.request_timeout,
                )
                data = resp.json()
                if resp.status_code != 200 or data.get("error"):
                    err = data.get("error", {}) if isinstance(data, dict) else {}
                    raise RuntimeError(f"{resp.status_code} {err.get('message', data)}")
                return data["choices"][0]["message"]
            except Exception as e:  # noqa: BLE001
                if _is_quota_exhausted_error(e):
                    # rotate the other keys first; only wait when every quota is exhausted
                    if len(self.keys) > 1 and quota_keys_tried < len(self.keys) - 1:
                        quota_keys_tried += 1
                        self._idx = (self._idx + 1) % len(self.keys)
                        print(f"  [Planner] quota exhausted, switching key -> {self.keys[self._idx][:8]}...")
                        continue
                    if quota_waits < QUOTA_MAX_WAITS:
                        quota_waits += 1
                        quota_keys_tried = 0
                        print(f"  [Planner] budget/quota exhausted({str(e)[:120]}), "
                              f"continuing after sleeping {QUOTA_WAIT_SECONDS:.0f}s "
                              f"({quota_waits}/{QUOTA_MAX_WAITS})")
                        time.sleep(QUOTA_WAIT_SECONDS)
                        continue
                    raise
                if _is_rate_limit_error(e) and len(self.keys) > 1:
                    tried_keys += 1
                    self._idx = (self._idx + 1) % len(self.keys)
                    if tried_keys < len(self.keys):
                        print(f"  [Planner] rate limited, switching key -> {self.keys[self._idx][:8]}...")
                        continue
                attempt += 1
                if attempt >= max_retries:
                    raise
                wait = min(120.0, 15.0 * (2 ** (attempt - 1)))
                print(f"  [Planner] LLM call failed({e}), retrying in {wait:.0f}s "
                      f"({attempt}/{max_retries})")
                time.sleep(wait)
        raise RuntimeError("planner LLM retries exhausted")


def create_planner_llm(planner_cfg: Dict[str, Any]) -> Optional[PlannerLLM]:
    cfg = planner_cfg or {}
    if not cfg.get("enable", True):
        print("  [Planner] LLM disabled, using the deterministic rule engine")
        return None
    keys_env = cfg.get("keys_env", "DASHSCOPE_API_KEY")
    raw = os.environ.get(keys_env, "").strip()
    if not raw:
        print(f"  [Planner] environment variable {keys_env} is not set, using the deterministic rule engine")
        return None
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    try:
        return PlannerLLM(
            keys=keys,
            base_url=cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=cfg.get("model", "qwen3.7-plus"),
            temperature=float(cfg.get("temperature", 0.2)),
            max_tokens=int(cfg.get("max_tokens", 2048)),
            request_timeout=int(cfg.get("request_timeout", 600)),
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [Planner] failed to create the LLM client({e}), using the deterministic rule engine")
        return None


# ============================================================
#  Tool definitions (the function schema exposed to the LLM)
# ============================================================

PLANNER_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_transition_eval",
            "description": ("对当前成片重新运行指定维度的评测（B1 转场时间 / D2 转场特效 / "
                            "D3 转场音画关系），返回分数、逐转场明细与失败清单。"
                            "首轮评测结果已在对话中给出，只有需要复核时才调用。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["B1", "D2", "D3"]},
                        "description": "要评测的维度，缺省为全部三个",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_transition_audio",
            "description": ("体检某个转场两侧分镜 clip 的首尾声音：返回整体/首窗/尾窗 dBFS、"
                            "0.25s 粒度能量曲线、首个与最后一个有声时刻，以及可执行结论。"
                            "用于判断 D3 失败是因为 clip 首尾静音（需改 prompt 声音描述）"
                            "还是因为音轨偏置时间不足（需改 timing_offset）。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "transition_index": {
                        "type": "integer",
                        "description": "转场序号，0 表示第 1 镜与第 2 镜之间的切点",
                    },
                },
                "required": ["transition_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_plan",
            "description": ("读取当前生成规划：每镜净时长/生成时长/实际 clip 时长，"
                            "以及每个转场的特效类型、渲染特效时长、音画关系、音轨偏置时间、"
                            "prompt 中的声音描述片段。"),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_repairs",
            "description": ("提交本轮修复动作清单并结束分析。每个动作必须精确对应一个失败转场，"
                            "且遵守：D2 问题只用 adjust_effect_duration（不重新生成镜头）；"
                            "B1 问题用 adjust_cut_timing；D3 问题用 adjust_audio_relation。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "整体诊断结论（简洁）"},
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": list(ACTION_TYPES)},
                                "target_dimension": {"type": "string", "enum": ["B1", "D2", "D3"]},
                                "transition_index": {
                                    "type": "integer",
                                    "description": "目标转场序号（adjust_effect_duration / "
                                                   "adjust_audio_relation 必填）",
                                },
                                "shot_id": {
                                    "type": "integer",
                                    "description": "目标镜头 shot_id（adjust_cut_timing 必填）",
                                },
                                "delta_seconds": {
                                    "type": "number",
                                    "description": f"adjust_cut_timing: 该镜净时长调整量，"
                                                   f"绝对值 <= {MAX_CUT_DELTA}s，"
                                                   f"相邻镜自动反向补偿以保持成片总时长不变",
                                },
                                "transition_duration_seconds": {
                                    "type": "number",
                                    "description": f"adjust_effect_duration: 新的特效渲染时长，"
                                                   f"范围 [{MIN_EFFECT_DURATION}, {MAX_EFFECT_DURATION}]s，"
                                                   f"默认 {TRANSITION_EFFECT_SECONDS}s。"
                                                   f"被判成直切说明太短要加长，未被检出/被判成模糊过渡说明要缩短",
                                },
                                "timing_offset_seconds": {
                                    "type": "number",
                                    "description": f"adjust_audio_relation: 新的 J/L-cut 音轨偏置时间，"
                                                   f"范围 [{MIN_JL_OFFSET}, {MAX_JL_OFFSET}]",
                                },
                                "sound_prompt_addition": {
                                    "type": "string",
                                    "description": "adjust_audio_relation: 追加到目标镜头 prompt 的"
                                                   "声音描述（英文），用于修正 clip 首尾静音。"
                                                   "填了本字段就会重新生成对应镜头",
                                },
                                "sound_prompt_target_shot_ids": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "声音描述要追加到哪些 shot_id（通常是该转场的"
                                                   "outgoing 镜和/或 incoming 镜）",
                                },
                                "reason": {"type": "string", "description": "该动作的依据"},
                            },
                            "required": ["type", "target_dimension", "reason"],
                        },
                    },
                },
                "required": ["actions"],
            },
        },
    },
]


_SYSTEM_PROMPT = f"""你是专业剪辑 agent 的中央规划器，负责在成片生成后做「滚动评测 -> 归因 -> 修复」。

评测三个维度与通过阈值：
- B1 转场时间：阈值 0.90。检测到的镜头时长与指令时长的相对误差；失败说明感知到的切镜时刻偏离了指令时刻。
- D2 转场特效：阈值 0.70。溶解/擦除/闪白/闪黑/硬切的五分类命中率。
- D3 转场音画关系：阈值 0.50。J-cut(声先画后)/L-cut(画先声后)/straight(直切) 判定命中率。

当前统一渲染参数：默认特效时长 {TRANSITION_EFFECT_SECONDS}s（可调范围 {MIN_EFFECT_DURATION}s~{MAX_EFFECT_DURATION}s），J/L-cut 音轨偏置 {JL_CUT_OFFSET_SECONDS}s。

归因与修复规则（必须严格遵守）：
1. D2 失败 -> 只调特效时长（adjust_effect_duration），绝对不要重新生成镜头。
   - pred_type=hard_cut 但 GT 是渐变特效：特效切得太快被判成直切，应【加长】特效时长；
   - 没检测到转场(no_pred) 或被判成其它渐变类型：可能太长导致过渡模糊，应【缩短】特效时长；
   - 闪白/闪黑被漏检：适度加长。
2. B1 失败 -> 用 adjust_cut_timing 微调切镜时间。评测里的 cut_offset_seconds 是实测切点减去
   计划切点：正值说明切晚了，应把该镜净时长调小；负值说明切早了，应调大。
   delta_seconds 取 -cut_offset_seconds 的量级即可，不要过度修正。
3. D3 失败 -> 用 adjust_audio_relation。必须先调用 inspect_transition_audio 查看两侧 clip 的
   首尾声音，再决定：
   - outgoing clip 尾部静音（L-cut 无声可延续）或 incoming clip 开头静音（J-cut 无声可提前）：
     必须给 sound_prompt_addition，要求该镜在对应位置产生明确的画面内声音；
   - incoming clip 声音起始时刻晚于当前偏置：把 timing_offset_seconds 增大到「首个有声时刻 + 0.4s」；
   - 被判成 straight 但 GT 是 J/L：增大偏置；被判成 J/L 但 GT 是 straight：减小偏置到 {MIN_JL_OFFSET}。
4. 一个失败转场最多给一个动作。不要给已经通过阈值维度的动作。
5. 分析完成后必须调用 propose_repairs 提交动作清单结束；没有可靠修复手段时提交空 actions。
"""


# ============================================================
#  Planner core
# ============================================================

class CentralPlanner:
    """Central planner with tool-calling capability."""

    def __init__(self, llm: Optional[PlannerLLM], tools: Dict[str, Callable[..., Any]],
                 max_tool_rounds: int = 8):
        self.llm = llm
        self.tools = tools
        self.max_tool_rounds = max_tool_rounds
        self.transcript: List[Dict[str, Any]] = []

    def plan_repairs(self, eval_result: Dict[str, Any], plan_snapshot: Dict[str, Any],
                     round_index: int, max_rounds: int) -> Dict[str, Any]:
        """Return {"reasoning", "actions", "source", "tool_calls"}."""
        failed = eval_result.get("failed_dimensions") or []
        if not failed:
            return {"reasoning": "every dimension already passed", "actions": [], "source": "no_repair_needed",
                    "tool_calls": []}

        if self.llm is None:
            return fallback_repair_rules(eval_result, plan_snapshot)

        try:
            return self._plan_with_llm(eval_result, plan_snapshot, round_index, max_rounds)
        except Exception as e:  # noqa: BLE001
            print(f"  [Planner] LLM planning failed({e}), falling back to the deterministic rule engine")
            result = fallback_repair_rules(eval_result, plan_snapshot)
            result["llm_error"] = str(e)[:300]
            return result

    def _plan_with_llm(self, eval_result: Dict[str, Any], plan_snapshot: Dict[str, Any],
                       round_index: int, max_rounds: int) -> Dict[str, Any]:
        user_payload = {
            "repair_round": round_index,
            "max_repair_rounds": max_rounds,
            "scores": eval_result.get("scores"),
            "thresholds": eval_result.get("thresholds"),
            "passed": eval_result.get("passed"),
            "failed_dimensions": eval_result.get("failed_dimensions"),
            "failed_transitions": eval_result.get("failed_transitions"),
            "n_planned_shots": eval_result.get("n_planned_shots"),
            "n_detected_shots": eval_result.get("n_detected_shots"),
            "b1_per_transition": ((eval_result.get("dimensions", {}).get("B1") or {})
                                  .get("per_transition")),
            "b1_per_shot": ((eval_result.get("dimensions", {}).get("B1") or {})
                            .get("per_shot")),
            "d2_per_transition": ((eval_result.get("dimensions", {}).get("D2") or {})
                                  .get("per_transition")),
            "d3_per_transition": [
                {k: v for k, v in t.items() if k != "signal_gate"}
                for t in ((eval_result.get("dimensions", {}).get("D3") or {})
                          .get("per_transition") or [])
            ],
            "current_plan": plan_snapshot,
        }
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content":
                "这是当前成片的滚动评测结果与生成规划，请归因并提交修复动作：\n"
                + json.dumps(user_payload, ensure_ascii=False)[:14000]},
        ]

        tool_calls_log: List[Dict[str, Any]] = []
        for _ in range(self.max_tool_rounds):
            msg = self.llm.chat(messages, tools=PLANNER_TOOLS)
            tool_calls = msg.get("tool_calls") or []
            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            })
            if not tool_calls:
                # no tool call and no submission -> try to salvage actions from the free text
                parsed = _extract_actions_from_text(msg.get("content") or "")
                if parsed is not None:
                    return {"reasoning": (msg.get("content") or "")[:800],
                            "actions": parsed, "source": "llm_text",
                            "tool_calls": tool_calls_log}
                messages.append({"role": "user", "content":
                                 "请调用 propose_repairs 工具提交修复动作清单。"})
                continue

            for call in tool_calls:
                fn = (call.get("function") or {})
                name = fn.get("name")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls_log.append({"name": name, "arguments": args})

                if name == "propose_repairs":
                    return {
                        "reasoning": str(args.get("reasoning", ""))[:1000],
                        "actions": args.get("actions") or [],
                        "source": "llm_tool_call",
                        "tool_calls": tool_calls_log,
                    }

                handler = self.tools.get(name)
                if handler is None:
                    result: Any = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        result = handler(**args)
                    except Exception as e:  # noqa: BLE001
                        result = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                print(f"  [Planner] tool call {name}({json.dumps(args, ensure_ascii=False)[:160]})")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(result, ensure_ascii=False)[:5000],
                })

        print("  [Planner] tool call rounds exhausted without a submission, falling back to the deterministic rule engine")
        result = fallback_repair_rules(eval_result, plan_snapshot)
        result["tool_calls"] = tool_calls_log
        return result


def _extract_actions_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last <= first:
        return None
    try:
        obj = json.loads(text[first:last + 1])
    except json.JSONDecodeError:
        return None
    actions = obj.get("actions")
    return actions if isinstance(actions, list) else None


# ============================================================
#  Deterministic rule engine (fallback when the LLM is unavailable or its output is invalid)
# ============================================================

def fallback_repair_rules(eval_result: Dict[str, Any],
                          plan_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-free repair rules, matching the attribution rules in the system prompt."""
    actions: List[Dict[str, Any]] = []
    notes: List[str] = []
    shots = plan_snapshot.get("shots", [])

    def shot_by_index(idx: int) -> Dict[str, Any]:
        return shots[idx] if 0 <= idx < len(shots) else {}

    for ft in eval_result.get("failed_transitions", []):
        idx = int(ft["transition_index"])
        dims = {i["dimension"] for i in ft["issues"]}
        issues = {i["dimension"]: i for i in ft["issues"]}

        # D2: only change the effect duration
        if "D2" in dims:
            issue = issues["D2"]
            current = issue.get("rendered_effect_seconds")
            current = float(current) if current else TRANSITION_EFFECT_SECONDS
            if issue.get("pred_type") == "hard_cut":
                new_value = min(MAX_EFFECT_DURATION, round(current + 0.15, 3))
                reason = "the effect was judged a hard cut, lengthen it so the gradual transition is detected"
            elif not issue.get("pred_type"):
                new_value = max(MIN_EFFECT_DURATION, round(current - 0.08, 3))
                reason = "this cut point was not detected as a transition, shorten the effect to sharpen it"
            else:
                new_value = min(MAX_EFFECT_DURATION, round(current + 0.10, 3))
                reason = f"the effect was misjudged as {issue.get('pred_type')}, nudging the effect duration"
            # a new value equal to the current one means the bound is reached, so skip the no-op
            if abs(new_value - current) < 0.01:
                notes.append(f"transition {idx}: the D2 effect duration is already at the bound {current}s, "
                             f"no further adjustment is possible within [{MIN_EFFECT_DURATION}, {MAX_EFFECT_DURATION}]")
                continue
            actions.append({
                "type": "adjust_effect_duration", "target_dimension": "D2",
                "transition_index": idx, "transition_duration_seconds": new_value,
                "reason": reason,
            })
            continue

        # D3: change the audio offset (+ the prompt sound description when the clip edges are silent)
        if "D3" in dims:
            issue = issues["D3"]
            gt = issue.get("gt_relation")
            pred = issue.get("predicted_relation")
            current = issue.get("rendered_timing_offset_seconds")
            current = float(current) if current else JL_CUT_OFFSET_SECONDS
            action: Dict[str, Any] = {
                "type": "adjust_audio_relation", "target_dimension": "D3",
                "transition_index": idx,
            }
            if gt in ("j-cut", "l-cut") and pred in ("straight", "unclear", "overlap-unclear"):
                action["timing_offset_seconds"] = min(MAX_JL_OFFSET, round(current + 0.4, 3))
                target_shot = (shot_by_index(idx + 1) if gt == "j-cut" else shot_by_index(idx))
                position = "the very beginning" if gt == "j-cut" else "the final moments"
                action["sound_prompt_addition"] = (
                    f"Audio timing requirement for this shot: from {position} of the shot, "
                    f"produce a clear, continuously audible diegetic sound owned by this shot "
                    f"(on-screen speech, Foley, or physical action sound) that lasts at least "
                    f"1.2 seconds without silence, so it can be perceived across the adjacent cut."
                )
                action["sound_prompt_target_shot_ids"] = (
                    [target_shot.get("shot_id")] if target_shot.get("shot_id") is not None else [])
                action["reason"] = (f"GT is {gt} but it was judged {pred}, increasing the audio offset and requiring "
                                   f"an identifiable diegetic sound at the {'beginning' if gt == 'j-cut' else 'end'} of this shot")
            elif gt == "straight" and pred in ("j-cut", "l-cut", "overlap-unclear"):
                action["timing_offset_seconds"] = MIN_JL_OFFSET
                action["reason"] = f"GT is a straight cut but it was judged {pred}, tightening the audio offset to sync sound with picture"
            else:
                action["timing_offset_seconds"] = min(MAX_JL_OFFSET, round(current + 0.25, 3))
                action["reason"] = f"GT={gt} / pred={pred}, slightly increasing the audio offset to strengthen the audio-visual displacement evidence"
            actions.append(action)
            continue

        # B1: nudge the cut time by its drift (each cut point is fixed once, so a shared cut is not corrected twice)
        if "B1" in dims:
            issue = issues["B1"]
            if not issue.get("cut_matched"):
                notes.append(f"transition {idx}: the cut point was not detected, cannot nudge the cut timing")
                continue
            delta = float(issue.get("suggested_delta_seconds") or 0)
            if abs(delta) < 0.08:
                notes.append(f"transition {idx}: cut drift {delta:.3f}s is too small, skipped")
                continue
            delta = max(-MAX_CUT_DELTA, min(MAX_CUT_DELTA, round(delta, 3)))
            # this cut point is the outgoing shot's out point, so changing its net duration shifts the cut
            target_shot_id = issue.get("outgoing_shot_id")
            if target_shot_id is None:
                target_shot_id = shot_by_index(idx).get("shot_id")
            actions.append({
                "type": "adjust_cut_timing", "target_dimension": "B1",
                "shot_id": target_shot_id, "delta_seconds": delta,
                "reason": (f"cut {idx} measured {issue.get('cut_time')}s vs planned "
                           f"{issue.get('planned_cut')}s (drift "
                           f"{issue.get('cut_offset_seconds'):+.3f}s), "
                           f"shifting this cut by {delta:+.3f}s"),
            })

    return {
        "reasoning": "deterministic rule engine attribution: " + ("; ".join(notes) if notes else "repair actions generated per dimension and per transition"),
        "actions": actions,
        "source": "deterministic_rules",
        "tool_calls": [],
    }


# ============================================================
#  Repair action validation and application
# ============================================================

def _recompute_shot_timing(decisions: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Recompute each shot's headroom and gen_duration from the modified net_duration / transition_out.

    Same formulas as generation_planner.plan_generation:
      overlap_out          = effect duration of a dissolve/wipe (0 for flash white/black and hard cuts)
      audio_extension      = incoming_j + outgoing_l (+0.5s guard band)
      required_clip_seconds= net + max(overlap_out, audio_extension) (lower bound needed by stitching)
      gen_duration         = clamp(ceil(required + audio_headroom), [min, max])
    """
    wan_min = int(cfg.get("wan_min_duration", 2))
    wan_max = int(cfg.get("wan_max_duration", 15))
    headroom = max(0.0, float((cfg.get("audio") or {}).get("headroom_seconds", 1.0)))
    for idx, dec in enumerate(decisions):
        trans = dec.get("transition_out") or None
        prev_trans = decisions[idx - 1].get("transition_out") if idx > 0 else None

        overlap_out = 0.0
        if trans:
            compose = trans.get("compose") or {}
            effect_dur = float(trans.get("transition_duration_seconds") or 0)
            if (compose.get("mode") == "xfade"
                    and not trans.get("occlusion_first_frame_handled")
                    and (compose.get("xfade") or "") not in ("fadewhite", "fadeblack")):
                overlap_out = effect_dur
            trans["overlap_seconds"] = overlap_out

        incoming_j = 0.0
        if prev_trans and map_audio_relation(
                prev_trans.get("audio_visual_relation") or prev_trans.get("audio_relation")) == "j-cut":
            incoming_j = max(0.0, float(prev_trans.get("timing_offset_seconds") or 0))
        outgoing_l = 0.0
        outgoing_offset = 0.0
        if trans:
            outgoing_offset = max(0.0, float(trans.get("timing_offset_seconds") or 0))
            if map_audio_relation(trans.get("audio_visual_relation")
                                  or trans.get("audio_relation")) == "l-cut":
                outgoing_l = outgoing_offset

        audio_extension = max(incoming_j + outgoing_l, outgoing_offset)
        audio_extension_with_guard = audio_extension + 0.5 if audio_extension > 0 else 0.0
        net = float(dec.get("net_duration", dec.get("duration", 0)) or 0)
        required = net + max(overlap_out, audio_extension_with_guard)
        # keep the same audio headroom so a regenerated shot also has material for later offset tuning
        gen_duration = min(int(math.ceil(required + headroom)), wan_max)
        gen_duration = max(gen_duration, wan_min)

        dec["overlap_out_seconds"] = overlap_out
        dec["incoming_j_offset_seconds"] = incoming_j
        dec["outgoing_l_offset_seconds"] = outgoing_l
        dec["audio_extension_seconds"] = audio_extension_with_guard
        dec["audio_headroom_seconds"] = headroom
        dec["gen_duration"] = gen_duration
        dec["required_clip_seconds"] = round(required, 4)


def _footage_sufficient(dec: Dict[str, Any], clip_duration: Optional[float]) -> bool:
    """Whether the already generated clip is still long enough (long enough means restitch only, no regeneration)."""
    if clip_duration is None or clip_duration <= 0:
        return False
    required = float(dec.get("required_clip_seconds") or 0)
    return clip_duration + 0.05 >= required


def apply_repair_actions(decisions: List[Dict[str, Any]], actions: List[Dict[str, Any]],
                         cfg: Dict[str, Any],
                         clip_durations: Optional[Dict[Any, float]] = None
                         ) -> Dict[str, Any]:
    """Apply the repair actions to decisions (in place) and return the outcome.

    Returns {"applied": [...], "rejected": [...], "regen_shot_ids": [...], "recompose": bool}
    """
    clip_durations = clip_durations or {}
    applied: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    regen_shot_ids: set = set()
    by_shot_id = {d["shot_id"]: d for d in decisions}
    target_duration = float(cfg.get("target_duration", 15))

    for action in actions or []:
        atype = str(action.get("type", ""))
        if atype not in ACTION_TYPES:
            rejected.append({"action": action, "reason": f"unknown action type: {atype}"})
            continue

        if atype == "adjust_effect_duration":
            idx = action.get("transition_index")
            if idx is None or not (0 <= int(idx) < len(decisions) - 1):
                rejected.append({"action": action, "reason": "transition_index out of range"})
                continue
            dec = decisions[int(idx)]
            trans = dec.get("transition_out")
            if not trans or (trans.get("compose") or {}).get("mode") != "xfade":
                rejected.append({"action": action,
                                 "reason": "this transition is a hard cut / wipe-by, it has no adjustable effect duration"})
                continue
            try:
                value = float(action.get("transition_duration_seconds"))
            except (TypeError, ValueError):
                rejected.append({"action": action, "reason": "invalid transition_duration_seconds"})
                continue
            value = round(max(MIN_EFFECT_DURATION, min(MAX_EFFECT_DURATION, value)), 3)
            old = trans.get("transition_duration_seconds")
            trans["transition_duration_seconds"] = value
            trans["effect_duration_policy"] = f"planner_repair_{value:g}s"
            applied.append({**action, "old_value": old, "new_value": value,
                            "requires_regeneration": False})
            continue

        if atype == "adjust_audio_relation":
            idx = action.get("transition_index")
            if idx is None or not (0 <= int(idx) < len(decisions) - 1):
                rejected.append({"action": action, "reason": "transition_index out of range"})
                continue
            idx = int(idx)
            dec = decisions[idx]
            trans = dec.get("transition_out")
            if not trans:
                rejected.append({"action": action, "reason": "this shot has no transition_out"})
                continue
            record: Dict[str, Any] = {**action}
            relation = map_audio_relation(trans.get("audio_visual_relation")
                                          or trans.get("audio_relation"))
            if action.get("timing_offset_seconds") is not None:
                if relation == "straight":
                    record["offset_skipped"] = "GT is a straight cut, the offset is pinned to 0 and left unchanged"
                else:
                    try:
                        offset = float(action["timing_offset_seconds"])
                    except (TypeError, ValueError):
                        offset = JL_CUT_OFFSET_SECONDS
                    offset = round(max(MIN_JL_OFFSET, min(MAX_JL_OFFSET, offset)), 3)
                    record["old_timing_offset_seconds"] = trans.get("timing_offset_seconds")
                    trans["timing_offset_seconds"] = offset
                    trans["jl_offset_policy"] = f"planner_repair_{offset:g}s"
                    record["new_timing_offset_seconds"] = offset

            addition = (action.get("sound_prompt_addition") or "").strip()
            touched_shots: List[Any] = []
            if addition:
                targets = action.get("sound_prompt_target_shot_ids") or []
                if not targets:
                    # unspecified: infer from the audio-visual relation, a J-cut fixes the incoming shot, an L-cut the outgoing one
                    targets = ([decisions[idx + 1]["shot_id"]] if relation == "j-cut"
                               else [dec["shot_id"]])
                for shot_id in targets:
                    target_dec = by_shot_id.get(shot_id)
                    if target_dec is None:
                        rejected.append({"action": action,
                                         "reason": f"sound_prompt target shot {shot_id} does not exist"})
                        continue
                    target_dec["prompt"] = f"{target_dec['prompt']}\n\n{addition}"
                    target_dec.setdefault("repair_prompt_additions", []).append(addition)
                    regen_shot_ids.add(shot_id)
                    touched_shots.append(shot_id)
            record["prompt_updated_shot_ids"] = touched_shots
            record["requires_regeneration"] = bool(touched_shots)
            applied.append(record)
            continue

        # adjust_cut_timing
        shot_id = action.get("shot_id")
        dec = by_shot_id.get(shot_id)
        if dec is None:
            rejected.append({"action": action, "reason": f"shot_id {shot_id} does not exist"})
            continue
        try:
            delta = float(action.get("delta_seconds"))
        except (TypeError, ValueError):
            rejected.append({"action": action, "reason": "invalid delta_seconds"})
            continue
        delta = max(-MAX_CUT_DELTA, min(MAX_CUT_DELTA, delta))
        pos = decisions.index(dec)
        neighbor = decisions[pos + 1] if pos + 1 < len(decisions) else (
            decisions[pos - 1] if pos > 0 else None)
        if neighbor is None:
            rejected.append({"action": action, "reason": "only one shot, the total duration cannot be compensated"})
            continue
        old_net = float(dec.get("net_duration", dec.get("duration", 0)) or 0)
        old_neighbor = float(neighbor.get("net_duration", neighbor.get("duration", 0)) or 0)
        new_net = round(old_net + delta, 3)
        new_neighbor = round(old_neighbor - delta, 3)
        if new_net < MIN_NET_DURATION or new_neighbor < MIN_NET_DURATION:
            rejected.append({"action": action,
                             "reason": f"the adjusted net duration falls below the {MIN_NET_DURATION}s floor "
                                       f"({new_net} / {new_neighbor})"})
            continue
        dec["net_duration"] = dec["duration"] = new_net
        neighbor["net_duration"] = neighbor["duration"] = new_neighbor
        applied.append({**action, "shot_id": shot_id,
                        "old_net_duration": old_net, "new_net_duration": new_net,
                        "compensated_shot_id": neighbor["shot_id"],
                        "compensated_new_net_duration": new_neighbor})

    # recompute headroom / gen duration uniformly, then decide which shots lack material and must be regenerated
    _recompute_shot_timing(decisions, cfg)
    for dec in decisions:
        if not _footage_sufficient(dec, clip_durations.get(dec["shot_id"])):
            regen_shot_ids.add(dec["shot_id"])

    total_net = sum(float(d.get("net_duration", 0) or 0) for d in decisions)
    if abs(total_net - target_duration) > 0.05:
        # push the float residue onto the last shot so the total length equals target_duration exactly
        last = decisions[-1]
        last["net_duration"] = last["duration"] = round(
            float(last["net_duration"]) + (target_duration - total_net), 3)
        _recompute_shot_timing(decisions, cfg)

    for rec in applied:
        if rec["type"] == "adjust_cut_timing":
            ids = [rec.get("shot_id"), rec.get("compensated_shot_id")]
        elif rec["type"] == "adjust_audio_relation":
            # a prompt change always regenerates; for offset-only changes, regenerate when the clip's audio headroom is too small
            idx = rec.get("transition_index")
            affected = list(rec.get("prompt_updated_shot_ids") or [])
            if idx is not None and 0 <= int(idx) < len(decisions) - 1:
                affected += [decisions[int(idx)]["shot_id"], decisions[int(idx) + 1]["shot_id"]]
            ids = affected
        else:
            ids = []
        rec["regenerated_shot_ids"] = sorted(
            {i for i in ids if i in regen_shot_ids}, key=lambda x: (x is None, x))
        rec["requires_regeneration"] = bool(rec["regenerated_shot_ids"])

    return {
        "applied": applied,
        "rejected": rejected,
        "regen_shot_ids": sorted(regen_shot_ids, key=lambda x: (x is None, x)),
        "recompose": bool(applied),
        "total_net_duration": round(sum(float(d.get("net_duration", 0) or 0)
                                        for d in decisions), 3),
    }


def build_plan_snapshot(decisions: List[Dict[str, Any]],
                        clip_durations: Optional[Dict[Any, float]] = None,
                        prompt_tail_chars: int = 400) -> Dict[str, Any]:
    """Snapshot of the current plan for the planner (net duration / effect / offset / tail of the prompt sound description)."""
    clip_durations = clip_durations or {}
    shots = []
    cursor = 0.0
    for idx, dec in enumerate(decisions):
        net = float(dec.get("net_duration", dec.get("duration", 0)) or 0)
        trans = dec.get("transition_out") or {}
        shots.append({
            "shot_index": idx,
            "shot_id": dec.get("shot_id"),
            "net_duration": round(net, 3),
            "planned_cut_time": round(cursor + net, 3) if idx < len(decisions) - 1 else None,
            "gen_duration": dec.get("gen_duration"),
            "actual_clip_duration": (round(float(clip_durations[dec["shot_id"]]), 3)
                                     if dec.get("shot_id") in clip_durations else None),
            "required_clip_seconds": dec.get("required_clip_seconds"),
            "transition_out": {
                "optical_effect": trans.get("optical_effect"),
                "rendered_effect_seconds": trans.get("transition_duration_seconds"),
                "gt_effect_seconds": trans.get("gt_transition_duration_seconds"),
                "audio_visual_relation": trans.get("audio_visual_relation"),
                "audio_relation": trans.get("audio_relation"),
                "rendered_timing_offset_seconds": trans.get("timing_offset_seconds"),
                "gt_timing_offset_seconds": trans.get("gt_timing_offset_seconds"),
                "overlap_seconds": trans.get("overlap_seconds"),
            } if trans else None,
            "prompt_tail": str(dec.get("prompt", ""))[-prompt_tail_chars:],
            "repair_prompt_additions": dec.get("repair_prompt_additions", []),
        })
        cursor += net
    return {"shots": shots, "total_net_duration": round(cursor, 3)}
