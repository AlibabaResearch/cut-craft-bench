"""Stage 1 -- shot segmentation.

For one case from prompt.json:
  - walk shots[]
  - parse the [start-end s] timestamp inside each description_prompt
  - compute an integer-second duration per shot (cumulative alignment avoids drift, clamped to WAN's range)
  - extract camera{} and transition_to_next (the last shot has none)
Outputs shot_plan (list[dict]), serialisable as shot_plan.json.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Matches timestamps like [00.00-02.20s] / [02.20-04.70 s] / [0-3s]
_TIME_RE = re.compile(
    r"\[\s*(\d+(?:\.\d+)?)\s*[-–~]\s*(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*\]",
    re.IGNORECASE,
)


def parse_shot_timing(description_prompt: str) -> Optional[Dict[str, float]]:
    """Parse the [start-end] timestamp from a shot description, returning {start, end} in seconds, or None."""
    if not description_prompt:
        return None
    m = _TIME_RE.search(description_prompt)
    if not m:
        return None
    start = float(m.group(1))
    end = float(m.group(2))
    if end < start:
        start, end = end, start
    return {"start": start, "end": end}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _allocate_integer_durations(
    raw_durations: List[float],
    target_duration: int,
    wan_min_duration: int,
    wan_max_duration: int,
) -> List[int]:
    """[Deprecated, kept for compatibility] Allocate integer seconds proportional to the raw durations.

    History: when target < wan_min_duration * n (e.g. 6 shots * a 3s floor = 18s > 15s), this also
    clamped each shot's net_duration (the final edited length) to wan_min_duration, so the sum of net
    durations systematically exceeded target_duration and Stage5's final trim cut away almost the whole last shot.
    Net durations now come from `_allocate_net_durations()` (float, summing exactly to target_duration),
    fully decoupled from the integer floor sent to the model (wan_min_duration). segment_case no longer calls this.
    """
    n = len(raw_durations)
    if n == 0:
        return []
    target = int(round(target_duration))
    min_total = wan_min_duration * n
    if target < min_total:
        return [_clamp(int(round(d)) or 1, wan_min_duration, wan_max_duration) for d in raw_durations]

    weights = [max(float(d), 0.001) for d in raw_durations]
    weight_sum = sum(weights) or float(n)
    available = target - min_total
    scaled_extras = [(w / weight_sum) * available for w in weights]
    durations = [wan_min_duration + int(extra) for extra in scaled_extras]
    durations = [min(d, wan_max_duration) for d in durations]

    remaining = target - sum(durations)
    order = sorted(
        range(n),
        key=lambda i: (scaled_extras[i] - int(scaled_extras[i]), weights[i]),
        reverse=True,
    )
    while remaining > 0:
        changed = False
        for idx in order:
            if durations[idx] < wan_max_duration:
                durations[idx] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break

    while remaining < 0:
        changed = False
        for idx in reversed(order):
            if durations[idx] > wan_min_duration:
                durations[idx] -= 1
                remaining += 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return durations


def _allocate_net_durations(
    raw_durations: List[float],
    target_duration: float,
    min_floor: float = 0.2,
) -> List[float]:
    """Allocate float-second net durations proportional to the raw (prompt GT) durations, summing exactly to target_duration.

    net_duration is only the final Stage5 trim length; it is fully decoupled from the integer seconds /
    provider minimum (wan_min_duration) sent to the model -- however long a provider constraint (e.g.
    a 3s per-task floor) stretches the generated clip, the final cut is trimmed back to the net duration
    allocated here, so sum(net_duration) == target_duration and the cut matches the target exactly.
    """
    n = len(raw_durations)
    if n == 0:
        return []
    target = float(target_duration)
    weights = [max(float(d), 0.001) for d in raw_durations]
    weight_sum = sum(weights) or float(n)
    durations = [target * (w / weight_sum) for w in weights]
    # Guard against near-zero durations from extreme ratios (normal GT timestamps never hit this)
    durations = [max(d, min(min_floor, target / n)) for d in durations]
    # Renormalise after clamping to the floor, so the sum matches target exactly (float error <1ms)
    adj_sum = sum(durations)
    if adj_sum > 0:
        durations = [d * (target / adj_sum) for d in durations]
    # Remove accumulated float error: push the residue onto the last shot
    diff = target - sum(durations)
    if durations:
        durations[-1] = max(0.05, durations[-1] + diff)
    return [round(d, 3) for d in durations]


def segment_case(
    case: Dict[str, Any],
    wan_min_duration: int = 2,
    wan_max_duration: int = 15,
    target_duration: int = 15,
) -> Dict[str, Any]:
    """Split one case into a shot plan.

    Returns:
        {
          "id", "title", "source_seed",
          "overall_description_prompt", "global_editing_style",
          "number_of_shots", "total_duration",
          "shots": [ {shot_id, start, end, raw_duration, duration,
                      description_prompt, camera, transition_to_next}, ... ]
        }
    """
    shots_in = case.get("shots", []) or []
    shots_out: List[Dict[str, Any]] = []

    # Parse the raw times first, for the total duration and the fallback
    parsed_times: List[Optional[Dict[str, float]]] = [
        parse_shot_timing(s.get("description_prompt", "")) for s in shots_in
    ]

    # Estimate the case duration: prefer the last shot's end, otherwise default to 15s
    total_duration = 15.0
    last_valid_end = None
    for t in parsed_times:
        if t is not None:
            last_valid_end = t["end"]
    if last_valid_end is not None:
        total_duration = last_valid_end

    n = len(shots_in)
    case_label_keys = (
        "audio_reference_labels", "shot_combination_labels",
        "event_coherence_labels", "shot_labels", "labels",
    )
    case_labels = None
    for key in case_label_keys:
        values = case.get(key)
        if isinstance(values, list) and len(values) == n:
            case_labels = values
            break

    # Net duration (net_duration/duration): float seconds proportional to the prompt GT timestamps,
    # summing exactly to target_duration, fully decoupled from the integer floor wan_min_duration
    # sent to the model. A longer actual generation duration required by a provider constraint (e.g.
    # a 3s per-task floor) is computed separately as gen_duration by Stage3
    # (generation_planner.plan_generation) and clamped to [wan_min_duration, wan_max_duration];
    # Stage5 trims the clip back to the net duration, so rounding or a floor never overruns the target.
    net_durations = _allocate_net_durations(
        [
            (t["end"] - t["start"]) if t is not None else (total_duration / n if n else total_duration)
            for t in parsed_times
        ],
        target_duration=target_duration,
    )

    for idx, shot in enumerate(shots_in):
        t = parsed_times[idx]
        if t is not None:
            start, end = t["start"], t["end"]
            raw_duration = end - start
        else:
            # Missing timestamp: split total_duration evenly
            avg = total_duration / n if n else total_duration
            start = round(idx * avg, 2)
            end = round((idx + 1) * avg, 2)
            raw_duration = end - start

        duration = net_durations[idx] if idx < len(net_durations) else round(max(raw_duration, 0.2), 3)

        transition = shot.get("transition_to_next")  # the last shot usually has no such field

        # Audio reference label: shots sharing a label share one reference clip (e.g. the [0,1,0,1,1]
        # grouping from event_coherence_label). Prefer the case-level parallel array, else the shot's own field.
        if case_labels is not None:
            audio_label = case_labels[idx]
        else:
            audio_label = shot.get("event_coherence_label")

        shots_out.append({
            "shot_id": shot.get("shot_id", idx + 1),
            "start": start,
            "end": end,
            "raw_duration": round(raw_duration, 2),
            "duration": duration,            # edited net duration (float seconds), decoupled from the generation-side integer floor
            "description_prompt": shot.get("description_prompt", ""),
            "camera": shot.get("camera", {}) or {},
            "transition_to_next": transition,
            "rewritten_prompt": None,        # filled in by Stage 2
            "audio_label": audio_label,      # audio reference group label
        })

    return {
        "id": case.get("id"),
        "title": case.get("title", ""),
        "source_seed": case.get("source_seed", ""),
        "overall_description_prompt": case.get("overall_description_prompt", ""),
        "global_editing_style": case.get("global_editing_style", ""),
        "number_of_shots": case.get("number_of_shots", n),
        "total_duration": target_duration,
        "shots": shots_out,
    }


def load_cases(prompt_json_path: str) -> List[Dict[str, Any]]:
    """Read prompt.json and return the list of cases."""
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def find_case_by_id(cases: List[Dict[str, Any]], case_id: int) -> Optional[Dict[str, Any]]:
    for c in cases:
        if c.get("id") == case_id:
            return c
    return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="shot segmentation self-test")
    ap.add_argument("--prompt-json", required=True)
    ap.add_argument("--id", type=int, default=None, help="segment only the given case id")
    args = ap.parse_args()

    cases = load_cases(args.prompt_json)
    targets = [find_case_by_id(cases, args.id)] if args.id else cases[:1]
    for c in targets:
        if c is None:
            print("the given case was not found")
            continue
        plan = segment_case(c)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
