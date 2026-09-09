"""
A3 counterfactual causal-chain evaluation.

This module replaces the old Mode C A3 multiple-choice score with a
reverse-order counterfactual signal:

    score = C_orig * (lambda + (1 - lambda) * clip01(C_orig - C_rev))

Important: reverse order means reordering aligned shot clips while each shot
itself is played forward. It never applies pixel-level reverse filters.
"""
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from mode_b_eval import call_vlm


LAMBDA_ORDER_PRIOR = 0.3
NON_DIRECTIONAL_MONTAGE = {"lyrical", "contrast", "attractions", "metaphorical", "reflexive"}

MONTAGE_DEFINITIONS = {
    "sequential": "All events unfold in chronological order, with no special treatment.",
    "parallel": "Two event chains progress separately in different times and spaces, and may eventually converge, serving to advance the plot.",
    "crosscut": "Two event chains progress separately in different spaces at the same time, and may eventually converge, serving to advance the plot.",
    "repetition": "Multiple shots repeatedly present the same or similar imagery, with the same subject within the frame.",
    "dialogue": "In one shot, a question is raised through speech, and in the next shot, it is answered through visual language rather than direct verbal response.",
    "lyrical": "In normal narration, empty scenic shots with no characters are inserted to express emotion.",
    "psychological": "Virtual scenes such as hallucination, dream, memory, or imagination are inserted to heighten atmosphere.",
    "metaphorical": "Recurring or associated imagery across scenes conveys metaphorical meaning related to the plot.",
    "contrast": "Opposing content or forms are juxtaposed to create conflict and reinforce the theme.",
    "accumulative": "Homogeneous content or forms are rapidly assembled to build atmosphere.",
    "attractions": "A shot unrelated to the plot and not belonging to the same scene is inserted to provoke emotion or idea.",
    "reflexive": "A plot-unrelated object from within the same scene is inserted to express emotion.",
    "ideological": "Existing materials such as news footage or documentaries are arranged and edited to argue for a viewpoint.",
}

INTRA_SYSTEM_PROMPT = (
    "You are a professional film editing analyst. The video below is assembled from several forward-playing shots "
    "belonging to the same event thread. Judge whether the CURRENT shot order is natural in terms of causality "
    "and time. Do not assume the video has been manipulated; evaluate only what you see. You must output ONLY "
    "valid JSON, with no markdown or extra text."
)

INTER_SYSTEM_PROMPT = (
    "You are a montage analyst. The video below is assembled from multiple forward-playing shots that may belong "
    "to parallel event threads. Judge whether the current arrangement realizes the intended montage type. You must "
    "output ONLY valid JSON, with no markdown or extra text."
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _safe_unlink(path: Optional[str]):
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _extract_json_from_response(response: str) -> Optional[dict]:
    if not response or response.upper().strip().startswith("ERROR:"):
        return None
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _normalize_montage_type(prompt: dict) -> str:
    candidates = [
        prompt.get("montage_type", ""),
        prompt.get("source_seed", "").split("_")[0],
        prompt.get("global_editing_style", ""),
    ]
    aliases = {
        "montage of attractions": "attractions",
        "attraction": "attractions",
        "attractions": "attractions",
        "crosscut montage": "crosscut",
        "crosscutting": "crosscut",
    }
    known = sorted(MONTAGE_DEFINITIONS.keys(), key=len, reverse=True)
    for raw in candidates:
        text = str(raw or "").strip().lower()
        if not text:
            continue
        compact = text.replace("-", " ")
        for alias, canonical in aliases.items():
            if alias in compact:
                return canonical
        for key in known:
            if key in compact:
                return key
    return "sequential"


def _montage_display_name(montage_type: str) -> str:
    if montage_type == "attractions":
        return "Montage of Attractions"
    return f"{montage_type.capitalize()} Montage"


def _get_shot_label(gt_shot: dict, gt_idx: int):
    return gt_shot.get("event_coherence_label", f"shot_{gt_idx}")


def _build_real_chains(aligned_shots: List[dict], gt_shots: List[dict]) -> Dict[str, List[int]]:
    chains: Dict[str, List[int]] = {}
    for gt_idx, aligned in enumerate(aligned_shots):
        if gt_idx >= len(gt_shots):
            continue
        if aligned.get("status") not in ("matched", "merged") or not aligned.get("time_range"):
            continue
        label = str(_get_shot_label(gt_shots[gt_idx], gt_idx))
        chains.setdefault(label, []).append(gt_idx)

    for label, shot_indices in chains.items():
        shot_indices.sort(key=lambda idx: aligned_shots[idx].get("time_range", [0, 0])[0])
    return chains


def _get_gt_chain_count(gt_shots: List[dict]) -> int:
    labels = {str(_get_shot_label(shot, idx)) for idx, shot in enumerate(gt_shots)}
    return len(labels)


def _shot_range(aligned_shots: List[dict], gt_idx: int) -> Optional[Tuple[float, float]]:
    if gt_idx >= len(aligned_shots):
        return None
    time_range = aligned_shots[gt_idx].get("time_range")
    if not time_range or len(time_range) < 2:
        return None
    start, end = float(time_range[0]), float(time_range[1])
    if end <= start:
        return None
    return start, end


def build_concat_clip(video_path: str, shot_indices: List[int], aligned_shots: List[dict]) -> Optional[str]:
    """Build a temporary clip by concatenating aligned shots in the requested order.

    Each shot segment is trimmed from the original video and kept in normal playback
    direction. The output is video-only to avoid audio stream incompatibilities.
    """
    ranges = [_shot_range(aligned_shots, idx) for idx in shot_indices]
    ranges = [r for r in ranges if r is not None]
    if not ranges:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    filter_parts = []
    concat_inputs = []
    for i, (start, end) in enumerate(ranges):
        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v{i}]"
        )
        concat_inputs.append(f"[v{i}]")
    filter_complex = ";".join(filter_parts) + ";" + "".join(concat_inputs) + f"concat=n={len(ranges)}:v=1:a=0[outv]"

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        tmp.name,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        if result.returncode != 0 or not os.path.exists(tmp.name) or os.path.getsize(tmp.name) == 0:
            _safe_unlink(tmp.name)
            return None
        return tmp.name
    except Exception:
        _safe_unlink(tmp.name)
        return None


def _run_vlm_json(system_prompt: str, user_text: str, video_path: str, max_tokens: int = 1200) -> Tuple[Optional[dict], str]:
    last_response = ""
    for _ in range(2):
        last_response = call_vlm(system_prompt, user_text, video_path, max_tokens=max_tokens, temperature=0.0)
        parsed = _extract_json_from_response(last_response)
        if parsed is not None:
            return parsed, last_response
    return None, last_response


def _score_intra_response(parsed: dict) -> Optional[float]:
    cuts = parsed.get("cuts") if isinstance(parsed, dict) else None
    if not isinstance(cuts, list) or not cuts:
        return None
    label_scores = {"FORWARD": 1.0, "INDEPENDENT": 0.5, "REVERSED": 0.0}
    scores = []
    for cut in cuts:
        label = str(cut.get("label", "")).strip().upper()
        if label in label_scores:
            scores.append(label_scores[label])
    return _mean(scores)


def run_intra_vlm(clip_path: str, n_shots: int) -> dict:
    user_text = (
        f"The video contains {n_shots} shots, creating {max(n_shots - 1, 0)} adjacent cuts.\n"
        "For each cut from shot i to shot i+1, classify the temporal/causal relation of the later shot to the earlier shot:\n"
        "- FORWARD: the later shot is a natural causal/time continuation.\n"
        "- REVERSED: the later shot should have happened before the earlier shot.\n"
        "- INDEPENDENT: there is no clear direction; either order would be plausible.\n\n"
        "Use only visible evidence such as object state changes, action starts/ends, character position, and scene progression. "
        "If concrete evidence is missing, choose INDEPENDENT.\n\n"
        "Output JSON exactly like: {\"cuts\":[{\"from_shot\":1,\"to_shot\":2,\"label\":\"FORWARD|REVERSED|INDEPENDENT\",\"evidence\":\"...\"}]}"
    )
    parsed, response = _run_vlm_json(INTRA_SYSTEM_PROMPT, user_text, clip_path)
    score = _score_intra_response(parsed) if parsed is not None else None
    return {"score": score, "parsed": parsed, "vlm_response": response[:500] if response else ""}


def _score_inter_response(parsed: dict, expected_cross_cuts: int) -> Optional[float]:
    if not isinstance(parsed, dict):
        return None
    overall_map = {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}
    overall = overall_map.get(str(parsed.get("overall_realized", "")).strip().upper())
    cross_cuts = parsed.get("cross_cuts", [])
    if overall is None:
        return None
    motivated = 0
    valid = 0
    if isinstance(cross_cuts, list):
        for cut in cross_cuts:
            label = str(cut.get("label", "")).strip().upper()
            if label in ("MOTIVATED", "ARBITRARY"):
                valid += 1
                if label == "MOTIVATED":
                    motivated += 1
    denominator = valid if valid > 0 else expected_cross_cuts
    motivated_ratio = motivated / denominator if denominator > 0 else 0.0
    return 0.5 * overall + 0.5 * motivated_ratio


def run_inter_vlm(clip_path: str, montage_type: str, montage_definition: str,
                  ordered_labels: List[str]) -> dict:
    label_sequence = " -> ".join(str(label) for label in ordered_labels)
    expected_cross_cuts = sum(1 for i in range(len(ordered_labels) - 1) if ordered_labels[i] != ordered_labels[i + 1])
    user_text = (
        f"Intended montage type: {_montage_display_name(montage_type)} ({montage_definition})\n"
        f"The assembled video contains {len(ordered_labels)} shots. Event-thread label sequence in this clip: {label_sequence}.\n"
        f"There are {expected_cross_cuts} adjacent cuts where the event-thread label changes.\n\n"
        "Tasks:\n"
        f"1. Judge whether switching between event threads serves the expressive purpose of {_montage_display_name(montage_type)}.\n"
        "2. For each cross-thread cut, classify the editing motivation:\n"
        "   - MOTIVATED: the jump has a clear purpose and fits the montage type.\n"
        "   - ARBITRARY: the jump looks random or unmotivated.\n\n"
        "Output JSON exactly like: {\"overall_realized\":\"YES|PARTIAL|NO\",\"cross_cuts\":[{\"at\":1,\"label\":\"MOTIVATED|ARBITRARY\",\"evidence\":\"...\"}]}"
    )
    parsed, response = _run_vlm_json(INTER_SYSTEM_PROMPT, user_text, clip_path, max_tokens=1400)
    score = _score_inter_response(parsed, expected_cross_cuts) if parsed is not None else None
    return {
        "score": score,
        "parsed": parsed,
        "vlm_response": response[:500] if response else "",
        "expected_cross_cuts": expected_cross_cuts,
    }


def eval_a3_counterfactual(video_path: str, alignment_result: dict, prompt: dict) -> dict:
    gt_shots = prompt.get("shots", []) if prompt else []
    aligned_shots = alignment_result.get("aligned_shots", []) if alignment_result else []
    flags = {}

    if not gt_shots or not aligned_shots:
        return {
            "A3": 0.0,
            "intra_score": None,
            "inter_score": None,
            "chain_coverage": 0.0,
            "insufficient_structure": True,
            "error": "missing prompt shots or alignment_result",
        }

    chains = _build_real_chains(aligned_shots, gt_shots)
    k_gt = _get_gt_chain_count(gt_shots)
    k_real = len(chains)
    chain_coverage = k_real / max(k_gt, 1)

    chain_details = []
    chain_scores = []
    for label, shot_indices in chains.items():
        detail = {"label": label, "shot_indices": [idx + 1 for idx in shot_indices]}
        if len(shot_indices) < 2:
            detail["skipped"] = "single_shot_chain"
            chain_details.append(detail)
            continue

        clip_fwd = build_concat_clip(video_path, shot_indices, aligned_shots)
        clip_rev = build_concat_clip(video_path, list(reversed(shot_indices)), aligned_shots)
        try:
            if not clip_fwd or not clip_rev:
                detail["skipped"] = "concat_failed"
                chain_details.append(detail)
                continue
            orig = run_intra_vlm(clip_fwd, len(shot_indices))
            rev = run_intra_vlm(clip_rev, len(shot_indices))
            c_orig = orig.get("score")
            c_rev = rev.get("score")
            detail.update({
                "C_orig": c_orig,
                "C_rev": c_rev,
                "orig_response": orig.get("vlm_response", ""),
                "rev_response": rev.get("vlm_response", ""),
                "orig_parsed": orig.get("parsed"),
                "rev_parsed": rev.get("parsed"),
            })
            if c_orig is None or c_rev is None:
                detail["skipped"] = "vlm_parse_failed"
                chain_details.append(detail)
                continue
            d_value = _clip01(c_orig - c_rev)
            chain_score = c_orig * (LAMBDA_ORDER_PRIOR + (1.0 - LAMBDA_ORDER_PRIOR) * d_value)
            detail.update({"D": d_value, "chain_score": chain_score})
            chain_scores.append(chain_score)
            chain_details.append(detail)
        finally:
            _safe_unlink(clip_fwd)
            _safe_unlink(clip_rev)

    intra_score = _mean(chain_scores)

    inter_score = None
    inter_detail = None
    if k_gt >= 2 and k_real == 1:
        flags["montage_structure_collapsed"] = True
    elif k_real >= 2:
        montage_type = _normalize_montage_type(prompt)
        montage_definition = MONTAGE_DEFINITIONS.get(montage_type, prompt.get("global_editing_style", ""))
        all_indices = sorted(
            [idx for shot_indices in chains.values() for idx in shot_indices],
            key=lambda idx: aligned_shots[idx].get("time_range", [0, 0])[0],
        )
        ordered_labels = [str(_get_shot_label(gt_shots[idx], idx)) for idx in all_indices]
        clip_fwd = build_concat_clip(video_path, all_indices, aligned_shots)
        clip_rev = None
        inter_detail = {
            "montage_type": montage_type,
            "montage_definition": montage_definition,
            "shot_indices": [idx + 1 for idx in all_indices],
            "label_sequence": ordered_labels,
            "non_directional": montage_type in NON_DIRECTIONAL_MONTAGE,
        }
        try:
            if clip_fwd:
                orig = run_inter_vlm(clip_fwd, montage_type, montage_definition, ordered_labels)
                c_inter_orig = orig.get("score")
                inter_detail.update({
                    "C_inter_orig": c_inter_orig,
                    "orig_response": orig.get("vlm_response", ""),
                    "orig_parsed": orig.get("parsed"),
                    "expected_cross_cuts": orig.get("expected_cross_cuts", 0),
                })
                if c_inter_orig is not None and montage_type in NON_DIRECTIONAL_MONTAGE:
                    inter_score = c_inter_orig
                elif c_inter_orig is not None:
                    rev_indices = list(reversed(all_indices))
                    rev_labels = [str(_get_shot_label(gt_shots[idx], idx)) for idx in rev_indices]
                    clip_rev = build_concat_clip(video_path, rev_indices, aligned_shots)
                    if clip_rev:
                        rev = run_inter_vlm(clip_rev, montage_type, montage_definition, rev_labels)
                        c_inter_rev = rev.get("score")
                        inter_detail.update({
                            "C_inter_rev": c_inter_rev,
                            "rev_response": rev.get("vlm_response", ""),
                            "rev_parsed": rev.get("parsed"),
                        })
                        if c_inter_rev is not None:
                            d_inter = _clip01(c_inter_orig - c_inter_rev)
                            inter_score = c_inter_orig * (LAMBDA_ORDER_PRIOR + (1.0 - LAMBDA_ORDER_PRIOR) * d_inter)
                            inter_detail.update({"D_inter": d_inter})
                    else:
                        inter_detail["skipped"] = "reverse_concat_failed"
                else:
                    inter_detail["skipped"] = "vlm_parse_failed"
            else:
                inter_detail["skipped"] = "concat_failed"
        finally:
            _safe_unlink(clip_fwd)
            _safe_unlink(clip_rev)

    if intra_score is None and inter_score is None:
        flags["insufficient_structure"] = True
        a3_score = 0.0
    elif intra_score is not None and inter_score is not None:
        a3_score = (intra_score + inter_score) / 2.0
    else:
        a3_score = intra_score if intra_score is not None else inter_score

    return {
        "A3": _clip01(a3_score),
        "intra_score": intra_score,
        "inter_score": inter_score,
        "chain_coverage": chain_coverage,
        "K_gt": k_gt,
        "K_real": k_real,
        "chains": chain_details,
        "inter_detail": inter_detail,
        **flags,
    }
