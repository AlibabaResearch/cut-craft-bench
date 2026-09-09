"""
VLM-assisted shot alignment module.
====================
A unified shot-alignment preprocessing layer, after TransNetV2 segmentation and before any dimension evaluation.

Flow:
  raw TransNetV2 segmentation -> VLM-assisted alignment -> aligned_shots, consumed by every dimension

Core logic:
  1. take the raw TransNetV2 segmentation (N_detected shots)
  2. take the GT shot descriptions from the prompt (N_gt shots)
  3. fast path: when N_detected == N_gt and every segment has a plausible duration, map them in order
  4. otherwise run the two-step VLM alignment:
     Step A: global shot semantics (the VLM watches the whole video)
     Step B: fine-grained per-shot alignment (TransNetV2 segments + GT descriptions)
  5. emit the aligned_shots structure for downstream use
"""

import os
import re
import json
import hashlib
import time
from typing import List, Dict, Tuple, Optional

# Import VLM call function from mode_b_eval
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mode_b_eval import call_vlm, compress_video_for_vlm, video_to_base64


# ============================================================
# Configuration
# ============================================================
# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alignment_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Fast-path threshold: minimum duration (seconds) of a detected segment; shorter counts as a fragment
MIN_SHOT_DURATION = 0.5

# System prompt for VLM alignment
# ============================================================
# Skill config loading (Plan B: prompts.yaml is the single source of truth)
# ============================================================
# SYSTEM_ALIGNMENT_STEP_A / SYSTEM_ALIGNMENT_STEP_B are now loaded from
# benchmark/skills/shot-alignment-skill/prompts.yaml (step_a.system_prompt /
# step_b.system_prompt), so they can be tuned without touching this file.
# Falls back to the historical hardcoded text (with a printed warning) if
# the skill config is missing/broken, since alignment is a prerequisite for
# many other dimensions and should not hard-crash the whole pipeline.
try:
    from skill_loader import load_skill
    _shot_align_cfg = load_skill("shot-alignment-skill")
    SYSTEM_ALIGNMENT_STEP_A = _shot_align_cfg["step_a"]["system_prompt"]
    SYSTEM_ALIGNMENT_STEP_B = _shot_align_cfg["step_b"]["system_prompt"]
except Exception as _shot_align_err:
    print(f"[WARN] Failed to load shot-alignment-skill config, using "
          f"hardcoded fallback prompts: {_shot_align_err}")
    SYSTEM_ALIGNMENT_STEP_A = (
        "You are a professional film editor and shot segmentation expert. "
        "Your task is to watch the video and determine the actual shot structure by comparing "
        "the video content with the intended shot descriptions.\n\n"
        "A 'shot' is defined as a continuous segment filmed from one camera setup/angle without cuts. "
        "A cut/transition creates a new shot.\n\n"
        "You must output ONLY valid JSON, with no extra text or explanation."
    )
    SYSTEM_ALIGNMENT_STEP_B = (
        "You are a professional film editor analyzing video content. "
        "Your task is to classify each video segment by its VISUAL CONTENT — "
        "NOT by temporal position or timing.\n\n"
        "CRITICAL RULES:\n"
        "1. For EACH detected segment, watch the actual visual content and describe what you see.\n"
        "2. Then match it to the GT shot description that best fits the CONTENT.\n"
        "3. DO NOT assume segments map to GT shots in sequential order.\n"
        "4. Multiple consecutive segments CAN belong to the same GT shot — "
        "this happens when one shot was incorrectly split into fragments.\n"
        "5. Some GT shots may NOT exist in the video at all (the model failed to generate them).\n"
        "6. If a segment is a transition artifact (<0.3s), label it -1.\n"
        "7. Base your decision ONLY on visual content similarity, IGNORE temporal position.\n\n"
        "You must output ONLY valid JSON, with no extra text or explanation."
    )


# ============================================================
# Utility Functions
# ============================================================

def _compute_cache_key(video_path: str, prompt_id: int) -> str:
    """Cache key (from the video path plus the prompt ID)."""
    key_str = f"{os.path.abspath(video_path)}_{prompt_id}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_cache(cache_key: str) -> Optional[dict]:
    """Try to load an alignment result from the cache."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _save_cache(cache_key: str, result: dict):
    """Save an alignment result to the cache."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except IOError:
        pass


def _parse_time_range(description: str) -> Tuple[float, float]:
    """Parse the time range [start-end] out of a shot description."""
    match = re.search(r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)s?\]', description)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


def _get_shot_boundaries_from_transnet(transnetv2_result: dict) -> List[Tuple[float, float]]:
    """Extract the shot boundaries from a TransNetV2 result."""
    shots = transnetv2_result.get("shots", [])
    if not shots:
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


def _extract_json_from_response(response: str) -> Optional[dict]:
    """Extract a JSON object from a VLM response (fault tolerant)."""
    if not response:
        return None

    # Try a direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting from a markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try the span between the first { and the last }
    first_brace = response.find('{')
    last_brace = response.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(response[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# Core Alignment Functions
# ============================================================

def _vlm_step_a_global_analysis(video_path: str, gt_shots: List[dict]) -> Optional[dict]:
    """
    Step A: global shot semantics.
    The VLM watches the whole video alongside the GT descriptions and decides whether each GT shot exists.
    """
    n_gt = len(gt_shots)

    # Build the list of GT shot descriptions
    shot_descriptions = []
    for i, shot in enumerate(gt_shots):
        desc = shot.get("description_prompt", "")
        # Take a short description (the first 100 chars after the time range is stripped)
        clean_desc = re.sub(r'Shot \d+ \[\d+\.?\d*-\d+\.?\d*s?\]:\s*', '', desc)
        shot_descriptions.append(f"  Shot {i+1}: {clean_desc[:120]}")

    shot_list_text = "\n".join(shot_descriptions)

    user_text = (
        f"This video is intended to have {n_gt} shots with the following descriptions:\n"
        f"{shot_list_text}\n\n"
        f"Watch the ACTUAL video carefully and determine:\n"
        f"1. How many distinct shots (separated by cuts/transitions) are actually present?\n"
        f"2. For each intended shot (1 to {n_gt}), determine if it is:\n"
        f"   - 'present': clearly visible as a distinct shot\n"
        f"   - 'missing': not present in the video at all\n"
        f"   - 'merged': its content appears to be merged with an adjacent shot (no visible cut)\n\n"
        f"Output JSON format:\n"
        f'{{"actual_shot_count": <int>, "shots": ['
        f'{{"gt_idx": 1, "status": "present/missing/merged", "approx_start": <float>, "approx_end": <float>}}, ...]}}\n\n'
        f"For missing shots, set approx_start and approx_end to -1.\n"
        f"For merged shots, use the time range of the combined segment."
    )

    response = call_vlm(SYSTEM_ALIGNMENT_STEP_A, user_text, video_path,
                        max_tokens=1000, temperature=0.1)

    return _extract_json_from_response(response)


def _vlm_step_b_segment_alignment(video_path: str,
                                   detected_boundaries: List[Tuple[float, float]],
                                   gt_shots: List[dict]) -> Optional[dict]:
    """
    Step B: segment-centric classification.
    
    Key change: instead of asking the VLM to group segments under GT shots, the VLM labels each segment
    with the GT shot it belongs to, so its judgement stays strictly within the TransNetV2 segmentation.
    
    Output format: {"segment_labels": [0, 1, 1, 2, -1]}
    - each value is a 0-based GT shot index
    - -1 means the segment belongs to no GT shot (and should be discarded)
    """
    n_gt = len(gt_shots)
    n_detected = len(detected_boundaries)

    # Build the list of detected segments
    segment_lines = []
    for i, (start, end) in enumerate(detected_boundaries):
        duration = end - start
        segment_lines.append(f"  Segment {i}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
    segments_text = "\n".join(segment_lines)

    # Build the list of GT shot descriptions (without time ranges, so the VLM cannot match by position)
    gt_lines = []
    for i, shot in enumerate(gt_shots):
        desc = shot.get("description_prompt", "")
        # Strip the "Shot N [xx.xx-xx.xxs]: " prefix and keep only the content description
        clean_desc = re.sub(r'Shot \d+ \[\d+\.?\d*-\d+\.?\d*s?\]:\s*', '', desc)
        gt_lines.append(f"  GT Shot {i}: {clean_desc[:150]}")
    gt_text = "\n".join(gt_lines)

    user_text = (
        f"A video was automatically split into {n_detected} segments by cut detection:\n"
        f"{segments_text}\n\n"
        f"The video was INTENDED to contain {n_gt} shots. Here are the CONTENT DESCRIPTIONS "
        f"(what each shot should show):\n"
        f"{gt_text}\n\n"
        f"IMPORTANT CONTEXT: The video generation model often makes mistakes:\n"
        f"- It may SPLIT one intended shot into multiple segments (fragments)\n"
        f"- It may SKIP some intended shots entirely (not generate them)\n"
        f"- The LAST GT shot (index {n_gt-1}) is the most commonly skipped one\n"
        f"- So the number of segments ({n_detected}) does NOT necessarily equal "
        f"the number of actually realized shots\n\n"
        f"YOUR TASK:\n"
        f"For each segment, watch the actual video content in that time range and determine "
        f"which GT shot description (0 to {n_gt-1}) best matches what you SEE.\n\n"
        f"Think carefully:\n"
        f"- If two consecutive segments show the SAME scene/subject/action continuing, "
        f"they should have the SAME label (they are fragments of one shot)\n"
        f"- If an intended shot's content does NOT appear anywhere in the video, "
        f"no segment should get that label\n"
        f"- Do NOT just assign labels 0,1,2,...,{n_gt-1} in order — "
        f"that assumption is usually WRONG\n\n"
        f"Output JSON with exactly {n_detected} integer labels:\n"
        f'{{"segment_labels": [<int>, <int>, ...]}}\n'
        f"Each label: 0 to {n_gt-1} (GT shot index) or -1 (discard)."
    )

    response = call_vlm(SYSTEM_ALIGNMENT_STEP_B, user_text, video_path,
                        max_tokens=2000, temperature=0.0)

    result = _extract_json_from_response(response)
    
    # Compatibility: convert an old-format alignment returned by the VLM into segment_labels
    if result and "alignment" in result and "segment_labels" not in result:
        # old format -> new format
        labels = [-1] * n_detected
        for entry in result.get("alignment", []):
            gt_idx = entry.get("gt_idx", -1)
            for seg in entry.get("segments", []):
                if 0 <= seg < n_detected:
                    labels[seg] = gt_idx
        result["segment_labels"] = labels
    
    return result


def _build_alignment_from_vlm(step_a_result: Optional[dict],
                               step_b_result: Optional[dict],
                               detected_boundaries: List[Tuple[float, float]],
                               gt_shots: List[dict]) -> dict:
    """
    Combine the Step A and Step B VLM outputs into the final alignment result.
    Step B is preferred (more precise), with Step A as supplementary validation.
    """
    n_gt = len(gt_shots)
    n_detected = len(detected_boundaries)

    # Prefer the Step B result (segment_labels, or the old alignment format)
    if step_b_result and ("segment_labels" in step_b_result or "alignment" in step_b_result):
        return _build_from_step_b(step_b_result, detected_boundaries, gt_shots)

    # Fall back to the Step A result
    if step_a_result and "shots" in step_a_result:
        return _build_from_step_a(step_a_result, detected_boundaries, gt_shots)

    # Both steps failed: use the sequential truncating alignment
    return _build_fallback_alignment(detected_boundaries, gt_shots)


def _build_from_step_b(step_b_result: dict,
                        detected_boundaries: List[Tuple[float, float]],
                        gt_shots: List[dict]) -> dict:
    """Build the alignment result from Step B's segment_labels.
    
    New (segment-centric) logic:
    - segment_labels: [0, 1, 1, 2] gives the GT shot each segment belongs to
    - adjacent segments with the same label are merged automatically
    - a GT shot with no segment is marked missing
    
    Post-processing corrections:
    - when the labels skip an index (e.g. [0,1,1,3] skips 2) and the skip happens in the last segment,
      the last segment is relabelled with the skipped GT index.
      Reason: the VLM tends to map the last segment to the last GT, while the last GT is the one most often skipped.
    """
    labels = step_b_result.get("segment_labels", [])
    n_gt = len(gt_shots)
    n_detected = len(detected_boundaries)

    # Validate the label count
    if len(labels) != n_detected:
        # Truncate or pad
        if len(labels) > n_detected:
            labels = labels[:n_detected]
        else:
            labels = labels + [-1] * (n_detected - len(labels))

    # Validate the label values
    validated_labels = []
    for lbl in labels:
        if isinstance(lbl, int) and -1 <= lbl < n_gt:
            validated_labels.append(lbl)
        else:
            validated_labels.append(-1)
    labels = validated_labels

    # ---- Post-processing correction #1: fix non-adjacent merges ----
    # Detection: when one GT shot owns non-adjacent segments (separated by another GT's segments),
    # the outlier segment is reassigned to a missing GT shot.
    # Reason: a continuous shot cannot physically be sandwiched inside another shot,
    # so this is usually a VLM mistake (e.g. in [0,1,1,0], segment 3 really belongs to GT2).
    def _fix_non_adjacent_merges(labels, n_gt, detected_boundaries):
        """Fix non-adjacent merges by reassigning the outlier segment to a missing GT shot."""
        fixed = False
        max_iterations = 5  # guard against an infinite loop
        for _ in range(max_iterations):
            # Build each GT's list of segment positions
            gt_positions = {i: [] for i in range(n_gt)}
            for seg_idx, lbl in enumerate(labels):
                if lbl >= 0:
                    gt_positions[lbl].append(seg_idx)
            
            # Find the missing GT shots
            missing_gts = sorted(gt_idx for gt_idx in range(n_gt) 
                                 if not gt_positions[gt_idx])
            if not missing_gts:
                break  # no missing GT, nothing to fix
            
            # Detect non-adjacent merges
            found_fix = False
            for gt_idx, positions in gt_positions.items():
                if len(positions) < 2:
                    continue
                # Check whether these positions are contiguous (adjacency allowed)
                for k in range(len(positions) - 1):
                    gap_start = positions[k] + 1
                    gap_end = positions[k + 1]
                    # Another GT's segments in between mean a non-adjacent merge
                    has_other_gt_in_gap = any(
                        labels[g] >= 0 and labels[g] != gt_idx
                        for g in range(gap_start, gap_end)
                    )
                    if has_other_gt_in_gap:
                        # Decide which one is the outlier segment:
                        # keep the earlier contiguous block and give the later outlier block to a missing GT
                        # (the VLM usually maps a later segment back onto an earlier GT by mistake)
                        outlier_positions = positions[k + 1:]  # the later half is the outlier
                        
                        # Pick the best missing GT to receive it: the closest in time
                        outlier_start_time = detected_boundaries[outlier_positions[0]][0]
                        best_missing_gt = None
                        best_time_dist = float('inf')
                        for mgt in missing_gts:
                            # Estimate the expected position from the centre of the GT time range
                            gt_desc = gt_shots[mgt].get("description_prompt", "")
                            gt_range = _parse_time_range(gt_desc)
                            gt_center = (gt_range[0] + gt_range[1]) / 2
                            dist = abs(outlier_start_time - gt_center)
                            if dist < best_time_dist:
                                best_time_dist = dist
                                best_missing_gt = mgt
                        
                        if best_missing_gt is not None:
                            for pos in outlier_positions:
                                labels[pos] = best_missing_gt
                            print(f"  [Alignment] Non-adjacent fix: segments {outlier_positions} "
                                  f"reassigned GT{gt_idx}→GT{best_missing_gt} "
                                  f"(non-adjacent merge detected)")
                            fixed = True
                            found_fix = True
                            break
                if found_fix:
                    break
            if not found_fix:
                break  # nothing left to fix
        return labels, fixed
    
    labels, non_adj_fixed = _fix_non_adjacent_merges(labels, n_gt, detected_boundaries)

    # ---- Post-processing correction #2: fix skipped labels ----
    # Detection: when a GT index is skipped (absent from labels) while a larger index is used,
    # and that larger index appears only in the last non-merged segment, relabel it with the skipped index.
    # Example: [0,1,1,3] -> GT2 is skipped and GT3 only appears in the last segment -> corrected to [0,1,1,2]
    used_indices = set(lbl for lbl in labels if lbl >= 0)
    all_gt_indices = set(range(n_gt))
    missing_indices = all_gt_indices - used_indices
    
    if missing_indices and used_indices:
        max_used = max(used_indices)
        # Find the skipped indices smaller than max_used
        skipped_before_max = sorted(idx for idx in missing_indices if idx < max_used)
        
        if skipped_before_max:
            # Check whether max_used appears only in the last segment (or the last run of identical labels)
            last_seg_idx = len(labels) - 1
            if labels[last_seg_idx] == max_used:
                # Check whether max_used occurs only in the last segment (merged runs excluded)
                max_used_positions = [i for i, lbl in enumerate(labels) if lbl == max_used]
                # When max_used only occupies the tail of one contiguous block
                if max_used_positions == list(range(max_used_positions[0], last_seg_idx + 1)):
                    # Relabel those positions with the smallest skipped index
                    fix_target = skipped_before_max[-1]  # take the largest skipped index (the closest one)
                    for pos in max_used_positions:
                        labels[pos] = fix_target
                    print(f"  [Alignment] Post-fix: labels corrected "
                          f"(GT{max_used}→GT{fix_target} for last segment, "
                          f"GT{max_used} likely missing)")
    # ---- Post-processing correction #3: time-overlap mismatch correction ----
    # Detection: when a segment's time range barely overlaps (<10%) the GT shot it was labelled with,
    # while a "missing" GT shot overlaps it heavily (>50%),
    # the segment is reassigned to the GT with the higher overlap.
    # Typical case: the segment content looks ambiguous to the VLM, which labels a GT that does not match in time;
    # e.g. seg[4](12.5-15s) labelled GT#2 (expected 5-7.5s) when it should be GT#5 (expected 12.5-15s).
    def _fix_time_overlap_mismatch(labels, n_gt, detected_boundaries, gt_shots):
        """Fix clearly wrong labels using time-overlap validation."""
        fixed = False
        # Parse every GT's expected time range
        gt_ranges = []
        for gt_idx in range(n_gt):
            desc = gt_shots[gt_idx].get("description_prompt", "")
            gt_ranges.append(_parse_time_range(desc))
        
        # Find the currently missing GTs
        used_indices = set(lbl for lbl in labels if lbl >= 0)
        missing_gts = [i for i in range(n_gt) if i not in used_indices]
        if not missing_gts:
            return labels, False
        
        for seg_idx, lbl in enumerate(labels):
            if lbl < 0:
                continue
            seg_start, seg_end = detected_boundaries[seg_idx]
            seg_dur = seg_end - seg_start
            if seg_dur <= 0:
                continue
            
            # Overlap with the currently labelled GT
            gt_start, gt_end = gt_ranges[lbl]
            overlap_start = max(seg_start, gt_start)
            overlap_end = min(seg_end, gt_end)
            current_overlap = max(0, overlap_end - overlap_start) / seg_dur
            
            # Only handle a very low overlap (< 10%)
            if current_overlap >= 0.10:
                continue
            
            # Find the best match among the missing GTs
            best_missing_gt = None
            best_overlap = 0.0
            for mgt in missing_gts:
                mgt_start, mgt_end = gt_ranges[mgt]
                if mgt_end <= mgt_start:
                    continue
                ov_start = max(seg_start, mgt_start)
                ov_end = min(seg_end, mgt_end)
                ov_ratio = max(0, ov_end - ov_start) / seg_dur
                if ov_ratio > best_overlap:
                    best_overlap = ov_ratio
                    best_missing_gt = mgt
            
            # The replacement candidate must overlap by > 50%
            if best_missing_gt is not None and best_overlap > 0.50:
                old_label = lbl
                labels[seg_idx] = best_missing_gt
                missing_gts.remove(best_missing_gt)
                # The old label's GT may become missing (when no other segment remains)
                if old_label not in labels:
                    missing_gts.append(old_label)
                print(f"  [Alignment] Time-overlap fix: segment[{seg_idx}] "
                      f"({seg_start:.1f}-{seg_end:.1f}s) reassigned "
                      f"GT{old_label}→GT{best_missing_gt} "
                      f"(overlap {current_overlap:.0%}→{best_overlap:.0%})")
                fixed = True
        
        return labels, fixed
    
    labels, time_fix_applied = _fix_time_overlap_mismatch(
        labels, n_gt, detected_boundaries, gt_shots)
    # ---- end of post-processing corrections ----

    # Aggregate: find the segment indices of every GT shot
    gt_to_segments = {i: [] for i in range(n_gt)}
    discarded_segments = []
    for seg_idx, lbl in enumerate(labels):
        if lbl == -1:
            discarded_segments.append(seg_idx)
        else:
            gt_to_segments[lbl].append(seg_idx)

    # Build aligned_shots
    aligned_shots = []
    for gt_idx in range(n_gt):
        desc = gt_shots[gt_idx].get("description_prompt", "")
        seg_indices = gt_to_segments[gt_idx]

        if not seg_indices:
            # this GT shot is missing
            aligned_shots.append({
                "gt_shot_idx": gt_idx,
                "gt_description": desc[:150],
                "status": "missing",
                "detected_indices": [],
                "time_range": None,
            })
        elif len(seg_indices) == 1:
            # single-segment match
            seg_idx = seg_indices[0]
            start, end = detected_boundaries[seg_idx]
            aligned_shots.append({
                "gt_shot_idx": gt_idx,
                "gt_description": desc[:150],
                "status": "matched",
                "detected_indices": seg_indices,
                "time_range": [start, end],
            })
        else:
            # merged fragments
            start = detected_boundaries[seg_indices[0]][0]
            end = detected_boundaries[seg_indices[-1]][1]
            aligned_shots.append({
                "gt_shot_idx": gt_idx,
                "gt_description": desc[:150],
                "status": "merged",
                "detected_indices": seg_indices,
                "time_range": [start, end],
            })

    n_matched = sum(1 for a in aligned_shots if a["status"] == "matched")
    n_missing = sum(1 for a in aligned_shots if a["status"] == "missing")
    n_merged = sum(1 for a in aligned_shots if a["status"] == "merged")
    n_present = n_matched + n_merged

    return {
        "aligned_shots": aligned_shots,
        "discarded_segments": sorted(discarded_segments),
        "n_matched": n_matched,
        "n_missing": n_missing,
        "n_merged": n_merged,
        "shot_accuracy": n_present / max(n_gt, 1),
        "method": "vlm_step_b",
        "segment_labels": labels,
    }


def _build_from_step_a(step_a_result: dict,
                        detected_boundaries: List[Tuple[float, float]],
                        gt_shots: List[dict]) -> dict:
    """Build the alignment result from Step A's VLM output."""
    shots_info = step_a_result.get("shots", [])
    n_gt = len(gt_shots)

    aligned_shots = []
    # Track the detected segment indices already used
    next_seg_idx = 0

    for entry in shots_info:
        gt_idx = entry.get("gt_idx", 0)
        # The VLM may return a 1-based index
        if gt_idx >= 1:
            gt_idx -= 1
        if gt_idx < 0 or gt_idx >= n_gt:
            continue

        status = entry.get("status", "present").lower()
        approx_start = entry.get("approx_start", -1)
        approx_end = entry.get("approx_end", -1)
        desc = gt_shots[gt_idx].get("description_prompt", "")

        if status == "missing" or approx_start < 0:
            aligned_shots.append({
                "gt_shot_idx": gt_idx,
                "gt_description": desc[:150],
                "status": "missing",
                "detected_indices": [],
                "time_range": None,
            })
        else:
            # Find the detected segments inside the time range
            matched_segs = []
            for i, (s, e) in enumerate(detected_boundaries):
                if i < next_seg_idx:
                    continue
                # Check whether the segment falls inside the time range the VLM gave
                mid = (s + e) / 2
                if approx_start - 0.5 <= mid <= approx_end + 0.5:
                    matched_segs.append(i)

            if matched_segs:
                next_seg_idx = max(matched_segs) + 1
                start = detected_boundaries[matched_segs[0]][0]
                end = detected_boundaries[matched_segs[-1]][1]
                shot_status = "merged" if len(matched_segs) > 1 else "matched"
            else:
                # Use the time range the VLM gave
                start = approx_start
                end = approx_end
                shot_status = "matched" if status == "present" else "merged"

            aligned_shots.append({
                "gt_shot_idx": gt_idx,
                "gt_description": desc[:150],
                "status": shot_status,
                "detected_indices": matched_segs,
                "time_range": [start, end],
            })

    # Fill in the GT shots that were not covered
    covered_gt = {a["gt_shot_idx"] for a in aligned_shots}
    for i in range(n_gt):
        if i not in covered_gt:
            desc = gt_shots[i].get("description_prompt", "")
            aligned_shots.append({
                "gt_shot_idx": i,
                "gt_description": desc[:150],
                "status": "missing",
                "detected_indices": [],
                "time_range": None,
            })

    aligned_shots.sort(key=lambda x: x["gt_shot_idx"])

    # Statistics
    used_segs = set()
    for a in aligned_shots:
        for s in a.get("detected_indices", []):
            used_segs.add(s)
    discarded = [i for i in range(len(detected_boundaries)) if i not in used_segs]

    n_matched = sum(1 for a in aligned_shots if a["status"] == "matched")
    n_missing = sum(1 for a in aligned_shots if a["status"] == "missing")
    n_merged = sum(1 for a in aligned_shots if a["status"] == "merged")
    n_present = n_matched + n_merged

    return {
        "aligned_shots": aligned_shots,
        "discarded_segments": discarded,
        "n_matched": n_matched,
        "n_missing": n_missing,
        "n_merged": n_merged,
        "shot_accuracy": n_present / max(n_gt, 1),
        "method": "vlm_step_a",
    }


def _build_fallback_alignment(detected_boundaries: List[Tuple[float, float]],
                               gt_shots: List[dict]) -> dict:
    """Fallback: sequential truncating alignment (the original logic)."""
    n_gt = len(gt_shots)
    n_detected = len(detected_boundaries)
    n = min(n_gt, n_detected)

    aligned_shots = []
    for i in range(n_gt):
        desc = gt_shots[i].get("description_prompt", "")
        if i < n_detected:
            start, end = detected_boundaries[i]
            aligned_shots.append({
                "gt_shot_idx": i,
                "gt_description": desc[:150],
                "status": "matched",
                "detected_indices": [i],
                "time_range": [start, end],
            })
        else:
            aligned_shots.append({
                "gt_shot_idx": i,
                "gt_description": desc[:150],
                "status": "missing",
                "detected_indices": [],
                "time_range": None,
            })

    # Extra detected segments count as discarded
    discarded = list(range(n_gt, n_detected)) if n_detected > n_gt else []

    n_matched = min(n_gt, n_detected)
    n_missing = max(0, n_gt - n_detected)

    return {
        "aligned_shots": aligned_shots,
        "discarded_segments": discarded,
        "n_matched": n_matched,
        "n_missing": n_missing,
        "n_merged": 0,
        "shot_accuracy": n_matched / max(n_gt, 1),
        "method": "fallback_sequential",
    }


# ============================================================
# Main Entry Point
# ============================================================

def align_shots_with_vlm(video_path: str,
                          transnetv2_result: dict,
                          prompt: dict,
                          use_cache: bool = True) -> dict:
    """
    Main VLM-assisted shot alignment entry point.

    Args:
        video_path: path to the video file
        transnetv2_result: raw TransNetV2 output
        prompt: prompt JSON config (holding the shots array)
        use_cache: whether to use the cache

    Returns:
        {
            "aligned_shots": [
                {
                    "gt_shot_idx": 0,          # GT shot index (0-based)
                    "gt_description": "...",     # GT shot description (truncated)
                    "status": "matched",         # matched / missing / merged
                    "detected_indices": [0],     # the matching TransNetV2 segment indices
                    "time_range": [0.0, 2.2],   # final time range [start, end]
                },
                ...
            ],
            "discarded_segments": [3],  # detected segments judged invalid fragments
            "n_matched": 3,             # shots matched by a single segment
            "n_missing": 1,             # missing shots
            "n_merged": 1,              # shots rebuilt from merged fragments
            "shot_accuracy": 0.75,      # valid shot rate = (matched+merged) / n_gt
            "method": "vlm_step_b",     # alignment method used
        }
    """
    prompt_id = prompt.get("id", 0)
    gt_shots = prompt.get("shots", [])
    n_gt = len(gt_shots)

    if n_gt == 0:
        return {
            "aligned_shots": [],
            "discarded_segments": [],
            "n_matched": 0, "n_missing": 0, "n_merged": 0,
            "shot_accuracy": 0.0,
            "method": "empty",
        }

    # Try the cache
    if use_cache and os.environ.get("EVAL_NO_CACHE"):
        use_cache = False
    if use_cache:
        cache_key = _compute_cache_key(video_path, prompt_id)
        cached = _load_cache(cache_key)
        if cached is not None:
            # Cache invalidation check #0: the TransNetV2 detection count changed
            # A different segment count from the cached run invalidates the cache completely
            cached_n_detected = len(cached.get("segment_labels", []))
            current_boundaries = _get_shot_boundaries_from_transnet(transnetv2_result)
            current_n_detected = len(current_boundaries)
            
            if cached_n_detected > 0 and current_n_detected != cached_n_detected:
                print(f"  [Alignment] Cache INVALIDATED for video ID={prompt_id} "
                      f"(TransNetV2 segments changed: {cached_n_detected}→{current_n_detected})")
                # Cache fully invalid: skip it and recompute
                cached = None
        
        if cached is not None:
            # Cache revalidation: look for non-adjacent merges or time-overlap mismatches
            # and, if found, rebuild the alignment from the stored segment_labels (without calling the VLM again)
            needs_revalidation = False
            if cached.get("segment_labels") and cached.get("method") == "vlm_step_b":
                aligned_shots = cached.get("aligned_shots", [])
                # Check #1: non-adjacent merge
                for ashot in aligned_shots:
                    indices = ashot.get("detected_indices", [])
                    if len(indices) >= 2:
                        for k in range(len(indices) - 1):
                            if indices[k+1] - indices[k] > 1:
                                seg_labels = cached["segment_labels"]
                                gap_has_other = any(
                                    seg_labels[g] >= 0 and seg_labels[g] != ashot["gt_shot_idx"]
                                    for g in range(indices[k]+1, indices[k+1])
                                    if g < len(seg_labels)
                                )
                                if gap_has_other:
                                    needs_revalidation = True
                                    break
                    if needs_revalidation:
                        break
                
                # Check #2: time-overlap mismatch (a matched shot barely overlaps its expected GT range)
                if not needs_revalidation and cached.get("n_missing", 0) > 0:
                    for ashot in aligned_shots:
                        if ashot["status"] != "matched" or not ashot.get("time_range"):
                            continue
                        gt_idx = ashot["gt_shot_idx"]
                        gt_desc = gt_shots[gt_idx].get("description_prompt", "")
                        gt_range = _parse_time_range(gt_desc)
                        if gt_range[1] <= gt_range[0]:
                            continue
                        seg_start, seg_end = ashot["time_range"]
                        seg_dur = seg_end - seg_start
                        if seg_dur <= 0:
                            continue
                        ov_start = max(seg_start, gt_range[0])
                        ov_end = min(seg_end, gt_range[1])
                        overlap_ratio = max(0, ov_end - ov_start) / seg_dur
                        if overlap_ratio < 0.10:
                            needs_revalidation = True
                            break
            
            if needs_revalidation:
                print(f"  [Alignment] Cache hit for video ID={prompt_id} "
                      f"- REVALIDATING (post-processing fix applicable)")
                detected_boundaries = _get_shot_boundaries_from_transnet(transnetv2_result)
                step_b_result = {"segment_labels": cached["segment_labels"]}
                result = _build_from_step_b(step_b_result, detected_boundaries, gt_shots)
                _save_cache(cache_key, result)
                return result
            else:
                print(f"  [Alignment] Cache hit for video ID={prompt_id}")
                return cached

    # Get the TransNetV2 detection boundaries
    detected_boundaries = _get_shot_boundaries_from_transnet(transnetv2_result)
    n_detected = len(detected_boundaries)

    print(f"  [Alignment] Video ID={prompt_id}: GT={n_gt} shots, Detected={n_detected} segments")

    # No fast path: even with N_detected == N_gt the VLM must compare content segment by segment,
    # because equal shot counts do not mean the content lines up (e.g. shot B fragments into B1+B2
    # while D is missing)

    # VLM-assisted alignment is required
    print(f"  [Alignment] Calling VLM for alignment (mismatch: {n_detected} vs {n_gt})...")

    # Step A: global semantics
    step_a_result = None
    try:
        t0 = time.time()
        step_a_result = _vlm_step_a_global_analysis(video_path, gt_shots)
        print(f"  [Alignment] Step A done in {time.time()-t0:.1f}s: "
              f"actual_shots={step_a_result.get('actual_shot_count', '?') if step_a_result else 'FAILED'}")
    except Exception as e:
        print(f"  [Alignment] Step A failed: {e}")

    # Step B: fine-grained alignment
    step_b_result = None
    try:
        t0 = time.time()
        step_b_result = _vlm_step_b_segment_alignment(video_path, detected_boundaries, gt_shots)
        has_labels = step_b_result is not None and "segment_labels" in (step_b_result or {})
        labels_str = step_b_result.get("segment_labels", []) if step_b_result else []
        print(f"  [Alignment] Step B done in {time.time()-t0:.1f}s: "
              f"has_labels={has_labels}, labels={labels_str}")
    except Exception as e:
        print(f"  [Alignment] Step B failed: {e}")

    # Combine both into the final alignment result
    result = _build_alignment_from_vlm(step_a_result, step_b_result,
                                        detected_boundaries, gt_shots)

    print(f"  [Alignment] Result: matched={result['n_matched']}, "
          f"merged={result['n_merged']}, missing={result['n_missing']}, "
          f"accuracy={result['shot_accuracy']:.2f}, method={result['method']}")

    # Save the cache
    if use_cache:
        _save_cache(cache_key, result)

    return result


# ============================================================
# Helper: Get aligned transitions
# ============================================================

def get_aligned_transitions(alignment_result: dict) -> List[Dict]:
    """
    Extract the list of valid transitions from an alignment result.
    A transition is valid only when both neighbouring aligned_shots exist (status != 'missing').

    Returns:
        [
            {
                "from_gt_idx": 0,
                "to_gt_idx": 1,
                "cut_time": 2.2,  # end time of the previous shot
                "evaluable": True,
            },
            ...
        ]
    """
    aligned_shots = alignment_result.get("aligned_shots", [])
    transitions = []

    for i in range(len(aligned_shots) - 1):
        current = aligned_shots[i]
        next_shot = aligned_shots[i + 1]

        evaluable = (current["status"] != "missing" and next_shot["status"] != "missing")

        cut_time = None
        if evaluable and current["time_range"] is not None:
            cut_time = current["time_range"][1]  # end time of the current shot

        transitions.append({
            "from_gt_idx": current["gt_shot_idx"],
            "to_gt_idx": next_shot["gt_shot_idx"],
            "cut_time": cut_time,
            "evaluable": evaluable,
        })

    return transitions


def get_aligned_shot_clip_range(alignment_result: dict, gt_shot_idx: int) -> Optional[Tuple[float, float]]:
    """
    Aligned time range of a given GT shot (used to cut out the shot clip).

    Returns:
        (start_sec, end_sec) or None if missing
    """
    for shot in alignment_result.get("aligned_shots", []):
        if shot["gt_shot_idx"] == gt_shot_idx:
            if shot["status"] != "missing" and shot["time_range"] is not None:
                return tuple(shot["time_range"])
            return None
    return None


def get_first_segment_clip_range(alignment_result: dict, gt_shot_idx: int,
                                  detected_boundaries: List[Tuple[float, float]] = None
                                  ) -> Optional[Tuple[float, float]]:
    """
    Time range of the FIRST detected segment of a given GT shot (for content evaluations such as E1).
    
    For a merged shot (e.g. B1+B2 -> B), only the first fragment (B1) is evaluated,
    because a content evaluation should judge the first segment.
    
    For a matched shot this equals get_aligned_shot_clip_range().
    For a missing shot it returns None.
    
    Args:
        alignment_result: the alignment result
        gt_shot_idx: GT shot index (0-based)
        detected_boundaries: TransNetV2 boundaries (optional, to pin down the first segment)
    
    Returns:
        (start_sec, end_sec) or None if missing
    """
    for shot in alignment_result.get("aligned_shots", []):
        if shot["gt_shot_idx"] == gt_shot_idx:
            if shot["status"] == "missing" or shot["time_range"] is None:
                return None
            
            # matched: return time_range directly
            if shot["status"] == "matched":
                return tuple(shot["time_range"])
            
            # merged: take only the first detected segment's time range
            detected_indices = shot.get("detected_indices", [])
            if detected_indices and detected_boundaries:
                first_seg_idx = detected_indices[0]
                if first_seg_idx < len(detected_boundaries):
                    return detected_boundaries[first_seg_idx]
            
            # Without detected_boundaries, fall back to the full time_range
            # (this should not happen, but it is a safe fallback)
            return tuple(shot["time_range"])
    return None


# ============================================================
# Standalone Test
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM Shot Alignment Test")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--prompt", required=True, help="Prompt JSON file path")
    parser.add_argument("--video-id", type=int, required=True, help="Video ID in prompt")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    args = parser.parse_args()

    # Load prompt
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    prompt = None
    for p in prompts:
        if p["id"] == args.video_id:
            prompt = p
            break

    if prompt is None:
        print(f"ERROR: Video ID={args.video_id} not found in prompt file")
        sys.exit(1)

    # Call TransNetV2
    from mode_b_eval import call_transnetv2
    print(f"Calling TransNetV2 for {args.video}...")
    transnet_result = call_transnetv2(args.video)
    print(f"TransNetV2: {transnet_result.get('num_shots', '?')} shots detected")

    # Run alignment
    result = align_shots_with_vlm(args.video, transnet_result, prompt,
                                   use_cache=not args.no_cache)

    print(f"\n{'='*60}")
    print("Alignment Result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
