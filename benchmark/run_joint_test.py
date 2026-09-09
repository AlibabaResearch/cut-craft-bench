"""
Joint Mode A + Mode B + Mode C test script.
- Mode A: direct expert-model computation (A1,A4,B1,B2,D1,D2,D3,E2,E3_ModeA,F1,F2)
- Mode B: VLM-assisted judgement (C1,D1_mode_b,E1-5sub)
- Mode C: VLM question answering (A2,A3,C1,E3 plus supplementary E1/D1 questions)

Final aggregation rules:
- C1: mean of the Mode B C1 score and the Mode C C1 score
- E1: mean Mode B E1 sub-dimension score x 0.5 + Mode C E1 question score x 0.5
- D1: flat mean over all transitions (Mode A+B) x 0.5 + Mode C D1 question score x 0.5
- A4: (Mode A A4 (CLIP style match) + Mode A E3 (DINOv2 style consistency)) / 2
- E3: the Mode C score directly (physical consistency questions)
- A2, A3: the Mode C score directly
- everything else: that mode's own score

Test data paths:
  The video directory / prompt / question bank are all injected as environment variables by the
  "dataset path configuration (single source of truth)" block at the top of benchmark/run_model_sequence_with_watchdog.sh:
    EVAL_VIDEO_DIR_TEMPLATE / EVAL_PROMPT_FILE / EVAL_QUESTION_BANK
  This file no longer hardcodes any file name; export those three variables before running it standalone.

Usage:
  # test every sample
  python run_joint_test.py

  # test by ID (the id field in the prompt)
  python run_joint_test.py --ids 3 8 14 21 26 48 63 92 106 129 171 204 238 254 270 284 313

  # test by 1-based JSON index range (inclusive on both ends)
  python run_joint_test.py --range 1-10
  python run_joint_test.py --range 20-30

  # combined
  python run_joint_test.py --ids 1 5 --range 20-25
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# Directory holding ffmpeg / ffprobe. The versions on PATH are used by default; to pin a dedicated
# environment, export AV_PROCESS_BIN=/path/to/env/bin before running.
AV_PROCESS_BIN = os.environ.get("AV_PROCESS_BIN", "")
if AV_PROCESS_BIN and AV_PROCESS_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = AV_PROCESS_BIN + os.pathsep + os.environ.get("PATH", "")

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent

sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# Configuration
# ============================================================
# The dataset / video / output paths come solely from the config block at the top of
# benchmark/run_model_sequence_with_watchdog.sh, injected as environment variables. No hardcoded copy
# is kept here, so the copies cannot drift apart (a prompt / question_bank batch mismatch once made mode_c fail silently).
def _require_eval_env(name: str) -> str:
    """Read the path config the outer script must inject; fail loudly instead of guessing a default."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"[CONFIG ERROR] missing environment variable {name}.\n"
            f"  Dataset paths are centralized at the top of benchmark/run_model_sequence_with_watchdog.sh,\n"
            f"  launch the evaluation through that script; to run this file standalone, export {name}=... first."
        )
    return value


MODEL_NAME= "seedance2.0"
TEST_VIDEO_DIR = _require_eval_env("EVAL_VIDEO_DIR_TEMPLATE").format(model=MODEL_NAME)
PROMPT_FILE = _require_eval_env("EVAL_PROMPT_FILE")
EVAL_RESULT_BASE = os.environ.get("EVAL_RESULT_BASE", "final")
# Root directory for evaluation artifacts; defaults to logs/ next to the repo, EVAL_LOG_ROOT can point to an external disk.
EVAL_LOG_ROOT = os.environ.get("EVAL_LOG_ROOT", str(REPO_ROOT / "logs"))
OUTPUT_DIR = os.path.join(EVAL_LOG_ROOT, EVAL_RESULT_BASE, MODEL_NAME)

# E1/D1 weighting: Mode B x w + Mode C x (1-w)
WEIGHT_MODE_B = 0.5
WEIGHT_MODE_C = 0.5


def get_test_videos():
    """List the test videos."""
    videos = sorted(Path(TEST_VIDEO_DIR).glob("*.mp4"))
    return [str(v) for v in videos]


def load_prompts():
    """Load the prompt config, returning the list and an ID index dict."""
    with open(PROMPT_FILE) as f:
        prompts = json.load(f)
    return prompts


def build_prompt_index(prompts):
    """Build the prompt ID -> prompt index dict."""
    return {p["id"]: p for p in prompts}


def run_mode_a(video_path: str, prompt: dict, alignment_result: dict = None) -> dict:
    """Run the Mode A evaluation.
    
    Args:
        alignment_result: VLM shot alignment result (optional; Mode A computes it internally when absent)
    
    Returns:
        The Mode A result, including the alignment_result field
    """
    from mode_a_eval import evaluate_single_video, TestLogger
    # Note: a per-video md log file is no longer written
    # log_path = os.path.join(OUTPUT_DIR, f"mode_a_{os.path.basename(video_path)}.md")
    log_path = "/dev/null"  # md log output disabled
    logger = TestLogger(log_path=log_path)
    result = evaluate_single_video(video_path, prompt, logger,
                                   alignment_result=alignment_result)
    return result


def run_mode_b(video_path: str, prompt: dict, mode_a_raw=None,
               alignment_result: dict = None) -> dict:
    """Run the Mode B evaluation.
    
    Args:
        alignment_result: VLM shot alignment result (optional; used by the per-shot E1 and the D1 transition evaluation)
    """
    from mode_b_eval import evaluate_mode_b
    result = evaluate_mode_b(video_path, prompt, mode_a_raw,
                             alignment_result=alignment_result)
    return result


def run_mode_c(video_path: str, video_id: int,
               alignment_result: dict = None,
               prompt: dict = None) -> dict:
    """Run the Mode C evaluation.
    
    Args:
        alignment_result: VLM shot alignment result (optional; adds shot information to the A2 event-chain judgement)
    """
    from mode_c_eval import evaluate_mode_c
    result = evaluate_mode_c(video_path, video_id,
                             alignment_result=alignment_result,
                             prompt=prompt)
    return result


def compute_final_dimensions(mode_a: dict, mode_b: dict, mode_c: dict) -> dict:
    """Compute final dimension scores combining all three modes.
    Merge the three modes' scores into the final per-dimension score.

    Rules:
    - A1, B1, B2, D2, D3, E2, F1, F2: from Mode A
    - A4: (Mode A A4 (CLIP style match) + Mode A E3 (DINOv2 style consistency)) / 2
    - A2: the Mode C per-event execution score (already covers shot presence + the VLM execution quality)
    - A3: Mode C legacy QA and counterfactual reverse-order score weighted average
    - B3: from Mode B
    - C1: avg(Mode B C1, Mode C C1)
    - D1: flat mean over all transitions (Mode A+B) x 0.5 + Mode C D1 QA x 0.5
    - E1: avg of Mode B E1 sub-dims × 0.5 + Mode C E1 QA × 0.5
    - E3: Mode C E3 only (physical consistency questions)
    """
    final = {}

    # --- Mode A dimensions (taken directly, except A4 which merges E3_ModeA) ---
    mode_a_scores = mode_a.get("dimension_scores", {})
    for dim in ["A1", "B1", "B2", "D2", "D3", "E2", "F1", "F2"]:
        if dim in mode_a_scores and mode_a_scores[dim] is not None:
            final[dim] = mode_a_scores[dim]

    # --- A4: (Mode A A4 (CLIP style match) + Mode A E3 (DINOv2 style consistency)) / 2 ---
    a4_score = mode_a_scores.get("A4")
    e3_mode_a = mode_a_scores.get("E3")
    if a4_score is not None and e3_mode_a is not None:
        final["A4"] = (a4_score + e3_mode_a) / 2.0
    elif a4_score is not None:
        final["A4"] = a4_score
    elif e3_mode_a is not None:
        final["A4"] = e3_mode_a

    # --- Mode C pure dimensions (A2, A3) ---
    mode_c_scores = mode_c.get("dimension_scores", {})
    for dim in ["A3"]:
        if dim in mode_c_scores:
            final[dim] = mode_c_scores[dim]
    
    # --- A2: per-event execution score (already covers shot presence + content execution quality) ---
    # New logic: the A2 score is the per-event mean computed in Mode C
    # That score already combines:
    #   - a missing shot -> the event scores 0
    #   - a present shot -> the VLM rates the execution: complete=1.0, partial=0.5, failed=0.1
    # so multiplying by shot_execution_rate is no longer needed
    a2_score = mode_c_scores.get("A2")
    if a2_score is not None:
        final["A2"] = a2_score

    # --- C1: avg(Mode B C1, Mode C C1) ---
    mode_b_scores = mode_b.get("dimension_scores", {})
    c1_b = mode_b_scores.get("C1")
    c1_c = mode_c_scores.get("C1")
    if c1_b is not None and c1_c is not None:
        final["C1"] = (c1_b + c1_c) / 2.0
    elif c1_b is not None:
        final["C1"] = c1_b
    elif c1_c is not None:
        final["C1"] = c1_c

    # --- D1: flat sum of every transition score / total transitions x 0.5 + Mode C QA x 0.5 ---
    # Mode A covers: similar_visual, similar_action, similar_audio, contrast, camera_motion
    # Mode B covers: the semantic types (Logical, Causal, POV, Exit/Entry, ...)
    # Aggregation: sum every transition score and divide by the total transition count, scoring 0 for the unevaluated ones
    d1_c = mode_c.get("d1_qa_score")
    
    # Per-transition detail scores from Mode A per_transition
    d1_detail = mode_a.get("detailed_results", {}).get("D1", {})
    per_transition = d1_detail.get("per_transition", [])
    total_transitions = d1_detail.get("num_transitions", 0)
    
    # Collect the Mode A type transition scores (method != None)
    d1_all_scores = []
    for t in per_transition:
        if t.get("method") is not None:  # Mode A type
            score = t.get("score")
            d1_all_scores.append(float(score) if score is not None else 0.0)
    
    # Append the Mode B type transition scores (recovered from D1_mode_b_avg)
    d1_b_info = mode_b.get("dimension_results", {}).get("D1_mode_b_avg")
    if d1_b_info and isinstance(d1_b_info, dict):
        d1_b_avg = d1_b_info.get("score", 0.0)
        d1_b_n = d1_b_info.get("n_transitions", 0)
        d1_b_sum = d1_b_avg * d1_b_n
        d1_all_scores_sum = sum(d1_all_scores) + d1_b_sum
        d1_all_count = len(d1_all_scores) + d1_b_n
    else:
        d1_all_scores_sum = sum(d1_all_scores)
        d1_all_count = len(d1_all_scores)
    
    # Flat mean: total / total transitions (total_transitions makes the unevaluated ones count as 0)
    if total_transitions > 0:
        d1_expert = d1_all_scores_sum / total_transitions
    elif d1_all_count > 0:
        d1_expert = d1_all_scores_sum / d1_all_count
    else:
        d1_expert = None
    
    if d1_expert is not None and d1_c is not None:
        final["D1"] = d1_expert * WEIGHT_MODE_B + d1_c * WEIGHT_MODE_C
    elif d1_expert is not None:
        final["D1"] = d1_expert
    elif d1_c is not None:
        final["D1"] = d1_c

    # --- E1: Mode B E1 avg × 0.5 + Mode C E1 QA × 0.5 ---
    e1_subdims = ["E1-camera_motion", "E1-shot_scale", "E1-angle", "E1-dof"]
    e1_b_scores = [mode_b_scores[d] for d in e1_subdims if d in mode_b_scores]
    e1_b_avg = float(np.mean(e1_b_scores)) if e1_b_scores else None
    e1_c = mode_c.get("e1_qa_score")
    if e1_b_avg is not None and e1_c is not None:
        final["E1"] = e1_b_avg * WEIGHT_MODE_B + e1_c * WEIGHT_MODE_C
    elif e1_b_avg is not None:
        final["E1"] = e1_b_avg
    elif e1_c is not None:
        final["E1"] = e1_c

    # --- E3: Mode C only (physical consistency questions) ---
    e3_c = mode_c_scores.get("E3")
    if e3_c is not None:
        final["E3"] = e3_c

    # --- B3: from Mode B ---
    b3 = mode_b_scores.get("B3")
    if b3 is not None:
        final["B3"] = b3

    return final


def generate_test_report(all_results: list, output_path: str):
    """Generate the full test report (including the all-dimension summary table)."""
    report_lines = []
    report_lines.append("# Mode A + Mode B + Mode C full-dimension joint evaluation report")
    report_lines.append(f"\n**test time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**videos tested**: {len(all_results)}")
    try:
        from mode_b_eval import VLM_MODEL as report_vlm_model
    except Exception:
        report_vlm_model = "unknown"
    report_lines.append(f"**VLM model**: {report_vlm_model} (DashScope)")
    report_lines.append(f"**evaluation modes**: Mode A (expert models) + Mode B (VLM assisted) + Mode C (VLM QA)")
    report_lines.append("")

    # Dimension order
    DIM_ORDER = [
        "A1", "A2", "A3", "A4",
        "B1", "B2", "B3",
        "C1",
        "D1", "D2", "D3",
        "E1", "E2", "E3",
        "F1", "F2",
    ]

    # Per-video details
    for result in all_results:
        video_name = os.path.basename(result["video_path"])
        report_lines.append(f"\n---\n## video: {video_name}")
        report_lines.append(f"- **ID**: {result.get('prompt_id', '?')}")
        report_lines.append(f"- **title**: {result.get('title', 'N/A')}")

        # Shot alignment info
        sa = result.get("shot_alignment")
        if sa:
            report_lines.append(
                f"- **shot alignment**: matched={sa['n_matched']}, merged={sa['n_merged']}, "
                f"missing={sa['n_missing']}, accuracy={sa.get('shot_accuracy', 0):.2f}, "
                f"method={sa.get('method', '?')}"
            )
        else:
            report_lines.append(f"- **shot alignment**: not run (fallback mode)")

        # Final dimension scores
        final_dims = result.get("final_dimensions", {})
        report_lines.append(f"\n### final dimension scores")
        report_lines.append("| dimension | score | source |")
        report_lines.append("|------|------|------|")
        for dim in DIM_ORDER:
            if dim in final_dims:
                score = final_dims[dim]
                source = _get_source_label(dim)
                report_lines.append(f"| {dim} | {score:.4f} | {source} |")

        overall = float(np.mean(list(final_dims.values()))) if final_dims else 0
        report_lines.append(f"\n**overall mean**: {overall:.4f}")

        # Mode C question details
        mode_c = result.get("mode_c", {})
        if mode_c.get("question_details"):
            report_lines.append(f"\n### Mode C answer details")
            report_lines.append("| question | dimension | correct | predicted | score |")
            report_lines.append("|------|------|------|------|------|")
            for qd in mode_c["question_details"]:
                report_lines.append(
                    f"| {qd['question_id']} | {qd['dimension']} | "
                    f"{qd['correct_answer']} | {qd['predicted_answer']} | "
                    f"{qd['score']:.0f} |"
                )

    # Summary table
    report_lines.append("\n\n---\n## full-dimension summary table")
    header = "| video |" + " | ".join(DIM_ORDER) + " | mean |"
    sep = "|------|" + "|".join(["------"] * len(DIM_ORDER)) + "|------|"
    report_lines.append(header)
    report_lines.append(sep)
    for result in all_results:
        name = os.path.basename(result["video_path"])[:25]
        final_dims = result.get("final_dimensions", {})
        row = f"| {name} |"
        scores_list = []
        for dim in DIM_ORDER:
            if dim in final_dims:
                row += f" {final_dims[dim]:.2f} |"
                scores_list.append(final_dims[dim])
            else:
                row += " - |"
        avg = float(np.mean(scores_list)) if scores_list else 0
        row += f" {avg:.2f} |"
        report_lines.append(row)

    # Overall average per dimension
    report_lines.append("")
    dim_totals = {d: [] for d in DIM_ORDER}
    for result in all_results:
        final_dims = result.get("final_dimensions", {})
        for dim in DIM_ORDER:
            if dim in final_dims:
                dim_totals[dim].append(final_dims[dim])

    row = "| **mean** |"
    for dim in DIM_ORDER:
        if dim_totals[dim]:
            row += f" **{np.mean(dim_totals[dim]):.2f}** |"
        else:
            row += " - |"
    all_scores = [s for scores in dim_totals.values() for s in scores]
    row += f" **{np.mean(all_scores):.2f}** |" if all_scores else " - |"
    report_lines.append(row)

    # Shot alignment global statistics
    alignment_stats = [r["shot_alignment"] for r in all_results if r.get("shot_alignment")]
    if alignment_stats:
        report_lines.append("\n\n---\n## shot alignment statistics")
        total_matched = sum(s["n_matched"] for s in alignment_stats)
        total_merged = sum(s["n_merged"] for s in alignment_stats)
        total_missing = sum(s["n_missing"] for s in alignment_stats)
        total_shots = total_matched + total_merged + total_missing
        avg_accuracy = np.mean([s.get("shot_accuracy", 0) for s in alignment_stats])
        methods = {}
        for s in alignment_stats:
            m = s.get("method", "unknown")
            methods[m] = methods.get(m, 0) + 1

        report_lines.append(f"- **videos evaluated**: {len(alignment_stats)}")
        report_lines.append(f"- **total shots (GT)**: {total_shots}")
        report_lines.append(f"- **matched**: {total_matched} ({total_matched/max(total_shots,1)*100:.1f}%)")
        report_lines.append(f"- **fragments merged**: {total_merged} ({total_merged/max(total_shots,1)*100:.1f}%)")
        report_lines.append(f"- **shots missing**: {total_missing} ({total_missing/max(total_shots,1)*100:.1f}%)")
        report_lines.append(f"- **mean alignment accuracy**: {avg_accuracy:.4f}")
        report_lines.append(f"- **alignment method distribution**: {methods}")

    report_text = "\n".join(report_lines)
    with open(output_path, "w") as f:
        f.write(report_text)

    return report_text


def _get_source_label(dim: str) -> str:
    """Get source label for a dimension"""
    source_map = {
        "A1": "Mode A", "A2": "Mode C × shot execution rate", "A3": "Mode C legacy × 0.5 + counterfactual reordering × 0.5", "A4": "Mode A",
        "B1": "Mode A", "B2": "Mode A", "B3": "Mode B",
        "C1": "Mode B×0.5 + Mode C×0.5",
        "D1": "flat mean(A+B)×0.5 + Mode C×0.5",
        "D2": "Mode A", "D3": "Mode A",
        "E1": "Mode B×0.5 + Mode C×0.5",
        "E2": "Mode A", "E3": "Mode A×0.5 + Mode C×0.5",
        "F1": "Mode A", "F2": "Mode A",
    }
    return source_map.get(dim, "Unknown")


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Mode A+B+C joint evaluation")
    parser.add_argument("--ids", nargs="+", type=int, default=None,
                        help="filter test samples by prompt ID, e.g. --ids 1 2 4 6 7")
    parser.add_argument("--range", dest="index_range", type=str, default=None,
                        help="filter by JSON list index range (1-based, inclusive), e.g. --range 1-10 or --range 20-30")
    parser.add_argument("--model", type=str, default=None,
                        help="model name (overrides MODEL_NAME in the script)")
    return parser.parse_args()


def filter_prompts(prompts, ids=None, index_range=None):
    """Filter the prompts by an ID list or an index range.
    
    Args:
        prompts: the full prompt list
        ids: the prompt IDs to keep (e.g. [1, 2, 4, 6, 7])
        index_range: index range string (e.g. "1-10", 1-based and inclusive)
    
    Returns:
        The filtered prompt list
    """
    if ids is None and index_range is None:
        return prompts
    
    selected_indices = set()
    
    # Filter by ID
    if ids is not None:
        id_set = set(ids)
        for i, p in enumerate(prompts):
            if p.get("id") in id_set:
                selected_indices.add(i)
    
    # Filter by index range (1-based)
    if index_range is not None:
        parts = index_range.split("-")
        if len(parts) == 2:
            start = int(parts[0]) - 1  # to 0-based
            end = int(parts[1])        # inclusive right end
            for i in range(max(0, start), min(end, len(prompts))):
                selected_indices.add(i)
        else:
            # A single index
            idx = int(parts[0]) - 1
            if 0 <= idx < len(prompts):
                selected_indices.add(idx)
    
    # Return in the original order
    return [prompts[i] for i in sorted(selected_indices)]


def main():
    args = parse_args()

    # Allow the command line to override the model name
    global MODEL_NAME, TEST_VIDEO_DIR, OUTPUT_DIR
    if args.model:
        MODEL_NAME = args.model
        TEST_VIDEO_DIR = _require_eval_env("EVAL_VIDEO_DIR_TEMPLATE").format(model=MODEL_NAME)
        OUTPUT_DIR = os.path.join(EVAL_LOG_ROOT, EVAL_RESULT_BASE, MODEL_NAME)

    # Create output only when this script is executed directly. run_final_joint_test.py imports
    # helpers from this module and must not create the default model's directory as an import side effect.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Mode A + Mode B + Mode C full-dimension joint evaluation")
    print("=" * 60)

    videos = get_test_videos()
    prompts = load_prompts()
    prompt_index = build_prompt_index(prompts)

    # Filter the samples
    selected_prompts = filter_prompts(prompts, ids=args.ids, index_range=args.index_range)
    selected_ids = {p["id"] for p in selected_prompts}

    if args.ids or args.index_range:
        filter_desc = []
        if args.ids:
            filter_desc.append(f"IDs: {args.ids}")
        if args.index_range:
            filter_desc.append(f"index range: {args.index_range}")
        print(f"\nfilters: {', '.join(filter_desc)}")
        print(f"samples after filtering: {len(selected_prompts)} / {len(prompts)}")
        print(f"selected IDs: {sorted(selected_ids)}")
    else:
        print(f"\ntesting every sample")

    print(f"test video directory: {TEST_VIDEO_DIR}")
    print(f"available videos: {len(videos)}")
    print(f"prompt entries: {len(prompts)}")

    all_results = []

    for i, video_path in enumerate(videos):
        video_name = os.path.basename(video_path)
        video_id = int(video_name.split("_")[0])

        # Skip the samples that were not selected
        if selected_ids and video_id not in selected_ids:
            continue

        # Look the prompt up by ID (not by positional index)
        prompt = prompt_index.get(video_id)
        if prompt is None:
            print(f"\n⚠️  skipping {video_name} (ID={video_id} has no matching prompt)")
            continue

        print(f"\n{'='*60}")
        print(f"[{len(all_results)+1}/{len(selected_prompts)}] {video_name}")
        print(f"  Title: {prompt.get('title', 'N/A')}")
        print(f"  Shots: {prompt.get('number_of_shots', '?')}")
        print(f"{'='*60}")

        result = {
            "video_path": video_path,
            "prompt_id": prompt.get("id"),
            "title": prompt.get("title"),
        }

        # ===== Mode A =====
        print("\n▶ Running Mode A...")
        t0 = time.time()
        try:
            mode_a_result = run_mode_a(video_path, prompt)
            result["mode_a"] = mode_a_result
            print(f"  Mode A done in {time.time()-t0:.1f}s, score={mode_a_result.get('overall_score', 0):.4f}")
        except Exception as e:
            print(f"  Mode A FAILED: {e}")
            mode_a_result = {"overall_score": 0, "error": str(e), "dimension_scores": {}, "raw_outputs": {}}
            result["mode_a"] = mode_a_result

        # ===== Extract alignment result from Mode A =====
        # Mode A already computed the VLM shot alignment: extract it and pass it to Mode B and Mode C
        alignment_result = None
        if isinstance(result.get("mode_a"), dict):
            alignment_result = result["mode_a"].get("alignment_result")
        if alignment_result:
            n_m = alignment_result.get("n_matched", 0)
            n_mg = alignment_result.get("n_merged", 0)
            n_ms = alignment_result.get("n_missing", 0)
            method = alignment_result.get("method", "?")
            print(f"  📐 Shot alignment: matched={n_m}, merged={n_mg}, "
                  f"missing={n_ms}, method={method}")
            result["shot_alignment"] = {
                "n_matched": n_m, "n_merged": n_mg,
                "n_missing": n_ms, "method": method,
                "shot_accuracy": alignment_result.get("shot_accuracy", 0),
            }
        else:
            print(f"  📐 Shot alignment: not available (fallback mode)")
            result["shot_alignment"] = None

        # ===== Mode B =====
        print("\n▶ Running Mode B...")
        t0 = time.time()
        try:
            mode_b_result = run_mode_b(video_path, prompt,
                                       mode_a_result.get("raw_outputs"),
                                       alignment_result=alignment_result)
            result["mode_b"] = mode_b_result
            print(f"  Mode B done in {time.time()-t0:.1f}s, score={mode_b_result.get('overall_score', 0):.4f}")
        except Exception as e:
            print(f"  Mode B FAILED: {e}")
            result["mode_b"] = {"overall_score": 0, "error": str(e), "dimension_scores": {}}

        # ===== Mode C =====
        print("\n▶ Running Mode C...")
        t0 = time.time()
        try:
            mode_c_result = run_mode_c(video_path, video_id,
                                       alignment_result=alignment_result,
                                       prompt=prompt)
            result["mode_c"] = mode_c_result
            c_scores = mode_c_result.get("dimension_scores", {})
            print(f"  Mode C done in {time.time()-t0:.1f}s, scores={c_scores}")
        except Exception as e:
            print(f"  Mode C FAILED: {e}")
            result["mode_c"] = {"error": str(e), "dimension_scores": {},
                                "e1_qa_score": None, "d1_qa_score": None}

        # ===== Compute final dimensions =====
        final_dims = compute_final_dimensions(
            result.get("mode_a", {}),
            result.get("mode_b", {}),
            result.get("mode_c", {})
        )
        result["final_dimensions"] = final_dims
        overall = float(np.mean(list(final_dims.values()))) if final_dims else 0
        print(f"\n  📊 Final dimensions ({len(final_dims)}): overall={overall:.4f}")

        all_results.append(result)

    # Generate the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_joint_test_report_{timestamp}.md")
    json_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_joint_test_results_{timestamp}.json")

    report = generate_test_report(all_results, report_path)

    # Save the detailed JSON result
    with open(json_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  evaluation finished!")
    print(f"  report: {report_path}")
    print(f"  details: {json_path}")
    print(f"{'='*60}")
    print(report)


if __name__ == "__main__":
    main()
