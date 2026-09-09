"""
Mode A + Mode B + Mode C final joint test with resume support.

This file is the long-running, resumable variant of run_joint_test.py.
It keeps the final JSON structure identical to run_joint_test.py: a list of
per-video result dictionaries. After each completed sample, it rewrites the
resume JSON file by appending the new result. If the same model is restarted,
completed prompt IDs in the resume JSON are skipped and the run continues from
the next unfinished sample.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Directory holding ffmpeg / ffprobe. The versions on PATH are used by default; to pin a dedicated
# environment, export AV_PROCESS_BIN=/path/to/env/bin before running.
AV_PROCESS_BIN = os.environ.get("AV_PROCESS_BIN", "")
if AV_PROCESS_BIN and AV_PROCESS_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = AV_PROCESS_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, os.path.dirname(__file__))

from run_joint_test import (  # noqa: E402
    EVAL_LOG_ROOT,
    _require_eval_env,
    build_prompt_index,
    compute_final_dimensions,
    filter_prompts,
    generate_test_report,
    load_prompts,
    run_mode_a,
    run_mode_b,
    run_mode_c,
)

# ============================================================
# Configuration
# ============================================================
# The video directory and dataset paths come solely from the config block at the top of
# benchmark/run_model_sequence_with_watchdog.sh, injected as environment variables; no hardcoded copy is kept here.
# MODEL_NAME is still switched by the outer script's set_model_name(), which rewrites the line below.
MODEL_NAME= "seedance2.5"
TEST_VIDEO_DIR = _require_eval_env("EVAL_VIDEO_DIR_TEMPLATE").format(model=MODEL_NAME)

# Same source as run_joint_test.PROMPT_FILE (both read EVAL_PROMPT_FILE), so they can no longer drift.
# Note: load_prompts(), which actually loads the prompt, lives in run_joint_test and uses that module's constant;
# this copy exists only for logging / report echoing.
PROMPT_FILE = _require_eval_env("EVAL_PROMPT_FILE")
# Output root segment: the outer script (run_model_sequence_with_watchdog.sh) overrides it by exporting
# EVAL_RESULT_BASE; running this script standalone falls back to the default "final".
EVAL_RESULT_BASE = os.environ.get("EVAL_RESULT_BASE", "final")
_EVAL_SEED_NAME = os.environ.get("EVAL_SEED_NAME")
if _EVAL_SEED_NAME:
    OUTPUT_DIR = os.path.join(EVAL_LOG_ROOT, EVAL_RESULT_BASE, _EVAL_SEED_NAME, MODEL_NAME)
else:
    OUTPUT_DIR = os.path.join(EVAL_LOG_ROOT, EVAL_RESULT_BASE, MODEL_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_test_videos():
    """List the test videos."""
    videos = sorted(Path(TEST_VIDEO_DIR).glob("*.mp4"))
    return [str(v) for v in videos]


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Mode A+B+C joint evaluation (resumable)")
    parser.add_argument("--ids", nargs="+", type=int, default=None,
                        help="filter test samples by prompt ID, e.g. --ids 1 2 4 6 7")
    parser.add_argument("--range", dest="index_range", type=str, default=None,
                        help="filter by JSON list index range (1-based, inclusive), e.g. --range 1-10 or --range 20-30")
    parser.add_argument("--model", type=str, default=None,
                        help="model name (overrides MODEL_NAME in the script)")
    parser.add_argument("--resume-json", type=str, default=None,
                        help="resume JSON path; defaults to {MODEL_NAME}_joint_test_results_resume.json under the output directory")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any existing resume file, start from scratch and overwrite the resume JSON")
    parser.add_argument("--status-only", action="store_true",
                        help="only scan the resume JSON and write the sample status file, without running the evaluation")
    return parser.parse_args()


def _atomic_write_json(path: str, data) -> None:
    """Atomic JSON write, so an interruption cannot leave a half-written file."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    os.replace(tmp_path, path)


def _load_resume_results(path: str) -> list:
    """Read the staged results; the format must match the final JSON, i.e. list[dict]."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Resume JSON must be a list with the same structure as final results: {path}")
    return data


def _dedupe_results(results: list) -> list:
    """Deduplicate by prompt_id, keeping the last result and the first-seen order."""
    order = []
    by_id = {}
    for item in results:
        pid = item.get("prompt_id") if isinstance(item, dict) else None
        if pid is None:
            continue
        if pid not in by_id:
            order.append(pid)
        by_id[pid] = item
    return [by_id[pid] for pid in order]


def _is_complete_result(result: dict) -> bool:
    """Only a sample that fully succeeded in all three modes may be skipped when resuming."""
    if not isinstance(result, dict) or result.get("prompt_id") is None:
        return False
    for mode_key in ("mode_a", "mode_b", "mode_c"):
        mode_result = result.get(mode_key)
        if not isinstance(mode_result, dict) or mode_result.get("error"):
            return False
    final_dims = result.get("final_dimensions")
    return isinstance(final_dims, dict) and bool(final_dims)


def _make_resume_path() -> str:
    return os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_joint_test_results_resume.json")


def _make_status_path() -> str:
    return os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_sample_status.json")


def _result_error(result: dict) -> str:
    """Extract a sample's failure reason, for the status file."""
    if not isinstance(result, dict):
        return "invalid result"
    errors = []
    for mode_key in ("mode_a", "mode_b", "mode_c"):
        mode_result = result.get(mode_key)
        if not isinstance(mode_result, dict):
            errors.append(f"{mode_key}: missing")
        elif mode_result.get("error"):
            errors.append(f"{mode_key}: {mode_result.get('error')}")
    if not result.get("final_dimensions"):
        errors.append("final_dimensions: missing")
    return "; ".join(errors)


def _build_status_payload(target_video_items: list, raw_results: list,
                          running_id: int = None, resume_path: str = None,
                          filter_info: dict = None) -> dict:
    """Build the success/failed/running/pending status of every sample."""
    now = datetime.now().isoformat(timespec="seconds")
    by_id = {
        item.get("prompt_id"): item
        for item in _dedupe_results(raw_results)
        if isinstance(item, dict) and item.get("prompt_id") is not None
    }
    samples = []
    summary = {"success": 0, "failed": 0, "running": 0, "pending": 0}
    for video_path, video_id, prompt in target_video_items:
        result = by_id.get(video_id)
        if running_id == video_id:
            state = "running"
            error = None
        elif result is None:
            state = "pending"
            error = None
        elif _is_complete_result(result):
            state = "success"
            error = None
        else:
            state = "failed"
            error = _result_error(result)
        summary[state] += 1
        samples.append({
            "prompt_id": video_id,
            "video_name": os.path.basename(video_path),
            "title": prompt.get("title"),
            "state": state,
            "error": error,
            "updated_at": now,
        })
    return {
        "model_name": MODEL_NAME,
        "generated_at": now,
        "test_video_dir": TEST_VIDEO_DIR,
        "resume_json": resume_path or _make_resume_path(),
        "filter": filter_info or {},
        "total_samples": len(target_video_items),
        "summary": summary,
        "samples": samples,
    }


def _write_status_file(path: str, target_video_items: list,
                       raw_results: list, running_id: int = None,
                       resume_path: str = None, filter_info: dict = None) -> dict:
    payload = _build_status_payload(
        target_video_items, raw_results, running_id=running_id,
        resume_path=resume_path, filter_info=filter_info,
    )
    _atomic_write_json(path, payload)
    return payload


def _print_status_summary(payload: dict, status_path: str) -> None:
    summary = payload.get("summary", {})
    print(
        "status file: "
        f"success={summary.get('success', 0)}, "
        f"failed={summary.get('failed', 0)}, "
        f"running={summary.get('running', 0)}, "
        f"pending={summary.get('pending', 0)} -> {status_path}"
    )


def main():
    args = parse_args()

    # Allow the command line to override the model name
    global MODEL_NAME, TEST_VIDEO_DIR, OUTPUT_DIR
    if args.model:
        MODEL_NAME = args.model
        TEST_VIDEO_DIR = _require_eval_env("EVAL_VIDEO_DIR_TEMPLATE").format(model=MODEL_NAME)
        if _EVAL_SEED_NAME:
            OUTPUT_DIR = os.path.join(EVAL_LOG_ROOT, EVAL_RESULT_BASE, _EVAL_SEED_NAME, MODEL_NAME)
        else:
            OUTPUT_DIR = os.path.join(EVAL_LOG_ROOT, EVAL_RESULT_BASE, MODEL_NAME)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    resume_path = args.resume_json or _make_resume_path()
    status_path = _make_status_path()

    print("=" * 60)
    print("  Mode A + Mode B + Mode C full-dimension joint evaluation (resumable)")
    print("=" * 60)

    videos = get_test_videos()
    prompts = load_prompts()
    prompt_index = build_prompt_index(prompts)

    # Filter the samples
    selected_prompts = filter_prompts(prompts, ids=args.ids, index_range=args.index_range)
    selected_ids = {p["id"] for p in selected_prompts}
    filter_info = {
        "ids": args.ids,
        "index_range": args.index_range,
        "is_filtered": bool(args.ids or args.index_range),
    }

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
        print("\ntesting every sample")

    print(f"model under test: {MODEL_NAME}")
    print(f"test video directory: {TEST_VIDEO_DIR}")
    print(f"available videos: {len(videos)}")
    print(f"prompt entries: {len(prompts)}")
    print(f"resume JSON: {resume_path}")
    print(f"status JSON: {status_path}")

    if args.no_resume:
        raw_resume_results = []
        all_results = []
        dropped_incomplete = 0
        print("resume: disabled, starting from scratch and overwriting the resume file")
    else:
        raw_resume_results = _dedupe_results(_load_resume_results(resume_path))
        all_results = [r for r in raw_resume_results if _is_complete_result(r)]
        dropped_incomplete = len(raw_resume_results) - len(all_results)
        print(f"resume: loaded {len(all_results)} complete staged results")
        if dropped_incomplete:
            print(f"resume: found {dropped_incomplete} incomplete/failed staged results, those samples will be re-run")

    completed_ids = {r.get("prompt_id") for r in all_results if isinstance(r, dict)}
    if all_results and not args.status_only:
        _atomic_write_json(resume_path, all_results)

    target_video_items = []
    for video_path in videos:
        video_name = os.path.basename(video_path)
        try:
            video_id = int(video_name.split("_")[0])
        except ValueError:
            print(f"\n⚠️  skipping {video_name} (cannot parse the ID from the filename)")
            continue
        if selected_ids and video_id not in selected_ids:
            continue
        prompt = prompt_index.get(video_id)
        if prompt is None:
            print(f"\n⚠️  skipping {video_name} (ID={video_id} has no matching prompt)")
            continue
        target_video_items.append((video_path, video_id, prompt))

    total_targets = len(target_video_items)
    status_payload = _write_status_file(
        status_path, target_video_items, raw_resume_results,
        resume_path=resume_path, filter_info=filter_info,
    )
    print(f"target samples to evaluate: {total_targets}")
    print(f"already completed samples: {len(completed_ids & {item[1] for item in target_video_items})}")
    _print_status_summary(status_payload, status_path)
    if args.status_only:
        print("status-only: only refreshing the status file, not running the evaluation")
        return

    for video_path, video_id, prompt in target_video_items:
        video_name = os.path.basename(video_path)

        if video_id in completed_ids:
            print(f"\n⏭️  skipping the already completed sample: {video_name} (ID={video_id})")
            continue

        done_count = len(completed_ids & {item[1] for item in target_video_items})
        print(f"\n{'='*60}")
        print(f"[{done_count + 1}/{total_targets}] {video_name}")
        print(f"  Title: {prompt.get('title', 'N/A')}")
        print(f"  Shots: {prompt.get('number_of_shots', '?')}")
        print(f"{'='*60}")

        result = {
            "video_path": video_path,
            "prompt_id": prompt.get("id"),
            "title": prompt.get("title"),
        }
        _write_status_file(
            status_path, target_video_items, all_results, running_id=video_id,
            resume_path=resume_path, filter_info=filter_info,
        )

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
        alignment_result = None
        if isinstance(result.get("mode_a"), dict):
            alignment_result = result["mode_a"].get("alignment_result")
        if alignment_result:
            n_m = alignment_result.get("n_matched", 0)
            n_mg = alignment_result.get("n_merged", 0)
            n_ms = alignment_result.get("n_missing", 0)
            method = alignment_result.get("method", "?")
            print(f"  📐 Shot alignment: matched={n_m}, merged={n_mg}, missing={n_ms}, method={method}")
            result["shot_alignment"] = {
                "n_matched": n_m, "n_merged": n_mg,
                "n_missing": n_ms, "method": method,
                "shot_accuracy": alignment_result.get("shot_accuracy", 0),
            }
        else:
            print("  📐 Shot alignment: not available (fallback mode)")
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
        all_results = _dedupe_results(all_results)
        if _is_complete_result(result):
            completed_ids.add(video_id)
        _atomic_write_json(resume_path, all_results)
        status_payload = _write_status_file(
            status_path, target_video_items, all_results,
            resume_path=resume_path, filter_info=filter_info,
        )
        print(f"  💾 staged {len(all_results)} results -> {resume_path}")
        _print_status_summary(status_payload, status_path)

    # Generate the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_joint_test_report_{timestamp}.md")
    json_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_joint_test_results_{timestamp}.json")

    report = generate_test_report(all_results, report_path)

    # Save the final detailed JSON, with exactly the same structure as the staged JSON
    _atomic_write_json(json_path, all_results)

    print(f"\n{'='*60}")
    print("  evaluation finished!")
    print(f"  report: {report_path}")
    print(f"  details: {json_path}")
    print(f"  staged: {resume_path}")
    print(f"  status: {status_path}")
    print(f"{'='*60}")
    print(report)


if __name__ == "__main__":
    main()
