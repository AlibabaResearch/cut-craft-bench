"""Main orchestrator of the editing baseline agent (pipeline.py).

Chains 6 stages, one case at a time:
  Stage1 shot segmentation -> Stage2 per-shot prompt rewrite with Qwen3.7 -> Stage3 generation planning
  -> Stage4 WAN shot generation (t2v / i2v for wipe-by transitions) -> Stage5 ffmpeg transition stitching
  -> Stage6 final cut + render_report

Usage:
  python pipeline.py --ids 1,2        # process the given case ids
  python pipeline.py --first-n 3      # process the first N cases
  python pipeline.py --id 1 --dry-run # only segmentation/rewrite/planning, no real video generation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Allow running this file directly as a script (put the package root on sys.path)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.shot_segmenter import segment_case, load_cases, find_case_by_id  # noqa: E402
from lib.config_paths import resolve_config_paths  # noqa: E402
from lib.llm_client import create_qwen_client  # noqa: E402
from lib.consistency import analyze_case  # noqa: E402
from lib.prompt_rewriter import rewrite_plan  # noqa: E402
from lib.generation_planner import plan_generation  # noqa: E402
from lib.sound_planner import enrich_sound_plan  # noqa: E402
from lib.shot_runner import ShotGenerator, compose_case, write_json as _write_json  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return resolve_config_paths(yaml.safe_load(f))


def process_case(
    case: Dict[str, Any],
    cfg: Dict[str, Any],
    client,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Process a single case and return its render_report."""
    case_id = case.get("id")
    title = case.get("title", "")
    print(f"\n{'='*60}\n  Processing case {case_id}: {title}\n{'='*60}")

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"]) / str(case_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "id": case_id, "title": title, "shots": [], "errors": [],
        "audio_references": {}, "visual_references": {},
        "scene_audio_beds": [], "audio_processing": {},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    audio_cfg = cfg.get("audio") or {}

    # ---- Stage 1: shot segmentation ----
    plan = segment_case(
        case,
        wan_min_duration=cfg.get("wan_min_duration", 2),
        wan_max_duration=cfg.get("wan_max_duration", 15),
        target_duration=cfg.get("target_duration", 15),
    )
    sound_design_cfg = audio_cfg.get("sound_design") or {}
    plan = enrich_sound_plan(plan, enabled=bool(sound_design_cfg.get("enabled", True)))
    _write_json(plan, str(artifacts_dir / "shot_plan.json"))
    print(f"  [Stage1] segmented into {len(plan['shots'])} shots, sound-event planning={'on' if sound_design_cfg.get('enabled', True) else 'off'}")

    # ---- Stage 1.5: global consistency anchoring + narrative structure analysis ----
    cons_cfg = cfg.get("consistency") or {}
    if cons_cfg.get("enable", True):
        analysis = analyze_case(
            plan, client, max_tokens=cons_cfg.get("max_tokens", 4096),
        )
        _write_json(analysis, str(artifacts_dir / "consistency.json"))
        n_ent = len(analysis.get("entities", []))
        n_set = len(analysis.get("settings", []))
        n_line = len(analysis.get("storylines", []))
        tag = "fallback" if analysis.get("_fallback") else "LLM"
        print(f"  [Stage1.5] consistency analysis({tag}): entities={n_ent} settings={n_set} storylines={n_line}")
    else:
        analysis = None
        print("  [Stage1.5] consistency analysis is disabled")

    # ---- Stage 2: per-shot prompt rewrite with Qwen3.7 (injects the consistency bible + logical predecessor shot) ----
    plan = rewrite_plan(plan, client, analysis=analysis)
    _write_json(plan, str(artifacts_dir / "shot_plan.json"))
    print(f"  [Stage2] prompt rewrite done (LLM={'on' if client else 'off/fallback'})")

    # ---- Stage 3: generation planning ----
    edit_decisions = plan_generation(
        plan, first_frame_continuity_types=cfg.get("first_frame_continuity_types"),
        wan_min_duration=cfg.get("wan_min_duration", 2),
        wan_max_duration=cfg.get("wan_max_duration", 15),
        suppress_clip_bgm=(audio_cfg.get("suppress_clip_bgm", True)),
        negative_prompt_extra=audio_cfg.get("negative_prompt_extra"),
        speech_reference_only=audio_cfg.get("speech_reference_only", True),
        audio_cfg=audio_cfg,
        single_shot_guard=cfg.get("single_shot_guard"),
        audio_headroom_seconds=float(audio_cfg.get("headroom_seconds", 1.0)),
    )
    plan["scene_audio_beds"] = edit_decisions.get("scene_audio_beds", [])
    report["scene_audio_beds"] = edit_decisions.get("scene_audio_beds", [])
    _write_json(plan, str(artifacts_dir / "shot_plan.json"))
    _write_json(edit_decisions, str(artifacts_dir / "edit_decisions.json"))
    n_i2v = sum(1 for d in edit_decisions["decisions"] if d["gen_mode"] == "i2v")
    print(f"  [Stage3] planning done: {len(edit_decisions['decisions'])} shots, i2v(wipe-by continuity)={n_i2v}")

    if dry_run:
        print("  [dry-run] skipping video generation and stitching")
        report["dry_run"] = True
        report["edit_decisions"] = edit_decisions
        _write_json(report, str(artifacts_dir / "render_report.json"))
        return report

    # ---- Stage 4: shot generation ----
    print(f"  [Stage4] generating with provider={str(cfg.get('provider', 'wan')).lower()}")
    generator = ShotGenerator(cfg, case_id, artifacts_dir, report)
    clip_paths = generator.generate_all(edit_decisions["decisions"])

    if not clip_paths:
        report["status"] = "failed"
        report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_json(report, str(artifacts_dir / "render_report.json"))
        print("  [Stage5] no usable clip, skipping the stitching")
        return report

    # ---- Stage 5: transition stitching ----
    # only successfully generated clips and their matching decisions
    ok_ids = {s["shot_id"] for s in report["shots"] if s["ok"]}
    used_decisions = [d for d in edit_decisions["decisions"] if d["shot_id"] in ok_ids]
    try:
        compose_case(
            cfg=cfg, case_id=case_id, title=title, plan=plan,
            edit_decisions=edit_decisions, used_decisions=used_decisions,
            clip_paths=clip_paths, report=report, artifacts_dir=artifacts_dir,
        )
        report["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"compose failed: {e}")
        report["status"] = "compose_failed"
        print(f"  [Stage5] compose failed: {e}")

    # ---- Stage 6: render_report ----
    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(report, str(artifacts_dir / "render_report.json"))
    return report


def parse_args():
    ap = argparse.ArgumentParser(description="Edit baseline agent main orchestrator")
    ap.add_argument("--config", default=str(_ROOT / "config.yaml"))
    ap.add_argument("--prompt-json", default=None, help="override the prompt.json path from config")
    ap.add_argument("--ids", default=None, help="comma separated case ids, e.g. 1,2,3")
    ap.add_argument("--first-n", type=int, default=None, help="process only the first N cases")
    ap.add_argument("--id", type=int, default=None, help="process a single case id")
    ap.add_argument("--output-dir", default=None, help="override the output directory from config")
    ap.add_argument("--dry-run", action="store_true", help="run segmentation/rewrite/planning only, no video generation")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.prompt_json:
        cfg["paths"]["prompt_json"] = args.prompt_json
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir

    cases = load_cases(cfg["paths"]["prompt_json"])

    # pick the cases to process
    if args.id is not None:
        targets = [find_case_by_id(cases, args.id)]
    elif args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        targets = [c for c in cases if c.get("id") in want]
    elif args.first_n:
        targets = cases[:args.first_n]
    else:
        targets = cases
    targets = [c for c in targets if c]
    if not targets:
        print("no case matched")
        return

    # LLM client (may be None -> fall back to template assembly)
    client = create_qwen_client(cfg.get("llm", {})) if not args.dry_run or cfg.get("llm", {}).get("enable") else None

    print(f"cases to process: {len(targets)} | dry_run={args.dry_run}")
    summary = []
    for case in targets:
        try:
            rep = process_case(case, cfg, client, dry_run=args.dry_run)
            summary.append((case.get("id"), rep.get("status", "dry-run"), len(rep.get("errors", []))))
        except Exception as e:  # noqa: BLE001
            print(f"  case {case.get('id')} raised: {e}")
            summary.append((case.get("id"), "exception", 1))

    print(f"\n{'='*60}\n  Batch finished")
    for cid, status, nerr in summary:
        print(f"    case {cid}: {status} (errors={nerr})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
