"""Editing agent main loop (agent_loop.py) -- generation + rolling evaluation + central planner repair.

Compared with pipeline.py (the pure multi-stage generation baseline), this entry point closes the loop after a full cut exists:

  Stage1 shot segmentation -> Stage1.5 consistency -> Stage2 prompt rewrite -> Stage3 generation planning
  -> Stage4 shot generation -> Stage5 transition stitching -> [Stage7 rolling evaluation] -> [Stage8 central planner repair]
                                              ^                              |
                                              +---- at most 2 rounds ---------+

The three rolling dimensions (B1 transition timing / D2 transition effect / D3 transition audio-visual
relation) are implemented self-contained in agent_eval/transition_eval.py, and the expert microservices use offset ports (8101/8104/8105/8107 by default), fully isolated from a running formal evaluation.

Repair actions and whether a shot is regenerated:
  - D2 (transition effect): only the effect duration changes -> restitch only, nothing regenerated;
  - B1 (transition timing): the cut time is nudged -> restitch when the material is long enough, regenerate otherwise;
  - D3 (audio-visual relation): audio offset and, if needed, the prompt sound description -> a prompt change always regenerates.

Usage:
  python agent_loop.py --id 1
  python agent_loop.py --ids 1,2 --max-repair-rounds 2
  python agent_loop.py --id 1 --eval-only --eval-video /path/final.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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
from lib.frame_utils import probe_duration  # noqa: E402
from lib.shot_runner import ShotGenerator, compose_case, safe_title, write_json  # noqa: E402
from lib.central_planner import (  # noqa: E402
    CentralPlanner, apply_repair_actions, build_plan_snapshot, create_planner_llm,
)
from lib.audio_gate import (  # noqa: E402
    apply_sound_boost, inspect_shots as inspect_shots_audio, summarize as summarize_gate,
)
from agent_eval.transition_eval import (  # noqa: E402
    DEFAULT_THRESHOLDS, SERVICE_PORTS, ServiceUnavailable,
    evaluate_video, inspect_clip_audio, inspect_transition_audio, require_services,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return resolve_config_paths(yaml.safe_load(f))


def _clip_durations(decisions: List[Dict[str, Any]],
                    generator: ShotGenerator) -> Dict[Any, float]:
    """Probe the real duration of each generated clip, to judge whether the material allows a restitch."""
    durations: Dict[Any, float] = {}
    for dec in decisions:
        path = generator.clip_path_for(dec["shot_id"])
        if Path(path).exists():
            dur = probe_duration(path)
            if dur:
                durations[dec["shot_id"]] = float(dur)
    return durations


def _eval_quality_key(eval_result: Dict[str, Any]) -> tuple:
    """Sort key for cut quality: first the number of passing dimensions, then how far each is above threshold."""
    scores = eval_result.get("scores", {})
    thresholds = eval_result.get("thresholds", DEFAULT_THRESHOLDS)
    n_passed = sum(1 for d, ok in (eval_result.get("passed") or {}).items() if ok)
    ratio = 0.0
    if scores:
        ratio = sum(min(1.0, float(s) / max(float(thresholds.get(d, 1.0)), 1e-6))
                    for d, s in scores.items()) / len(scores)
    raw = sum(float(s) for s in scores.values()) / max(len(scores), 1)
    return (n_passed, round(ratio, 6), round(raw, 6))


def run_audio_gate(decisions: List[Dict[str, Any]], generator: ShotGenerator,
                   gate_cfg: Dict[str, Any], report: Dict[str, Any],
                   stage_label: str, attempt_counter: Dict[Any, int]) -> Dict[str, Any]:
    """D3 pre-check audio gate: measure each clip's loudness, reinforce the prompt and regenerate the silent ones.

    D3 can only hold when a clip really carries sound, so silent clips must be replaced BEFORE D3 runs;
    otherwise no amount of audio offset tuning helps. Regenerations per shot are capped by attempt_counter.
    """
    if not gate_cfg.get("enabled", True):
        return {"enabled": False, "regenerated": []}

    max_attempts = int(gate_cfg.get("max_regen_attempts", 1))
    gate_report = inspect_shots_audio(
        decisions, generator.clip_path_for, inspect_clip_audio, gate_cfg)
    print(f"  ---- [Stage6.5] {stage_label} audio gate (before D3) ----")
    print(f"    {summarize_gate(gate_report)}")

    # only touch silent shots that still have regeneration budget
    eligible = [s for s in gate_report["shots"]
                if s["silent"] and attempt_counter.get(s["shot_id"], 0) < max_attempts]
    exhausted = [s["shot_id"] for s in gate_report["shots"]
                 if s["silent"] and attempt_counter.get(s["shot_id"], 0) >= max_attempts]
    if exhausted:
        print(f"    silent shots that already used up their regeneration budget: {exhausted} (generator capability limit)")

    if not eligible:
        gate_report["regenerated"] = []
        gate_report["exhausted_shot_ids"] = exhausted
        report.setdefault("agent", {}).setdefault("audio_gate", []).append(
            {"stage": stage_label, "n_silent": gate_report["n_silent"],
             "silent_shot_ids": gate_report["silent_shot_ids"],
             "regenerated": [], "exhausted_shot_ids": exhausted})
        return gate_report

    sub_report = {"shots": [s for s in gate_report["shots"]
                            if s["shot_id"] in {e["shot_id"] for e in eligible}]}
    boosted = apply_sound_boost(decisions, sub_report)
    for b in boosted:
        print(f"    shot{b['shot_id']} silent -> reinforcing the sound prompt and regenerating: "
              f"{'; '.join(b['reasons'])[:160]}")

    regenerated = []
    for b in boosted:
        shot_id = b["shot_id"]
        dec = next((d for d in decisions if d["shot_id"] == shot_id), None)
        if dec is None:
            continue
        pos = decisions.index(dec)
        generator.prev_clip = (generator.clip_path_for(decisions[pos - 1]["shot_id"])
                               if pos > 0 else None)
        shot_report = generator.generate_shot(dec)
        shot_report["audio_gate_attempt"] = attempt_counter.get(shot_id, 0) + 1
        report.setdefault("shots", []).append(shot_report)
        attempt_counter[shot_id] = attempt_counter.get(shot_id, 0) + 1
        regenerated.append({"shot_id": shot_id, "ok": shot_report["ok"]})
    generator.prev_clip = (generator.clip_path_for(decisions[-1]["shot_id"])
                           if decisions else None)

    # measure again after regeneration and record the improvement in the report
    after = inspect_shots_audio(decisions, generator.clip_path_for, inspect_clip_audio, gate_cfg)
    print(f"    after regeneration: {summarize_gate(after)}")
    gate_report["regenerated"] = regenerated
    gate_report["exhausted_shot_ids"] = exhausted
    gate_report["after"] = after
    report.setdefault("agent", {}).setdefault("audio_gate", []).append({
        "stage": stage_label,
        "n_silent_before": gate_report["n_silent"],
        "silent_shot_ids_before": gate_report["silent_shot_ids"],
        "regenerated": regenerated,
        "exhausted_shot_ids": exhausted,
        "n_silent_after": after["n_silent"],
        "silent_shot_ids_after": after["silent_shot_ids"],
        "per_shot_after": after["shots"],
    })
    return gate_report


def _build_planner_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build the external tools the central planner may call (the closure captures the current round's state)."""

    def run_transition_eval(dimensions: Optional[List[str]] = None) -> Dict[str, Any]:
        dims = [d.upper() for d in (dimensions or ["B1", "D2", "D3"])]
        cached = state.get("eval_result") or {}
        cached_dims = set((cached.get("dimensions") or {}).keys())
        if cached_dims and set(dims).issubset(cached_dims):
            return {
                "cached": True,
                "video": cached.get("video"),
                "scores": {d: cached["scores"][d] for d in dims if d in cached.get("scores", {})},
                "thresholds": cached.get("thresholds"),
                "passed": {d: cached["passed"][d] for d in dims if d in cached.get("passed", {})},
                "failed_transitions": cached.get("failed_transitions"),
            }
        result = evaluate_video(
            video_path=state["final_path"], decisions=state["decisions"],
            dimensions=dims, use_vlm=state["use_vlm"], thresholds=state["thresholds"],
        )
        return {
            "cached": False, "video": result["video"], "scores": result["scores"],
            "thresholds": result["thresholds"], "passed": result["passed"],
            "failed_transitions": result["failed_transitions"],
        }

    def inspect_transition_audio_tool(transition_index: int) -> Dict[str, Any]:
        decisions = state["decisions"]
        idx = int(transition_index)
        if not (0 <= idx < len(decisions) - 1):
            return {"error": f"transition_index {idx} out of range (0..{len(decisions) - 2})"}
        generator: ShotGenerator = state["generator"]
        prev_clip = generator.clip_path_for(decisions[idx]["shot_id"])
        next_clip = generator.clip_path_for(decisions[idx + 1]["shot_id"])
        trans = decisions[idx].get("transition_out") or {}
        report = inspect_transition_audio(
            prev_clip=prev_clip, next_clip=next_clip,
            timing_offset_seconds=float(trans.get("timing_offset_seconds") or 0),
            audio_relation=(trans.get("audio_visual_relation")
                            or trans.get("audio_relation") or ""),
        )
        # the energy curve is long, so keep only the conclusion and the key metrics for the LLM
        for key in ("outgoing_clip", "incoming_clip"):
            clip_report = report.get(key) or {}
            clip_report.pop("energy_envelope_0p25s", None)
        report["transition_index"] = idx
        report["outgoing_shot_id"] = decisions[idx]["shot_id"]
        report["incoming_shot_id"] = decisions[idx + 1]["shot_id"]
        return report

    def get_current_plan() -> Dict[str, Any]:
        return build_plan_snapshot(state["decisions"], state.get("clip_durations") or {})

    return {
        "run_transition_eval": run_transition_eval,
        "inspect_transition_audio": inspect_transition_audio_tool,
        "get_current_plan": get_current_plan,
    }


def run_case(case: Dict[str, Any], cfg: Dict[str, Any], client,
             planner_llm=None, max_repair_rounds: int = 2,
             dimensions: Optional[List[str]] = None,
             use_vlm: bool = True) -> Dict[str, Any]:
    """Generate + rolling evaluation + repair; returns the agent_report."""
    case_id = case.get("id")
    title = case.get("title", "")
    print(f"\n{'=' * 60}\n  Agent processing case {case_id}: {title}\n{'=' * 60}")

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"]) / str(case_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir = artifacts_dir / "agent_rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg.get("evaluation") or {}
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update({k.upper(): float(v) for k, v in (eval_cfg.get("thresholds") or {}).items()})
    dims = [d.upper() for d in (dimensions or eval_cfg.get("dimensions") or ["B1", "D2", "D3"])]

    report: Dict[str, Any] = {
        "id": case_id, "title": title, "shots": [], "errors": [],
        "audio_references": {}, "visual_references": {},
        "scene_audio_beds": [], "audio_processing": {},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent": {"thresholds": thresholds, "dimensions": dims,
                  "max_repair_rounds": max_repair_rounds,
                  "eval_service_ports": dict(SERVICE_PORTS), "rounds": []},
    }
    audio_cfg = cfg.get("audio") or {}

    # ---- Stage 1~3: identical to the baseline ----
    plan = segment_case(
        case,
        wan_min_duration=cfg.get("wan_min_duration", 2),
        wan_max_duration=cfg.get("wan_max_duration", 15),
        target_duration=cfg.get("target_duration", 15),
    )
    sound_design_cfg = audio_cfg.get("sound_design") or {}
    plan = enrich_sound_plan(plan, enabled=bool(sound_design_cfg.get("enabled", True)))
    print(f"  [Stage1] segmented into {len(plan['shots'])} shots")

    cons_cfg = cfg.get("consistency") or {}
    analysis = None
    if cons_cfg.get("enable", True):
        analysis = analyze_case(plan, client, max_tokens=cons_cfg.get("max_tokens", 4096))
        write_json(analysis, str(artifacts_dir / "consistency.json"))
        print(f"  [Stage1.5] consistency analysis: entities={len(analysis.get('entities', []))} "
              f"settings={len(analysis.get('settings', []))} storylines={len(analysis.get('storylines', []))}")

    plan = rewrite_plan(plan, client, analysis=analysis)
    write_json(plan, str(artifacts_dir / "shot_plan.json"))
    print(f"  [Stage2] prompt rewrite done (LLM={'on' if client else 'off/fallback'})")

    edit_decisions = plan_generation(
        plan, first_frame_continuity_types=cfg.get("first_frame_continuity_types"),
        wan_min_duration=cfg.get("wan_min_duration", 2),
        wan_max_duration=cfg.get("wan_max_duration", 15),
        suppress_clip_bgm=audio_cfg.get("suppress_clip_bgm", True),
        negative_prompt_extra=audio_cfg.get("negative_prompt_extra"),
        speech_reference_only=audio_cfg.get("speech_reference_only", True),
        audio_cfg=audio_cfg,
        single_shot_guard=cfg.get("single_shot_guard"),
        audio_headroom_seconds=float(audio_cfg.get("headroom_seconds", 1.0)),
    )
    plan["scene_audio_beds"] = edit_decisions.get("scene_audio_beds", [])
    report["scene_audio_beds"] = edit_decisions.get("scene_audio_beds", [])
    write_json(edit_decisions, str(artifacts_dir / "edit_decisions.json"))
    decisions = edit_decisions["decisions"]
    print(f"  [Stage3] planning done: {len(decisions)} shots")

    # ---- Stage 4: full generation ----
    print(f"  [Stage4] generating with provider={str(cfg.get('provider', 'wan')).lower()}")
    generator = ShotGenerator(cfg, case_id, artifacts_dir, report)
    clip_paths = generator.generate_all(decisions)
    if not clip_paths:
        report["status"] = "failed"
        report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(report, str(artifacts_dir / "agent_report.json"))
        print("  [Stage5] no usable clip, skipping the stitching and evaluation")
        return report

    ok_ids = [s["shot_id"] for s in report["shots"] if s["ok"]]
    used_decisions = [d for d in decisions if d["shot_id"] in set(ok_ids)]

    # ---- Stage 6.5: D3 pre-check audio gate (regenerate silent clips before stitching and evaluating) ----
    gate_cfg = dict(eval_cfg.get("audio_gate") or {})
    gate_attempts: Dict[Any, int] = {}
    if "D3" in dims:
        run_audio_gate(used_decisions, generator, gate_cfg, report,
                       stage_label="after the first generation", attempt_counter=gate_attempts)

    output_dir = Path(cfg["paths"].get("agent_output_dir") or cfg["paths"]["output_dir"])
    official_final = str(output_dir / f"{case_id}_{safe_title(title)}.mp4")

    def compose_round(round_index: int) -> str:
        out = str(rounds_dir / f"round_{round_index}.mp4")
        current_clips = [generator.clip_path_for(d["shot_id"]) for d in used_decisions]
        compose_case(
            cfg=cfg, case_id=case_id, title=title, plan=plan,
            edit_decisions=edit_decisions, used_decisions=used_decisions,
            clip_paths=current_clips, report=report, artifacts_dir=artifacts_dir,
            final_path=out, work_subdir=f"agent_rounds/work_{round_index}",
        )
        return out

    try:
        final_path = compose_round(0)
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"first compose failed: {e}")
        report["status"] = "compose_failed"
        report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(report, str(artifacts_dir / "agent_report.json"))
        print(f"  [Stage5] first compose failed: {e}")
        return report

    # ---- Stage 5.5: B1/D2 pre-evaluation + quick repair (only needs TransNetV2, so it is cheap) ----
    # ---- Stage 7/8: rolling evaluation + central planner repair ----

    # check TransNetV2 first (B1/D2/D3 all rely on it for cut detection)
    try:
        require_services(["transnetv2"])
    except ServiceUnavailable as e:
        report["errors"].append(str(e))
        report["agent"]["eval_skipped"] = str(e)
        shutil.copyfile(final_path, official_final)
        report["final_video"] = official_final
        report["status"] = "ok_without_eval"
        report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(report, str(artifacts_dir / "agent_report.json"))
        print(f"  [Stage5.5] TransNetV2 unavailable, skipping every evaluation: {e}")
        return report

    state: Dict[str, Any] = {
        "final_path": final_path, "decisions": used_decisions, "generator": generator,
        "use_vlm": use_vlm, "thresholds": thresholds, "eval_result": None,
        "clip_durations": _clip_durations(used_decisions, generator),
    }
    planner = CentralPlanner(planner_llm, _build_planner_tools(state))

    # B1/D2 pre-evaluation: right after stitching, check transition timing and effect with TransNetV2 only.
    # Fixing errors here keeps the shot structure broadly valid before the D3 audio-visual evaluation.
    pre_eval_dims = [d for d in dims if d in ("B1", "D2")]
    pre_eval_max_rounds = 2
    if pre_eval_dims:
        print(f"\n  ---- [Stage5.5] B1/D2 pre-evaluation + quick repair (up to {pre_eval_max_rounds} rounds) ----")
        pre_eval_rounds: List[Dict[str, Any]] = []
        for pre_round in range(pre_eval_max_rounds + 1):
            print(f"    [Pre-eval Round {pre_round}] evaluating {', '.join(pre_eval_dims)} ...")
            try:
                pre_eval_result = evaluate_video(
                    video_path=state["final_path"], decisions=used_decisions,
                    dimensions=pre_eval_dims, use_vlm=False, thresholds=thresholds)
            except Exception as e:  # noqa: BLE001
                report["errors"].append(f"pre-eval round {pre_round} evaluation failed: {e}")
                print(f"    [Pre-eval Round {pre_round}] evaluation failed: {e}")
                break
            state["eval_result"] = pre_eval_result
            write_json(pre_eval_result, str(rounds_dir / f"pre_eval_round_{pre_round}.json"))

            for dim, score in pre_eval_result["scores"].items():
                flag = "PASS" if pre_eval_result["passed"][dim] else "FAIL"
                print(f"      {dim}: {score:.4f} (threshold {thresholds[dim]:.2f}) [{flag}]")

            pre_round_record: Dict[str, Any] = {
                "pre_eval_round": pre_round,
                "video": state["final_path"],
                "scores": pre_eval_result["scores"],
                "passed": pre_eval_result["passed"],
                "failed_dimensions": pre_eval_result["failed_dimensions"],
                "failed_transitions": pre_eval_result["failed_transitions"],
            }

            if not pre_eval_result["failed_dimensions"]:
                pre_round_record["decision"] = "all_b1d2_passed"
                pre_eval_rounds.append(pre_round_record)
                print("      B1/D2 pre-evaluation fully passed")
                break
            if pre_round >= pre_eval_max_rounds:
                pre_round_record["decision"] = "pre_eval_budget_exhausted"
                pre_eval_rounds.append(pre_round_record)
                print(f"      B1/D2 pre-evaluation used up its {pre_eval_max_rounds}-round repair budget")
                break

            # central planner attribution (B1/D2 only)
            print(f"    [Pre-eval] B1/D2 repair round {pre_round + 1}: central planner attribution")
            state["clip_durations"] = _clip_durations(used_decisions, generator)
            plan_snapshot = build_plan_snapshot(used_decisions, state["clip_durations"])
            repair_plan = planner.plan_repairs(pre_eval_result, plan_snapshot,
                                                round_index=pre_round + 1,
                                                max_rounds=pre_eval_max_rounds)
            print(f"      plan source={repair_plan.get('source')} "
                  f"actions={len(repair_plan.get('actions') or [])}")
            if repair_plan.get("reasoning"):
                print(f"      attribution: {str(repair_plan['reasoning'])[:300]}")
            pre_round_record["repair_plan"] = repair_plan

            actions = repair_plan.get("actions") or []
            if not actions:
                pre_round_record["decision"] = "planner_no_action"
                pre_eval_rounds.append(pre_round_record)
                print("      planner proposed no repair action, ending the B1/D2 pre-evaluation")
                break

            apply_result = apply_repair_actions(
                used_decisions, actions, cfg, clip_durations=state["clip_durations"])
            pre_round_record["apply_result"] = apply_result
            print(f"      applied {len(apply_result['applied'])} actions, "
                  f"rejected {len(apply_result['rejected'])}, "
                  f"shots to regenerate={apply_result['regen_shot_ids'] or 'none'}")
            for rej in apply_result["rejected"]:
                print(f"        [rejected] {rej['reason']}")
            if not apply_result["applied"]:
                pre_round_record["decision"] = "all_actions_rejected"
                pre_eval_rounds.append(pre_round_record)
                break

            # regenerate the failed shots (D2-class repairs never reach here)
            regen_ids = apply_result["regen_shot_ids"]
            regen_reports = []
            for shot_id in regen_ids:
                dec = next((d for d in used_decisions if d["shot_id"] == shot_id), None)
                if dec is None:
                    continue
                pos = used_decisions.index(dec)
                generator.prev_clip = (generator.clip_path_for(used_decisions[pos - 1]["shot_id"])
                                       if pos > 0 else None)
                print(f"      [Pre-eval] regenerating shot {shot_id} "
                      f"(gen_duration={dec.get('gen_duration')}s)")
                shot_report = generator.generate_shot(dec)
                shot_report["pre_eval_repair_round"] = pre_round + 1
                report["shots"].append(shot_report)
                regen_reports.append({"shot_id": shot_id, "ok": shot_report["ok"]})
            pre_round_record["regenerated_shots"] = regen_reports
            generator.prev_clip = (generator.clip_path_for(used_decisions[-1]["shot_id"])
                                   if used_decisions else None)

            # a repair-regenerated shot may be silent, so run the audio gate again before restitching
            if regen_ids and "D3" in dims:
                gate_after_pre = run_audio_gate(
                    used_decisions, generator, gate_cfg, report,
                    stage_label=f"after B1/D2 pre-evaluation repair round {pre_round + 1}",
                    attempt_counter=gate_attempts)
                pre_round_record["audio_gate"] = {
                    "n_silent": gate_after_pre.get("n_silent"),
                    "silent_shot_ids": gate_after_pre.get("silent_shot_ids"),
                    "regenerated": gate_after_pre.get("regenerated"),
                }

            write_json(edit_decisions, str(rounds_dir / f"pre_eval_decisions_round_{pre_round + 1}.json"))

            try:
                state["final_path"] = compose_round(100 + pre_round + 1)
                pre_round_record["decision"] = "repaired_and_recomposed"
            except Exception as e:  # noqa: BLE001
                report["errors"].append(f"pre-eval round {pre_round + 1} recompose failed: {e}")
                pre_round_record["decision"] = f"recompose_failed: {e}"
                pre_eval_rounds.append(pre_round_record)
                print(f"      recompose failed: {e}")
                break
            pre_eval_rounds.append(pre_round_record)
            state["eval_result"] = None
            state["clip_durations"] = _clip_durations(used_decisions, generator)

        report["agent"]["pre_eval_rounds"] = pre_eval_rounds
        # update final_path for the later Stage7
        final_path = state["final_path"]

    # check the extra services D3 needs (Whisper / Demucs / PANNs)
    if "D3" in dims:
        try:
            require_services(["whisper", "demucs", "panns"])
        except ServiceUnavailable as e:
            report["errors"].append(str(e))
            report["agent"]["d3_eval_skipped"] = str(e)
            print(f"  [Stage7] D3 services unavailable, skipping the D3 evaluation: {e}")
            dims = [d for d in dims if d != "D3"]

    if not dims:
        shutil.copyfile(final_path, official_final)
        report["final_video"] = official_final
        report["status"] = "ok_without_eval"
        report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(report, str(artifacts_dir / "agent_report.json"))
        print("  [Stage7] no evaluation dimension available, emitting the cut directly")
        return report

    best_final, best_key, best_eval = None, None, None
    for round_index in range(max_repair_rounds + 1):
        print(f"\n  ---- [Stage7] rolling evaluation round {round_index} ({', '.join(dims)}) ----")
        try:
            eval_result = evaluate_video(
                video_path=state["final_path"], decisions=used_decisions,
                dimensions=dims, use_vlm=use_vlm, thresholds=thresholds)
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"round {round_index} evaluation failed: {e}")
            print(f"  [Stage7] evaluation failed: {e}")
            break
        state["eval_result"] = eval_result
        write_json(eval_result, str(rounds_dir / f"eval_round_{round_index}.json"))

        for dim, score in eval_result["scores"].items():
            flag = "PASS" if eval_result["passed"][dim] else "FAIL"
            print(f"    {dim}: {score:.4f} (threshold {thresholds[dim]:.2f}) [{flag}]")

        key = _eval_quality_key(eval_result)
        if best_key is None or key > best_key:
            best_final, best_key, best_eval = state["final_path"], key, eval_result

        round_record: Dict[str, Any] = {
            "round": round_index,
            "video": state["final_path"],
            "scores": eval_result["scores"],
            "passed": eval_result["passed"],
            "failed_dimensions": eval_result["failed_dimensions"],
            "failed_transitions": eval_result["failed_transitions"],
        }

        if not eval_result["failed_dimensions"]:
            round_record["decision"] = "all_dimensions_passed"
            report["agent"]["rounds"].append(round_record)
            print("    every dimension passed, ending the repair loop")
            break
        if round_index >= max_repair_rounds:
            round_record["decision"] = "repair_budget_exhausted"
            report["agent"]["rounds"].append(round_record)
            print(f"    used up the {max_repair_rounds}-round repair budget, keeping the best version")
            break

        # ---- central planner attribution (may call external tools on its own) ----
        print(f"  ---- [Stage8] repair round {round_index + 1}: central planner attribution ----")
        state["clip_durations"] = _clip_durations(used_decisions, generator)
        plan_snapshot = build_plan_snapshot(used_decisions, state["clip_durations"])
        repair_plan = planner.plan_repairs(eval_result, plan_snapshot,
                                          round_index=round_index + 1,
                                          max_rounds=max_repair_rounds)
        print(f"    plan source={repair_plan.get('source')} "
              f"actions={len(repair_plan.get('actions') or [])} "
              f"tool calls={len(repair_plan.get('tool_calls') or [])}")
        if repair_plan.get("reasoning"):
            print(f"    attribution: {str(repair_plan['reasoning'])[:300]}")
        round_record["repair_plan"] = repair_plan

        actions = repair_plan.get("actions") or []
        if not actions:
            round_record["decision"] = "planner_proposed_no_action"
            report["agent"]["rounds"].append(round_record)
            print("    planner proposed no actionable repair, ending the repair loop")
            break

        apply_result = apply_repair_actions(
            used_decisions, actions, cfg, clip_durations=state["clip_durations"])
        round_record["apply_result"] = apply_result
        print(f"    applied {len(apply_result['applied'])} actions, "
              f"rejected {len(apply_result['rejected'])}, "
              f"shots to regenerate={apply_result['regen_shot_ids'] or 'none'}")
        for rej in apply_result["rejected"]:
            print(f"      [rejected] {rej['reason']}")
        if not apply_result["applied"]:
            round_record["decision"] = "all_actions_rejected"
            report["agent"]["rounds"].append(round_record)
            break

        # ---- regenerate only the failed shots (D2-class repairs never reach here) ----
        regen_ids = apply_result["regen_shot_ids"]
        regen_reports = []
        for shot_id in regen_ids:
            dec = next((d for d in used_decisions if d["shot_id"] == shot_id), None)
            if dec is None:
                continue
            pos = used_decisions.index(dec)
            generator.prev_clip = (generator.clip_path_for(used_decisions[pos - 1]["shot_id"])
                                   if pos > 0 else None)
            print(f"    [Stage4-repair] regenerating shot {shot_id} "
                  f"(gen_duration={dec.get('gen_duration')}s)")
            shot_report = generator.generate_shot(dec)
            shot_report["repair_round"] = round_index + 1
            report["shots"].append(shot_report)
            regen_reports.append({"shot_id": shot_id, "ok": shot_report["ok"]})
        round_record["regenerated_shots"] = regen_reports
        # generation state may have changed: reset prev_clip to the last shot to keep sequential semantics
        generator.prev_clip = (generator.clip_path_for(used_decisions[-1]["shot_id"])
                               if used_decisions else None)

        # a repair-regenerated shot may be silent again, so run the audio gate once more before restitching
        if regen_ids and "D3" in dims:
            gate_after_repair = run_audio_gate(
                used_decisions, generator, gate_cfg, report,
                stage_label=f"after repair round {round_index + 1} regeneration",
                attempt_counter=gate_attempts)
            round_record["audio_gate"] = {
                "n_silent": gate_after_repair.get("n_silent"),
                "silent_shot_ids": gate_after_repair.get("silent_shot_ids"),
                "regenerated": gate_after_repair.get("regenerated"),
            }

        write_json(edit_decisions, str(rounds_dir / f"edit_decisions_round_{round_index + 1}.json"))

        try:
            state["final_path"] = compose_round(round_index + 1)
            round_record["decision"] = "repaired_and_recomposed"
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"round {round_index + 1} recompose failed: {e}")
            round_record["decision"] = f"recompose_failed: {e}"
            report["agent"]["rounds"].append(round_record)
            print(f"    recompose failed: {e}")
            break
        report["agent"]["rounds"].append(round_record)
        state["eval_result"] = None
        state["clip_durations"] = _clip_durations(used_decisions, generator)

    # ---- write out the best version ----
    chosen = best_final or state["final_path"]
    Path(official_final).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(chosen, official_final)
    report["final_video"] = official_final
    report["agent"]["chosen_round_video"] = chosen
    report["agent"]["final_scores"] = (best_eval or {}).get("scores")
    report["agent"]["final_passed"] = (best_eval or {}).get("passed")
    report["status"] = "ok"
    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(report, str(artifacts_dir / "agent_report.json"))
    write_json(edit_decisions, str(artifacts_dir / "edit_decisions_final.json"))
    print(f"\n  [Agent] final cut: {official_final}")
    print(f"  [Agent] final scores: {report['agent']['final_scores']}")
    return report


def evaluate_only(cfg: Dict[str, Any], video_path: str, decisions_path: str,
                 dimensions: List[str], use_vlm: bool, output: Optional[str]) -> int:
    """Run the three-dimension evaluation once on an existing cut (debug only: no generation, no repair)."""
    with open(decisions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    decisions = data.get("decisions", data) if isinstance(data, dict) else data
    eval_cfg = cfg.get("evaluation") or {}
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update({k.upper(): float(v) for k, v in (eval_cfg.get("thresholds") or {}).items()})
    result = evaluate_video(video_path=video_path, decisions=decisions,
                            dimensions=dimensions, use_vlm=use_vlm, thresholds=thresholds)
    for dim, score in result["scores"].items():
        flag = "PASS" if result["passed"][dim] else "FAIL"
        print(f"  {dim}: {score:.4f} (threshold {thresholds[dim]:.2f}) [{flag}]")
    if output:
        write_json(result, output)
        print(f"evaluation result written to: {output}")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Edit agent (generation + rolling evaluation + central planner repair)")
    ap.add_argument("--config", default=str(_ROOT / "config.yaml"))
    ap.add_argument("--prompt-json", default=None, help="override the prompt.json path from config")
    ap.add_argument("--ids", default=None, help="comma separated case ids")
    ap.add_argument("--first-n", type=int, default=None, help="process only the first N cases")
    ap.add_argument("--id", type=int, default=None, help="process a single case id")
    ap.add_argument("--output-dir", default=None, help="override the final cut output directory")
    ap.add_argument("--max-repair-rounds", type=int, default=None,
                    help="max repair rounds (defaults to config.agent_loop.max_repair_rounds, usually 2)")
    ap.add_argument("--dimensions", default=None, help="evaluation dimensions, e.g. B1,D2,D3")
    ap.add_argument("--no-vlm", action="store_true", help="skip the VLM for D3, use signal arbitration only")
    ap.add_argument("--eval-only", action="store_true", help="evaluate an existing cut only, no generation or repair")
    ap.add_argument("--eval-video", default=None, help="cut path for --eval-only")
    ap.add_argument("--eval-decisions", default=None, help="edit_decisions.json for --eval-only")
    ap.add_argument("--eval-output", default=None, help="evaluation result output path for --eval-only")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.prompt_json:
        cfg["paths"]["prompt_json"] = args.prompt_json
    if args.output_dir:
        cfg["paths"]["agent_output_dir"] = args.output_dir

    dims = ([d.strip().upper() for d in args.dimensions.split(",") if d.strip()]
            if args.dimensions else None)

    if args.eval_only:
        if not args.eval_video or not args.eval_decisions:
            print("--eval-only requires both --eval-video and --eval-decisions")
            return 1
        return evaluate_only(cfg, args.eval_video, args.eval_decisions,
                             dims or ["B1", "D2", "D3"], not args.no_vlm, args.eval_output)

    loop_cfg = cfg.get("agent_loop") or {}
    max_repair_rounds = (args.max_repair_rounds if args.max_repair_rounds is not None
                         else int(loop_cfg.get("max_repair_rounds", 2)))

    cases = load_cases(cfg["paths"]["prompt_json"])
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
        return 1

    client = create_qwen_client(cfg.get("llm", {}))
    planner_llm = create_planner_llm(cfg.get("planner", {}))

    print(f"cases to process: {len(targets)} | max repair rounds={max_repair_rounds} | "
          f"eval service ports={SERVICE_PORTS}")
    summary = []
    for case in targets:
        try:
            rep = run_case(case, cfg, client, planner_llm=planner_llm,
                           max_repair_rounds=max_repair_rounds,
                           dimensions=dims, use_vlm=not args.no_vlm)
            summary.append((case.get("id"), rep.get("status"),
                            (rep.get("agent") or {}).get("final_scores")))
        except Exception as e:  # noqa: BLE001
            print(f"  case {case.get('id')} raised: {e}")
            summary.append((case.get("id"), "exception", None))

    print(f"\n{'=' * 60}\n  Batch finished")
    for cid, status, scores in summary:
        print(f"    case {cid}: {status}  scores={scores}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
