"""
Mode C evaluation module -- VLM question answering (multiple choice / true-false).
Evaluates dimensions: A2, A3, C1, E3 (pure Mode C)
Also provides E1/D1 supplementary questions for weighted scoring with Mode B.

Flow:
1. load the questions from question_bank.json
2. send the video plus the question to the VLM
3. the VLM picks an answer, which is compared with correct_answer
4. 1 point for a correct answer, 0 for a wrong one

The A2 dimension has its own three-step flow:
  Step 1: the VLM picks the best-matching event chain
  -> a wrong pick scores A2 0 immediately and skips the remaining steps
  Step 2: alignment_result decides which shots were executed (present / missing)
  Step 3: for the executed shots, the VLM judges the per-event execution quality (complete / partial / failed)
  Final score: the mean over all events (missing=0, complete=1.0, partial=0.5, failed=0.1)
"""
import os
import sys
import json
from typing import Optional, List

# Directory holding ffmpeg / ffprobe. The versions on PATH are used by default; to pin a dedicated
# environment, export AV_PROCESS_BIN=/path/to/env/bin before running.
AV_PROCESS_BIN = os.environ.get("AV_PROCESS_BIN", "")
if AV_PROCESS_BIN and AV_PROCESS_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = AV_PROCESS_BIN + os.pathsep + os.environ.get("PATH", "")

sys.path.insert(0, os.path.dirname(__file__))
from mode_b_eval import call_vlm, parse_choice, extract_shot_clip, VLMRetryExhausted
from a3_counterfactual_eval import eval_a3_counterfactual

# ============================================================
# Configuration
# ============================================================
# The question bank and prompt paths come solely from the config block at the top of
# benchmark/run_model_sequence_with_watchdog.sh, injected as environment variables; this file no
# longer hardcodes a file name. Note that evaluate_mode_c only records a missing-question error
# into the result JSON without raising, so a prompt / question_bank pair whose video ids do not
# intersect fails silently: always swap the two files together.
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


QUESTION_BANK_PATH = _require_eval_env("EVAL_QUESTION_BANK")
PROMPT_FILE = _require_eval_env("EVAL_PROMPT_FILE")
A3_COUNTERFACTUAL_WEIGHT = float(os.getenv("A3_COUNTERFACTUAL_WEIGHT", "0.5"))

# System prompt for Mode C QA
SYSTEM_QA = (
    "You are a professional film analysis expert. "
    "Watch the video carefully, then answer the question by choosing "
    "the SINGLE best option. Reply with ONLY the option letter (e.g., A, B, C, D, or E). "
    "Do NOT provide any explanation.\n\n"
    "IMPORTANT CONTEXT: These videos are artistic creations that may deliberately "
    "include non-realistic imagery — metaphorical, hallucinatory, surreal, dreamlike, "
    "or psychologically subjective shots (e.g., metaphor montage, contrast montage, "
    "psychological montage). Such stylized or non-literal shots are INTENTIONAL and do "
    "NOT by themselves make the video wrong, unnatural, or physically implausible. "
    "When a question asks whether motion, action, or content is plausible, natural, or "
    "coherent, judge it WITHIN the artistic context and creative intent of the video — "
    "assess whether it is consistent and reasonable for that intended style, rather than "
    "whether it strictly matches literal real-world physics."
)

# ============================================================
# Skill config loading (Plan B: prompts.yaml is the single source of truth)
# ============================================================
# E3's Mode C physical-consistency QA prompt used to share the generic
# SYSTEM_QA constant above. It is now split out into its own SYSTEM_QA_E3,
# loaded from benchmark/skills/e3-event-coherence-skill/prompts.yaml
# (mode_c.system_prompt / mode_c.user_template), so E3's Mode C prompt can
# be tuned independently of A3/D1/E1 without touching this Python file.
# Falls back to the historical SYSTEM_QA text if the skill config is
# missing/broken, with a printed warning, since this module evaluates many
# unrelated dimensions and should not crash entirely because of E3.
try:
    from skill_loader import load_skill
    _e3_mode_c_cfg = load_skill("e3-event-coherence-skill").get("mode_c", {})
    SYSTEM_QA_E3 = _e3_mode_c_cfg["system_prompt"]
    USER_TEMPLATE_E3 = _e3_mode_c_cfg["user_template"]
except Exception as _e3_mode_c_err:
    print(f"[WARN] Failed to load e3-event-coherence-skill mode_c config, "
          f"falling back to generic SYSTEM_QA for E3: {_e3_mode_c_err}")
    SYSTEM_QA_E3 = SYSTEM_QA
    USER_TEMPLATE_E3 = (
        "Question: {question}\n\n"
        "Options:\n{options}\n\n"
        "Answer with ONLY the option letter (A or B):"
    )

# System prompt for A2 event alignment
# The VLM judges the events in order and picks the best-matching event chain
# ============================================================
# Skill config loading: montage-classification-skill (C1 mode_c) +
# event-fidelity-skill (A2 step1/step3). Both are Plan B: prompts.yaml is
# the single source of truth, loaded via benchmark/skill_loader.py.
# ============================================================
try:
    _c1_mode_c_cfg = load_skill("montage-classification-skill").get("mode_c", {})
    SYSTEM_QA_C1 = _c1_mode_c_cfg["system_prompt"]
    USER_TEMPLATE_C1 = _c1_mode_c_cfg["user_template"]
except Exception as _c1_mode_c_err:
    print(f"[WARN] Failed to load montage-classification-skill mode_c config, "
          f"falling back to generic SYSTEM_QA for C1: {_c1_mode_c_err}")
    SYSTEM_QA_C1 = SYSTEM_QA
    USER_TEMPLATE_C1 = (
        "Question: {question}\n\n"
        "Options:\n{options}\n\n"
        "Answer with ONLY the option letter (A/B/C/D):"
    )

try:
    _event_fidelity_cfg = load_skill("event-fidelity-skill")
    SYSTEM_QA_A2 = _event_fidelity_cfg["step1"]["system_prompt"]
    SYSTEM_QA_A2_FIDELITY = _event_fidelity_cfg["step3"]["system_prompt"]
    USER_TEMPLATE_A2_FIDELITY = _event_fidelity_cfg["step3"]["user_template"]
except Exception as _event_fidelity_err:
    print(f"[WARN] Failed to load event-fidelity-skill config, using hardcoded "
          f"fallback prompts for A2: {_event_fidelity_err}")
    SYSTEM_QA_A2 = (
        "You are a professional film editing analyst with expertise in shot segmentation. "
        "Your task is to select the event chain option that BEST matches the ACTUAL shots in the video.\n\n"
        "CRITICAL RULES:\n"
        "1. Watch the video carefully and observe the sequence of events/shots.\n"
        "2. Each option uses ' --> ' to separate events describing a sequence of shots.\n"
        "3. Select the option whose event descriptions match the video content IN ORDER "
        "from the beginning. The best option is the one with the MOST consecutive events "
        "matching the actual video content in the correct sequence.\n"
        "4. Even if no option perfectly matches ALL shots in the video (e.g., the video has fewer "
        "shots than described), choose the option that matches the MOST events from the start "
        "in the correct order.\n"
        "5. Focus on: (a) correct temporal ORDER of events, (b) accurate DESCRIPTION of each event.\n\n"
        "Reply with ONLY the option letter (A/B/C/D). Do NOT provide any explanation."
    )
    SYSTEM_QA_A2_FIDELITY = (
        "You are a professional video content analyst specializing in evaluating "
        "how well a video shot realizes its intended content description.\n\n"
        "Watch the video clip carefully, then evaluate whether the described event/content "
        "was executed in this clip.\n\n"
        "Answer with ONLY the option letter (A/B/C). Do NOT provide any explanation."
    )
    USER_TEMPLATE_A2_FIDELITY = (
        "The intended content for this video clip is:\n"
        "\"{event_desc}\"\n\n"
        "Based on what you actually SEE in the video clip, evaluate the execution quality "
        "of the described content:\n"
        "A. Fully Executed - The described content is completely and accurately shown in this clip\n"
        "B. Partially Executed - Some elements of the described content are present "
        "but incomplete, inaccurate, or only partially realized\n"
        "C. Execution Failed - The described content is barely recognizable or "
        "completely absent from this clip\n\n"
        "Answer with ONLY the option letter (A/B/C):"
    )

try:
    _d1_mode_c_cfg = load_skill("transition-semantics-skill").get("mode_c", {})
    SYSTEM_QA_D1 = _d1_mode_c_cfg["system_prompt"]
    USER_TEMPLATE_D1 = _d1_mode_c_cfg["user_template"]
    D1_SHOT_CONTEXT_TEMPLATE = _d1_mode_c_cfg["shot_context_template"]
except Exception as _d1_mode_c_err:
    print(f"[WARN] Failed to load transition-semantics-skill mode_c config, "
          f"falling back to generic SYSTEM_QA for D1: {_d1_mode_c_err}")
    SYSTEM_QA_D1 = SYSTEM_QA
    USER_TEMPLATE_D1 = (
        "Question: {question}\n\n"
        "{shot_context}Options:\n{options}\n\n"
        "Answer with ONLY the option letter (A/B/C/D/E):"
    )
    D1_SHOT_CONTEXT_TEMPLATE = (
        "[SHOT GENERATION INFO]\n"
        "This video was intended to have {n_gt} shots, but only {n_actual} shots "
        "were actually generated. "
        "If the question asks about a shot number that is GREATER than {n_actual} "
        "(i.e., that shot was NOT generated), you MUST choose option E "
        "('The shot(s) mentioned in this question were not generated in the video')."
    )


# ============================================================
# Core Functions
# ============================================================
def load_question_bank() -> list:
    """Load question bank"""
    with open(QUESTION_BANK_PATH, "r") as f:
        data = json.load(f)
    return data.get("question_bank", [])


def load_prompt_by_id(video_id: int) -> Optional[dict]:
    """Load prompt metadata by video ID for A3 counterfactual evaluation."""
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        for prompt in prompts:
            if prompt.get("id") == video_id:
                return prompt
    except Exception as e:
        print(f"[WARN] Failed to load prompt for A3 counterfactual: {e}")
    return None


def evaluate_single_question(video_path: str, question: dict,
                             alignment_result: dict = None) -> dict:
    """Evaluate a single question with VLM

    Args:
        video_path: Path to the video file
        question: Question dict from question_bank.json
        alignment_result: (Optional) VLM shot alignment result from vlm_shot_alignment module.
            When provided and dimension is A2, alignment info is injected into the prompt
            to give VLM accurate shot count reference.

    Returns:
        dict with score (0 or 1), predicted answer, correct answer, etc.
        For A2 dimension, also includes 'selected_option_text' field.
    """
    q_text = question["question"]
    options = question["options"]
    correct = question["correct_answer"]
    dimension = question["dimension"]
    q_type = question.get("type", "single_choice")

    # Build user text with question and options
    # Build the user prompt text
    options_text = "\n".join(options)

    # For A2 dimension, use specialized system prompt - select best matching chain
    # A2 uses its dedicated system prompt to pick the best-matching event chain
    if dimension == "A2":
        # Build alignment context if available
        # With a VLM alignment result, pass the confirmed shot count and time ranges as extra context
        alignment_context = ""
        if alignment_result is not None:
            n_matched = alignment_result.get("n_matched", 0) + alignment_result.get("n_merged", 0)
            n_missing = alignment_result.get("n_missing", 0)
            aligned_shots = alignment_result.get("aligned_shots", [])

            # Build per-shot time range info for matched/merged shots
            shot_info_lines = []
            for ashot in aligned_shots:
                if ashot["status"] in ("matched", "merged"):
                    tr = ashot.get("time_range")
                    if tr:
                        shot_info_lines.append(
                            f"  Shot {ashot['gt_shot_idx']+1}: {tr[0]:.1f}s - {tr[1]:.1f}s ({ashot['status']})"
                        )
                else:
                    shot_info_lines.append(
                        f"  Shot {ashot['gt_shot_idx']+1}: MISSING (not generated)"
                    )

            alignment_context = (
                f"\n\n[SHOT ANALYSIS RESULT]\n"
                f"A prior shot segmentation analysis has confirmed that this video "
                f"contains {n_matched} effective shots"
                f"{f' ({n_missing} intended shots are missing)' if n_missing > 0 else ''}.\n"
                f"Detected shot time ranges:\n"
                + "\n".join(shot_info_lines) + "\n"
                f"Use this information to help identify which events were actually executed.\n"
            )

        user_text = (
            f"Question: {q_text}\n\n"
            f"IMPORTANT: Each option below describes a chain of events (separated by ' --> '). "
            f"The video may NOT have executed ALL events in the chain. "
            f"Select the option that BEST matches the video — the one whose events "
            f"match the actual video content in the correct ORDER from the beginning, "
            f"with the MOST consecutive events matching.\n\n"
            f"{alignment_context}"
            f"Options:\n{options_text}\n\n"
            f"Answer with ONLY the option letter (A/B/C/D):"
        )
        system_prompt = SYSTEM_QA_A2
    elif dimension == "D1":
        # D1 transition-type identification now has its own dedicated prompt
        # (SYSTEM_QA_D1 / USER_TEMPLATE_D1 / D1_SHOT_CONTEXT_TEMPLATE), loaded
        # from benchmark/skills/transition-semantics-skill/prompts.yaml (mode_c.*).
        # Previously D1 shared the generic SYSTEM_QA with A3/E1.
        shot_context = ""
        if alignment_result is not None:
            n_actual = alignment_result.get("n_matched", 0) + alignment_result.get("n_merged", 0)
            n_gt = alignment_result.get("n_matched", 0) + alignment_result.get("n_merged", 0) + alignment_result.get("n_missing", 0)
            ctx = D1_SHOT_CONTEXT_TEMPLATE.format(n_gt=n_gt, n_actual=n_actual)
            shot_context = f"\n\n{ctx}\n"

        user_text = USER_TEMPLATE_D1.format(
            question=q_text,
            shot_context=shot_context,
            options=options_text,
        )
        system_prompt = SYSTEM_QA_D1
    elif dimension in ("A3", "E1") and alignment_result is not None:
        # A3/E1: these dimensions ask about specific shot numbers
        # When a question refers to a shot number beyond the shots actually generated, the VLM should answer E
        n_actual = alignment_result.get("n_matched", 0) + alignment_result.get("n_merged", 0)
        n_gt = alignment_result.get("n_matched", 0) + alignment_result.get("n_merged", 0) + alignment_result.get("n_missing", 0)

        shot_context = (
            f"\n\n[SHOT GENERATION INFO]\n"
            f"This video was intended to have {n_gt} shots, but only {n_actual} shots "
            f"were actually generated. "
            f"If the question asks about a shot number that is GREATER than {n_actual} "
            f"(i.e., that shot was NOT generated), you MUST choose option E "
            f"('The shot(s) mentioned in this question were not generated in the video').\n"
        )

        user_text = (
            f"Question: {q_text}\n\n"
            f"{shot_context}"
            f"Options:\n{options_text}\n\n"
            f"Answer with ONLY the option letter (A/B/C/D/E):"
        )
        system_prompt = SYSTEM_QA
    elif dimension == "E3":
        # E3 physical-consistency QA now has its own dedicated prompt
        # (SYSTEM_QA_E3 / USER_TEMPLATE_E3), loaded from
        # benchmark/skills/e3-event-coherence-skill/prompts.yaml.
        # E3 questions are true_false (options are only A/B), so the user
        # prompt explicitly asks for "A or B" instead of "A/B/C/D".
        user_text = USER_TEMPLATE_E3.format(
            question=q_text,
            options=options_text,
        )
        system_prompt = SYSTEM_QA_E3
    elif dimension == "C1":
        # C1 montage-classification Mode C QA now has its own dedicated
        # prompt (SYSTEM_QA_C1 / USER_TEMPLATE_C1), loaded from
        # benchmark/skills/montage-classification-skill/prompts.yaml.
        user_text = USER_TEMPLATE_C1.format(
            question=q_text,
            options=options_text,
        )
        system_prompt = SYSTEM_QA_C1
    else:
        user_text = (
            f"Question: {q_text}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Answer with ONLY the option letter (A/B/C/D):"
        )
        system_prompt = SYSTEM_QA

    # Call VLM
    response = call_vlm(system_prompt, user_text, video_path)

    # Parse response to extract choice letter
    valid_options = [opt[0] for opt in options]  # ["A", "B", "C", "D"]
    pred = parse_choice(response, valid_options)

    # Score: 1 if correct, 0 otherwise
    score = 1.0 if pred == correct else 0.0

    result = {
        "question_id": question.get("question_id", ""),
        "dimension": dimension,
        "type": q_type,
        "question": q_text[:100],
        "correct_answer": correct,
        "predicted_answer": pred,
        "vlm_response": response[:200] if response else "",
        "score": score,
    }

    # For A2: also return the selected option text for per-event fidelity evaluation
    # For A2: also return the chosen option text, for the per-event evaluation that follows
    if dimension == "A2" and pred is not None:
        for opt in options:
            if opt.startswith(pred + "."):
                result["selected_option_text"] = opt
                break

    return result


def eval_a2_per_event_fidelity(video_path: str, selected_option_text: str,
                               alignment_result: dict) -> dict:
    """A2 Step 3: Per-event execution quality evaluation.
    A2 step 3: per-event execution quality.

    For every event in the chosen chain:
    - the corresponding shot is missing (alignment says missing) -> that event scores 0
    - the shot exists -> the VLM rates the execution: complete=1.0, partial=0.5, failed=0.1

    Args:
        video_path: Path to the full video
        selected_option_text: The option text VLM selected (e.g., "B. event1 --> event2 --> ...")
        alignment_result: VLM shot alignment result

    Returns:
        dict with per_event_results, avg_score, details
    """
    # Parse events from the selected option text
    # Parse the event list out of the chosen option text
    chain_text = selected_option_text
    # Remove option letter prefix (e.g., "B. ")
    if len(chain_text) > 2 and chain_text[1] == '.':
        chain_text = chain_text[3:].strip()

    events = [e.strip() for e in chain_text.split(' --> ')]

    # Get aligned shots
    aligned_shots = alignment_result.get("aligned_shots", []) if alignment_result else []

    per_event_results = []
    for i, event_desc in enumerate(events):
        # Determine if this shot was generated
        if i < len(aligned_shots):
            shot_info = aligned_shots[i]
            shot_status = shot_info.get("status", "missing")
            time_range = shot_info.get("time_range")
        else:
            shot_status = "missing"
            time_range = None

        if shot_status == "missing" or time_range is None:
            # Event not generated → score = 0
            # The shot for this event was not generated -> score 0
            per_event_results.append({
                "event_idx": i,
                "event_desc": event_desc[:120],
                "status": "not_generated",
                "score": 0.0,
            })
            print(f"      A2 Event {i+1}: 0.0 (shot missing)")
        else:
            # Extract shot clip and ask VLM about execution quality
            # Cut out the shot clip and let the VLM rate the execution
            start_sec, end_sec = time_range[0], time_range[1]
            shot_clip = extract_shot_clip(video_path, start_sec, end_sec)

            user_text = USER_TEMPLATE_A2_FIDELITY.format(event_desc=event_desc)

            response = call_vlm(SYSTEM_QA_A2_FIDELITY, user_text, shot_clip)
            pred = parse_choice(response, ["A", "B", "C"])

            # Map to score
            score_map = {"A": 1.0, "B": 0.5, "C": 0.1}
            score = score_map.get(pred, 0.1)  # Default to 0.1 if parse fails

            per_event_results.append({
                "event_idx": i,
                "event_desc": event_desc[:120],
                "status": "evaluated",
                "vlm_response": response[:100] if response else "",
                "vlm_choice": pred,
                "score": score,
            })
            print(f"      A2 Event {i+1}: {score:.1f} (choice={pred})")

            # Cleanup temp file
            if shot_clip and os.path.exists(shot_clip) and shot_clip != video_path:
                try:
                    os.unlink(shot_clip)
                except OSError:
                    pass

    # Calculate average score
    event_scores = [r["score"] for r in per_event_results]
    avg_score = sum(event_scores) / len(event_scores) if event_scores else 0.0

    return {
        "per_event_results": per_event_results,
        "avg_score": avg_score,
        "n_events": len(events),
        "n_generated": sum(1 for r in per_event_results if r["status"] == "evaluated"),
        "n_missing": sum(1 for r in per_event_results if r["status"] == "not_generated"),
    }


def evaluate_mode_c(video_path: str, video_id: int,
                    alignment_result: dict = None,
                    prompt: dict = None) -> dict:
    """Evaluate all Mode C questions for a given video.
    Answer every Mode C question for the given video.

    The A2 dimension has its own three-step flow:
      Step 1: the VLM picks the best-matching event chain (multiple choice)
      Step 2: alignment decides which shots were executed
      Step 3: the VLM rates the per-event execution quality (complete / partial / failed)
      Final A2 score = the mean of the per-event scores

    Args:
        video_path: Path to video file
        video_id: 1-based video ID (matches question_bank video_id)
        alignment_result: (Optional) VLM shot alignment result.

    Returns:
        dict with:
          - dimension_scores: {A2: score, A3: score, C1: score, E3: score, ...}
          - question_details: list of per-question results
          - a2_fidelity: detailed per-event fidelity results (if A2 evaluated)
          - e1_qa_score: E1 question score (for weighted average with Mode B)
          - d1_qa_score: D1 question score (for weighted average with Mode B)
    """
    question_bank = load_question_bank()
    if prompt is None:
        prompt = load_prompt_by_id(video_id)

    # Find questions for this video
    video_questions = None
    for entry in question_bank:
        if entry["video_id"] == video_id:
            video_questions = entry["questions"]
            break

    if not video_questions:
        return {
            "mode": "C",
            "video_path": video_path,
            "video_id": video_id,
            "error": f"No questions found for video_id={video_id}",
            "dimension_scores": {},
            "question_details": [],
            "a2_fidelity": None,
            "e1_qa_score": None,
            "d1_qa_score": None,
            "a3_counterfactual": None,
            "a3_qa_legacy_score": None,
        }

    # Evaluate each question
    question_details = []
    dimension_scores = {}
    a2_fidelity_result = None
    a3_legacy_scores = []

    for q in video_questions:
        print(f"    Mode C [{q['dimension']}]: ", end="", flush=True)
        # Pass alignment_result for A2/A3/E1/D1 dimension questions
        # A2: supplies the real shot timings to help judge the event chain
        # A3/E1/D1: used to tell whether the shot a question refers to was really generated
        q_alignment = alignment_result if q.get("dimension") in ("A2", "A3", "E1", "D1") else None
        result = evaluate_single_question(video_path, q, alignment_result=q_alignment)
        question_details.append(result)

        dim = result["dimension"]
        score = result["score"]
        pred = result["predicted_answer"]
        correct = result["correct_answer"]
        print(f"{score:.0f} (pred={pred}, gt={correct})")

        # === The old A3 Mode C multiple choice is kept for diagnostics only, no longer the main A3 score ===
        if dim == "A3":
            result["diagnostic_only"] = True
            a3_legacy_scores.append(score)
            continue

        # === A2 special handling: three-step evaluation ===
        # Step 1 is done (the multiple choice above); now check whether the event chain was picked correctly
        if dim == "A2" and pred is not None:
            # A wrong event chain scores 0 outright, with no per-event judging
            if pred != correct:
                print(f"    A2: Event chain WRONG (pred={pred}, gt={correct}) → score=0")
                if "A2" not in dimension_scores:
                    dimension_scores["A2"] = []
                dimension_scores["A2"].append(0.0)
                continue
        
            # The event chain is right, so continue to Step 3: per-event execution quality
            selected_text = result.get("selected_option_text")
            if selected_text and alignment_result is not None:
                print(f"    A2 Step 3: Per-event fidelity evaluation...")
                a2_fidelity_result = eval_a2_per_event_fidelity(
                    video_path, selected_text, alignment_result
                )
                # Final A2 score = mean per-event score (already covers missing=0, complete=1, partial=0.5, failed=0.1)
                a2_final_score = a2_fidelity_result["avg_score"]
                print(f"    A2 Final Score: {a2_final_score:.3f} "
                      f"(generated={a2_fidelity_result['n_generated']}/{a2_fidelity_result['n_events']})")
        
                # Override A2 score with the per-event fidelity score
                # Override the A2 score with the per-event execution score
                if "A2" not in dimension_scores:
                    dimension_scores["A2"] = []
                dimension_scores["A2"].append(a2_final_score)
                continue  # Skip the normal score storage below
            else:
                # Fallback: the VLM picked correctly but there is no alignment_result, so Step 3 is impossible
                # correct pick = 1 point
                print(f"    A2: No alignment available, using selection score only (correct)")

        # Store score per dimension
        if dim not in dimension_scores:
            dimension_scores[dim] = []
        dimension_scores[dim].append(score)

    # Average scores per dimension
    avg_scores = {}
    for dim, scores in dimension_scores.items():
        avg_scores[dim] = sum(scores) / len(scores) if scores else 0.0

    # === A3 counterfactual reverse-order evaluation: fused with the old A3 multiple-choice score ===
    a3_counterfactual_result = None
    a3_qa_legacy_score = sum(a3_legacy_scores) / len(a3_legacy_scores) if a3_legacy_scores else None
    a3_counterfactual_score = None
    a3_weight = max(0.0, min(1.0, A3_COUNTERFACTUAL_WEIGHT))

    if alignment_result is not None and prompt is not None:
        print("    A3 Counterfactual: reorder aligned shots and evaluate...")
        try:
            a3_counterfactual_result = eval_a3_counterfactual(video_path, alignment_result, prompt)
            a3_counterfactual_score = a3_counterfactual_result.get("A3", 0.0)
            print(
                f"    A3 Counterfactual Score: {a3_counterfactual_score:.3f} "
                f"(intra={a3_counterfactual_result.get('intra_score')}, "
                f"inter={a3_counterfactual_result.get('inter_score')}, "
                f"coverage={a3_counterfactual_result.get('chain_coverage'):.3f})"
            )
        except Exception as e:
            if isinstance(e, VLMRetryExhausted):
                raise
            print(f"    A3 Counterfactual FAILED: {e}")
            a3_counterfactual_result = {"A3": 0.0, "error": str(e)}
            a3_counterfactual_score = 0.0

    if a3_qa_legacy_score is not None and a3_counterfactual_score is not None:
        avg_scores["A3"] = (1.0 - a3_weight) * a3_qa_legacy_score + a3_weight * a3_counterfactual_score
        print(
            f"    A3 Final Mixed Score: {avg_scores['A3']:.3f} "
            f"(legacy={a3_qa_legacy_score:.3f}, counterfactual={a3_counterfactual_score:.3f}, "
            f"counterfactual_weight={a3_weight:.2f})"
        )
    elif a3_counterfactual_score is not None:
        avg_scores["A3"] = a3_counterfactual_score
        print("    A3 legacy QA unavailable; using counterfactual score only")
    elif a3_qa_legacy_score is not None:
        avg_scores["A3"] = a3_qa_legacy_score
        print("    A3 Counterfactual unavailable; falling back to legacy A3 QA score")

    # Extract E1 and D1 QA scores for weighted average
    e1_qa_score = avg_scores.get("E1", None)
    d1_qa_score = avg_scores.get("D1", None)

    return {
        "mode": "C",
        "video_path": video_path,
        "video_id": video_id,
        "dimension_scores": avg_scores,
        "question_details": question_details,
        "a2_fidelity": a2_fidelity_result,
        "a3_counterfactual": a3_counterfactual_result,
        "a3_qa_legacy_score": a3_qa_legacy_score,
        "a3_counterfactual_score": a3_counterfactual_score,
        "a3_counterfactual_weight": a3_weight,
        "e1_qa_score": e1_qa_score,
        "d1_qa_score": d1_qa_score,
    }


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    else:
        raise SystemExit("usage: python mode_c_eval.py <video_dir>")
    videos = sorted([
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if f.endswith(".mp4")
    ])

    for video in videos:
        vid_id = int(os.path.basename(video).split("_")[0])
        print(f"\n{'='*50}")
        print(f"Video {vid_id}: {os.path.basename(video)}")
        print(f"{'='*50}")
        result = evaluate_mode_c(video, vid_id)
        print(f"\nDimension scores: {result['dimension_scores']}")
        print(f"E1 QA: {result['e1_qa_score']}, D1 QA: {result['d1_qa_score']}")
