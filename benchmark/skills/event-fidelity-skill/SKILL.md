---
name: event-fidelity-skill
description: Evaluates A2 event fidelity with the existing Mode C three-step VLM workflow: event-chain selection, shot-alignment missing-shot handling, and per-event execution quality scoring. Use when working on A2 consistency, event execution scoring, missing generated shots, or Mode C event-chain QA prompts.
---

# Event Fidelity Skill

## Runtime Status: WIRED (Plan B)

`benchmark/skills/event-fidelity-skill/prompts.yaml` is the single source
of truth for both VLM steps:
- **step1** (`step1.system_prompt`): loaded as `SYSTEM_QA_A2`. The user
  prompt for this step stays in Python (it dynamically injects the
  alignment_context block).
- **step3** (`step3.system_prompt` / `step3.user_template`): loaded as
  `SYSTEM_QA_A2_FIDELITY` / `USER_TEMPLATE_A2_FIDELITY`, used directly in
  `eval_a2_per_event_fidelity()`.

Both are loaded via `benchmark/skill_loader.py` at import time in
`mode_c_eval.py`, falling back to the historical hardcoded text with a
printed warning if the YAML is missing/broken.

## Purpose

Use this skill for the current A2 event execution / event fidelity VLM evaluation in Bench-A-V-Phys. It wraps the prompts and scoring rules currently implemented in `benchmark/mode_c_eval.py`.

## Source Code

- `benchmark/mode_c_eval.py`
- System prompts: `SYSTEM_QA_A2`, `SYSTEM_QA_A2_FIDELITY`
- Functions: `evaluate_single_question(...)`, `eval_a2_per_event_fidelity(...)`

## Current A2 Workflow

A2 uses a three-step evaluation flow:

1. VLM selects the event-chain option that best matches the actual video.
2. Shot alignment determines whether each intended shot exists or is missing.
3. For each generated shot, VLM scores execution fidelity as Fully / Partially / Failed.

Missing shots directly receive `0.0`. Generated shots receive VLM quality scores.

## Step 1 System Prompt: Event Chain Selection

```text
You are a professional film editing analyst with expertise in shot segmentation. Your task is to select the event chain option that BEST matches the ACTUAL shots in the video.

CRITICAL RULES:
1. Watch the video carefully and observe the sequence of events/shots.
2. Each option uses ' --> ' to separate events describing a sequence of shots.
3. Select the option whose event descriptions match the video content IN ORDER from the beginning. The best option is the one with the MOST consecutive events matching the actual video content in the correct sequence.
4. Even if no option perfectly matches ALL shots in the video (e.g., the video has fewer shots than described), choose the option that matches the MOST events from the start in the correct order.
5. Focus on: (a) correct temporal ORDER of events, (b) accurate DESCRIPTION of each event.

Reply with ONLY the option letter (A/B/C/D). Do NOT provide any explanation.
```

## Step 1 User Prompt Template

```text
Question: {q_text}

IMPORTANT: Each option below describes a chain of events (separated by ' --> '). The video may NOT have executed ALL events in the chain. Select the option that BEST matches the video — the one whose events match the actual video content in the correct ORDER from the beginning, with the MOST consecutive events matching.

{alignment_context}
Options:
{options_text}

Answer with ONLY the option letter (A/B/C/D):
```

## Alignment Context Template

When `alignment_result` is available, inject this context:

```text
[SHOT ANALYSIS RESULT]
A prior shot segmentation analysis has confirmed that this video contains {n_matched} effective shots{missing_suffix}.
Detected shot time ranges:
  Shot {gt_shot_idx+1}: {start:.1f}s - {end:.1f}s ({status})
  Shot {gt_shot_idx+1}: MISSING (not generated)
Use this information to help identify which events were actually executed.
```

## Step 3 System Prompt: Per-Event Fidelity

```text
You are a professional video content analyst specializing in evaluating how well a video shot realizes its intended content description.

Watch the video clip carefully, then evaluate whether the described event/content was executed in this clip.

Answer with ONLY the option letter (A/B/C). Do NOT provide any explanation.
```

## Step 3 User Prompt Template

```text
The intended content for this video clip is:
"{event_desc}"

Based on what you actually SEE in the video clip, evaluate the execution quality of the described content:
A. Fully Executed - The described content is completely and accurately shown in this clip
B. Partially Executed - Some elements of the described content are present but incomplete, inaccurate, or only partially realized
C. Execution Failed - The described content is barely recognizable or completely absent from this clip

Answer with ONLY the option letter (A/B/C):
```

## Current Scoring Rule

- If corresponding aligned shot is missing or has no `time_range`: score `0.0`.
- If corresponding aligned shot exists, extract its clip and ask VLM with the Step 3 prompt.
- Score map:
  - `A` Fully Executed -> `1.0`
  - `B` Partially Executed -> `0.5`
  - `C` Execution Failed -> `0.1`
- If parsing fails, default to `0.1`.
- A2 final event-fidelity score is the average of all per-event scores in the selected event chain.

## Current Inputs and Outputs

### Inputs

- `video_path`: full generated video.
- `question`: Mode C A2 question from `question_bank.json`.
- `alignment_result`: VLM shot alignment result, including `aligned_shots`, `status`, and `time_range`.
- `selected_option_text`: event chain selected in Step 1.

### Output Fields

```text
per_event_results: list of event-level records
avg_score: mean event score
n_events: number of events in selected chain
```

Each event record may include:

```text
event_idx
event_desc
status: not_generated or evaluated
vlm_response
vlm_choice
score
```

## Downstream Connections

- A2 final score in `run_joint_test.py` uses Mode C A2 directly.
- The alignment result comes from the Shot Alignment Skill and determines missing-shot penalties.
- This skill explains cases where an intended event receives `0.0` because no valid aligned shot was generated.

## Notes for Future Editing

- The Step 1 prompt prioritizes longest matching prefix order, not perfect full-chain matching.
- The Step 3 prompt currently forces A/B/C without explanation.
- The default parse-failure score is `0.1`, matching Execution Failed rather than `0.0`.
- Edit `prompts.yaml` to tune Step 1 or Step 3 system prompts (and the Step 3 user template) independently; no Python changes are needed for prompt-text modifications.
