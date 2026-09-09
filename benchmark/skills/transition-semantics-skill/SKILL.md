---
name: transition-semantics-skill
description: Evaluates D1 transition semantics with the VLM yes/no prompts (Mode B) for logical, POV, exit-entry, occlusion, cutaway, Gilligan, whip-pan, and generic transitions, plus the D1 single-choice transition-type QA (Mode C). Use when working on D1 semantic transition judging, Mode B transition routing, Mode C D1 QA, or stabilizing transition prompt behavior.
---

# Transition Semantics Skill

## Runtime Status: WIRED (Plan B)

This skill is now integrated into the live evaluation pipeline. `benchmark/skills/transition-semantics-skill/prompts.yaml` is the single source of truth for all D1 prompts, loaded at runtime via `benchmark/skill_loader.load_skill("transition-semantics-skill")`:

- **Mode B** (`benchmark/mode_b_eval.py`): `SYSTEM_D1_TRANSITION` and the 8 per-type user prompt templates (`logical`, `pov`, `exit_enter`, `obstruction`, `cutaway`, `flag`, `whip_pan`, `generic`) are loaded from `prompts.yaml`'s `mode_b.system_prompt` / `mode_b.templates.*` and used by `eval_d1_transition()` / `_eval_d1_*` helpers.
- **Mode C** (`benchmark/mode_c_eval.py`): D1 previously shared the generic `SYSTEM_QA` with A3/E1. It now has its own dedicated `elif dimension == "D1":` branch using `SYSTEM_QA_D1` / `USER_TEMPLATE_D1` / `D1_SHOT_CONTEXT_TEMPLATE`, loaded from `prompts.yaml`'s `mode_c.*`.

Both call sites wrap the loading in `try/except`; if `prompts.yaml` is missing or malformed, they print a `[WARN] Failed to load transition-semantics-skill ... config` message and fall back to the historical hardcoded prompts, so the pipeline never crashes because of this skill. To change any D1 prompt wording, edit `prompts.yaml` directly — no Python code changes needed.

## Purpose

Use this skill for the current D1 transition evaluation in Bench-A-V-Phys (both Mode B semantic yes/no judging and Mode C transition-type single-choice QA).

## Source Code

- `benchmark/mode_b_eval.py`
- System prompt: `SYSTEM_D1_TRANSITION`
- Functions: `eval_d1_transition(...)` and `_eval_d1_*` helpers

## Current System Prompt

```text
You are a professional film editor specializing in transition analysis. Based on the expert model evidence and the video, determine whether the described transition technique was correctly executed. Answer with ONLY 'Yes' or 'No'.
```

## Current Routing Rules

```text
Logical or Causal -> _eval_d1_logical
POV or Point-of-View -> _eval_d1_pov
Exit, Entry, or Walk -> _eval_d1_exit_enter
Occlusion, Wipe-By, or Foreground -> _eval_d1_obstruction
Cutaway, Empty, or Insert -> _eval_d1_cutaway
Gilligan or Flag -> _eval_d1_flag
Whip or Camera-Movement -> _eval_d1_whip_pan
Otherwise -> _eval_d1_generic
```

## Common Scoring Rule

- Ask VLM with the system prompt and transition-specific user prompt.
- Parse only `Yes` or `No`.
- If English parsing fails, parse Chinese `是` or `否` and map to `Yes` / `No`.
- Score `1.0` for `Yes`, otherwise `0.0`.

## Prompt Templates by Transition Type

### Logical / Causal Cut

Evidence:

```text
Cross-shot semantic similarity at cut point: {sim_val:.3f}
Intended transition design: {desc[:150]}
```

User prompt:

```text
At the transition from Shot {shot_idx+1} to Shot {shot_idx+2}:
{evidence}

Question: Does this transition establish a clear causal/logical relationship (i.e., the action or event in the preceding shot naturally motivates or leads to the content of the following shot)?
Answer: Yes or No
```

### Point-of-View Cut

Evidence:

```text
Head pose data: {sixdrepnet_summary[:150]}
Person detection: avg_count={avg_person_count}
```

User prompt:

```text
At the transition from Shot {shot_idx+1} to Shot {shot_idx+2}:
{evidence}

Question: Does this transition correctly implement a Point-of-View cut (one shot shows a character looking, the next shot shows what they see from their visual perspective)?
Answer: Yes or No
```

### Exit-and-Entry Transition

User prompt:

```text
At the transition from Shot {shot_idx+1} to Shot {shot_idx+2}:
Object detection edge analysis: {edge_analysis[:150]}

Question: Does the principal subject exit the frame in one shot and a subject enters the frame in the following shot (Exit-and-Entry transition)?
Answer: Yes or No
```

### Foreground-Occlusion / Wipe-By Transition

User prompt:

```text
At the transition from Shot {shot_idx+1} to Shot {shot_idx+2}:
Saliency mask coverage: {max_mask_ratio:.2f}

Question: Is the transition masked by a foreground element passing through and momentarily occluding the frame (Wipe-By / Foreground-Occlusion transition)?
Answer: Yes or No
```

### Cutaway / Empty Shot / Insert Transition

Evidence:

```text
Person detection: {avg_person_count}
Scene classification: {top_scene}
```

User prompt:

```text
At the transition from Shot {shot_idx+1} to Shot {shot_idx+2}:
{evidence}

Question: Does this transition use a cutaway to scenery / empty shot (a landscape or object-only shot devoid of characters) as a bridging element?
Answer: Yes or No
```


### Whip-Pan / Camera-Movement Transition

User prompt:

```text
At the transition from Shot {shot_idx+1} to Shot {shot_idx+2}:

Question: Does this transition use rapid camera motion (whip pan, swish pan, or fast camera movement) to bridge between two disparate scenes?
Answer: Yes or No
```

### Generic Transition

User prompt:

```text
At the transition from Shot {shot_idx+1} to Shot {shot_idx+2}:
Intended transition type: {ctype}
Design intent: {desc[:150]}

Question: Does this transition successfully convey the intended cinematographic effect described above?
Answer: Yes or No
```

## Current Inputs and Outputs

### Inputs

- `video_path`: full generated video.
- `prompt`: prompt dict with `shots[].transition_to_next`.
- `shot_idx`: transition index from shot `i` to shot `i+1`.
- `expert_results`: Mode A / expert model evidence, including CLIP, SixDRepNet, YOLOv8, saliency, and Places365 when available.

### Output Fields

```text
transition_idx
type
score
vlm_answer
vlm_response
```

## Downstream Connections

- D1 Mode B transition scores are aggregated with Mode A D1 and Mode C D1 in `run_joint_test.py`.
- Mode A skips D1 semantic categories and leaves them to this Mode B VLM path.
- Shot alignment affects whether adjacent transitions are evaluable in Mode A, but this skill focuses on semantic execution judgment.

## Notes for Future Editing

- Current prompts are binary and do not ask for rationale.
- Some categories rely heavily on expert evidence snippets; if those services change, update the evidence text here.
- Generic fallback is broad and may hide taxonomy drift, so new transition types should ideally get their own explicit prompt.
- Edit `prompts.yaml` to change Mode B system/user prompts (`mode_b.system_prompt`, `mode_b.templates.*`) or the Mode C system/user/shot-context prompts (`mode_c.system_prompt`, `mode_c.user_template`, `mode_c.shot_context_template`). Keep the placeholder names (`{shot_from}`, `{shot_to}`, `{evidence}`, `{edge_analysis}`, `{mask_ratio}`, `{sim_val}`, `{ctype}`, `{desc}` for Mode B; `{question}`, `{shot_context}`, `{options}`, `{n_gt}`, `{n_actual}` for Mode C) intact since the Python code formats these templates with those exact keyword names.
