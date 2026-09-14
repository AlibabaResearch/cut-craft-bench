---
name: e3-event-coherence-skill
description: Defines E3 event-coherence labeling and scoring using the existing VLM label prompt plus reconstructed-video DINOv2 scoring by event label. Use when working on event_coherence_label generation, E3 style consistency, montage label grouping, reconstructed label videos, or prompt-label weighted aggregation.
---

# E3 Event Coherence Skill

## Runtime Status: WIRED (Plan B)

Unlike the other 4 VLM skills, this skill is no longer documentation-only.
`benchmark/skills/e3-event-coherence-skill/prompts.yaml` is now the SINGLE
SOURCE OF TRUTH for the labeling prompts, the target montage taxonomy, and
the DINOv2 scoring parameters. It is loaded at runtime through
`benchmark/skill_loader.py`:

- `data_generator/prompt/generate_event_coherence_labels.py` loads
  `labeling.system_prompt`, `labeling.user_template`, and
  `labeling.target_montage_prefixes` at import time (raises loudly if the
  YAML is missing or incomplete, since labeling is this script's only job).
- `benchmark/mode_a_eval.py` loads `scoring.sample_fps` and
  `scoring.max_frames` at import time as the defaults for
  `eval_e3_style_consistency(...)` (falls back to `4.0`/`80` with a printed
  warning if the YAML is missing/broken, since this file evaluates many
  unrelated dimensions and should not crash entirely because of E3).

**To change labeling prompts, the target montage list, or DINOv2 sampling
parameters, edit `prompts.yaml` directly — no Python changes needed.** The
text below mirrors `prompts.yaml` for human readability, but `prompts.yaml`
is authoritative if they ever disagree.

## Purpose

Use this skill for E3 event-coherence label generation and the current E3 Mode A scoring semantics in CutCraft. It combines the existing VLM label prompt from `generate_event_coherence_labels.py` with the current reconstructed-video scoring logic from `mode_a_eval.py`.

## Source Code

- `benchmark/skills/e3-event-coherence-skill/prompts.yaml` (authoritative prompt/param source)
- `benchmark/skill_loader.py` (`load_skill("e3-event-coherence-skill")`)
- `data_generator/prompt/generate_event_coherence_labels.py`
- `benchmark/mode_a_eval.py`
- Label prompt constants: `SYSTEM_PROMPT`, `USER_TEMPLATE` (now loaded from `prompts.yaml`, not hardcoded)
- Scoring function: `eval_e3_style_consistency(...)`

## Target Montage Types for Labeling

Only these montage prefixes receive VLM event-coherence labels. Non-target montage types fall back to all-zero labels.

```text
Parallel Montage
Crosscut Montage
Lyrical Montage
Metaphorical Montage
Contrast Montage
Montage of Attractions
Reflexive Montage
Ideological Montage
```

## Current Labeling System Prompt

```text
You are a professional film editing analyst.
Your task is to assign categorical event-coherence labels to shots in a montage prompt.
The labels are category IDs only; they do NOT indicate order or priority.
Shots belonging to the same continuous scene/event/thread should share the same integer label.
Different scenes/events/threads, metaphorical insert layers, attraction inserts, documentary evidence layers, or contrast strands should use different labels when they are not expected to be visually/style-continuous with each other.
Return ONLY valid JSON in the exact form: {"labels": [0, 1, 0]}.
```

## Current Labeling User Prompt Template

```text
Montage source_seed: {source_seed}
Global editing style: {global_editing_style}
Number of shots: {number_of_shots}

Shot descriptions:
{shot_lines}

Assign one integer event-coherence label for each shot.
Rules:
- Same label means E3 may compare visual/style consistency within that event/thread.
- Different labels mean cross-label visual jumps are intended and should not be compared directly.
- Use the smallest practical number of labels.
- The output labels list length MUST equal {number_of_shots}.
- Return ONLY JSON, no explanation.
```

`shot_lines` is built as:

```text
Shot {shot_id}: {description_prompt}
```

## Current Label Validation Rules

- Output must be valid JSON with a `labels` list.
- Label count must equal the number of prompt shots.
- Labels must be non-negative integers, not booleans.
- Labels are normalized to contiguous category IDs by first occurrence.
- Labels are categorical IDs only; they do not encode order or priority.

## Current Deterministic Fallback Heuristics

Fallback is used only when explicitly enabled or when label generation cannot use the model.

- Non-target montage -> all labels `0`.
- Parallel / Crosscut:
  - Convergence / shared-space shots use label `0`.
  - Other shots alternate labels by index.
- Contrast -> alternating `0, 1, 0, 1, ...`.
- Insert/metaphor/attraction/reflection/empty-shot keywords create separate labels; normal story shots use `0`.

## Current E3 Scoring Semantics

E3 currently evaluates style consistency by event label, not by comparing all adjacent frames across the full video.

1. Read `prompt.shots[].event_coherence_label`.
2. Build `aligned_by_gt` from `alignment_result.aligned_shots`.
3. For each GT shot, skip missing shots or shots without valid `time_range`.
4. Group valid aligned shot time ranges by label.
5. For each label:
   - Weight = count of prompt shots with that label / total prompt shots.
   - If no valid aligned shot exists for the label, label score is `0.0`.
   - Extract each valid aligned shot segment from the original video.
   - If the label has multiple segments, concatenate them into one reconstructed temporary video.
   - If the label has one segment, evaluate that segment directly.
   - Run DINOv2 on the reconstructed label video with `sample_fps=4.0` and `max_frames=80`.
   - Use mean `temporal_consistency` if present; otherwise use `mean_similarity`.
6. Final E3 score = weighted average of per-label scores.

## Current E3 Output Fields

```text
dimension: E3
metric: 风格一致性
score: weighted label score
mean_temporal_sim: rounded weighted score
num_frame_pairs: total DINOv2 frame-pair count
method: event_coherence_reconstructed_video
used_alignment: True
per_label_scores: label-level details
num_event_labels: number of labels
weighting: prompt_label_shot_ratio
```

Each label entry may include:

```text
score
weight
weighted_score
shot_count
valid_shot_count
shot_ids
num_frame_pairs
reconstructed
note: no valid aligned shots for this label
error
```

## Important Interpretation

- Same-label shots are treated as belonging to the same visual/style continuity group.
- Different-label shots are intentionally not compared directly, because visual jumps across labels may be part of the montage design.
- If a label has prompt shots but no valid aligned generated shots, that label contributes `0.0` with its full prompt-shot weight.
- Example: labels `[0, 1, 0, 1, 0]` give label `0` weight `0.6` and label `1` weight `0.4`.

## Downstream Connections

- Label quality depends on Montage Classification Skill taxonomy and prompt semantics.
- Scoring quality depends on Shot Alignment Skill because `aligned_shots.time_range` determines which video segments are used.
- Final E3 in `run_joint_test.py` combines Mode A E3 and Mode C E3 with the configured weights.

## Notes for Future Editing

- The current label prompt does not request reasoning, only JSON.
- The current E3 score penalizes missing labels by their prompt-shot ratio.
- If label semantics change, update `prompts.yaml` (the authoritative source); this SKILL.md section should be kept in sync manually as documentation.
- If you add new tunable parameters, add them under `prompts.yaml`'s `scoring:` or `labeling:` sections and load them through `benchmark/skill_loader.load_skill(...)` rather than hardcoding new Python constants.
