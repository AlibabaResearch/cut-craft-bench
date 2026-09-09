---
name: montage-classification-skill
description: Classifies C1 montage structure using the Mode B VLM prompt, full-video input, global CLIP cross-shot semantic evidence, event_coherence_label grouped CLIP evidence (within-label and between-label similarities), and the 13-type montage taxonomy. Use when working on C1 montage classification, global_editing_style matching, montage prompt options, or dependencies between C1 and event-coherence labeling.
---

# Montage Classification Skill

## Runtime Status: WIRED (Plan B)

`benchmark/skills/montage-classification-skill/prompts.yaml` is the single
source of truth for BOTH modes:
- **Mode B** (`mode_b.system_prompt`): loaded by `mode_b_eval.py` as
  `SYSTEM_C1_MONTAGE`. Mode B sends the full video to VLM and provides both
  global CLIP cross-shot evidence and `event_coherence_label` grouped CLIP
  evidence (label sequence, label groups, label switches, within-label
  shot-pair similarity statistics, and between-label shot-pair similarity
  statistics). Falls back to hardcoded text with a warning.
- **Mode C** (`mode_c.system_prompt` / `mode_c.user_template`): loaded by
  `mode_c_eval.py` as `SYSTEM_QA_C1` / `USER_TEMPLATE_C1`. The C1 dimension
  now has its own `elif dimension == "C1"` branch in
  `evaluate_single_question()` instead of sharing the generic `SYSTEM_QA`
  used by A3/D1/E1. Falls back to `SYSTEM_QA` + generic template with a
  warning.

## Purpose

Use this skill for the current C1 montage-type VLM evaluation in Bench-A-V-Phys. It wraps the prompt and rubric currently implemented in `benchmark/mode_b_eval.py` for `eval_c1_montage`.

## Source Code

- `benchmark/mode_b_eval.py`
- System prompt: `SYSTEM_C1_MONTAGE`
- Function: `eval_c1_montage(video_path, prompt, clip_result)`

## Current System Prompt

```text
You are an expert in film editing theory and montage classification. Based on the full video, the global cross-shot semantic analysis, and the event-label grouped CLIP analysis (including both within-label and between-label similarities), identify the montage structure used in this video. Event-label alternation (the label sequence/switching pattern) indicates intercutting between coherent event threads/storylines and is general structural evidence for Parallel or Crosscut montage rather than simple Sequential montage.

Scope rule for the between-label CLIP similarity score: this numeric score is ONLY used as supporting evidence when deciding among Lyrical Montage, Contrast Montage, Montage of Attractions, and Reflexive Montage. For all other montage types (Sequential, Parallel, Crosscut, Repetition, Dialogue, Psychological, Metaphorical, Accumulative, Ideological), the VLM must decide from the full video and the label sequence/switching pattern alone, WITHOUT relying on the between-label CLIP score. If the video does not fit any of the four CLIP-sensitive types, the VLM ignores the between-label CLIP score and freely judges among the remaining nine types.

... (13-type Explanation of Montage Categories + 5 easily-confused-scenario rules) ...

Answer with ONLY the option letter.
```

## Current User Prompt Template

```text
Cross-shot Semantic Analysis:
- Cross-shot CLIP similarities: {[f'{s:.3f}' for s in cross_sims[:5]]}
- Average cross-shot CLIP similarity: {clip_result.get('avg_cross_shot_sim', 'N/A')}
- Number of shots: {prompt.get('number_of_shots', '?')}

Event-label Grouped CLIP Analysis (from prompt.shots[].event_coherence_label):
- Interpretation rule: event_coherence_label marks intended coherent event threads/storylines. Repeated alternation between labels is structured evidence for intercutting between threads (e.g., Parallel/Crosscut montage), while a single dominant label is stronger evidence for Sequential montage.
- Label sequence by intended shot: {label_sequence}
- Label switches across adjacent cuts: {label_transitions}/{num_cuts}; same-label adjacent cuts: {same_label_adjacent_cuts}
- Within-label CLIP similarity (how visually/semantically coherent each event thread is):
  - within label {label}: shots {shot_ids}, mean_within_label_clip={mean_within_label_clip}, n_shot_pairs={n_pairs}
- Between-label CLIP similarity (how similar or distinct different event threads are). SCOPE NOTE: only use this score to help decide among Lyrical / Contrast / Montage of Attractions / Reflexive Montage; for all other montage types, ignore this score and rely on the full video and the label sequence/switching pattern instead:
  - between label {label_a} and {label_b}: shots {shot_ids_a} vs {shot_ids_b}, mean_between_label_clip={mean_between_label_clip}, n_shot_pairs={n_pairs}

Watch this video and identify which montage structure best describes its editing. Use the full-video visual judgment together with the label sequence/switching pattern above as general structural evidence. The between-label CLIP similarity score is a specialized signal: use it ONLY when deciding among Lyrical Montage, Contrast Montage, Montage of Attractions, or Reflexive Montage. For all other montage types, decide from the full video and narrative structure WITHOUT relying on that CLIP score. If none of these four CLIP-sensitive types fit, ignore the between-label CLIP score and choose the best match among the remaining categories on your own.

A. Sequential Montage (events in strict chronological order)
B. Parallel Montage (multiple storylines intercut, converging later)
C. Crosscut Montage (rapid alternation of simultaneous events for tension)
D. Repetition Montage (meaningful shot repeated at key moments)
E. Dialogue Montage (a conversation split across different scenes/time)
F. Lyrical Montage (insert scenic/poetic shots to evoke emotion)
G. Psychological Montage (visualize dreams, memories, hallucinations, imagination)
H. Metaphorical Montage (visual analogy to imply deeper meaning)
I. Contrast Montage (juxtapose opposites for dramatic conflict)
J. Accumulative Montage (rapid succession of similar shots to build intensity)
K. Montage of Attractions (insert unrelated shots to provoke emotion/idea)
L. Reflexive Montage (metaphor drawn from objects already in the scene)
M. Ideological Montage (re-edit existing footage to argue a thesis/ideology)

Answer with the option letter (A-M):
```

## Option Mapping

```text
A -> Sequential
B -> Parallel
C -> Crosscut
D -> Repetition
E -> Dialogue
F -> Lyrical
G -> Psychological
H -> Metaphorical
I -> Contrast
J -> Accumulative
K -> Attractions
L -> Reflexive
M -> Ideological
```

## Current Scoring Rule

- Parse VLM response with valid choices `A-M`.
- Map the predicted option letter to a montage type.
- Compare `pred_type.lower()` against `prompt.global_editing_style.lower()`.
- Score `1.0` if the predicted montage type appears in the GT style string, otherwise `0.0`.

## Current Inputs and Outputs

### Inputs

- `video_path`: full generated video.
- `prompt`: prompt dict, especially `global_editing_style`, `number_of_shots`, and `shots[].event_coherence_label`.
- `clip_result`: CLIP result, especially `cross_shot_similarities`, `avg_cross_shot_sim`, `frame_times`, and `frame_sim_matrix`.

### Output Fields

```text
dimension: C1
metric: montage_type
score: 0.0 or 1.0
pred_type: mapped montage type
gt_style: prompt.global_editing_style truncated to 100 chars
vlm_choice: option letter
vlm_response: raw VLM response truncated to 200 chars
event_label_clip_stats: label sequence/groups/switches, within-label CLIP similarity statistics, and between-label CLIP similarity statistics
```

## Downstream Connections

- C1 final score is combined with Mode C C1 in `run_joint_test.py`.
- E3 event-coherence label generation depends on montage taxonomy, especially Parallel, Crosscut, Lyrical, Metaphorical, Contrast, Attractions, Reflexive, and Ideological montage categories.
- This skill should stay aligned with the taxonomy in `generate_event_coherence_labels.py`.

## Notes for Future Editing

- The current Mode B prompt only asks for one best label; it does not request reasoning.
- Mode B scoring is strict binary matching against `global_editing_style`.
- Mode B now uses `event_coherence_label` as structured evidence: alternating labels (the label sequence/switching pattern) indicate intercutting between coherent event threads/storylines, used as general evidence for distinguishing Parallel/Crosscut montage from simple Sequential montage.
- The between-label CLIP similarity score is now scope-restricted: it is only presented as decisive evidence for Lyrical / Contrast / Montage of Attractions / Reflexive Montage (these four types hinge on whether an inserted/alternating shot group is visually/semantically similar to or distinct from the main event). For the remaining nine montage types, the VLM is instructed to ignore this score and rely on the full video plus the label sequence/switching pattern, since those types typically differ by plot/narrative details rather than by overall shot-level visual similarity.
- Edit `prompts.yaml` to tune either mode's prompt independently; no
  Python changes are needed for prompt-text modifications.
- If the taxonomy (13-type option list in Mode B) changes, also update
  the `montage_map` dict in `eval_c1_montage()` and the event-coherence
  skill's `target_montage_prefixes` list.
