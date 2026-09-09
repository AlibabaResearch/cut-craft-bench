---
name: shot-alignment-skill
description: Two-step VLM shot alignment skill that maps TransNetV2-detected video segments to GT prompt shots (present/missing/merged), producing aligned_shots used by B1, E1, E3, D1/D2/D3, A2 and other dimensions. Use when working on vlm_shot_alignment.py, debugging alignment accuracy, or designing new alignment prompts.
---

# Shot Alignment Skill

## Runtime Status: WIRED (Plan B)

`benchmark/skills/shot-alignment-skill/prompts.yaml` is the single source
of truth for the two system prompts (`step_a.system_prompt`,
`step_b.system_prompt`), loaded via `benchmark/skill_loader.py` at import
time in `vlm_shot_alignment.py`. Falls back to the historical hardcoded
text (with a printed warning) if the YAML is missing/broken, since
alignment is a prerequisite for many other dimensions and must not
hard-crash the whole pipeline.

Only the two SYSTEM prompts are externalized (as requested, reusing the
existing prompt text as-is). The user prompt templates for both steps stay
in Python because they are built dynamically from many runtime-computed
fields (shot lists, segment lists, GT descriptions).

## 用途

将生成视频经 TransNetV2 检测到的片段（segments）与 prompt 中的 GT 镜头（shots）进行语义对齐，输出每个 GT shot 的状态（matched / merged / missing）及其在视频中的实际 time_range。这是 B1、E1、E3、D1/D2/D3、A2 等多个评测维度的共同基础。

来源代码：[vlm_shot_alignment.py](../../vlm_shot_alignment.py)

## 当前实现：两步法

### Step A — 全局镜头语义判断

**System Prompt**（[L43-L50](../../vlm_shot_alignment.py#L43-L50)）：

```
You are a professional film editor and shot segmentation expert. Your task is to watch the video and determine the actual shot structure by comparing the video content with the intended shot descriptions.

A 'shot' is defined as a continuous segment filmed from one camera setup/angle without cuts. A cut/transition creates a new shot.

You must output ONLY valid JSON, with no extra text or explanation.
```

**User Prompt 模板**（[L246-L260](../../vlm_shot_alignment.py#L246-L260)）：

```
This video is intended to have {n_gt} shots with the following descriptions:
{shot_list_text}

Watch the ACTUAL video carefully and determine:
1. How many distinct shots (separated by cuts/transitions) are actually present?
2. For each intended shot (1 to {n_gt}), determine if it is:
   - 'present': clearly visible as a distinct shot
   - 'missing': not present in the video at all
   - 'merged': its content appears to be merged with an adjacent shot (no visible cut)

Output JSON format:
{"actual_shot_count": <int>, "shots": [{"gt_idx": 1, "status": "present/missing/merged", "approx_start": <float>, "approx_end": <float>}, ...]}

For missing shots, set approx_start and approx_end to -1.
For merged shots, use the time range of the combined segment.
```

调用参数：`max_tokens=1000, temperature=0.1`。

### Step B — 逐段分类（更精确，优先采用）

**System Prompt**（[L52-L66](../../vlm_shot_alignment.py#L52-L66)）：

```
You are a professional film editor analyzing video content. Your task is to classify each video segment by its VISUAL CONTENT — NOT by temporal position or timing.

CRITICAL RULES:
1. For EACH detected segment, watch the actual visual content and describe what you see.
2. Then match it to the GT shot description that best fits the CONTENT.
3. DO NOT assume segments map to GT shots in sequential order.
4. Multiple consecutive segments CAN belong to the same GT shot — this happens when one shot was incorrectly split into fragments.
5. Some GT shots may NOT exist in the video at all (the model failed to generate them).
6. If a segment is a transition artifact (<0.3s), label it -1.
7. Base your decision ONLY on visual content similarity, IGNORE temporal position.

You must output ONLY valid JSON, with no extra text or explanation.
```

**User Prompt 模板**（[L300-L325](../../vlm_shot_alignment.py#L300-L325)）：

```
A video was automatically split into {n_detected} segments by cut detection:
{segments_text}

The video was INTENDED to contain {n_gt} shots. Here are the CONTENT DESCRIPTIONS (what each shot should show):
{gt_text}

IMPORTANT CONTEXT: The video generation model often makes mistakes:
- It may SPLIT one intended shot into multiple segments (fragments)
- It may SKIP some intended shots entirely (not generate them)
- The LAST GT shot (index {n_gt-1}) is the most commonly skipped one
- So the number of segments ({n_detected}) does NOT necessarily equal the number of actually realized shots

YOUR TASK:
For each segment, watch the actual video content in that time range and determine which GT shot description (0 to {n_gt-1}) best matches what you SEE.

Think carefully:
- If two consecutive segments show the SAME scene/subject/action continuing, they should have the SAME label (they are fragments of one shot)
- If an intended shot's content does NOT appear anywhere in the video, no segment should get that label
- Do NOT just assign labels 0,1,2,...,{n_gt-1} in order — that assumption is usually WRONG

Output JSON with exactly {n_detected} integer labels:
{"segment_labels": [<int>, <int>, ...]}
Each label: 0 to {n_gt-1} (GT shot index) or -1 (discard).
```

调用参数：`max_tokens=2000, temperature=0.0`。

GT shot 描述在传给 VLM 前会去掉时间范围前缀（正则 `Shot \d+ \[\d+\.?\d*-\d+\.?\d*s?\]:\s*`），防止 VLM 按时间位置而非视觉内容匹配。

## 综合与后处理逻辑

- 优先采用 Step B 结果（`segment_labels` 或旧版 `alignment` 格式），Step A 仅作补充验证；两步均失败则回退到顺序截断对齐（[_build_alignment_from_vlm](../../vlm_shot_alignment.py#L346-L366)）。
- Step B 结果需做三重后处理修正（[_build_from_step_b](../../vlm_shot_alignment.py#L369-L640)）：
  1. **非相邻合并修复**：同一 GT 镜头被分配了不相邻的 segments 时，重新分配离群段。
  2. **跳跃标签修复**：labels 中出现跳跃（如 `[0,1,1,3]` 跳过了 2）且发生在最后一段时，修正为被跳过的 GT 索引。
  3. **时间重叠验证修复**：按检测到的实际时间重叠比例校正误判的 GT 索引。
- 最终输出 `aligned_shots`：每项含 `gt_shot_idx / status(matched/merged/missing) / detected_indices / time_range`，以及全局统计 `n_matched / n_merged / n_missing / shot_accuracy`。
- 结果带缓存（`alignment_cache/`），键为视频路径哈希 + prompt_id，TransNetV2 段数变化则失效。

## 下游依赖

- 转场类维度（D1/D2/D3/B2）：仅当相邻两镜头都存在时该转场才「可评测」，否则记 0 分。
- 按镜头维度（E1、B1、E3）：用对齐后 time_range 截取镜头，缺失镜头记 0 分或不参与计算。
- Mode C（A2/A3/E1/D1 题）：告知 VLM 实际镜头数与缺失情况。

## 待优化方向（供后续修改参考）

- Step A/B 的 system prompt 现在已外置到 `prompts.yaml`，可直接编辑该文件调整措辞；user prompt 模板仍是纯Python硬编码（因依赖大量运行时变量）。
- 缺少显式的 few-shot 反例（如"最后一镜头最常被跳过"这条经验规则目前只在 prompt 里提示一次，没有具体示例佐证）。
- 后处理修正规则（非相邻合并/跳跃修复/重叠验证）目前是硬编码启发式，未与 VLM 输出的置信度关联。
