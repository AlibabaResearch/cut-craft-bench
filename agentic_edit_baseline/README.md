# agentic_edit_baseline

A WAN-based montage editing agent. Given a case prompt, it segments the story into
shots, rewrites each shot prompt with Qwen3.7, generates every shot with WAN, and
stitches them into a single final cut with transitions and a coherent audio track.

Two entry points share the same 6-stage generation core:

| Entry | Script | What it does |
| --- | --- | --- |
| **Baseline** | `run_baseline.sh` → `pipeline.py` | Pure multi-stage generation. No evaluation, no repair. |
| **Agent** | `run_agent.sh` → `agent_loop.py` | Baseline generation, then rolling B1/D2/D3 self-evaluation and central-planner repair (≤ 2 rounds). |

Generation pipeline (both entries):

```
Stage1 shot segmentation
  -> Stage1.5 cross-shot consistency / narrative anchoring
  -> Stage2 per-shot prompt rewrite (Qwen3.7)
  -> Stage3 generation planning (which transitions need first-frame continuity)
  -> Stage4 WAN shot generation (t2v / i2v / r2v)
  -> Stage5 ffmpeg transition stitching + audio mixing
  -> Stage6 final cut + render_report
```

The agent path adds `Stage7 rolling evaluation` and `Stage8 central-planner repair`
on top of Stage6.

## Prerequisites

1. **`DASHSCOPE_API_KEY`** — required for WAN video generation, the Qwen3.7 prompt
   rewriter, the central planner, and the D3 VLM. Comma-separate multiple keys to rotate.
   ```bash
   export DASHSCOPE_API_KEY=sk-xxxx
   ```
2. **ffmpeg / ffprobe** on `PATH` (concat, frame extraction, audio analysis). To pin a
   specific environment instead of `PATH`, export `AV_PROCESS_BIN=/path/to/env/bin`.
3. **Python deps**: `pyyaml`, `numpy`, and (for the agent's D3 rolling evaluation)
   `librosa`. To pin the interpreter for the agent, export `EDIT_AGENT_PYTHON=/path/to/python`.
4. **`prompt.json`** — the case list, at `../prompt/prompt.json` by default
   (override with `--prompt-json`). Each case needs an `id`.

## Quick start

### Baseline (pure generation)

```bash
export DASHSCOPE_API_KEY=sk-xxxx

./run_baseline.sh                  # all cases
./run_baseline.sh --first-n 3      # first 3 cases
./run_baseline.sh --ids "1,2,5"    # specific case ids
./run_baseline.sh --id 1 --dry-run # segmentation/rewrite/planning only, no video generation
./run_baseline.sh --output-dir DIR # override the final output directory
```

### Agent (generation + self-eval + repair)

First start the self-eval expert services (offset ports, isolated from any formal
evaluation on 8001-8015), then run the agent:

```bash
bash agent_eval/start_agent_services.sh start   # TransNetV2 8101 / Whisper 8104 / Demucs 8105 / PANNs 8107
bash agent_eval/start_agent_services.sh status  # health check

./run_agent.sh --id 1
./run_agent.sh --ids "1,2,5"
./run_agent.sh --id 1 --max-repair-rounds 2
./run_agent.sh --id 1 --no-vlm     # D3 by signal arbitration only, saves VLM calls
```

If the self-eval services are down, the agent skips the rolling evaluation and degrades
to pure generation. Stop them with `bash agent_eval/start_agent_services.sh stop`.

### Batch retry (regenerate only missing final cuts)

Scans the output directory and re-runs only the case ids whose final `.mp4` is missing:

```bash
bash run_batch_retry.sh                 # detect + regenerate missing cases
bash run_batch_retry.sh --only-detect   # list missing ids, generate nothing
bash run_batch_retry.sh --batch-size 5  # ids per run_baseline.sh batch (default 10)
bash run_batch_retry.sh --output-dir /path/to/wan
```

## Configuration

All behavior is driven by [`config.yaml`](config.yaml). Key sections:

| Section | Purpose |
| --- | --- |
| `provider` / `t2v_model` / `i2v_model` / `r2v_model` | WAN model ids per generation mode |
| `resolution` / `ratio` / `fps` / `target_duration` | output frame parameters and forced final duration |
| `llm` | Qwen3.7 text model for prompt rewriting (OpenAI-compatible endpoint) |
| `consistency` | cross-shot entity/setting bible + narrative anchoring |
| `audio` | diegetic sound policy, loudness normalisation, room-tone floor, scene beds, final master |
| `paths` | `prompt_json`, `artifacts_dir`, `output_dir`, `agent_output_dir` |
| `planner` | central-planner LLM (`agent_loop.py` only) |
| `evaluation` | rolling B1/D2/D3 dimensions, thresholds, and the D3 audio gate (`agent_loop.py` only) |
| `agent_loop` | `max_repair_rounds` |

Relative paths in `paths` resolve against this directory. To write artifacts to a disk
outside the repo, export `EDIT_BASELINE_OUT=/path/to/data`.

## Outputs

- `paths.artifacts_dir/<id>/` — intermediate artifacts: `shot_plan.json`,
  `consistency.json`, `edit_decisions.json`, per-shot clips, `render_report`.
- `paths.output_dir/` — baseline final cuts (`pipeline.py`).
- `paths.agent_output_dir/` — agent final cuts (`agent_loop.py`); empty reuses `output_dir`.

## Layout

```
agentic_edit_baseline/
├── run_baseline.sh          # baseline entry  -> pipeline.py
├── run_agent.sh             # agent entry     -> agent_loop.py
├── run_batch_retry.sh       # regenerate missing final cuts
├── pipeline.py              # 6-stage generation orchestrator
├── agent_loop.py            # generation + rolling eval + central-planner repair
├── config.yaml              # global configuration
├── pipeline_defs/           # pipeline definition (montage-edit.yaml)
├── lib/                     # segmentation, rewrite, planning, audio, consistency, shot runner
├── tools/                   # WAN t2v / i2v callers and video compositor
└── agent_eval/              # self-eval services + transition_eval (B1/D2/D3)
```
