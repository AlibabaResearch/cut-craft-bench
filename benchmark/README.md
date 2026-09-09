# CutCrafter Benchmark

Full-dimension evaluation of video-editing / montage generations. Every sample is
scored across 16 dimensions (A1–A4, B1–B3, C1, D1–D3, E1–E3, F1–F2) by combining
three complementary modes:

- **Mode A** — direct expert-model computation ([`mode_a_eval.py`](mode_a_eval.py))
- **Mode B** — VLM-assisted judgement over expert evidence ([`mode_b_eval.py`](mode_b_eval.py))
- **Mode C** — VLM question answering ([`mode_c_eval.py`](mode_c_eval.py))

The final per-dimension score merges the three modes; the merge rules live in
`compute_final_dimensions()` inside [`run_joint_test.py`](run_joint_test.py).
All dimension scores follow the **adjusted** aggregation (the only reported
formula).

## Prerequisites

1. **Expert model microservices.** The benchmark talks to 14 FastAPI services over
   HTTP (ports 8001–8015). They live in `../third_party/` — see
   [`third_party/README.md`](../third_party/README.md) for weights and launch. The
   runner script below starts/monitors them for you.

   | Port | Service | Port | Service |
   |------|---------|------|---------|
   | 8001 | transnetv2 | 8009 | yolov8 |
   | 8002 | raft       | 8010 | clip |
   | 8003 | dinov2     | 8011 | u2net (saliency) |
   | 8004 | whisper_asr| 8013 | movieshots |
   | 8005 | demucs     | 8014 | sixdrepnet |
   | 8006 | e2quality  | 8015 | monst3r |
   | 8007 | panns      |      | |
   | 8008 | dnsmos     |      | |

2. **Python environment** with `openai`, `numpy`, `scipy`, `librosa`, `requests`
   plus `ffmpeg` / `ffprobe` on `PATH`.

3. **DashScope API key** for the VLM (Mode B / Mode C / shot alignment):

   ```bash
   export DASHSCOPE_API_KEY=sk-xxxx
   ```

4. **Dataset** — provided in [`../prompt/`](../prompt):
   - `prompt.json` — shot descriptions (ground truth)
   - `question_bank.json` — Mode C questions keyed by the same video IDs

   The two files must always be swapped as a pair.

5. **Videos under evaluation** — place them in [`../videos/`](../videos), one subdirectory per
   model, named exactly as the model appears in the `MODELS` list of the watchdog script:

   ```
   ../videos/wan2.7/100_First_Sip_Twice.mp4
   ../videos/wan2.7/101_Activation_Loop.mp4
   ```

   The leading number is the video id joined against `prompt.json` / `question_bank.json`.
   This is the default (`EVAL_VIDEO_DIR_TEMPLATE=<repo>/videos/{model}`); override it only if
   the videos live outside the repo (`{model}` is substituted with the model name).

## Recommended: run everything through the watchdog

[`run_model_sequence_with_watchdog.sh`](run_model_sequence_with_watchdog.sh) is the
top-level entry point. It verifies the dataset, starts the 14 expert services once,
runs the evaluation, restarts any service that dies mid-run, and tears everything
down at the end.

```bash
export DASHSCOPE_API_KEY=sk-xxxx

# edit the MODELS=( ... ) list near the top of the script first, then:
bash run_model_sequence_with_watchdog.sh                # all samples
bash run_model_sequence_with_watchdog.sh --range 1-10   # by 1-based index range
bash run_model_sequence_with_watchdog.sh --ids 103 204 313
```

It fails fast if a model in `MODELS` has no `../videos/<model>/` directory or that directory
holds no `.mp4` files.

Key environment overrides (all optional except `DASHSCOPE_API_KEY`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `EVAL_VIDEO_DIR_TEMPLATE` | `<repo>/videos/{model}` | Video dir template, `{model}` substituted |
| `DASHSCOPE_API_KEY` | *(required)* | VLM API key |
| `EVAL_PYTHON` | `python` | Interpreter used for the evaluation |
| `AV_PROCESS_BIN` | — | A `bin/` dir prepended to `PATH` (e.g. the env holding ffmpeg) |
| `EXPERT_SERVICE_DIR` | `../third_party` | Directory holding the service subdirs |
| `EVAL_LOG_ROOT` | `../logs` | Root for evaluation artifacts |
| `EVAL_RESULT_BASE` | `final` | Output segment under `EVAL_LOG_ROOT` |
| `PROMPT_DIR` | `../prompt` | Directory of `prompt.json` / `question_bank.json` |
| `EVAL_PROMPT_FILE` | `$PROMPT_DIR/prompt.json` | Prompt file |
| `EVAL_QUESTION_BANK` | `$PROMPT_DIR/question_bank.json` | Mode C question bank |
| `SERVICE_LOG_DIR` | `/tmp/bench_services` | Service logs + pid files |

Results and a Markdown report land under
`${EVAL_LOG_ROOT}/${EVAL_RESULT_BASE}/<model>/`.

## Resumable single-model run

[`run_final_joint_test.py`](run_final_joint_test.py) evaluates one model with
resume support: after each sample it appends to a resume JSON, so a restart skips
the samples that already fully succeeded. This is what the watchdog script invokes,
but it can be run standalone once the services are already up and the dataset env
vars are exported:

```bash
export DASHSCOPE_API_KEY=sk-xxxx
export EVAL_PROMPT_FILE=../prompt/prompt.json
export EVAL_QUESTION_BANK=../prompt/question_bank.json
export EVAL_VIDEO_DIR_TEMPLATE=../videos/{model}

python run_final_joint_test.py --model wan2.7 --range 1-50
python run_final_joint_test.py --model wan2.7 --ids 3 8 14
python run_final_joint_test.py --model wan2.7 --no-resume      # ignore staged results
python run_final_joint_test.py --model wan2.7 --status-only    # only write the status file
```

## One-shot (non-resumable) run

[`run_joint_test.py`](run_joint_test.py) runs the same three-mode pipeline in a
single pass (no resume file). Useful for a quick check on a small ID set:

```bash
python run_joint_test.py --ids 1 5 --range 20-25
```

## Files

| File | Role |
|------|------|
| `run_model_sequence_with_watchdog.sh` | Top-level runner: services + watchdog + sequence |
| `run_final_joint_test.py` | Resumable single-model driver (recommended entry) |
| `run_joint_test.py` | One-pass driver + final-dimension aggregation + report |
| `mode_a_eval.py` | Mode A: expert-model dimension scoring |
| `mode_b_eval.py` | Mode B: VLM-assisted judgement |
| `mode_c_eval.py` | Mode C: VLM question answering |
| `a3_counterfactual_eval.py` | A3 counterfactual (reverse-order) evaluation |
| `vlm_shot_alignment.py` | VLM-assisted GT-shot ↔ detected-segment alignment |
| `service_client.py` | HTTP client for the expert microservices |
| `skill_loader.py` | Loads skill prompt configs from `skills/` |
| `skills/` | Per-skill `prompts.yaml` (single source of truth for some prompts) |
| `alignment_cache/` | Cached shot-alignment results |
