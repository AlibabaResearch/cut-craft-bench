#!/bin/bash
# ============================================================
# Edit baseline agent batch entry point.
#
# Usage:
#   ./run_baseline.sh                 # all cases
#   ./run_baseline.sh --first-n 3     # first 3 cases
#   ./run_baseline.sh --ids "1,2,5"   # specific case ids
#   ./run_baseline.sh --id 1 --dry-run  # segmentation/rewrite/planning only, no video generation
#   ./run_baseline.sh --output-dir DIR  # override the final output directory
#
# Required environment:
#   DASHSCOPE_API_KEY -- shared by WAN video generation and the Qwen3.7 LLM
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/pipeline.py"

# ---- ffmpeg / ffprobe (concat, frame extraction, audio extraction) ----
# Uses PATH by default; to pin a specific env, export AV_PROCESS_BIN=/path/to/env/bin
AV_PROCESS_BIN="${AV_PROCESS_BIN:-}"
if [ -n "${AV_PROCESS_BIN}" ] && [ -d "${AV_PROCESS_BIN}" ]; then
    export PATH="${AV_PROCESS_BIN}:${PATH}"
fi

# ---- API key ----
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"

if [ -z "${DASHSCOPE_API_KEY}" ]; then
    echo "[WARN] the DASHSCOPE_API_KEY environment variable is not set."
    echo "       Run first: export DASHSCOPE_API_KEY=sk-xxxx"
    if [[ " $* " != *" --dry-run "* ]]; then
        echo "[ERROR] a non dry-run needs DASHSCOPE_API_KEY, aborting."
        exit 1
    fi
fi

if [[ " $* " != *" --dry-run "* ]]; then
    if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
        echo "[ERROR] ffmpeg/ffprobe not found: cannot extract audio or stitch video."
        echo "        Install ffmpeg, or add the directory containing ffmpeg and ffprobe to PATH."
        exit 1
    fi
fi

python3 "${PY}" "$@"
