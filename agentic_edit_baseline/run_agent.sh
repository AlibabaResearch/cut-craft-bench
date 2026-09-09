#!/bin/bash
# ============================================================
# Edit agent entry point (generation + rolling evaluation + central-planner repair).
#
# Difference from run_baseline.sh:
#   run_baseline.sh -> pipeline.py   pure multi-stage generation, no eval, no repair
#   run_agent.sh    -> agent_loop.py evaluates B1/D2/D3 after generation, then the central
#                                    planner (qwen3.7-plus) attributes and repairs, max 2 rounds
#
# Usage:
#   ./run_agent.sh --id 1
#   ./run_agent.sh --ids "1,2,5"
#   ./run_agent.sh --id 1 --max-repair-rounds 2
#   ./run_agent.sh --id 1 --no-vlm            # D3 by signal arbitration only, saves VLM calls
#   ./run_agent.sh --eval-only --eval-video X.mp4 --eval-decisions edit_decisions.json
#
# Requirements:
#   1) DASHSCOPE_API_KEY -- shared by WAN generation, Qwen3.7 rewrite, the qwen3.7-plus planner and the D3 VLM
#   2) the agent self-eval expert services on the offset ports 8101/8104/8105/8107:
#        bash agent_eval/start_agent_services.sh start
#      the formal evaluation ports 8001-8015 are never accessed by this path.
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_ENTRY="${SCRIPT_DIR}/agent_loop.py"

# ---- Python: rolling evaluation needs librosa/numpy; set EDIT_AGENT_PYTHON to pin an interpreter ----
PYTHON="${EDIT_AGENT_PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1 && [ ! -x "${PYTHON}" ]; then
    echo "[WARN] ${PYTHON} not found, falling back to python3 (D3 evaluation is unavailable without librosa)"
    PYTHON="python3"
fi

# ---- ffmpeg / ffprobe: uses PATH by default; set AV_PROCESS_BIN to pin a specific env ----
AV_PROCESS_BIN="${AV_PROCESS_BIN:-}"
if [ -n "${AV_PROCESS_BIN}" ] && [ -d "${AV_PROCESS_BIN}" ]; then
    export PATH="${AV_PROCESS_BIN}:${PATH}"
fi

export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"
if [ -z "${DASHSCOPE_API_KEY}" ]; then
    echo "[WARN] DASHSCOPE_API_KEY is not set: video generation / planner / D3 VLM are all unavailable."
    echo "       Run first: export DASHSCOPE_API_KEY=sk-xxxx"
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "[ERROR] ffmpeg/ffprobe not found: cannot stitch video or analyse audio."
    exit 1
fi

# ---- Self-eval service health check (offset ports only, never the formal eval ports) ----
MISSING=""
for port in 8101 8104 8105 8107; do
    if ! curl -s --max-time 3 "http://localhost:${port}/health" >/dev/null 2>&1; then
        MISSING="${MISSING} ${port}"
    fi
done
if [ -n "${MISSING}" ]; then
    echo "[WARN] agent self-eval services are not ready (ports:${MISSING})."
    echo "       Start them with: bash ${SCRIPT_DIR}/agent_eval/start_agent_services.sh start"
    echo "       While they are down the agent skips the rolling evaluation and degrades to pure generation."
fi

exec "${PYTHON}" "${PY_ENTRY}" "$@"
