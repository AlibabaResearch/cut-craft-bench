#!/bin/bash
# ============================================================
# Expert-model microservices dedicated to the edit agent's own evaluation (offset ports).
#
# The benchmark's formal evaluation permanently occupies ports 8001-8015 and is watchdog
# guarded, so the agent's rolling self-evaluation runs a separate set of instances at a
# +100 port offset on idle GPUs; otherwise the two would preempt or kill each other.
#
#   TransNetV2 : 8101  (B1 shot duration / D2 transition-effect 5-way / D3 cut point)
#   Whisper    : 8104  (D3 word-level speech timestamps)
#   Demucs     : 8105  (D3 vocal / object-sound separation)
#   PANNs      : 8107  (D3 object sound-event tags)
#
# Usage: bash start_agent_services.sh [start|stop|status|restart]
#
# Safety constraints:
#   - only kills processes on 8101/8104/8105/8107, never touches 8001-8015;
#   - defaults to GPU 1,2,3 (formal eval services use GPU 4-7, generation uses GPU 0).
# ============================================================

set -uo pipefail

# In-repo paths are derived from this script: agent_eval/ -> agentic_edit_baseline/ -> repo root
AGENT_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${AGENT_EVAL_DIR}/../.." && pwd)"

PYTHON="${AGENT_EVAL_PYTHON:-python3}"
# Expert-model service code directory; defaults to third_party/ inside the repo
BASE_DIR="${AGENT_EVAL_SERVICE_DIR:-${REPO_ROOT}/third_party}"
LOG_DIR="${AGENT_EVAL_LOG_DIR:-/tmp/edit_agent_eval_services}"
# ffmpeg / ffprobe: uses whatever is on PATH; set AV_PROCESS_BIN to pin a specific env
if [ -n "${AV_PROCESS_BIN:-}" ] && [ -d "${AV_PROCESS_BIN}" ]; then
    export PATH="${AV_PROCESS_BIN}:$PATH"
fi

mkdir -p "$LOG_DIR"

# service_dir:port:gpu
SERVICES=(
    "transnetv2:8101:1"
    "whisper_asr:8104:2"
    "demucs:8105:3"
    "panns:8107:1"
)

AGENT_PORTS=(8101 8104 8105 8107)

# Port allowlist check: never operate on the formal evaluation ports
assert_agent_port() {
    local port="$1"
    for p in "${AGENT_PORTS[@]}"; do
        if [ "$p" = "$port" ]; then
            return 0
        fi
    done
    echo "  [FATAL] port $port is not on the agent self-eval allowlist, refusing (protects the formal eval services)"
    exit 1
}

start_services() {
    echo "Starting edit-agent eval services (offset ports, isolated from benchmark 8001-8015)..."
    for entry in "${SERVICES[@]}"; do
        IFS=':' read -r service port gpu <<< "$entry"
        assert_agent_port "$port"
        if curl -s --max-time 2 "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "  - $service already listening on $port, skip"
            continue
        fi
        echo "  - starting $service on port $port (GPU $gpu)"
        (
            cd "$BASE_DIR/$service" || exit 1
            CUDA_VISIBLE_DEVICES="$gpu" SERVICE_PORT="$port" \
                nohup "$PYTHON" app.py >"$LOG_DIR/${service}_${port}.log" 2>&1 &
            echo "    PID: $!"
        )
    done
    echo ""
    echo "Waiting for services to initialize (25s)..."
    sleep 25
    status_services
}

stop_services() {
    echo "Stopping edit-agent eval services (only offset ports)..."
    for entry in "${SERVICES[@]}"; do
        IFS=':' read -r service port gpu <<< "$entry"
        assert_agent_port "$port"
        pid=$(ss -ltnp 2>/dev/null | grep -oP "(?<=:)$port\s.*pid=\K[0-9]+" | head -1)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null && echo "  - stopped $service (port $port, pid $pid)"
        else
            echo "  - $service (port $port) not running"
        fi
    done
}

status_services() {
    echo "Edit-agent eval service status:"
    echo "============================================"
    local all_ok=0
    for entry in "${SERVICES[@]}"; do
        IFS=':' read -r service port gpu <<< "$entry"
        health=$(curl -s --max-time 3 "http://localhost:$port/health" 2>/dev/null)
        if [ -n "$health" ]; then
            echo "  OK   $service (port $port): $health"
        else
            echo "  DOWN $service (port $port)  -> log: $LOG_DIR/${service}_${port}.log"
            all_ok=1
        fi
    done
    echo "============================================"
    return $all_ok
}

case "${1:-start}" in
    start)   start_services ;;
    stop)    stop_services ;;
    status)  status_services ;;
    restart) stop_services; sleep 3; start_services ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
