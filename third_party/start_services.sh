#!/bin/bash
# Expert model microservice launcher
# Usage: bash start_services.sh [start|stop|status]
#
# Overridable via environment:
#   EXPERT_SERVICE_PYTHON  python interpreter used to run each service (default: python)
#   EXPERT_SERVICE_LOG_DIR directory for service logs (default: /tmp)
#   MODEL_ROOT             weights root shared by all services (default: <this dir>/models)

PYTHON="${EXPERT_SERVICE_PYTHON:-python}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${EXPERT_SERVICE_LOG_DIR:-/tmp}"
export MODEL_ROOT="${MODEL_ROOT:-$BASE_DIR/models}"

declare -A SERVICES
# Mode A expert models (8001-8008)
# GPU 0-3 may be occupied by video generation jobs; keep evaluator services on GPU 4-7.
SERVICES[transnetv2]="8001:4"
SERVICES[raft]="8002:5"
SERVICES[dinov2]="8003:6"
SERVICES[whisper_asr]="8004:7"
SERVICES[demucs]="8005:4"
SERVICES[e2quality]="8006:5"
SERVICES[panns]="8007:6"
SERVICES[dnsmos]="8008:7"      # CPU-only (librosa signal analysis), the GPU ID is a placeholder
# Mode B expert models (8009-8015)
SERVICES[yolov8]="8009:4"
SERVICES[clip]="8010:5"
SERVICES[u2net]="8011:7"       # CPU-only, the GPU ID is a placeholder
# places365 (8012) is intentionally not started: nothing calls it (see README).
SERVICES[movieshots]="8013:7"
SERVICES[sixdrepnet]="8014:4"
SERVICES[monst3r]="8015:5"

start_services() {
    echo "Starting all expert model services..."
    mkdir -p "$LOG_DIR"
    for service in "${!SERVICES[@]}"; do
        IFS=':' read -r port gpu <<< "${SERVICES[$service]}"
        echo "  Starting $service on port $port (GPU $gpu)..."
        cd "$BASE_DIR/$service"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON app.py &>"$LOG_DIR/${service}.log" &
        echo "    PID: $!"
    done
    echo ""
    echo "Waiting for services to initialize (20s)..."
    sleep 20
    status_services
}

stop_services() {
    echo "Stopping all services..."
    for service in "${!SERVICES[@]}"; do
        IFS=':' read -r port gpu <<< "${SERVICES[$service]}"
        pkill -f "python app.py.*$port" 2>/dev/null
        echo "  Stopped $service (port $port)"
    done
    # Also kill by port binding
    for port in 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8013 8014 8015; do
        fuser -k $port/tcp 2>/dev/null
    done
    echo "All services stopped."
}

status_services() {
    echo "Service Status:"
    echo "============================================"
    for service in "${!SERVICES[@]}"; do
        IFS=':' read -r port gpu <<< "${SERVICES[$service]}"
        health=$(curl -s --max-time 2 "http://localhost:$port/health" 2>/dev/null)
        if [ -n "$health" ]; then
            echo "  ✓ $service (port $port): $health"
        else
            echo "  ✗ $service (port $port): NOT RUNNING"
        fi
    done
    echo "============================================"
}

case "${1:-start}" in
    start)
        stop_services 2>/dev/null
        sleep 2
        start_services
        ;;
    stop)
        stop_services
        ;;
    status)
        status_services
        ;;
    restart)
        stop_services
        sleep 2
        start_services
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
