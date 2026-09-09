#!/bin/bash
# ============================================================
# Serial multi-model full evaluation with expert-service watchdog.
#
# Usage:
#   bash run_model_sequence_with_watchdog.sh
#   bash run_model_sequence_with_watchdog.sh --range 1-10
#   bash run_model_sequence_with_watchdog.sh --ids 103 204 313
#
# Requirements:
#   - DASHSCOPE_API_KEY must be set
#   - edit the MODELS list below to change the test order
#
# Environment overrides:
#   EVAL_PYTHON              python interpreter (default: python)
#   AV_PROCESS_BIN           prepended to PATH, e.g. an env's bin/ holding ffmpeg
#   EXPERT_SERVICE_DIR       directory holding the service subdirs (default: <repo>/third_party)
#   EVAL_LOG_ROOT            evaluation artifact root (default: <repo>/logs)
#   EVAL_RESULT_BASE         output root segment under EVAL_LOG_ROOT (default: final1)
#   PROMPT_DIR               prompt.json / question_bank.json directory (default: <repo>/prompt)
#   EVAL_PROMPT_FILE         prompt file (default: $PROMPT_DIR/prompt.json)
#   EVAL_QUESTION_BANK       mode_c question bank (default: $PROMPT_DIR/question_bank.json)
#   EVAL_VIDEO_DIR_TEMPLATE  videos under evaluation, {model} is substituted
#                            (default: <repo>/videos/{model})
#   SERVICE_LOG_DIR          service logs and pid files (default: /tmp/bench_services)
# ============================================================

set -o pipefail
if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ]; then
  echo "ERROR: bash 4 or newer is required, found ${BASH_VERSION:-unknown}" >&2
  echo "       macOS ships bash 3.2; please run this script with a newer bash." >&2
  exit 1
fi

# ============ Configurable: model test order ============
MODELS=(
  # "seedance2.0"
  # "kling"
  # "MiniMax-H3"
  # "wan2.7"
  # "happyhorse"
  # "veo3.1"
  # "vidu"
  # "ltx2"
  # "Davinci"
  # "JavisDiT++"
  # "ovi"
  # "mova"
  "seedance2.5"
)
RESTORE_MODEL_NAME_ON_EXIT=0
SKIP_COMPLETED_MODELS=1

# ============ Configurable: watchdog tuning ============
WATCHDOG_INTERVAL=15        # health check interval while an evaluation runs (seconds)
HEALTH_TIMEOUT=20           # per-request health check timeout (seconds)
HEALTH_FAIL_THRESHOLD=2     # consecutive failures before a service is declared down
STARTUP_WAIT=60             # grace period after the initial service launch (seconds)
MAX_RESTART_PER_SERVICE=5   # max restarts per service, counted per model
RESTART_COOLDOWN=45         # wait after a restart before re-checking health (seconds)

# ============ Path configuration ============
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$BENCHMARK_DIR")"
PYTHON="${EVAL_PYTHON:-python}"
EVAL_SCRIPT="run_final_joint_test.py"
RUN_FILE="$BENCHMARK_DIR/$EVAL_SCRIPT"
SERVICE_DIR="${EXPERT_SERVICE_DIR:-$REPO_ROOT/third_party}"
if [ -n "${AV_PROCESS_BIN:-}" ]; then
  export PATH="${AV_PROCESS_BIN}:$PATH"
fi

EVAL_RESULT_BASE="${EVAL_RESULT_BASE:-final}"
export EVAL_RESULT_BASE
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-$REPO_ROOT/logs}"
export EVAL_LOG_ROOT
LOG_BASE="${EVAL_LOG_ROOT}/${EVAL_RESULT_BASE}/watchdog_sequence"

# ============ VLM configuration (inherited by Mode A/B/C) ============
# DASHSCOPE_API_KEY must be injected by the caller; never store it in source code.
DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
VLM_MODEL="${VLM_MODEL:-qwen3.5-omni-plus}"
export DASHSCOPE_API_KEY DASHSCOPE_BASE_URL VLM_MODEL

SERVICE_LOG_DIR="${SERVICE_LOG_DIR:-/tmp/bench_services}"

# ============ Dataset paths (single source of truth) ============
PROMPT_DIR="${PROMPT_DIR:-$REPO_ROOT/prompt}"
EVAL_PROMPT_FILE="${EVAL_PROMPT_FILE:-$PROMPT_DIR/prompt.json}"
EVAL_QUESTION_BANK="${EVAL_QUESTION_BANK:-$PROMPT_DIR/question_bank.json}"
EVAL_VIDEO_DIR_TEMPLATE="${EVAL_VIDEO_DIR_TEMPLATE:-$REPO_ROOT/videos/{model}}"

export EVAL_PROMPT_FILE EVAL_QUESTION_BANK EVAL_VIDEO_DIR_TEMPLATE

# ============ Expert services: single source of truth ============
declare -A SERVICES
SERVICES[transnetv2]="8001:1"
SERVICES[raft]="8002:2"
SERVICES[dinov2]="8003:2"
SERVICES[whisper_asr]="8004:3"
SERVICES[demucs]="8005:3"
SERVICES[e2quality]="8006:1"
SERVICES[panns]="8007:0"
SERVICES[dnsmos]="8008:0"
SERVICES[yolov8]="8009:2"
SERVICES[clip]="8010:1"
SERVICES[u2net]="8011:3"
SERVICES[movieshots]="8013:0"
SERVICES[sixdrepnet]="8014:0"
SERVICES[monst3r]="8015:1"

declare -A RESTART_COUNT
declare -A FAIL_COUNT

EVAL_PID=""
CLEANUP_DONE=0
SERVICES_STARTED=0

# ============ Logging ============
ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
log_warn() { echo "[$(ts)] ⚠️  $*"; }
log_error() { echo "[$(ts)] ❌ $*"; }
log_ok() { echo "[$(ts)] ✅ $*"; }
fail() {
  echo "[$(ts)] ERROR: $*" >&2
  exit 1
}

# ============ Model-status helpers ============
get_current_model_name() {
  "$PYTHON" - "$RUN_FILE" <<'PY'
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
match = re.search(r'(?m)^MODEL_NAME\s*=\s*["\']([^"\']+)["\']\s*$', text)
if not match:
    sys.exit(2)
print(match.group(1))
PY
}

set_model_name() {
  local model="$1"
  "$PYTHON" - "$RUN_FILE" "$model" <<'PY'
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
model = sys.argv[2]
text = path.read_text(encoding="utf-8")
new_text, n = re.subn(
    r'(?m)^MODEL_NAME\s*=\s*["\'][^"\']+["\']\s*$',
    f'MODEL_NAME= "{model}"',
    text,
    count=1,
)
if n != 1:
    print(f"Failed to replace MODEL_NAME in {path}", file=sys.stderr)
    sys.exit(2)
path.write_text(new_text, encoding="utf-8")
print(f'MODEL_NAME= "{model}"')
PY
}

# Reads the per-model status file written by run_final_joint_test.py --status-only.
# mode=print -> echo a one-line summary; mode=check -> exit 0 only when fully complete.
model_status() {
  local model="$1" mode="$2"
  "$PYTHON" - "$model" "$mode" <<'PY'
import json
import os
import sys
from pathlib import Path
model, mode = sys.argv[1], sys.argv[2]
_base = os.environ.get("EVAL_RESULT_BASE", "final")
_root = os.environ["EVAL_LOG_ROOT"]
path = Path(_root) / _base / model / f"{model}_sample_status.json"
if not path.exists():
    if mode == "print":
        print("missing_status")
    sys.exit(1)
with path.open(encoding="utf-8") as f:
    data = json.load(f)
summary = data.get("summary", {})
total = int(data.get("total_samples", 0) or 0)
success = int(summary.get("success", 0) or 0)
failed = int(summary.get("failed", 0) or 0)
pending = int(summary.get("pending", 0) or 0)
running = int(summary.get("running", 0) or 0)
if mode == "print":
    print(f"total={total} success={success} failed={failed} "
          f"running={running} pending={pending} status={path}")
    sys.exit(0)
complete = total > 0 and success == total and failed == 0 and pending == 0 and running == 0
sys.exit(0 if complete else 1)
PY
}

# ============ Service management ============
check_service_health() {
  local service=$1
  IFS=':' read -r port gpu <<< "${SERVICES[$service]}"
  local response
  response=$(curl -s --max-time "$HEALTH_TIMEOUT" "http://localhost:$port/health" 2>/dev/null)
  if [ $? -eq 0 ] && [ -n "$response" ]; then
    if echo "$response" | grep -q '"loaded"[[:space:]]*:[[:space:]]*false'; then
      return 1
    fi
    if echo "$response" | grep -q '"clip_loaded"[[:space:]]*:[[:space:]]*false'; then
      return 1
    fi
    FAIL_COUNT[$service]=0
    return 0
  fi
  return 1
}

start_single_service() {
  local service=$1
  IFS=':' read -r port gpu <<< "${SERVICES[$service]}"
  fuser -k "$port/tcp" 2>/dev/null
  sleep 1
  log "  Starting $service on port $port (GPU $gpu)..."
  cd "$SERVICE_DIR/$service" || return 1
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" app.py >"$SERVICE_LOG_DIR/${service}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$SERVICE_LOG_DIR/${service}.pid"
  log "  PID: $pid"
}

restart_service() {
  local service=$1
  local count=${RESTART_COUNT[$service]:-0}
  if [ "$count" -ge "$MAX_RESTART_PER_SERVICE" ]; then
    log_error "$service has been restarted $count times (max=$MAX_RESTART_PER_SERVICE), skipping"
    return 1
  fi
  RESTART_COUNT[$service]=$((count + 1))
  log_warn "Restarting $service (attempt ${RESTART_COUNT[$service]}/$MAX_RESTART_PER_SERVICE)..."
  start_single_service "$service"
  sleep "$RESTART_COOLDOWN"
  if check_service_health "$service"; then
    log_ok "$service restarted successfully"
    return 0
  fi
  log_error "$service failed to restart"
  return 1
}

start_all_services() {
  log "====== Starting all expert model services ======"
  SERVICES_STARTED=1
  log "Stopping any existing services..."
  local service port gpu
  for service in "${!SERVICES[@]}"; do
    IFS=':' read -r port gpu <<< "${SERVICES[$service]}"
    fuser -k "$port/tcp" 2>/dev/null
  done
  sleep 3
  for service in "${!SERVICES[@]}"; do
    start_single_service "$service"
    RESTART_COUNT[$service]=0
  done
  log "Waiting ${STARTUP_WAIT}s for services to initialize..."
  sleep "$STARTUP_WAIT"
}

wait_all_services_ready() {
  local max_rounds=3 round service port gpu
  for round in $(seq 1 $max_rounds); do
    log "--- Health check round $round/$max_rounds ---"
    local all_ready=true
    local failed_services=()
    for service in "${!SERVICES[@]}"; do
      IFS=':' read -r port gpu <<< "${SERVICES[$service]}"
      if check_service_health "$service"; then
        echo "  ✓ $service (port $port): OK"
      else
        echo "  ✗ $service (port $port): NOT READY"
        all_ready=false
        failed_services+=("$service")
      fi
    done
    if $all_ready; then
      log_ok "All services are ready!"
      return 0
    fi
    if [ $round -lt $max_rounds ]; then
      log_warn "Some services not ready: ${failed_services[*]}"
      log "Restarting failed services and waiting ${RESTART_COOLDOWN}s..."
      local svc
      for svc in "${failed_services[@]}"; do
        start_single_service "$svc"
      done
      sleep "$RESTART_COOLDOWN"
    fi
  done
  log_error "Some services still not ready after $max_rounds rounds"
  log "Proceeding anyway (watchdog will keep monitoring)..."
  return 1
}

# Cheap pre-model check: services stay up across models, so only revive the unhealthy ones
# instead of tearing all 15 down and paying the full reload cost again.
ensure_services_healthy() {
  local service
  local down=()
  for service in "${!SERVICES[@]}"; do
    check_service_health "$service" || down+=("$service")
  done
  if [ "${#down[@]}" -eq 0 ]; then
    log_ok "All ${#SERVICES[@]} services healthy"
    return 0
  fi
  log_warn "Services not healthy before this model: ${down[*]}"
  for service in "${down[@]}"; do
    restart_service "$service"
  done
}

# A pid file may be left over from a previous round, and a recycled PID can point at an
# unrelated process, so check /proc/<pid>/cmdline really is a service app.py before killing.
is_service_pid() {
  local pid="$1"
  [ -n "$pid" ] || return 1
  [ -r "/proc/$pid/cmdline" ] || return 1
  local cl
  cl=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || return 1
  case "$cl" in
    *python*app.py*) return 0 ;;
    *) return 1 ;;
  esac
}

stop_all_services() {
  log "====== Stopping all expert model services ======"
  local service port gpu pid pid_file
  for service in "${!SERVICES[@]}"; do
    pid_file="$SERVICE_LOG_DIR/${service}.pid"
    [ -f "$pid_file" ] || continue
    pid=$(cat "$pid_file" 2>/dev/null)
    if is_service_pid "$pid"; then
      kill "$pid" 2>/dev/null
    fi
  done
  sleep 3
  for service in "${!SERVICES[@]}"; do
    IFS=':' read -r port gpu <<< "${SERVICES[$service]}"
    pid_file="$SERVICE_LOG_DIR/${service}.pid"
    if [ -f "$pid_file" ]; then
      pid=$(cat "$pid_file" 2>/dev/null)
      if is_service_pid "$pid"; then
        log_warn "  $service (PID=$pid) still alive after TERM, sending KILL"
        kill -9 "$pid" 2>/dev/null
      fi
      rm -f "$pid_file"
    fi
    # fallback: kill leftovers this script did not start but that still hold the port
    fuser -k "$port/tcp" 2>/dev/null
  done
  log_ok "All services stopped"
}

# ============ Evaluation process + watchdog ============
stop_eval_process() {
  [ -n "$EVAL_PID" ] || return 0
  kill -0 "$EVAL_PID" 2>/dev/null || return 0
  log "Stopping evaluation process (PID=$EVAL_PID)..."
  kill "$EVAL_PID" 2>/dev/null
  local i
  for i in $(seq 1 10); do
    kill -0 "$EVAL_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$EVAL_PID" 2>/dev/null; then
    log_warn "Evaluation process did not exit on TERM, sending KILL"
    kill -9 "$EVAL_PID" 2>/dev/null
  fi
  wait "$EVAL_PID" 2>/dev/null
}

# Watches service health until the evaluation process exits.
run_watchdog() {
  local eval_pid=$1 service port gpu fc
  log "====== Watchdog started (interval=${WATCHDOG_INTERVAL}s) ======"
  while true; do
    if ! kill -0 "$eval_pid" 2>/dev/null; then
      log_ok "Evaluation process (PID=$eval_pid) has finished. Watchdog exiting."
      break
    fi
    for service in "${!SERVICES[@]}"; do
      if ! check_service_health "$service"; then
        IFS=':' read -r port gpu <<< "${SERVICES[$service]}"
        fc=${FAIL_COUNT[$service]:-0}
        fc=$((fc + 1))
        FAIL_COUNT[$service]=$fc
        if [ "$fc" -lt "$HEALTH_FAIL_THRESHOLD" ]; then
          log_warn "$service (port $port) health check failed ($fc/$HEALTH_FAIL_THRESHOLD), retrying next cycle..."
        else
          log_warn "$service (port $port) is DOWN! ($fc consecutive failures) Attempting restart..."
          FAIL_COUNT[$service]=0
          if [ -f "$SERVICE_LOG_DIR/${service}.log" ]; then
            echo "  Last 5 lines of ${service}.log:"
            tail -5 "$SERVICE_LOG_DIR/${service}.log" | sed 's/^/    /'
          fi
          restart_service "$service"
        fi
      fi
    done
    sleep "$WATCHDOG_INTERVAL"
  done
}

print_restart_summary() {
  local scope="$1" service count total=0
  log "====== Service restart summary (${scope}) ======"
  for service in "${!SERVICES[@]}"; do
    count=${RESTART_COUNT[$service]:-0}
    if [ "$count" -gt 0 ]; then
      echo "  ⚠️  $service: restarted $count time(s)"
      total=$((total + count))
    fi
  done
  if [ "$total" -eq 0 ]; then
    echo "  ✅ No services required restart"
  else
    echo "  Total restarts: $total"
  fi
}

# Runs one model end to end; returns the evaluation exit code.
run_one_model() {
  local model="$1"
  local rc=0 service

  # Restart budget is per model, matching the behaviour of the previous per-model subprocess
  for service in "${!SERVICES[@]}"; do
    RESTART_COUNT[$service]=0
    FAIL_COUNT[$service]=0
  done

  local current_line
  current_line="$(set_model_name "$model")" || fail "failed to set MODEL_NAME=$model"
  log "Set: $current_line"

  mkdir -p "${EVAL_LOG_ROOT}/${EVAL_RESULT_BASE}/$model"

  if [ "$SKIP_COMPLETED_MODELS" = "1" ]; then
    log "Refreshing model status: $model"
    "$PYTHON" "$RUN_FILE" --status-only "${EVAL_ARGS[@]}" \
      >"$LOG_BASE/${model}_status_refresh.log" 2>&1 \
      || fail "failed to refresh the status of model $model (see $LOG_BASE/${model}_status_refresh.log)"
    log "Status: $(model_status "$model" print)"
    if model_status "$model" check; then
      log_ok "Model $model is already fully complete, skipping"
      return 0
    fi
  fi

  ensure_services_healthy

  log "====== Starting evaluation: $model ======"
  log "Command: $PYTHON $RUN_FILE ${EVAL_ARGS[*]}"
  cd "$BENCHMARK_DIR" || fail "cannot enter $BENCHMARK_DIR"
  "$PYTHON" "$EVAL_SCRIPT" "${EVAL_ARGS[@]}" &
  EVAL_PID=$!
  log "Evaluation started with PID=$EVAL_PID"

  run_watchdog "$EVAL_PID"
  wait "$EVAL_PID"
  rc=$?
  EVAL_PID=""

  if [ "$rc" -eq 0 ]; then
    log_ok "Model $model evaluation finished (exit code: $rc)"
  else
    log_error "Model $model evaluation failed (exit code: $rc)"
  fi
  print_restart_summary "model=$model"
  return "$rc"
}

# ============ Cleanup / signals ============
cleanup() {
  local reason="${1:-exit}"
  [ "$CLEANUP_DONE" = "1" ] && return 0
  CLEANUP_DONE=1
  if [ "$SERVICES_STARTED" != "1" ]; then
    # Preflight bailed out before anything was launched; nothing to clean up.
    return 0
  fi
  log ""
  log "====== Cleanup (${reason}) ======"
  stop_eval_process
  if [ "$RESTORE_MODEL_NAME_ON_EXIT" = "1" ] && [ -n "${ORIGINAL_MODEL:-}" ]; then
    log "Restoring the original MODEL_NAME: $ORIGINAL_MODEL"
    set_model_name "$ORIGINAL_MODEL" >/dev/null || true
  fi
  stop_all_services
}

on_signal() {
  echo ""
  log_warn "Received $1, stopping the evaluation and all services..."
  cleanup signal
  exit "$2"
}

trap 'on_signal SIGINT 130'  SIGINT
trap 'on_signal SIGTERM 143' SIGTERM
trap 'cleanup exit'          EXIT

# ============ Main ============
# Arguments are forwarded to run_final_joint_test.py
EVAL_ARGS=("$@")
for arg in "${EVAL_ARGS[@]}"; do
  if [ "$arg" = "--model" ] || [[ "$arg" == --model=* ]]; then
    fail "do not pass --model; configure the model order in the MODELS list at the top of this script"
  fi
done

[ -n "${DASHSCOPE_API_KEY:-}" ] || fail "set the DASHSCOPE_API_KEY environment variable first"
[ -f "$RUN_FILE" ] || fail "evaluation script not found: $RUN_FILE"
[ -d "$SERVICE_DIR" ] || fail "expert service directory not found: ${SERVICE_DIR} (override with EXPERT_SERVICE_DIR)"
command -v "$PYTHON" >/dev/null 2>&1 || fail "Python unavailable: ${PYTHON} (override with EVAL_PYTHON)"
[ "${#MODELS[@]}" -gt 0 ] || fail "the MODELS list is empty"
[ -f "$EVAL_PROMPT_FILE" ] || fail "prompt file not found: ${EVAL_PROMPT_FILE}"
[ -f "$EVAL_QUESTION_BANK" ] || fail "question bank file not found: ${EVAL_QUESTION_BANK}"

# Every model in MODELS needs its own video subdirectory holding that model's .mp4 files.
for _model in "${MODELS[@]}"; do
  _video_dir="${EVAL_VIDEO_DIR_TEMPLATE//\{model\}/$_model}"
  [ -d "$_video_dir" ] || fail "video directory not found for model '${_model}': ${_video_dir} (override with EVAL_VIDEO_DIR_TEMPLATE)"
  case "$(echo "$_video_dir"/*.mp4)" in
    *'*.mp4') fail "no .mp4 files under ${_video_dir}" ;;
  esac
done
unset _model _video_dir

mkdir -p "$LOG_BASE" "$SERVICE_LOG_DIR"
RUN_LOG="$LOG_BASE/sequence_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

ORIGINAL_MODEL="$(get_current_model_name)" || fail "cannot read the current MODEL_NAME"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     CutCrafter multi-model evaluation (resume + watchdog)    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
log "Dataset: $EVAL_PROMPT_FILE"
log "Question bank: $EVAL_QUESTION_BANK"
log "Model order: ${MODELS[*]}"
log "Evaluation script: $RUN_FILE"
log "Evaluation args: ${EVAL_ARGS[*]:-(none, full run)}"
log "Expert services: ${#SERVICES[@]} in total, directory $SERVICE_DIR"
log "Service logs: $SERVICE_LOG_DIR"
log "Run log: $RUN_LOG"
log "MODEL_NAME before start: $ORIGINAL_MODEL"
echo ""

start_all_services
wait_all_services_ready

SEQUENCE_RC=0
for model in "${MODELS[@]}"; do
  echo ""
  log "========================================================"
  log "Starting evaluation of model: $model"
  log "========================================================"
  # Capture the status directly: inside `if ! cmd`, $? in the then-branch is always 0
  run_one_model "$model"
  model_rc=$?
  if [ "$model_rc" -ne 0 ]; then
    SEQUENCE_RC=$model_rc
    log_error "Model $model failed, remaining models were skipped. Log: $RUN_LOG"
    break
  fi
done

echo ""
log "========================================================"
if [ "$SEQUENCE_RC" -eq 0 ]; then
  log_ok "All models evaluated: ${MODELS[*]}"
else
  log_error "Evaluation sequence interrupted, exit code: $SEQUENCE_RC"
fi
log "========================================================"

exit "$SEQUENCE_RC"
