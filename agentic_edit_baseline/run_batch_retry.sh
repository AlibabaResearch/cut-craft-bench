#!/bin/bash
# ============================================================
# Edit baseline agent batch retry.
# Regenerates only the final videos missing from the output directory.
#
# Detection: read every id from prompt_json, scan output_dir for .mp4 files whose name
# starts with a numeric id; the difference is the set of ids to re-run.
#
# Usage:
#   bash run_batch_retry.sh
#   bash run_batch_retry.sh --only-detect
#   bash run_batch_retry.sh --batch-size 5
#   bash run_batch_retry.sh --output-dir /path/to/wan
#   bash run_batch_retry.sh --prompt-json /path/to/prompt.json
#   bash run_batch_retry.sh --dry-run
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_baseline.sh"
DEFAULT_CONFIG="${SCRIPT_DIR}/config.yaml"

CONFIG="${DEFAULT_CONFIG}"
PROMPT_JSON_OVERRIDE=""
OUTPUT_DIR_OVERRIDE=""
BATCH_SIZE=10
MIN_SIZE_BYTES=1024
ONLY_DETECT=false
DRY_RUN=false
EXTRA_ARGS=()

usage() {
    sed -n '1,18p' "$0"
    echo "Options:"
    echo "  --config PATH             path to the edit_baseline config.yaml"
    echo "  --prompt-json PATH        override config.paths.prompt_json"
    echo "  --output-dir DIR          override config.paths.output_dir, and detect finished mp4 files there"
    echo "  --batch-size N            number of ids per run_baseline.sh batch, default 10"
    echo "  --min-size-bytes N        an mp4 counts as finished only above this size, default 1024"
    echo "  --only-detect             only detect the missing ids, do not generate"
    echo "  --dry-run                 pass --dry-run to run_baseline.sh: refresh planning only, no video"
    echo "  --                        everything after this is forwarded verbatim to run_baseline.sh"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --prompt-json)
            PROMPT_JSON_OVERRIDE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --min-size-bytes)
            MIN_SIZE_BYTES="$2"
            shift 2
            ;;
        --only-detect)
            ONLY_DETECT=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            echo "[ERROR] unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ ! -f "${RUN_SCRIPT}" ]; then
    echo "[ERROR] entry script not found: ${RUN_SCRIPT}"
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] config file not found: ${CONFIG}"
    exit 1
fi

if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [ "${BATCH_SIZE}" -lt 1 ]; then
    echo "[ERROR] --batch-size must be a positive integer"
    exit 1
fi

if ! [[ "${MIN_SIZE_BYTES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --min-size-bytes must be a non-negative integer"
    exit 1
fi

detect_missing() {
    CONFIG_PATH="${CONFIG}" \
    BASE_DIR="${SCRIPT_DIR}" \
    PROMPT_JSON_OVERRIDE="${PROMPT_JSON_OVERRIDE}" \
    OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE}" \
    MIN_SIZE_BYTES="${MIN_SIZE_BYTES}" \
    python3 - <<'PY'
import json
import os
import re
import sys

sys.path.insert(0, os.environ["BASE_DIR"])

try:
    import yaml
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] PyYAML is required to read config.yaml: {exc}", file=sys.stderr)
    sys.exit(2)

from lib.config_paths import resolve_config_paths, resolve_path

config_path = os.environ["CONFIG_PATH"]
prompt_override = os.environ.get("PROMPT_JSON_OVERRIDE") or ""
output_override = os.environ.get("OUTPUT_DIR_OVERRIDE") or ""
min_size = int(os.environ.get("MIN_SIZE_BYTES") or "0")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = resolve_config_paths(yaml.safe_load(f) or {})
paths = cfg.get("paths") or {}
prompt_json = resolve_path(prompt_override) or paths.get("prompt_json")
output_dir = resolve_path(output_override) or paths.get("output_dir")

if not prompt_json or not os.path.isfile(prompt_json):
    print(f"[ERROR] prompt_json not found: {prompt_json}", file=sys.stderr)
    sys.exit(3)
if not output_dir:
    print("[ERROR] config.paths.output_dir is empty and --output-dir was not given", file=sys.stderr)
    sys.exit(4)

with open(prompt_json, "r", encoding="utf-8") as f:
    cases = json.load(f)

all_ids = sorted(int(c["id"]) for c in cases if "id" in c)
generated_ids = set()
pattern = re.compile(r"^(\d+)(?:_|\b).*\.mp4$", re.IGNORECASE)

if os.path.isdir(output_dir):
    for fname in os.listdir(output_dir):
        match = pattern.match(fname)
        if not match:
            continue
        path = os.path.join(output_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) <= min_size:
                continue
        except OSError:
            continue
        generated_ids.add(int(match.group(1)))

missing = [cid for cid in all_ids if cid not in generated_ids]
print(prompt_json)
print(output_dir)
print(len(all_ids))
print(len(generated_ids))
print(len(missing))
print(" ".join(str(x) for x in missing))
PY
}

echo "============================================================"
echo "  edit_baseline batch retry: detecting missing final cuts"
echo "============================================================"

DETECT_OUTPUT=$(detect_missing)
DETECT_STATUS=$?
if [ "${DETECT_STATUS}" -ne 0 ]; then
    exit "${DETECT_STATUS}"
fi
mapfile -t DETECT_INFO <<< "${DETECT_OUTPUT}"

PROMPT_JSON="${DETECT_INFO[0]:-}"
OUTPUT_DIR="${DETECT_INFO[1]:-}"
TOTAL_COUNT="${DETECT_INFO[2]:-0}"
GENERATED_COUNT="${DETECT_INFO[3]:-0}"
MISSING_COUNT="${DETECT_INFO[4]:-0}"
MISSING_IDS="${DETECT_INFO[5]:-}"

echo "  Prompt JSON: ${PROMPT_JSON}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Total cases: ${TOTAL_COUNT}"
echo "  Finished:    ${GENERATED_COUNT}"
echo "  Missing:     ${MISSING_COUNT}"

if [ -z "${MISSING_IDS}" ]; then
    echo "  All final cuts already exist, nothing to retry."
    exit 0
fi

echo "  Missing ids: ${MISSING_IDS}"
echo "============================================================"

if [ "${ONLY_DETECT}" = true ]; then
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

IFS=' ' read -ra RETRY_IDS <<< "${MISSING_IDS}"
TOTAL_RETRY=${#RETRY_IDS[@]}
BATCH_OK=0
BATCH_FAILED=0

echo "  Re-running missing cases: ${TOTAL_RETRY} total, ${BATCH_SIZE} per batch"
echo "============================================================"

for ((start=0; start<TOTAL_RETRY; start+=BATCH_SIZE)); do
    batch=("${RETRY_IDS[@]:start:BATCH_SIZE}")
    batch_ids=$(IFS=,; echo "${batch[*]}")
    batch_no=$((start / BATCH_SIZE + 1))

    echo ""
    echo "------------------------------------------------------------"
    echo "  [BATCH ${batch_no}] ids=${batch_ids}"
    echo "------------------------------------------------------------"

    cmd=(bash "${RUN_SCRIPT}" --config "${CONFIG}" --ids "${batch_ids}")
    if [ -n "${PROMPT_JSON_OVERRIDE}" ]; then
        cmd+=(--prompt-json "${PROMPT_JSON_OVERRIDE}")
    fi
    if [ -n "${OUTPUT_DIR_OVERRIDE}" ]; then
        cmd+=(--output-dir "${OUTPUT_DIR_OVERRIDE}")
    fi
    if [ "${DRY_RUN}" = true ]; then
        cmd+=(--dry-run)
    fi
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    if "${cmd[@]}"; then
        BATCH_OK=$((BATCH_OK + 1))
    else
        BATCH_FAILED=$((BATCH_FAILED + 1))
        echo "  [WARN] batch failed: ids=${batch_ids}"
    fi
done

echo ""
echo "============================================================"
echo "  Retry commands finished, re-scanning the output directory"
echo "============================================================"
FINAL_OUTPUT=$(detect_missing)
mapfile -t FINAL_INFO <<< "${FINAL_OUTPUT}"
FINAL_MISSING_COUNT="${FINAL_INFO[4]:-0}"
FINAL_MISSING_IDS="${FINAL_INFO[5]:-}"

echo "  Batches ok: ${BATCH_OK} | batches failed: ${BATCH_FAILED}"
echo "  Still missing: ${FINAL_MISSING_COUNT}"
if [ -n "${FINAL_MISSING_IDS}" ]; then
    echo "  Still missing ids: ${FINAL_MISSING_IDS}"
else
    echo "  Every missing final cut has been generated."
fi
echo "  Output dir: ${OUTPUT_DIR}"
echo "============================================================"
