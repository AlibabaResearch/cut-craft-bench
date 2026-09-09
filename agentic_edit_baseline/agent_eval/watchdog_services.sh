#!/bin/bash
# Evaluation-service watchdog: checks every 60s and restarts whatever died.
# Usage: nohup bash watchdog_services.sh > /tmp/edit_agent_eval_services/watchdog.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL=60
MAX_RESTART=3
restart_count=0

while true; do
    ALL_OK=true
    for port in 8101 8104 8105 8107; do
        if ! curl -s --max-time 3 "http://localhost:${port}/health" >/dev/null 2>&1; then
            ALL_OK=false
            echo "[$(date)] Port ${port} DOWN"
        fi
    done

    if [ "$ALL_OK" = false ]; then
        echo "[$(date)] Service down detected, trying to restart..."
        bash "${SCRIPT_DIR}/start_agent_services.sh" start 2>&1
        restart_count=$((restart_count + 1))
        if [ $restart_count -ge $MAX_RESTART ]; then
            echo "[$(date)] Reached the maximum of ${MAX_RESTART} restarts, exiting the watchdog"
            break
        fi
    else
        # reset the counter while the service is healthy
        restart_count=0
    fi

    sleep $INTERVAL
done
