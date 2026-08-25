#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/artifacts/logs"
RUN_LOG="$LOG_DIR/solution_scenarios.log"
STATUS_FILE="$LOG_DIR/solution_scenarios.status"

cd "$REPO_ROOT" || exit 1
mkdir -p "$LOG_DIR"

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  printf 'ERROR: project Python is missing at %s\n' "$REPO_ROOT/.venv/bin/python" |
    tee "$STATUS_FILE"
  exit 1
fi

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export RAY_local_fs_capacity_threshold=0.98
export RAY_DEDUP_LOGS=0
export UV_HTTP_TIMEOUT=600

printf 'RUNNING: started %s\n' "$(date --iso-8601=seconds)" | tee "$STATUS_FILE"

"$REPO_ROOT/.venv/bin/python" -m ssfl.experiments.run_suite \
  --matrix "$REPO_ROOT/configs/experiments_solution.yaml" \
  --resume \
  >>"$RUN_LOG" 2>&1
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
  title="SSFL experiments completed"
  message="Scenarios 1, 2, and 3 completed successfully."
  sound="/usr/share/sounds/freedesktop/stereo/complete.oga"
  status="SUCCESS"
else
  title="SSFL experiments failed"
  message="The solution suite exited with code $exit_code. Check solution_scenarios.log."
  sound="/usr/share/sounds/freedesktop/stereo/dialog-error.oga"
  status="FAILED"
fi

printf '%s: %s (exit=%s)\n' "$status" "$(date --iso-8601=seconds)" "$exit_code" |
  tee "$STATUS_FILE"

if command -v notify-send >/dev/null 2>&1; then
  DISPLAY="${DISPLAY:-:0}" \
    DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}" \
    notify-send --urgency=critical "$title" "$message" >/dev/null 2>&1 || true
fi

if command -v paplay >/dev/null 2>&1 && [[ -f "$sound" ]]; then
  paplay "$sound" >/dev/null 2>&1 || true
fi

exit "$exit_code"
