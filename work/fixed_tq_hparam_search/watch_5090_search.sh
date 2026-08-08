#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/paichichi/projects/TCC-core"
SEARCH_DIR="$ROOT/work/fixed_tq_hparam_search"
STATE_DIR="$ROOT/wsl_result/tcc_core_runs/launchers/managed_hparam_search"
SERVICE="tcc-hps-5090-first.service"
TIMER="tcc-hps-5090-first.timer"
INITIAL_SERVICE="tcc-hps-5090-initial.service"
INITIAL_TIMER="tcc-hps-5090-initial.timer"
ENTRYPOINT="$ROOT/run_5090_first_full40k.sh"
LOG="$STATE_DIR/watchdog.log"
NOT_BEFORE_EPOCH=1785955896

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/watchdog.lock"
flock -n 9 || exit 0

if [[ -f "$SEARCH_DIR/SEARCH_ALL_COMPLETE" ]]; then
    exit 0
fi
if [[ "$(date +%s)" -lt "$NOT_BEFORE_EPOCH" ]]; then
    exit 0
fi
if systemctl --user is-active --quiet "$SERVICE" ||
   systemctl --user is-active --quiet "$INITIAL_SERVICE"; then
    exit 0
fi
# The initial three-hour delay has not elapsed yet.
if systemctl --user is-active --quiet "$TIMER" ||
   systemctl --user is-active --quiet "$INITIAL_TIMER"; then
    exit 0
fi
# Avoid colliding with a manually started worker even if it is outside the
# transient service cgroup.
if pgrep -f \
    'train_multiview_softdtw_hparam.py|run_5090_managed_queue.sh|python -m rvt.(train|eval)' \
    >/dev/null; then
    exit 0
fi

used_mib="$(
    nvidia-smi --query-gpu=memory.used \
        --format=csv,noheader,nounits -i 0 | tr -d ' '
)"
if [[ "$used_mib" -ge 2048 ]]; then
    printf '%s search inactive but gpu busy: memory.used=%s MiB\n' \
        "$(date -Iseconds)" "$used_mib" >> "$LOG"
    exit 0
fi

printf '%s search service inactive; restarting\n' "$(date -Iseconds)" >> "$LOG"
systemctl --user reset-failed "$SERVICE" 2>/dev/null || true
if systemctl --user start "$SERVICE" 2>/dev/null; then
    printf '%s restarted existing service\n' "$(date -Iseconds)" >> "$LOG"
    exit 0
fi

systemd-run --user \
    --unit=tcc-hps-5090-first \
    --description="Resume managed 5090 hyperparameter search" \
    "$ENTRYPOINT"
printf '%s recreated transient service\n' "$(date -Iseconds)" >> "$LOG"
