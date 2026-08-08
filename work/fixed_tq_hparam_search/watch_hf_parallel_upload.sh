#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="/home/paichichi/data/rh20t_prd03_transfer_20260806"
STATE_DIR="${PACK_DIR}/hf_parallel_state"
SERVICE="tcc-hf-parallel-switch.service"
LOG="${STATE_DIR}/watchdog.log"
UPLOAD_LOG="${STATE_DIR}/parallel_upload.log"
PROGRESS_LOG="${STATE_DIR}/watchdog_progress.log"

mkdir -p "$STATE_DIR"
exec 9>"${STATE_DIR}/watchdog.lock"
flock -n 9 || exit 0

if [[ -f "${STATE_DIR}/HF_UPLOAD_COMPLETE" ]]; then
    printf '%s\tCOMPLETE\n' "$(date -Iseconds)" >> "$PROGRESS_LOG"
    exit 0
fi

if systemctl --user is-active --quiet "$SERVICE"; then
    latest_progress="$(
        tail -c 200000 "$UPLOAD_LOG" 2>/dev/null \
            | tr '\r' '\n' \
            | grep -E '\[\+ [0-9]+ files\]' \
            | tail -1 \
            | perl -pe 's/\e\[[0-9;?]*[A-Za-z]//g' \
            || true
    )"
    printf '%s\tACTIVE\t%s\n' \
        "$(date -Iseconds)" "$latest_progress" >> "$PROGRESS_LOG"
    exit 0
fi

printf '%s upload service inactive; restarting\n' "$(date -Iseconds)" >> "$LOG"
systemctl --user reset-failed "$SERVICE" 2>/dev/null || true
if systemctl --user start "$SERVICE" 2>/dev/null; then
    printf '%s restarted existing service\n' "$(date -Iseconds)" >> "$LOG"
    exit 0
fi

# A collected transient unit can disappear after it exits. Recreate it with
# the same stable name and entrypoint; upload-large-folder resumes from its
# local metadata and skips files already present in the dataset repository.
systemd-run --user \
    --unit=tcc-hf-parallel-switch \
    --description="Resume parallel RH20T Hugging Face upload" \
    "${PACK_DIR}/switch_hf_upload_to_parallel.sh"
printf '%s recreated transient service\n' "$(date -Iseconds)" >> "$LOG"
