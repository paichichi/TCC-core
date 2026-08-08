#!/usr/bin/env bash
set -euo pipefail

SHARE="/Users/xzha593/.local/share/tcc-hparam-search"
STATE="/Users/xzha593/.local/state/tcc-search-coordinator"
LIBEXEC="/Users/xzha593/.local/libexec"
LOCAL_HOST="paichichi@192.168.68.84"
LOCAL_SEARCH="/home/paichichi/projects/TCC-core/work/fixed_tq_hparam_search"
REMOTE_HOST="prd03"
REMOTE_SEARCH="/data/xzha593/project/TCC-core/hparam_search"
PYTHON="/usr/bin/python3"
LOG="$STATE/coordinator.log"
LOCK="$STATE/lock"

now() {
    date "+%Y-%m-%dT%H:%M:%S%z"
}

mkdir -p "$STATE"
if ! mkdir "$LOCK" 2>/dev/null; then
    exit 0
fi
cleanup() {
    rmdir "$LOCK" 2>/dev/null || true
}
trap cleanup EXIT

if [[ ! -f "$STATE/current_stage" ]]; then
    printf 'A1\n' > "$STATE/current_stage"
fi
stage="$(cat "$STATE/current_stage")"
if [[ "$stage" == "DONE" ]]; then
    exit 0
fi
if [[ ! "$stage" =~ ^(A1|A2|B|C)$ ]]; then
    printf '%s\tINVALID_STAGE\t%s\n' "$(now)" "$stage" >> "$LOG"
    exit 2
fi

manifest="$SHARE/${stage}_comparison.tsv"
if [[ ! -f "$manifest" ]]; then
    printf '%s\tWAITING_MANIFEST\t%s\n' "$(now)" "$manifest" >> "$LOG"
    exit 0
fi
if ! rsync -a "$LOCAL_HOST:$LOCAL_SEARCH/rvt2_lite_results.csv" \
    "$STATE/rvt2_lite_results.csv" < /dev/null; then
    printf '%s\tWAITING_LITE_RESULTS\t%s\n' "$(now)" "$stage" >> "$LOG"
    exit 0
fi

selection="$STATE/${stage}_selection.tsv"
audit="$STATE/${stage}_selection_audit.tsv"
if ! "$PYTHON" "$LIBEXEC/select_stage_winners.py" \
    --manifest "$manifest" \
    --lite-results "$STATE/rvt2_lite_results.csv" \
    --selection-output "$selection" \
    --audit-output "$audit" \
    --stage-label "$stage" \
    2> "$STATE/${stage}_selection_error.log"; then
    printf '%s\tWAITING_COMPLETE_LITE\t%s\n' "$(now)" "$stage" >> "$LOG"
    exit 0
fi

if [[ "$stage" == "A1" ]]; then
    "$PYTHON" "$LIBEXEC/summarize_platform_calibration.py" \
        --calibration "$SHARE/stageA_platform_calibration.tsv" \
        --lite-results "$STATE/rvt2_lite_results.csv" \
        --output "$STATE/stageA_platform_calibration_summary.csv"
    rsync -a "$STATE/stageA_platform_calibration_summary.csv" \
        "$LOCAL_HOST:$LOCAL_SEARCH/" < /dev/null
    rsync -a "$STATE/stageA_platform_calibration_summary.csv" \
        "$REMOTE_HOST:$REMOTE_SEARCH/" < /dev/null
fi

rsync -a "$selection" "$audit" "$LOCAL_HOST:$LOCAL_SEARCH/" < /dev/null
rsync -a "$selection" "$audit" "$REMOTE_HOST:$REMOTE_SEARCH/" < /dev/null

if [[ "$stage" == "C" ]]; then
    final_report_dir="$STATE/final_report"
    "$PYTHON" "$LIBEXEC/build_search_final_report.py" \
        --state-dir "$STATE" \
        --lite-results "$STATE/rvt2_lite_results.csv" \
        --provenance "$SHARE/experiment_provenance.yaml" \
        --output-dir "$final_report_dir"
    ssh -n -o BatchMode=yes "$LOCAL_HOST" \
        "mkdir -p '$LOCAL_SEARCH/final_report'"
    rsync -a "$final_report_dir/" \
        "$LOCAL_HOST:$LOCAL_SEARCH/final_report/" < /dev/null
    ssh -n -o BatchMode=yes "$REMOTE_HOST" \
        "mkdir -p '$REMOTE_SEARCH/final_report'"
    rsync -a "$final_report_dir/" \
        "$REMOTE_HOST:$REMOTE_SEARCH/final_report/" < /dev/null
    ssh -n -o BatchMode=yes "$LOCAL_HOST" \
        "touch '$LOCAL_SEARCH/SEARCH_ALL_COMPLETE'"
    printf 'DONE\n' > "$STATE/current_stage.tmp"
    mv "$STATE/current_stage.tmp" "$STATE/current_stage"
    printf '%s\tSEARCH_COMPLETE\n' "$(now)" >> "$LOG"
    exit 0
fi

case "$stage" in
    A1)
        next_stage="A2"
        generator_stage="lr"
        local_family="r3m_bn_bi"
        ;;
    A2)
        next_stage="B"
        generator_stage="q"
        local_family="vit_imagenet"
        ;;
    B)
        next_stage="C"
        generator_stage="weights"
        local_family="r3m_bn_bi"
        ;;
esac

new_manifest="$STATE/${next_stage}_new.tsv"
comparison_manifest="$SHARE/${next_stage}_comparison.tsv"
h200_manifest="$STATE/${next_stage}_h200.tsv"
local_manifest="$STATE/${next_stage}_local.tsv"
assignment_manifest="$STATE/${next_stage}_assignment.tsv"

"$PYTHON" "$LIBEXEC/generate_followup_manifests.py" "$generator_stage" \
    --selection "$selection" \
    --output "$new_manifest" \
    --comparison-output "$comparison_manifest"
"$PYTHON" "$LIBEXEC/partition_followup_manifest.py" \
    --input "$new_manifest" \
    --local-family "$local_family" \
    --h200-output "$h200_manifest" \
    --local-output "$local_manifest" \
    --assignment-output "$assignment_manifest"

"$PYTHON" "$LIBEXEC/merge_tsv_by_run_name.py" \
    --base "$SHARE/upstream_5090_all.tsv" \
    --additions "$local_manifest" \
    --output "$SHARE/upstream_5090_all.tsv"
"$PYTHON" "$LIBEXEC/merge_tsv_by_run_name.py" \
    --base "$SHARE/lite_intake_all.tsv" \
    --additions "$assignment_manifest" \
    --output "$SHARE/lite_intake_all.tsv"

rsync -a "$SHARE/upstream_5090_all.tsv" "$SHARE/lite_intake_all.tsv" \
    "$comparison_manifest" "$assignment_manifest" \
    "$LOCAL_HOST:$LOCAL_SEARCH/" < /dev/null
ssh -n -o BatchMode=yes "$LOCAL_HOST" \
    "rm -f '$LOCAL_SEARCH/ALL_CURRENT_LITE_COMPLETE'"

remote_h200_manifest="$REMOTE_SEARCH/${next_stage}_h200.tsv"
rsync -a "$h200_manifest" "$REMOTE_HOST:$remote_h200_manifest" < /dev/null
rsync -a "$comparison_manifest" "$assignment_manifest" "$selection" "$audit" \
    "$REMOTE_HOST:$REMOTE_SEARCH/" < /dev/null
ssh -n -o BatchMode=yes "$REMOTE_HOST" \
    "'$REMOTE_SEARCH/submit_followup_stage_prd03.sh' '$next_stage' '$remote_h200_manifest'"

printf '%s\n' "$next_stage" > "$STATE/current_stage.tmp"
mv "$STATE/current_stage.tmp" "$STATE/current_stage"
printf '%s\tADVANCED\t%s\t%s\n' "$(now)" "$stage" "$next_stage" >> "$LOG"
