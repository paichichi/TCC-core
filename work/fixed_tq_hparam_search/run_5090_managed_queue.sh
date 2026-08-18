#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/paichichi/projects/TCC-core"
SEARCH_DIR="$ROOT/work/fixed_tq_hparam_search"
UPSTREAM_MANIFEST="$SEARCH_DIR/upstream_5090_all.tsv"
LITE_MANIFEST="$SEARCH_DIR/lite_intake_all.tsv"
PYTHON="/home/paichichi/miniconda3/envs/tcc-core-parity/bin/python"
TRAIN_SCRIPT="$ROOT/scripts/train_multiview_softdtw_hparam.py"
OUTPUT_ROOT="$ROOT/wsl_result/tcc_core_runs"
STATE_DIR="$OUTPUT_ROOT/launchers/managed_hparam_search"
LITE_ROOT="$ROOT/wsl_result/downstream_rvt2_lite_runs/hparam_search"
STATUS="$SEARCH_DIR/status_5090.tsv"
SKIP_UPSTREAM="$SEARCH_DIR/skip_upstream_5090.txt"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
if [[ "$GPU_NAME" != *"RTX 5090"* ]]; then
    echo "Managed local queue requires RTX 5090; found: $GPU_NAME" >&2
    exit 3
fi
mkdir -p "$STATE_DIR" "$LITE_ROOT"

config_for() {
    case "$1" in
        vit_imagenet)
            echo "$ROOT/configs/linux_method3_vit_ln_single_8ts1v_b20_seed1_i40000.yaml"
            ;;
        r3m_bn_bi)
            echo "$ROOT/configs/linux_method3_resnet_bn_affine_symmetric_two_branch_single_8ts1v_b20_seed1_i40000.yaml"
            ;;
        *)
            return 2
            ;;
    esac
}

lite_family_for() {
    [[ "$1" == "vit_imagenet" ]] && echo vit || echo r3m_bn
}

skip_upstream_run() {
    local run_name="$1"
    [[ -f "$SKIP_UPSTREAM" ]] && grep -Fxq "$run_name" "$SKIP_UPSTREAM"
}

run_pending_lite() {
    local found=1 lite_status
    while IFS=$'\t' read -r run_name backbone _rest; do
        [[ "$run_name" == "run_name" || -z "$run_name" ]] && continue
        checkpoint="$OUTPUT_ROOT/$run_name/checkpoint_040000.pt"
        marker="$LITE_ROOT/$run_name/LITE_COMPLETE"
        if [[ -f "$checkpoint" && ! -f "$marker" ]]; then
            printf '%s\t%s\t%s\tLITE_RUNNING\n' \
                "$(date -Iseconds)" "$run_name" "$backbone" >> "$STATUS"
            "$SEARCH_DIR/run_rvt2_lite_candidate_5090.sh" \
                "$(lite_family_for "$backbone")" "$run_name" "$checkpoint"
            lite_status=$?
            if [[ "$lite_status" -ne 0 ]]; then
                printf '%s\t%s\t%s\tLITE_FAILED_EXIT_%s\n' \
                    "$(date -Iseconds)" "$run_name" "$backbone" \
                    "$lite_status" >> "$STATUS"
                return 2
            fi
            printf '%s\t%s\t%s\tLITE_COMPLETE\n' \
                "$(date -Iseconds)" "$run_name" "$backbone" >> "$STATUS"
            found=0
            break
        fi
    done < "$LITE_MANIFEST"
    return "$found"
}

all_lite_complete() {
    local run_name backbone rest
    while IFS=$'\t' read -r run_name backbone rest; do
        [[ "$run_name" == "run_name" || -z "$run_name" ]] && continue
        [[ -f "$LITE_ROOT/$run_name/LITE_COMPLETE" ]] || return 1
    done < "$LITE_MANIFEST"
    return 0
}

while true; do
    # Drain every completed candidate's RVT2-lite work before starting another
    # upstream 40k run.
    set +e
    run_pending_lite
    lite_status=$?
    set -e
    case "$lite_status" in
        0)
            continue
            ;;
        1)
            ;;
        *)
            exit "$lite_status"
            ;;
    esac

    pending_found=0
    while IFS=$'\t' read -r run_name backbone rho epsilon lr max_forward_step lambda_q lambda_sa lambda_mv; do
        [[ "$run_name" == "run_name" || -z "$run_name" ]] && continue
        if skip_upstream_run "$run_name"; then
            continue
        fi
        run_dir="$OUTPUT_ROOT/$run_name"
        [[ -f "$run_dir/checkpoint_040000.pt" ]] && continue
        pending_found=1
        config="$(config_for "$backbone")"
        resume_args=()
        current_step=0
        checkpoint="$(find "$run_dir" -maxdepth 1 -type f -name 'checkpoint_*.pt' 2>/dev/null | sort -V | tail -1 || true)"
        if [[ -n "$checkpoint" ]]; then
            checkpoint_name="$(basename "$checkpoint")"
            if [[ ! "$checkpoint_name" =~ ^checkpoint_([0-9]+)\.pt$ ]]; then
                echo "Cannot parse checkpoint step: $checkpoint" >&2
                exit 4
            fi
            current_step=$((10#${BASH_REMATCH[1]}))
            if [[ "$current_step" -ge 40000 ]]; then
                echo "Final checkpoint has unexpected name: $checkpoint" >&2
                exit 4
            fi
            "$PYTHON" "$SEARCH_DIR/prepare_resume_losses.py" "$run_dir" "$checkpoint"
            resume_args=(--resume "$checkpoint")
        fi
        target_step=$((current_step + 5000))
        [[ "$target_step" -le 40000 ]] || target_step=40000

        printf '%s\t%s\t%s\tUPSTREAM_RUNNING_TO_%s\n' \
            "$(date -Iseconds)" "$run_name" "$backbone" "$target_step" \
            >> "$STATUS"
        cd "$ROOT"
        CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TRAIN_SCRIPT" \
            --config "$config" \
            --output-root "$OUTPUT_ROOT" \
            --run-name "$run_name" \
            --max-iters "$target_step" \
            --log-every 50 \
            --save-every 5000 \
            --soft-alignment-rho "$rho" \
            --soft-alignment-epsilon "$epsilon" \
            --soft-alignment-max-forward-step "$max_forward_step" \
            --soft-alignment-struct-lambda "$lambda_q" \
            --lambda-sa "$lambda_sa" \
            --lambda-mv "$lambda_mv" \
            --lr "$lr" \
            "${resume_args[@]}" \
            >> "$STATE_DIR/${run_name}.log" 2>&1
        expected_checkpoint="$(printf '%s/checkpoint_%06d.pt' "$run_dir" "$target_step")"
        if [[ ! -f "$expected_checkpoint" ]]; then
            echo "Training segment did not create $expected_checkpoint" >&2
            exit 5
        fi
        printf '%s\t%s\t%s\tUPSTREAM_SEGMENT_COMPLETE_%s\n' \
            "$(date -Iseconds)" "$run_name" "$backbone" "$target_step" \
            >> "$STATUS"
        if [[ "$target_step" -eq 40000 ]]; then
            printf '%s\t%s\t%s\tUPSTREAM_COMPLETE\n' \
                "$(date -Iseconds)" "$run_name" "$backbone" >> "$STATUS"
            "$PYTHON" "$SEARCH_DIR/summarize_hparam_runs.py" \
                --manifest "$UPSTREAM_MANIFEST" \
                --output-root "$OUTPUT_ROOT" \
                --output "$SEARCH_DIR/upstream_5090_all_summary.csv" \
                --lite-results "$SEARCH_DIR/rvt2_lite_results.csv"
        fi
        break
    done < "$UPSTREAM_MANIFEST"

    if [[ "$pending_found" -eq 0 ]]; then
        if all_lite_complete; then
            touch "$SEARCH_DIR/ALL_CURRENT_LITE_COMPLETE"
            if [[ -f "$SEARCH_DIR/SEARCH_ALL_COMPLETE" ]]; then
                touch "$SEARCH_DIR/STAGE_A_5090_COMPLETE"
                exit 0
            fi
            printf '%s\tWAITING_FOR_NEXT_SEARCH_STAGE\n' "$(date -Iseconds)" \
                >> "$STATUS"
            sleep 900
            continue
        fi
        rm -f "$SEARCH_DIR/ALL_CURRENT_LITE_COMPLETE"
        printf '%s\tWAITING_FOR_IMPORTED_H200_LITE\n' "$(date -Iseconds)" \
            >> "$STATUS"
        sleep 900
    fi
done
