#!/usr/bin/env bash
set -euo pipefail

repo="/home/paichichi/projects/TCC-core"
python_bin="/home/paichichi/miniconda3/envs/tcc-core-parity/bin/python"
train_script="scripts/train_multiview_softdtw_hparam.py"
config="configs/linux_method3_vit_ln_single_8ts1v_b8_seed1_i40000.yaml"
run_name="hpsA_vit_imagenet_rho0p25_eps0p05_lr7p5e5_i40000"
run_dir="$repo/wsl_result/tcc_core_runs/$run_name"
state_dir="$repo/wsl_result/tcc_core_runs/launchers/managed_hparam_search"
log_file="$state_dir/${run_name}.log"
pack_complete="/home/paichichi/data/rh20t_prd03_transfer_20260806/PACK_COMPLETE"
search_dir="$repo/work/fixed_tq_hparam_search"
lite_marker="$repo/wsl_result/downstream_rvt2_lite_runs/hparam_search/$run_name/LITE_COMPLETE"

mkdir -p "$state_dir"

# Avoid competing with the one-time scan of millions of dataset files.
while [[ ! -f "$pack_complete" ]]; do
    printf '%s waiting for dataset packing to complete\n' "$(date -Iseconds)" \
        > "$state_dir/${run_name}.waiting"
    sleep 60
done

wait_for_gpu() {
    # Do not collide with an unexpected process that starts using the 5090.
    # The service remains pending and checks once per minute.
    while true; do
        used_mib="$(
            nvidia-smi --query-gpu=memory.used \
                --format=csv,noheader,nounits -i 0 | tr -d ' '
        )"
        if [[ "$used_mib" -lt 2048 ]]; then
            break
        fi
        printf '%s waiting: gpu0 memory.used=%s MiB\n' \
            "$(date -Iseconds)" "$used_mib" \
            > "$state_dir/${run_name}.waiting"
        sleep 60
    done
    rm -f "$state_dir/${run_name}.waiting"
}

rm -f "$state_dir/${run_name}.failed"
wait_for_gpu

cd "$repo"
if [[ ! -f "$run_dir/checkpoint_040000.pt" ]]; then
    resume_args=()
    checkpoint="$(
        find "$run_dir" -maxdepth 1 -type f -name 'checkpoint_*.pt' \
            2>/dev/null | sort -V | tail -1 || true
    )"
    if [[ -n "$checkpoint" ]]; then
        "$python_bin" "$search_dir/prepare_resume_losses.py" \
            "$run_dir" "$checkpoint"
        resume_args=(--resume "$checkpoint")
    fi

    printf '%s pid=%s\n' "$(date -Iseconds)" "$$" \
        > "$state_dir/${run_name}.running"
    set +e
    CUDA_VISIBLE_DEVICES=0 "$python_bin" "$train_script" \
        --config "$config" \
        --run-name "$run_name" \
        --max-iters 40000 \
        --log-every 50 \
        --save-every 5000 \
        --soft-alignment-rho 0.25 \
        --soft-alignment-epsilon 0.05 \
        --lr 0.000075 \
        "${resume_args[@]}" \
        >> "$log_file" 2>&1
    status=$?
    set -e
    rm -f "$state_dir/${run_name}.running"
    if [[ "$status" -ne 0 || ! -f "$run_dir/checkpoint_040000.pt" ]]; then
        [[ "$status" -ne 0 ]] || status=5
        printf '%s exit_code=%s\n' "$(date -Iseconds)" "$status" \
            > "$state_dir/${run_name}.failed"
        exit "$status"
    fi
fi

"$python_bin" "$search_dir/summarize_hparam_runs.py" \
    --manifest "$search_dir/stageA_5090.tsv" \
    --output-root "$repo/wsl_result/tcc_core_runs" \
    --output "$search_dir/stageA_5090_summary.csv"

if [[ ! -f "$lite_marker" ]]; then
    printf '%s\n' "$(date -Iseconds) lite running" \
        > "$state_dir/${run_name}.lite_running"
    set +e
    "$search_dir/run_rvt2_lite_candidate_5090.sh" \
        vit \
        "$run_name" \
        "$run_dir/checkpoint_040000.pt"
    status=$?
    set -e
    rm -f "$state_dir/${run_name}.lite_running"
    if [[ "$status" -ne 0 || ! -f "$lite_marker" ]]; then
        [[ "$status" -ne 0 ]] || status=6
        printf '%s lite_exit_code=%s\n' "$(date -Iseconds)" "$status" \
            > "$state_dir/${run_name}.failed"
        exit "$status"
    fi
fi

printf '%s\n' "$(date -Iseconds) first candidate complete" \
    > "$state_dir/${run_name}.done"
"$search_dir/run_5090_managed_queue.sh"
