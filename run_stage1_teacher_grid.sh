#!/usr/bin/env bash
set -euo pipefail

repo=/home/paichichi/projects/TCC-core
python_bin=/home/paichichi/miniconda3/envs/tcc-core-parity/bin/python
train_script=scripts/train_multiview_softdtw_hparam.py
launch_dir="$repo/wsl_result/tcc_core_runs/launchers/fixed_tq_stage1_teacher_grid_20260806"

mkdir -p "$launch_dir"
cd "$repo"

rhos=(0.125 0.25 0.5 1.0 2.0)
epsilons=(0.025 0.05 0.1 0.2)

tag_number() {
  printf '%s' "$1" | sed 's/\./p/g'
}

run_family() {
  local family=$1
  local config=$2
  for rho in "${rhos[@]}"; do
    for epsilon in "${epsilons[@]}"; do
      local rho_tag
      local epsilon_tag
      rho_tag=$(tag_number "$rho")
      epsilon_tag=$(tag_number "$epsilon")
      local run_name="hps1_${family}_rho${rho_tag}_eps${epsilon_tag}_i00500"
      local run_dir="$repo/wsl_result/tcc_core_runs/$run_name"
      local run_log="$launch_dir/${run_name}.log"
      if [[ -e "$run_dir" ]]; then
        echo "SKIP existing $run_name"
        continue
      fi
      echo "START $run_name rho=$rho epsilon=$epsilon"
      CUDA_VISIBLE_DEVICES=0 "$python_bin" "$train_script" \
        --config "$config" \
        --run-name "$run_name" \
        --max-iters 500 \
        --log-every 50 \
        --save-every 500 \
        --soft-alignment-rho "$rho" \
        --soft-alignment-epsilon "$epsilon" \
        >"$run_log" 2>&1
      echo "DONE $run_name"
    done
  done
}

run_family \
  "vit_imagenet" \
  "configs/linux_method3_vit_ln_single_8ts1v_b8_seed1_i40000.yaml"

run_family \
  "r3m_bn_bi" \
  "configs/linux_method3_resnet_bn_affine_symmetric_two_branch_single_8ts1v_b20_seed1_i40000.yaml"

echo "STAGE1_COMPLETE"
