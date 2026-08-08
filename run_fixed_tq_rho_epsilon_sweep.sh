#!/usr/bin/env bash
set -euo pipefail

repo=/home/paichichi/projects/TCC-core
python_bin=/home/paichichi/miniconda3/envs/tcc-core-parity/bin/python
train_script=scripts/train_multiview_softdtw.py
config=configs/linux_method3_vit_ln_d4r_kinetics_single_8ts1v_b8_seed1_i40000.yaml
launch_dir="$repo/wsl_result/tcc_core_runs/launchers/fixed_tq_rho_epsilon_sweep_20260806"

mkdir -p "$launch_dir"
cd "$repo"

specs=(
  "0.25:0.025:r0p25:e0p025"
  "0.25:0.05:r0p25:e0p05"
  "0.25:0.1:r0p25:e0p1"
  "0.5:0.025:r0p5:e0p025"
  "0.5:0.1:r0p5:e0p1"
  "1.0:0.025:r1p0:e0p025"
  "1.0:0.05:r1p0:e0p05"
  "1.0:0.1:r1p0:e0p1"
)

for spec in "${specs[@]}"; do
  IFS=: read -r rho epsilon rho_tag epsilon_tag <<<"$spec"
  run_name="linux_vit_kinetics_single_fixedtq_${rho_tag}_${epsilon_tag}_i01000"
  run_dir="$repo/wsl_result/tcc_core_runs/$run_name"
  run_log="$launch_dir/${run_name}.log"
  if [[ -e "$run_dir" ]]; then
    echo "SKIP existing $run_name"
    continue
  fi
  echo "START $run_name rho=$rho epsilon=$epsilon"
  CUDA_VISIBLE_DEVICES=0 "$python_bin" "$train_script" \
    --config "$config" \
    --run-name "$run_name" \
    --max-iters 1000 \
    --log-every 50 \
    --save-every 1000 \
    --soft-alignment-rho "$rho" \
    --soft-alignment-epsilon "$epsilon" \
    >"$run_log" 2>&1
  echo "DONE $run_name"
done

echo "SWEEP_COMPLETE"
