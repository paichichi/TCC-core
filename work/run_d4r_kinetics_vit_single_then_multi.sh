#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/paichichi/projects/TCC-core
PYTHON=/home/paichichi/miniconda3/envs/tcc-core-parity/bin/python
SINGLE_CFG=configs/linux_method3_vit_ln_d4r_kinetics_single_8ts1v_b8_seed1_i40000.yaml
MULTI_CFG=configs/linux_method3_vit_ln_d4r_kinetics_multi_8ts4v_b8_seed1_i40000.yaml

cd "${ROOT}"

echo "queue_started=$(date --iso-8601=seconds)"

if [[ ! -f wsl_result/tcc_core_runs/linux_method3_vit_ln_d4r_kinetics_single_8ts1v_b8_seed1_i40000/checkpoint_040000.pt ]]; then
  echo "single_started=$(date --iso-8601=seconds)"
  "${PYTHON}" train.py --exp_cfg_path "${SINGLE_CFG}" --device 0
  echo "single_finished=$(date --iso-8601=seconds)"
else
  echo "single_already_complete=$(date --iso-8601=seconds)"
fi

if [[ ! -f wsl_result/tcc_core_runs/linux_method3_vit_ln_d4r_kinetics_multi_8ts4v_b8_seed1_i40000/checkpoint_040000.pt ]]; then
  echo "multi_started=$(date --iso-8601=seconds)"
  "${PYTHON}" train.py --exp_cfg_path "${MULTI_CFG}" --device 0
  echo "multi_finished=$(date --iso-8601=seconds)"
else
  echo "multi_already_complete=$(date --iso-8601=seconds)"
fi

echo "queue_finished=$(date --iso-8601=seconds)"
