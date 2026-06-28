#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
JOB_SCRIPT="${SCRIPT_DIR}/train_2a100_nesi.sh"

declare -a JOBS=(
  "tcc_vit_ln_8ts4v:configs/nesi_vit_ln_8ts4v.yaml"
  "tcc_vit_ln_8ts3v:configs/nesi_vit_ln_8ts3v.yaml"
  "tcc_r3m_bn_8ts4v:configs/nesi_r3m_bn_affine_8ts4v.yaml"
  "tcc_r3m_late_8ts4v:configs/nesi_r3m_late_adapter_8ts4v.yaml"
)

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Missing job script: ${JOB_SCRIPT}" >&2
  exit 1
fi

for item in "${JOBS[@]}"; do
  job_name="${item%%:*}"
  cfg="${item#*:}"
  cfg_path="${PROJECT_ROOT}/${cfg}"

  if [[ ! -f "${cfg_path}" ]]; then
    echo "Missing config for ${job_name}: ${cfg_path}" >&2
    exit 1
  fi

  echo "Submitting ${job_name} with ${cfg}"
  sbatch \
    --job-name="${job_name}" \
    --export=ALL,EXP_CFG_PATH="${cfg}" \
    "${JOB_SCRIPT}"
done
