#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TCC_REPO_ROOT="${TCC_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
JOB_SCRIPT="${SCRIPT_DIR}/train_2a100_nesi.sh"
DRY_RUN="${DRY_RUN:-0}"

CONFIGS=(
  "configs/nesi_vit_ln_8ts4v.yaml"
  "configs/nesi_vit_ln_8ts3v.yaml"
  "configs/nesi_r3m_bn_affine_8ts4v.yaml"
  "configs/nesi_r3m_late_adapter_8ts4v.yaml"
)

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Missing job script: ${JOB_SCRIPT}" >&2
  exit 1
fi

submit_job() {
  local cfg="$1"
  local stem
  stem="$(basename "${cfg}" .yaml)"
  local job_name="tcc_${stem#nesi_}"
  local cfg_path="${TCC_REPO_ROOT}/${cfg}"

  if [[ ! -f "${cfg_path}" ]]; then
    echo "Missing config: ${cfg_path}" >&2
    exit 1
  fi

  echo "Submitting ${job_name} with ${cfg}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN sbatch --job-name=${job_name} --export=ALL,TCC_REPO_ROOT=${TCC_REPO_ROOT} ${JOB_SCRIPT} ${cfg}"
  else
    sbatch \
      --job-name="${job_name}" \
      --export=ALL,TCC_REPO_ROOT="${TCC_REPO_ROOT}" \
      "${JOB_SCRIPT}" \
      "${cfg}"
  fi
}

echo "TCC_REPO_ROOT=${TCC_REPO_ROOT}"
echo "JOB_SCRIPT=${JOB_SCRIPT}"
echo "DRY_RUN=${DRY_RUN}"

for cfg in "${CONFIGS[@]}"; do
  submit_job "${cfg}"
done
