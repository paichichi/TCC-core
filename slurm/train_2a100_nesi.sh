#!/usr/bin/env bash
#SBATCH --job-name=hralign_2a100
#SBATCH --account=uoa04758
#SBATCH --partition=genoa,milan
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/nesi/project/uoa04758/xzha593/logs/%x-%j.out
#SBATCH --error=/nesi/project/uoa04758/xzha593/logs/%x-%j.err

set -euo pipefail

TCC_REPO_ROOT=${TCC_REPO_ROOT:-/nesi/project/uoa04758/xzha593/GitHub/TCC-core}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-/nesi/project/uoa04758/xzha593/envs/activate_tcc_core.sh}
LOG_DIR=${LOG_DIR:-/nesi/project/uoa04758/xzha593/logs}

EXP_CFG_PATH=${1:-${EXP_CFG_PATH:-}}
DEVICES=${TRAIN_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}

mkdir -p "${LOG_DIR}"

if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "Missing activate script: ${ACTIVATE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${TCC_REPO_ROOT}" ]]; then
  echo "Missing TCC repo root: ${TCC_REPO_ROOT}" >&2
  exit 1
fi

source "${ACTIVATE_SCRIPT}"
cd "${TCC_REPO_ROOT}"

if [[ -z "${EXP_CFG_PATH}" ]]; then
  echo "Usage: sbatch ${0} configs/nesi_vit_ln_8ts4v.yaml" >&2
  echo "Or set EXP_CFG_PATH=configs/nesi_vit_ln_8ts4v.yaml" >&2
  exit 1
fi

if [[ ! -f "${EXP_CFG_PATH}" ]]; then
  echo "Missing config: ${EXP_CFG_PATH}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "TCC_REPO_ROOT=${TCC_REPO_ROOT}"
echo "ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "TRAIN_DEVICES=${DEVICES}"
echo "EXP_CFG_PATH=${EXP_CFG_PATH}"
which python
nvidia-smi

python train.py \
  --exp_cfg_path "${EXP_CFG_PATH}" \
  --device "${DEVICES}"
