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

TCC_REPO_ROOT=/nesi/project/uoa04758/xzha593/GitHub/TCC-core
ACTIVATE_SCRIPT=/nesi/project/uoa04758/xzha593/envs/activate_tcc_core.sh

EXP_CFG_PATH=${EXP_CFG_PATH:-}
RUN_NAME=${RUN_NAME:-train_2a100_lr7p5e5_20k}
NUM_TIMESTAMPS=${NUM_TIMESTAMPS:-8}
NUM_MULTI_VIEW=${NUM_MULTI_VIEW:-4}
BATCH_EPISODE_PAIRS=${BATCH_EPISODE_PAIRS:-${BATCH_PAIRS:-16}}
LR=${LR:-0.000075}
MAX_ITERS=${MAX_ITERS:-20000}
LOG_EVERY=${LOG_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-1000}
DEVICES=${TRAIN_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}

mkdir -p /nesi/project/uoa04758/xzha593/logs

source "${ACTIVATE_SCRIPT}"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "EXP_CFG_PATH=${EXP_CFG_PATH:-configs/train_2a100.yaml}"
echo "RUN_NAME=${RUN_NAME}"
echo "NUM_TIMESTAMPS=${NUM_TIMESTAMPS}"
echo "NUM_MULTI_VIEW=${NUM_MULTI_VIEW}"
echo "BATCH_EPISODE_PAIRS=${BATCH_EPISODE_PAIRS}"
echo "LR=${LR}"
echo "MAX_ITERS=${MAX_ITERS}"
echo "TRAIN_DEVICES=${DEVICES}"
which python
nvidia-smi

if [[ -n "${EXP_CFG_PATH}" ]]; then
  if [[ ! -f "${EXP_CFG_PATH}" ]]; then
    echo "Missing config: ${EXP_CFG_PATH}" >&2
    exit 1
  fi
  python train.py \
    --exp_cfg_path "${EXP_CFG_PATH}" \
    --device "${DEVICES}"
else
  python train.py \
    --exp_cfg_path configs/train_2a100.yaml \
    --device "${DEVICES}" \
    -- \
    --run-name "${RUN_NAME}" \
    --num-timestamps "${NUM_TIMESTAMPS}" \
    --num-multi-view "${NUM_MULTI_VIEW}" \
    --batch-episode-pairs "${BATCH_EPISODE_PAIRS}" \
    --lr "${LR}" \
    --max-iters "${MAX_ITERS}" \
    --log-every "${LOG_EVERY}" \
    --save-every "${SAVE_EVERY}"
fi
