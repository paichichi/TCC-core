#!/usr/bin/env bash
#SBATCH --job-name=hralign_r3m_l
#SBATCH --account=uoa04758
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nesi/project/uoa04758/xzha593/GitHub/TCC-core}"
CONFIG="${1:-configs/hralign_r3m_l_nesi.yaml}"
ACTIVATE="${ACTIVATE:-/nesi/project/uoa04758/xzha593/envs/activate_tcc_core.sh}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"

cd "${PROJECT_ROOT}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${PROJECT_ROOT}/${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${ACTIVATE}" ]]; then
  echo "Missing environment activation script: ${ACTIVATE}" >&2
  exit 1
fi
source "${ACTIVATE}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this file with sbatch, or use the torchrun debug command in slurm/README.md." >&2
  exit 1
fi

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="$((10000 + SLURM_JOB_ID % 50000))"

export PROJECT_ROOT CONFIG GPUS_PER_NODE MASTER_ADDR MASTER_PORT
export ACTIVATE

echo "nodes=${SLURM_NNODES} gpus_per_node=${GPUS_PER_NODE}"
echo "rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
echo "config=${CONFIG}"

srun \
  --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  bash -c '
  set -euo pipefail
  source "${ACTIVATE}"
  cd "${PROJECT_ROOT}"
  exec torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --node-rank="${SLURM_NODEID}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    train.py \
    --config "${CONFIG}"
'
