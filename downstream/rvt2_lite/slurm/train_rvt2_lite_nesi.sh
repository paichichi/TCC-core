#!/usr/bin/env bash
#SBATCH --job-name=rvt2_lite
#SBATCH --account=uoa04758
#SBATCH --partition=genoa,milan
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/nesi/project/uoa04758/xzha593/logs/%x-%j.out
#SBATCH --error=/nesi/project/uoa04758/xzha593/logs/%x-%j.err

set -euo pipefail

RVT_ROOT=${RVT_ROOT:-/home/paichichi/projects/rvt-3d-policy-head-adaption}
ACTIVATE_SCRIPT=${ACTIVATE_SCRIPT:-/nesi/project/uoa04758/xzha593/envs/activate_hralign.sh}
TCC_ROOT=${TCC_ROOT:-/home/paichichi/projects/TCC-core}
DEVICE=${DEVICE:-0}
MVT_CFG=${MVT_CFG:-mvt/configs/rvt2.yaml}

CONFIGS=${CONFIGS:-"d4r.yaml hrp.yaml ours_6ts4v.yaml ours_8ts3v.yaml"}

mkdir -p /nesi/project/uoa04758/xzha593/logs

source "${ACTIVATE_SCRIPT}"
cd "${RVT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RVT_ROOT}:${RVT_ROOT}/rvt:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "RVT_ROOT=${RVT_ROOT}"
echo "TCC_ROOT=${TCC_ROOT}"
echo "CONFIGS=${CONFIGS}"
which python
nvidia-smi

for cfg in ${CONFIGS}; do
  exp_cfg="${TCC_ROOT}/downstream/rvt2_lite/configs/${cfg}"
  mvt_cfg_path="${RVT_ROOT}/rvt/${MVT_CFG}"
  echo "Training ${exp_cfg}"
  python -m rvt.train \
    --exp_cfg_path "${exp_cfg}" \
    --mvt_cfg_path "${mvt_cfg_path}" \
    --device "${DEVICE}"
done
