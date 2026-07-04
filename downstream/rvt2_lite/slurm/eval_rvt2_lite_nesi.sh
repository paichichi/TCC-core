#!/usr/bin/env bash
#SBATCH --job-name=rvt2_lite_eval
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
TASKS=${TASKS:-"open_drawer push_buttons"}
EVAL_EPISODES=${EVAL_EPISODES:-10}
EPISODE_LENGTH=${EPISODE_LENGTH:-25}
RUNS=${RUNS:-"d4r hrp ours_6ts4v ours_8ts3v"}

source "${ACTIVATE_SCRIPT}"
cd "${RVT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RVT_ROOT}:${RVT_ROOT}/rvt:${RVT_ROOT}/rvt/libs/YARR:${RVT_ROOT}/rvt/libs/RLBench:${RVT_ROOT}/rvt/libs/PyRep:${RVT_ROOT}/rvt/libs/peract:${RVT_ROOT}/rvt/libs/peract_colab:${RVT_ROOT}/rvt/libs/point-renderer:${PYTHONPATH:-}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/paichichi/software/CoppeliaSim_4_1_0}"
export TORCH_LIB_DIR="${TORCH_LIB_DIR:-/home/paichichi/miniconda3/envs/${CONDA_ENV}/lib/python3.9/site-packages/torch/lib}"
export CONDA_LIB_DIR="${CONDA_LIB_DIR:-/home/paichichi/miniconda3/envs/${CONDA_ENV}/lib}"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CONDA_LIB_DIR}:${COPPELIASIM_ROOT}:${COPPELIASIM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

for run in ${RUNS}; do
  model_folder="${TCC_ROOT}/downstream/rvt2_lite/runs/base/${run}"
  echo "Evaluating ${model_folder}"
  python -m rvt.eval \
    --model-folder "${model_folder}" \
    --model-name model_0.pth \
    --tasks ${TASKS} \
    --eval-datafolder /home/paichichi/data/rvt/train/replay/replay_train \
    --eval-episodes "${EVAL_EPISODES}" \
    --episode-length "${EPISODE_LENGTH}" \
    --headless \
    --device "${DEVICE}" \
    --log-name lite_eval
done
