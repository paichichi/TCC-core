#!/usr/bin/env bash
#SBATCH --job-name=tcc_2a100
#SBATCH --account=uoa04758
#SBATCH --partition=genoa,milan
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/nesi/project/uoa04758/xzha593/logs/%x-%j.out
#SBATCH --error=/nesi/project/uoa04758/xzha593/logs/%x-%j.err

set -euo pipefail

TCC_REPO_ROOT=/nesi/project/uoa04758/xzha593/GitHub/TCC-core
ACTIVATE_SCRIPT=/nesi/project/uoa04758/xzha593/envs/activate_tcc_core.sh

RUN_NAME=${RUN_NAME:-d4rIN_cammatch_8ts4v_bp16x2_mv0p5_sdtwdiv_lr7p5e5_20k_s0}
NUM_TIMESTAMPS=${NUM_TIMESTAMPS:-8}
NUM_MULTI_VIEW=${NUM_MULTI_VIEW:-4}
BATCH_PAIRS=${BATCH_PAIRS:-16}
LR=${LR:-0.000075}
MAX_ITERS=${MAX_ITERS:-20000}
LOG_EVERY=${LOG_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-1000}

DEVICES=${TRAIN_DEVICES:-0,1}

LOG_DIR=/nesi/project/uoa04758/xzha593/logs
mkdir -p "${LOG_DIR}"

# ===== resource monitor start =====
MONITOR_INTERVAL=${MONITOR_INTERVAL:-60}
MONITOR_LOG="${LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.monitor.log"
SACCT_LOG="${LOG_DIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.sacct.log"

monitor_resources() {
  while true; do
    echo "============================================================"
    echo "[monitor] time=$(date)"
    echo "[monitor] host=$(hostname)"
    echo "[monitor] job_id=${SLURM_JOB_ID}"
    echo "[monitor] pwd=$(pwd)"
    echo

    echo "[free -h]"
    free -h || true
    echo

    echo "[top processes by RSS memory]"
    ps -u "$USER" -o pid,ppid,stat,pcpu,pmem,rss,vsz,comm,args --sort=-rss | head -30 || true
    echo

    echo "[nvidia-smi csv]"
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv || true
    echo

    echo "[slurm sstat]"
    sstat -j "${SLURM_JOB_ID}.batch" \
      --format=AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize \
      2>/dev/null || true
    echo

    sleep "${MONITOR_INTERVAL}"
  done
}

monitor_resources > "${MONITOR_LOG}" 2>&1 &
MONITOR_PID=$!

cleanup_monitor() {
  local exit_code=$?

  echo "============================================================" >> "${MONITOR_LOG}"
  echo "[monitor] stopping at $(date), exit_code=${exit_code}" >> "${MONITOR_LOG}"

  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
  fi

  echo "============================================================" > "${SACCT_LOG}"
  echo "[sacct final report] $(date)" >> "${SACCT_LOG}"

  sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU \
    >> "${SACCT_LOG}" 2>&1 || true

  exit "${exit_code}"
}

trap cleanup_monitor EXIT
# ===== resource monitor end =====

source "${ACTIVATE_SCRIPT}"
cd "${TCC_REPO_ROOT}"

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "RUN_NAME=${RUN_NAME}"
echo "NUM_TIMESTAMPS=${NUM_TIMESTAMPS}"
echo "NUM_MULTI_VIEW=${NUM_MULTI_VIEW}"
echo "BATCH_PAIRS=${BATCH_PAIRS}"
echo "LR=${LR}"
echo "MAX_ITERS=${MAX_ITERS}"
echo "TRAIN_DEVICES=${DEVICES}"

which python
python --version

ls -lh train.py
ls -lh configs/train_2a100.yaml
ls -lh /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/training_index.pt
ls -lh /nesi/nobackup/uoa04758/xzha593/datasets/HRAlign/pretrains/D4R_IN_1M.pth

nvidia-smi

python train.py \
  --exp_cfg_path configs/train_2a100.yaml \
  --device "${DEVICES}" \
  -- \
  --run-name "${RUN_NAME}" \
  --num-timestamps "${NUM_TIMESTAMPS}" \
  --num-multi-view "${NUM_MULTI_VIEW}" \
  --batch-pairs "${BATCH_PAIRS}" \
  --lr "${LR}" \
  --max-iters "${MAX_ITERS}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}"