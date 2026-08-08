#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/paichichi/projects/rvt-3d-policy-head-adaption
RVT_ROOT="${PROJECT_ROOT}/rvt"
TCC_ROOT=/home/paichichi/projects/TCC-core
MODEL_RUN=${MODEL_RUN:-linux_method3_resnet_four_branch_tc_multi_8ts4v_b24_seed1_i40000_rvt2}
MODEL_FOLDER="${RVT_ROOT}/runs/${MODEL_RUN}"
MODEL_NAME=model_4.pth
DATA_ROOT=/home/paichichi/data/AGNOSTOS/unseen_tasks/test
OUTPUT_TAG=${OUTPUT_TAG:-fourbranch_multi_40k_eval_1x10}
LOG_PREFIX=${LOG_PREFIX:-fourbranch_multi_40k_lite}
OUTPUT_ROOT="${TCC_ROOT}/downstream/rvt2_lite/runs/${OUTPUT_TAG}"
CONTROLLER_LOG="${OUTPUT_ROOT}/controller.log"
EPISODES=10
EPISODE_LENGTH=25
GPU_ID=0

LEVEL1_TASKS=(
  close_fridge
  close_microwave
  close_laptop_lid
  toilet_seat_down
  open_grill
  phone_on_base
)

LEVEL2_TASKS=(
  take_usb_out_of_computer
  take_lid_off_saucepan
  turn_oven_on
  beat_the_buzz
  water_plants
  unplug_charger
)

TASKS=("${LEVEL1_TASKS[@]}" "${LEVEL2_TASKS[@]}")

set +u
source /home/paichichi/miniconda3/etc/profile.d/conda.sh
conda activate tcc-core-parity
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${RVT_ROOT}:${RVT_ROOT}/libs/RLBench:${RVT_ROOT}/libs/PyRep:${RVT_ROOT}/libs/YARR:${RVT_ROOT}/libs/peract_colab:${RVT_ROOT}/libs/peract:${RVT_ROOT}/libs/point-renderer:${PYTHONPATH:-}"
export COPPELIASIM_ROOT=/home/paichichi/software/CoppeliaSim_4_1_0
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${COPPELIASIM_ROOT}/libcrypto.so.1.1:${COPPELIASIM_ROOT}/libssl.so.1.1"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
unset QT_QPA_PLATFORM
unset QT_PLUGIN_PATH

test -s "${MODEL_FOLDER}/${MODEL_NAME}"
mkdir -p "${OUTPUT_ROOT}"
cd "${RVT_ROOT}"

echo "START $(date '+%F %T')" | tee -a "${CONTROLLER_LOG}"
echo "MODEL=${MODEL_FOLDER}/${MODEL_NAME}" | tee -a "${CONTROLLER_LOG}"
echo "PROTOCOL=12 tasks x 1 run x 10 episodes, episode_length=${EPISODE_LENGTH}" | tee -a "${CONTROLLER_LOG}"

for task in "${TASKS[@]}"; do
  for run_id in 1; do
    run_dir="${OUTPUT_ROOT}/${task}/run${run_id}"
    log_name="${LOG_PREFIX}_${task}_run${run_id}"
    eval_result_dir="${MODEL_FOLDER}/eval/${log_name}"

    rm -rf "${run_dir}" "${eval_result_dir}"
    mkdir -p "${run_dir}"

    echo "EVAL task=${task} run=${run_id} START $(date '+%F %T')" | tee -a "${CONTROLLER_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" xvfb-run -a \
      -e /dev/null \
      -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
      python -X faulthandler -u eval.py \
        --model-folder "${MODEL_FOLDER}" \
        --model-name "${MODEL_NAME}" \
        --tasks "${task}" \
        --eval-datafolder "${DATA_ROOT}" \
        --eval-episodes "${EPISODES}" \
        --episode-length "${EPISODE_LENGTH}" \
        --log-name "${log_name}" \
        --device 0 \
        --headless \
      2>&1 | tee "${run_dir}/stdout.log"

    result_csv="${eval_result_dir}/${MODEL_NAME%.pth}/eval_results.csv"
    test -s "${result_csv}"
    cp "${result_csv}" "${run_dir}/eval_results.csv"
    echo "EVAL task=${task} run=${run_id} DONE $(date '+%F %T')" | tee -a "${CONTROLLER_LOG}"
  done
done

echo "FINISH $(date '+%F %T')" | tee -a "${CONTROLLER_LOG}"
