#!/usr/bin/env bash
set -euo pipefail

RVT_ROOT=${RVT_ROOT:-/home/paichichi/projects/rvt-3d-policy-head-adaption}
TCC_ROOT=${TCC_ROOT:-/home/paichichi/projects/TCC-core}
TCC_PYTHON=${TCC_PYTHON:-/home/paichichi/miniconda3/envs/tcc-core/bin/python}
CONDA_EXE=${CONDA_EXE:-/home/paichichi/miniconda3/bin/conda}
CONDA_ENV=${CONDA_ENV:-tcc-core}
DEVICE=${DEVICE:-0}
STAMP=${STAMP:-method3_orig_weight_10k_$(date +%Y%m%d_%H%M%S)}

BASE="${TCC_ROOT}/wsl_result/downstream_rvt2_lite_runs"
RUN_ROOT="${BASE}/method3_orig_weight_10k_${STAMP}"
TCC_LOG_DIR="${TCC_ROOT}/wsl_result/tcc_core_logs"
TRAIN_LOG_DIR="${BASE}/method3_orig_weight_10k_train_logs_${STAMP}"
EVAL_LOG_DIR="${BASE}/method3_orig_weight_10k_eval_logs_${STAMP}"
PRETRAIN_DIR="${TCC_ROOT}/wsl_result/downstream_pretrains"
RAW="${BASE}/method3_orig_weight_10k_${STAMP}.csv"
TABLE="${BASE}/method3_orig_weight_10k_table_${STAMP}.csv"

RUNS=(
  "method3_vit_8ts4v_sa1_mv05_i10k:${TCC_ROOT}/configs/wsl_lite_method3_soft_alignment_8ts4v.yaml:${TCC_ROOT}/downstream/rvt2_lite/configs/wsl_method3_vit_ln_8ts4v.yaml:vit"
  "method3_resnet_asym_v2_8ts4v_sa1_mv05_i10k:${TCC_ROOT}/configs/wsl_lite_method3_resnet_ln_8ts4v.yaml:${TCC_ROOT}/downstream/rvt2_lite/configs/method3_wsl_resnet_ln_8ts4v.yaml:resnet"
)

TRAIN_TASKS="close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,place_wine_at_rack_location,push_buttons,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,slide_block_to_color_target,stack_blocks,stack_cups,sweep_to_dustpan_of_size,turn_tap"
EVAL_TASKS=(
  close_fridge close_microwave close_laptop_lid toilet_seat_down open_grill phone_on_base
  take_usb_out_of_computer take_lid_off_saucepan turn_oven_on beat_the_buzz water_plants unplug_charger
)

mkdir -p "${RUN_ROOT}" "${TCC_LOG_DIR}" "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}" "${PRETRAIN_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RVT_ROOT}:${RVT_ROOT}/rvt:${RVT_ROOT}/rvt/libs/YARR:${RVT_ROOT}/rvt/libs/RLBench:${RVT_ROOT}/rvt/libs/PyRep:${RVT_ROOT}/rvt/libs/peract:${RVT_ROOT}/rvt/libs/peract_colab:${RVT_ROOT}/rvt/libs/point-renderer:${PYTHONPATH:-}"
export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-/home/paichichi/software/CoppeliaSim_4_1_0}
export TORCH_LIB_DIR=${TORCH_LIB_DIR:-/home/paichichi/miniconda3/envs/${CONDA_ENV}/lib/python3.9/site-packages/torch/lib}
export CONDA_LIB_DIR=${CONDA_LIB_DIR:-/home/paichichi/miniconda3/envs/${CONDA_ENV}/lib}
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CONDA_LIB_DIR}:${COPPELIASIM_ROOT}:${COPPELIASIM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}

convert_resnet_checkpoint() {
  local source_ckpt="$1"
  local target_ckpt="$2"
  local converter="${TCC_ROOT}/scripts/convert_tcc_r3m_to_rvt.py"
  "${TCC_PYTHON}" "${converter}" convert \
    "${source_ckpt}" "${target_ckpt}" --force
  "${TCC_PYTHON}" "${converter}" validate \
    "${target_ckpt}" --source "${source_ckpt}"
}

require_rvt_transfer_contract() {
  grep -Fq 'TCC_RVT_TRANSFER_FORMAT = "tcc_rvt_resnet_v1"' \
    "${RVT_ROOT}/rvt/train.py"
  grep -Fq 'R3M_LATE_ADAPTER_LAYOUT = "post_layer4_sequential_v1"' \
    "${RVT_ROOT}/rvt/mvt/resnet.py"
}

require_strict_load_log() {
  local log_path="$1"
  if ! grep -Fq "Strict TCC/RVT convnet load OK" "${log_path}"; then
    echo "ERROR: RVT did not confirm the strict Method3 transfer load: ${log_path}" >&2
    tail -100 "${log_path}" >&2 || true
    exit 1
  fi
}

echo "STAMP=${STAMP}"
echo "RUN_ROOT=${RUN_ROOT}"

RVT_RUNS=()
cd "${TCC_ROOT}"
for item in "${RUNS[@]}"; do
  IFS=: read -r run tcc_cfg rvt_cfg backbone <<< "${item}"
  checkpoint="${TCC_ROOT}/wsl_result/tcc_core_runs/${run}/checkpoint_010000.pt"
  if [[ -s "${checkpoint}" ]]; then
    echo "TCC SKIP ${run}"
  else
    echo "TCC START ${run} $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES="${DEVICE}" "${TCC_PYTHON}" -u train.py \
      --exp_cfg_path "${tcc_cfg}" --device 0 -- \
      --run-name "${run}" --lambda-sa 1.0 --lambda-mv 0.5 \
      --max-iters 10000 --save-every 10000 --log-every 100 \
      > "${TCC_LOG_DIR}/${run}_${STAMP}.log" 2>&1
    echo "TCC DONE ${run} $(date '+%F %T')"
  fi
  [[ -s "${checkpoint}" ]]

  if [[ "${backbone}" == "resnet" ]]; then
    pretrain="${PRETRAIN_DIR}/${run}_rvt2_resnet_pretrain.pt"
    echo "CONVERT+VALIDATE ${run} ${pretrain}"
    convert_resnet_checkpoint "${checkpoint}" "${pretrain}"
  else
    pretrain="${checkpoint}"
  fi
  RVT_RUNS+=("${run}:${rvt_cfg}:${pretrain}")
done

require_rvt_transfer_contract
cd "${RVT_ROOT}"
for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run rvt_cfg pretrain <<< "${item}"
  output="${RUN_ROOT}/${run}"
  if [[ -s "${output}/model_0.pth" ]]; then
    echo "TRAIN SKIP ${run}"
  else
    echo "TRAIN START ${run} $(date '+%F %T')"
    "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.train \
      --exp_cfg_path "${rvt_cfg}" \
      --mvt_cfg_path "${RVT_ROOT}/rvt/mvt/configs/rvt2.yaml" \
      --device "${DEVICE}" \
      --exp_cfg_opts "tasks ${TRAIN_TASKS} train_iterations 25000 pretrain ${pretrain} overwriter_log_dir ${output}" \
      > "${TRAIN_LOG_DIR}/${run}.log" 2>&1
    echo "TRAIN DONE ${run} $(date '+%F %T')"
  fi
  if [[ "${run}" == *"resnet"* ]]; then
    require_strict_load_log "${TRAIN_LOG_DIR}/${run}.log"
  fi
  [[ -s "${output}/model_0.pth" ]]
done

echo 'run,repeat,task,success_rate,length,total_transitions,status' > "${RAW}"
for rep in 1 2 3; do
  for item in "${RVT_RUNS[@]}"; do
    IFS=: read -r run _ _ <<< "${item}"
    log_name="orig_weight_10k_${STAMP}_rep${rep}"
    csv="${RUN_ROOT}/${run}/eval/${log_name}/model_0/eval_results.csv"
    log="${EVAL_LOG_DIR}/${run}_rep${rep}.log"
    if [[ -s "${csv}" ]] && [[ "$(awk 'END {print NR}' "${csv}")" -ge 13 ]]; then
      echo "EVAL SKIP ${run} rep=${rep}"
    else
      echo "EVAL START ${run} rep=${rep} $(date '+%F %T')"
      "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.eval \
        --model-folder "${RUN_ROOT}/${run}" --model-name model_0.pth \
        --tasks "${EVAL_TASKS[@]}" \
        --eval-datafolder /home/paichichi/data/AGNOSTOS/unseen_tasks/test \
        --eval-episode-list 0,1,2,3,4,5,6,7,8,9 --episode-length 25 \
        --headless --device "${DEVICE}" --log-name "${log_name}" \
        --no-tensorboard --no-env-shutdown > "${log}" 2>&1
      echo "EVAL DONE ${run} rep=${rep} $(date '+%F %T')"
    fi
    awk -F, -v run="${run}" -v rep="${rep}" \
      'NR > 1 {gsub(/\r/, "", $4); print run "," rep "," $1 "," $2 "," $3 "," $4 ",ok"}' \
      "${csv}" >> "${RAW}"
  done
done

"${TCC_PYTHON}" - "${RAW}" "${TABLE}" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

raw, output = map(Path, sys.argv[1:])
level1 = {"close_fridge", "close_microwave", "close_laptop_lid", "toilet_seat_down", "open_grill", "phone_on_base"}
level2 = {"take_usb_out_of_computer", "take_lid_off_saucepan", "turn_oven_on", "beat_the_buzz", "water_plants", "unplug_charger"}
scores = defaultdict(lambda: defaultdict(list))
for row in csv.DictReader(raw.open()):
    score = float(row["success_rate"])
    scores[row["run"]]["overall"].append(score)
    if row["task"] in level1:
        scores[row["run"]]["level1"].append(score)
    elif row["task"] in level2:
        scores[row["run"]]["level2"].append(score)
with output.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["model", "overall", "level1", "level2"])
    for run in sorted(scores):
        values = scores[run]
        writer.writerow([run] + [f"{sum(values[key]) / len(values[key]):.2f}" for key in ("overall", "level1", "level2")])
print(output.read_text())
PY

echo "ALL DONE $(date '+%F %T')"
