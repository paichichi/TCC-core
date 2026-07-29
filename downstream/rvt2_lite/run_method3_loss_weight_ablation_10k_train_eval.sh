#!/usr/bin/env bash
set -euo pipefail

RVT_ROOT=${RVT_ROOT:-/home/paichichi/projects/rvt-3d-policy-head-adaption}
TCC_ROOT=${TCC_ROOT:-/home/paichichi/projects/TCC-core}
TCC_PYTHON=${TCC_PYTHON:-/home/paichichi/miniconda3/envs/tcc-core/bin/python}
CONDA_ENV=${CONDA_ENV:-tcc-core}
CONDA_EXE=${CONDA_EXE:-/home/paichichi/miniconda3/bin/conda}
DEVICE=${DEVICE:-0}
MVT_CFG=${MVT_CFG:-${RVT_ROOT}/rvt/mvt/configs/rvt2.yaml}
TCC_ITERS=${TCC_ITERS:-10000}
RVT_TRAIN_ITERATIONS=${RVT_TRAIN_ITERATIONS:-25000}
EPISODES=${EPISODES:-0,1,2,3,4,5,6,7,8,9}
REPEATS=${REPEATS:-3}

STAMP=${STAMP:-method3_loss_weight_10k_$(date +%Y%m%d_%H%M%S)}
BASE="${TCC_ROOT}/wsl_result/downstream_rvt2_lite_runs"
TCC_LOG_DIR="${TCC_ROOT}/wsl_result/tcc_core_logs"
PRETRAIN_DIR="${TCC_ROOT}/wsl_result/downstream_pretrains"
RUN_ROOT="${BASE}/method3_loss_weight_10k_${STAMP}"
TRAIN_LOG_DIR="${BASE}/method3_loss_weight_10k_train_logs_${STAMP}"
EVAL_LOG_DIR="${BASE}/method3_loss_weight_10k_eval_logs_${STAMP}"
RAW="${BASE}/method3_loss_weight_10k_${STAMP}.csv"
SUMMARY="${BASE}/method3_loss_weight_10k_summary_${STAMP}.csv"
OVERALL="${BASE}/method3_loss_weight_10k_overall_${STAMP}.csv"
LEVELS="${BASE}/method3_loss_weight_10k_level_summary_${STAMP}.csv"
TABLE="${BASE}/method3_loss_weight_10k_table_${STAMP}.csv"

VIT_TCC_CFG="${TCC_ROOT}/configs/wsl_lite_method3_soft_alignment_8ts4v.yaml"
RESNET_TCC_CFG="${TCC_ROOT}/configs/wsl_lite_method3_resnet_ln_8ts4v.yaml"
VIT_RVT_CFG="${TCC_ROOT}/downstream/rvt2_lite/configs/wsl_method3_vit_ln_8ts4v.yaml"
RESNET_RVT_CFG="${TCC_ROOT}/downstream/rvt2_lite/configs/method3_wsl_resnet_ln_8ts4v.yaml"

TCC_RUNS=(
  "method3_vit_8ts4v_sa1_mv01_i10k:${VIT_TCC_CFG}:1.0:0.1:vit"
  "method3_vit_8ts4v_sa05_mv01_i10k:${VIT_TCC_CFG}:0.5:0.1:vit"
  "method3_resnet_asym_v2_8ts4v_sa1_mv01_i10k:${RESNET_TCC_CFG}:1.0:0.1:resnet"
  "method3_resnet_asym_v2_8ts4v_sa05_mv01_i10k:${RESNET_TCC_CFG}:0.5:0.1:resnet"
)

TRAIN_TASKS=(
  close_jar
  insert_onto_square_peg
  light_bulb_in
  meat_off_grill
  open_drawer
  place_cups
  place_shape_in_shape_sorter
  place_wine_at_rack_location
  push_buttons
  put_groceries_in_cupboard
  put_item_in_drawer
  put_money_in_safe
  reach_and_drag
  slide_block_to_color_target
  stack_blocks
  stack_cups
  sweep_to_dustpan_of_size
  turn_tap
)

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

TRAIN_TASKS_CSV=$(IFS=,; echo "${TRAIN_TASKS[*]}")
EVAL_TASKS=("${LEVEL1_TASKS[@]}" "${LEVEL2_TASKS[@]}")

mkdir -p "${TCC_LOG_DIR}" "${PRETRAIN_DIR}" "${RUN_ROOT}" "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RVT_ROOT}:${RVT_ROOT}/rvt:${RVT_ROOT}/rvt/libs/YARR:${RVT_ROOT}/rvt/libs/RLBench:${RVT_ROOT}/rvt/libs/PyRep:${RVT_ROOT}/rvt/libs/peract:${RVT_ROOT}/rvt/libs/peract_colab:${RVT_ROOT}/rvt/libs/point-renderer:${PYTHONPATH:-}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/paichichi/software/CoppeliaSim_4_1_0}"
export TORCH_LIB_DIR="${TORCH_LIB_DIR:-/home/paichichi/miniconda3/envs/${CONDA_ENV}/lib/python3.9/site-packages/torch/lib}"
export CONDA_LIB_DIR="${CONDA_LIB_DIR:-/home/paichichi/miniconda3/envs/${CONDA_ENV}/lib}"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CONDA_LIB_DIR}:${COPPELIASIM_ROOT}:${COPPELIASIM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

echo "STAMP=${STAMP}"
echo "TCC_ITERS=${TCC_ITERS}"
echo "RVT_TRAIN_ITERATIONS=${RVT_TRAIN_ITERATIONS}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TCC_LOG_DIR=${TCC_LOG_DIR}"
echo "TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
echo "EVAL_LOG_DIR=${EVAL_LOG_DIR}"
echo "TABLE=${TABLE}"

convert_tcc_resnet_checkpoint() {
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

RVT_RUNS=()

cd "${TCC_ROOT}"
for item in "${TCC_RUNS[@]}"; do
  IFS=: read -r run cfg lambda_sa lambda_mv backbone_type <<< "${item}"
  test -s "${cfg}"
  ckpt="${TCC_ROOT}/wsl_result/tcc_core_runs/${run}/checkpoint_$(printf '%06d' "${TCC_ITERS}").pt"

  if [ -s "${ckpt}" ]; then
    echo "TCC ${run} SKIP existing ${ckpt} $(date '+%F %T')"
  else
    echo "TCC ${run} START $(date '+%F %T') lambda_sa=${lambda_sa} lambda_mv=${lambda_mv}"
    CUDA_VISIBLE_DEVICES="${DEVICE}" "${TCC_PYTHON}" -u train.py \
      --exp_cfg_path "${cfg}" \
      --device 0 \
      -- \
      --run-name "${run}" \
      --lambda-sa "${lambda_sa}" \
      --lambda-mv "${lambda_mv}" \
      --max-iters "${TCC_ITERS}" \
      --save-every "${TCC_ITERS}" \
      --log-every 100 \
      > "${TCC_LOG_DIR}/${run}_${STAMP}.log" 2>&1
    echo "TCC ${run} DONE $(date '+%F %T')"
    grep -E "loaded paired_episodes|step |done in|losses=|NaN|nan|Traceback|RuntimeError" "${TCC_LOG_DIR}/${run}_${STAMP}.log" | tail -80 || true
  fi
  test -s "${ckpt}"

  if [ "${backbone_type}" = "resnet" ]; then
    pretrain="${PRETRAIN_DIR}/${run}_rvt2_resnet_pretrain.pt"
    echo "CONVERT+VALIDATE ${run} START $(date '+%F %T')"
    convert_tcc_resnet_checkpoint "${ckpt}" "${pretrain}"
    rvt_cfg="${RESNET_RVT_CFG}"
  else
    pretrain="${ckpt}"
    rvt_cfg="${VIT_RVT_CFG}"
  fi
  test -s "${pretrain}"
  RVT_RUNS+=("${run}:${rvt_cfg}:${pretrain}")
done

for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run cfg pretrain <<< "${item}"
  test -s "${cfg}"
  test -s "${pretrain}"
  echo "CHECK ${run}"
  echo "  cfg=${cfg}"
  echo "  pretrain=${pretrain}"
done

require_rvt_transfer_contract
cd "${RVT_ROOT}"
for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run cfg pretrain <<< "${item}"
  out_dir="${RUN_ROOT}/${run}"
  if [ -s "${out_dir}/model_0.pth" ]; then
    echo "TRAIN ${run} SKIP existing ${out_dir}/model_0.pth $(date '+%F %T')"
    if [[ "${run}" == *"resnet"* ]]; then
      require_strict_load_log "${TRAIN_LOG_DIR}/${run}.log"
    fi
    continue
  fi

  echo "TRAIN ${run} START $(date '+%F %T')"
  "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.train \
    --exp_cfg_path "${cfg}" \
    --mvt_cfg_path "${MVT_CFG}" \
    --device "${DEVICE}" \
    --exp_cfg_opts "tasks ${TRAIN_TASKS_CSV} train_iterations ${RVT_TRAIN_ITERATIONS} pretrain ${pretrain} overwriter_log_dir ${out_dir}" \
    > "${TRAIN_LOG_DIR}/${run}.log" 2>&1
  echo "TRAIN ${run} DONE $(date '+%F %T')"
  if [[ "${run}" == *"resnet"* ]]; then
    require_strict_load_log "${TRAIN_LOG_DIR}/${run}.log"
  fi
  grep -E "MVT_|ViTB16Backbone|Manually loading|matched keys|\\[Finish\\]|total_loss|trans_loss|nan|NaN|Traceback|RuntimeError" "${TRAIN_LOG_DIR}/${run}.log" | tail -80 || true
done

for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run _cfg _pretrain <<< "${item}"
  test -s "${RUN_ROOT}/${run}/model_0.pth"
  ls -lh "${RUN_ROOT}/${run}/model_0.pth"
done

echo 'run,repeat,task,success_rate,length,total_transitions,status' > "${RAW}"

for rep in $(seq 1 "${REPEATS}"); do
  for item in "${RVT_RUNS[@]}"; do
    IFS=: read -r run _cfg _pretrain <<< "${item}"
    log_name="method3_loss_weight_10k_${STAMP}_rep${rep}"
    outlog="${EVAL_LOG_DIR}/${run}_rep${rep}.log"
    echo "EVAL ${run} repeat=${rep} START $(date '+%F %T')"
    if "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.eval \
      --model-folder "${RUN_ROOT}/${run}" \
      --model-name model_0.pth \
      --tasks "${EVAL_TASKS[@]}" \
      --eval-datafolder /home/paichichi/data/AGNOSTOS/unseen_tasks/test \
      --eval-episode-list "${EPISODES}" \
      --episode-length 25 \
      --headless \
      --device "${DEVICE}" \
      --log-name "${log_name}" \
      --no-tensorboard \
      --no-env-shutdown \
      > "${outlog}" 2>&1; then
      csv="${RUN_ROOT}/${run}/eval/${log_name}/model_0/eval_results.csv"
      awk -F, -v run="${run}" -v rep="${rep}" 'NR>1 {gsub(/\r/,"",$4); print run","rep","$1","$2","$3","$4",ok"}' "${csv}" >> "${RAW}"
      echo "EVAL ${run} repeat=${rep} DONE $(date '+%F %T')"
    else
      echo "${run},${rep},ALL,,,,error" >> "${RAW}"
      echo "EVAL ${run} repeat=${rep} ERROR $(date '+%F %T')"
      tail -80 "${outlog}" || true
      exit 1
    fi
  done
done

"${TCC_PYTHON}" - "${RAW}" "${SUMMARY}" "${OVERALL}" "${LEVELS}" "${TABLE}" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

raw, summary, overall, levels, table = map(Path, sys.argv[1:])
level1 = {
    "close_fridge", "close_microwave", "close_laptop_lid",
    "toilet_seat_down", "open_grill", "phone_on_base",
}
level2 = {
    "take_usb_out_of_computer", "take_lid_off_saucepan", "turn_oven_on",
    "beat_the_buzz", "water_plants", "unplug_charger",
}
rows = list(csv.DictReader(raw.open()))
ok = [r for r in rows if r["status"] == "ok"]

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs):
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(max(0.0, sum((x - m) ** 2 for x in xs) / len(xs)))

by_run_task = defaultdict(list)
by_run_level = defaultdict(list)
by_run = defaultdict(list)
for row in ok:
    score = float(row["success_rate"])
    length = float(row["length"])
    run = row["run"]
    task = row["task"]
    by_run_task[(run, task)].append((score, length))
    by_run[run].append((score, length))
    if task in level1:
        by_run_level[(run, "level1")].append((score, length))
    elif task in level2:
        by_run_level[(run, "level2")].append((score, length))

with summary.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run", "task", "repeats", "mean_success_rate",
        "std_success_rate", "mean_length", "std_length",
    ])
    for (run, task), vals in sorted(by_run_task.items()):
        scores = [v[0] for v in vals]
        lengths = [v[1] for v in vals]
        writer.writerow([
            run, task, len(vals), f"{mean(scores):.2f}", f"{std(scores):.2f}",
            f"{mean(lengths):.2f}", f"{std(lengths):.2f}",
        ])

with overall.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "n", "overall_mean_success_rate", "overall_mean_length"])
    for run, vals in sorted(by_run.items()):
        writer.writerow([
            run, len(vals), f"{mean([v[0] for v in vals]):.2f}",
            f"{mean([v[1] for v in vals]):.2f}",
        ])

with levels.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "level", "n", "mean_success_rate", "mean_length"])
    for (run, level), vals in sorted(by_run_level.items()):
        writer.writerow([
            run, level, len(vals), f"{mean([v[0] for v in vals]):.2f}",
            f"{mean([v[1] for v in vals]):.2f}",
        ])

with table.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "overall", "level1", "level2"])
    for run in sorted(by_run):
        overall_score = mean([v[0] for v in by_run[run]])
        level1_score = mean([v[0] for v in by_run_level.get((run, "level1"), [])])
        level2_score = mean([v[0] for v in by_run_level.get((run, "level2"), [])])
        writer.writerow([
            run, f"{overall_score:.2f}", f"{level1_score:.2f}",
            f"{level2_score:.2f}",
        ])

print(table)
print(table.read_text())
PY

echo "FINISH $(date '+%F %T')"
