#!/usr/bin/env bash
set -Eeuo pipefail

RVT_ROOT=/home/paichichi/projects/rvt-3d-policy-head-adaption
TCC_ROOT=/home/paichichi/projects/TCC-core
CONDA_ENV=tcc-core-parity
CONDA_EXE=/home/paichichi/miniconda3/bin/conda
CONDA_PREFIX_PATH=/home/paichichi/miniconda3/envs/tcc-core-parity
DEVICE=0
MVT_CFG="${RVT_ROOT}/rvt/mvt/configs/rvt2.yaml"
BASE_CFG="${TCC_ROOT}/downstream/rvt2_lite/configs/unadapted_r3m.yaml"
PRETRAIN="${TCC_ROOT}/wsl_result/downstream_pretrains/linux_method3_resnet_bn_affine_symmetric_two_branch_single_8ts1v_b20_seed1_i40000.pt"
TRAIN_ITERATIONS=25000
EPISODES_PER_REPEAT=10
REPEATS=3

STAMP=bn_affine_symmetric_40k_25k_3x10_fair_restart_20260802
BASE="${TCC_ROOT}/wsl_result/downstream_rvt2_lite_runs"
RUN_NAME=bn_affine_symmetric_two_branch_single_40k
RUN_ROOT="${BASE}/resnet_25k_${STAMP}"
OUT_DIR="${RUN_ROOT}/${RUN_NAME}"
TRAIN_LOG_DIR="${BASE}/resnet_25k_train_logs_${STAMP}"
EVAL_LOG_DIR="${BASE}/resnet_25k_eval_logs_${STAMP}"
RAW="${BASE}/resnet_25k_${STAMP}.csv"
SUMMARY="${BASE}/resnet_25k_summary_${STAMP}.csv"
OVERALL="${BASE}/resnet_25k_overall_${STAMP}.csv"
LEVELS="${BASE}/resnet_25k_level_summary_${STAMP}.csv"
TABLE="${BASE}/resnet_25k_table_${STAMP}.csv"
COMPARISON="${BASE}/resnet_25k_bn_direction_comparison_${STAMP}.csv"

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

mkdir -p "${RUN_ROOT}" "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}"
test -s "${BASE_CFG}"
test -s "${PRETRAIN}"
test -s "${MVT_CFG}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RVT_ROOT}:${RVT_ROOT}/rvt:${RVT_ROOT}/rvt/libs/YARR:${RVT_ROOT}/rvt/libs/RLBench:${RVT_ROOT}/rvt/libs/PyRep:${RVT_ROOT}/rvt/libs/peract:${RVT_ROOT}/rvt/libs/peract_colab:${RVT_ROOT}/rvt/libs/point-renderer:${PYTHONPATH:-}"
export COPPELIASIM_ROOT=/home/paichichi/software/CoppeliaSim_4_1_0
export LD_LIBRARY_PATH="${CONDA_PREFIX_PATH}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${COPPELIASIM_ROOT}/libcrypto.so.1.1:${COPPELIASIM_ROOT}/libssl.so.1.1"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM

echo "START $(date '+%F %T')"
echo "PROTOCOL=25k policy training; 12 tasks x 3 runs x 10 episodes"
echo "PRETRAIN=${PRETRAIN}"
echo "OUT_DIR=${OUT_DIR}"

if [[ ! -s "${OUT_DIR}/model_0.pth" ]]; then
  echo "TRAIN START $(date '+%F %T')"
  cd "${RVT_ROOT}"
  "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.train \
    --exp_cfg_path "${BASE_CFG}" \
    --mvt_cfg_path "${MVT_CFG}" \
    --device "${DEVICE}" \
    --exp_cfg_opts \
      "tasks ${TRAIN_TASKS_CSV} train_iterations ${TRAIN_ITERATIONS} pretrain ${PRETRAIN} overwriter_log_dir ${OUT_DIR}" \
    > "${TRAIN_LOG_DIR}/${RUN_NAME}.log" 2>&1
  echo "TRAIN DONE $(date '+%F %T')"
else
  echo "TRAIN SKIP existing ${OUT_DIR}/model_0.pth"
fi

test -s "${OUT_DIR}/model_0.pth"
ls -lh "${OUT_DIR}/model_0.pth"

echo 'run,repeat,task,success_rate,length,total_transitions,status' > "${RAW}"

cd "${RVT_ROOT}"
for rep in $(seq 1 "${REPEATS}"); do
  log_name="resnet_25k_${STAMP}_rep${rep}"
  outlog="${EVAL_LOG_DIR}/${RUN_NAME}_rep${rep}.log"
  eval_result_dir="${OUT_DIR}/eval/${log_name}"
  rm -rf "${eval_result_dir}"

  echo "EVAL repeat=${rep} START $(date '+%F %T')"
  if xvfb-run -a -e /dev/null \
    -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
    "${CONDA_PREFIX_PATH}/bin/python" -m rvt.eval \
      --model-folder "${OUT_DIR}" \
      --model-name model_0.pth \
      --tasks "${EVAL_TASKS[@]}" \
      --eval-datafolder /home/paichichi/data/AGNOSTOS/unseen_tasks/test \
      --start-episode 0 \
      --eval-episodes "${EPISODES_PER_REPEAT}" \
      --episode-length 25 \
      --headless \
      --device "${DEVICE}" \
      --log-name "${log_name}" \
      > "${outlog}" 2>&1; then
    csv="${eval_result_dir}/model_0/eval_results.csv"
    test -s "${csv}"
    awk -F, -v run="${RUN_NAME}" -v rep="${rep}" \
      'NR>1 {gsub(/\r/,"",$4); print run","rep","$1","$2","$3","$4",ok"}' \
      "${csv}" >> "${RAW}"
    echo "EVAL repeat=${rep} DONE $(date '+%F %T')"
  else
    echo "${RUN_NAME},${rep},ALL,,,,error" >> "${RAW}"
    echo "EVAL repeat=${rep} ERROR $(date '+%F %T')"
    tail -80 "${outlog}" || true
    exit 1
  fi
done

/home/paichichi/miniconda3/envs/tcc-core-parity/bin/python - \
  "${RAW}" "${SUMMARY}" "${OVERALL}" "${LEVELS}" "${TABLE}" "${COMPARISON}" <<'PY'
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

raw, summary, overall, levels, table, comparison = map(Path, sys.argv[1:])
level1 = {
    "close_fridge", "close_microwave", "close_laptop_lid",
    "toilet_seat_down", "open_grill", "phone_on_base",
}
level2 = {
    "take_usb_out_of_computer", "take_lid_off_saucepan", "turn_oven_on",
    "beat_the_buzz", "water_plants", "unplug_charger",
}
rows = list(csv.DictReader(raw.open()))
ok = [row for row in rows if row["status"] == "ok"]
if len(ok) != 36:
    raise ValueError(f"Expected 36 completed task-run rows, found {len(ok)}")

def mean(values):
    return sum(values) / len(values)

def std(values):
    avg = mean(values)
    return math.sqrt(max(0.0, sum((value - avg) ** 2 for value in values) / len(values)))

by_task = defaultdict(list)
by_level = defaultdict(list)
all_scores = []
for row in ok:
    score = float(row["success_rate"])
    task = row["task"]
    by_task[task].append(score)
    by_level["level1" if task in level1 else "level2"].append(score)
    all_scores.append(score)

with summary.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["task", "runs", "mean_success_rate", "std_success_rate"])
    for task in sorted(by_task):
        values = by_task[task]
        writer.writerow([task, len(values), f"{mean(values):.2f}", f"{std(values):.2f}"])

with overall.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["run", "n", "overall_mean_success_rate"])
    writer.writerow(["bn_affine_symmetric_two_branch_single_40k", len(all_scores), f"{mean(all_scores):.2f}"])

with levels.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["run", "level", "n", "mean_success_rate"])
    for level in ("level1", "level2"):
        values = by_level[level]
        writer.writerow(["bn_affine_symmetric_two_branch_single_40k", level, len(values), f"{mean(values):.2f}"])

sym_l1 = mean(by_level["level1"])
sym_l2 = mean(by_level["level2"])
sym_all = mean(all_scores)
with table.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["model", "overall", "level1", "level2"])
    writer.writerow(["bn_affine_symmetric_two_branch_single_40k", f"{sym_all:.2f}", f"{sym_l1:.2f}", f"{sym_l2:.2f}"])

oneway = {
    "model": "bn_affine_two_branch_single_40k",
    "overall": 29.44,
    "level1": 23.89,
    "level2": 35.00,
}
with comparison.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["model", "overall", "level1", "level2", "delta_overall_vs_oneway", "delta_level1_vs_oneway", "delta_level2_vs_oneway"])
    writer.writerow([oneway["model"], oneway["overall"], oneway["level1"], oneway["level2"], "0.00", "0.00", "0.00"])
    writer.writerow([
        "bn_affine_symmetric_two_branch_single_40k",
        f"{sym_all:.2f}", f"{sym_l1:.2f}", f"{sym_l2:.2f}",
        f"{sym_all - oneway['overall']:.2f}",
        f"{sym_l1 - oneway['level1']:.2f}",
        f"{sym_l2 - oneway['level2']:.2f}",
    ])

print(comparison.read_text())
PY

echo "FINISH $(date '+%F %T')"
cat "${COMPARISON}"
