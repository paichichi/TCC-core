#!/usr/bin/env bash
set -euo pipefail

RVT_ROOT=${RVT_ROOT:-/home/paichichi/projects/rvt-3d-policy-head-adaption}
TCC_ROOT=${TCC_ROOT:-/home/paichichi/projects/TCC-core}
TCC_PYTHON=${TCC_PYTHON:-/home/paichichi/miniconda3/envs/tcc-core/bin/python}
CONDA_ENV=${CONDA_ENV:-hralign_py39}
CONDA_EXE=${CONDA_EXE:-/home/paichichi/miniconda3/bin/conda}
DEVICE=${DEVICE:-0}
MVT_CFG=${MVT_CFG:-${RVT_ROOT}/rvt/mvt/configs/rvt2.yaml}
TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-25000}
EPISODES_PER_REPEAT=${EPISODES_PER_REPEAT:-10}
REPEATS=${REPEATS:-3}
SUITE=${SUITE:-six}

STAMP=${STAMP:-resnet_25k_$(date +%Y%m%d_%H%M%S)}
BASE="${TCC_ROOT}/wsl_result/downstream_rvt2_lite_runs"
PRETRAIN_DIR="${TCC_ROOT}/wsl_result/downstream_pretrains"
RUN_ROOT="${BASE}/resnet_25k_${STAMP}"
TRAIN_LOG_DIR="${BASE}/resnet_25k_train_logs_${STAMP}"
EVAL_LOG_DIR="${BASE}/resnet_25k_eval_logs_${STAMP}"
RAW="${BASE}/resnet_25k_${STAMP}.csv"
SUMMARY="${BASE}/resnet_25k_summary_${STAMP}.csv"
OVERALL="${BASE}/resnet_25k_overall_${STAMP}.csv"
LEVELS="${BASE}/resnet_25k_level_summary_${STAMP}.csv"
TABLE="${BASE}/resnet_25k_table_${STAMP}.csv"

TCC_RUNS=(
  "method3_wsl_resnet_ln_8ts4v:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method3_resnet_ln_8ts4v_b4_i3000/checkpoint_003000.pt:${PRETRAIN_DIR}/wsl_lite_method3_resnet_ln_8ts4v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method3_wsl_resnet_ln_8ts3v:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method3_resnet_ln_8ts3v_b4_i3000/checkpoint_003000.pt:${PRETRAIN_DIR}/wsl_lite_method3_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method1_wsl_resnet_ln_8ts3v:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_resnet_ln_8ts3v_b4_i3000/checkpoint_003000.pt:${PRETRAIN_DIR}/wsl_lite_method1_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method1_wsl_resnet_ln_8ts4v:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_resnet_ln_8ts4v_b4_i3000/checkpoint_003000.pt:${PRETRAIN_DIR}/wsl_lite_method1_resnet_ln_8ts4v_b4_i3000_rvt2_resnet_pretrain.pt"
)

RVT_RUNS=(
  "method3_wsl_resnet_ln_8ts4v:${TCC_ROOT}/downstream/rvt2_lite/configs/method3_wsl_resnet_ln_8ts4v.yaml:${PRETRAIN_DIR}/wsl_lite_method3_resnet_ln_8ts4v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method3_wsl_resnet_ln_8ts3v:${TCC_ROOT}/downstream/rvt2_lite/configs/method3_wsl_resnet_ln_8ts3v.yaml:${PRETRAIN_DIR}/wsl_lite_method3_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt"
  "unadapted_r3m:${TCC_ROOT}/downstream/rvt2_lite/configs/unadapted_r3m.yaml:/home/paichichi/data/pretrain/UnadaptedR3M.pt"
  "adapted_r3m:${TCC_ROOT}/downstream/rvt2_lite/configs/adapted_r3m.yaml:/home/paichichi/data/pretrain/AdaptedR3M.pyth"
  "method1_wsl_resnet_ln_8ts3v:${TCC_ROOT}/downstream/rvt2_lite/configs/method1_wsl_resnet_ln_8ts3v.yaml:${PRETRAIN_DIR}/wsl_lite_method1_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method1_wsl_resnet_ln_8ts4v:${TCC_ROOT}/downstream/rvt2_lite/configs/method1_wsl_resnet_ln_8ts4v.yaml:${PRETRAIN_DIR}/wsl_lite_method1_resnet_ln_8ts4v_b4_i3000_rvt2_resnet_pretrain.pt"
)

if [[ "${SUITE}" == "8ts1v_three" ]]; then
  TCC_RUNS=(
    "four_branch_control_single_40k:${TCC_ROOT}/wsl_result/tcc_core_runs/linux_method3_resnet_four_branch_tc_single_8ts1v_b24_seed1_i40000_from25000/checkpoint_040000.pt:${PRETRAIN_DIR}/linux_method3_resnet_four_branch_tc_single_8ts1v_b24_seed1_i40000_rvt2_resnet_pretrain.pt"
    "four_branch_control_spatial_single_40k:${TCC_ROOT}/wsl_result/tcc_core_runs/linux_method3_resnet_four_branch_tc_spatial_single_8ts1v_b24_seed1_i40000/checkpoint_040000.pt:${PRETRAIN_DIR}/linux_method3_resnet_four_branch_tc_spatial_single_8ts1v_b24_seed1_i40000_rvt2_resnet_pretrain.pt"
    "bn_affine_two_branch_single_40k:${TCC_ROOT}/wsl_result/tcc_core_runs/linux_method3_resnet_bn_affine_two_branch_single_8ts1v_b20_seed1_i40000/checkpoint_040000.pt:${PRETRAIN_DIR}/linux_method3_resnet_bn_affine_two_branch_single_8ts1v_b20_seed1_i40000_rvt2_resnet_pretrain.pt"
  )

  RVT_RUNS=(
    "four_branch_control_single_40k:${TCC_ROOT}/downstream/rvt2_lite/configs/adapted_r3m.yaml:${PRETRAIN_DIR}/linux_method3_resnet_four_branch_tc_single_8ts1v_b24_seed1_i40000_rvt2_resnet_pretrain.pt"
    "four_branch_control_spatial_single_40k:${TCC_ROOT}/downstream/rvt2_lite/configs/adapted_r3m.yaml:${PRETRAIN_DIR}/linux_method3_resnet_four_branch_tc_spatial_single_8ts1v_b24_seed1_i40000_rvt2_resnet_pretrain.pt"
    "bn_affine_two_branch_single_40k:${TCC_ROOT}/downstream/rvt2_lite/configs/unadapted_r3m.yaml:${PRETRAIN_DIR}/linux_method3_resnet_bn_affine_two_branch_single_8ts1v_b20_seed1_i40000_rvt2_resnet_pretrain.pt"
  )
elif [[ "${SUITE}" != "six" ]]; then
  echo "Unknown SUITE=${SUITE}; expected six or 8ts1v_three" >&2
  exit 2
fi

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

mkdir -p "${PRETRAIN_DIR}" "${RUN_ROOT}" "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RVT_ROOT}:${RVT_ROOT}/rvt:${RVT_ROOT}/rvt/libs/YARR:${RVT_ROOT}/rvt/libs/RLBench:${RVT_ROOT}/rvt/libs/PyRep:${RVT_ROOT}/rvt/libs/peract:${RVT_ROOT}/rvt/libs/peract_colab:${RVT_ROOT}/rvt/libs/point-renderer:${PYTHONPATH:-}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/paichichi/software/CoppeliaSim_4_1_0}"
export CONDA_PREFIX_PATH="${CONDA_PREFIX_PATH:-/home/paichichi/miniconda3/envs/${CONDA_ENV}}"
export CONDA_LIB_DIR="${CONDA_LIB_DIR:-${CONDA_PREFIX_PATH}/lib}"
export LD_LIBRARY_PATH="${CONDA_LIB_DIR}:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${COPPELIASIM_ROOT}/libcrypto.so.1.1:${COPPELIASIM_ROOT}/libssl.so.1.1"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM

echo "STAMP=${STAMP}"
echo "SUITE=${SUITE}"
echo "TRAIN_ITERATIONS=${TRAIN_ITERATIONS}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
echo "EVAL_LOG_DIR=${EVAL_LOG_DIR}"
echo "RAW=${RAW}"
echo "TABLE=${TABLE}"

convert_tcc_resnet_checkpoint() {
  local source_ckpt="$1"
  local target_ckpt="$2"
  "${TCC_PYTHON}" - "$source_ckpt" "$target_ckpt" <<'PY'
import sys
from pathlib import Path
import torch

source = Path(sys.argv[1])
target = Path(sys.argv[2])
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
state = checkpoint.get("model", checkpoint)
converted = {}
for key, value in state.items():
    if key.startswith("backbone.convnet."):
        converted[key.removeprefix("backbone.")] = value
target.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model": converted,
        "source_checkpoint": str(source),
        "stripped_prefix": "backbone.",
    },
    target,
)
print(f"converted {len(converted)} tensors: {source} -> {target}")
PY
}

for item in "${TCC_RUNS[@]}"; do
  IFS=: read -r run ckpt converted <<< "${item}"
  test -s "${ckpt}"
  if [ -s "${converted}" ]; then
    echo "CONVERT ${run} SKIP existing ${converted}"
  else
    echo "CONVERT ${run} START $(date '+%F %T')"
    convert_tcc_resnet_checkpoint "${ckpt}" "${converted}"
  fi
  test -s "${converted}"
done

for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run cfg pretrain <<< "${item}"
  test -s "${cfg}"
  test -s "${pretrain}"
  echo "CHECK ${run}"
  echo "  cfg=${cfg}"
  echo "  pretrain=${pretrain}"
done

cd "${RVT_ROOT}"
for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run cfg pretrain <<< "${item}"
  out_dir="${RUN_ROOT}/${run}"
  if [ -s "${out_dir}/model_0.pth" ]; then
    echo "TRAIN ${run} SKIP existing ${out_dir}/model_0.pth $(date '+%F %T')"
    continue
  fi

  echo "TRAIN ${run} START $(date '+%F %T')"
  "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.train \
    --exp_cfg_path "${cfg}" \
    --mvt_cfg_path "${MVT_CFG}" \
    --device "${DEVICE}" \
    --exp_cfg_opts "tasks ${TRAIN_TASKS_CSV} train_iterations ${TRAIN_ITERATIONS} pretrain ${pretrain} overwriter_log_dir ${out_dir}" \
    > "${TRAIN_LOG_DIR}/${run}.log" 2>&1
  echo "TRAIN ${run} DONE $(date '+%F %T')"
  grep -E "MVT_Resnet|Manually loading|matched keys|Adapter layers|\\[Finish\\]|total_loss|trans_loss|nan|NaN" "${TRAIN_LOG_DIR}/${run}.log" | tail -60 || true
done

for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run _cfg _pretrain <<< "${item}"
  test -s "${RUN_ROOT}/${run}/model_0.pth"
  ls -lh "${RUN_ROOT}/${run}/model_0.pth"
done

echo 'run,repeat,task,success_rate,length,total_transitions,status' > "${RAW}"

for rep in $(seq 1 "${REPEATS}"); do
  start_episode=0
  for item in "${RVT_RUNS[@]}"; do
    IFS=: read -r run _cfg _pretrain <<< "${item}"
    log_name="resnet_25k_${STAMP}_rep${rep}"
    outlog="${EVAL_LOG_DIR}/${run}_rep${rep}.log"
    eval_result_dir="${RUN_ROOT}/${run}/eval/${log_name}"
    rm -rf "${eval_result_dir}"
    echo "EVAL ${run} repeat=${rep} START $(date '+%F %T')"
    if xvfb-run -a -e /dev/null \
      -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
      "${CONDA_PREFIX_PATH}/bin/python" -m rvt.eval \
      --model-folder "${RUN_ROOT}/${run}" \
      --model-name model_0.pth \
      --tasks "${EVAL_TASKS[@]}" \
      --eval-datafolder /home/paichichi/data/AGNOSTOS/unseen_tasks/test \
      --start-episode "${start_episode}" \
      --eval-episodes "${EPISODES_PER_REPEAT}" \
      --episode-length 25 \
      --headless \
      --device "${DEVICE}" \
      --log-name "${log_name}" \
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

by_task = defaultdict(list)
by_run = defaultdict(list)
by_level = defaultdict(list)
for r in ok:
    sr = float(r["success_rate"])
    ln = float(r["length"])
    by_task[(r["run"], r["task"])].append((sr, ln))
    by_run[r["run"]].append((sr, ln))
    level = "level1" if r["task"] in level1 else "level2" if r["task"] in level2 else "unknown"
    by_level[(r["run"], level)].append((sr, ln))

with summary.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "task", "repeats", "mean_success_rate", "std_success_rate", "mean_length", "std_length"])
    for key in sorted(by_task):
        vals = by_task[key]
        srs = [v[0] for v in vals]
        lens = [v[1] for v in vals]
        w.writerow([key[0], key[1], len(vals), f"{mean(srs):.2f}", f"{std(srs):.2f}", f"{mean(lens):.2f}", f"{std(lens):.2f}"])

with overall.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "n", "overall_mean_success_rate", "overall_mean_length"])
    for run in sorted(by_run):
        vals = by_run[run]
        w.writerow([run, len(vals), f"{mean([v[0] for v in vals]):.2f}", f"{mean([v[1] for v in vals]):.2f}"])

with levels.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "level", "n", "mean_success_rate", "mean_length"])
    for key in sorted(by_level):
        vals = by_level[key]
        w.writerow([key[0], key[1], len(vals), f"{mean([v[0] for v in vals]):.2f}", f"{mean([v[1] for v in vals]):.2f}"])

with table.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "overall", "level1", "level2"])
    for run in sorted(by_run):
        vals = by_run[run]
        l1 = by_level[(run, "level1")]
        l2 = by_level[(run, "level2")]
        w.writerow([
            run,
            f"{mean([v[0] for v in vals]):.2f}",
            f"{mean([v[0] for v in l1]):.2f}",
            f"{mean([v[0] for v in l2]):.2f}",
        ])

for path in (summary, overall, levels, table):
    print(path)
print(table.read_text())
PY

echo "FINISH $(date '+%F %T')"
