#!/usr/bin/env bash
set -euo pipefail

RVT_ROOT=${RVT_ROOT:-/home/paichichi/projects/rvt-3d-policy-head-adaption}
TCC_ROOT=${TCC_ROOT:-/home/paichichi/projects/TCC-core}
TCC_PYTHON=${TCC_PYTHON:-/home/paichichi/miniconda3/envs/tcc-core/bin/python}
CONDA_ENV=${CONDA_ENV:-hralign_py39}
CONDA_EXE=${CONDA_EXE:-/home/paichichi/miniconda3/bin/conda}
DEVICE=${DEVICE:-0}
MVT_CFG=${MVT_CFG:-${RVT_ROOT}/rvt/mvt/configs/rvt2.yaml}
VIT_TRAIN_ITERATIONS=${VIT_TRAIN_ITERATIONS:-50000}
RESNET_TRAIN_ITERATIONS=${RESNET_TRAIN_ITERATIONS:-10000}
EPISODES=${EPISODES:-0,1,2,3,4,5,6,7,8,9}
REPEATS=${REPEATS:-3}

STAMP=${STAMP:-method1_lite_$(date +%Y%m%d_%H%M%S)}
BASE="${TCC_ROOT}/wsl_result/downstream_rvt2_lite_runs"
TCC_LOG_DIR="${TCC_ROOT}/wsl_result/tcc_core_logs"
PRETRAIN_DIR="${TCC_ROOT}/wsl_result/downstream_pretrains"
RUN_ROOT="${BASE}/method1_lite_${STAMP}"
TRAIN_LOG_DIR="${BASE}/method1_lite_train_logs_${STAMP}"
EVAL_LOG_DIR="${BASE}/method1_lite_eval_logs_${STAMP}"
RAW="${BASE}/method1_lite_${STAMP}.csv"
SUMMARY="${BASE}/method1_lite_summary_${STAMP}.csv"
OVERALL="${BASE}/method1_lite_overall_${STAMP}.csv"
LEVELS="${BASE}/method1_lite_level_summary_${STAMP}.csv"
TABLE="${BASE}/method1_lite_table_${STAMP}.csv"
VIT_TABLE="${BASE}/vit_lite_table_${STAMP}.csv"
RESNET_TABLE="${BASE}/resnet_lite_table_with_method1_${STAMP}.csv"

TCC_RUNS=(
  "method1_wsl_vit_ln_8ts3v:${TCC_ROOT}/configs/wsl_lite_method1_vit_ln_8ts3v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_vit_ln_8ts3v_b4_i3000/checkpoint_003000.pt:"
  "method1_wsl_vit_ln_6ts4v:${TCC_ROOT}/configs/wsl_lite_method1_vit_ln_6ts4v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_vit_ln_6ts4v_b4_i3000/checkpoint_003000.pt:"
  "method1_wsl_vit_ln_8ts4v:${TCC_ROOT}/configs/wsl_lite_method1_vit_ln_8ts4v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_vit_ln_8ts4v_b4_i3000/checkpoint_003000.pt:"
  "method1_wsl_resnet_ln_8ts3v:${TCC_ROOT}/configs/wsl_lite_method1_resnet_ln_8ts3v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_resnet_ln_8ts3v_b4_i3000/checkpoint_003000.pt:${PRETRAIN_DIR}/wsl_lite_method1_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method1_wsl_resnet_ln_8ts4v:${TCC_ROOT}/configs/wsl_lite_method1_resnet_ln_8ts4v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_resnet_ln_8ts4v_b4_i3000/checkpoint_003000.pt:${PRETRAIN_DIR}/wsl_lite_method1_resnet_ln_8ts4v_b4_i3000_rvt2_resnet_pretrain.pt"
)

RVT_RUNS=(
  "method1_wsl_vit_ln_8ts3v:${TCC_ROOT}/downstream/rvt2_lite/configs/method1_wsl_vit_ln_8ts3v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_vit_ln_8ts3v_b4_i3000/checkpoint_003000.pt:${VIT_TRAIN_ITERATIONS}"
  "method1_wsl_vit_ln_6ts4v:${TCC_ROOT}/downstream/rvt2_lite/configs/method1_wsl_vit_ln_6ts4v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_vit_ln_6ts4v_b4_i3000/checkpoint_003000.pt:${VIT_TRAIN_ITERATIONS}"
  "method1_wsl_vit_ln_8ts4v:${TCC_ROOT}/downstream/rvt2_lite/configs/method1_wsl_vit_ln_8ts4v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method1_vit_ln_8ts4v_b4_i3000/checkpoint_003000.pt:${VIT_TRAIN_ITERATIONS}"
  "method1_wsl_resnet_ln_8ts3v:${TCC_ROOT}/downstream/rvt2_lite/configs/method1_wsl_resnet_ln_8ts3v.yaml:${PRETRAIN_DIR}/wsl_lite_method1_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt:${RESNET_TRAIN_ITERATIONS}"
  "method1_wsl_resnet_ln_8ts4v:${TCC_ROOT}/downstream/rvt2_lite/configs/method1_wsl_resnet_ln_8ts4v.yaml:${PRETRAIN_DIR}/wsl_lite_method1_resnet_ln_8ts4v_b4_i3000_rvt2_resnet_pretrain.pt:${RESNET_TRAIN_ITERATIONS}"
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
echo "VIT_TRAIN_ITERATIONS=${VIT_TRAIN_ITERATIONS}"
echo "RESNET_TRAIN_ITERATIONS=${RESNET_TRAIN_ITERATIONS}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "RAW=${RAW}"
echo "TABLE=${TABLE}"
echo "VIT_TABLE=${VIT_TABLE}"
echo "RESNET_TABLE=${RESNET_TABLE}"

convert_tcc_resnet_checkpoint() {
  local source_ckpt="$1"
  local target_ckpt="$2"
  "${TCC_PYTHON}" - "$source_ckpt" "$target_ckpt" <<'PY'
import sys
from pathlib import Path
import torch

source = Path(sys.argv[1])
target = Path(sys.argv[2])
checkpoint = torch.load(source, map_location="cpu")
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

cd "${TCC_ROOT}"
for item in "${TCC_RUNS[@]}"; do
  IFS=: read -r run cfg ckpt converted <<< "${item}"
  test -s "${cfg}"
  if [ -s "${ckpt}" ]; then
    echo "TCC ${run} SKIP existing ${ckpt} $(date '+%F %T')"
  else
    echo "TCC ${run} START $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES="${DEVICE}" "${TCC_PYTHON}" -u train.py \
      --exp_cfg_path "${cfg}" \
      --device 0 \
      > "${TCC_LOG_DIR}/${run}_${STAMP}.log" 2>&1
    echo "TCC ${run} DONE $(date '+%F %T')"
    tail -40 "${TCC_LOG_DIR}/${run}_${STAMP}.log"
  fi
  test -s "${ckpt}"
  if [ -n "${converted}" ]; then
    if [ -s "${converted}" ]; then
      echo "CONVERT ${run} SKIP existing ${converted}"
    else
      convert_tcc_resnet_checkpoint "${ckpt}" "${converted}"
    fi
    test -s "${converted}"
  fi
done

for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run cfg pretrain iterations <<< "${item}"
  test -s "${cfg}"
  test -s "${pretrain}"
  echo "CHECK ${run}"
  echo "  cfg=${cfg}"
  echo "  pretrain=${pretrain}"
  echo "  train_iterations=${iterations}"
done

cd "${RVT_ROOT}"
for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run cfg pretrain iterations <<< "${item}"
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
    --exp_cfg_opts "tasks ${TRAIN_TASKS_CSV} train_iterations ${iterations} pretrain ${pretrain} overwriter_log_dir ${out_dir}" \
    > "${TRAIN_LOG_DIR}/${run}.log" 2>&1
  echo "TRAIN ${run} DONE $(date '+%F %T')"
  grep -E "MVT_|ViTB16Backbone|MVT_Resnet|Manually loading|matched keys|Adapter layers|\\[Finish\\]|total_loss|trans_loss|nan|NaN" "${TRAIN_LOG_DIR}/${run}.log" | tail -60 || true
done

for item in "${RVT_RUNS[@]}"; do
  IFS=: read -r run _cfg _pretrain _iterations <<< "${item}"
  test -s "${RUN_ROOT}/${run}/model_0.pth"
  ls -lh "${RUN_ROOT}/${run}/model_0.pth"
done

echo 'run,repeat,task,success_rate,length,total_transitions,status' > "${RAW}"

for rep in $(seq 1 "${REPEATS}"); do
  for item in "${RVT_RUNS[@]}"; do
    IFS=: read -r run _cfg _pretrain _iterations <<< "${item}"
    log_name="method1_lite_${STAMP}_rep${rep}"
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

"${TCC_PYTHON}" - "${RAW}" "${SUMMARY}" "${OVERALL}" "${LEVELS}" "${TABLE}" "${VIT_TABLE}" "${RESNET_TABLE}" <<'PY'
import csv
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

raw, summary, overall, levels, table, vit_table, resnet_table = map(Path, sys.argv[1:])
rows = list(csv.DictReader(raw.open()))

latest = OrderedDict()
for row in rows:
    if row["status"] != "ok":
        continue
    latest[(row["run"], row["repeat"], row["task"])] = row
clean_rows = list(latest.values())

level1 = set("close_fridge close_microwave close_laptop_lid toilet_seat_down open_grill phone_on_base".split())
level2 = set("take_usb_out_of_computer take_lid_off_saucepan turn_oven_on beat_the_buzz water_plants unplug_charger".split())

by_task = defaultdict(list)
by_run = defaultdict(list)
for row in clean_rows:
    by_task[(row["run"], row["task"])].append(row)
    by_run[row["run"]].append(row)

with summary.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "task", "repeats", "errors", "mean_success_rate", "std_success_rate", "mean_length", "std_length"])
    for (run, task), items in sorted(by_task.items()):
        vals = [float(x["success_rate"]) for x in items]
        lens = [float(x["length"]) for x in items]
        writer.writerow([
            run,
            task,
            len(items),
            0,
            f"{statistics.mean(vals):.2f}",
            f"{statistics.pstdev(vals) if len(vals) > 1 else 0.0:.2f}",
            f"{statistics.mean(lens):.2f}",
            f"{statistics.pstdev(lens) if len(lens) > 1 else 0.0:.2f}",
        ])

overall_rows = {}
with overall.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "n", "overall_mean_success_rate", "overall_mean_length"])
    for run, items in sorted(by_run.items()):
        vals = [float(x["success_rate"]) for x in items]
        lens = [float(x["length"]) for x in items]
        overall_rows[run] = {
            "n": len(items),
            "overall": f"{statistics.mean(vals):.2f}",
            "length": f"{statistics.mean(lens):.2f}",
        }
        writer.writerow([run, len(items), overall_rows[run]["overall"], overall_rows[run]["length"]])

level_rows = {}
with levels.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "level", "n", "mean_success_rate", "mean_length"])
    for run, items in sorted(by_run.items()):
        for label, tasks in (("level1", level1), ("level2", level2)):
            sub = [x for x in items if x["task"] in tasks]
            vals = [float(x["success_rate"]) for x in sub]
            lens = [float(x["length"]) for x in sub]
            level_rows[(run, label)] = {
                "n": len(sub),
                "success": f"{statistics.mean(vals):.2f}",
                "length": f"{statistics.mean(lens):.2f}",
            }
            writer.writerow([run, label, len(sub), level_rows[(run, label)]["success"], level_rows[(run, label)]["length"]])

def write_table(path: Path, runs: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n", "overall", "level1", "level2", "source"])
        for run in runs:
            writer.writerow([
                run,
                overall_rows[run]["n"],
                overall_rows[run]["overall"],
                level_rows[(run, "level1")]["success"],
                level_rows[(run, "level2")]["success"],
                "method1 current clean run",
            ])

all_runs = [
    "method1_wsl_vit_ln_8ts3v",
    "method1_wsl_vit_ln_6ts4v",
    "method1_wsl_vit_ln_8ts4v",
    "method1_wsl_resnet_ln_8ts3v",
    "method1_wsl_resnet_ln_8ts4v",
]
write_table(table, all_runs)

vit_rows = [
    ["D4R ImageNet", 36, "30.83", "31.67", "30.00", "previous vit lite table"],
    ["HRP ImageNet", 36, "33.61", "38.89", "28.33", "previous vit lite table"],
    ["method2_wsl_vit_ln_8ts4v", 36, "33.34", "32.78", "33.89", "previous vit lite table"],
    ["method3_wsl_vit_ln_8ts4v", 36, "38.06", "40.56", "35.56", "previous vit lite table"],
    ["method3_wsl_vit_ln_8ts3v", 36, "30.28", "27.78", "32.78", "previous vit lite table"],
]
for run in ["method1_wsl_vit_ln_8ts3v", "method1_wsl_vit_ln_6ts4v", "method1_wsl_vit_ln_8ts4v"]:
    vit_rows.append([
        run,
        overall_rows[run]["n"],
        overall_rows[run]["overall"],
        level_rows[(run, "level1")]["success"],
        level_rows[(run, "level2")]["success"],
        "method1 current clean run",
    ])
with vit_table.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "n", "overall", "level1", "level2", "source"])
    writer.writerows(vit_rows)

resnet_rows = []
previous = Path("/home/paichichi/projects/TCC-core/wsl_result/downstream_rvt2_lite_runs/resnet_lite_table_resnet_lite_20260706_042246.csv")
if previous.exists():
    for row in csv.DictReader(previous.open()):
        resnet_rows.append([row["model"], row["n"], row["overall"], row["level1"], row["level2"], row["source"]])
for run in ["method1_wsl_resnet_ln_8ts3v", "method1_wsl_resnet_ln_8ts4v"]:
    resnet_rows.append([
        run,
        overall_rows[run]["n"],
        overall_rows[run]["overall"],
        level_rows[(run, "level1")]["success"],
        level_rows[(run, "level2")]["success"],
        "method1 current clean run",
    ])
with resnet_table.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "n", "overall", "level1", "level2", "source"])
    writer.writerows(resnet_rows)

print("SUMMARY", summary)
print("OVERALL", overall)
print("LEVELS", levels)
print("METHOD1_TABLE", table)
print("VIT_TABLE", vit_table)
print("RESNET_TABLE", resnet_table)
print(table.read_text())
PY

echo "FINISH $(date '+%F %T')"
cat "${TABLE}"
echo "VIT_TABLE"
cat "${VIT_TABLE}"
echo "RESNET_TABLE"
cat "${RESNET_TABLE}"
