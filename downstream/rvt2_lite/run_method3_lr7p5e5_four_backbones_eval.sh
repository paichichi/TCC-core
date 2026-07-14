#!/usr/bin/env bash
set -euo pipefail

RVT_ROOT=${RVT_ROOT:-/home/paichichi/projects/rvt-3d-policy-head-adaption}
TCC_ROOT=${TCC_ROOT:-/home/paichichi/projects/TCC-core}
TCC_PYTHON=${TCC_PYTHON:-/home/paichichi/miniconda3/envs/tcc-core/bin/python}
CONDA_ENV=${CONDA_ENV:-hralign_py39}
CONDA_EXE=${CONDA_EXE:-/home/paichichi/miniconda3/bin/conda}
DEVICE=${DEVICE:-0}
EPISODES=${EPISODES:-0,1,2,3,4,5,6,7,8,9}
REPEATS=${REPEATS:-3}

STAMP=${STAMP:-method3_lr7p5e5_eval_$(date +%Y%m%d_%H%M%S)}
BASE="${TCC_ROOT}/wsl_result/downstream_rvt2_lite_runs"
RUN_ROOT="${BASE}/method3_lr7p5e5_eval_${STAMP}"
EVAL_LOG_DIR="${BASE}/method3_lr7p5e5_eval_logs_${STAMP}"
RAW="${BASE}/method3_lr7p5e5_eval_${STAMP}.csv"
SUMMARY="${BASE}/method3_lr7p5e5_eval_summary_${STAMP}.csv"
OVERALL="${BASE}/method3_lr7p5e5_eval_overall_${STAMP}.csv"
LEVELS="${BASE}/method3_lr7p5e5_eval_level_summary_${STAMP}.csv"
TABLE="${BASE}/method3_lr7p5e5_eval_table_${STAMP}.csv"

RUNS=(
  "method3_vit_ln_8ts4v_lr7p5e5:${BASE}/wsl_method3_default_method3_default_20260705_104441/wsl_method3_vit_ln_8ts4v"
  "method3_vit_ln_8ts3v_lr7p5e5:${BASE}/wsl_method3_default_method3_default_20260705_104441/wsl_method3_vit_ln_8ts3v"
  "method3_resnet_ln_8ts4v_lr7p5e5:${BASE}/resnet_lite_resnet_lite_20260706_042246/method3_wsl_resnet_ln_8ts4v"
  "method3_resnet_ln_8ts3v_lr7p5e5:${BASE}/resnet_lite_resnet_lite_20260706_042246/method3_wsl_resnet_ln_8ts3v"
)

LEVEL1_TASKS=(
  close_fridge close_microwave close_laptop_lid toilet_seat_down open_grill
  phone_on_base
)
LEVEL2_TASKS=(
  take_usb_out_of_computer take_lid_off_saucepan turn_oven_on beat_the_buzz
  water_plants unplug_charger
)
EVAL_TASKS=("${LEVEL1_TASKS[@]}" "${LEVEL2_TASKS[@]}")

mkdir -p "${RUN_ROOT}" "${EVAL_LOG_DIR}"

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
echo "RUN_ROOT=${RUN_ROOT}"
echo "RAW=${RAW}"
echo "TABLE=${TABLE}"

for item in "${RUNS[@]}"; do
  IFS=: read -r run model_dir <<< "${item}"
  test -s "${model_dir}/model_0.pth"
  echo "FOUND ${run}: ${model_dir}/model_0.pth"
done

cd "${RVT_ROOT}"
echo 'run,repeat,task,success_rate,length,total_transitions,status' > "${RAW}"
for rep in $(seq 1 "${REPEATS}"); do
  for item in "${RUNS[@]}"; do
    IFS=: read -r run model_dir <<< "${item}"
    log_name="method3_lr7p5e5_eval_${STAMP}_rep${rep}"
    outlog="${EVAL_LOG_DIR}/${run}_rep${rep}.log"
    echo "EVAL ${run} repeat=${rep} START $(date '+%F %T')"
    if "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.eval \
      --model-folder "${model_dir}" \
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
      csv="${model_dir}/eval/${log_name}/model_0/eval_results.csv"
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
    w.writerow(["run", "task", "repeats", "errors", "mean_success_rate", "std_success_rate", "mean_length", "std_length"])
    for key in sorted(by_task):
        vals = by_task[key]
        srs = [v[0] for v in vals]
        lens = [v[1] for v in vals]
        errors = sum(1 for r in rows if r["run"] == key[0] and r["task"] == key[1] and r["status"] != "ok")
        w.writerow([key[0], key[1], len(vals), errors, f"{mean(srs):.2f}", f"{std(srs):.2f}", f"{mean(lens):.2f}", f"{std(lens):.2f}"])

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
