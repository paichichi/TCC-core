#!/usr/bin/env bash
set -euo pipefail

RVT_ROOT=${RVT_ROOT:-/home/paichichi/projects/rvt-3d-policy-head-adaption}
TCC_ROOT=${TCC_ROOT:-/home/paichichi/projects/TCC-core}
TCC_PYTHON=${TCC_PYTHON:-/home/paichichi/miniconda3/envs/tcc-core/bin/python}
CONDA_ENV=${CONDA_ENV:-hralign_py39}
CONDA_EXE=${CONDA_EXE:-/home/paichichi/miniconda3/bin/conda}
DEVICE=${DEVICE:-0}
MVT_CFG=${MVT_CFG:-${RVT_ROOT}/rvt/mvt/configs/rvt2.yaml}
TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-10000}
EPISODES=${EPISODES:-0,1,2,3,4,5,6,7,8,9}
REPEATS=${REPEATS:-3}

STAMP=${STAMP:-resnet_lite_$(date +%Y%m%d_%H%M%S)}
BASE="${TCC_ROOT}/wsl_result/downstream_rvt2_lite_runs"
TCC_LOG_DIR="${TCC_ROOT}/wsl_result/tcc_core_logs"
PRETRAIN_DIR="${TCC_ROOT}/wsl_result/downstream_pretrains"
RUN_ROOT="${BASE}/resnet_lite_${STAMP}"
TRAIN_LOG_DIR="${BASE}/resnet_lite_train_logs_${STAMP}"
EVAL_LOG_DIR="${BASE}/resnet_lite_eval_logs_${STAMP}"
RAW="${BASE}/resnet_lite_${STAMP}.csv"
SUMMARY="${BASE}/resnet_lite_summary_${STAMP}.csv"
OVERALL="${BASE}/resnet_lite_overall_${STAMP}.csv"
LEVELS="${BASE}/resnet_lite_level_summary_${STAMP}.csv"

TCC_RUNS=(
  "method2_wsl_resnet_ln_8ts3v:${TCC_ROOT}/configs/wsl_lite_method2_resnet_ln_8ts3v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method2_resnet_ln_8ts3v_b4_i3000/checkpoint_003000.pt:${PRETRAIN_DIR}/wsl_lite_method2_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method3_wsl_resnet_ln_8ts4v:${TCC_ROOT}/configs/wsl_lite_method3_resnet_ln_8ts4v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method3_resnet_ln_8ts4v_b4_i3000/checkpoint_003000.pt:${PRETRAIN_DIR}/wsl_lite_method3_resnet_ln_8ts4v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method3_wsl_resnet_ln_8ts3v:${TCC_ROOT}/configs/wsl_lite_method3_resnet_ln_8ts3v.yaml:${TCC_ROOT}/wsl_result/tcc_core_runs/wsl_lite_method3_resnet_ln_8ts3v_b4_i3000/checkpoint_003000.pt:${PRETRAIN_DIR}/wsl_lite_method3_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt"
)

RVT_RUNS=(
  "method2_wsl_resnet_ln_8ts3v:${TCC_ROOT}/downstream/rvt2_lite/configs/method2_wsl_resnet_ln_8ts3v.yaml:${PRETRAIN_DIR}/wsl_lite_method2_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method3_wsl_resnet_ln_8ts4v:${TCC_ROOT}/downstream/rvt2_lite/configs/method3_wsl_resnet_ln_8ts4v.yaml:${PRETRAIN_DIR}/wsl_lite_method3_resnet_ln_8ts4v_b4_i3000_rvt2_resnet_pretrain.pt"
  "method3_wsl_resnet_ln_8ts3v:${TCC_ROOT}/downstream/rvt2_lite/configs/method3_wsl_resnet_ln_8ts3v.yaml:${PRETRAIN_DIR}/wsl_lite_method3_resnet_ln_8ts3v_b4_i3000_rvt2_resnet_pretrain.pt"
  "unadapted_r3m:${TCC_ROOT}/downstream/rvt2_lite/configs/unadapted_r3m.yaml:/home/paichichi/data/pretrain/UnadaptedR3M.pt"
  "adapted_r3m:${TCC_ROOT}/downstream/rvt2_lite/configs/adapted_r3m.yaml:/home/paichichi/data/pretrain/AdaptedR3M.pyth"
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
echo "TRAIN_ITERATIONS=${TRAIN_ITERATIONS}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TCC_LOG_DIR=${TCC_LOG_DIR}"
echo "TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
echo "EVAL_LOG_DIR=${EVAL_LOG_DIR}"
echo "RAW=${RAW}"
echo "SUMMARY=${SUMMARY}"
echo "OVERALL=${OVERALL}"
echo "LEVELS=${LEVELS}"

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
  if [ -s "${converted}" ]; then
    echo "CONVERT ${run} SKIP existing ${converted}"
  else
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
  for item in "${RVT_RUNS[@]}"; do
    IFS=: read -r run _cfg _pretrain <<< "${item}"
    log_name="resnet_lite_${STAMP}_rep${rep}"
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

{
  echo 'run,task,repeats,errors,mean_success_rate,std_success_rate,mean_length,std_length'
  awk -F, 'NR>1 {
    key=$1","$3
    if ($7=="ok") {
      n[key]++; sum[key]+=$4; sumsq[key]+=$4*$4; len[key]+=$5; lensq[key]+=$5*$5
    } else { err[key]++ }
  }
  END {
    for (k in n) {
      mean=sum[k]/n[k]; lmean=len[k]/n[k]
      var=(sumsq[k]/n[k])-(mean*mean); if (var < 0) var=0
      lvar=(lensq[k]/n[k])-(lmean*lmean); if (lvar < 0) lvar=0
      printf "%s,%d,%d,%.2f,%.2f,%.2f,%.2f\n", k, n[k], err[k]+0, mean, sqrt(var), lmean, sqrt(lvar)
    }
    for (k in err) if (!(k in n)) printf "%s,0,%d,nan,nan,nan,nan\n", k, err[k]
  }' "${RAW}" | sort
} > "${SUMMARY}"

{
  echo 'run,n,overall_mean_success_rate,overall_mean_length'
  awk -F, 'NR>1 && $7=="ok" {n[$1]++; sum[$1]+=$4; len[$1]+=$5} END {for (r in n) printf "%s,%d,%.2f,%.2f\n", r, n[r], sum[r]/n[r], len[r]/n[r]}' "${RAW}" | sort
} > "${OVERALL}"

{
  echo 'run,level,n,mean_success_rate,mean_length'
  awk -F, '
    BEGIN {
      split("close_fridge close_microwave close_laptop_lid toilet_seat_down open_grill phone_on_base", l1, " ")
      for (i in l1) level[l1[i]]="level1"
      split("take_usb_out_of_computer take_lid_off_saucepan turn_oven_on beat_the_buzz water_plants unplug_charger", l2, " ")
      for (i in l2) level[l2[i]]="level2"
    }
    NR>1 && $7=="ok" {
      key=$1","level[$3]
      n[key]++; sum[key]+=$4; len[key]+=$5
    }
    END {
      for (k in n) printf "%s,%d,%.2f,%.2f\n", k, n[k], sum[k]/n[k], len[k]/n[k]
    }' "${RAW}" | sort
} > "${LEVELS}"

echo "FINISH $(date '+%F %T')"
cat "${SUMMARY}"
echo "OVERALL"
cat "${OVERALL}"
echo "LEVELS"
cat "${LEVELS}"
