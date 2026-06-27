#!/usr/bin/env bash
set -euo pipefail

RVT_ROOT=${RVT_ROOT:-/home/paichichi/projects/rvt-3d-policy-head-adaption}
TCC_ROOT=${TCC_ROOT:-/home/paichichi/projects/TCC-core}
CONDA_ENV=${CONDA_ENV:-hralign_py39}
CONDA_EXE=${CONDA_EXE:-/home/paichichi/miniconda3/bin/conda}
DEVICE=${DEVICE:-0}
MVT_CFG=${MVT_CFG:-${RVT_ROOT}/rvt/mvt/configs/rvt2.yaml}
TRAIN_ITERATIONS=${TRAIN_ITERATIONS:-25000}

STAMP=${STAMP:-$(date +%H%M%S)}
BASE="${TCC_ROOT}/analysis/rvt2_lite"
TRAIN_LOG_DIR="${BASE}/seen18_unseen6_train_logs_${STAMP}"
EVAL_LOG_DIR="${BASE}/seen18_unseen6_eval_logs_${STAMP}"
RAW="${BASE}/seen18_unseen6_${STAMP}.csv"
SUMMARY="${BASE}/seen18_unseen6_summary_${STAMP}.csv"
OVERALL="${BASE}/seen18_unseen6_overall_${STAMP}.csv"

RUNS=(d4r hrp ours_6ts4v ours_8ts3v)
CONFIGS=(d4r hrp ours_6ts4v ours_8ts3v)
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
TEST_TASKS=(
  close_fridge
  close_laptop_lid
  open_grill
  take_usb_out_of_computer
  toilet_seat_down
  turn_oven_on
)
TRAIN_TASKS_CSV=$(IFS=,; echo "${TRAIN_TASKS[*]}")
EPISODES=0,1,2,3,4,5,6,7,8,9

mkdir -p "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RVT_ROOT}:${RVT_ROOT}/rvt:${PYTHONPATH:-}"

echo "STAMP=${STAMP}"
echo "TRAIN_ITERATIONS=${TRAIN_ITERATIONS}"
echo "TRAIN_TASKS=${TRAIN_TASKS_CSV}"
echo "TEST_TASKS=${TEST_TASKS[*]}"
echo "TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
echo "EVAL_LOG_DIR=${EVAL_LOG_DIR}"
echo "RAW=${RAW}"
echo "SUMMARY=${SUMMARY}"
echo "OVERALL=${OVERALL}"
echo "START_TRAIN $(date '+%F %T')"

cd "${RVT_ROOT}"

for i in "${!RUNS[@]}"; do
  run="${RUNS[$i]}"
  cfg="${CONFIGS[$i]}"
  out_dir="${BASE}/seen18_unseen6_runs/${run}"
  if [ -s "${out_dir}/model_0.pth" ]; then
    echo "TRAIN ${run} SKIP existing ${out_dir}/model_0.pth $(date '+%F %T')"
    continue
  fi
  echo "TRAIN ${run} START $(date '+%F %T')"
  "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.train \
    --exp_cfg_path "${TCC_ROOT}/downstream/rvt2_lite/configs/${cfg}.yaml" \
    --mvt_cfg_path "${MVT_CFG}" \
    --device "${DEVICE}" \
    --exp_cfg_opts "tasks ${TRAIN_TASKS_CSV} train_iterations ${TRAIN_ITERATIONS} overwriter_log_dir ${out_dir}" \
    > "${TRAIN_LOG_DIR}/${run}.log" 2>&1
  echo "TRAIN ${run} DONE $(date '+%F %T')"
  grep -E "^\{|'total_loss'|'trans_loss'|'rot_loss_z'|\\[Finish\\]" "${TRAIN_LOG_DIR}/${run}.log" | tail -20 || true
done

for run in "${RUNS[@]}"; do
  ckpt="${BASE}/seen18_unseen6_runs/${run}/model_0.pth"
  test -s "${ckpt}"
  ls -lh "${ckpt}"
done

echo "START_EVAL $(date '+%F %T')"
echo 'run,repeat,task,success_rate,length,total_transitions,status' > "${RAW}"

for rep in 1 2 3; do
  for run in "${RUNS[@]}"; do
    log_name="seen18_unseen6_${STAMP}_rep${rep}"
    outlog="${EVAL_LOG_DIR}/${run}_rep${rep}.log"
    echo "EVAL ${run} repeat=${rep} START $(date '+%F %T')"
    if "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.eval \
      --model-folder "${BASE}/seen18_unseen6_runs/${run}" \
      --model-name model_0.pth \
      --tasks "${TEST_TASKS[@]}" \
      --eval-datafolder /home/paichichi/data/AGNOSTOS/unseen_tasks/test \
      --eval-episode-list "${EPISODES}" \
      --episode-length 25 \
      --headless \
      --device "${DEVICE}" \
      --log-name "${log_name}" \
      --no-tensorboard \
      --no-env-shutdown \
      > "${outlog}" 2>&1; then
      csv="${BASE}/seen18_unseen6_runs/${run}/eval/${log_name}/model_0/eval_results.csv"
      awk -F, -v run="${run}" -v rep="${rep}" 'NR>1 {gsub(/\r/,"",$4); print run","rep","$1","$2","$3","$4",ok"}' "${csv}" >> "${RAW}"
      echo "EVAL ${run} repeat=${rep} DONE $(date '+%F %T')"
    else
      echo "${run},${rep},ALL,,,,error" >> "${RAW}"
      echo "EVAL ${run} repeat=${rep} ERROR $(date '+%F %T')"
      tail -40 "${outlog}" || true
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

echo "FINISH $(date '+%F %T')"
cat "${SUMMARY}"
echo "OVERALL"
cat "${OVERALL}"
