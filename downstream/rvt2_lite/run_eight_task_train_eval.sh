#!/usr/bin/env bash
set -euo pipefail

RVT_ROOT=${RVT_ROOT:-/home/paichichi/projects/rvt-3d-policy-head-adaption}
TCC_ROOT=${TCC_ROOT:-/home/paichichi/projects/TCC-core}
CONDA_ENV=${CONDA_ENV:-hralign_py39}
CONDA_EXE=${CONDA_EXE:-/home/paichichi/miniconda3/bin/conda}
DEVICE=${DEVICE:-0}
MVT_CFG=${MVT_CFG:-${RVT_ROOT}/rvt/mvt/configs/rvt2.yaml}

STAMP=${STAMP:-$(date +%H%M%S)}
BASE="${TCC_ROOT}/analysis/rvt2_lite"
TRAIN_LOG_DIR="${BASE}/eight_train_logs_${STAMP}"
EVAL_LOG_DIR="${BASE}/eight_rvt2_style_3x10_logs_${STAMP}"
RAW="${BASE}/eight_rvt2_style_3x10_${STAMP}.csv"
SUMMARY="${BASE}/eight_rvt2_style_3x10_summary_${STAMP}.csv"

CONFIGS=(eight_d4r eight_hrp eight_ours_6ts4v eight_ours_8ts3v)
RUNS=(d4r hrp ours_6ts4v ours_8ts3v)
TASKS=(push_buttons slide_block_to_color_target sweep_to_dustpan_of_size meat_off_grill turn_tap place_shape_in_shape_sorter close_jar insert_onto_square_peg)
EPISODES=0,1,2,3,4,5,6,7,8,9

mkdir -p "${TRAIN_LOG_DIR}" "${EVAL_LOG_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RVT_ROOT}:${RVT_ROOT}/rvt:${PYTHONPATH:-}"

echo "STAMP=${STAMP}"
echo "TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
echo "EVAL_LOG_DIR=${EVAL_LOG_DIR}"
echo "RAW=${RAW}"
echo "SUMMARY=${SUMMARY}"
echo "START_TRAIN $(date '+%F %T')"

cd "${RVT_ROOT}"

for cfg in "${CONFIGS[@]}"; do
  echo "TRAIN ${cfg} START $(date '+%F %T')"
  "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.train \
    --exp_cfg_path "${TCC_ROOT}/downstream/rvt2_lite/configs/${cfg}.yaml" \
    --mvt_cfg_path "${MVT_CFG}" \
    --device "${DEVICE}" \
    > "${TRAIN_LOG_DIR}/${cfg}.log" 2>&1
  echo "TRAIN ${cfg} DONE $(date '+%F %T')"
  grep -E "^\{|'total_loss'|'trans_loss'|'rot_loss_z'|\\[Finish\\]" "${TRAIN_LOG_DIR}/${cfg}.log" | tail -20 || true
done

for run in "${RUNS[@]}"; do
  ckpt="${BASE}/eight_runs/${run}/model_0.pth"
  test -s "${ckpt}"
  ls -lh "${ckpt}"
done

echo "START_EVAL $(date '+%F %T')"
echo 'run,repeat,task,success_rate,length,total_transitions,status' > "${RAW}"

for rep in 1 2 3; do
  for run in "${RUNS[@]}"; do
    log_name="eight_rvt2_style_3x10_${STAMP}_rep${rep}"
    outlog="${EVAL_LOG_DIR}/${run}_rep${rep}.log"
    echo "EVAL ${run} repeat=${rep} START $(date '+%F %T')"
    if "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m rvt.eval \
      --model-folder "${BASE}/eight_runs/${run}" \
      --model-name model_0.pth \
      --tasks "${TASKS[@]}" \
      --eval-datafolder /home/paichichi/data/AGNOSTOS/seen_tasks/train \
      --eval-episode-list "${EPISODES}" \
      --episode-length 25 \
      --headless \
      --device "${DEVICE}" \
      --log-name "${log_name}" \
      --no-tensorboard \
      --no-env-shutdown \
      > "${outlog}" 2>&1; then
      csv="${BASE}/eight_runs/${run}/eval/${log_name}/model_0/eval_results.csv"
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

echo "FINISH $(date '+%F %T')"
cat "${SUMMARY}"
