#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 <vit|r3m_bn> <run_name> <checkpoint_040000.pt>" >&2
    exit 2
fi

BACKBONE="$1"
RUN_NAME="$2"
CHECKPOINT="$3"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Unsafe run name: $RUN_NAME" >&2
    exit 2
fi
if [[ "$BACKBONE" != "vit" && "$BACKBONE" != "r3m_bn" ]]; then
    echo "Backbone must be vit or r3m_bn" >&2
    exit 2
fi
if [[ "$BACKBONE" == "vit" ]]; then
    RESULT_BACKBONE="vit_imagenet"
else
    RESULT_BACKBONE="r3m_bn_bi"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
if [[ "$GPU_NAME" != *"RTX 5090"* ]]; then
    echo "RVT2-lite is restricted to RTX 5090; found: $GPU_NAME" >&2
    exit 3
fi

RVT_ROOT="/home/paichichi/projects/rvt-3d-policy-head-adaption"
TCC_ROOT="/home/paichichi/projects/TCC-core"
SEARCH_DIR="$TCC_ROOT/work/fixed_tq_hparam_search"
CONDA_ENV="tcc-core-parity"
CONDA_EXE="/home/paichichi/miniconda3/bin/conda"
CONDA_PREFIX_PATH="/home/paichichi/miniconda3/envs/tcc-core-parity"
TCC_PYTHON="$CONDA_PREFIX_PATH/bin/python"
MVT_CFG="$RVT_ROOT/rvt/mvt/configs/rvt2.yaml"
DEVICE=0
TRAIN_ITERATIONS=25000
EPISODES_PER_REPEAT=10
REPEATS=3

BASE="$TCC_ROOT/wsl_result/downstream_rvt2_lite_runs/hparam_search"
RUN_ROOT="$BASE/$RUN_NAME"
POLICY_DIR="$RUN_ROOT/policy"
LOG_DIR="$RUN_ROOT/logs"
RAW="$RUN_ROOT/eval_raw.csv"
TASK_SUMMARY="$RUN_ROOT/eval_task_summary.csv"
CENTRAL="$SEARCH_DIR/rvt2_lite_results.csv"

TRAIN_TASKS=(
  close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer
  place_cups place_shape_in_shape_sorter place_wine_at_rack_location push_buttons
  put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag
  slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size
  turn_tap
)
LEVEL1_TASKS=(
  close_fridge close_microwave close_laptop_lid toilet_seat_down open_grill
  phone_on_base
)
LEVEL2_TASKS=(
  take_usb_out_of_computer take_lid_off_saucepan turn_oven_on beat_the_buzz
  water_plants unplug_charger
)
TRAIN_TASKS_CSV="$(IFS=,; echo "${TRAIN_TASKS[*]}")"
EVAL_TASKS=("${LEVEL1_TASKS[@]}" "${LEVEL2_TASKS[@]}")

mkdir -p "$RUN_ROOT" "$LOG_DIR"
test -s "$CHECKPOINT"
test -s "$MVT_CFG"

if [[ "$BACKBONE" == "vit" ]]; then
    BASE_CFG="$TCC_ROOT/downstream/rvt2_lite/configs/wsl_tcc_vit_ln_8ts4v.yaml"
    PRETRAIN="$CHECKPOINT"
else
    BASE_CFG="$TCC_ROOT/downstream/rvt2_lite/configs/unadapted_r3m.yaml"
    PRETRAIN="$RUN_ROOT/${RUN_NAME}_rvt2_resnet_pretrain.pt"
    REFERENCE="/home/paichichi/data/pretrain/UnadaptedR3M.pt"
    if [[ ! -s "$PRETRAIN" ]]; then
        "$TCC_PYTHON" "$SEARCH_DIR/convert_bn_affine_to_rvt.py" convert \
            "$CHECKPOINT" "$PRETRAIN" --reference "$REFERENCE"
    fi
    "$TCC_PYTHON" "$SEARCH_DIR/convert_bn_affine_to_rvt.py" validate \
        "$PRETRAIN" --source "$CHECKPOINT" --reference "$REFERENCE"
fi
test -s "$BASE_CFG"
test -s "$PRETRAIN"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RVT_ROOT}:${RVT_ROOT}/rvt:${RVT_ROOT}/rvt/libs/YARR:${RVT_ROOT}/rvt/libs/RLBench:${RVT_ROOT}/rvt/libs/PyRep:${RVT_ROOT}/rvt/libs/peract:${RVT_ROOT}/rvt/libs/peract_colab:${RVT_ROOT}/rvt/libs/point-renderer:${PYTHONPATH:-}"
export COPPELIASIM_ROOT="/home/paichichi/software/CoppeliaSim_4_1_0"
export LD_LIBRARY_PATH="${CONDA_PREFIX_PATH}/lib:${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${COPPELIASIM_ROOT}/libcrypto.so.1.1:${COPPELIASIM_ROOT}/libssl.so.1.1"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM

if [[ ! -s "$POLICY_DIR/model_0.pth" ]]; then
    cd "$RVT_ROOT"
    "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" python -m rvt.train \
        --exp_cfg_path "$BASE_CFG" \
        --mvt_cfg_path "$MVT_CFG" \
        --device "$DEVICE" \
        --exp_cfg_opts \
          "tasks ${TRAIN_TASKS_CSV} train_iterations ${TRAIN_ITERATIONS} pretrain ${PRETRAIN} overwriter_log_dir ${POLICY_DIR}" \
        > "$LOG_DIR/policy_train.log" 2>&1
fi
test -s "$POLICY_DIR/model_0.pth"

echo 'run,repeat,task,success_rate,length,total_transitions,status' > "$RAW"
cd "$RVT_ROOT"
for repeat in $(seq 1 "$REPEATS"); do
    log_name="hps_lite_${RUN_NAME}_rep${repeat}"
    eval_result_dir="$POLICY_DIR/eval/$log_name"
    rm -rf -- "$eval_result_dir"
    if xvfb-run -a -e /dev/null \
        -s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
        "$CONDA_PREFIX_PATH/bin/python" -m rvt.eval \
          --model-folder "$POLICY_DIR" \
          --model-name model_0.pth \
          --tasks "${EVAL_TASKS[@]}" \
          --eval-datafolder /home/paichichi/data/AGNOSTOS/unseen_tasks/test \
          --start-episode 0 \
          --eval-episodes "$EPISODES_PER_REPEAT" \
          --episode-length 25 \
          --headless \
          --device "$DEVICE" \
          --log-name "$log_name" \
          > "$LOG_DIR/eval_rep${repeat}.log" 2>&1; then
        result_csv="$eval_result_dir/model_0/eval_results.csv"
        test -s "$result_csv"
        awk -F, -v run="$RUN_NAME" -v rep="$repeat" \
          'NR>1 {gsub(/\r/,"",$4); print run","rep","$1","$2","$3","$4",ok"}' \
          "$result_csv" >> "$RAW"
    else
        echo "$RUN_NAME,$repeat,ALL,,,,error" >> "$RAW"
        tail -80 "$LOG_DIR/eval_rep${repeat}.log" || true
        exit 1
    fi
done

"$TCC_PYTHON" "$SEARCH_DIR/summarize_rvt2_lite.py" \
    --raw "$RAW" \
    --run-name "$RUN_NAME" \
    --backbone "$RESULT_BACKBONE" \
    --central "$CENTRAL" \
    --task-summary "$TASK_SUMMARY" \
    --episodes-per-task "$EPISODES_PER_REPEAT"
touch "$RUN_ROOT/LITE_COMPLETE"
