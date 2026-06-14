#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <gpus> <experiment_name> <config_yaml> [extra pretrain.py flags...]"
  echo "Example single GPU: $0 0 ln_debug configs/rh20t/pretraining/paired_tcc_vit_d4r_in_layernorm.yaml --max_iters=100"
  echo "Example multi GPU : $0 0,1,2,3 ln_ddp configs/rh20t/pretraining/paired_tcc_vit_d4r_in_layernorm.yaml --max_iters=10000"
  exit 1
fi

GPUS="${1// /}"
EXPERIMENT_NAME="$2"
CONFIG_YAML="$3"
shift 3

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS="${#GPU_ARRAY[@]}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="${PYTHONPATH:-}:/home/paichichi/projects/TCC-core"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

PYTHON_BIN="${PYTHON_BIN:-/home/paichichi/miniconda3/envs/tcc-core/bin/python}"

if [[ "$NUM_GPUS" -eq 1 ]]; then
  exec "$PYTHON_BIN" pretrain.py \
    --gpus="$GPUS" \
    --experiment_name="$EXPERIMENT_NAME" \
    --config_yaml="$CONFIG_YAML" \
    "$@"
fi

exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  pretrain.py \
  --gpus="$GPUS" \
  --experiment_name="$EXPERIMENT_NAME" \
  --config_yaml="$CONFIG_YAML" \
  "$@"
