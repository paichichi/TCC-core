#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/train_2a100_nesi.sh"

sbatch \
  --job-name=hralign_8ts3v \
  --export=ALL,RUN_NAME=train_2a100_d4r_8ts3v_lr7p5e5_20k,NUM_TIMESTAMPS=8,NUM_MULTI_VIEW=3 \
  "${JOB_SCRIPT}"

sbatch \
  --job-name=hralign_6ts4v \
  --export=ALL,RUN_NAME=train_2a100_d4r_6ts4v_lr7p5e5_20k,NUM_TIMESTAMPS=6,NUM_MULTI_VIEW=4 \
  "${JOB_SCRIPT}"

sbatch \
  --job-name=hralign_6ts3v \
  --export=ALL,RUN_NAME=train_2a100_d4r_6ts3v_lr7p5e5_20k,NUM_TIMESTAMPS=6,NUM_MULTI_VIEW=3 \
  "${JOB_SCRIPT}"
