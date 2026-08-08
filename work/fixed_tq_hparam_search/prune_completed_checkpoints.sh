#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/paichichi/projects/TCC-core"
SEARCH_DIR="$ROOT/work/fixed_tq_hparam_search"
OUTPUT_ROOT="$ROOT/wsl_result/tcc_core_runs"
MANIFEST="$SEARCH_DIR/upstream_5090_all.tsv"
PYTHON="/home/paichichi/miniconda3/envs/tcc-core-parity/bin/python"

while IFS=$'\t' read -r run_name _rest; do
    [[ "$run_name" == "run_name" || -z "$run_name" ]] && continue

    run_dir="$OUTPUT_ROOT/$run_name"
    final_checkpoint="$run_dir/checkpoint_040000.pt"
    marker="$run_dir/.intermediate_checkpoints_pruned"
    [[ -f "$final_checkpoint" && ! -f "$marker" ]] || continue

    # Do not remove recovery points until the final checkpoint can be fully read.
    "$PYTHON" - "$final_checkpoint" <<'PY'
import os
import sys
import torch

path = sys.argv[1]
if os.path.getsize(path) < 1_000_000:
    raise SystemExit(f"final checkpoint is unexpectedly small: {path}")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if not isinstance(checkpoint, dict):
    raise SystemExit(f"final checkpoint is not a dictionary: {path}")
PY

    removed=0
    for checkpoint in "$run_dir"/checkpoint_*.pt; do
        [[ -e "$checkpoint" ]] || continue
        [[ "$checkpoint" == "$final_checkpoint" ]] && continue
        rm -f -- "$checkpoint"
        removed=$((removed + 1))
    done

    {
        printf 'pruned_at=%s\n' "$(date -Iseconds)"
        printf 'final_checkpoint=%s\n' "$final_checkpoint"
        printf 'removed=%s\n' "$removed"
    } > "$marker"
done < "$MANIFEST"
