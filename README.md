# TCC-core

Minimal RH20T multi-view pretraining code.

The current training path is no longer vanilla TCC. It uses:

- same-side timestamp-aligned multi-view fusion
- H/R matched camera combos
- contrastive Soft-DTW over human/robot fused sequences
- auxiliary multi-view VVCL over disjoint view subsets

## Data

Expected data root:

```text
/home/paichichi/data/RH20T/TCC_RH20T/
  train/
  manifest.csv
  lookup.csv
  tcn_timestamp_groups.csv
  training_index.pt
```

`training_index.pt` is a local dataset index. It is not committed to GitHub.
Rebuild it from `tcn_timestamp_groups.csv` with:

```bash
python scripts/build_universal_matched_group_index.py
```

## Train

Default training arguments already match the current main experiment:

```text
8 timestamps
4 views
H/R exact matched camera combo
unique-task batches when possible
full 4-view fusion for Soft-DTW
2-view vs 2-view disjoint subsets for MV-VVCL
```

Minimal debug run:

```bash
conda run --no-capture-output -n tcc-core \
  python -u scripts/train_multiview_softdtw.py \
  --config configs/debug_4080s.json
```

Config values can be overridden from the command line:

```bash
conda run --no-capture-output -n tcc-core \
  python -u scripts/train_multiview_softdtw.py \
  --config configs/debug_4080s.json \
  --out-dir /tmp/tcc-core/multiview_softdtw_runs/smoke \
  --max-iters 1
```

Useful overrides:

```bash
--batch-pairs 4
--training-index /home/paichichi/data/RH20T/TCC_RH20T/training_index.pt
--data-root /home/paichichi/data/RH20T/TCC_RH20T
--num-timestamps 8
--max-views-per-group 4
--lambda-mv 0.5
--temperature 0.1
--gamma 0.1
--device cuda:0
```

The default matched index path is:

```text
/home/paichichi/data/RH20T/TCC_RH20T/training_index.pt
```

## Review Export

Export copied-frame folders for visual inspection:

```bash
python scripts/export_matched_combo_review_folders.py
```

## Core Files

```text
scripts/train_multiview_softdtw.py
configs/debug_4080s.json
configs/train_2a100.json
scripts/build_tcn_timestamp_group_index.py
scripts/build_universal_matched_group_index.py
scripts/export_matched_combo_review_folders.py
xirl/losses.py
xirl/models.py
```
