# HR-Align R3M-Align-L Reproduction

This branch is an isolated, evidence-grounded reconstruction of the upstream
training pipeline for **R3M-Align-L** from:

> Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for
> Robotic Manipulation, CVPR 2025.

It intentionally contains no TCC, multi-view fusion, Soft-DTW, soft alignment,
timestamp-group loss, or RVT2-lite experiment code.

## Reproduction Status

The authors released the paper, downstream RLBench code, `UnadaptedR3M.pt`,
and `AdaptedR3M.pyth`, but not the adaptation trainer or the exact RH20T split.
Therefore this repository makes two precise claims:

1. **Release-compatible architecture:** the exported `model_state` has the
   same 438 keys and tensor shapes as the released `AdaptedR3M.pyth`.
2. **Paper-faithful objective:** the trainer implements the three visual
   streams, task-aware pooling, and Eq. (6) contrastive loss from the paper.

It does not claim bit-exact reproduction of the private training run. The full
evidence audit and all unresolved gaps are documented in
[docs/REPRODUCTION_AUDIT.md](docs/REPRODUCTION_AUDIT.md).

## Implemented Pipeline

One dataset item is one same-camera human/robot video pair:

```text
human clip H_i ---- frozen R3M F -------------------- h_i^f
robot clip R_i ---- frozen R3M F -------------------- r_i^f
robot clip R_i ---- frozen R3M + 3 late adapters --- r_i^t
task text L_i ----- frozen DistilBERT + trainable FC --- query l_i

spatial-temporal features + l_i
        |
        +-- task-aware attention pooling
        |
        +-- h_bar_i^f, r_bar_i^f, r_bar_i^t
        |
        +-- symmetric HR contrastive alignment, Eq. (6)
```

For pair `i`, `(h_i^f, r_i^t)` is positive. The denominator contains:

- the paired unadapted baseline `(h_i^f, r_i^f)`;
- every unpaired adapted robot feature in the global batch;
- the symmetric robot-to-human direction.

There is no temporal alignment loss and no multi-view fusion in HR-Align.

## Environment

Python 3.10 or newer is recommended.

```bash
conda create -n hralign python=3.10 -y
conda activate hralign
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The R3M checkpoint contains the DistilBERT weights, but not the tokenizer
vocabulary. The first run must be able to obtain
`distilbert-base-uncased`, or `text.tokenizer_name` must point to a local copy.

## Required Files

Local defaults in `configs/hralign_r3m_l.yaml` expect:

```text
/home/paichichi/data/RH20T/TCC_RH20T/
  lookup.csv
  manifest.csv
  train/episode_xxxxxx/sequence_id/000000.jpg

/home/paichichi/data/pretrain/UnadaptedR3M.pt
```

The repository includes the official English RH20T task descriptions at:

```text
metadata/rh20t_task_descriptions.json
```

`TCC_RH20T` stores each cropped sequence with compact frame IDs beginning at
zero. Keep:

```yaml
sampling:
  frame_index_mode: compact
```

Use `manifest_offset` only if JPEG filenames preserve the original RH20T frame
indices.

## Audit Before Training

Check the deterministic 56k subset and every selected sequence boundary:

```bash
python scripts/audit_dataset.py \
  --root /home/paichichi/data/RH20T/TCC_RH20T \
  --lookup /home/paichichi/data/RH20T/TCC_RH20T/lookup.csv \
  --manifest /home/paichichi/data/RH20T/TCC_RH20T/manifest.csv \
  --task-descriptions metadata/rh20t_task_descriptions.json \
  --max-pairs 56000 \
  --frame-index-mode compact \
  --check-files
```

Audit the released model against original R3M:

```bash
python scripts/audit_release_checkpoint.py \
  --unadapted /home/paichichi/data/pretrain/UnadaptedR3M.pt \
  --adapted /home/paichichi/data/pretrain/AdaptedR3M.pyth
```

## Smoke Test

One real-image forward/backward pass without DistilBERT:

```bash
python scripts/smoke_test.py \
  --root /home/paichichi/data/RH20T/TCC_RH20T \
  --lookup /home/paichichi/data/RH20T/TCC_RH20T/lookup.csv \
  --manifest /home/paichichi/data/RH20T/TCC_RH20T/manifest.csv \
  --task-descriptions metadata/rh20t_task_descriptions.json \
  --pretrain /home/paichichi/data/pretrain/UnadaptedR3M.pt \
  --device cuda
```

Add `--with-text` to test the exact R3M DistilBERT path:

```bash
python scripts/smoke_test.py \
  --root /home/paichichi/data/RH20T/TCC_RH20T \
  --lookup /home/paichichi/data/RH20T/TCC_RH20T/lookup.csv \
  --manifest /home/paichichi/data/RH20T/TCC_RH20T/manifest.csv \
  --task-descriptions metadata/rh20t_task_descriptions.json \
  --pretrain /home/paichichi/data/pretrain/UnadaptedR3M.pt \
  --device cuda \
  --with-text
```

## Training

Single-GPU debug:

```bash
python train.py \
  --config configs/hralign_r3m_l.yaml \
  --set train.batch_size_per_gpu=8 \
  --set train.expected_global_batch_size=8 \
  --set train.max_steps=10 \
  --set train.save_every=0
```

Paper-scale contrastive batch:

```bash
torchrun \
  --standalone \
  --nproc-per-node=4 \
  train.py \
  --config configs/hralign_r3m_l.yaml
```

The paper uses 4 GPUs, 50 pairs per GPU, and global batch 200. Global batch is
part of the method because all other pairs become negatives. Gradient
accumulation does not recreate those missing negatives.

CLI overrides use dotted YAML paths:

```bash
python train.py \
  --config configs/hralign_r3m_l.yaml \
  --set train.max_steps=100 \
  --set train.output_dir=runs/debug_100
```

Resume:

```bash
python train.py \
  --config configs/hralign_r3m_l.yaml \
  --resume runs/hralign_r3m_l/checkpoint_001000.pt
```

## NeSI

The strict NeSI script requests two nodes with two A100 GPUs per node:

```bash
sbatch slurm/train_hralign_nesi.sh configs/hralign_r3m_l_nesi.yaml
```

See [slurm/README.md](slurm/README.md) for path checks and a two-GPU
interactive smoke test.

## Outputs

Each run writes:

```text
config_resolved.yaml
losses.csv
checkpoint_XXXXXX.pt
checkpoint_last.pt
AdaptedR3M_reproduced.pyth
```

`checkpoint_*.pt` is a compact resume checkpoint containing trainable tensors,
adapted BN buffers, optimizer, scheduler, and scaler state.

`AdaptedR3M_reproduced.pyth` uses the public HumanRobotAlign layout and can be
loaded by its downstream `adaptedR3M.yaml` path.

## Tests

```bash
python -m pytest -q
```

The tests cover:

- exact vectorization of paper Eq. (6);
- released adapter parameter count and initialization behavior;
- compact and manifest-offset frame layouts;
- all 438 released checkpoint keys and tensor shapes.

## Primary Sources

- [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_Mitigating_the_Human-Robot_Domain_Discrepancy_in_Visual_Pre-training_for_Robotic_CVPR_2025_paper.pdf)
- [Supplementary material](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Zhou_Mitigating_the_Human-Robot_CVPR_2025_supplemental.pdf)
- [Official HumanRobotAlign repository](https://github.com/jiaming-zhou/HumanRobotAlign)
- [Official R3M repository](https://github.com/facebookresearch/r3m)
- [Official RH20T task descriptions](https://rh20t.github.io/static/task_description.json)
