# TCC-core

TCC-core is a trimmed Temporal Cycle Consistency pretraining codebase for
fine-tuning visual backbones on RH20T-style paired videos.

This repository keeps the pieces we need for backbone pretraining:

- TCC losses in `xirl/losses.py`
- video dataset loading and frame sampling in `xirl/dataset.py`,
  `xirl/frame_samplers.py`, and `xirl/video_samplers.py`
- TCC training logic in `xirl/trainers/tcc.py`
- ResNet18 and ViT-B/16 backbone wrappers in `xirl/models.py`
- RH20T paired configs in `configs/rh20t/pretraining/`

The original XIRL reward, downstream evaluation, imitation learning,
X-MAGICAL launchers, and SAC policy-training code have been removed.

## Dataset Shape

The paired RH20T sampler expects a root like:

```text
/home/paichichi/data/RH20T/TCC_RH20T/
  train/
    task_0001/
      000000/
        000000.jpg
        ...
      000001/
        000000.jpg
        ...
  manifest.csv
```

The `manifest.csv` must include:

```text
sequence_id,paired_sequence_id,episode_id,task_id,role,num_frames,embodiment_id,camera_id
```

For paired TCC, a batch is ordered as:

```text
[h0, r0, h1, r1, ...]
```

and TCC is computed only inside each adjacent human-robot pair.

For paired sampling, the number of frames per video is controlled by:

```yaml
data:
  video_sampler_seed: 1
  paired_fixed_frames: -1
  paired_frame_sample_ratio: 1.0
  paired_max_frames: 40
  paired_min_frames: 16
```

`video_sampler_seed` controls the shuffled pair order. Use the same value to
reproduce the same random pair sequence, or change it to sample a different
order.

With the default RH20T configs, the sampled frame count is:

```text
T = min(40, shortest_video_len_in_batch)
```

This avoids padding short videos while capping normal/long videos at 40 frames.
Set `paired_fixed_frames` to a positive value, such as `40`, only when you want
to force a fixed T; videos shorter than T are padded by repeating their final
frame and clamping repeated `frame_idxs` to the true final frame.

More generally, dynamic ratio-based sampling uses:

```text
T = max(paired_min_frames, floor(shortest_video_len_in_batch * paired_frame_sample_ratio))
```

In dynamic mode, `paired_max_frames` adds an explicit upper bound and T is
finally clamped so it never exceeds the shortest video in that batch.

## Useful Commands

Debug one batch:

```bash
PYTHONPATH=/home/paichichi/projects/TCC-core \
python debug_dataset.py \
  --config=configs/rh20t/pretraining/paired_tcc_vit_d4r_in.py \
  --debug
```

Start D4R ViT paired TCC pretraining:

```bash
PYTHONPATH=/home/paichichi/projects/TCC-core \
python pretrain.py \
  --experiment_name=debug_layernorm \
  --config_yaml=configs/rh20t/pretraining/paired_tcc_vit_d4r_in_layernorm.yaml
```

Run on a specific GPU:

```bash
scripts/run_pretrain.sh \
  0 \
  ln_gpu0 \
  configs/rh20t/pretraining/paired_tcc_vit_d4r_in_layernorm.yaml
```

Run with DDP on multiple GPUs:

```bash
scripts/run_pretrain.sh \
  0,1,2,3 \
  ln_ddp_4gpu \
  configs/rh20t/pretraining/paired_tcc_vit_d4r_in_layernorm.yaml
```

Short loss-curve smoke test:

```bash
scripts/run_pretrain.sh \
  0 \
  ln_loss_debug \
  configs/rh20t/pretraining/paired_tcc_vit_d4r_in_layernorm.yaml \
  --max_iters=100 \
  --log_every=1 \
  --eval_every=0 \
  --checkpoint_every=0
```

The current ViT configs use letterbox resize to fit native RH20T frames into a
224x224 ViT-B/16 input while preserving aspect ratio.

## Fine-Tuning Scope

ViT fine-tuning is controlled by one config value:

```yaml
model:
  trainable_scope: layernorm_head
```

Supported values:

- `layernorm_head`: train ViT LayerNorm gamma/beta plus the TCC linear head.
- `all`: train the full ViT backbone plus the TCC linear head.
- `head`: train only the TCC linear head.

Two D4R-IN YAML templates are provided:

- `configs/rh20t/pretraining/paired_tcc_vit_d4r_in_layernorm.yaml`
- `configs/rh20t/pretraining/paired_tcc_vit_d4r_in_full.yaml`
