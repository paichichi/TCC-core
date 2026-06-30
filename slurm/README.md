# NeSI Slurm Quick Commands

This folder contains scripts for launching the four main 2xA100 pretraining jobs.

## 1. Interactive Smoke Test

Use this on an interactive GPU node before submitting jobs.

```bash
cd /nesi/project/uoa04758/xzha593/GitHub/TCC-core
source /nesi/project/uoa04758/xzha593/envs/activate_tcc_core.sh

which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

### ViT LN Smoke Test

Use the same timestamp/view shape as the real job. Do not use a tiny
`2 timestamps / 2 views` override here, because that samples a wider episode
pool and can hit files that are not part of the formal `8ts4v` setting.

### First-Version ViT LN Smoke Test

This checks the original fixed camera-slot fusion version.

```bash
python train.py \
  --exp_cfg_path configs/nesi_vit_ln_first_version_8ts4v.yaml \
  --device 0 \
  -- \
  --batch-episode-pairs 1 \
  --num-timestamps 8 \
  --num-multi-view 4 \
  --max-iters 1 \
  --log-every 1 \
  --save-every 0
```

Expected first line should include:

```text
fusion_mode=fixed_slot ... mv_soft_temporal=False ... num_camera_slots=...
```

```bash
python train.py \
  --exp_cfg_path configs/nesi_vit_ln_8ts4v.yaml \
  --device 0 \
  -- \
  --batch-episode-pairs 1 \
  --num-timestamps 8 \
  --num-multi-view 4 \
  --max-iters 1 \
  --log-every 1 \
  --save-every 0
```

For the `8ts3v` ViT job, use:

```bash
python train.py \
  --exp_cfg_path configs/nesi_vit_ln_8ts3v.yaml \
  --device 0 \
  -- \
  --batch-episode-pairs 1 \
  --num-timestamps 8 \
  --num-multi-view 3 \
  --max-iters 1 \
  --log-every 1 \
  --save-every 0
```

### R3M BN-Affine Smoke Test

```bash
python train.py \
  --exp_cfg_path configs/nesi_r3m_bn_affine_8ts4v.yaml \
  --device 0 \
  -- \
  --batch-episode-pairs 1 \
  --num-timestamps 8 \
  --num-multi-view 4 \
  --max-iters 1 \
  --log-every 1 \
  --save-every 0
```

### R3M Late-Adapter Smoke Test

```bash
python train.py \
  --exp_cfg_path configs/nesi_r3m_late_adapter_8ts4v.yaml \
  --device 0 \
  -- \
  --batch-episode-pairs 1 \
  --num-timestamps 8 \
  --num-multi-view 4 \
  --max-iters 1 \
  --log-every 1 \
  --save-every 0
```

### Near-Full Shape Smoke Test

Use this after the tiny smoke tests pass.

```bash
python train.py \
  --exp_cfg_path configs/nesi_vit_ln_8ts4v.yaml \
  --device 0 \
  -- \
  --batch-episode-pairs 2 \
  --num-timestamps 8 \
  --num-multi-view 4 \
  --max-iters 1 \
  --log-every 1 \
  --save-every 0
```

Healthy output should include:

```text
loaded paired_episodes=...
step 00001 total=... sdtw=... aux=... emb_std=...
done in ...
```

For R3M BN-affine, this is expected:

```text
R3MResNet50Backbone loaded 318 tensors ... missing=0
```

For R3M late-adapter, this is expected:

```text
HRAlignR3MBackbone loaded 318 tensors ... missing=18
```

`missing=18` is normal because the late adapters are newly inserted.

## 2. Dry-Run Job Submission

This checks the four job commands without submitting anything.

```bash
cd /nesi/project/uoa04758/xzha593/GitHub/TCC-core
DRY_RUN=1 ./slurm/submit_four_backbone_experiments_nesi.sh
```

Expected jobs:

```text
tcc_vit_ln_8ts4v              -> configs/nesi_vit_ln_8ts4v.yaml
tcc_vit_ln_8ts3v              -> configs/nesi_vit_ln_8ts3v.yaml
tcc_r3m_bn_affine_8ts4v       -> configs/nesi_r3m_bn_affine_8ts4v.yaml
tcc_r3m_late_adapter_8ts4v    -> configs/nesi_r3m_late_adapter_8ts4v.yaml
```

## 3. Submit All Four Jobs

Only run this after the smoke tests and dry-run pass.

```bash
cd /nesi/project/uoa04758/xzha593/GitHub/TCC-core
./slurm/submit_four_backbone_experiments_nesi.sh
```

To submit only one YAML:

```bash
sbatch --job-name=tcc_vit_ln_8ts4v \
  ./slurm/train_2a100_nesi.sh \
  configs/nesi_vit_ln_8ts4v.yaml
```

Check queue status:

```bash
squeue -u "$USER"
```

Check logs:

```bash
ls -lh /nesi/project/uoa04758/xzha593/logs
tail -f /nesi/project/uoa04758/xzha593/logs/tcc_vit_ln_8ts4v-*.out
```

## 4. Important Paths To Verify

Before submitting, make sure these paths exist on NeSI:

```bash
ls /nesi/nobackup/uoa04758/xzha593/datasets/RH20T
ls /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/TCC_RH20T
ls /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/TCC_RH20T/train
ls /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/training_index.pt
ls /nesi/nobackup/uoa04758/xzha593/datasets/HRAlign/pretrains/D4R_IN_1M.pth
ls /nesi/nobackup/uoa04758/xzha593/datasets/HRAlign/pretrains/UnadaptedR3M.pt
```

If a path is wrong, edit the corresponding YAML in `configs/nesi_*.yaml`.

## 5. Audit Missing Training Images

If training fails with `FileNotFoundError`, check whether the current
`training_index.pt` matches the copied RH20T image directory:

```bash
python scripts/audit_training_index_files.py \
  --data-root /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/TCC_RH20T \
  --training-index /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/training_index.pt
```

If only a small number of image refs are missing, create a filtered index:

```bash
python scripts/audit_training_index_files.py \
  --data-root /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/TCC_RH20T \
  --training-index /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/training_index.pt \
  --output /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/training_index.filtered.pt
```

Then point the `training_index` field in the NeSI YAML files to
`training_index.filtered.pt`.
