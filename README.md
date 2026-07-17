# TCC-core

RH20T multi-view pretraining code.

The current training objective is not vanilla TCC. It uses:

- timestamp-aligned multi-view fusion
- camera-matched H/R pairing
- Human/Robot sequence-level contrastive Soft-DTW
- same-side multi-view VVCL auxiliary loss

## 1. Environment

Install a PyTorch build that matches the server CUDA version first. Then install this repo.

```bash
conda create -n tcc-core python=3.10 -y
conda activate tcc-core
```

Install PyTorch:

```bash
# Choose torch / torchvision according to the server CUDA version.
# Do not blindly reuse a command from another machine.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For the local RTX 5090 (`sm_120`), the validated isolated environment is:

```bash
conda create -n tcc-core-sm120 python=3.10 pip -y
conda activate tcc-core-sm120
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy pillow pyyaml pytest
python -m pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

The verified architecture list includes `sm_120`. Keep the older `tcc-core`
environment unchanged for historical runs.

Install project dependencies:

```bash
pip install numpy pillow pyyaml
pip install -e . --no-deps
```

Check the environment:

```bash
python - <<'PY'
import torch
import torchvision
import yaml
from xirl.models import ViTB16Backbone

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())
print("imports OK")
PY
```

## 2. Data

Expected data directory:

```text
TCC_RH20T/
  train/
  manifest.csv
  lookup.csv
  tcn_timestamp_groups.csv
  training_index.pt
```

`training_index.pt` is a local dataset index. Do not commit it to GitHub.

If you only have `tcn_timestamp_groups.csv`, rebuild the index:

```bash
python scripts/build_universal_matched_group_index.py \
  --index /path/to/TCC_RH20T/tcn_timestamp_groups.csv \
  --output /path/to/TCC_RH20T/training_index.pt
```

## 3. Config

Main config files:

```text
configs/debug_4080s.yaml
configs/train_2a100.yaml
```

When moving to a new server, update these paths:

```yaml
data_root: /path/to/TCC_RH20T
training_index: /path/to/TCC_RH20T/training_index.pt
pretrain_path: /path/to/D4R_IN_1M.pth
output_root: /path/to/output_runs
```

Commonly tuned settings:

```yaml
batch_episode_pairs: 16
num_timestamps: 8
num_multi_view: 4
lr: 0.000075
max_iters: 20000
lambda_mv: 0.5  # If MV-VVCL dominates, try 0.1 first.
```

`batch_episode_pairs` is per GPU/process. With 2 GPUs, total processed H/R episode pairs per step is roughly `batch_episode_pairs * 2`.

## 4. Local Debug

Single GPU:

```bash
python train.py \
  --exp_cfg_path configs/debug_4080s.yaml \
  --device 0
```

Fast smoke test:

```bash
python train.py \
  --exp_cfg_path configs/debug_4080s.yaml \
  --device 0 \
  -- \
  --batch-episode-pairs 2 \
  --max-iters 1 \
  --log-every 1 \
  --save-every 0
```

## 5. Multi-GPU Training

2 GPUs:

```bash
python train.py \
  --exp_cfg_path configs/train_2a100.yaml \
  --device 0,1
```

4 GPUs:

```bash
python train.py \
  --exp_cfg_path configs/train_2a100.yaml \
  --device 0,1,2,3
```

## 6. Slurm

Edit:

```text
slurm/train_2a100_nesi.sh
```

Main fields to update:

```bash
PROJECT_ROOT=/path/to/TCC-core
ACTIVATE_SCRIPT=/path/to/activate_env.sh
```

Submit one experiment:

```bash
sbatch slurm/train_2a100_nesi.sh
```

Submit the 3-job ablation:

```bash
bash slurm/submit_d4r_ablation_2a100_nesi.sh
```

## 7. Output

Each run writes to:

```text
output_root/run_name/
  losses.csv
  checkpoint_001000.pt
  checkpoint_002000.pt
  ...
```

Checkpoint format:

```python
ckpt = torch.load("checkpoint_001000.pt", map_location="cpu")
model_state = ckpt["model"]
args = ckpt["args"]
```

For RVT transfer, the main object you need is `ckpt["model"]`.

New R3M late-adapter checkpoints also contain:

```python
ckpt["model_format"]["r3m_late_adapter_layout"]
# post_layer4_sequential_v1
```

ResNet late-adapter checkpoints without this field use the legacy,
shape-compatible but computation-incompatible adapter placement. Retrain those
upstream checkpoints before treating them as the release-compatible Method 3
ResNet result. ViT checkpoints are unaffected.

## 8. Core Files

```text
train.py
scripts/train_multiview_softdtw.py
xirl/models.py
xirl/losses.py
configs/default.yaml
configs/debug_4080s.yaml
configs/train_2a100.yaml
slurm/train_2a100_nesi.sh
```
