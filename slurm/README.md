# NeSI Training

## 1. Check Paths

```bash
ls /nesi/project/uoa04758/xzha593/GitHub/TCC-core
ls /nesi/project/uoa04758/xzha593/envs/activate_tcc_core.sh
ls /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/TCC_RH20T/lookup.csv
ls /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/TCC_RH20T/manifest.csv
ls /nesi/nobackup/uoa04758/xzha593/datasets/RH20T/TCC_RH20T/train
ls /nesi/nobackup/uoa04758/xzha593/datasets/HRAlign/pretrains/UnadaptedR3M.pt
```

Check every path in:

```text
configs/hralign_r3m_l_nesi.yaml
```

## 2. Install

```bash
cd /nesi/project/uoa04758/xzha593/GitHub/TCC-core
source /nesi/project/uoa04758/xzha593/envs/activate_tcc_core.sh
python -m pip install -r requirements.txt
```

The compute nodes may not have internet access. Cache
`distilbert-base-uncased` before submitting, then set:

```yaml
text:
  cache_dir: /your/persistent/huggingface/cache
  local_files_only: true
```

## 3. Interactive Two-GPU Smoke Test

This is only a connectivity test. It uses global batch 2, not the paper's
global batch 200.

```bash
cd /nesi/project/uoa04758/xzha593/GitHub/TCC-core
source /nesi/project/uoa04758/xzha593/envs/activate_tcc_core.sh

torchrun \
  --standalone \
  --nproc-per-node=2 \
  train.py \
  --config configs/hralign_r3m_l_nesi.yaml \
  --set data.max_pairs=16 \
  --set sampling.num_frames=2 \
  --set sampling.sampling_rate=3 \
  --set sampling.crop_size=96 \
  --set 'sampling.train_jitter_scales=[96,112]' \
  --set train.output_dir=/tmp/hralign_smoke \
  --set train.batch_size_per_gpu=1 \
  --set train.expected_global_batch_size=2 \
  --set train.num_workers=0 \
  --set train.persistent_workers=false \
  --set train.max_steps=1 \
  --set train.warmup_steps=1 \
  --set train.schedule_total_steps=1 \
  --set train.log_every=1 \
  --set train.save_every=0
```

Healthy output ends with:

```text
step 00001 loss=...
done steps=1 ...
```

## 4. Strict Paper Batch

The submission script requests:

```text
2 nodes
2 A100 GPUs per node
50 pairs per GPU
global contrastive batch 200
```

Submit:

```bash
cd /nesi/project/uoa04758/xzha593/GitHub/TCC-core
sbatch slurm/train_hralign_nesi.sh configs/hralign_r3m_l_nesi.yaml
```

Check:

```bash
squeue -u "$USER"
tail -f slurm-hralign_r3m_l-JOBID.out
```

The first lines should report:

```text
nodes=2 gpus_per_node=2
pairs=56000
batch_per_gpu=50
world_size=4
contrastive_batch=200
```

## 5. Resume

Set the resume checkpoint in YAML:

```yaml
train:
  resume: /path/to/checkpoint_001000.pt
```

or submit a copied config with that path. Do not resume from
`AdaptedR3M_reproduced.pyth`; it is the downstream export, not the compact
training checkpoint.
