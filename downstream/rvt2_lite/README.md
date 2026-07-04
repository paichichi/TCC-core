# RVT2-Lite Probe

Cheap downstream probe for deciding whether a TCC-core checkpoint is worth a
full RVT2 run.

This probe uses the latest RVT repo:

```text
/home/paichichi/projects/rvt-3d-policy-head-adaption
```

The idea is to keep the downstream setup close to RVT2, but shrink the budget:

- small RLBench task set
- few demos per task
- short training
- real rollout evaluation

## Directory Layout

```text
downstream/rvt2_lite/
  configs/      # RVT2-lite experiment configs
  slurm/        # NeSI job wrappers for this downstream probe
  runs/         # RVT2-lite policy checkpoints, logs, raw CSVs, and summaries
```

Keep RVT2-lite outputs under `downstream/rvt2_lite/runs/`. The repo-level
`downstream/analysis/` directory is reserved for TCC-core
representation/retrieval analysis, not downstream policy results.

## Models

We compare four visual initializers:

```text
D4R_IN_1M
HRP_IN
Ours 6ts4v checkpoint
Ours 8ts3v checkpoint
```

For Ours checkpoints, RVT only uses the ViT backbone weights. The RH20T fusion
head and projectors are pretraining scaffolding and are not used by the policy.

## Default Probe

Current lite config:

```text
tasks: open_drawer,push_buttons
num_train: 10 demos/task
train_iterations: 10000 per epoch
epochs: 1
model: MVT_ViT
stage_two: true
```

This is not meant to replace the full RVT2 result. It is a decision experiment:

```text
If Ours beats D4R/HRP here, it is worth running full RVT2.
```

## Train

From this repo:

```bash
cd /home/paichichi/projects/rvt-3d-policy-head-adaption
export PYTHONPATH=/home/paichichi/projects/rvt-3d-policy-head-adaption:/home/paichichi/projects/rvt-3d-policy-head-adaption/rvt
```

Train one initializer:

```bash
python -m rvt.train \
  --exp_cfg_path /home/paichichi/projects/TCC-core/downstream/rvt2_lite/configs/ours_8ts3v.yaml \
  --mvt_cfg_path /home/paichichi/projects/rvt-3d-policy-head-adaption/rvt/mvt/configs/rvt2.yaml \
  --device 0
```

Train all four:

```bash
for cfg in \
  d4r.yaml \
  hrp.yaml \
  ours_6ts4v.yaml \
  ours_8ts3v.yaml
do
  python -m rvt.train \
    --exp_cfg_path /home/paichichi/projects/TCC-core/downstream/rvt2_lite/configs/${cfg} \
    --mvt_cfg_path /home/paichichi/projects/rvt-3d-policy-head-adaption/rvt/mvt/configs/rvt2.yaml \
    --device 0
done
```

## Evaluate

After training, each run writes:

```text
downstream/rvt2_lite/runs/base/<run_name>/model_0.pth
```

Evaluate a trained model:

```bash
python -m rvt.eval \
  --model-folder /home/paichichi/projects/TCC-core/downstream/rvt2_lite/runs/base/ours_8ts3v \
  --model-name model_0.pth \
  --tasks open_drawer push_buttons \
  --eval-datafolder /home/paichichi/data/rvt/train/replay/replay_train \
  --eval-episodes 10 \
  --episode-length 25 \
  --headless \
  --device 0 \
  --log-name lite_eval
```

## Smoke Test

Run this first before the full lite probe:

```bash
cd /home/paichichi/projects/rvt-3d-policy-head-adaption
export PYTHONPATH=/home/paichichi/projects/rvt-3d-policy-head-adaption:/home/paichichi/projects/rvt-3d-policy-head-adaption/rvt

conda run --no-capture-output -n hralign_py39 \
  python -m rvt.train \
  --exp_cfg_path /home/paichichi/projects/TCC-core/downstream/rvt2_lite/configs/ours_8ts3v.yaml \
  --mvt_cfg_path /home/paichichi/projects/rvt-3d-policy-head-adaption/rvt/mvt/configs/rvt2.yaml \
  --device 0 \
  --exp_cfg_opts "tasks open_drawer num_train 2 bs 2 num_workers 0 train_iterations 2 overwriter_log_dir /home/paichichi/projects/TCC-core/downstream/rvt2_lite/runs/smoke/ours_8ts3v"
```

## Readout

Primary metric:

```text
rollout success rate per task
```

Secondary metrics:

```text
training loss
convergence speed
```

Decision rule:

```text
If Ours improves average success by >= 10 percentage points over D4R/HRP,
run full RVT2.
```
