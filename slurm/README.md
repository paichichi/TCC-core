那就跑最小 smoke test，每个只跑 `1 iter` 就够了。

先进入环境：

```bash
cd /nesi/project/uoa04758/xzha593/GitHub/TCC-core
source /nesi/project/uoa04758/xzha593/envs/activate_tcc_core.sh
```

**ViT Method3 8ts4v**

```bash
python train.py \
  --exp_cfg_path configs/nesi_method3_vit_ln_8ts4v.yaml \
  --device 0 \
  -- \
  --batch-episode-pairs 1 \
  --num-timestamps 8 \
  --num-multi-view 4 \
  --max-iters 1 \
  --log-every 1 \
  --save-every 0
```

**ResNet Method3 8ts4v**

```bash
python train.py \
  --exp_cfg_path configs/nesi_method3_resnet_ln_8ts4v.yaml \
  --device 0 \
  -- \
  --batch-episode-pairs 1 \
  --num-timestamps 8 \
  --num-multi-view 4 \
  --max-iters 1 \
  --log-every 1 \
  --save-every 0
```

如果这两个都能打印出类似：

```text
step 00001 ...
done in ...
losses=...
```

就说明代码、环境、数据路径、pretrain 路径都通了。