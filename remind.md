我会设计一个 **三层快速评估**，不要一上来跑 RVT2/AGNOSTOS。目标是先回答两个问题：

```text
1. 这两个新 checkpoint 有没有学到 H/R task-progress alignment？
2. 6ts4v 和 8ts3v 哪个更值得拿去跑昂贵 downstream？
```

我建议这样做。

**Level 1: Loss Curve 对比**
先从你已有的文件开始，最快。

每个 run 有：

```text
losses.csv          # rank0
losses_rank1.csv    # rank1
checkpoint_020000.pt
```

画这些曲线：

```text
loss_softdtw
softdtw_top1
softdtw_pos_dist
softdtw_off_dist
loss_aux
aux_top1_h / aux_top1_r
emb_std
```

重点看：

```text
softdtw_pos_dist 是否下降
softdtw_off_dist 是否保持高于 pos
softdtw_top1 是否上升
emb_std 是否没有塌缩
aux loss 是否稳定下降
```

这一步可以快速判断有没有 collapse。  
如果一个 checkpoint 的 `pos/off` 分离明显更好，它优先进入下一步。

**Level 2: Held-out Episode Retrieval**
这是我最推荐的快速评估。  
不需要 RVT，不需要 rollout，只评估 representation。

做法：

```text
从没有参与训练或随机保留的一批 episode 中采样 N 个 episode-level H/R pairs
每个 episode:
  human sequence: T timestamp groups × V views
  robot sequence: T timestamp groups × V views

用 checkpoint 编码：
  H_i -> joint-view embedding sequence
  R_i -> joint-view embedding sequence

计算所有 H_i 和 R_j 的 Soft-DTW distance
得到 N × N distance matrix
```

指标：

```text
H->R retrieval top1
R->H retrieval top1
top5
mean rank
median rank
positive distance
negative distance
pos/neg margin
```

这和训练目标高度一致，但在 held-out episodes 上评估，所以很快、很直接。

你可以比较：

```text
train_2a100_d4r_6ts4v_lr7p5e5_20k
train_2a100_d4r_8ts3v_lr7p5e5_20k
```

这一步能回答：

```text
6 timestamps + 4 views 更好？
还是 8 timestamps + 3 views 更好？
```

我认为这是你现在最该做的评估。

**Level 3: Temporal Alignment 可视化**
从 held-out 里抽 20 个 episode，画每个正样本的 Soft-DTW cost matrix：

```text
8 × 8 或 6 × 6 cost matrix
x-axis: robot timestamps
y-axis: human timestamps
```

好的模型应该出现比较清晰的低 cost 对齐带，而不是整张图乱七八糟。

同时对比：

```text
positive pair: H_i vs R_i
negative pair: H_i vs R_j
```

你想看到：

```text
positive cost matrix 有结构
negative cost matrix 更乱、更高
```

这个特别适合放进汇报图里，视觉上很有说服力。

**D4R / HRP 怎么比较**
这里要小心。你的两个训练 checkpoint 有：

```text
ViT backbone
fusion head
projector_sdtw
projector_aux
```

但原始：

```text
D4R_IN_1M.pth
HRP_IN_1M.pth
```

只有 backbone 权重，没有你训练出来的 fusion/projector。所以不能直接公平地拿它们跑同一个 `projector_sdtw`。

比较方式可以做成两类 baseline：

```text
Baseline A:
  D4R backbone only
  multi-view feature = average over views
  sequence distance = Soft-DTW over averaged backbone features

Baseline B:
  HRP backbone only
  multi-view feature = average over views
  sequence distance = Soft-DTW over averaged backbone features
```

然后你的模型是：

```text
Ours:
  D4R backbone + learned fusion + learned projector_sdtw
```

这样比较虽然不是完全同架构，但很合理：

```text
pretrained backbone feature alone
vs
our H/R-aligned multi-view representation
```

如果你的 checkpoint 在 held-out H/R retrieval 上明显超过 D4R/HRP backbone-only baseline，就很有价值。

**我会优先做的实验表**
做一个表就够清楚：

```text
Model                    T   V   H->R Top1   R->H Top1   Top5   Mean Rank   Pos Dist   Neg Dist
D4R backbone avg-view     8   4
HRP backbone avg-view     8   4
Ours 6ts4v                6   4
Ours 8ts3v                8   3
```

也可以补一组 same setting：

```text
Ours 6ts4v 用 T=6,V=4 eval
Ours 8ts3v 用 T=8,V=3 eval
```

不要混太多 setting，先简单。

**最快结论路线**
我建议你按这个顺序来：

```text
1. 画训练曲线，确认没有 collapse
2. 做 held-out H/R retrieval，比较两个 checkpoint
3. 画 positive/negative Soft-DTW cost matrix
4. 胜出的那个 checkpoint 再拿去跑 RVT2 / AGNOSTOS
```

一句话：  
**先用 representation-level H/R retrieval 筛 checkpoint，再用 rollout 做最终验证。**

这会比直接四个 checkpoint 都跑 RVT2 高效很多，而且结论也更干净。