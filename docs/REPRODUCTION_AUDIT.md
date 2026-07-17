# HR-Align R3M-Align-L Reproduction Audit

## 1. Verdict

This implementation is the strongest reconstruction supported by the public
evidence, but an exact reproduction claim would be scientifically incorrect.

The public material provides:

1. the paper and supplement;
2. downstream RLBench model code;
3. original `UnadaptedR3M.pt`;
4. released `AdaptedR3M.pyth`;
5. the released checkpoint's embedded training configuration.

It does not provide:

1. the upstream adaptation trainer;
2. the private `R3M_MultiAdapter_Pair_v2` model;
3. the private `Rh20t_pair` dataset loader;
4. the exact 56k pair list and view split;
5. the initialization checkpoint referenced inside `AdaptedR3M.pyth`.

The branch therefore targets:

```text
paper-faithful learning objective
+ release-compatible R3M-Align-L architecture
+ checkpoint-forensics-informed defaults
```

It does not target:

```text
bit-exact recreation of the authors' private run
```

## 2. Evidence Hierarchy

Decisions use the following priority:

1. released tensor shapes and values;
2. released checkpoint configuration;
3. official downstream HumanRobotAlign source;
4. paper and supplement;
5. official R3M source;
6. clearly labeled inference.

This order matters when the paper diagram and released implementation differ.

## 3. Paper Method

### 3.1 Training sample

The paper defines:

```text
D = {(H_i, R_i, L_i)} from i=1 to N
```

where:

- `H_i` is one human demonstration video;
- `R_i` is the paired robot demonstration video;
- `L_i` is the task description;
- human and robot videos perform the same task.

For the local RH20T export, one `lookup.csv` row is used as one training pair.
It is a same-camera, single-view human/robot pair. It is not an
episode-level multi-view set.

### 3.2 Three visual streams

For a batch of `M` pairs, the paper randomly samples `T` frames per video:

```text
h_i^f = F(sample(H_i))
r_i^f = F(sample(R_i))
r_i^t = T(sample(R_i))
```

`F` is original frozen R3M. `T` is the adapted robot encoder formed by adding
late adapters to `F`.

The paper marks the R3M weights as shared. This implementation therefore uses
one R3M instance for all three streams. It evaluates the base backbone
separately for human, frozen robot, and adapted robot, then applies the three
late adapters only to the final robot base features. Backbone parameters never
receive gradients.

### 3.3 Task-aware pooling

R3M's frozen DistilBERT produces a 768-D task feature. A trainable linear layer
maps it to the 2048-D ResNet channel dimension:

```text
l_i = Linear(DistilBERT(L_i))
```

Each visual stream produces features with shape:

```text
[T, H, W, C]
```

The temporal and spatial axes are flattened into tokens. Language is the query,
and visual tokens are keys and values:

```text
A = softmax(tokens dot l_i)
pooled = sum(A * tokens)
```

The same task-aware pooling is applied to frozen human, frozen robot, and
adapted robot features.

### 3.4 Contrastive alignment

Let the three pooled features be:

```text
h_bar_i^f, r_bar_i^f, r_bar_i^t
```

and:

```text
S(x, y) = exp(x^T y / tau)
```

For human-to-robot:

```text
positive:
  S(h_bar_i^f, r_bar_i^t)

denominator:
  positive
  + S(h_bar_i^f, r_bar_i^f)
  + sum over j != i of S(h_bar_i^f, r_bar_j^t)
```

The robot-to-human direction is symmetric. The final loss averages both
directions. This is implemented directly in `hralign/losses.py` and is tested
against a loop-based transcription of Eq. (6).

The paired unadapted robot feature is not a normal in-batch negative. It is a
special baseline candidate. The objective asks the adapted robot feature to
match its human pair better than both:

1. the same robot clip under unadapted R3M;
2. all mismatched examples in the batch.

## 4. Paper Hyperparameters

The paper explicitly reports:

| Setting | Value |
|---|---:|
| RH20T pair subset | about 56k |
| Frames per video | 5 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 200 |
| Temperature | 0.1 |
| Adapter location | after last backbone layer |
| Training length | about 8k steps |
| Hardware | 4 NVIDIA A6000 GPUs |

The global batch size is structurally important. With Eq. (6), a batch of 200
provides 199 mismatched adapted candidates per query. A smaller batch changes
the objective. Accumulating gradients from several smaller contrastive
batches does not reconstruct one batch with 200 negatives.

## 5. Released Adapter Architecture

The paper describes a two-convolution bottleneck:

```text
x + Conv_up(ReLU(Conv_down(x)))
```

The official downstream source and checkpoint use a different, more specific
module:

```text
D_fc1:     grouped 1x1 Conv2d, 2048 -> 512, groups=8
ReLU
D_mapping: dense 1x1 Conv2d, 512 -> 512
ReLU
D_fc2:     grouped 1x1 Conv2d, 512 -> 2048, groups=8
residual add
```

Three identical adapters are applied sequentially after the complete ResNet
`layer4`:

```text
layer4
  -> late_adapter_1
  -> late_adapter_2
  -> late_adapter_3
```

They are not inserted after the three individual layer4 bottleneck blocks.

The released configuration names this variant:

```text
late.layer.3.k.1.down.4.g.8
```

Exact trainable visual adapter parameters:

```text
3 * 527,360 = 1,582,080
```

The paper's parameter table reports roughly 1.6M, which matches the visual
adapters.

The trainable language projection adds:

```text
768 * 2048 + 2048 = 1,574,912
```

Total parameters present in the released optimizer:

```text
1,582,080 + 1,574,912 = 3,156,992
```

The 1.6M paper number therefore describes visual adapter overhead, not every
parameter optimized during alignment.

### 5.1 Why the target is R3M-Align-L

The paper compares four insertion strategies on two Adroit tasks:

| Variant | Visual adapter parameters | Average success |
|---|---:|---:|
| R3M | 0M | 74.0 |
| R3M-Align-E | 0.1M | 79.6 |
| R3M-Align-M | 3.5M | 81.3 |
| R3M-Align-L | 1.6M | 81.3 |
| R3M-Align-E.M.L | 5.2M | 80.3 |

`L` reaches the best reported average while using less than half the visual
adapter capacity of `M`. The released `AdaptedR3M.pyth` also uses the late
variant, so R3M-Align-L is the uniquely well-supported reproduction target.

## 6. Adapter Initialization Finding

The public adapter initializes:

```text
D_mapping.weight = 0
D_mapping.bias   = 0
D_fc1.bias       = 0
D_fc2.bias       = 0
```

Because `D_mapping` is followed by ReLU, the exact initial forward pass is an
identity mapping. More importantly, the initial data-gradient pattern is:

```text
D_fc1 weight/bias:       zero data gradient
D_mapping weight/bias:  zero data gradient
D_fc2 weight:            zero data gradient
D_fc2 bias:              non-zero data gradient
```

The real backward smoke test reproduces this signature.

The released checkpoint further shows:

- all six `D_mapping` tensors remain exactly zero;
- optimizer first moments for all `D_mapping` tensors remain zero;
- all three `D_fc1` biases remain zero;
- all three `D_mapping` biases remain zero;
- all three `D_fc2` biases are learned and numerically near-identical
  (maximum pairwise difference below `3e-9`), though not bitwise equal.

`D_fc1` and `D_fc2` weights have non-zero Adam moments, but this is consistent
with coupled Adam weight decay acting on random non-zero weights even when
their data gradients are blocked.

The evidence therefore indicates that the released visual adapters behave
primarily as three learned output biases, while robot-domain BN statistics and
the language projection carry substantial adaptation. This repository keeps
that behavior because changing it would no longer reproduce the release.

## 7. BatchNorm Finding

Comparing shared tensors in `UnadaptedR3M.pt` and `AdaptedR3M.pyth` gives:

```text
changed learned backbone parameters: 0
changed shared tensors:              159
changed tensor type:                 BN running buffers only
```

ResNet-50 has 53 BatchNorm2d modules. Each contributes:

```text
running_mean
running_var
num_batches_tracked
```

Thus:

```text
53 * 3 = 159
```

All shared learned weights, including BN affine parameters, are identical to
original R3M.

The released config says `MODEL.FROZEN_BN=True`, but its configuration schema
comes from SlowFast. SlowFast's public `frozen_bn_stats` helper only calls
`eval()` on `nn.BatchNorm3d`. R3M uses `nn.BatchNorm2d`, so that helper does
not freeze R3M running statistics. This is consistent with the checkpoint.

Every one of the 53 `num_batches_tracked` buffers has the same exact change:

```text
AdaptedR3M - UnadaptedR3M = 16,980
```

SlowFast names `checkpoint_epoch_00030.pyth` after 30 completed epochs. The
released alignment checkpoint has 283 batches per epoch and 10 completed
epochs. The observed counter change decomposes exactly as:

```text
predecessor robot-only stage:
  30 epochs * 283 batches * 1 backbone call = 8,490

released pair-alignment stage:
  10 epochs * 283 batches * 3 visual streams = 8,490

total:
  8,490 + 8,490 = 16,980
```

This is stronger than merely observing changed BN means and variances. It
supports one shared R3M called separately for the human, frozen-robot, and
adapted-robot streams during pair alignment. The default is therefore:

```yaml
model:
  adapted_bn_mode: shared_stream_stats
```

The interpretation still depends on the unavailable predecessor checkpoint
and private model source, so the code exposes two diagnostic alternatives:

```text
robot_stats: one BN update per step, from adapted robot only
frozen:      no BN running-stat updates
```

In every mode all learned backbone parameters, including BN affine weight and
bias, remain frozen.

## 8. Released Checkpoint Lineage

The released `AdaptedR3M.pyth` contains:

```text
epoch: 9
optimizer step: 2830
current optimizer LR: about 9.9965e-5
model tensors: 438
optimizer tensors: 20
```

This does not look like the paper's stated final point of about 8k steps.

More importantly, the embedded configuration contains:

```text
TRAIN.LOAD_CHECKPOINT_MANUAL = True
```

and points to:

```text
...Adapter.NO.bn_R50.../checkpoint_epoch_00030.pyth
```

That checkpoint was not released. The output directory of the public model
also contains `BN.R.e30`, reinforcing that a 30-epoch predecessor was part of
the actual lineage.

Consequences:

1. The public model is not transparently derivable from
   `UnadaptedR3M.pt -> 2830 released optimizer steps`.
2. Its path and the BN counters strongly support a 30-epoch robot-only BN
   adaptation stage, but its objective and exact forward path remain private.
3. The predecessor may also contain language-projection state, although the
   current checkpoint's optimizer state alone cannot establish that.
4. Reconstructing the predecessor loss or exact data order from a path string
   would be speculation.

The default config therefore follows the paper's direct 8k-step description,
while reproducing the released architecture. It does not invent a private
stage-one algorithm.

## 9. Data Reconstruction

### 9.1 Available local index

The local `lookup.csv` contains:

```text
64,830 same-camera H/R pair rows
9,587 episode-level H/R groups
145 tasks
30 camera IDs
```

The paper reports about 56k pairs but does not publish their identities. The
released optimizer contains `step=2830` at the end of epoch 9. Its stored LR
is exactly the SlowFast warmup LR for 283 steps per epoch. At global batch
200, the private dataset therefore contained between 56,600 and 56,799 rows
before `drop_last`.

That range is forensic evidence about the authors' private run, not a reliable
rule for selecting rows from our separately collected index. The training
configs therefore use every local pair:

```yaml
data:
  max_pairs: null
  subset_seed: 0
```

This gives 64,830 selected pairs and 324 complete batches at global batch 200.
The remaining 30 rows are omitted from each epoch by `drop_last`; epoch-wise
reshuffling changes which rows are omitted. Consequently, our epoch-based
learning-rate trace intentionally follows the full local index and does not
match the private 283-batch trace. Setting an integer `max_pairs` remains
available only for deterministic debug subsets.

### 9.2 Frame sampling

The released config contains:

```text
NUM_FRAMES = 5
SAMPLING_RATE = 12
TARGET_FPS = 30
USE_OFFSET_SAMPLING = True
```

RH20T RGB is 10 Hz. SlowFast's full-video decoder computes the native clip
size as:

```text
12 * 5 / 30 * 10 = 20 frames
```

It selects a random clip start and applies
`torch.linspace(start, start + 19, 5).long()`, producing relative indices
`[0, 4, 9, 14, 19]` when the start is zero. The implementation transcribes
this behavior for the local JPEG sequences.

This is stronger evidence than replacing the released sampling fields with
five uniformly spaced frames. However, exact decoding and offset behavior in
the private loader remains unknown.

### 9.3 Spatial processing

The released configuration provides:

```text
crop: 224 x 224
random horizontal flip: true
ImageNet normalization
AUG.ENABLE: false
TRAIN_JITTER_SCALES: [256, 320]
TRAIN_JITTER_SCALES_RELATIVE: [0.08, 1.0]
TRAIN_JITTER_ASPECT_RELATIVE: [0.75, 1.3333]
```

In SlowFast's standard training loader, non-empty relative scale and aspect
fields select Inception-style random resized crop, taking precedence over the
absolute short-side jitter values. The default reproduces that path:

```yaml
sampling:
  spatial_mode: random_resized_crop
  relative_crop_scale: [0.08, 1.0]
  relative_crop_aspect: [0.75, 1.3333]
```

`AUG.ENABLE=false` disables the additional augmentation subsystem. It does
not disable base random crop and horizontal flip. One crop and flip are shared
by every frame in a clip; human and robot clips receive independent draws.
`short_side_jitter` remains available as a diagnostic alternative because the
private `Rh20t_pair` loader is unavailable.

One spatial transform is shared by all frames in a clip. Human and robot clips
receive independently sampled transforms.

### 9.4 Compact frame IDs

`manifest.csv.start_frame` refers to the source video. The local
`TCC_RH20T/train` export has already cropped and reindexed each sequence from
zero. Adding `start_frame` a second time produces missing-file errors.

The default is therefore:

```yaml
sampling:
  frame_index_mode: compact
```

The complete 64,830-pair boundary audit finds zero missing first/last frames
in this mode. Neither the loader nor the index audit reads timestamp groups.

### 9.5 Reproducible resume

Each sampled index carries its epoch. Temporal sampling and spatial transforms
use an independent deterministic random stream for:

```text
(training seed, epoch, pair index, human-or-robot stream)
```

The checkpoint step determines both the next epoch and its within-epoch batch
offset. Resuming therefore skips already completed sampler indices without
decoding them and preserves the remaining samples and augmentations. This
changes neither the sampling distribution nor the contrastive objective; it
removes accidental epoch-prefix repetition after preemption.

## 10. Language Encoder

Official R3M uses:

```text
distilbert-base-uncased
padding=True
last_hidden_state.mean(dim=1)
```

The mean is not masked by `attention_mask`, so an embedding can depend on the
longest sentence in the current batch. This implementation reproduces that
behavior when:

```yaml
text:
  cache_by_text: false
  embeddings: ""
```

Precomputing one vector per task is supported for speed, but it changes this
padding-dependent detail and is not the strict path.

## 11. Feature Normalization

The paper defines dot-product similarity but does not explicitly state whether
pooled vectors are normalized. The released run directory includes:

```text
lang.feat.norm
```

The default interprets that clue as L2-normalizing pooled features before Eq.
(6):

```yaml
model:
  normalize_language_query: false
  normalize_visual_tokens_for_attention: false
  normalize_pooled_features: true
```

This is an inference. It is exposed in the config because the private feature
normalization source is unavailable.

## 12. Optimizer and Schedule

The released optimizer confirms:

```text
Adam betas: (0.9, 0.999)
base LR: 1e-4
weight decay: 1e-4
zero weight decay for 1-D parameters and biases
gradient L2 clip: 1.0
warmup start LR: 1e-6
warmup: 10 epochs
cosine end LR: 1e-6
configured schedule: 300 epochs
```

The released checkpoint provides a stronger schedule check than the rounded
paper count:

```text
optimizer step = 2830
saved epoch = 9
steps per epoch = 283
10 warmup epochs = 2830 steps
300 schedule epochs = 84,900 steps
stored LR = 9.996501766784452e-5
```

SlowFast sets LR before each update from
`epoch_exact = cur_epoch + cur_iter / steps_per_epoch`. The 2,830th update
therefore uses `epoch_exact = 9 + 282/283`, which reproduces the stored LR
exactly. The scheduler follows that epoch-based rule instead of approximating
it with fixed step counts.

The paper stops adaptation at about 8,000 steps, long before the configured
300-epoch cosine endpoint. The implementation preserves this distinction:

```yaml
train:
  max_steps: 8000
  warmup_epochs: 10.0
  schedule_epochs: 300.0
```

## 13. DDP Semantics

Each rank encodes its local pairs. Differentiable all-gather constructs global:

```text
human frozen features
robot frozen features
robot adapted features
```

Eq. (6) is then evaluated over the full global batch. This makes a 4 x 50 run
equivalent in candidate structure to batch 200.

The code does not use four independent local losses with only 49 negatives.
The repository also includes a two-process numerical audit that compares the
DDP parameter gradient against one single-process global batch.

That gradient equivalence does not make `1 x 200` numerically identical to the
paper's `4 x 50` execution geometry. Two batch-local operations differ:

1. With no synchronized BN, every official rank computes visual features from
   50 pairs (250 frames per visual stream), while a single-GPU batch of 200 uses
   1,000 frames per stream for BatchNorm2d.
2. R3M averages the DistilBERT padded token dimension without masking it.
   Padding is therefore determined by 50 local task descriptions in the
   reported run, rather than all 200 descriptions at once.

The global contrastive candidate set and DDP trainable-parameter gradient are
correct in both cases, but the visual and language features are not guaranteed
to match. The closest hardware-semantic reproduction remains four processes
with 50 pairs per process.

BN is not synchronized because the released config says:

```text
BN.SYNC_BN = false
BN.GLOBAL_SYNC = false
```

## 14. Export Compatibility

The released checkpoint contains 438 model tensors. The reconstructed export
contains:

```text
ResNet-50 backbone and BN buffers
18 adapter tensors
100 DistilBERT tensors
2 language projection tensors
```

Automated comparison reports:

```text
produced tensors: 438
reference tensors: 438
missing keys: 0
unexpected keys: 0
shape mismatches: 0
```

The exported file is:

```text
AdaptedR3M_reproduced.pyth
```

Its `model_state` can be consumed by the official HumanRobotAlign downstream
loader.

## 15. Paper Evaluation Context

The upstream adaptation is evaluated by freezing the resulting visual
backbone during downstream policy learning.

Reported paper-level evidence includes:

```text
Adroit two-task average:
  R3M:        74.0
  R3M-Align:  81.3  (+7.3)

RLBench 18-task average:
  R3M:        50.3
  R3M-Align:  59.2  (+8.9)

Real-world five-task average improvement:
  R3M-Align over R3M: +11 percentage points
```

The supplement reports an Adroit average of 80.0 after removing
language-guided feature enhancement, versus 81.3 with it. This supports
including the frozen DistilBERT query and trainable projection in the upstream
trainer rather than treating language as downstream-only metadata.

For RLBench, the supplement says the authors:

1. replace RVT's four intra-image attention layers with the frozen pretrained
   or adapted visual backbone;
2. use one attention layer to fuse image and language features;
3. remove visual-backbone spatial downsampling used by the normal classifier;
4. keep the visual representation frozen during policy learning.

The official repository exposes this downstream stage, not the upstream
alignment trainer. This branch focuses on reconstructing the missing upstream
stage and exports a checkpoint compatible with that released downstream code.

## 16. What Was Deliberately Removed

This branch does not contain or invoke:

- ViT LayerNorm tuning;
- multi-view InfoNCE;
- fixed camera slots;
- attention pooling over camera sets;
- view masking or feature augmentation;
- Soft-DTW;
- TCC;
- Sinkhorn soft alignment;
- timestamp-group indices;
- B x B episode alignment;
- RVT2-lite result tables.

Those belong to separate method branches and would obscure the HR-Align
baseline.

## 17. Verification Performed

The implementation has passed:

```text
21 unit tests
real RH20T image forward/backward
real R3M DistilBERT forward
full one-step trainer and final export
resume from compact training checkpoint
two-process DDP global-negative smoke test and gradient-equivalence audit
64,830-pair boundary-file audit
released checkpoint forensic audit
438/438 export layout comparison
```

Observed smoke-test gradient signature matches the released adapter behavior:

```text
non-zero data gradients:
  3 x D_fc2.bias
  lang_linear.weight
  lang_linear.bias

zero initial data gradients:
  all remaining adapter tensors
```

## 18. Remaining Reproduction Risks

Ordered by likely impact:

1. Missing 30-epoch predecessor checkpoint referenced by the release.
2. Missing exact 56k pair/view split.
3. Missing private BN stream-update logic.
4. Missing exact feature normalization implementation.
5. Missing private video decode and offset-sampling implementation.
6. Unknown whether paired clips share spatial augmentation randomness.
7. Released 2830-step snapshot versus paper's roughly 8k-step statement.

These should be disclosed in any paper, report, or baseline table. A suitable
label is:

```text
HR-Align R3M-Align-L, evidence-grounded reproduction
```

Do not label it:

```text
official HR-Align training code
```

## 19. Source Map

- Paper:
  <https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_Mitigating_the_Human-Robot_Domain_Discrepancy_in_Visual_Pre-training_for_Robotic_CVPR_2025_paper.pdf>
- Supplement:
  <https://openaccess.thecvf.com/content/CVPR2025/supplemental/Zhou_Mitigating_the_Human-Robot_CVPR_2025_supplemental.pdf>
- Official project:
  <https://jiaming-zhou.github.io/projects/HumanRobotAlign/>
- Official repository:
  <https://github.com/jiaming-zhou/HumanRobotAlign>
- Exact public adapter source inspected:
  <https://github.com/jiaming-zhou/HumanRobotAlign/blob/0da2d96ed2410076aaa19316833915cd6627c5cd/rvt/mvt/resnet.py>
- Exact R3M language source inspected:
  <https://github.com/facebookresearch/r3m/blob/b2334e726887fa0206962d7984c69c5fb09cceab/r3m/models/models_language.py>
- SlowFast BN helper:
  <https://github.com/facebookresearch/SlowFast/blob/287ec0076846560f44a9327e931a5a2360240533/slowfast/utils/misc.py>
- RH20T task descriptions:
  <https://rh20t.github.io/static/task_description.json>
