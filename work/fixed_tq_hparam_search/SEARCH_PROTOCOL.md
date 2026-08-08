# Fixed-\(D,Q\) Hyperparameter Search Protocol

## Purpose

This search calibrates one frozen method definition for the paper.  It is not
allowed to change the architecture while tuning, and it does not select a model
from the final 23-task evaluation suite.

The two backbone lanes are:

- `vit_imagenet`: D4R ImageNet ViT-B/16, trainable LayerNorm affine parameters.
- `r3m_bn_bi`: R3M ResNet-50, trainable BatchNorm affine parameters.

Both lanes use the single-view two-branch method, bidirectional within-domain
InfoNCE, `aux_teacher_stop_grad=false`, `aux_global_negatives=false`, no control
loss, and no spatial-preservation loss.  The ResNet lane retains its registered
`temporal_centered` feature preprocessing; the ViT lane retains `raw`.

## Fixed choices

- 40,000 upstream iterations for every candidate.
- Seed 1 during search.
- Eight ordered timestamps per episode.
- `soft_alignment_temperature=0.1`.
- `soft_alignment_teacher_feature_weight=1`.
- 20 Sinkhorn iterations.
- No candidate is ranked by final scalar training loss.
- Checkpoints are written every 5,000 iterations and resume with optimizer
  state.

Temperature and Sinkhorn iteration count are fixed a priori to control the
number of researcher degrees of freedom.  They may receive a post-selection
sensitivity plot, but they are not used to choose the headline model.

## Stages

### A1 — global-prior teacher

Full factorial grid, independently executed for both backbone lanes:

- \(\rho \in \{0.25, 0.5, 1.0\}\)
- \(\epsilon \in \{0.025, 0.05, 0.1\}\)

All other values remain at their preregistered defaults.

### A2 — optimizer scale

For the A1-selected teacher, evaluate:

- ViT learning rate: \(0.5\times, 1\times, 2\times\) the default
  \(7.5\times10^{-5}\).
- R3M learning rate: \(0.5\times, 1\times, 2\times\) the default
  \(10^{-4}\).

Learning rate is allowed to remain backbone-specific because the trainable
normalization parameterizations differ.

### B — local transition prior

For the selected A2 setting:

- `max_forward_step` \(\in \{1,3,7\}\).  With eight sampled timestamps, 7 is
  the effective no-large-forward-jump ablation.
- \(\lambda_Q \in \{0.05,0.1,0.2\}\).

This factorial stage tests both the allowed transition range and the strength
of the local prior.

### C — relative objective weight

Set \(\lambda_{\mathrm{SA}}=1\) for identifiability and search:

- \(\lambda_{\mathrm{InfoNCE}}\in\{0.25,0.5,1.0\}\).

Searching both loss coefficients independently would duplicate their common
scale.  Global optimization scale is already tested in Stage A2; Stage C
therefore identifies the meaningful loss ratio.

## Downstream screening protocol

Every complete upstream checkpoint is converted, when needed, and receives the
same RVT2-lite procedure on RTX 5090:

- one 25,000-iteration policy training run;
- 12 validation tasks (six Level 1 and six Level 2);
- three evaluation repeats;
- ten episodes per task per repeat;
- episode length 25.

The final 23-task evaluation is not consulted during search.  After selection,
the final report must additionally show the aggregate on the 11 tasks not used
by RVT2-lite, calculated from task-level final-evaluation logs.

## Candidate validity and selection

1. Require a valid 40k checkpoint, finite logged metrics, and a complete
   `3 x 12 x 10` RVT2-lite result.
2. Primary ranking metric: overall RVT2-lite mean.
3. Secondary reporting metrics: Level-1 mean, Level-2 mean, standard deviation,
   alignment-loss trajectory, teacher entropy/diagonal mass, embedding
   standard deviation, and gradient norms.
4. Apply a one-standard-error rule.  If the raw best is not separated from
   another candidate by the pooled evaluation standard error, choose the
   candidate closest to the preregistered defaults instead of exploiting the
   noisy maximum.
5. Report the full candidate table and heatmaps for both backbones.  Distinct
   per-backbone winners are permitted, but the paper must also report whether a
   shared near-optimal region exists.

## Confirmation

The search winner is not yet a headline result.  Confirm the selected
configuration with additional upstream/policy seeds and the complete
evaluation protocol.  The final report must preserve:

- all manifests and code/config/data hashes;
- run-to-device assignment;
- checkpoints and loss trajectories;
- all RVT2-lite task-level results;
- raw best versus one-SE-selected candidate;
- held-out-task and complete-task aggregates.

## Resource assignment

- RTX 5090: first A1 calibration candidate, then every RVT2-lite evaluation;
  when no lite candidate is available, execute one resumable upstream segment.
- RTX 5090 also repeats the R3M
  \((\rho,\epsilon)=(0.25,0.05)\) point.  Its matched H200 run is retained as
  a platform-calibration pair and is not counted as a second independent
  hyperparameter candidate.
- prd03 H200 #1: ViT upstream lane.
- prd03 H200 #2: R3M upstream lane.
- Dataset upload/download is allowed to overlap RTX 5090 training.  If sustained
  upload throughput falls by more than 30% relative to its pre-training window,
  preserve data readiness by reducing local data-loader pressure or pausing the
  upstream run at its next 5k checkpoint.
