# Downstream Workspace

All downstream-related scripts, logs, checkpoints, and evaluation summaries live
under this directory.

```text
downstream/
  analysis/      # TCC-core representation/retrieval analysis outputs
  rvt2_lite/     # RVT2-lite downstream policy probe
```

`downstream/analysis/` is for upstream checkpoint analysis such as level-1 loss
curves and level-2 retrieval.

`downstream/rvt2_lite/runs/` is for RVT2-lite policy checkpoints, rollout
logs, and downstream CSV summaries.
