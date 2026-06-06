# Configs

`configs/rh20t/pretraining/` contains the active TCC-core experiments.

- `paired_tcc.py`: ResNet18 paired TCC smoke/debug config.
- `paired_tcc_vit.py`: shared ViT-B/16 paired TCC base config.
- `paired_tcc_vit_d4r_*.py`: D4R checkpoint variants.
- `paired_tcc_vit_d4r_in_layernorm.yaml`: D4R-IN, LayerNorm gamma/beta + head.
- `paired_tcc_vit_d4r_in_full.yaml`: D4R-IN, full ViT fine-tuning.

In the paired YAML configs, `paired_max_frames: -1` means the sampled frame
count is controlled by `paired_frame_sample_ratio` instead of
`frame_sampler.num_frames_per_sequence`.
