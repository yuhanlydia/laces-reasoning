# scripts/ - Launch, Sampling, and Evaluation Scripts

## Overview

Scripts support the current RELAY/state-hijacking line centered on `train_state_hijacking_dit.py`.

## Where To Look

| Task | Location | Notes |
|------|----------|-------|
| Train launch helpers | `training/unconditional/`, `training/conditional/prefix_suffix/`, `training/cluster/` | S0/S1/S2 launch wrappers |
| Active evaluation | `scripts/eval/` | Cola-DLM, PPL, multichoice, trajectory grids |
| Active sampling | `scripts/eval/sample_prefix_suffix_cfg.py`, `scripts/eval/sample_prefix_suffix_trajectory_cfg.py` | Prefix/suffix CFG samplers |
| Preprocess | `scripts/preprocess/` | Dataset/token preprocessing helpers |
| Tools | `scripts/tools/` | Offline/download/benchmark helpers |

## Notes

- Prefer `README.md` commands for current runs.
- Do not add generated eval outputs to commits unless explicitly requested.
- Old failed Block-Causal, thinker/CoT, and RWKV diffusion-denoiser scripts were removed from the active tree.
