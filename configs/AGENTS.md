# configs/ - Active Hydra Configs

## Overview

Active configs are limited to RELAY/state-hijacking DiT VAE32 runs for RWKV-7 `0.4B`, `2.9B`, and `13.3B` backbones.

## Where To Look

| Task | Pattern |
|------|---------|
| Single-z unconditional | `rwkv_relay_*_state_hijack_dit_vae32.yaml` |
| Single-z prefix/suffix | `rwkv_relay_*_state_hijack_dit_vae32_prefix_suffix_s*.yaml` |
| Trajectory unconditional | `rwkv_relay_*_state_hijack_dit_vae32_traj32x16.yaml` |
| Trajectory prefix/suffix | `rwkv_relay_*_state_hijack_dit_vae32_prefix_suffix_traj32x16_*.yaml` |

## Notes

- Use `README.md` as the source of truth for current commands.
- Old `rwkv_mmdit`, bidirectional, block-causal, and ablation configs were removed from the active config set.
- Keep new configs scoped to `train_state_hijacking_dit.py` unless explicitly restoring a legacy path.
