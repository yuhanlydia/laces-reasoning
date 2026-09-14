# 16-block LACES S2 GRPO: GSM8K and MATH

This experiment operationalizes the 2026-09-11 block-diffusion design. It trains only the
original step-30k LACES `trajectory_dit` (S2) while keeping S0, dynamic S1, state scaling,
and RWKV frozen.

## What is implemented

- Pinned, leakage-guarded GSM8K and Hendrycks-MATH preparation with sealed official tests.
- Conservative answer verification: exact numeric/fraction matching plus strict symbolic
  surface matching for supported MATH answers. Unsupported forms receive zero credit.
- Causal execution of the native 16-block latent plan: write block `h`, generate up to 32
  tokens, carry the RWKV cache into block `h+1`.
- Frozen-model potential shaping with block reward `r_h = Phi_h - Phi_{h-1}` plus terminal
  exact-answer / valid-format bonuses. Boundary potentials use a transient cache copy, so the
  native path scores in O(H) block boundaries rather than replaying O(H^2) previous blocks.
- Per-block diffusion transition ratios, block return-to-go, and default Dr.GRPO-style
  center-only advantages. Standard-deviation-normalized GRPO remains an explicit ablation.
- Defaults: group size 8, LR `3e-7`, KL coefficient `0.05`, 32 reverse transitions,
  16 latent blocks, 32 downstream tokens per block.
- Held-out scaling reports at block budgets 1/2/4/8/16 for raw RWKV, frozen parent LACES,
  and the current S2. Checkpoint selection uses the fixed 16-block development score only.

## Status

The implementation and CPU contract tests were completed in the ChatGPT execution
environment. That environment does not contain the 2.9B checkpoint or CUDA runtime, so no
GSM8K/MATH GPU accuracy result is recorded here. Do not treat implementation completion as a
successful reasoning result. Run the real-model preflights first and only scale training if
block reward variance, score-function gradients, KL, and held-out accuracy are healthy.

See `commands.sh` for the exact preparation, preflight, pilot, and evaluation commands.
