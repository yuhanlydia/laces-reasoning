# LACES + Dynamic-Basis + Recurrent Latent Reasoning

Extension of the LACES / State-Hijacking RELAY line: replace the fixed 16-basis S1 writer with
input-dependent writable directions, and add a trained recurrent latent reasoner for multi-hop
fact fusion.

## What's here

- `models/state_hijacking_dit.py` — core model. S1 writer supports `s1_writer_type`:
  - `fixed` (default): original `sum_k alpha_k(z) B_k` (16 basis per layer)
  - `dynlowrank` (Plan A): `dS_l = U_l(z) V_l(z)^T` (dynamic low-rank, `s1_rank=32`)
  - `mixture` (Plan B): `S = sum_m g_m(z) sum_k a_{m,k}(z) B^{(m)}_k` (bank selection, `s1_num_banks=8`)
- `train_state_hijacking_dit.py` — joint S1+S2 co-adapt training entry (champion recipe).
- `scripts/eval/` — diagnostic + writer/reasoner training scripts:
  - `e9_decisive_writer.py` — FixedBasis-16/32 vs DynamicUV-r32 (fixed 32-D z input)
  - `train_recurrent_reasoner.py` — recurrent latent reasoner (step-wise state trajectory distillation)
  - `train_dynamic_writer*.py`, `train_behavioral*.py` — writer training variants (E5-E8)
  - `diag_capacity_audit.py`, `diag_rank_sweep.py`, `diag_behavioral_sensitivity.py` — capacity diagnostics (E1-E4)
- `experiments/2026-08-27/`, `experiments/2026-08-28/` — experiment notes + results (checkpoints excluded).

## Key findings (see `experiments/2026-08-2*/NOTES.md`)

- The fixed 16-basis span is nearly orthogonal to the multi-hop reasoning correction (`e16_delta ~ 1`).
- A rank-32 low-rank writer can carry the correction in the oracle (per-task) setting, but a
  trained writer overfits and does not generalize to held-out entity combinations.
- State-matching MSE is misaligned with answer correctness: near-floor MSE (6.3) still gives 0% acc.
- Zero-order behavioral loss is ineffective (reward is flat except along the gold direction).

## Checkpoint

Champion (frozen backbone + S0): `outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000`
(hosted on HuggingFace `SII-Jiaquan/StateDiffRWKV-2.9B-512-pretrained`, not in this repo).
