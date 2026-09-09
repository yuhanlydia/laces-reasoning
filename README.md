# LACES + Dynamic-Basis + Recurrent Latent Reasoning

## ARC-AGI-1 recurrent latent reasoning

The ARC pipeline freezes the RWKV 2.9B backbone and the 30k joint LACES model,
caches projected token/state features once, then trains a 32-dimensional recurrent
latent reasoner, dynamic state writer, and direct grid decoder with supervision at
R=1/2/4/8. It uses all 400 official ARC-AGI-1 training tasks with a deterministic
360/40 train/development split and keeps the 400 official evaluation tasks isolated.

See
[`experiments/2026-09-09/arc_agi1_recurrent_reasoner/README.md`](experiments/2026-09-09/arc_agi1_recurrent_reasoner/README.md)
for the staged commands, capability gates, checkpoints, and evaluation outputs.

Extension of the LACES / State-Hijacking RELAY line: replace the fixed 16-basis S1 writer with
input-dependent writable directions, and add a trained recurrent latent reasoner for multi-hop
fact fusion.

## Status

**S1+S2 joint training is running.** The dynamic-basis writers (`dynlowrank`, `mixture`)
are implemented and tested. The current 2.9B rank-32 run uses a frozen S0, batch size 4,
and a 50,000-step schedule; its step-10,000 checkpoint is available on
[Hugging Face](https://huggingface.co/humanlong/laces-2.9b-dynlowrank-r32-s0frozen-joint-b4-50k-fla03).
The recurrent latent reasoner remains a separate follow-up training stage.

## What's here

- `models/state_hijacking_dit.py` — core model. S1 writer supports `s1_writer_type`:
  - `fixed` (default): original `sum_k alpha_k(z) B_k` (16 basis per layer)
  - `dynlowrank` (Plan A): `dS_l = U_l(z) V_l(z)^T` (dynamic low-rank, `s1_rank=32`)
  - `mixture` (Plan B): `S = sum_m g_m(z) sum_k a_{m,k}(z) B^{(m)}_k` (bank selection, `s1_num_banks=8`)
- `train_state_hijacking_dit.py` — joint S1+S2 co-adapt training entry (champion recipe).
- `scripts/eval/` — diagnostic + writer/reasoner training scripts (E1-E10):
  - `e9_decisive_writer.py` — FixedBasis-16/32 vs DynamicUV-r32 (fixed 32-D z input)
  - `train_recurrent_reasoner.py` — variable-budget recurrent latent reasoner with per-step evidence re-query and cumulative state writes
  - `train_dynamic_writer*.py`, `train_behavioral*.py` — writer training variants (E5-E8)
  - `diag_capacity_audit.py`, `diag_rank_sweep.py`, `diag_behavioral_sensitivity.py` — capacity diagnostics (E1-E4)
- `experiments/2026-08-27/`, `experiments/2026-08-28/` — full notes + results (checkpoints excluded).


## Variable-depth reasoning training

The recurrent reasoner is a post-training stage on top of a trained LACES checkpoint. It keeps
its workspace in 32 dimensions but re-queries token-level facts at every compute step:

```text
H_facts, H_query
      -> z0 in R^32
      -> z1 -> z2 -> ... -> zR
      -> C1, C2, ... -> CR
```

Here `C_r` is the **cumulative** RWKV-state correction after step `r`. Inference injects only
`C_R`; it does not sum a full state write at every step. Training unrolls to `R_max=8` by
default, supervises every intermediate budget, and trains steps after the oracle hop depth to
remain at a stable solved state. This makes one checkpoint usable at external budgets
`R in {1,2,4,8}`.

Run:

```bash
CKPT_DIR=/path/to/trained-dynamic-basis-checkpoint \
GPU=0 TRAIN_MAX_STEPS=8 EVAL_DEPTHS="1 2 4 8" RANK=32 \
bash training/run_recurrent_reasoner.sh
```

The default evaluation mode is `sweep`, which reports accuracy and state error separately for
`R=1,2,4,8`. `R_mode=auto` uses the known synthetic hop count and is an **oracle diagnostic**,
not a deployable adaptive policy. Optional `EARLY_STOP=1` enables validation-calibrated
convergence stopping based on latent and cumulative-state changes; there is not yet a learned
halting head.

## Key findings (E1-E10, on frozen champion + lightweight writers)

### 1. The fixed 16-basis is a hard bottleneck (E1-E4)

- `e16_full ≈ e16_delta ≈ 0.9999`: the 16-basis span captures ~0.01% of the gold correction
  `dS* = S_gold - S_context`. The basis span is nearly orthogonal to the multi-hop reasoning correction.
- E3 behavioral sensitivity: the gold direction lifts answer log-prob by ~+9.7, but all 16 basis
  directions are ~inert (≈ random). The basis is a *language-steering* subspace, not a
  *reasoning-writable* subspace.

### 2. A wider/dynamic writer CAN carry the correction — in the oracle (E2)

- Per-task rank sweep of `dS_l = U_l C_l V_l^T`: rank-16 → 0% acc, **rank-32 → 60%**, rank-64 → 100%.
- So `behavioral capacity ≠ 16 fixed directions`; a rank-32 low-rank writer is the right order.

### 3. But state-matching MSE is misaligned with answer correctness (E9)

- A *trained* FixedBasis-16/32 writer reaches MSE 6.3 (≈ the rank-32 floor) yet **0% acc**,
  while the E2 oracle at the same MSE gives 60%.
- Interpretation: the answer depends on a low-rank *behavioral* subspace; MSE residual is
  concentrated exactly in those behaviorally-critical directions. `MSE floor ≠ answer`.

### 4. Trained writers overfit, don't generalize (E5/E6)

- E5 (mean-pool input + fixed per-layer dirs): MSE 10.7, 0% acc.
- E6 (token cross-attn + per-head dynamic dirs): overfit MSE 8.17, held-out 12.27, 0% acc.
- The `facts → dS*` mapping is essentially the RWKV's own recurrent computation; a small writer
  can't generalize it from the tiny synthetic entity pool (18 subjects × 16 links × 16 places).

### 5. Zero-order behavioral loss is ineffective (E7/E8)

- The reward `r(dS) = log p(gold | S + dS)` is flat/sharp: random-direction NES gives ~0 signal
  (E7), and even the gold-direction signal is ~0 when the writer is far from the correction (E8).
- FLA's RWKV-7 kernel does **not** backprop through the recurrent state, so no `∇_dS L_answer`.
- Layer weighting fixed training stability (MSE 32→19.7) but still 0% acc; curriculum didn't help.

### 6. Hard constraint: no answer-loss gradient through the frozen RWKV

- `logits` have no `grad_fn` w.r.t. the injected recurrent state (verified). All supervision
  must be state-matching (smooth but misaligned) or zero-order (flat).

## Implication for the dynamic basis (this repo's actual contribution)

The fixed linear writable interface (`alpha B`) is the bottleneck, not the 32-D latent. The two
writers here (`dynlowrank`, `mixture`) make the writable directions input-dependent. **The open
question this repo is meant to answer:** does a *trained* dynamic-basis S1 (in full S1+S2 joint
training) break through the 0% acc wall that the fixed basis hit — and does it need the
recurrent reasoner on top, or is dynamic directions alone enough?

## Checkpoint

Champion (frozen backbone + S0): `outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000`
(hosted on HuggingFace `SII-Jiaquan/StateDiffRWKV-2.9B-512-pretrained`, not in this repo).
## MMLU-Pro recurrent reasoning

The language-native recurrent reasoning experiment uses frozen RWKV features and
base answer logits on MMLU-Pro, then trains only the recurrent reasoner and dynamic
state writer residual head:

```bash
bash training/run_mmlu_pro_recurrent_reasoner.sh
```

See `experiments/2026-09-09/mmlu_pro_recurrent_reasoner/README.md` for the split
semantics and evaluation protocol.
