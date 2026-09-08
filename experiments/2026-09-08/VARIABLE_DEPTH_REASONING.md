# E11 — Variable-depth recurrent latent reasoning

## Question

Can one trained 32-D recurrent latent reasoner trade inference-time compute for reasoning
quality without using the oracle hop count?

## Model

For token-level fact features `H_f` and question features `H_q`:

```math
q_r = Q_\rho([z_r;\bar h_q]),\qquad
c_r = \operatorname{CrossAttn}(q_r,H_f,H_f),
```

```math
z_{r+1}=\operatorname{LN}(\operatorname{GRU}_\rho(c_r,z_r)),
\qquad z_r\in\mathbb R^{32}.
```

A shared layer/head-conditioned hypernetwork produces both low-rank factors for every
RWKV head:

```math
C_{\ell,a}^{(r)}
=U_{\ell,a}(z_r)V_{\ell,a}(z_r)^\top.
```

`C^(r)` is cumulative relative to one question-context state. Inference injects only
`C^(R)`; it never sums full corrections across reasoning steps.

## Training length

Default:

```text
train_max_steps = 8
```

Every sample is unrolled for eight shared-parameter transitions. For an `L`-hop sample,
steps `r <= L` match the cumulative oracle state after `r` facts. Steps `r > L` match the
same final target and receive latent/state stability regularization. Thus the same
checkpoint is explicitly trained at all budgets from 1 to 8.

## Inference length

Primary operating points:

```text
LOW=1, MEDIUM=2, STANDARD=4, HIGH=8
```

The main experiment reports the complete accuracy/latency curve. The default deployment
budget is the smallest `R` whose validation accuracy is within 1 percentage point of
`R=8`. `R_mode=auto` is retained only as an oracle-hop diagnostic. Optional convergence
stopping is threshold-based, not learned halting.

## Commands

Smoke test:

```bash
CKPT_DIR=/path/to/dynamic-basis-checkpoint \
N_TRAIN=4 N_TEST=2 EPOCHS=2 TRAIN_MAX_STEPS=4 EVAL_DEPTHS="1 2 4" \
OUTPUT=results/capacity_audit/e11_smoke.json \
bash training/run_recurrent_reasoner.sh
```

Pilot:

```bash
CKPT_DIR=/path/to/dynamic-basis-checkpoint \
N_TRAIN=200 N_TEST=50 EPOCHS=100 TRAIN_MAX_STEPS=8 \
EVAL_DEPTHS="1 2 4 8" RANK=32 \
OUTPUT=results/capacity_audit/e11_variable_depth.json \
bash training/run_recurrent_reasoner.sh
```

Convergence-stopping diagnostic:

```bash
CKPT_DIR=/path/to/dynamic-basis-checkpoint EARLY_STOP=1 \
OUTPUT=results/capacity_audit/e11_early_stop.json \
bash training/run_recurrent_reasoner.sh
```

## Required checks before a long run

1. `inject_gold` must remain a meaningful ceiling under the aligned token-position path.
2. Same facts with different questions must produce different `z_0` and `C_R`.
3. `C_R`, rather than `sum_r C_r`, must be injected.
4. Gradients must reach the recurrent transition and shared dynamic writer.
5. Report `Acc@1`, `Acc@2`, `Acc@4`, `Acc@8`, relative state MSE at each depth, and
   mean stopping depth when convergence stopping is enabled.

## Gate

Do not claim test-time compute scaling unless both hold on held-out entities:

```math
\operatorname{Acc}@4 > \operatorname{Acc}@1
```

and at least one deeper budget improves gold-answer behavior without degrading the
`inject_gold` protocol. If accuracy remains flat while state MSE falls, the remaining
failure is behavioral alignment, not reasoning-length selection.
