# LACES + Dynamic-Basis + Recurrent Latent Reasoning

Extension of the LACES / State-Hijacking RELAY line: replace the fixed 16-basis S1 writer with
input-dependent writable directions, and investigate recurrent latent reasoning for multi-hop
fact fusion.

## Status — audited 2026-09-10

**Dynamic-basis LACES has a trained step-30,000 checkpoint and committed generation outputs.**
The 2.9B rank-32 S1+S2 joint run uses a frozen S0 and batch size 4. `50k` in the run name is
the scheduled training length, not evidence that 50,000 steps have completed. The previous
README's step-10,000 status was stale.

- Current checkpoint used in the committed runs:
  `outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000`.
- [Hugging Face model repository](https://huggingface.co/humanlong/laces-2.9b-dynlowrank-r32-s0frozen-joint-b4-50k-fla03).
  The 30k step is confirmed by the committed runtime records below; this audit did not download
  or validate the Hub checkpoint binaries.
- [30k generation record](experiments/2026-09-09/laces_cfg_sweep/cfg2.json):
  `checkpoint_step=30000`, with actual generated text.
- Reasoning training has already been run, including E5-E8 and the
  [closed-loop recurrent run](experiments/2026-09-09/recurrent_reasoner_closed_loop/README.md).
  It is **not** an unimplemented proposal. However, the current E10 reasoning path bypasses
  the pretrained LACES S0/S1/S2 modules; see the integration audit below.

**Loading the LACES checkpoint is not the same as using the trained LACES inference path.**
The current E10 results do not establish the accuracy of full 30k LACES plus reasoning.

## What's here

- `models/state_hijacking_dit.py` — core model. S1 supports `s1_writer_type`:
  - `fixed` (default): original `sum_k alpha_k(z) B_k`.
  - `dynlowrank` (Plan A): input-dependent `U_l(z) V_l(z)^T`, with `s1_rank=32`.
  - `mixture` (Plan B): bank selection, with `s1_num_banks=8`.
  These are planned-state outputs, including `state_scale`; the generation interface blends
  or replaces recurrent states. Do not silently reinterpret them as additive corrections.
- `train_state_hijacking_dit.py` — joint S1+S2 co-adaptation training entry.
- `scripts/eval/sample_prefix_suffix_trajectory_cfg.py` — full LACES generation:
  S0 prefix encoding, S2 conditional trajectory sampling, S1 state writing, RWKV rendering.
- `scripts/eval/train_recurrent_reasoner.py` — current standalone E10 reasoner/writer training
  on raw RWKV features and raw recurrent-state targets, not yet pretrained-S1 integration.
- `models/recurrent_latent_reasoner.py` — recurrent evidence re-query, pooled state feedback,
  and a separately initialized `SharedDynamicStateWriter`.
- `scripts/eval/train_dynamic_writer*.py`, `train_behavioral*.py`, and
  `e9_decisive_writer.py` — earlier writer training variants and diagnostics.
- `experiments/2026-08-27/`, `experiments/2026-08-28/`, and `experiments/2026-09-09/` —
  notes and results; weights are not committed here.

## Reproduce the recorded 30k generation setting

From the repository root, with the checkpoint and its configured local RWKV/tokenizer files
already available:

```bash
export CKPT_DIR=outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000
python scripts/eval/sample_prefix_suffix_trajectory_cfg.py \
  --ckpt_dir "$CKPT_DIR" \
  --prompt "Question: Why does the sky appear blue? Answer in one clear short paragraph:" \
  --device cuda --output outputs_eval/laces_30k_cfg2.json \
  --diffusion_sampler ddim --steps 1000 --cfg_scale 2.0 \
  --trajectory_s1_mode independent --trajectory_state_blend 0.7 \
  --temperature 0.2 --top_k 50 --top_p 0.9 \
  --repetition_penalty 1.2 --max_new_tokens 512 --seed 42
```

This reproduces the settings of a single-prompt, single-seed generation record, not a
validated best configuration across reasoning tasks. Its readable opening does not remove
later repetition or factual inconsistencies. The generation pipeline also removes certain
control/replacement characters from displayed text; retain token IDs and raw decoded text
when auditing output quality.

## Reasoning integration audit — code at `7798557`

### What E10 actually uses

| Component | Full LACES generation | Current E10 reasoning |
|---|---|---|
| Checkpoint loader | `load_relay_model(...)` | Same loader; Python default points to 30k |
| Frozen RWKV | Prefix features and token rendering | Fact/query features, targets and token rendering |
| S0 encoder | `encode_prefix` calls `model._encode_pooled` | Not called; a new `z0_head` is used |
| Trained S1 writer | `model.predict_states` / `predict_trajectory_states` | Not called; a new `SharedDynamicStateWriter` is used |
| Trained S2 planner | Sampler calls `model.trajectory_dit` | Not called |
| State semantics | Blend/replace with trained planned states | Add a learned cumulative correction to query state |

Sources: [loader](scripts/eval/relay_utils.py),
[E10 entry point](scripts/eval/train_recurrent_reasoner.py),
[reasoner constructor](models/recurrent_latent_reasoner.py),
[full generation entry point](scripts/eval/sample_prefix_suffix_trajectory_cfg.py), and
[trained S1 / cache interfaces](models/state_hijacking_dit.py).

The current E10 computation is:

```text
facts / query
  -> raw model.rwkv_model hidden features
  -> new projections + new z0_head + recurrent GRU
  -> new SharedDynamicStateWriter
  -> cumulative correction C_R
  -> raw RWKV query cache + C_R
  -> raw RWKV answer generation
```

`run()` constructs `RecurrentReasoner(...)` after loading LACES. It passes model dimensions,
not the trained S0/S1 modules or their weights. Setting `_prefix_suffix_trajectory_s2=True`
does not execute S2: the E10 call path never invokes the trajectory sampler. The fresh
writer's layer/head-conditioned hypernetwork also differs from LACES's trained
`s1_trunk`, `s1_u_head`, `s1_v_head`, and `state_scale` implementation. Equal latent width
or rank does not make these the same trained model.

### Other evaluation boundaries

- E10 trains `variable_depth_state_loss` against raw RWKV recurrent-state targets. Answer
  generation is an evaluation path under `no_grad`; the current training loss is not
  answer-token cross-entropy through the pretrained LACES writer.
- `text_concat` calls the raw RWKV `raw_text` helper. It is not a full-LACES baseline.
  `inject_gold` replaces only recurrent matrices through a correction and resets
  `conv_state` / `ffn_state` rather than restoring the donor's complete cache. It is a
  recurrent-state intervention diagnostic, not an exact full-cache oracle. Even a zero
  correction still triggers those cache resets in this injection function.
- The raw-text control joins facts and question with a newline, whereas target construction
  uses a space. Use identical tokenized prompts and preserve the relevant auxiliary cache
  state before claiming oracle equivalence. The effect size of these differences has not
  been measured by this source audit.
- The loader calls `load_state_dict(..., strict=False)` but does not inspect the returned
  missing/unexpected keys. That is a checkpoint-coverage blind spot, not proof that the
  particular 30k weights failed to load. Record checkpoint `step`, writer configuration,
  and component-level key coverage before the next full-model run.

### What the existing results establish

The [2026-09-09 closed-loop record](experiments/2026-09-09/recurrent_reasoner_closed_loop/README.md)
reports 18 training items, 9 test items, 20 epochs, train loss `1.2934`, and relative state MSE
`0.943932` at R=1 versus `0.940925` at R=8. `text_concat`, `inject_gold`, and learned writes
all scored `0/9` on those synthetic prompts.

Those numbers remain unchanged. They describe the standalone/raw-RWKV paths above. The
historical note's phrase "base 30k checkpoint" must not be read as a matched full S0+S2+S1
LACES evaluation. The record does not establish either success or failure of **pretrained
30k LACES + recurrent reasoning**. The previously reported `67 passed, 7 skipped` belongs
to that earlier run; this documentation audit did not rerun GPU training or inference.

### Next integration check — not implemented by this README update

Keep the existing 30k checkpoint. First compare raw RWKV and the full LACES generation path
on identical tasks/prompts, and log actual S0/S1/S2 calls. Then integrate reasoning with the
pretrained latent/state interface rather than silently replacing it with a fresh writer.

Preserve S0's latent coordinates and scaling and S1's blend/replace semantics. A new
32-dimensional GRU output is not automatically a valid S0/S2 latent; directly adding
`model.predict_states(z)` as `C_R` is not a correct drop-in fix. A frozen pretrained S1 plus
a trained latent adapter/refiner is an integration candidate, not a validated result.
S2 should be retained in the full generative baseline; deliberate S2-free variants must be
labeled separately. Do not launch another long reasoning run before this wiring check.

## Reproduce the standalone variable-depth diagnostic

This command runs the existing diagnostic described above, **not the pending full-LACES
integration**. The shell launcher requires an explicit `CKPT_DIR`; it does not silently
select an old champion checkpoint.

```bash
CKPT_DIR=outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000 \
GPU=0 TRAIN_MAX_STEPS=8 EVAL_DEPTHS="1 2 4 8" RANK=32 \
bash training/run_recurrent_reasoner.sh
```

The workspace has 32 dimensions and re-queries token-level facts at every step. `C_r` is a
cumulative correction: inference injects only `C_R`, not the sum of all cumulative writes.
Training unrolls to eight steps by default, supervises intermediate states, and applies
stability losses after the synthetic hop depth. This supports externally selected budgets
`R in {1,2,4,8}`; it does not establish that accuracy improves with more steps.

`R_mode=auto` uses the known synthetic hop count and is an oracle diagnostic.
`EARLY_STOP=1` enables convergence stopping using latent/state changes, not a learned halting
head. Calibrate stopping thresholds on separate validation data. The current summary's
`recommended_default_R` is selected from evaluated test accuracies and is descriptive only;
it must not be reported as an independently validation-selected budget.

## Historical findings (E1-E9, frozen champion + separately trained writers)

These findings and interpretations come from the earlier diagnostic setup, not a completed
30k full-LACES-plus-reasoning comparison. Preserve that scope when citing them.

### 1. Fixed 16-basis limitation in the audited tasks (E1-E4)

- `e16_full ≈ e16_delta ≈ 0.9999`: the 16-basis span captures approximately 0.01% of the gold
  correction `dS* = S_gold - S_context` in the recorded audit.
- E3: the gold direction lifts answer log-probability by approximately +9.7, while the 16
  basis directions are approximately inert. The notes interpret this as a language-steering
  versus reasoning-writable subspace mismatch.

### 2. Oracle low-rank capacity (E2)

- Per-task rank sweep of `dS_l = U_l C_l V_l^T`: rank-16 gives 0% accuracy, rank-32 gives 60%,
  and rank-64 gives 100% in the recorded oracle experiment.
- This tests oracle representational capacity, not learned writer generalization.

### 3. State MSE is not a correctness metric (E9)

- A trained FixedBasis-16/32 writer reaches MSE 6.3 yet 0% accuracy, while the E2 oracle at a
  similar MSE gives 60%. Matching average state error does not establish answer equivalence.

### 4. Writer training and generalization (E5/E6)

- E5: MSE 10.7 and 0% accuracy.
- E6: overfit MSE 8.17, held-out MSE 12.27, and 0% accuracy.
- The notes discuss the small synthetic entity pool and the difficulty of recovering raw
  RWKV state changes as possible explanations, not a proof that all learned writers fail.

### 5. Behavioral supervision attempts (E7/E8)

- Random-direction NES produced approximately zero useful behavioral signal. Gold-direction
  probing was also ineffective when the predicted correction was far from the target.
- Layer weighting improved the reported MSE from approximately 32 to 19.7 without improving
  accuracy. These variants and curriculum runs were already executed.

### 6. Historical gradient-path limitation

- The earlier diagnostic path reported logits with no `grad_fn` with respect to injected
  recurrent state. This applies to the tested implementation and configuration; freezing
  model weights alone is not a general proof that gradients to input states are impossible.
- The current E10 path explicitly uses state matching and no-grad answer evaluation. A new
  answer-loss integration needs its own gradient check rather than relying on that old result.

## Research question and checkpoint separation

Dynamic-basis S1+S2 joint training has reached the recorded 30k checkpoint. The remaining
question is whether reusing its trained latent/state interface improves multi-hop answers,
and whether recurrent reasoning adds value beyond dynamic directions alone. Current E10
results do not answer that comparison because they bypass the trained interface.

The historical fixed-basis champion used for earlier diagnostics is:
`outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000`
(Hugging Face: `SII-Jiaquan/StateDiffRWKV-2.9B-512-pretrained`). It is a historical baseline,
not the current dynamic-basis step-30,000 checkpoint.
