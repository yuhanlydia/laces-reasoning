# Reasoning through the existing 30k LACES checkpoint

## What changed

The old E10 script loaded LACES but then bypassed its S0 encoder, S2 prior and S1 writer.
It learned a new `z0_head` and a new `SharedDynamicStateWriter` against raw-state MSE.
That exact historical script is retained as `scripts/eval/train_standalone_state_reasoner.py`.
Its results/checkpoints have not been deleted or relabeled as full-LACES results.

The default entry `scripts/eval/train_recurrent_reasoner.py` now calls
`train_laces_reasoner.py`. It uses this explicit path:

```text
prefix (facts + question; no gold answer)
  -> frozen RWKV hidden sequence
  -> EXISTING S0._encode_pooled -> z_prefix
  -> EXISTING S2 trajectory sampler -> Z_initial [1,H,32]
  -> trainable, shared recurrent LATENT refiner, R steps
  -> EXISTING S1.predict_states for each output chunk
  -> EXISTING state_scale + blend/replace operator
  -> frozen RWKV native answer tokens
```

All S0/S1/S2/backbone weights remain frozen. No new basis, state decoder, U/V head or
initial-latent encoder is created. The checkpoint's actual S1 architecture is retained,
including its sharing of V across heads within a layer; it is not silently replaced by
the previous standalone per-head hypernetwork.

This is a latent-refinement integration, not a claim that the old standalone architecture
was already trained in S0 coordinates. The old pooled-state-feedback loop remains in the
legacy diagnostic. The new loop re-queries the full prefix features and modifies S2's
native latent trajectory. It does not execute RWKV as a verifier at each internal step.

H is the existing output-chunk horizon (16 in the recorded 30k setup). R is a separately
chosen reasoning budget. Each coordinate is 32-D, but a full trajectory has H*32 scalar
coordinates. No claim of a single 32-D total workspace is made for the chunked path.
The shared update head starts at zero, so any number of untrained refinement steps is
exactly the identity in native latent coordinates. Normalization is internal to the update
network; it is never applied to replace/renormalize the pretrained latent.

## Cache-boundary correctness

`LACESRuntime` uses one token-level path for teacher-forced scoring and free generation.

- `aligned` (new default): prefill all but the last prefix token, write S1's planned state,
  then consume the last prefix token exactly once. The first answer logit therefore depends
  on S1. At later boundaries, write before consuming the preceding generated anchor token.
- `legacy`: reproduce the historical sampler's timing, including its first-token blindness:
  the logit used immediately after a state write was computed before that write. This is
  available for replay, but training with this protocol is rejected.

The two protocols are intentionally labeled. New aligned R=0 output is a full-LACES
baseline under the corrected boundary protocol, NOT a promise of byte equality with the
historical 512-token example. CPU tests additionally verify that legacy R=0 reproduces
`sample_prefix_suffix_trajectory_cfg.generate` using the same latents and greedy settings.

For both protocols, R=0 bypasses refinement entirely. Nonzero state blends use the existing
S1 blend operator (and its auxiliary-cache policy); planned states are never blindly added
as if they were residuals. `blend=0` is a true no-op and does not clear auxiliary caches.
Raw RWKV and LACES baselines now receive exactly the same prefix IDs. No partial-cache
`inject_gold` result is mislabeled as an exact full-cache oracle.

## Checkpoint and load audit

The new runtime requires `step=30000` by default, independent trajectory S1, and dynlowrank
S1. `--expected_step` can explicitly select another training step. It validates every
required S0/S1/S2 parameter/buffer against `checkpoint['trainable_state']`, after the
loader's dtype conversion. Missing keys, wrong shapes, or unequal loaded tensors are fatal.
Frozen raw RWKV keys are intentionally not required in the LACES adapter checkpoint.

The audit records component key counts, a fingerprint of the active tensors, the configured
backbone/tokenizer path, and actual forward-hook counts for S0/S1/S2. This does not hash the
backbone files themselves. Keep the backbone files and tokenizer at that path unchanged.
`--rwkv_path` overrides their local location without editing checkpoint files.

The new refiner checkpoint stores only refiner/optimizer state and provenance. Resuming
against different active LACES weights or a different backbone path fails. Old E10 `.pt`
files fail with an explicit standalone-format error; their newly learned coordinates are
not silently assumed compatible with S0. S0/S1/S2 are neither discarded nor retrained.

## Training objective (explicit new post-training stage)

State-MSE imitation is not the default objective of this integrated path.

`answer_pg` (default for the historically non-differentiable FLA inference path):

1. Sample a training budget R from `--train_depths` and compute the refined native latent.
2. Sample G Gaussian candidates around the active answer-chunk coordinates. Noise is in
   latent units, not in the millions of recurrent-state entries. Other chunks are unchanged.
3. Score every candidate through the pretrained S1 and frozen RWKV using mean gold-token
   log probability. Rewards and sampled actions are detached.
4. Optimize group-centered/standardized score-function loss `-mean(A * log_prob(action))`,
   plus a small normalized latent-drift penalty. Only the refiner parameters are updated.

This is answer-supervised policy optimization, not training-free reasoning, not a claim of
unbiased gradient estimation after advantage normalization, and not evidence it improves
this checkpoint yet. It is distinct from the previous state-space random-direction NES.
Flat-reward examples are explicitly counted/skipped; an all-flat epoch aborts rather than
reporting drift-only updates as reasoning training. No ground-truth answer is used to choose
inference candidates: inference uses the deterministic refined mean, with no weight update.

`answer_ce`: direct native answer cross-entropy. Preflight must establish a finite, nonzero
state-input gradient (using a short answer probe); training aborts on detached loss. This
option does NOT replace FLA with a differentiable implementation and does not assume the
historical kernel limitation has been solved. There is no silent CE-to-PG fallback.

PyTorch references for the two distinct gradient mechanisms:
- https://docs.pytorch.org/docs/stable/distributions.html#score-function
- https://docs.pytorch.org/docs/stable/notes/autograd.html#setting-requires-grad

## Run order

Run CPU contracts first (they use real small LACES S0/S1/S2 modules and a tiny test renderer):

```bash
python -m pytest tests/test_pretrained_laces_reasoning.py -q
```

GPU preflight on the existing checkpoint:

```bash
CKPT_DIR=outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000 \
GPU=0 MODE=preflight \
bash training/run_recurrent_reasoner.sh
```

If its original local RWKV path is absent, add `RWKV_PATH=/actual/local/rwkv-directory`.
Inspect `outputs_eval/laces_pretrained_reasoner/preflight.json`: required modules must have
actual calls, R0 identity must pass, raw/full-LACES texts must be inspected, and the selected
objective must have a usable signal. A readable example alone is not a reasoning gate.

Small training pilot (not a benchmark-sized experiment):

```bash
GPU=0 MODE=train EPOCHS=2 N_TRAIN=8 N_VALIDATION=3 N_TEST=3 \
OBJECTIVE=answer_pg GROUP_SIZE=4 \
bash training/run_recurrent_reasoner.sh
```

Native S0/S2 prefix representations are cached once per provenance/prefix/seed on disk, not
recomputed for every candidate or epoch. No giant full-state teacher target cache is built.
The recorded DDIM1000/CFG2/blend0.7 defaults come from the 30k readability run, not from a
validation sweep on reasoning. A reduced PLAN_STEPS is a separate configuration, not a
reproduction of that record. This launcher is single-GPU with one example/group per update.
No 16GB or 24GB memory guarantee is made until the real checkpoint has been smoke-tested.

For actual training data, supply disjoint JSONL files with either schema:

```json
{"prefix":"Facts ... Question: ... Answer:","answer":"the verified answer"}
{"facts":["fact one","fact two"],"question":"Question: ... Answer:","answer":"the verified answer"}
```

```bash
MODE=train EPOCHS=10 TRAIN_JSONL=data/train.jsonl \
VALIDATION_JSONL=data/validation.jsonl TEST_JSONL=data/test.jsonl \
OUTPUT_DIR=outputs_eval/laces_reasoner_formal \
bash training/run_recurrent_reasoner.sh
```

Prefix and answer are tokenized separately; when needed a leading answer space is inserted.
There is no silent prefix or answer truncation. Set `--max_prefix_tokens` explicitly for
longer inputs; answers must fit the existing S2 horizon. Exact prompt overlap across splits
is rejected; independent semantic/template splits remain the dataset author's responsibility.

Resume training: set `RESUME=.../refiner_last.pt`, `MODE=train`, and EPOCHS to the TOTAL desired
count. Evaluate that refiner: set the same RESUME with `MODE=eval`. Keep sampler/blend/data
settings fixed for comparisons. R is externally selected; there is no learned halting here.

Outputs:
- `preflight.json`: provenance, real component calls, R0 equality, signal checks, raw texts.
- `training_history.jsonl`: task loss, drift, sampled depth, gradient norm, reward variance.
- `refiner_last.pt`: resumable NEW refiner only; frozen LACES is referenced, not copied.
- `metrics.json`: separate raw RWKV and full LACES R=0/1/2/4/8 results, token IDs and uncleaned
  text, exact match, boundary-aware answer containment, and native answer log probability.
  Default-budget selection uses validation only; test is not used to tune the budget.

## Legacy reproduction and evidence limits

Use `python scripts/eval/train_recurrent_reasoner.py --legacy_standalone ...OLD_FLAGS...` to
run the preserved standalone diagnostic. Old flags such as `--r_s`, `--R_mode auto`, or
`--train_max_steps` are not silently accepted by the corrected interface. Rank is inherited
from the loaded S1; new training/evaluation budgets use `--train_depths`/`--eval_depths`.

This patch fixes integration and provides runnable post-training with tests. It does not
claim that the real 30k GPU run has passed, that older 0/9 results were all caused by one bug,
or that R=8 now outperforms R=1. Compare the new full-LACES R0 baseline and trained refiner
on matched settings before making any such claim.
