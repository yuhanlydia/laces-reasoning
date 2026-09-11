# Direct post-training of the original LACES S2 on MMLU-Pro

Audited upstream: `yuhanlydia/laces-reasoning`, main commit
`82c456765b9f9fa4819ab7a53ebd3e2ae38727a2`. This is an **additive** implementation.
It does not overwrite previous experiments, the pretrained-interface refiner, the
old GRPO probes, or any 30k weights. It introduces no new writer, GRU or classifier.

## 1. What is being trained

```text
question + all available choices (no gold answer or rationale)
    -> original frozen RWKV -> original frozen S0 condition c
    -> original trainable trajectory_dit (S2), Z in [1,H,d]
    -> original frozen S1 + state_scale + planned-state blending
    -> original frozen RWKV vocabulary head
```

The default parent is the user's **30k dynamic-basis LACES**. Load a local
`step_00030000/model.pt` using its matching local RWKV model and tokenizer.
The run name's `50k` is the planned schedule, not a completed-step claim.
Only parameters of the existing `model.trajectory_dit` enter the optimizer.
All active S0/S1/S2 tensors are checked against the loaded checkpoint before any
S2 update; missing weights, wrong step and a silently substituted writer fail.
The original model remains untouched on disk. Output checkpoints contain only
S2, its frozen reference S2, optimizer/RNG state, settings and provenance.

Three experiments are supported:

1. **Direct S2 distillation.** Default is reward-weighted *candidate self-distillation*.
2. **Direct S2-GRPO.** A clipped, group-relative diffusion-transition policy update.
3. **Distillation -> GRPO.** Initialize S2 from a distillation output; reset optimizer
   and use the initialized S2 as the new frozen KL reference. The unchanged original
   30k S2 is still the matched evaluation baseline.

An optional `DISTILL_SOURCE=rationale` uses supplied training-only teacher traces.
It is not the same objective as candidate self-distillation; report them separately.

## 2. What BDH-CQ does, and what this implementation does not claim

The referenced paper is *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*
(arXiv:2608.09888v1). It evaluates public ARC-AGI-1 and controlled ARC-like/ConceptARC
cases. It does not report MMLU-Pro. Its public equations distinguish recurrent
context memory from query-specific iterative latent computation. It describes
ARC-style supervision and different reasoning efforts, but its complete internal
training recipe and detailed architecture are proprietary.

We borrow the motivation for internal computation, **not an undocumented claim that
BDH-CQ uses GRPO or this distillation method**. This repository addition is direct
post-training of the existing LACES diffusion planner, not a BDH-CQ reproduction.

Sources:
- https://arxiv.org/html/2608.09888v1 (sections 3-4, 5-7)
- https://arxiv.org/abs/2305.13301 (diffusion denoising as a decision process / DDPO)
- https://arxiv.org/abs/2402.03300 (GRPO reference)
- https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/README.md
- https://github.com/TIGER-AI-Lab/MMLU-Pro

## 3. Data protocol: there is no official MMLU-Pro training split

The official dataset card lists **70 validation and 12,032 test items**. Questions
have up to ten options (some fewer). The adapter uses the actual options, A..J,
not four hardcoded labels. Dataset revision is resolved to a Hub commit and saved.

### Default: explicitly labeled validation-adaptation pilot

For each category, reserve one validation item for development and use the others
for a small training pilot. With the official five examples per 14 categories,
this is **56 train / 14 dev**, while all 12,032 test items remain sealed.
This is NOT the standard five-shot unadapted evaluation protocol. It is much too
small to establish a broad data-scaling result. It is enough to test wiring and
whether the trained interface supplies a useful reward gradient.

### Independent training data

Supply `--external-train-jsonl ... --source-id ...`. The official validation set
is then development-only, and test remains untouched. The independent training
file must contain question, options, category and answer/index. A teacher text
and teacher source are optional. No external model/API calls or paid generation
are performed automatically.

```json
{"question_id":"train-001","question":"Your independently sourced training question", "options":["choice one","choice two","choice three"],"answer":"B","category":"math","teacher_text":"A short verified explanation. The answer is (B).","teacher_source":"your-teacher-run/version"}
```

Preparation rejects train/dev/test problem overlap after normalization, independent
of option order. The hash includes the normalized question AND unordered choices:
generic question stems with different choices are different problems. This is an
exact normalized check, not a guarantee against paraphrases, semantic overlap or
contamination already in pretrained weights. Source provenance still matters.
Test rationale fields are removed from the sealed file. File hashes are checked
on every run. Test bytes may be hashed for integrity, but the training path never
parses test questions/labels or uses them as rewards. Do not fabricate source IDs
for copies or derivatives of test questions.

## 4. Objectives and exact probability convention

### Candidate self-distillation (default, no external teacher needed)

For one training question, sample G fresh plans from the current original S2,
score each through original S1/RWKV, and compute

`w_i = softmax((reward_i - max(reward)) / temperature)`.

Treat the sampled native plan as a detached target and apply the original epsilon
prediction objective:

`loss = sum_i w_i E_{t,eps} ||eps_psi(sqrt(a_t) Z_i + sqrt(1-a_t) eps, t, c) - eps||^2`.

This is reward-weighted self-distillation, **not teacher knowledge transfer, not
exact trajectory maximum likelihood, and not GRPO**. Uniform weights under a flat
reward only imitate existing candidates; they are not evidence of reasoning progress.
Logs report reward variance and effective candidate count.

### Optional teacher-rationale distillation

The supplied `teacher_text`/official training-side `cot_content` is encoded by the
original frozen S0 using isolated chunks, matching the independent checkpoint's
native coordinate convention. S2 denoising is supervised on those latents. Unused
chunks are masked. An explicit teacher final letter that contradicts the training
label fails validation. This does not verify every statement in the rationale.
Missing traces or traces exceeding H*chunk_size fail **before training**, rather
than silently truncating them or inventing a teacher trace. The short last chunk
is encoded at its actual length. This is native-coordinate latent target imitation;
its decoding quality must be measured, not assumed.

### Diffusion-transition GRPO

The actual rollout policy is a diagonal Gaussian at EVERY reverse step, including
the terminal step:

`p_psi(Z_{t-1}|Z_t,c) = Normal(mu_psi(Z_t,t,c), sigma_t^2 I)`.

The mean uses the ancestral-DDIM formula. `sigma_t` has a positive configurable
floor (`MIN_STD=0.02` initially). The floor, including terminal stochasticity, is an
explicit policy modification; it is not claimed to reproduce deterministic DDIM.
There is no unscored, parameter-dependent final denoising map.

Rollouts are under no_grad. States/actions/old means are stored detached on CPU.
At optimization, replay a stored state/action under the new S2:

`log_ratio = sum_latent_dims [log p_new(action|state) - log p_old(action|state)]`.

This is the JOINT transition ratio, not a geometric-mean pseudo-probability.
Use group-standardized terminal rewards, a clipped PPO-style surrogate, and an
exact Gaussian reference KL (the latter normalized per coordinate and explicitly
reported as such). The default is one update per collected group. More inner
updates are configurable but a large joint log-ratio fails instead of silently
changing the probability definition. Reduce LR/inner updates when that gate fires.

Critically, `action.detach()` and detached observed states prevent the old bug:
`action = mean + sigma*noise` inside `log_prob(action)` without detach cancels the
score-function derivative with respect to mean. A regression test reproduces that
zero gradient and checks the corrected nonzero gradient. No reward gradient through
RWKV's injected state is required. Flat reward groups skip the optimizer, including
weight decay; they are counted, not described as successful learning.

### S2 training kernel and precision

The native BiRWKV denoiser selects a CUDA training/inference kernel using its
`training` flag. Both rollout and replay use the **training kernel** while dropout
and BatchNorm updates are disabled. Using `.eval()` indiscriminately can select an
inference-only path and disconnect S2 gradients. Optimizer/master parameters are
FP32; CUDA denoiser calls use BF16 autocast; transition means are FP32 and joint
log-density differences FP64. Original frozen S0/S1/RWKV dtypes are unchanged.
Only the user's real CUDA preflight can validate the production FLA kernel.

## 5. Rewards, answers and evaluation

**Default direct mode:** the prompt contains the question and all choices, but no
answer or rationale. Score full label continuations ` A` ... ` J` with the native
RWKV vocabulary head. Multi-token labels are handled by complete joint likelihoods;
no unverified assumption that each option is a single token. Tokenizer boundary
merges fail with an actionable delimiter error. The default training reward is
`log_softmax(option_joint_logprobs)[gold]`. Option argmax accuracy is also available.
No new answer classifier is introduced.

**Optional CoT mode:** `ANSWER_MODE=cot REWARD=accuracy`. Generate real text and parse
an explicit final answer (or a bare letter). Do not search for an arbitrary letter
inside an explanation. Invalid parses count as wrong; raw text and token IDs are
saved. Output length must fit the original latent horizon. CoT has sparse rewards
and is more expensive, so the direct mode is the initial interface diagnostic.

Important scope: direct short-answer scoring normally consumes only the first S1
chunk, although S2 jointly processes the whole trajectory. It does NOT prove that
all H latent chunks execute an H-step reasoning chain. Use the optional full CoT
path and appropriate ablations to test later-chunk use. Neither diffusion steps
nor output-chunk count is automatically an independently learned thinking depth.

Evaluation reports micro/macro and per-category accuracy, invalid outputs, raw
option log-likelihoods or generated IDs/text, plus matched timing. Multiple latent
samples are optionally aggregated by average option probabilities / CoT majority
vote, never by test-gold best-of-N. The first sample remains the primary accuracy;
ensemble accuracy is separate. Prompt encoding time is included in each arm's
reported per-item time (even though encoding is shared experimentally).

Arms are current S2, the unchanged original S2 under the same sampling/decode budget,
and raw RWKV. The GRPO KL anchor may be a distilled warm start, but the matched
parent control is still original 30k. A separate preflight DDIM1000 control checks
sampling degradation versus the shorter stochastic training policy. Do not count
more rollout samples or different samplers as free improvements.

The default is **zero-shot direct likelihood**, not the official five-shot CoT
leaderboard protocol. An evaluation at a different sampler, blend, answer mode or
diffusion schedule from training is rejected; use the recorded settings explicitly.
Only dev selects `best_dev.pt`. Test requires an explicit acknowledgment and never
selects checkpoint, hyperparameters or latent candidates with its labels.

## 6. Commands

Use the same environment as the trained checkpoint. Do NOT casually upgrade
PyTorch/FLA/Transformers. Additional data-preparation dependencies are listed in
`requirements-mmlu-pro.txt`; install them only if missing. CPU tests need pytest,
PyTorch and OmegaConf, but no Transformers download or 2.9B checkpoint.

```bash
python -m pytest tests/posttrain -q
python -m laces_posttrain.prepare --output data/mmlu_pro_pilot
export DATA_DIR="$PWD/data/mmlu_pro_pilot"
export CKPT_DIR=/absolute/path/to/step_00030000
# Only if the backbone/tokenizer path stored in model.pt no longer exists:
# export RWKV_PATH=/absolute/path/to/the-matching-rwkv-backbone
```

Offline preparation, using separately saved official split JSONL files:

```bash
python -m laces_posttrain.prepare --output data/mmlu_pro_pilot \
  --official-validation-jsonl /path/validation.jsonl \
  --official-test-jsonl /path/test.jsonl
```

Independent training source:

```bash
python -m laces_posttrain.prepare --output data/mmlu_pro_external \
  --external-train-jsonl /path/train.jsonl --source-id your-source/version
```

### A. Real 30k preflight

```bash
GPU=0 MODE=preflight GROUP_SIZE=4 DIFFUSION_STEPS=32 NATIVE_CONTROL_STEPS=1000 \
OUTPUT_DIR=results/mmlu_pro/preflight bash training/run_mmlu_pro_s2.sh
```

Inspect S0/S1/S2 call counts, score-gradient norm, reward_std, repeated-score
agreement, and raw/short-policy/DDIM1000 option scores. No GPU quality result has
been produced by the supplied CPU verification. A two-step diffusion run is an
engineering smoke only; it is not evidence of language quality.

### B. Short candidate-distillation smoke

```bash
GPU=0 MODE=train OBJECTIVE=distill DISTILL_SOURCE=candidates \
NUM_STEPS=2 TRAIN_LIMIT=4 DEV_LIMIT=2 EVAL_EVERY=2 SAVE_EVERY=1 \
OUTPUT_DIR=results/mmlu_pro/distill_smoke bash training/run_mmlu_pro_s2.sh
```

### C. Distillation pilot (after preflight)

```bash
GPU=0 MODE=train OBJECTIVE=distill NUM_STEPS=200 GROUP_SIZE=4 LR=1e-6 \
OUTPUT_DIR=results/mmlu_pro/distill_s42 bash training/run_mmlu_pro_s2.sh
```

For actual teacher-trace distillation instead, use `DISTILL_SOURCE=rationale` with
short attributed traces in the training file. This is a separate experimental arm.

### D. Direct GRPO and distillation -> GRPO

```bash
# Direct GRPO from original 30k:
GPU=0 MODE=train OBJECTIVE=grpo NUM_STEPS=200 GROUP_SIZE=4 LR=1e-6 \
OUTPUT_DIR=results/mmlu_pro/grpo_direct_s42 bash training/run_mmlu_pro_s2.sh

# GRPO from the dev-selected distillation S2, not a new writer:
GPU=0 MODE=train OBJECTIVE=grpo NUM_STEPS=200 GROUP_SIZE=4 LR=1e-6 \
INIT_S2=results/mmlu_pro/distill_s42/best_dev.pt \
OUTPUT_DIR=results/mmlu_pro/distill_grpo_s42 bash training/run_mmlu_pro_s2.sh
```

Each new run uses a new output directory. `RESUME=.../latest.pt` resumes the same
objective/data/settings with optimizer and RNG restored; `NUM_STEPS` is the total
step count, not additional steps. `INIT_S2` explicitly starts a new stage. Legacy
standalone reasoner/refiner checkpoints are rejected.

### E. Locked final test evaluation

First fix the checkpoint and protocol using dev. Then:

```bash
GPU=0 MODE=eval SPLIT=test ACKNOWLEDGE_TEST=1 \
S2_CHECKPOINT=results/mmlu_pro/distill_grpo_s42/best_dev.pt \
OUTPUT_DIR=results/mmlu_pro/final_test_s42 bash training/run_mmlu_pro_s2.sh
```

Match all training sampling/decode settings. Full test can be slow; `TEST_LIMIT`
is only a debug subset and must be labeled as such. Do not repeatedly optimize
against it. Use `SEED=42`, `43`, `44` for distinct training runs, not to search test
results. The default validation-adaptation pilot has a very small dev set: report
uncertainty and do not infer general scaling or publishable superiority from it.

Optional CoT is a separate run beginning with:
`ANSWER_MODE=cot REWARD=accuracy MAX_NEW_TOKENS=256`. Keep these settings at eval.
A historical fixed-writer ablation requires its own checkpoint and explicit
`EXPECTED_WRITER=fixed EXPECTED_STEP=26000`; it is not automatically a fair causal
basis-only comparison if the parent training histories differ.

## 7. Verification and remaining limits

Tests cover detached policy gradients, exact joint ratios, positive terminal
variance, training-kernel/dropout mode, reward-induced S2 changes, flat groups,
10-choice scoring, token boundaries, matched raw behavior, data guards, teacher
masks/length checks, strict parent loads, S2-only saving, warm starts, original
parent controls, exact interrupted-versus-continuous resume and shell quoting.
Native integration tests instantiate the real LACES S0/S1/S2 at tiny dimensions
with both DiT/MLP and BiRWKV/variational-encoder configurations, using a small
synthetic recurrent renderer. They do not substitute a fake denoiser
for all model tests, but they cannot validate the real 2.9B FLA backend or accuracy.

No generated success numbers are included. Previous fixed-basis observations and
old policy-gradient errors motivate this test; they do not prove that the new
30k dynamic interface must improve with RL. The result to measure is held-out
answer accuracy through the existing trained state interface, not state MSE.
