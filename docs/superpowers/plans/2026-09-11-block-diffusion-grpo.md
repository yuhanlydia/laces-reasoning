# Block Diffusion GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-guarded GSM8K/MATH preparation, conservative answer verification, causal 16-block generation, block-shaped rewards, and per-block diffusion GRPO for the original LACES S2.

**Architecture:** Keep the existing MMLU-Pro path intact. Add a separate math data/verifier layer and a block policy layer that factors each continuous Gaussian reverse transition across the 16 latent blocks. A dedicated runner freezes S0, dynamic S1, and RWKV, trains only S2, and evaluates raw, parent, and trained arms.

**Tech Stack:** Python 3.10, PyTorch 2.8, Hugging Face `datasets`, pytest, existing LACES/RWKV runtime.

**Spec:** `docs/superpowers/specs/2026-09-11-block-diffusion-grpo-design.md`

## Global Constraints

- Original S0, dynamic S1, and RWKV remain frozen; only original S2 may update.
- Checkpoint shape is 16 latent blocks × 32 downstream tokens; reverse diffusion uses 32 transitions in full runs.
- Official test splits remain sealed until explicit evaluation.
- Dataset revisions and file hashes are pinned; raw rows and checkpoints are not committed.
- Unsupported answers receive zero verifier credit rather than heuristic matches.
- Actions are detached and trained transitions remain stochastic.

---

### Task 1: Mathematical answer verifier

**Files:**
- Create: `laces_posttrain/math_verify.py`
- Test: `tests/posttrain/test_math_verify.py`

**Interfaces:**
- Produces: `extract_final_answer(text: str) -> str | None`, `canonical_numeric(text: str) -> str | None`, `verify_answer(text: str, gold: str, task: str) -> bool`.

- [ ] **Step 1: Write failing tests** for GSM8K integers/decimals/commas/signs, `####` answers, `\boxed{}` fractions, equivalent reduced fractions, final-answer anchoring, and rejection of prose substring matches.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/posttrain/test_math_verify.py -q`; expect import failure for `laces_posttrain.math_verify`.
- [ ] **Step 3: Implement** conservative extraction and exact `fractions.Fraction` comparison. MATH symbolic strings are normalized only for outer whitespace, `\boxed{}`, braces, and canonical numeric fractions; arbitrary algebraic equivalence is rejected.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/posttrain/test_math_verify.py -q`; expect all verifier tests to pass.
- [ ] **Step 5: Commit** `laces_posttrain/math_verify.py tests/posttrain/test_math_verify.py` with message `Add conservative GSM8K and MATH answer verifier`.

### Task 2: Pinned GSM8K and MATH bundles

**Files:**
- Create: `laces_posttrain/prepare_math.py`
- Create: `tests/posttrain/test_math_data.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces JSONL rows with `problem_id`, `problem`, `answer`, `reference_rationale`, `task`, `category`, `source`, and `problem_hash`.
- Produces manifest schema `laces_math_reasoning_v1` with revisions, split counts, hashes, and verifier version.

- [ ] **Step 1: Write failing tests** using local fixture rows to require deterministic train/dev partitioning, cross-split duplicate rejection, sealed-test provenance, rationale retention, and manifest hashes.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/posttrain/test_math_data.py -q`; expect missing module/entrypoint failure.
- [ ] **Step 3: Implement** local-fixture and Hugging Face loading paths. Pin resolved Hub revisions, split official train by normalized problem identity, retain official test as sealed, and never expose test rows through the training reader.
- [ ] **Step 4: Run** verifier/data tests together and verify deterministic manifests.
- [ ] **Step 5: Commit** source, tests, and ignore rules with message `Add pinned math reasoning data bundles`.

### Task 3: Causal latent-block generation

**Files:**
- Modify: `laces_posttrain/native.py`
- Test: `tests/posttrain/test_block_generation.py`

**Interfaces:**
- Produces: `NativeLACES.generate_blocks(prefix, z, *, tokens_per_block, max_blocks, eos_id) -> BlockGeneration`.
- `BlockGeneration` contains token IDs, decoded text, block end offsets, active block mask, and per-block frozen answer potentials supplied by a callback.

- [ ] **Step 1: Write failing fake-runtime tests** proving block `h` is injected before tokens `32h..32h+31`, cache is carried forward, EOS masks later blocks, and raw mode injects no latent state.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/posttrain/test_block_generation.py -q`; expect missing API failure.
- [ ] **Step 3: Refactor the existing `_stream` mechanics minimally** into a block generator without changing option scoring or historical generation behavior.
- [ ] **Step 4: Run** native, option-scoring, and block-generation tests.
- [ ] **Step 5: Commit** with message `Add causal LACES latent-block generation`.

### Task 4: Block-factor diffusion policy and shaped returns

**Files:**
- Create: `laces_posttrain/block_policy.py`
- Test: `tests/posttrain/test_block_policy.py`

**Interfaces:**
- Consumes existing `PolicyConfig`, `Transition`, and S2 denoiser semantics.
- Produces `block_log_ratio(action, new_mean, old_mean, std) -> Tensor[B,H]`, `potential_rewards(phi, exact, format, active_mask)`, `returns_to_go(rewards, active_mask)`, and `block_grpo_update(...)`.

- [ ] **Step 1: Write failing tests** showing block log-ratios equal a manual Normal calculation summed only over latent dimension, potential differences telescope, inactive blocks are zero, returns are reverse cumulative sums, and block advantages center within each group/block.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/posttrain/test_block_policy.py -q`; expect missing module failure.
- [ ] **Step 3: Implement** detached block-factor ratios, center-only and std-normalized advantage modes, PPO clipping, parent KL on the same `[T,H]` scale, flat-block masking, finite checks, and S2-only optimizer updates.
- [ ] **Step 4: Run** block-policy plus existing policy contract tests; verify the existing direct GRPO API is unchanged.
- [ ] **Step 5: Commit** with message `Add block-factor diffusion GRPO objective`.

### Task 5: Math block-GRPO workflow

**Files:**
- Create: `laces_posttrain/run_math_block_grpo.py`
- Create: `training/run_math_block_grpo.sh`
- Test: `tests/posttrain/test_math_block_workflow.py`

**Interfaces:**
- Consumes prepared math bundle, original 30k checkpoint, block policy, verifier, and causal generator.
- Produces run contract, metrics JSONL, dev reports, S2-only checkpoints, and raw/parent/current evaluation reports.

- [ ] **Step 1: Write failing fake-runtime workflow tests** for preflight, one update, checkpoint resume identity, held-out selection, explicit test acknowledgement, and parent/current/raw arms.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/posttrain/test_math_block_workflow.py -q`; expect missing runner failure.
- [ ] **Step 3: Implement** prompt encoding, grouped S2 rollouts, per-block frozen answer potential, exact/format terminal reward, block return-to-go, block GRPO update, logging, evaluation, and atomic checkpointing.
- [ ] **Step 4: Run** workflow tests and `bash -n training/run_math_block_grpo.sh`.
- [ ] **Step 5: Commit** with message `Add math block diffusion GRPO workflow`.

### Task 6: Real data and GPU pilots

**Files:**
- Create: `experiments/2026-09-11/block_diffusion_grpo/README.md`
- Create: `experiments/2026-09-11/block_diffusion_grpo/commands.sh`
- Add generated compact JSON/log reports under the experiment directory; do not add raw data or checkpoints.

**Interfaces:**
- Consumes Tasks 1–5 and the local original 30k model.
- Produces evidence for GSM8K and MATH pilots plus the existing MMLU-Pro evaluation.

- [ ] **Step 1: Run all CPU tests** with `.venv/bin/pytest tests/posttrain tests/test_config_identity.py -q` and save the exact output.
- [ ] **Step 2: Prepare pinned GSM8K and MATH bundles** in distinct local `data/` directories and audit counts/hashes/overlap.
- [ ] **Step 3: Run one-item real-model preflight per dataset** with reduced diffusion transitions; require all pretrained component calls, nonflat block rewards, finite block gradients, and S2-only trainable names.
- [ ] **Step 4: Run short GSM8K and MATH pilots** with identical policy hyperparameters and separate output directories; evaluate fixed development subsets against raw and parent controls.
- [ ] **Step 5: Record MMLU-Pro test status/results** without mixing its custom split into the math training claims.
- [ ] **Step 6: Audit checkpoints** for finite tensors and changed S2 parameters, summarize limitations, run `git diff --check`, and commit compact results with message `Record GSM8K and MATH block diffusion GRPO pilots`.

