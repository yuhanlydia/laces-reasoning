# ARC-AGI-1 Recurrent Latent Reasoner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the frozen-RWKV LACES recurrent reasoner on all 400 ARC-AGI-1 training tasks plus deterministic augmentation and evaluate exact grid accuracy on an isolated 400-task evaluation split.

**Architecture:** Official ARC grids are serialized for one-time frozen RWKV feature extraction. A compact cache stores projected evidence/query features and pooled RWKV state. The 32-dimensional closed-loop reasoner and an ARC grid decoder are trained end-to-end with shape and cell cross-entropy at recurrent budgets 1/2/4/8.

**Tech Stack:** Python 3.10, PyTorch 2.13, Hugging Face Transformers/FLA RWKV, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-arc-agi-1-recurrent-reasoner-design.md`

## Global Constraints

- Use all 400 official ARC-AGI-1 training tasks in the final run.
- Keep all 400 official evaluation tasks isolated from training and checkpoint selection.
- Expand training to at least 3,200 episodes with the eight dihedral transforms.
- Keep the 2.9B RWKV backbone frozen.
- Keep recurrent latent width at 32 and evaluate budgets 1, 2, 4, and 8.
- Do not truncate long ARC tasks to 512 tokens; process serialized input recurrently in chunks.
- Run on the local RTX 3090 and save resumable checkpoints.

---

### Task 1: Official ARC data, validation, transforms, and serialization

**Files:**
- Create: `models/arc_data.py`
- Create: `scripts/preprocess/prepare_arc_agi1.py`
- Create: `tests/test_arc_data.py`

**Interfaces:**
- Produces: `ArcPair`, `ArcTask`, `load_arc_split(path)`, `validate_task(task)`, `transform_task(task, transform_id)`, `serialize_arc_context(task, query_index)`, and `prepare_official_arc(root, source_url, source_ref)`.
- Data output: `preprocessed_data/arc_agi1/{training,evaluation}/*.json` plus `manifest.json`.

- [ ] **Step 1: Write failing tests for all eight invertible transforms, malformed-grid rejection, exact serialization round-trip, task-ID split isolation, and counts.**

```python
def test_all_dihedral_transforms_are_invertible(sample_task):
    for transform_id in range(8):
        transformed = transform_task(sample_task, transform_id)
        assert inverse_transform_task(transformed, transform_id) == sample_task

def test_context_serialization_round_trips(sample_task):
    text = serialize_arc_context(sample_task, 0)
    assert parse_arc_context(text) == (sample_task.train, sample_task.test[0].input)
```

- [ ] **Step 2: Run `python3 -m pytest -q tests/test_arc_data.py` and verify failures are missing imports/functions.**
- [ ] **Step 3: Implement immutable task dataclasses, strict colors 0..9 and dimensions 1..30 validation, D4 transforms, compact row-delimited serialization, official tarball download, source-ref manifest, and atomic extraction.**
- [ ] **Step 4: Run the tests, then run `prepare_arc_agi1.py --verify-only` and assert exactly 400 training and 400 evaluation files with disjoint IDs.**
- [ ] **Step 5: Commit data-pipeline code and tests.**

### Task 2: Frozen RWKV feature-cache builder

**Files:**
- Create: `models/arc_feature_cache.py`
- Create: `scripts/preprocess/cache_arc_rwkv_features.py`
- Create: `tests/test_arc_feature_cache.py`
- Modify: `models/recurrent_latent_reasoner.py`

**Interfaces:**
- Consumes: `ArcTask`, `serialize_arc_context`, the 30k LACES checkpoint, and `_cache_layer_state`.
- Produces: `ArcFeatureRecord(task_id, transform_id, query_index, evidence, query, base_state_features, target_grid)` and `RecurrentReasoner.forward(..., base_state_features=...)`.

- [ ] **Step 1: Write failing tests for deterministic 256-D projection, chunked-versus-single-pass cache equality, linear pooled-state composition, cache schema/version rejection, and direct pre-pooled reasoner input.**
- [ ] **Step 2: Run the focused tests and verify they fail for absent implementations.**
- [ ] **Step 3: Implement a seeded orthonormal hidden projection, sequential/chunked RWKV ingestion, 4x4 per-layer/head state pooling, and sharded `.pt` records with checksums. Add `base_state_features` to the reasoner so full matrices are unnecessary during ARC training.**
- [ ] **Step 4: Run focused tests and a two-task GPU cache smoke. Confirm the RWKV parameters all have `requires_grad=False` and peak memory stays below 24 GiB.**
- [ ] **Step 5: Commit feature-cache and reasoner-interface changes.**

### Task 3: ARC grid decoder and differentiable recurrent objective

**Files:**
- Create: `models/arc_grid_adapter.py`
- Create: `tests/test_arc_grid_adapter.py`
- Modify: `models/recurrent_latent_reasoner.py`

**Interfaces:**
- Consumes: the final or intermediate reasoner latent `[B,32]`, projected query features `[B,T,256]`, and pooled state features.
- Produces: `ArcDecoderOutput(row_logits, col_logits, cell_logits)` where shapes are `[B,30]`, `[B,30]`, and `[B,30,30,10]`; `arc_grid_loss(output, targets)`; `decode_grid(output)`.

- [ ] **Step 1: Write failing tests for output shapes, target masking outside dimensions, exact loss on a perfect prediction, invalid-dimension rejection, and gradients reaching the recurrent cell and state writer at depth 2.**
- [ ] **Step 2: Run focused tests and verify failure.**
- [ ] **Step 3: Implement query pooling, latent/state conditioning, row/column heads, a coordinate-conditioned 30x30 color head, masked cell CE, shape CE, stability loss, and state-magnitude regularization.**
- [ ] **Step 4: Run focused tests and confirm finite gradients for every trainable module.**
- [ ] **Step 5: Commit the decoder and objective.**

### Task 4: Staged training, resume, and exact ARC evaluation

**Files:**
- Create: `scripts/eval/train_arc_recurrent_reasoner.py`
- Create: `scripts/eval/eval_arc_recurrent_reasoner.py`
- Create: `models/arc_metrics.py`
- Create: `tests/test_arc_training_pipeline.py`

**Interfaces:**
- Consumes: cached `ArcFeatureRecord` shards, `RecurrentReasoner`, and `ArcGridDecoder`.
- Produces: resumable `.pt` checkpoints, JSONL training history, prediction JSON, and summary metrics for pair exact, whole-task exact, pass@1/pass@2, shape, cell accuracy, runtime, memory, and R=1/2/4/8.

- [ ] **Step 1: Write failing tests for deterministic 360/40 task splitting, no evaluation IDs in training, checkpoint round-trip including RNG/optimizer/scheduler/scaler, exact metrics, pass@2, and invalid prediction handling.**
- [ ] **Step 2: Run focused tests and verify failure.**
- [ ] **Step 3: Implement AMP training, gradient accumulation, periodic checkpointing, emergency non-finite checkpointing, resume, deep supervision, early stopping on the 40-task development split, and standalone evaluation.**
- [ ] **Step 4: Run focused tests, then overfit eight official training tasks. Require at least 95% exact pair accuracy before continuing.**
- [ ] **Step 5: Commit training and evaluation code.**

### Task 5: Full 400-task run and reproducible launcher

**Files:**
- Create: `training/run_arc_recurrent_reasoner.sh`
- Modify: `README.md`
- Create: `experiments/2026-09-09/arc_agi1_recurrent_reasoner/README.md`

**Interfaces:**
- Consumes: Tasks 1-4 command-line entry points.
- Produces: feature cache for 3,200+ training episodes, development checkpoint/metrics, final all-400 checkpoint, and isolated 400-task evaluation report.

- [ ] **Step 1: Add a launcher dry-run test that validates paths, split counts, checkpoint cadence, batch/accumulation settings, and output directories without loading the model.**
- [ ] **Step 2: Implement the launcher with `BATCH_SIZE=1`, automatic gradient accumulation, budgets `1 2 4 8`, periodic saves, and resume defaults suitable for one RTX 3090.**
- [ ] **Step 3: Build all 3,200 augmented feature records and record preprocessing throughput, disk usage, and checksum.**
- [ ] **Step 4: Run the 360/40 development training. Continue to the final run only if eight-task overfit reaches 95% and R=8 solves at least two more development pairs than R=1.**
- [ ] **Step 5: Train from scratch on all 400 training tasks, evaluate once on all 400 evaluation tasks, and record exact metrics and example predictions.**
- [ ] **Step 6: Run `python3 -m pytest -q`, `git diff --check`, document results and limitations, commit, and push `main`.**
