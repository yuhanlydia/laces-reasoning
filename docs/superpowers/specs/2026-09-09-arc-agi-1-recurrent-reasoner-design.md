# ARC-AGI-1 recurrent latent reasoner design

## Goal

Train and evaluate the LACES recurrent latent reasoner on the official ARC-AGI-1
task format. The final training run must use all 400 public training tasks and at
least 3,200 augmented episodes. The 400 public evaluation tasks remain isolated
until the final evaluation.

The experiment keeps the 2.9B RWKV backbone frozen. Trainable components are the
ARC adapter, the 32-dimensional recurrent reasoner, the dynamic state writer, and
the grid decoder. The main metric is exact output-grid accuracy, rather than text
substring accuracy or recurrent-state MSE.

## Data source and splits

Use the official `fchollet/ARC-AGI` ARC-AGI-1 files and record the source commit in
the prepared-data manifest. Verify exactly 400 JSON files under `data/training`
and 400 under `data/evaluation`. Each task contains a list of demonstration
input/output grids and one or more query grids with released target outputs.

Development proceeds in two phases:

1. A deterministic 360/40 split of the official training tasks is used for model
   and hyperparameter development. Task IDs, rather than augmented episodes, are
   the split unit.
2. After the recipe is frozen, train from a fresh initialization on all 400
   official training tasks. Evaluate once on all 400 public evaluation tasks.

No evaluation task, target, feature, or score is used for checkpoint selection.

## Augmentation

Apply the eight dihedral transforms to every task consistently across every
demonstration and query grid. This creates 3,200 episodes from the 400 official
training tasks. Demonstration order is shuffled online. Optional color
permutations preserve color 0 by default and are enabled only after an ablation
shows that they improve held-out training-task validation.

All transforms must be invertible and tested by applying the inverse transform.
The same transform is applied to inputs and outputs. Evaluation tasks receive no
augmentation other than optional test-time ensembling reported separately.

## ARC representation and frozen RWKV feature cache

Serialize grid cells with ten verified single-token color IDs plus explicit row,
grid, input/output, demonstration, and query delimiters. Demonstrations are fed
sequentially so the RWKV recurrent cache evolves after every example. Long tasks
are processed recurrently in chunks; they are never truncated to 512 tokens.

Run the frozen 30k LACES/RWKV checkpoint once during preprocessing. Store:

- a fixed seeded 256-dimensional projection of token hidden states for evidence
  and query features;
- a 4x4 adaptive-pooled feature grid for every layer and head of the final query
  recurrent state;
- task ID, augmentation transform, original grid shapes, token counts, and source
  checksum.

The full recurrent matrices are not cached for every augmented episode. Pooling
is linear, so the recurrent loop can combine the cached pooled base state with
the pooled dynamic correction. This keeps the cache small enough for the local
machine while preserving layer, head, and spatial association structure.

## Model

The existing recurrent reasoner remains the computation core:

```text
evidence features, query features, pooled base state
        -> z0 in R^32
        -> dynamic state write C0
        -> pooled(base state + C0)
        -> z1 -> C1 -> ... -> zR -> CR
```

At every step, `z` re-queries demonstration features and reads the state produced
by the preceding write. Writes are cumulative corrections relative to the base
query state; they are not summed across reasoning steps.

An ARC grid decoder consumes the current latent, query-grid features, and current
pooled state. It has two output branches:

- row and column classifiers for dimensions 1 through 30;
- a 30x30 field of ten-way color logits, masked outside the target dimensions.

The decoder supplies a differentiable task loss even though the installed FLA
RWKV kernel does not backpropagate answer loss through an injected recurrent
cache. The state writer still affects later latent steps through the differentiable
pooled readback.

## Objective and reasoning budgets

Train with deep supervision at budgets `R = 1, 2, 4, 8`. The total objective is:

```text
L = L_cell_ce + 0.25 * L_shape_ce
    + 0.05 * L_post_solution_stability
    + 0.01 * L_state_magnitude
```

`L_cell_ce` covers target cells only. `L_shape_ce` predicts exact output rows and
columns. Stability discourages oscillation after the final supervised step.
State-magnitude regularization prevents unbounded corrections. The previous
full-state MSE is retained only as an optional diagnostic on a small uncached
subset because earlier experiments showed that it is poorly aligned with answer
correctness.

Train with mixed precision, gradient accumulation, and batch size selected to use
the RTX 3090 without out-of-memory retries. Save checkpoints and validation metrics
at regular intervals. Resume must restore model, optimizer, scheduler, scaler,
epoch, and random-number states.

## Evaluation

Report the following separately for the 40-task development split and the final
400-task public evaluation split:

- exact accuracy per test pair;
- exact whole-task accuracy, requiring every test pair to be correct;
- pass@1 and pass@2;
- accuracy at R=1, R=2, R=4, and R=8;
- output-shape accuracy and cell accuracy for failed exact outputs;
- mean runtime and peak GPU memory per task.

The first capability gate is overfitting a small fixed subset of eight training
tasks to at least 95% exact pair accuracy. The second gate is a measurable depth
benefit on the 40-task development split: R=8 must exceed R=1 by at least two
exactly solved pairs. Failure of either gate stops the final 400-task run and
produces diagnostics instead of a misleading benchmark number.

## Components

- `scripts/preprocess/prepare_arc_agi1.py`: download/validate official data,
  generate augmentations, serialize grids, and build the frozen feature cache.
- `models/arc_grid_adapter.py`: ARC token metadata, pooled-state input adapter,
  and grid decoder.
- `scripts/eval/train_arc_recurrent_reasoner.py`: staged training, checkpointing,
  resume, and development evaluation.
- `scripts/eval/eval_arc_recurrent_reasoner.py`: isolated 400-task evaluation and
  exact ARC metrics.
- `training/run_arc_recurrent_reasoner.sh`: reproducible local launch settings.
- Unit tests cover data validation, transforms, inverse transforms, masking,
  shape loss, exact scoring, state feedback, and split isolation.

## Failure handling

Preprocessing fails on malformed colors, ragged grids, dimensions outside 1..30,
missing targets, duplicate task IDs, wrong split counts, or serialization that
does not round-trip exactly. Training fails on non-finite losses and writes an
emergency checkpoint. Evaluation records invalid shapes and colors as incorrect
predictions without crashing the remaining run.

The source manifest and deterministic seeds make every generated episode and
reported metric reproducible.
