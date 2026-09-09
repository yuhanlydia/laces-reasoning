# ARC-AGI-1 recurrent latent reasoner

This run trains the 32-dimensional LACES recurrent reasoner and dynamic rank-32
state writer on official ARC-AGI-1 tasks. The RWKV 2.9B backbone and the existing
30k joint LACES checkpoint remain frozen. A one-time cache stores 256-dimensional
token projections and 4x4 pooled recurrent state per layer and head.

The official 400 training tasks are split deterministically into 360 optimization
tasks and 40 development tasks. D4 transforms are used only inside each selected
training task; task IDs never cross the split. The 400 official evaluation tasks
remain isolated until final evaluation.

Run the stages with:

```bash
bash training/run_arc_recurrent_reasoner.sh prepare
bash training/run_arc_recurrent_reasoner.sh cache
bash training/run_arc_recurrent_reasoner.sh overfit
bash training/run_arc_recurrent_reasoner.sh train
bash training/run_arc_recurrent_reasoner.sh eval
```

The answer readout receives only the dynamic writer correction, so it cannot bypass
the writing interface through a direct latent, query, or base-state shortcut. During
training each sample draws one recurrent budget from R=1/2/4/8/16. Pooled feedback
is computed exactly from pooled U/V factors; full 64x64 state matrices are materialized
only when a final native-state injection is requested. Evaluation runs and times every
budget independently.

The eight-task gate requires at least 95% exact output-pair accuracy. Full training
does not start through the launcher unless that file-backed gate passes. Training
uses recurrent budgets 1, 2, 4, 8, and 16 and writes `best.pt`, `last.pt`, periodic
checkpoints, JSONL history, and a JSON summary with peak GPU memory.

The first live gate run uses:

```text
feature cache: preprocessed_data/arc_agi1_features_overfit8
output:        outputs_arc/overfit8_r32
seed:          20260909
episodes:      64 (8 tasks x 8 D4 transforms)
```
