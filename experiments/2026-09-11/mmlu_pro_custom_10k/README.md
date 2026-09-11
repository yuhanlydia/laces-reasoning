# Custom MMLU-Pro split: three 10,000-step runs

The user explicitly authorized combining official validation and test and repartitioning them for training/testing. These experiments are **not official MMLU-Pro benchmark evaluations**. Source implementation: c3d78abb13f0659f41b8354d2187516d8e2fe435.

12,102 input rows yield 11,944 unique normalized question + unordered-option identities, with no conflicting answer-text groups. Seed 42, stratified by subject, approximately 90% training allocation / 10% final test, then 5% of the training allocation reserved for development:

- Training: 10,212
- Development: 538
- Final test: 1,194

The preparation script asserts disjoint identities, unique rows, complete accounting, and compatibility with the training loader. Provenance is retained in original_sources; source and manifest explicitly identify the custom partition. Rationales are removed. Semantic near-duplicates are not ruled out. Existing official-split preparation behavior is unchanged.

## Training schedule

All three routes have a budget of 10,000 optimizer steps, group size 4, diffusion steps 32, LR 1e-6, seed 42, and original 30k S2. A (candidate distillation) and B (direct diffusion GRPO) start concurrently on GPU 0. C (GRPO initialized from A's best development checkpoint) starts after A succeeds. C performs 10,000 additional steps; its initialization is selected by validation, so it need not be A's final step.

The existing trainer samples one training question per step with replacement from the entire training pool. A 10,000-step budget is not an exhaustive epoch or a promise to visit every training example. There are no train/dev limits. Checkpoints are saved every 100 steps and the full 538-example development set is evaluated every 500 steps. Final test is held out from this training scheduler and is not automatically evaluated.

## Operations

Preparation (once): `PYTHONPATH=. .venv/bin/python experiments/2026-09-11/mmlu_pro_custom_10k/prepare_custom.py`

The detached coordinator was launched on 2026-09-11 from `run_10k.py`; it records process IDs, queued tasks, exits, and observed GPU memory. Do not relaunch into existing output directories. Artifacts:

- `outputs_eval/mmlu_pro_custom_10k/status.json`
- `outputs_eval/mmlu_pro_custom_10k/coordinator.log`
- `outputs_eval/mmlu_pro_custom_10k/{distill,grpo_direct,distill_grpo}.log`
- `results/mmlu_pro/custom_10k_{distill,grpo_direct,distill_grpo}_s42/metrics.jsonl`
- Checkpoints `latest.pt` and `best_dev.pt` in each result directory.

This record documents the split and launch, not completion or model improvement. A machine/container shutdown will interrupt these local processes; saved checkpoints support the trainer's explicit resume mechanism. No weights or raw dataset rows are committed.

## Progress snapshot

Captured 2026-09-11T06:41:20.927977+00:00. See `progress/snapshot.json` and per-route metrics and development reports.

- distill: 1500 completed steps. Latest completed validation (dev_step_001000): current 11.34%, raw_rwkv 23.23%, parent_matched 14.31%.
- grpo_direct: 487 completed steps.

Queued: distill_grpo. These are preliminary custom-split results; training remains active as recorded in the snapshot. Final test has not been evaluated.
