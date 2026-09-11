# Direct original-S2 signal pilot — 2026-09-11

Source: main `522b78912ecae4fb90426dd72cc0a1319f6a2587` plus the accompanying data-preparation fix. One A100-SXM4-40GB, PyTorch 2.8.0, Transformers 4.57.6, FLA 0.3.2. Parent: original LACES step 30000; backbone and checkpoint provenance are recorded in the sibling sampling_and_reasoning report.

## What was verified

- Direct-S2 preflight passed: S0/S1/S2 calls 1/7/2258, S2 score gradient norm 6777.04169, reward standard deviation 0.01526764, repeated-score maximum difference 0.
- Distillation, direct diffusion-GRPO, and distillation followed by GRPO each completed two optimizer steps. Only original S2 is trainable: 91,203,104 parameters across 473 tensors. All 473 tensors changed versus each run's initial reference; saved S2 values are finite.
- All three runs scored 0/2 on the selected development examples. This is an execution/gradient check, not evidence of improved reasoning or a benchmark result. The test split was not evaluated. Training reward statistics concern different sampled questions and should not be interpreted as a learning curve.
- CPU validation: 63 tests passed (`tests/posttrain/` and `tests/test_config_identity.py`); Python compilation and launcher shell syntax passed.

## Settings and GPU use

Seed 42, learning rate 1e-6, group size 4, 32 stochastic diffusion transitions, native control 1000 steps, 2 training steps per run, evaluation every 2 steps, development limit 2. No new writer, classifier, or GRU. `run_pilots.py` records the exact commands and local paths.

Distillation and direct GRPO ran concurrently on GPU 0; the third run initialized from distillation's best_dev.pt after distillation completed. Peak sampled aggregate GPU memory was 22,925 MiB of 39,936 MiB; sampled utilization reached 100%. Sampling occurred every five seconds, so these are observed values rather than exact instantaneous maxima. This does not establish a throughput speedup relative to sequential execution. Checkpoints remain on the GPU machine and are not included in Git.

## Dataset preparation fix

The official MMLU-Pro test split has 12,032 rows, 11,874 unique normalized questions, and 158 duplicate rows. The previous preparation code rejected that official split. The fix preserves every official test row in its original order while using unique question hashes for cross-split leakage checks. Train/dev duplicates remain rejected. Two regression tests cover duplicate preservation and overlap rejection; their initial failures are included.

Prepared split counts: 56 train / 14 dev / 12,032 sealed test. Dataset revision: `b189ec765aa7ed75c8acfea42df31fdae71f97be`. Manifest and hashes are included; raw dataset rows are not. These results do not cover the proposed 200-step experiments or establish research novelty.
