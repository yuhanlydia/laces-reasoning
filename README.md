# LACES: Dynamic Basis and Pretrained-Interface Reasoning

## Current checkpoint

**Dynamic-basis LACES has a trained step-30,000 checkpoint and committed generation records.**
The 2.9B, rank-32 S1+S2 joint run uses frozen S0 and batch size 4. `50k` in its name is the
scheduled training length, not a claim that 50,000 steps finished.

```text
outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000
```

[Hugging Face repository](https://huggingface.co/humanlong/laces-2.9b-dynlowrank-r32-s0frozen-joint-b4-50k-fla03)
· [Actual 30k output record](experiments/2026-09-09/laces_cfg_sweep/cfg2.json)
· [Readability sweep](experiments/2026-09-09/laces_cfg_sweep/README.md)

The output record confirms `checkpoint_step=30000`. It has a readable opening but also later
repetition/inconsistency; it is not a broad reasoning benchmark. Hub binaries and real GPU
accuracy were not revalidated by the integration patch below.

## Corrected reasoning entry

The earlier E10 script loaded LACES but bypassed trained S0/S1/S2 and created a new encoder
and writer. That code and its results remain available as a **standalone diagnostic**.
The default entry now reuses the actual pretrained interface:

```text
facts + question -> frozen RWKV features -> trained S0 -> trained S2 trajectory
                   -> small recurrent latent refiner (R iterations)
                   -> trained S1 + original state_scale/blend -> frozen RWKV answer
```

There is no new U/V writer or new z0 encoder. All LACES weights stay frozen; only the latent
refiner is trained. The native S2 trajectory is retained: H output chunks are distinct from
R reasoning iterations, and each chunk coordinate remains in the original 32-D latent units.
R=0 bypasses refinement; the zero-initialized update is initially an identity at every depth.

**[Full integration, protocols, objectives and running instructions](docs/PRETRAINED_REASONING.md)**

### First run

```bash
python -m pytest tests/test_pretrained_laces_reasoning.py -q

CKPT_DIR=outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000 \
GPU=0 MODE=preflight bash training/run_recurrent_reasoner.sh
```

Preflight checks the training step, all active checkpoint tensors, real S0/S1/S2 calls,
R0 equivalence, and answer signal. Add `RWKV_PATH=/actual/local/backbone` if the checkpoint's
saved local backbone/tokenizer directory does not exist on this machine.

After inspecting preflight outputs, run a small training pilot:

```bash
GPU=0 MODE=train EPOCHS=2 N_TRAIN=8 N_VALIDATION=3 N_TEST=3 \
OBJECTIVE=answer_pg GROUP_SIZE=4 bash training/run_recurrent_reasoner.sh
```

`answer_pg` uses forward-only native answer log-probability rewards for sampled latent
candidates; no state-input backward is needed. `answer_ce` is available ONLY when a real
state-gradient check passes. Neither objective is training-free or already proven successful
on the 30k model. Full-state MSE is no longer the main integrated training objective.

The default `aligned` protocol writes before consuming the boundary anchor token, so the
first answer token is influenced by S1. `legacy` preserves historical sampler timing for
replay, including first-token blindness, and is not accepted for training. The two protocols
must not be mixed in a claimed reasoning comparison. Raw RWKV and LACES R0 receive the same
prefix; no partial-cache injection is labeled an exact oracle.

Outputs are under `outputs_eval/laces_pretrained_reasoner/`: `preflight.json`,
`training_history.jsonl`, `refiner_last.pt`, and `metrics.json`. The latter includes raw RWKV,
full LACES R=0/1/2/4/8, token IDs, raw text, exact match, answer containment, and gold log
probability. Budget selection uses validation only. Successful execution is not proof of
accuracy improvement or test-time-compute scaling.

## Key files

- `models/state_hijacking_dit.py`: unchanged pretrained S0/S1/S2 and native state interface.
- `models/laces_latent_refiner.py`: frozen-interface audit, latent refiner, shared scorer/decoder.
- `scripts/eval/train_laces_reasoner.py`: integrated answer-supervised training and evaluation.
- `scripts/eval/train_recurrent_reasoner.py`: corrected default entry; explicit legacy dispatch.
- `scripts/eval/train_standalone_state_reasoner.py`: preserved E10 state-regression experiment.
- `models/recurrent_latent_reasoner.py`: earlier standalone writer/reasoner; not the new writer.
- `scripts/eval/sample_prefix_suffix_trajectory_cfg.py`: unchanged historical full-LACES sampler.

## Reproduce the historical 30k generation sample

```bash
python scripts/eval/sample_prefix_suffix_trajectory_cfg.py \
  --ckpt_dir outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000 \
  --prompt "Question: Why does the sky appear blue? Answer in one clear short paragraph:" \
  --device cuda --output outputs_eval/laces_30k_cfg2.json \
  --diffusion_sampler ddim --steps 1000 --cfg_scale 2.0 \
  --trajectory_s1_mode independent --trajectory_state_blend 0.7 \
  --temperature 0.2 --top_k 50 --top_p 0.9 \
  --repetition_penalty 1.2 --max_new_tokens 512 --seed 42
```

These are recorded single-prompt settings, not validation-selected reasoning hyperparameters.

## Historical experiments are retained

- [E1-E5 notes and capacity audit](experiments/2026-08-27/): fixed-basis alignment diagnostics
  and oracle rank sweeps. Oracle rank-32 success is not trained-writer generalization.
- [E6-E8 training records](experiments/2026-08-28/NOTES.md): dynamic-writer, behavioral-loss and
  curriculum experiments were already run; they are not merely proposed experiments.
- [Closed-loop E10 record](experiments/2026-09-09/recurrent_reasoner_closed_loop/README.md):
  the recorded 0/9 control and learned-write results came from the standalone/raw-RWKV path,
  not matched full S0+S2+S1 LACES reasoning. Numerical results are unchanged.
- [Original integration audit at 73af3a9](https://github.com/yuhanlydia/laces-reasoning/blob/73af3a9c7c86006f1b79ac79ea41fa97b6d6f545/README.md).

Legacy command: `python scripts/eval/train_recurrent_reasoner.py --legacy_standalone ...`.
Old standalone checkpoints are intentionally rejected by the new refiner loader.

The earlier fixed-basis champion at `outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000`
(Hub: `SII-Jiaquan/StateDiffRWKV-2.9B-512-pretrained`) remains a historical baseline,
not the current dynamic-basis step-30,000 model.

<!-- LACES-MMLU-PRO-S2-POSTTRAIN-V1 -->
## Direct S2 post-training on MMLU-Pro

The direct baseline updates the **existing pretrained `trajectory_dit` (S2)**.
S0, the native S1 dynamic writer/state scale and RWKV remain frozen. It does not
instantiate a new writer, encoder, classifier or recurrent refiner. The earlier
pretrained-interface refiner and historical diagnostics are retained separately.

[Full objectives, probability contracts, data protocol and commands](docs/MMLU_PRO_S2_POSTTRAINING.md)

```bash
python -m pytest tests/posttrain -q
python -m laces_posttrain.prepare --output data/mmlu_pro_pilot
export DATA_DIR="$PWD/data/mmlu_pro_pilot"
export CKPT_DIR=/absolute/path/to/step_00030000
GPU=0 MODE=preflight OUTPUT_DIR=results/mmlu_pro/preflight \
  bash training/run_mmlu_pro_s2.sh
```

`OBJECTIVE=distill` defaults to reward-weighted **candidate self-distillation**;
`DISTILL_SOURCE=rationale` instead uses provided training-only teacher traces.
`OBJECTIVE=grpo` uses detached diffusion transitions, clipped joint-density ratios
and a frozen S2 reference. `INIT_S2=.../best_dev.pt` enables distillation -> GRPO.

MMLU-Pro has no official training split. Default preparation makes a small,
explicitly labeled **validation-adaptation pilot** (normally 56 train / 14 dev),
not the official five-shot setting. Official test is never used for optimization
or checkpoint selection. Independent training data can be imported with provenance
and normalized problem-overlap checks. Only an explicit final-eval invocation may
read test labels. CPU tests establish software contracts, not actual 30k GPU
accuracy or a successful reproduction of BDH-CQ.
<!-- /LACES-MMLU-PRO-S2-POSTTRAIN-V1 -->

<!-- LACES-BLOCK-GRPO-V1 -->
## 16-block latent reasoning on GSM8K / MATH

MMLU-Pro direct-option experiments showed that the S2 policy receives a real answer-reward
gradient, but they do not exercise the full 16-block latent trajectory. The block-GRPO
workflow therefore uses GSM8K and Hendrycks MATH to train the **existing S2 only** while
keeping S0, dynamic S1/state scaling, and RWKV frozen.

```text
prompt -> frozen S0 -> S2 samples z1..z16
       -> [S1(z1) -> <=32 tokens] -> ... -> [S1(z16) -> <=32 tokens]
       -> exact/verifiable answer reward
```

Block reward uses frozen-model potential differences `Phi_h - Phi_{h-1}` plus terminal
exact-answer / format bonuses. PPO ratios and KL remain factorized by latent block, and the
default advantage is Dr.GRPO-style group centering without reward-standard-deviation
normalization. Native boundary potentials are scored from transient cache copies, avoiding
quadratic replay of all earlier blocks. Evaluation reports `Acc@1/2/4/8/16` for raw RWKV,
frozen parent LACES, and the trained S2; sealed test is never used to choose a budget or checkpoint.

```bash
python -m laces_posttrain.prepare_math --task gsm8k --output data/gsm8k_block_grpo
python -m laces_posttrain.prepare_math --task math --output data/math_block_grpo
export CKPT_DIR=/absolute/path/to/step_00030000
GPU=0 DATA_DIR="$PWD/data/gsm8k_block_grpo" MODE=preflight \
  OUTPUT_DIR=results/block_grpo/gsm8k_preflight bash training/run_math_block_grpo.sh
```

See [the implementation design](docs/superpowers/specs/2026-09-11-block-diffusion-grpo-design.md)
and [the runnable experiment commands](experiments/2026-09-14/block_diffusion_grpo/commands.sh).
No GPU accuracy claim is made by the code commit itself.
<!-- /LACES-BLOCK-GRPO-V1 -->
