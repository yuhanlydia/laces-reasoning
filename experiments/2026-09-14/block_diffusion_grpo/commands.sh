#!/usr/bin/env bash
set -euo pipefail

# Prepare pinned datasets (the preparer resolves the requested Hub revision to an immutable SHA).
python -m laces_posttrain.prepare_math --task gsm8k --output data/gsm8k_block_grpo
python -m laces_posttrain.prepare_math --task math   --output data/math_block_grpo

export CKPT_DIR=/absolute/path/to/step_00030000

# Real-model preflight: require all 16 active blocks, non-flat reward, and nonzero S2 gradient.
GPU=0 DATA_DIR="$PWD/data/gsm8k_block_grpo" MODE=preflight \
  OUTPUT_DIR=results/block_grpo/gsm8k_preflight bash training/run_math_block_grpo.sh
GPU=0 DATA_DIR="$PWD/data/math_block_grpo" MODE=preflight \
  OUTPUT_DIR=results/block_grpo/math_preflight bash training/run_math_block_grpo.sh

# Short matched pilots. Keep settings identical across GSM8K and MATH first.
GPU=0 DATA_DIR="$PWD/data/gsm8k_block_grpo" MODE=train NUM_STEPS=200 \
  GROUP_SIZE=8 MAX_BLOCKS=16 TOKENS_PER_BLOCK=32 ADVANTAGE_MODE=center \
  OUTPUT_DIR=results/block_grpo/gsm8k_pilot bash training/run_math_block_grpo.sh
GPU=1 DATA_DIR="$PWD/data/math_block_grpo" MODE=train NUM_STEPS=200 \
  GROUP_SIZE=8 MAX_BLOCKS=16 TOKENS_PER_BLOCK=32 ADVANTAGE_MODE=center \
  OUTPUT_DIR=results/block_grpo/math_pilot bash training/run_math_block_grpo.sh

# Explicit ablation: legacy std-normalized GRPO. Use a distinct output directory.
GPU=0 DATA_DIR="$PWD/data/gsm8k_block_grpo" MODE=train NUM_STEPS=200 \
  GROUP_SIZE=8 ADVANTAGE_MODE=normalized \
  OUTPUT_DIR=results/block_grpo/gsm8k_normalized_ablation bash training/run_math_block_grpo.sh

# Development evaluation with reasoning-depth frontier at 1/2/4/8/16 blocks.
GPU=0 DATA_DIR="$PWD/data/gsm8k_block_grpo" MODE=eval SPLIT=dev \
  S2_CHECKPOINT=results/block_grpo/gsm8k_pilot/best_dev.pt \
  BLOCK_BUDGETS="1 2 4 8 16" OUTPUT_DIR=results/block_grpo/gsm8k_eval \
  bash training/run_math_block_grpo.sh

# Final test is opt-in only. Do not run until the method/config is frozen.
# GPU=0 DATA_DIR="$PWD/data/gsm8k_block_grpo" MODE=eval SPLIT=test ACKNOWLEDGE_TEST=1 \
#   S2_CHECKPOINT=results/block_grpo/gsm8k_pilot/best_dev.pt \
#   OUTPUT_DIR=results/block_grpo/gsm8k_final bash training/run_math_block_grpo.sh
