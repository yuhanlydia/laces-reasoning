#!/bin/bash
# Plan B 13.3B (multi-GPU DDP): s0 (VAE) → s1 (alpha) → s2 (DiT) → sample
# Default 8×GPU; override with NPROC_PER_NODE=4 for local test
# Requires H200 (141GB) for 13.3B backbone — A100 (80GB) will OOM
set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Ensure working directory is project root (cluster jobs may start elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIG="rwkv_relay_13.3B_state_hijack_dit_vae32"
NAME="test-v6-13.3B"
GPU_COUNT="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
EFF_BS=$((GPU_COUNT * NNODES))

S0_BS="${S0_BS:-1}"
S1_BS="${S1_BS:-1}"
S2_BS="${S2_BS:-4}"
S0_WORKERS="${S0_WORKERS:-4}"
S1_WORKERS="${S1_WORKERS:-4}"
S2_WORKERS="${S2_WORKERS:-4}"

VISIBLE_GPU_COUNT=1
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  VISIBLE_GPU_COUNT=$(( $(tr -cd ',' <<< "$CUDA_VISIBLE_DEVICES" | wc -c) + 1 ))
fi
if [ "$GPU_COUNT" -gt "$VISIBLE_GPU_COUNT" ]; then
  echo "ERROR: NPROC_PER_NODE=${GPU_COUNT} exceeds visible CUDA devices (${CUDA_VISIBLE_DEVICES})" >&2
  exit 1
fi

S0_NAME="${NAME}-s0"
S1_NAME="${NAME}-s1"
S2_NAME="${NAME}-s2"

S0_CKPT="outputs_relay/${S0_NAME}/step_00050000"
S1_CKPT="outputs_relay/${S1_NAME}/step_00050000"
S2_CKPT="outputs_relay/${S2_NAME}/step_00150000"

TORCHRUN_ARGS="--nnodes=${NNODES} --nproc_per_node=${GPU_COUNT} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"

echo "========================================="
echo " Plan B 13.3B (${NNODES}×${GPU_COUNT}=${EFF_BS} GPU DDP): s0 → s1 → s2 → sample"
echo " Config : $CONFIG"
echo " Name   : $NAME"
echo " Node   : ${NODE_RANK}/${NNODES}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo " Effective batch size: ${GPU_COUNT} × ${NNODES} = ${EFF_BS}"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo " NCCL  : DEBUG=${NCCL_DEBUG} ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING} TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT}"
echo " OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo " Stage settings:"
echo "   S0: batch=${S0_BS} workers=${S0_WORKERS} effective_batch=$((S0_BS * EFF_BS))"
echo "   S1: batch=${S1_BS} workers=${S1_WORKERS} effective_batch=$((S1_BS * EFF_BS))"
echo "   S2: batch=${S2_BS} workers=${S2_WORKERS} effective_batch=$((S2_BS * EFF_BS))"
echo "========================================="

# ── Stage 0: VAE (50K steps) ──
echo "[$(date '+%H:%M:%S')] Stage 0 — VAE encoder+decoder"
torchrun ${TORCHRUN_ARGS} train_state_hijacking_dit.py \
  --config-name "$CONFIG" \
  training.num_train_steps=50000 training.stage=3 \
  training.save_every_n_steps=50000 data.num_workers="$S0_WORKERS" \
  training.train_batch_size="$S0_BS" \
  logging.run_name="$S0_NAME"

echo "[$(date '+%H:%M:%S')] Stage 0 done, waiting for $S0_CKPT ..."
while [ ! -f "${S0_CKPT}/model.pt" ]; do sleep 10; done

# ── Stage 1: Alpha predictors (50K steps) ──
echo "[$(date '+%H:%M:%S')] Stage 1 — Alpha predictors"
torchrun ${TORCHRUN_ARGS} train_state_hijacking_dit.py \
  --config-name "$CONFIG" \
  training.num_train_steps=50000 training.stage=1 \
  training.save_every_n_steps=50000 data.num_workers="$S1_WORKERS" \
  training.train_batch_size="$S1_BS" \
  training.resume="$S0_CKPT" \
  logging.run_name="$S1_NAME"

echo "[$(date '+%H:%M:%S')] Stage 1 done, waiting for $S1_CKPT ..."
while [ ! -f "${S1_CKPT}/model.pt" ]; do sleep 10; done

# ── Stage 2: DiT diffusion (150K steps) ──
echo "[$(date '+%H:%M:%S')] Stage 2 — DiT latent diffusion"
torchrun ${TORCHRUN_ARGS} train_state_hijacking_dit.py \
  --config-name "$CONFIG" \
  training.num_train_steps=150000 training.stage=2 training.gen_type=ddpm \
  training.save_every_n_steps=50000 data.num_workers="$S2_WORKERS" \
  training.train_batch_size="$S2_BS" \
  training.resume="$S1_CKPT" \
  logging.run_name="$S2_NAME"

echo "[$(date '+%H:%M:%S')] Stage 2 done, waiting for $S2_CKPT ..."
while [ ! -f "${S2_CKPT}/model.pt" ]; do sleep 10; done

# ── Sample (single-GPU) ──
echo "[$(date '+%H:%M:%S')] Sampling..."
CUDA_VISIBLE_DEVICES=0 python -u train_state_hijacking_dit.py --sample \
  --ckpt_dir "$S2_CKPT" \
  --prompt "The history of artificial intelligence" \
  --max_len 128 --temperature 0.7 --top_k 50 --seed 42

echo "[$(date '+%H:%M:%S')] 13.3B pipeline done!"
