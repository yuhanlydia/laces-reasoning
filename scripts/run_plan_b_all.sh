#!/bin/bash
# Plan B: Full pipeline — s0 (VAE) → s1 (alpha) → s2 (DiT) → sample
set -e
export PYTHONUNBUFFERED=1

CONFIG="rwkv_relay_0.4B_state_hijack_dit_vae32"
RUN_NAME="test-v6"
S0_NAME="${RUN_NAME}-s0"
S1_NAME="${RUN_NAME}-s1"
S2_NAME="${RUN_NAME}-s2"

S0_CKPT="outputs_relay/${S0_NAME}/step_00050000"
S1_CKPT="outputs_relay/${S1_NAME}/step_00050000"
S2_CKPT="outputs_relay/${S2_NAME}/step_00050000"

echo "========================================="
echo " Plan B Pipeline: s0 → s1 → s2 → sample"
echo " Config: $CONFIG"
echo " Run:    $RUN_NAME"
echo " GPU:    s0=0, s1=1, s2=2"
echo "========================================="

# ── Stage 0: VAE encoder + decoder ──
echo "[$(date '+%H:%M:%S')] Starting Stage 0 (VAE)..."
CUDA_VISIBLE_DEVICES=0 python -u train_state_hijacking_dit.py \
  --config-name "$CONFIG" \
  training.num_train_steps=50000 training.stage=3 \
  training.save_every_n_steps=50000 data.num_workers=0 \
  training.train_batch_size=64 \
  logging.run_name="$S0_NAME"

echo "[$(date '+%H:%M:%S')] Stage 0 done. Waiting for checkpoint..."
while [ ! -f "${S0_CKPT}/model.pt" ]; do sleep 10; done

# ── Stage 1: Alpha on frozen VAE ──
echo "[$(date '+%H:%M:%S')] Starting Stage 1 (Alpha)..."
CUDA_VISIBLE_DEVICES=1 python -u train_state_hijacking_dit.py \
  --config-name "$CONFIG" \
  training.num_train_steps=50000 training.stage=1 \
  training.save_every_n_steps=50000 data.num_workers=0 \
  training.train_batch_size=16 \
  training.resume="$S0_CKPT" \
  logging.run_name="$S1_NAME"

echo "[$(date '+%H:%M:%S')] Stage 1 done. Waiting for checkpoint..."
while [ ! -f "${S1_CKPT}/model.pt" ]; do sleep 10; done

# ── Stage 2: DiT on fixed z-space ──
echo "[$(date '+%H:%M:%S')] Starting Stage 2 (DiT)..."
CUDA_VISIBLE_DEVICES=2 python -u train_state_hijacking_dit.py \
  --config-name "$CONFIG" \
  training.num_train_steps=150000 training.stage=2 training.gen_type=ddpm \
  training.save_every_n_steps=50000 data.num_workers=0 \
  training.train_batch_size=16 \
  training.resume="$S1_CKPT" \
  logging.run_name="$S2_NAME"

echo "[$(date '+%H:%M:%S')] Stage 2 done. Waiting for checkpoint..."
while [ ! -f "${S2_CKPT}/model.pt" ]; do sleep 10; done

# ── Sample ──
echo "[$(date '+%H:%M:%S')] Sampling..."
CUDA_VISIBLE_DEVICES=2 python -u train_state_hijacking_dit.py --sample \
  --ckpt_dir "$S2_CKPT" \
  --prompt "The history of artificial intelligence" \
  --max_len 128 --temperature 0.7 --top_k 50 --seed 42

echo "[$(date '+%H:%M:%S')] All done!"
