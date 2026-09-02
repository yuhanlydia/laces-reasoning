#!/bin/bash
# Plan B 2.9B: resume from S0 → s1 (alpha) → s2 (DiT) → sample
# Single-GPU sequential; S0 already done, start from S1
set -e
export PYTHONUNBUFFERED=1

GPU="${GPU:-3}"
CONFIG="rwkv_relay_2.9B_state_hijack_dit_vae32"
NAME="test-v6-2.9B"

S0_CKPT="outputs_relay/${NAME}-s0/step_00050000"
S1_NAME="${NAME}-s1"
S2_NAME="${NAME}-s2"
S1_CKPT="outputs_relay/${S1_NAME}/step_00050000"
S2_CKPT="outputs_relay/${S2_NAME}/step_00050000"

echo "========================================="
echo " Plan B 2.9B: s1 → s2 → sample (resume S0)"
echo " Config : $CONFIG"
echo " GPU    : $GPU"
echo "========================================="

# ── Stage 1: Alpha predictors (50K steps) ──
echo "[$(date '+%H:%M:%S')] Stage 1 — Alpha predictors"
CUDA_VISIBLE_DEVICES="$GPU" python -u train_state_hijacking_dit.py \
  --config-name "$CONFIG" \
  training.num_train_steps=50000 training.stage=1 \
  training.save_every_n_steps=50000 data.num_workers=0 \
  training.train_batch_size=4 \
  training.resume="$S0_CKPT" \
  logging.run_name="$S1_NAME"

echo "[$(date '+%H:%M:%S')] Stage 1 done, waiting for $S1_CKPT ..."
while [ ! -f "${S1_CKPT}/model.pt" ]; do sleep 10; done

# ── Stage 2: DiT diffusion (150K steps) ──
echo "[$(date '+%H:%M:%S')] Stage 2 — DiT latent diffusion"
CUDA_VISIBLE_DEVICES="$GPU" python -u train_state_hijacking_dit.py \
  --config-name "$CONFIG" \
  training.num_train_steps=150000 training.stage=2 training.gen_type=ddpm \
  training.save_every_n_steps=50000 data.num_workers=0 \
  training.train_batch_size=4 \
  training.resume="$S1_CKPT" \
  logging.run_name="$S2_NAME"

echo "[$(date '+%H:%M:%S')] Stage 2 done, waiting for $S2_CKPT ..."
while [ ! -f "${S2_CKPT}/model.pt" ]; do sleep 10; done

# ── Sample ──
echo "[$(date '+%H:%M:%S')] Sampling..."
CUDA_VISIBLE_DEVICES="$GPU" python -u train_state_hijacking_dit.py --sample \
  --ckpt_dir "$S2_CKPT" \
  --prompt "The history of artificial intelligence" \
  --max_len 128 --temperature 0.7 --top_k 50 --seed 42

echo "[$(date '+%H:%M:%S')] 2.9B pipeline done!"
