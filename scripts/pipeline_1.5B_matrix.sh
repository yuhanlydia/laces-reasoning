#!/bin/bash
# Automated pipeline: wait for 1.5B S0 → train cross-source S1 → eval PPL
# Run: bash scripts/pipeline_1.5B_matrix.sh
set -e
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv

CKPT="outputs_relay/traj32x16-1.5B-s0/step_00050000/model.pt"
echo "[pipeline] waiting for 1.5B S0 checkpoint: $CKPT"
while [ ! -f "$CKPT" ]; do
    latest=$(ls -d outputs_relay/traj32x16-1.5B-s0/step_* 2>/dev/null | sort -V | tail -1)
    echo "[pipeline] $(date +%H:%M) waiting... latest=$latest"
    sleep 300
done
echo "[pipeline] 1.5B S0 ready. Starting cross-source S1 training..."

# Train 1.5B cross-source S1 (13.3B Z → 1.5B WKV states)
CUDA_VISIBLE_DEVICES=1 python scripts/train_cross_source_full.py \
  --backbone 1.5B --stage 1 \
  --latent_dir preprocessed_data/owt_13b_s2_denoised_z/train \
  --token_dir preprocessed_data/owt_rwkv_tokens/train \
  --save_dir outputs_relay/cross-source-1.5B-s1-denoised \
  --num_steps 30000 --batch_size 8 --lr 3e-5 --save_every 5000

echo "[pipeline] 1.5B cross-source S1 done."
