# Training launch matrix

This directory is the canonical launch surface for method experiments. Launchers are split first by conditioning:

```text
training/
  unconditional/
    512/<0.4B|2.9B|13.3B>/single_z/basis_<8|16|32|64>/<ddpm|rf|flow>.sh
    512/<0.4B|2.9B|13.3B>/trajectory/<dit|rwkv|birwkv>/basis_<8|16|32|64>/<ddpm|rf|flow>.sh
    4096/<0.4B|2.9B|13.3B>/single_z/basis_<8|16|32|64>/<ddpm|rf|flow>.sh
    4096/<0.4B|2.9B|13.3B>/trajectory/<dit|rwkv|birwkv>/basis_<8|16|32|64>/<ddpm|rf|flow>.sh

  conditional/
    prefix_suffix/
      512/<0.4B|2.9B|13.3B>/single_z/basis_<8|16|32|64>/<ddpm|rf|flow>.sh
      512/<0.4B|2.9B|13.3B>/trajectory/<dit|rwkv|birwkv>/basis_<8|16|32|64>/<ddpm|rf|flow>.sh
      512/<0.4B|2.9B|13.3B>/post_training/<planner_aware_s2|stochastic_continuation_s2|self_forcing_s2|single_z_s2_sft|response_sft>.sh
      4096/<0.4B|2.9B|13.3B>/single_z/basis_<8|16|32|64>/<ddpm|rf|flow>.sh
      4096/<0.4B|2.9B|13.3B>/trajectory/<dit|rwkv|birwkv>/basis_<8|16|32|64>/<ddpm|rf|flow>.sh
      4096/<0.4B|2.9B|13.3B>/post_training/<planner_aware_s2|stochastic_continuation_s2|self_forcing_s2|single_z_s2_sft|response_sft>.sh

  preprocess/4096/<full_rows|pack_existing|pack_stream>.sh
  cluster/unconditional/*.sh
  cluster/conditional/prefix_suffix/*.sh
```

The canonical cluster paths are under `training/cluster/unconditional/` and
`training/cluster/conditional/prefix_suffix/`. The old flat paths directly under
`training/cluster/*.sh` are kept as compatibility shims and forward to the
canonical nested launchers.

## 实验说明：Diffusion RWKV

我们的最新方法可以直接看成 **Diffusion RWKV**：RWKV 自己就是 token block 的
去噪器。训练时先保存进入当前 block 前的干净 `pre-state`，然后把当前 clean chunk
随机加噪成 masked chunk。比如 clean chunk 是 `[A, B, C, D]`，加噪后变成
`[A, M, C, M]`。去噪分支只在当前 block 内使用：从干净 `pre-state` 扫一遍
`[A, M, C, M]` 得到临时 `masked-state`，再从这个 `masked-state` 扫同一个
`[A, M, C, M]` 输出 logits，只对被 mask 的 `B` 和 `D` 算 loss。这个
`masked-state` 是一次性的临时状态，不会传给下一个 block；算完 loss 后，训练会回到
原来的干净 `pre-state`，扫描 ground-truth clean chunk `[A, B, C, D]`，得到下一个
block 的干净左边界 state。

RWKV 预测 logits 的方式是整块 query：第二遍 scan `[A, M, C, M]` 时，RWKV 对 block
里每个位置都输出一个 vocab logits。训练只取 mask 位置的 logits 参与 CE，所以这里
只监督位置 2 和位置 4，让它们分别预测 `B` 和 `D`；未 mask 的 `A`、`C` 只是上下文，
不作为 loss 目标。

训练阶段不是一步步 unmask。每个训练样本只随机采一次 mask pattern，例如一次变成
`[A, M, C, M]`，然后模型一次性预测所有被 mask 的位置并计算 loss。也就是说，训练
是在学习“任意 mask pattern 下如何把 masked token 还原出来”；多步 unmask 是推理时
为了从全 mask 逐步生成 clean block 才使用的采样过程。

推理时 diffusion 发生在“全 mask -> 多步 unmask”的反向去噪过程里：从
`[M, M, M, M]` 开始，每个 denoise step 对整个 block 做一次 prefill + query，一次
拿到所有位置的 logits。然后每个还没 unmask 的位置先从自己的 logits 里选一个候选
token：`temperature=0` 时就是 argmax，`temperature>0` 时可以按 softmax 采样，并可
叠加 top-k/top-p。候选 token 的 confidence 是它在原始 softmax 里的概率。最后按提交
策略决定这一轮哪些位置真正写回 block：`all` 是一次提交所有位置，`linear` 是每步
提交一定数量的高置信位置，`threshold` 是提交超过阈值的位置。没被提交的位置仍保持
mask，下一轮会基于新的 partial block 重新 prefill + query、重新预测。因此 unmask
不是固定从左到右，而是由 confidence/schedule 决定，例如
`[M, M, M, M] -> [M, b, M, d] -> [a, b, c, d]`。RWKV 的 scan 内部仍是左到右，但
token 的 unmask/commit 不是普通自回归那种一个 token 一个 token 从左到右生成，而是
block-level 的并行候选 + 多步提交。当前 block 完成后，再把 clean/generated chunk
扫进 RWKV state，作为下一个 block 的干净左边界。

当前训练入口是 `scripts/train_state_prefill_block_diffusion.py`，已经支持单机 H200×8
DDP。用 `torchrun --nproc_per_node=8` 启动时，脚本会按 `LOCAL_RANK` 绑定 GPU，使用
`DistributedSampler` 切分数据，并且只让 rank0 打日志和保存 checkpoint。这里的
`--batch_size` 是每张 GPU/rank 的 batch size；H200 上统一使用 `--batch_size 4`。
FineWeb 推荐直接用本地 `sample-10BT` 的 4096
packed 版本：`preprocessed_data/fineweb_4096_packed_full`，共 2,305,891 条 4096-token
packed 样本，约 9.45B token。B3D-RWKV 的公开训练配置是 8 卡、`B=32`、effective
batch 128，并在约 4.9B-token 的 SFT/trajectory 混合数据上训练约 2 epochs，约等于
9.8B raw content tokens。因为这里用的是不同数据集，不能直接照搬 epoch 数；更稳妥的
预算对齐是按 raw token 数对齐。FineWeb packed 一整遍约 9.45B token，已经接近
B3D-RWKV 的总 raw-token 预算：`2,305,891 / 128 ≈ 18,015` optimizer steps/epoch。
正式训练使用 5 epochs，即约 `18,015 * 5 = 90,075` optimizer steps；这是比
B3D-RWKV token-budget parity 更大的 FineWeb 训练预算。

H200×8 baseline（第一组，最基础配置）：

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29531 \
  scripts/train_state_prefill_block_diffusion.py \
  --rwkv_path /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B-world \
  --token_dir preprocessed_data/fineweb_4096_packed_full \
  --save_dir outputs_state_prefill_block/diffusion_rwkv_fineweb10bt_h200_baseline \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --block_size 32 \
  --max_steps 90075 \
  --lr 1e-6 \
  --weight_decay 0.0 \
  --dtype bf16 \
  --eos_id 0 \
  --min_mask_ratio 0.05 \
  --max_mask_ratio 0.20 \
  --full_mask_prob 0.0 \
  --num_workers 4 \
  --save_every 5000 \
  --log_every 10
```

H200×8 state-native strong（第二组，最有意义的增强配置）：

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29541 \
  scripts/train_state_prefill_block_diffusion.py \
  --rwkv_path /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B-world \
  --token_dir preprocessed_data/fineweb_4096_packed_full \
  --save_dir outputs_state_prefill_block/diffusion_rwkv_fineweb10bt_h200_state_logit_ar \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --block_size 32 \
  --max_steps 90075 \
  --lr 1e-6 \
  --weight_decay 0.0 \
  --dtype bf16 \
  --eos_id 0 \
  --min_mask_ratio 0.05 \
  --max_mask_ratio 0.20 \
  --full_mask_prob 0.0 \
  --lambda_state 0.01 \
  --state_mask_ratio_weight 2.0 \
  --lambda_logit 0.05 \
  --lambda_ar 0.05 \
  --num_workers 4 \
  --save_every 5000 \
  --log_every 10
```

H200×1 state-native strong（单卡先跑/看 loss）：

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv

CUDA_VISIBLE_DEVICES=1 python \
  scripts/train_state_prefill_block_diffusion.py \
  --rwkv_path /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B-world \
  --token_dir preprocessed_data/fineweb_4096_packed_full \
  --save_dir outputs_state_prefill_block/diffusion_rwkv_fineweb10bt_h200_1gpu_state_logit_ar \
  --batch_size 4 \
  --grad_accum_steps 32 \
  --block_size 32 \
  --max_steps 90075 \
  --lr 1e-6 \
  --weight_decay 0.0 \
  --dtype bf16 \
  --eos_id 0 \
  --min_mask_ratio 0.05 \
  --max_mask_ratio 0.20 \
  --full_mask_prob 0.0 \
  --lambda_state 0.01 \
  --state_mask_ratio_weight 2.0 \
  --lambda_logit 0.05 \
  --lambda_ar 0.05 \
  --num_workers 4 \
  --save_every 10000 \
  --log_every 10 \
  --micro_log_every 8
```

RWKV raw vocab 没有天然 `[MASK]` token。训练脚本会自动用模型 `vocab_size` 末尾的
unused id 作为 mask，例如 0.4B-world 是 `mask_id=65535`；`pad_id` 默认读取 tokenizer
的 pad token。不要手动把 `len(tokenizer)-1` 当 mask，因为那会落到 RWKV 的换行/EOS
token 上。

第二组 strong 命令启用 state-native 增强 loss：

```bash
  --lambda_state 0.01 \
  --state_mask_ratio_weight 2.0 \
  --lambda_logit 0.05 \
  --lambda_ar 0.05
```

第一组 baseline 不启用这些 loss；若要从 strong 命令退回纯 baseline，把这三项都设成 `0.0` 或删掉即可。含义：`lambda_state` 让 masked branch 的临时 state 靠近 clean branch state；
`state_mask_ratio_weight` 把 state loss 变成 `1 + weight * mask_ratio` 的 diffusion-style 加权；
`lambda_logit` 让 masked branch logits 蒸馏 clean branch logits；`lambda_ar` 保留原始
RWKV 的 next-token 能力。这里暂时不加 CAP loss，避免方法上过度贴近 B3D-RWKV。

Examples:

```bash
# Unconditional single-z, OWT-512.
BATCH_SIZE=8 GPU=0 training/unconditional/512/2.9B/single_z/basis_32/ddpm.sh

# Unconditional trajectory, FineWeb-4096.
BATCH_SIZE=8 GPU=2 training/unconditional/4096/2.9B/trajectory/birwkv/basis_32/rf.sh

# Prefix/suffix single-z CFG.
BATCH_SIZE=8 GPU=0 training/conditional/prefix_suffix/512/2.9B/single_z/basis_32/ddpm.sh

# Prefix/suffix trajectory CFG.
# Pass S0_CKPT when using older no-basis S0 checkpoint names; otherwise these
# wrappers wait for their default basis-aware S0 path.
# CFG_DROP_PROB defaults to 0.1; set CFG_DROP_PROB=0 only if you want to disable
# CFG dropout during S2 training.
BATCH_SIZE=8 GPU=2 CFG_DROP_PROB=0.1 S0_CKPT=outputs_relay/traj32x16-2.9B-s0/step_00050000 \
  training/conditional/prefix_suffix/512/2.9B/trajectory/rwkv/basis_32/rf.sh

# Response-only SFT from instruction/chat tokens with response_mask.
python scripts/preprocess/preprocess_response_sft.py \
  --dataset allenai/tulu-3-sft-mixture \
  --output_dir preprocessed_data/tulu3_response_sft_512 \
  --model_path /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF \
  --max_length 512 --local_files_only

DRY_RUN=1 SFT_TOKEN_DIR=preprocessed_data/tulu3_response_sft_512 SFT_RESUME=outputs_relay/test-v6-2.9B-s1/step_00050000 \
  training/conditional/prefix_suffix/512/2.9B/post_training/response_sft.sh
```

S2 trajectory SFT trains the latent trajectory sampler, not the response-only
S1 CE bridge. Use `MIX_MODE=maximal` for the largest math/code/agent trace mix:
Tulu3, OpenThoughts3, Open-R1 Mixture-of-Thoughts, Nemotron math/code,
OpenThoughts Agent SFT, Complete-FABLE.5 traces, AgentTrove, and Nemotron
Post-Training. In maximal mode each dataset defaults to full preprocessing;
set any `*_MAX` variable to cap one source for a smoke run. The preprocessor
keeps at least one trajectory chunk for both prompt and response, so S2 learns
`p(Z_response | Z_prompt)` from each dataset's native SFT prompt/response
boundary. `SFT_PROMPT_TOKENS` is optional; set it only when you explicitly want
to truncate every prompt to a fixed chunk-aligned length for an ablation. The S2
trajectory SFT trainer supports mixed `prompt_lengths` inside a batch, so the
default native split works with `BATCH_SIZE=8`.

```bash
# Dry-run first: prints every preprocessing command without writing token shards.
DRY_RUN=1 CONTEXT_LENGTH=512 MIX_MODE=maximal BACKBONE=2.9B \
  training/preprocess/s2_trajectory_sft_mix.sh

# Real 512-token maximal preprocess. This writes SFT token shards with
# input_ids, attention_mask, response_mask, and prompt_lengths. It may take a
# long time and requires the HF datasets/tokenizer to already be available
# locally because the launcher passes --local_files_only.
CONTEXT_LENGTH=512 MIX_MODE=maximal BACKBONE=2.9B \
  OUT_DIR=preprocessed_data/s2_traj_sft_512_maximal \
  training/preprocess/s2_trajectory_sft_mix.sh

# If OpenThoughts3 repeatedly stalls on HuggingFace RemoteDisconnected retries
# and the openthoughts3 shard count is no longer increasing, keep the already
# written shards, mark that source as intentionally partial, and continue the
# remaining maximal sources. Do not use this unless the OpenThoughts3 writer has
# clearly stopped making progress.
PIDS="$(pgrep -f 'python scripts/preprocess/preprocess_response_sft.py --dataset open-thoughts/OpenThoughts3-1.2M' || true)"
if [[ -n "${PIDS}" ]]; then kill ${PIDS}; fi
PIDS="$(pgrep -f 'bash training/preprocess/s2_trajectory_sft_mix.sh' || true)"
if [[ -n "${PIDS}" ]]; then kill ${PIDS}; fi
python - <<'PY'
from pathlib import Path
import json, time

root = Path("preprocessed_data/s2_traj_sft_512_maximal")
files = sorted(root.glob("openthoughts3_*_tokens.npz"))
if not files:
    raise SystemExit("No openthoughts3 shards found; do not create a partial manifest.")
payload = {
    "saved": len(files),
    "skipped": None,
    "filename_prefix": "openthoughts3",
    "start_index": 0,
    "max_length": 512,
    "min_response_tokens": 32,
    "min_prompt_tokens": 32,
    "fixed_prompt_tokens": None,
    "source": "open-thoughts/OpenThoughts3-1.2M",
    "partial": True,
    "skip_reason": "Skipped remaining OpenThoughts3 shards after repeated HuggingFace RemoteDisconnected retries; keeping already-written shards.",
    "last_file": files[-1].name,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(root / "manifest_openthoughts3.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
CONTEXT_LENGTH=512 MIX_MODE=maximal BACKBONE=2.9B \
  OUT_DIR=preprocessed_data/s2_traj_sft_512_maximal \
  training/preprocess/s2_trajectory_sft_mix.sh

# Safer/lower-risk 512-token preprocess without maximal-only trace sources.
CONTEXT_LENGTH=512 MIX_MODE=safe BACKBONE=2.9B \
  OUT_DIR=preprocessed_data/s2_traj_sft_512_safe \
  training/preprocess/s2_trajectory_sft_mix.sh

# One-command 512-token safe S2 trajectory SFT run:
# first tokenizes/preprocesses the safe SFT mix, then starts S2 trajectory SFT with BATCH_SIZE=8.
CONTEXT_LENGTH=512 MIX_MODE=safe BACKBONE=2.9B \
  OUT_DIR=preprocessed_data/s2_traj_sft_512_safe \
  training/preprocess/s2_trajectory_sft_mix.sh && \
GPU=2 BATCH_SIZE=8 SAVE_EVERY=5000 SFT_STEPS=20000 \
  SFT_TOKEN_DIR=preprocessed_data/s2_traj_sft_512_safe \
  SFT_RESUME=outputs_relay/owt512-traj32x16-2.9B-basis32-prefix-suffix-blend0p5-s2-rwkv-ddpm/step_00150000 \
  ARCH=rwkv BASIS=32 OBJECTIVE=ddpm \
  training/conditional/prefix_suffix/512/2.9B/post_training/s2_trajectory_sft.sh

# One-command 512-token maximal S2 trajectory SFT run:
# first tokenizes/preprocesses the maximal SFT mix, then starts S2 trajectory SFT with BATCH_SIZE=8.
CONTEXT_LENGTH=512 MIX_MODE=maximal BACKBONE=2.9B \
  OUT_DIR=preprocessed_data/s2_traj_sft_512_maximal \
  training/preprocess/s2_trajectory_sft_mix.sh && \
GPU=2 BATCH_SIZE=8 SAVE_EVERY=5000 SFT_STEPS=20000 \
  SFT_TOKEN_DIR=preprocessed_data/s2_traj_sft_512_maximal \
  SFT_RESUME=outputs_relay/owt512-traj32x16-2.9B-basis32-prefix-suffix-blend0p5-s2-rwkv-ddpm/step_00150000 \
  ARCH=rwkv BASIS=32 OBJECTIVE=ddpm \
  training/conditional/prefix_suffix/512/2.9B/post_training/s2_trajectory_sft.sh

# Single-z maximal SFT, recommended two-stage route.

# Train S1 only: response bridge SFT. This is response-token CE, not S2; it
# adapts z->state injection to SFT responses. With SAVE_EVERY=5000 and
# SFT_STEPS=20000, it saves step_00005000, step_00010000, step_00015000, and
# step_00020000 under:
# outputs_relay/owt512-2.9B-basis32-singlez-s1-response-sft-maximal-bs8/
GPU=1 BATCH_SIZE=8 SAVE_EVERY=5000 SFT_STEPS=20000 \
  SFT_TOKEN_DIR=preprocessed_data/s2_traj_sft_512_maximal \
  SFT_RESUME=outputs_relay/test-v6-2.9B-s1/step_00050000 \
  BASIS=32 SFT_STAGE=1 \
  RUN_NAME=owt512-2.9B-basis32-singlez-s1-response-sft-maximal-bs8 \
  training/conditional/prefix_suffix/512/2.9B/post_training/response_sft.sh

# Continue single-z S2 SFT from the completed 20k maximal checkpoint to 3.5M
# total steps. This writes checkpoints every 50k steps under the same run name.
GPU=2 BATCH_SIZE=4 SAVE_EVERY=50000 SFT_STEPS=3500000 \
  SFT_TOKEN_DIR=preprocessed_data/s2_traj_sft_512_maximal \
  SFT_RESUME=outputs_relay/owt512-2.9B-basis32-singlez-ddpm-s2-sft-maximal-bs8/step_00020000 \
  BASIS=32 OBJECTIVE=ddpm \
  RUN_NAME=owt512-2.9B-basis32-singlez-ddpm-s2-sft-maximal-bs8 \
  training/conditional/prefix_suffix/512/2.9B/post_training/single_z_s2_sft.sh

# Train trajectory S2 SFT with the completed BiRWKV DDPM prefix/suffix S2 checkpoint.
# Use a different GPU or wait for the single-z S2 job to finish before launching on the same 4090.
GPU=1 BATCH_SIZE=4 SAVE_EVERY=50000 SFT_STEPS=3500000 \
  SFT_TOKEN_DIR=preprocessed_data/s2_traj_sft_512_maximal \
  SFT_RESUME=outputs_relay/owt512-traj32x16-2.9B-basis32-prefix-suffix-blend0p5-s2-birwkv-ddpm/step_00150000 \
  ARCH=birwkv BASIS=32 OBJECTIVE=ddpm STATE_BLEND=0.5 \
  RUN_NAME=owt512-traj32x16-2.9B-basis32-birwkv-ddpm-s2-trajectory-sft-maximal-bs4 \
  training/conditional/prefix_suffix/512/2.9B/post_training/s2_trajectory_sft.sh

# Optional: train S2 directly from the original single-z S2 checkpoint, skipping
# S1 bridge SFT. Use this only for an ablation.
GPU=3 BATCH_SIZE=8 SAVE_EVERY=5000 SFT_STEPS=20000 \
  SFT_TOKEN_DIR=preprocessed_data/s2_traj_sft_512_maximal \
  SFT_RESUME=outputs_relay/test-v6-2.9B-s2/step_00150000 \
  BASIS=32 OBJECTIVE=ddpm \
  RUN_NAME=owt512-2.9B-basis32-singlez-ddpm-s2-direct-sft-maximal-bs8 \
  training/conditional/prefix_suffix/512/2.9B/post_training/single_z_s2_sft.sh

# Response-only CE remains available as the S1 bridge step/baseline; it is not S2.
# GPU=1 BATCH_SIZE=8 SFT_STAGE=0 ... post_training/response_sft.sh
```

Keep `MIX_MODE=safe` and `MIX_MODE=maximal` outputs in separate token dirs.
The maximal mix intentionally includes high-risk mixed-license trace sources;
use it for internal capability training unless the license review says otherwise.
For `nvidia/Nemotron-Post-Training-Dataset-v1`, maximal mode preprocesses the
`code`, `math`, and `tool_calling` splits separately.

Current `preprocessed_data/s2_traj_sft_512_maximal` status, 2026-06-26 UTC:
maximal preprocessing is complete and sealed; all source manifests and
`mix_manifest_maximal_512.json` exist, so this token directory is safe for S2
SFT. Completed/full sources: `tulu3_core`, `tulu3_if`, `openr1_mot`,
`nemotron_opencode_v1` (aggregate of the named OpenCode splits, 458,694
examples), `ot_agent_100k`, `agenttrove` (1,568,254 examples),
`nemotron_post_math` (1,934,168 examples), and `nemotron_post_tool` (310,050
examples). Completed partial sources: `openthoughts3` (818,659 shards kept after
repeated HuggingFace `RemoteDisconnected` retries), `nemotron_math_v4` (51,655
shards kept), `fable5_traces` (0 shards; dataset inaccessible from this
environment), and `nemotron_post_code` (0 shards; partial-skipped after no local
growth). The watchdog exited after logging `all manifests and mix manifest
present`. Single-z S2 SFT can now use this directory; the recommended continuation
command above resumes from
`outputs_relay/owt512-2.9B-basis32-singlez-ddpm-s2-sft-maximal-bs8/step_00020000`
and trains to 3.5M total steps with 50k-step checkpoints.

Current single-z S2 SFT progress, 2026-06-26 UTC: the first maximal single-z S2
run completed `step_00020000` with checkpoints at 5k/10k/15k/20k under
`outputs_relay/owt512-2.9B-basis32-singlez-ddpm-s2-sft-maximal-bs8/`. A quick
sample smoke from `step_00020000` confirmed the checkpoint loads and the sampler
injects nonzero sampled states, but output quality is still unstable: a simple
sky explanation starts correctly, while a basic arithmetic prompt collapses into
repetition even with conservative decoding. Treat 20k as a smoke/early-adaptation
checkpoint, not a final result. The next planned single-z run is the continuation
command above: `SFT_STEPS=3500000`, `SAVE_EVERY=50000`, same run name, resuming
from `step_00020000`. At batch size 4 this is about 2.2 maximal-data epochs; disk
use is acceptable because 50k-step saves produce roughly 70 checkpoints.

Runtime check, 2026-06-26 UTC: single-z 3.5M continuation has not been launched
yet because all visible 4090 GPUs were already occupied. The maximal trajectory
BiRWKV S2 SFT 3.5M job is running with
`data.token_dir=preprocessed_data/s2_traj_sft_512_maximal`,
`training.resume=outputs_relay/owt512-traj32x16-2.9B-basis32-prefix-suffix-blend0p5-s2-birwkv-ddpm/step_00150000`,
`training.num_train_steps=3500000`, `training.save_every_n_steps=50000`, and
`logging.run_name=owt512-traj32x16-2.9B-basis32-birwkv-ddpm-s2-trajectory-sft-maximal-bs4`.
Launch the single-z continuation only after a GPU is free, or intentionally stop
one of the other long jobs.

Runtime update, 2026-06-27 UTC: QZ auth is working again. The single-z maximal
S2 continuation was submitted to the Dai project as a 2xH200 job after a dry-run
validated the request body. QZ job id:
`job-b8742a76-1a92-48df-9f7c-6d914c77ba80`; QZ status at submit check:
`job_queuing`, `gpu_count=2`, spec id
`26ef0d6e-330d-4650-a18a-7e1fbe8f3717` (`2卡40核`, H200 141GB). The command uses
`torchrun --nproc_per_node=2`, `data.token_dir=preprocessed_data/s2_traj_sft_512_maximal`,
`training.resume=outputs_relay/owt512-2.9B-basis32-singlez-ddpm-s2-sft-maximal-bs8/step_00020000`,
`training.num_train_steps=3500000`, `training.save_every_n_steps=50000`, per-rank
`training.train_batch_size=4`, and keeps
`logging.run_name=owt512-2.9B-basis32-singlez-ddpm-s2-sft-maximal-bs8`. The local
trajectory BiRWKV maximal S2 SFT is still running and has now produced
`step_00050000` and `step_00100000` checkpoints in addition to the earlier
5k/10k/15k checkpoints.

```bash
# MMLU-Pro-style local JSONL rows can use question/options/answer fields.
python scripts/preprocess/preprocess_response_sft.py \
  --input_file data/sft/mmlu_pro_train.jsonl \
  --output_dir preprocessed_data/mmlu_pro_response_sft_512 \
  --model_path /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF \
  --max_length 512 --local_files_only

# IFEval-style local JSONL rows can use prompt/response or instruction/output fields.
python scripts/preprocess/preprocess_response_sft.py \
  --input_file data/sft/ifeval_style_train.jsonl \
  --output_dir preprocessed_data/ifeval_response_sft_512 \
  --model_path /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF \
  --max_length 512 --local_files_only

# OWT-512 2.9B prefix/suffix trajectory missing S2 matrix, split for four terminals/GPUs.
GPU=0 QUEUE=0 BATCH_SIZE=8 training/conditional/prefix_suffix/512/2.9B/trajectory/run_missing_s2_matrix.sh
GPU=1 QUEUE=1 BATCH_SIZE=4 training/conditional/prefix_suffix/512/2.9B/trajectory/run_missing_s2_matrix.sh
GPU=2 QUEUE=2 BATCH_SIZE=4 training/conditional/prefix_suffix/512/2.9B/trajectory/run_missing_s2_matrix.sh
GPU=3 QUEUE=3 BATCH_SIZE=4 training/conditional/prefix_suffix/512/2.9B/trajectory/run_missing_s2_matrix.sh


BATCH_SIZE=4 GPU=3 CFG_DROP_PROB=0.0 S0_CKPT=outputs_relay/traj32x16-2.9B-s0/step_00050000 \
  training/conditional/prefix_suffix/512/2.9B/trajectory/rwkv/basis_32/rf.sh


BATCH_SIZE=8 GPU=3 CFG_DROP_PROB=0.1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/conditional/prefix_suffix/4096/2.9B/trajectory/birwkv/basis_32/rf.sh


# 13.3B prefix/suffix trajectory CFG. Start with BATCH_SIZE=1 for 13.3B.
BATCH_SIZE=8 GPU=0 CFG_DROP_PROB=0.1 S0_CKPT=outputs_relay/traj32x16-13.3B-s0/step_00050000 \
  training/conditional/prefix_suffix/512/13.3B/trajectory/rwkv/basis_32/rf.sh


BATCH_SIZE=8 GPU=0 CFG_DROP_PROB=0.0 S0_CKPT=outputs_relay/traj32x16-13.3B-s0/step_00050000 \
  training/conditional/prefix_suffix/512/13.3B/trajectory/rwkv/basis_32/rf.sh


BATCH_SIZE=1 GPU=3 CFG_DROP_PROB=0.1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-13.3B-s0/step_00050000 \
  training/conditional/prefix_suffix/4096/13.3B/trajectory/birwkv/basis_32/rf.sh

# FineWeb preprocessing.
training/preprocess/4096/pack_existing.sh
```

4096 cluster examples:

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv

# Unconditional 4096 trajectory, H200×8.
BATCH_SIZE=1 DRY_RUN=0 training/cluster/unconditional/h200x8_4096_2.9B_traj_dit_rf.sh

# Prefix/suffix 4096 trajectory, RWKV DDPM basis32, compare blend=0.5 vs blend=1.0.
# Start H200×8 RWKV S1 with BATCH_SIZE=1. BATCH_SIZE is per GPU/rank;
# BATCH_SIZE=8 means global batch 64 and has OOMed on 2.9B 4096 RWKV S1.
BATCH_SIZE=1 DRY_RUN=0 STAGE=1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32_blend05.sh

BATCH_SIZE=1 DRY_RUN=0 STAGE=1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32_blend1.sh

# After S1 completes, run S2 separately.
BATCH_SIZE=1 DRY_RUN=0 STAGE=2 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32_blend1.sh

# Prefix/suffix 4096 trajectory, RWKV RF basis32, compare blend=0.5 vs blend=1.0.
BATCH_SIZE=1 DRY_RUN=0 STAGE=1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_rf_basis32_blend05.sh

BATCH_SIZE=1 DRY_RUN=0 STAGE=1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_rf_basis32_blend1.sh




# Same run through the old flat compatibility shim. This is useful for older notes/scripts.
BATCH_SIZE=1 DRY_RUN=0 STAGE=1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32_blend05.sh

# 2-GPU H200 smoke/probe version of the same prefix/suffix run. Use this when
# only two GPUs are allocated; do not use the h200x8 wrapper with two visible GPUs.
BATCH_SIZE=8 DRY_RUN=0 STAGE=all BACKBONE=2.9B ARCH=rwkv OBJECTIVE=ddpm BASIS=32 STATE_BLEND=0.5 \
  NPROC_PER_NODE=2 NNODES=1 NODE_RANK=0 \
  S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/launch_4096_prefix_suffix_trajectory_stage.sh

# Generic prefix/suffix 4096 launcher.
BATCH_SIZE=8 DRY_RUN=0 STAGE=all BACKBONE=2.9B ARCH=birwkv OBJECTIVE=ddpm BASIS=32 STATE_BLEND=0.5 \
  S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/launch_4096_prefix_suffix_trajectory_stage.sh
```

Why `S0_CKPT` is explicit above: new basis-aware wrappers default to
`outputs_relay/fineweb4096-traj64x64-2.9B-basis32-s0/step_00050000`, but the
existing 2.9B 4096 S0 checkpoint in this repo is the older no-basis name:

```text
outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000/model.pt
```

If you omit `S0_CKPT` and only the old no-basis checkpoint exists, the launcher
will stop with `Missing S0_CKPT`. Pass the existing checkpoint explicitly:

```bash
BATCH_SIZE=1 DRY_RUN=0 STAGE=1 \
S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
training/cluster/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32_blend05.sh
```

For H200×8 RWKV prefix/suffix 4096 S1, treat `BATCH_SIZE=1` as the safe
starting point. Try `BATCH_SIZE=2` only after the run is stable for hundreds of
steps; `BATCH_SIZE=4` is not guaranteed; `BATCH_SIZE=8` has OOMed because it is
per-rank and becomes global batch 64 with `NPROC_PER_NODE=8`.

That last command uses the old flat compatibility shim; the equivalent canonical
path is:

```bash
training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32_blend05.sh
```

Base wrappers call:

```text
unconditional/single_z      -> training/run_single_z.sh
unconditional/trajectory    -> training/run_trajectory.sh
conditional/prefix_suffix/single_z   -> training/run_prefix_suffix_single_z.sh
conditional/prefix_suffix/trajectory -> training/run_prefix_suffix_trajectory.sh
conditional/prefix_suffix/post_training -> training/run_post_training_route.sh
```

Override `BATCH_SIZE`, `GPU`, `NUM_WORKERS`, `S0_STEPS`, `S1_STEPS`, `S2_STEPS`, `SAVE_EVERY`, `TOKEN_DIR`, `STATE_BLEND`, `S0_CKPT`, or `S1_CKPT` as needed. Set `DRY_RUN=1` to print commands without launching training.
