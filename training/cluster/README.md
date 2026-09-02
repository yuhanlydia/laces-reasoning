# 4096 cluster launch templates

Cluster launchers are split by conditioning:

```text
training/cluster/
  unconditional/
    launch_4096_trajectory_stage.sh
    h200x8_4096_<2.9B|13.3B>_traj_dit_rf.sh
    2nodes16_4096_<2.9B|13.3B>_traj_dit_rf.sh

  conditional/prefix_suffix/
    launch_4096_prefix_suffix_trajectory_stage.sh
    h200x8_4096_<2.9B|13.3B>_prefix_suffix_traj_<dit|rwkv>_ddpm_basis32_<blend05|blend1>.sh
    h200x8_4096_2.9B_prefix_suffix_traj_dit_rf.sh
    2nodes16_4096_2.9B_prefix_suffix_traj_dit_rf.sh
```

These scripts use PyTorch DDP data parallelism. Each rank still loads the full frozen RWKV backbone and RELAY trainable modules; multi-GPU improves throughput but does not shard model memory.

## Step scaling for more GPUs

`training.num_train_steps` is an optimizer-step count. By default it stays fixed when you increase GPUs, so a 16-GPU run with the same per-rank `BATCH_SIZE` sees 4x the samples/tokens of a 4-GPU run.

For a fixed sample/token budget, enable global-batch scaling in the Hydra overrides. Use the 4xH200 high-util baseline as reference with `reference_world_size=4` and `reference_train_batch_size=4`:

```bash
+training.step_scale_mode=global_batch \
+training.step_scale_reference_world_size=4 \
+training.step_scale_reference_train_batch_size=4
```

Example: requested `training.num_train_steps=150000`, 16 GPUs, `BATCH_SIZE=4` gives actual global batch `64`; the scaler runs `ceil(150000 * 16 / 64) = 37500` optimizer steps. If you want a fixed optimizer-step experiment instead, omit these overrides. The trainer always saves a final checkpoint even when the scaled step count is below `training.save_every_n_steps`.

## Unconditional examples

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv

# Dry-run first.
BATCH_SIZE=1 DRY_RUN=1 training/cluster/unconditional/h200x8_4096_2.9B_traj_dit_rf.sh

# Launch H200×8.
BATCH_SIZE=1 DRY_RUN=0 training/cluster/unconditional/h200x8_4096_2.9B_traj_dit_rf.sh

# 2 nodes × 8 GPUs: run on both nodes with shared MASTER_ADDR.
BATCH_SIZE=1 MASTER_ADDR=<node0-ip> NODE_RANK=0 DRY_RUN=0 \
  training/cluster/unconditional/2nodes16_4096_2.9B_traj_dit_rf.sh
BATCH_SIZE=1 MASTER_ADDR=<node0-ip> NODE_RANK=1 DRY_RUN=0 \
  training/cluster/unconditional/2nodes16_4096_2.9B_traj_dit_rf.sh
```

Use `STAGE=0|1|2` for unconditional staged launches.

## Prefix/suffix examples

Prefix/suffix routes reuse a 4096 trajectory S0 checkpoint, then run conditional S1/S2. Recommended `STATE_BLEND` comparison is `0.5` vs `1.0`. For H200×8 RWKV prefix/suffix 4096 S1, start with `BATCH_SIZE=1`; `BATCH_SIZE` is per GPU/rank, so `BATCH_SIZE=8` becomes global batch 64 with `NPROC_PER_NODE=8` and can OOM the 2.9B RWKV rollout.

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv

# RWKV DDPM basis32, blend=0.5.
BATCH_SIZE=1 DRY_RUN=0 STAGE=1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32_blend05.sh

# RWKV DDPM basis32, blend=1.0.
BATCH_SIZE=1 DRY_RUN=0 STAGE=1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32_blend1.sh

# After S1 completes, run S2 separately.
BATCH_SIZE=1 DRY_RUN=0 STAGE=2 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32_blend1.sh

# RWKV RF basis32, blend=0.5.
BATCH_SIZE=1 DRY_RUN=0 STAGE=1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_rf_basis32_blend05.sh

# RWKV RF basis32, blend=1.0.
BATCH_SIZE=1 DRY_RUN=0 STAGE=1 S0_CKPT=outputs_relay/fineweb4096-traj64x64-2.9B-s0/step_00050000 \
  training/cluster/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_rf_basis32_blend1.sh

# Generic launcher for BiRWKV.
BATCH_SIZE=8 DRY_RUN=0 STAGE=all BACKBONE=2.9B ARCH=birwkv OBJECTIVE=ddpm BASIS=32 STATE_BLEND=0.5 \
  training/cluster/conditional/prefix_suffix/launch_4096_prefix_suffix_trajectory_stage.sh
```

For multi-node prefix/suffix runs, do not use `STAGE=all`; run `STAGE=1` on both nodes, wait for S1, then run `STAGE=2` on both nodes.

```bash
# node 0, S1
BATCH_SIZE=1 MASTER_ADDR=<node0-ip> NODE_RANK=0 STAGE=1 DRY_RUN=0 \
  BACKBONE=2.9B ARCH=rwkv OBJECTIVE=ddpm BASIS=32 STATE_BLEND=0.5 NNODES=2 \
  training/cluster/conditional/prefix_suffix/launch_4096_prefix_suffix_trajectory_stage.sh

# node 1, S1
BATCH_SIZE=1 MASTER_ADDR=<node0-ip> NODE_RANK=1 STAGE=1 DRY_RUN=0 \
  BACKBONE=2.9B ARCH=rwkv OBJECTIVE=ddpm BASIS=32 STATE_BLEND=0.5 NNODES=2 \
  training/cluster/conditional/prefix_suffix/launch_4096_prefix_suffix_trajectory_stage.sh
```

Once `BATCH_SIZE=1` is stable for hundreds of steps, `BATCH_SIZE=2` is the next
reasonable probe. `BATCH_SIZE=4` may still OOM; do not start with it for RWKV
4096 S1. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only as a
fragmentation guard; it does not replace reducing per-rank batch size.

Common variables: `BACKBONE`, `ARCH`, `BASIS`, `OBJECTIVE`, `STAGE`, `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `BATCH_SIZE`, `NUM_WORKERS`, `S0_STEPS`, `S1_STEPS`, `S2_STEPS`, `SAVE_EVERY`, `STATE_BLEND`, `S0_CKPT`, `S1_CKPT`, `DRY_RUN`.
