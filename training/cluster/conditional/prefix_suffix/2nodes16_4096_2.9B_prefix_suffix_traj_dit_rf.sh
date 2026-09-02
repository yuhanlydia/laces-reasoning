#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKBONE=2.9B ARCH=dit OBJECTIVE=rf BASIS="${BASIS:-16}" \
NNODES=2 NPROC_PER_NODE=8 NODE_RANK="${NODE_RANK:?Set NODE_RANK=0 on node0 and NODE_RANK=1 on node1}" \
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to node0 ip/hostname}" MASTER_PORT="${MASTER_PORT:-29500}" \
BATCH_SIZE="${BATCH_SIZE:-1}" DRY_RUN="${DRY_RUN:-1}" \
exec "${SCRIPT_DIR}/launch_4096_prefix_suffix_trajectory_stage.sh" "$@"
