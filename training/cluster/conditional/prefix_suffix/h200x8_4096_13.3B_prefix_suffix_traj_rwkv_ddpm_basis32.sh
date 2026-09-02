#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKBONE=13.3B ARCH=rwkv OBJECTIVE=ddpm BASIS=32 STATE_BLEND="${STATE_BLEND:-0.5}" \
NNODES=1 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
MASTER_PORT="${MASTER_PORT:-29500}" BATCH_SIZE="${BATCH_SIZE:-8}" DRY_RUN="${DRY_RUN:-1}" \
exec "${SCRIPT_DIR}/launch_4096_prefix_suffix_trajectory_stage.sh" "$@"
