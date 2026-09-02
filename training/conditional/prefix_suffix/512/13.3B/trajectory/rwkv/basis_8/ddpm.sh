#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=512 BACKBONE=13.3B ARCH=rwkv BASIS=8 OBJECTIVE=ddpm exec "${SCRIPT_DIR}/../../../../../../../run_prefix_suffix_trajectory.sh" "$@"
