#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=4096 BACKBONE=13.3B ARCH=birwkv BASIS=64 OBJECTIVE=rf exec "${SCRIPT_DIR}/../../../../../../../run_prefix_suffix_trajectory.sh" "$@"
