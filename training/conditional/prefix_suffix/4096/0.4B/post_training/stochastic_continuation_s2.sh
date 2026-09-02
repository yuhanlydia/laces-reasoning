#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=4096 BACKBONE=0.4B ROUTE=stochastic_continuation_s2 exec "${SCRIPT_DIR}/../../../../../run_post_training_route.sh" "$@"
