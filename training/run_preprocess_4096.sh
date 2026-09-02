#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${MODE:?Set MODE=full_rows,pack_existing,or pack_stream}"
: "${MAX_LENGTH:=4096}"
: "${SHARD_SIZE:=100000}"
: "${CACHE_DIR:=./data/huggingface}"
: "${MODEL:=RWKV/RWKV7-Goose-World3-1.5B-HF}"

print_header "FineWeb 4096 preprocessing mode=${MODE}"

case "${MODE}" in
  full_rows)
    run_cmd python scripts/preprocess/preprocess_fineweb.py --model "${MODEL}" --cache_dir "${CACHE_DIR}" --local_files_only --max_length "${MAX_LENGTH}" --max_samples 0 --shard_size "${SHARD_SIZE}" --out_dir preprocessed_data/fineweb_4096_full --out_file fineweb_4096_tokenized.pkl
    ;;
  pack_existing)
    run_cmd python scripts/preprocess/preprocess_fineweb.py --pack_existing_token_dir preprocessed_data/fineweb_4096_full --max_length "${MAX_LENGTH}" --shard_size "${SHARD_SIZE}" --out_dir preprocessed_data/fineweb_4096_packed_full --out_file fineweb_4096_packed.pkl
    ;;
  pack_stream)
    run_cmd python scripts/preprocess/preprocess_fineweb.py --model "${MODEL}" --cache_dir "${CACHE_DIR}" --local_files_only --max_length "${MAX_LENGTH}" --max_samples 0 --shard_size "${SHARD_SIZE}" --pack_sequences --out_dir preprocessed_data/fineweb_4096_packed_full --out_file fineweb_4096_packed.pkl
    ;;
  *) echo "Unsupported MODE=${MODE}; use full_rows, pack_existing, or pack_stream" >&2; exit 2 ;;
esac
