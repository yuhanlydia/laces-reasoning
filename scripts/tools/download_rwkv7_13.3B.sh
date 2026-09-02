#!/bin/bash
# Download + convert RWKV-7 G1f 13.3B (BlinkDL .pth → HF format).
#
# Why two steps? BlinkDL publishes 13.3B only as a raw PyTorch .pth file,
# which the HF transformers loader (and therefore our train/sample scripts)
# can't read directly. This script downloads the .pth and converts it to
# HF format so it can be used identically to fla-hub/rwkv7-7.2B-g0.
#
# Disk usage: ~28 GB (14 GB .pth + 14 GB HF safetensors). After conversion
# you can delete the .pth file.
#
# Usage:
#   bash scripts/tools/download_rwkv7_13.3B.sh
#   PTH_DIR=/data/blinkdl OUT_DIR=/data/RWKV7-G1f-13.3B-HF bash scripts/tools/download_rwkv7_13.3B.sh
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "${PROJECT_DIR}"

# ── paths ──────────────────────────────────────────────────────
PTH_DIR="${PTH_DIR:-./checkpoints/blinkdl}"
OUT_DIR="${OUT_DIR:-./checkpoints/RWKV7-G1f-13.3B-HF}"
PTH_FILE="${PTH_FILE:-rwkv7-g1f-13.3b-20260415-ctx8192.pth}"
HF_REPO="${HF_REPO:-BlinkDL/rwkv7-g1}"   # BlinkDL's official upload repo
TOKENIZER_SRC="${TOKENIZER_SRC:-fla-hub/rwkv7-7.2B-g0}"   # same World tokenizer
CTX="${CTX:-8192}"
DTYPE="${DTYPE:-bf16}"

mkdir -p "${PTH_DIR}" "${OUT_DIR}"

# ── 1) Download .pth ──────────────────────────────────────────
if [ ! -f "${PTH_DIR}/${PTH_FILE}" ]; then
    echo "=========================================="
    echo "Downloading ${PTH_FILE} from ${HF_REPO}"
    echo "  → ${PTH_DIR}/${PTH_FILE}"
    echo "  ~14 GB; may take a while"
    echo "=========================================="
    huggingface-cli download "${HF_REPO}" "${PTH_FILE}" \
        --local-dir "${PTH_DIR}" \
        --local-dir-use-symlinks False
else
    echo "Already downloaded: ${PTH_DIR}/${PTH_FILE}"
fi

# ── 2) Convert to HF format ──────────────────────────────────
if [ ! -f "${OUT_DIR}/config.json" ]; then
    echo "=========================================="
    echo "Converting .pth → HF (${OUT_DIR})"
    echo "=========================================="
    python scripts/tools/convert_blinkdl_to_hf.py \
        --pth "${PTH_DIR}/${PTH_FILE}" \
        --out_dir "${OUT_DIR}" \
        --tokenizer_src "${TOKENIZER_SRC}" \
        --max_position_embeddings "${CTX}" \
        --dtype "${DTYPE}" \
        --verify
else
    echo "Already converted: ${OUT_DIR}/config.json"
fi

# ── 3) Patch configs to point at converted dir (both identity and rwkv) ──
ABS_OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
for CFG_PATH in configs/rwkv_relay_13.3B_identity.yaml configs/rwkv_relay_13.3B_rwkv.yaml; do
    if [ -f "${CFG_PATH}" ]; then
        echo "Patching ${CFG_PATH} → rwkv_local_path=${ABS_OUT_DIR}"
        sed -i.bak -E "s|rwkv_local_path: .*|rwkv_local_path: \"${ABS_OUT_DIR}\"|" "${CFG_PATH}"
        rm -f "${CFG_PATH}.bak"
    fi
done

# ── 4) Clear stale HF transformers_modules cache for this dir ──────
# HuggingFace caches custom-code (hf_rwkv_tokenizer.py) under
# ~/.cache/huggingface/modules/transformers_modules/<basename>/. If an
# earlier broken conversion populated that cache, AutoTokenizer will
# reuse the cached (broken) version even after we fix the local files.
# Clear it preemptively.
BN="$(basename "${OUT_DIR}" | tr '.-' '_' | sed 's|_|_hyphen_|g')"
CACHE_DIR="${HOME}/.cache/huggingface/modules/transformers_modules"
for d in "${CACHE_DIR}/${BN}" "${CACHE_DIR}/$(basename "${OUT_DIR}")"; do
    if [ -d "${d}" ]; then
        echo "Clearing stale HF cache: ${d}"
        rm -rf "${d}"
    fi
done

echo ""
echo "=========================================="
echo "✅ DONE"
echo "  HF model dir:  ${ABS_OUT_DIR}"
echo "  Configs patched: rwkv_relay_13.3B_{identity,rwkv}.yaml"
echo ""
echo "  Next: train RELAY-DiT"
echo "    MODE=identity    BACKBONE=13.3B bash scripts/train/train_multi_node_dit.sh"
echo "    MODE=variational BACKBONE=13.3B bash scripts/train/train_multi_node_dit.sh"
echo ""
echo "    sample:"
echo "    python scripts/sample/sample_relay_cot.py \\"
echo "        --ckpt_dir outputs_relay/relay-13.3B-cot/step_XXXXX \\"
echo "        --prompt 'The history of...' --num_samples 3"
echo "=========================================="
