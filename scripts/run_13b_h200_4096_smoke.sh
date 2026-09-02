#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DIFFRWKV_13B_MODEL_PATH="${DIFFRWKV_13B_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-G1f-13.3B-HF}"
export DIFFRWKV_4096_DATA_DIR_WAS_SET="${DIFFRWKV_4096_DATA_DIR+x}"
export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_test}"
export DIFFRWKV_CUDA_VISIBLE_DEVICES_WAS_SET="${CUDA_VISIBLE_DEVICES+x}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_PREFIX="${RUN_PREFIX:-smoke-traj64x64-13.3B-h200}"
export EVIDENCE_DIR="${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}"
export CONFIG_NAME="rwkv_relay_13.3B_state_hijack_dit_vae32_traj32x16"
export CONFIG_PATH="configs/${CONFIG_NAME}.yaml"
export DIFFRWKV_MIN_GPU_MEMORY_GIB="${DIFFRWKV_MIN_GPU_MEMORY_GIB:-139}"
export DIFFRWKV_REQUIRED_GPU_NAME="${DIFFRWKV_REQUIRED_GPU_NAME:-}"
export S0_RUN_NAME="${RUN_PREFIX}-s0"
export S1_RUN_NAME="${RUN_PREFIX}-s1"
export S2_RUN_NAME="${RUN_PREFIX}-s2"
export S0_DIR="outputs_relay/${S0_RUN_NAME}"
export S1_DIR="outputs_relay/${S1_RUN_NAME}"
export S2_DIR="outputs_relay/${S2_RUN_NAME}"

mkdir -p "$EVIDENCE_DIR"

print_context() {
  echo "CONFIG_NAME=$CONFIG_NAME"
  echo "DIFFRWKV_13B_MODEL_PATH=$DIFFRWKV_13B_MODEL_PATH"
  echo "DIFFRWKV_4096_DATA_DIR=$DIFFRWKV_4096_DATA_DIR"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "RUN_PREFIX=$RUN_PREFIX"
  echo "EVIDENCE_DIR=$EVIDENCE_DIR"
  echo "S0_DIR=$S0_DIR"
  echo "S1_DIR=$S1_DIR"
  echo "S2_DIR=$S2_DIR"
  echo "H200_S0_STEPS=${H200_S0_STEPS:-10}"
  echo "H200_S1_STEPS=${H200_S1_STEPS:-10}"
  echo "H200_S2_STEPS=${H200_S2_STEPS:-10}"
  echo "H200_GATE_SAMPLE_STEPS=${H200_GATE_SAMPLE_STEPS:-4}"
  echo "H200_GATE_MAX_NEW_TOKENS=${H200_GATE_MAX_NEW_TOKENS:-64}"
}

step_dir() {
  local stage_dir="$1"
  local step="$2"
  printf "%s/step_%08d" "$stage_dir" "$step"
}

checkpoint_path() {
  local stage_dir="$1"
  local step="$2"
  printf "%s/model.pt" "$(step_dir "$stage_dir" "$step")"
}

assert_clean_log() {
  local log_file="$1"
  local label="$2"
  LOG_FILE="$log_file" LOG_LABEL="$label" python - <<'PY'
import os
import re
from pathlib import Path

log = Path(os.environ["LOG_FILE"])
label = os.environ["LOG_LABEL"]
if not log.is_file():
    raise SystemExit(f"{label}: missing log {log}")
text = log.read_text(errors="ignore")
patterns = [
    r"(?i)(?<![a-z])nan(?![a-z])",
    r"(?i)(?<![a-z])[+-]?inf(?![a-z])",
    r"Traceback \(most recent call last\)",
    r"RuntimeError:",
    r"CUDA out of memory",
    r"out of memory",
    r"size mismatch",
    r"PytorchStreamReader failed",
    r"NCCL.*(?:error|failed|timeout|unhandled)",
    r"(?:rank|Rank).*(?:failed|died|crashed)",
]
for pattern in patterns:
    if re.search(pattern, text):
        raise SystemExit(f"{label}: bad log pattern matched: {pattern}")
print(f"{label}: clean log {log}")
PY
}

checkpoint_integrity() {
  local label="$1"
  local stage_dir="$2"
  local step="$3"
  local expected_stage="$4"
  local expected_gen_type="${5:-}"
  local output_file="$6"
  local ckpt
  ckpt="$(checkpoint_path "$stage_dir" "$step")"
  CHECKPOINT_PATH="$ckpt" EXPECTED_STEP="$step" EXPECTED_STAGE="$expected_stage" EXPECTED_GEN_TYPE="$expected_gen_type" python - <<'PY' 2>&1 | tee "$output_file"
import math
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

ckpt_path = Path(os.environ["CHECKPOINT_PATH"])
expected_step = int(os.environ["EXPECTED_STEP"])
expected_stage = int(os.environ["EXPECTED_STAGE"])
expected_gen_type = os.environ.get("EXPECTED_GEN_TYPE", "")

print("checkpoint", ckpt_path)
if not ckpt_path.is_file() or ckpt_path.stat().st_size <= 0:
    raise SystemExit("missing or empty checkpoint")

ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
if int(ckpt.get("step", -1)) != expected_step:
    raise SystemExit(f"unexpected checkpoint step {ckpt.get('step')} != {expected_step}")
if "config" not in ckpt:
    raise SystemExit("checkpoint missing config")
if "trainable_state" not in ckpt or not isinstance(ckpt["trainable_state"], dict) or not ckpt["trainable_state"]:
    raise SystemExit("checkpoint missing nonempty trainable_state")

cfg = OmegaConf.create(ckpt["config"])
checks = {
    "training.stage": int(cfg.training.stage) == expected_stage,
    "data.max_length": int(cfg.data.max_length) == 4096,
    "model.trajectory_mode": bool(cfg.model.trajectory_mode),
    "model.trajectory_chunk_size": int(cfg.model.trajectory_chunk_size) == 64,
    "model.trajectory_horizon": int(cfg.model.trajectory_horizon) == 64,
}
if expected_gen_type:
    checks["training.gen_type"] = str(cfg.training.get("gen_type", "")) == expected_gen_type
for name, ok in checks.items():
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"checkpoint config check failed: {name}")

tensor_count = 0
numel = 0
for key, value in ckpt["trainable_state"].items():
    if not torch.is_tensor(value):
        continue
    tensor_count += 1
    numel += value.numel()
    if value.dtype.is_floating_point and not torch.isfinite(value).all().item():
        raise SystemExit(f"non-finite tensor in checkpoint: {key}")
if tensor_count == 0 or numel == 0:
    raise SystemExit("checkpoint trainable_state has no tensors")
print("checkpoint_step", ckpt.get("step"))
print("tensor_count", tensor_count)
print("tensor_numel", numel)
print("CHECKPOINT_INTEGRITY: PASS")
PY
  return "${PIPESTATUS[0]}"
}

resume_log_check() {
  local label="$1"
  local log_file="$2"
  local expected_ckpt="$3"
  local expected_step="$4"
  local output_file="$5"
  RESUME_LABEL="$label" RESUME_LOG="$log_file" RESUME_CKPT="$expected_ckpt" RESUME_STEP="$expected_step" python - <<'PY' 2>&1 | tee "$output_file"
import os
from pathlib import Path

label = os.environ["RESUME_LABEL"]
log = Path(os.environ["RESUME_LOG"])
expected_ckpt = os.environ["RESUME_CKPT"]
expected_step = os.environ["RESUME_STEP"]
print("resume_label", label)
print("resume_log", log)
print("expected_checkpoint", expected_ckpt)
if not log.is_file():
    raise SystemExit("missing resume log")
text = log.read_text(errors="ignore")
required = [
    f"Resuming from {expected_ckpt}",
    f"Loaded checkpoint at step {expected_step}",
]
for needle in required:
    ok = needle in text
    print(f"contains {needle!r}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"missing resume evidence: {needle}")
print("RESUME_VALIDATION: PASS")
PY
  return "${PIPESTATUS[0]}"
}

trajectory-data-check() {
  set -euo pipefail
  export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_test}"
  export EVIDENCE_DIR="${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}"
  export H200_DATA_CHECK_SAMPLES="${H200_DATA_CHECK_SAMPLES:-2048}"
  mkdir -p "$EVIDENCE_DIR"
  python - <<'PY' 2>&1 | tee "$EVIDENCE_DIR/trajectory-data-check.txt"
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np

data_dir = Path(os.environ["DIFFRWKV_4096_DATA_DIR"])
max_samples = int(os.environ.get("H200_DATA_CHECK_SAMPLES", "2048"))
expected_len = 4096
chunk = 64
horizon = 64

print("trajectory_data_dir", data_dir)
print("expected_len", expected_len)
print("trajectory_chunk_size", chunk)
print("trajectory_horizon", horizon)
print("sample_check_limit", max_samples)
if expected_len != chunk * horizon:
    raise SystemExit("internal harness mismatch: expected_len != chunk*horizon")
if not data_dir.is_dir():
    raise SystemExit(
        f"missing trajectory data dir {data_dir}; create packed data with: "
        "python scripts/preprocess/preprocess_fineweb.py --max_length 4096 --pack_sequences "
        "--out_dir preprocessed_data/fineweb_4096_packed_full --out_file fineweb_4096_packed.pkl"
    )

pkl_paths = sorted(data_dir.glob("*.pkl"))
print("pkl_count", len(pkl_paths))
if not pkl_paths:
    raise SystemExit("trajectory data dir has no .pkl shards")

manifest_paths = sorted(data_dir.glob("*_manifest.json"))
if manifest_paths:
    manifest_path = manifest_paths[0]
    manifest = json.loads(manifest_path.read_text())
    print("manifest", manifest_path)
    print("manifest_pack_sequences", manifest.get("pack_sequences"))
    print("manifest_max_length", manifest.get("max_length"))
    print("manifest_total_samples", manifest.get("total_samples"))
    if manifest.get("pack_sequences") is not True:
        raise SystemExit("trajectory 64x64 requires manifest pack_sequences=true")
    if int(manifest.get("max_length", -1)) != expected_len:
        raise SystemExit("trajectory packed data manifest max_length must be 4096")
    if int(manifest.get("total_samples", 0)) <= 0:
        raise SystemExit("trajectory packed data manifest total_samples must be positive")
    for shard in manifest.get("shards", []):
        shard_path = data_dir / shard.get("path", "")
        if not shard_path.is_file():
            raise SystemExit(f"manifest shard missing: {shard_path}")
else:
    print("manifest", "NONE")

checked = 0
min_len = expected_len
min_mask_sum = expected_len
bad_examples = []
for pkl_path in pkl_paths:
    if checked >= max_samples:
        break
    print("checking_pkl", pkl_path)
    with pkl_path.open("rb") as f:
        shard = pickle.load(f)
    for sample_idx, sample in enumerate(shard):
        if checked >= max_samples:
            break
        ids = sample["input_ids"] if isinstance(sample, dict) else sample[0]
        mask = sample["attention_mask"] if isinstance(sample, dict) else sample[1]
        ids = np.asarray(ids)
        mask = np.asarray(mask).astype(bool)
        length = int(ids.shape[-1])
        mask_sum = int(mask.sum())
        min_len = min(min_len, length)
        min_mask_sum = min(min_mask_sum, mask_sum)
        if length != expected_len or mask.shape[-1] != expected_len or mask_sum != expected_len or not bool(mask.all()):
            bad_examples.append((str(pkl_path), sample_idx, length, int(mask.shape[-1]), mask_sum))
            if len(bad_examples) >= 5:
                break
        checked += 1
    if bad_examples:
        break

print("checked_samples", checked)
print("min_seq_len", min_len)
print("min_attention_mask_sum", min_mask_sum)
if checked <= 0:
    raise SystemExit("no samples checked")
if bad_examples:
    print("bad_examples", bad_examples)
    raise SystemExit(
        "trajectory 64x64 requires packed fixed-length FineWeb: every inspected sample must have "
        "len(input_ids)=4096 and all-true attention_mask. Per-row fineweb_4096/fineweb_4096_full is invalid."
    )
print("TRAJECTORY_DATA_CHECK: PASS")
PY
}

preflight() {
  set -euo pipefail
  export WANDB_MODE=disabled
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export DIFFRWKV_13B_MODEL_PATH="${DIFFRWKV_13B_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-G1f-13.3B-HF}"
  export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_test}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RUN_PREFIX="${RUN_PREFIX:-smoke-traj64x64-13.3B-h200}"
  export EVIDENCE_DIR="${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}"
  mkdir -p "$EVIDENCE_DIR"
  {
    date
    pwd
    print_context
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    python - <<'PY'
import os, sys, torch
print('python', sys.version)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(i, props.name, props.total_memory // (1024**3), 'GiB')
PY
    python - <<'PY'
import glob, os
model=os.environ['DIFFRWKV_13B_MODEL_PATH']
data=os.environ['DIFFRWKV_4096_DATA_DIR']
pkls=sorted(glob.glob(os.path.join(data, '*.pkl'))) if os.path.isdir(data) else []
print('model_path', model, os.path.isdir(model))
print('data_path', data, os.path.isdir(data))
print('pkl_count', len(pkls))
print('first_pkl', pkls[0] if pkls else 'NONE')
if not os.path.isdir(model): raise SystemExit('missing model path')
if not os.path.isdir(data): raise SystemExit('missing data dir')
if not pkls: raise SystemExit('missing pkl shard/file')
PY
  } 2>&1 | tee "$EVIDENCE_DIR/preflight.txt"
}

config-check() {
  set -euo pipefail
  export WANDB_MODE=disabled
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export DIFFRWKV_13B_MODEL_PATH="${DIFFRWKV_13B_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-G1f-13.3B-HF}"
  export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_test}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RUN_PREFIX="${RUN_PREFIX:-smoke-traj64x64-13.3B-h200}"
  export EVIDENCE_DIR="${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}"
  export S0_RUN_NAME="${RUN_PREFIX}-s0"
  export S1_RUN_NAME="${RUN_PREFIX}-s1"
  export S2_RUN_NAME="${RUN_PREFIX}-s2"
  mkdir -p "$EVIDENCE_DIR"
  python - <<'PY' 2>&1 | tee "$EVIDENCE_DIR/config-check.txt"
import os
from omegaconf import OmegaConf

config_name = 'rwkv_relay_13.3B_state_hijack_dit_vae32_traj32x16'
cfg = OmegaConf.load(os.path.join('configs', f'{config_name}.yaml'))
overrides = OmegaConf.create({
    'data': {
        'token_dir': os.environ['DIFFRWKV_4096_DATA_DIR'],
        'max_length': 4096,
        'num_workers': 0,
    },
    'model': {
        'rwkv_local_path': os.environ['DIFFRWKV_13B_MODEL_PATH'],
        'trajectory_chunk_size': 64,
        'trajectory_horizon': 64,
    },
    'training': {
        'train_batch_size': 1,
    },
    'logging': {
        'run_name': os.environ['S0_RUN_NAME'],
    },
})
resolved = OmegaConf.merge(cfg, overrides)
run_names = [os.environ['S0_RUN_NAME'], os.environ['S1_RUN_NAME'], os.environ['S2_RUN_NAME']]

print('config_name', resolved.config_name)
print('trajectory_mode', bool(resolved.model.trajectory_mode))
print('trajectory_chunk_size', int(resolved.model.trajectory_chunk_size))
print('trajectory_horizon', int(resolved.model.trajectory_horizon))
print('data_max_length', int(resolved.data.max_length))
print('token_dir', resolved.data.token_dir)
print('rwkv_local_path', resolved.model.rwkv_local_path)
print('train_batch_size', int(resolved.training.train_batch_size))
print('data_num_workers', int(resolved.data.num_workers))
print('s0_run_name', run_names[0])
print('s1_run_name', run_names[1])
print('s2_run_name', run_names[2])

if resolved.config_name != config_name:
    raise SystemExit('wrong config_name')
if not bool(resolved.model.trajectory_mode):
    raise SystemExit('trajectory mode disabled')
if int(resolved.data.max_length) != 4096:
    raise SystemExit('expected data.max_length 4096')
if int(resolved.model.trajectory_chunk_size) != 64:
    raise SystemExit('expected trajectory_chunk_size 64')
if int(resolved.model.trajectory_horizon) != 64:
    raise SystemExit('expected trajectory_horizon 64')
if 'fineweb_4096' not in str(resolved.data.token_dir):
    raise SystemExit('expected FineWeb 4096 token_dir')
if 'owt_rwkv_tokens' in str(resolved.data.token_dir):
    raise SystemExit('unexpected OWT token_dir after overrides')
if bool(resolved.model.trajectory_mode) and 'packed' not in str(resolved.data.token_dir):
    raise SystemExit('trajectory 64x64 requires packed FineWeb 4096 token_dir')
if len(set(run_names)) != 3 or any('test-v6-13.3B' in name for name in run_names):
    raise SystemExit('run names are not unique smoke names')
if any('traj32x16' in name for name in run_names):
    raise SystemExit('run name suggests old traj32x16 checkpoint path')
PY
}

tokenizer-check() {
  export DIFFRWKV_13B_MODEL_PATH="${DIFFRWKV_13B_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-G1f-13.3B-HF}"
  export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_test}"
  export EVIDENCE_DIR="${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}"
  export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  set -euo pipefail
  mkdir -p "$EVIDENCE_DIR"
  python - <<'PY' 2>&1 | tee "$EVIDENCE_DIR/tokenizer_decode.txt"
import glob, os, pickle, numpy as np
from transformers import AutoTokenizer
model = os.environ['DIFFRWKV_13B_MODEL_PATH']
data_dir = os.environ['DIFFRWKV_4096_DATA_DIR']
pkls = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))
if not pkls: raise SystemExit('missing pkl shard/file')
data_file = pkls[0]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
with open(data_file, 'rb') as f:
    data = pickle.load(f)
first = data[0]
ids = first['input_ids'] if isinstance(first, dict) else first[0]
ids = np.asarray(ids)
print('num_samples_loaded', len(data))
print('data_file', data_file)
print('seq_len', int(ids.shape[0]))
print('min_id', int(ids.min()))
print('max_id', int(ids.max()))
print('vocab_size', getattr(tok, 'vocab_size', 'unknown'))
print('decoded_head')
print(tok.decode(ids[:128].tolist()))
if ids.shape[0] != 4096: raise SystemExit('expected seq_len 4096')
if getattr(tok, 'vocab_size', None) is not None and int(ids.max()) >= int(tok.vocab_size):
    raise SystemExit('token id exceeds tokenizer vocab size')
sample_count = min(256, len(data))
min_seq_len = 4096
min_mask_sum = 4096
for idx in range(sample_count):
    sample = data[idx]
    sample_ids = sample['input_ids'] if isinstance(sample, dict) else sample[0]
    sample_mask = sample['attention_mask'] if isinstance(sample, dict) else sample[1]
    sample_ids = np.asarray(sample_ids)
    sample_mask = np.asarray(sample_mask).astype(bool)
    min_seq_len = min(min_seq_len, int(sample_ids.shape[0]))
    min_mask_sum = min(min_mask_sum, int(sample_mask.sum()))
    if sample_ids.shape[0] != 4096 or sample_mask.shape[0] != 4096 or int(sample_mask.sum()) != 4096:
        raise SystemExit(f'packed trajectory sample check failed at index {idx}: seq_len={sample_ids.shape[0]} mask_sum={int(sample_mask.sum())}')
print('packed_sample_count_checked', sample_count)
print('packed_min_seq_len', min_seq_len)
print('packed_min_mask_sum', min_mask_sum)
PY
}

s0() {
  set -euo pipefail
  export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export DIFFRWKV_13B_MODEL_PATH="${DIFFRWKV_13B_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-G1f-13.3B-HF}"
  export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_test}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RUN_PREFIX="${RUN_PREFIX:-smoke-traj64x64-13.3B-h200}"
  export EVIDENCE_DIR="${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}"
  export S0_RUN_NAME="${RUN_PREFIX}-s0"
  export S0_DIR="outputs_relay/${S0_RUN_NAME}"
  mkdir -p "$EVIDENCE_DIR"
  test ! -e "$S0_DIR"
  (while true; do nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader; sleep 5; done) > "$EVIDENCE_DIR/s0-memory.txt" & MON_PID=$!
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -u train_state_hijacking_dit.py \
    --config-name rwkv_relay_13.3B_state_hijack_dit_vae32_traj32x16 \
    model.rwkv_local_path="$DIFFRWKV_13B_MODEL_PATH" \
    data.token_dir="$DIFFRWKV_4096_DATA_DIR" data.max_length=4096 \
    model.trajectory_chunk_size=64 model.trajectory_horizon=64 \
    training.stage=3 training.num_train_steps=10 training.save_every_n_steps=10 logging.log_freq=1 \
    training.train_batch_size=1 data.num_workers=0 \
    logging.run_name="$S0_RUN_NAME" \
    2>&1 | tee "$EVIDENCE_DIR/s0-train.txt"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  return "$TRAIN_STATUS"
}

s1() {
  set -euo pipefail
  export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export DIFFRWKV_13B_MODEL_PATH="${DIFFRWKV_13B_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-G1f-13.3B-HF}"
  export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_test}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RUN_PREFIX="${RUN_PREFIX:-smoke-traj64x64-13.3B-h200}"
  export EVIDENCE_DIR="${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}"
  export S0_RUN_NAME="${RUN_PREFIX}-s0"
  export S1_RUN_NAME="${RUN_PREFIX}-s1"
  export S0_DIR="outputs_relay/${S0_RUN_NAME}"
  export S1_DIR="outputs_relay/${S1_RUN_NAME}"
  mkdir -p "$EVIDENCE_DIR"
  test -f "$S0_DIR/step_00000010/model.pt"
  test ! -e "$S1_DIR"
  (while true; do nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader; sleep 5; done) > "$EVIDENCE_DIR/s1-memory.txt" & MON_PID=$!
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -u train_state_hijacking_dit.py \
    --config-name rwkv_relay_13.3B_state_hijack_dit_vae32_traj32x16 \
    model.rwkv_local_path="$DIFFRWKV_13B_MODEL_PATH" \
    data.token_dir="$DIFFRWKV_4096_DATA_DIR" data.max_length=4096 \
    model.trajectory_chunk_size=64 model.trajectory_horizon=64 \
    training.stage=1 training.num_train_steps=10 training.save_every_n_steps=10 logging.log_freq=1 \
    training.resume="$S0_DIR/step_00000010" \
    model.trajectory_s1_mode=transformer model.trajectory_state_blend=0.7 \
    training.train_batch_size=1 data.num_workers=0 \
    logging.run_name="$S1_RUN_NAME" \
    2>&1 | tee "$EVIDENCE_DIR/s1-train.txt"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  return "$TRAIN_STATUS"
}

s2() {
  set -euo pipefail
  export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export DIFFRWKV_13B_MODEL_PATH="${DIFFRWKV_13B_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-G1f-13.3B-HF}"
  export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_test}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export RUN_PREFIX="${RUN_PREFIX:-smoke-traj64x64-13.3B-h200}"
  export EVIDENCE_DIR="${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}"
  export S1_RUN_NAME="${RUN_PREFIX}-s1"
  export S2_RUN_NAME="${RUN_PREFIX}-s2"
  export S1_DIR="outputs_relay/${S1_RUN_NAME}"
  export S2_DIR="outputs_relay/${S2_RUN_NAME}"
  mkdir -p "$EVIDENCE_DIR"
  test -f "$S1_DIR/step_00000010/model.pt"
  test ! -e "$S2_DIR"
  (while true; do nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader; sleep 5; done) > "$EVIDENCE_DIR/s2-memory.txt" & MON_PID=$!
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -u train_state_hijacking_dit.py \
    --config-name rwkv_relay_13.3B_state_hijack_dit_vae32_traj32x16 \
    model.rwkv_local_path="$DIFFRWKV_13B_MODEL_PATH" \
    data.token_dir="$DIFFRWKV_4096_DATA_DIR" data.max_length=4096 \
    model.trajectory_chunk_size=64 model.trajectory_horizon=64 \
    training.stage=2 training.gen_type=rf training.num_train_steps=10 training.save_every_n_steps=10 logging.log_freq=1 \
    training.resume="$S1_DIR/step_00000010" \
    loss.trajectory_delta_loss_weight=0.02 \
    model.trajectory_s1_mode=transformer model.trajectory_denoiser_type=dit model.trajectory_state_blend=0.7 \
    model.dit_hidden=768 model.dit_depth=8 model.dit_num_heads=8 \
    model.trajectory_state_hidden=256 model.trajectory_state_depth=4 model.trajectory_state_num_heads=4 \
    training.train_batch_size=1 data.num_workers=0 \
    logging.run_name="$S2_RUN_NAME" \
    2>&1 | tee "$EVIDENCE_DIR/s2-train.txt"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  return "$TRAIN_STATUS"
}

report() {
  set -euo pipefail
  export EVIDENCE_DIR="${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}"
  export RUN_PREFIX="${RUN_PREFIX:-smoke-traj64x64-13.3B-h200}"
  export S0_RUN_NAME="${RUN_PREFIX}-s0"
  export S1_RUN_NAME="${RUN_PREFIX}-s1"
  export S2_RUN_NAME="${RUN_PREFIX}-s2"
  export S0_DIR="outputs_relay/${S0_RUN_NAME}"
  export S1_DIR="outputs_relay/${S1_RUN_NAME}"
  export S2_DIR="outputs_relay/${S2_RUN_NAME}"
  mkdir -p "$EVIDENCE_DIR"
  python - <<'PY'
import os
from pathlib import Path

evidence = Path(os.environ['EVIDENCE_DIR'])
run_prefix = os.environ['RUN_PREFIX']
s0_dir = Path(os.environ['S0_DIR'])
s1_dir = Path(os.environ['S1_DIR'])
s2_dir = Path(os.environ['S2_DIR'])
report_path = evidence / 'final-report.md'

def exists(rel):
    return (evidence / rel).is_file()

def nonempty(rel):
    p = evidence / rel
    return p.is_file() and p.stat().st_size > 0

def checkpoint(stage_dir):
    return stage_dir / 'step_00000010' / 'model.pt'

def train_log(stage_dir):
    return stage_dir / 'train.log'

def status(ok):
    return 'PASS' if ok else 'FAIL'

def has_bad_log(stage_dir):
    log = train_log(stage_dir)
    if not log.is_file():
        return True
    text = log.read_text(errors='ignore')
    needles = ['nan', 'NaN', 'inf', 'OOM', 'out of memory', 'size mismatch']
    return any(n in text for n in needles)

criteria = {
    'Preflight': exists('preflight.txt'),
    'Config check': exists('config-check.txt'),
    'Tokenizer check': exists('tokenizer_decode.txt'),
    'Trajectory data check': exists('trajectory-data-check.txt') and 'TRAJECTORY_DATA_CHECK: PASS' in (evidence / 'trajectory-data-check.txt').read_text(errors='ignore'),
    'S0': checkpoint(s0_dir).is_file() and train_log(s0_dir).is_file() and nonempty('s0-memory.txt') and not has_bad_log(s0_dir),
    'S1': checkpoint(s1_dir).is_file() and train_log(s1_dir).is_file() and nonempty('s1-memory.txt') and not has_bad_log(s1_dir),
    'S2': checkpoint(s2_dir).is_file() and train_log(s2_dir).is_file() and nonempty('s2-memory.txt') and not has_bad_log(s2_dir),
}
all_pass = all(criteria.values())
recommendation = 'All smoke criteria passed; consider a separate production-training plan.' if all_pass else 'Do not start full training; fix failed smoke criteria first.'

lines = [
    '# 13.3B H200 4096 Smoke Final Report',
    '',
    f'Run prefix: `{run_prefix}`',
    f'Evidence directory: `{evidence}`',
    '',
    'This 10-step smoke validates plumbing only; it does not validate generation quality or convergence.',
    '',
    '## Criteria',
]
for name, ok in criteria.items():
    lines.append(f'- {name}: {status(ok)}')
lines.extend([
    '',
    '## Checkpoints',
    f'- S0: `{checkpoint(s0_dir)}` exists={checkpoint(s0_dir).is_file()}',
    f'- S1: `{checkpoint(s1_dir)}` exists={checkpoint(s1_dir).is_file()}',
    f'- S2: `{checkpoint(s2_dir)}` exists={checkpoint(s2_dir).is_file()}',
    '',
    '## Memory Evidence',
    f'- S0 memory: `{evidence / "s0-memory.txt"}` exists={nonempty("s0-memory.txt")}',
    f'- S1 memory: `{evidence / "s1-memory.txt"}` exists={nonempty("s1-memory.txt")}',
    f'- S2 memory: `{evidence / "s2-memory.txt"}` exists={nonempty("s2-memory.txt")}',
    '',
    '## Next Recommendation',
    recommendation,
    '',
    '## How to inspect',
    f'- Preflight: `{evidence / "preflight.txt"}`',
    f'- Config: `{evidence / "config-check.txt"}`',
    f'- Tokenizer: `{evidence / "tokenizer_decode.txt"}`',
    f'- Trajectory data: `{evidence / "trajectory-data-check.txt"}`',
    f'- S0 train log: `{evidence / "s0-train.txt"}`',
    f'- S1 train log: `{evidence / "s1-train.txt"}`',
    f'- S2 train log: `{evidence / "s2-train.txt"}`',
    '',
])
report_path.write_text('\n'.join(lines))
print(f'wrote {report_path}')
print(f'overall {status(all_pass)}')
print(f'S0 checkpoint {checkpoint(s0_dir)}')
print(f'S1 checkpoint {checkpoint(s1_dir)}')
print(f'S2 checkpoint {checkpoint(s2_dir)}')
PY
}

integrated_env() {
  set -euo pipefail
  export WANDB_MODE=disabled
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export DIFFRWKV_13B_MODEL_PATH="${DIFFRWKV_13B_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-G1f-13.3B-HF}"
  export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_test}"
  if [ -z "${DIFFRWKV_CUDA_VISIBLE_DEVICES_WAS_SET:-}" ]; then
    local detected_cuda_devices
    detected_cuda_devices=$(python - <<'PY'
try:
    import subprocess
    lines = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
        text=True,
    ).strip().splitlines()
    print(','.join(line.strip() for line in lines if line.strip()))
except Exception:
    print('')
PY
)
    export CUDA_VISIBLE_DEVICES="${detected_cuda_devices:-0}"
  else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  fi
  export DIFFRWKV_MIN_GPU_MEMORY_GIB="${DIFFRWKV_MIN_GPU_MEMORY_GIB:-139}"
  export DIFFRWKV_REQUIRED_GPU_NAME="${DIFFRWKV_REQUIRED_GPU_NAME:-}"
  local visible_cuda_count
  visible_cuda_count=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
  if [ "${visible_cuda_count:-0}" -lt 1 ]; then
    visible_cuda_count=1
  fi
  export H200_NPROC_PER_NODE="${H200_NPROC_PER_NODE:-$visible_cuda_count}"
  export H200_MASTER_PORT="${H200_MASTER_PORT:-29517}"
  if [ "${RUN_PREFIX:-smoke-traj64x64-13.3B-h200}" = "smoke-traj64x64-13.3B-h200" ]; then
    export RUN_PREFIX="integrated-traj64x64-13.3B-adaptive-${H200_NPROC_PER_NODE}gpu"
  fi
  if [ "${EVIDENCE_DIR:-.omo/evidence/13b-h200-4096-smoke}" = ".omo/evidence/13b-h200-4096-smoke" ]; then
    export EVIDENCE_DIR=".omo/evidence/13b-h200-4096-integrated"
  fi
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
  export H200_S0_STEPS="${H200_S0_STEPS:-10}"
  export H200_S1_STEPS="${H200_S1_STEPS:-10}"
  export H200_S2_STEPS="${H200_S2_STEPS:-10}"
  export H200_GATE_SAMPLE_STEPS="${H200_GATE_SAMPLE_STEPS:-4}"
  export H200_GATE_MAX_NEW_TOKENS="${H200_GATE_MAX_NEW_TOKENS:-64}"
  export S0_RUN_NAME="${RUN_PREFIX}-s0"
  export S1_RUN_NAME="${RUN_PREFIX}-s1"
  export S2_RUN_NAME="${RUN_PREFIX}-s2"
  export S0_DIR="outputs_relay/${S0_RUN_NAME}"
  export S1_DIR="outputs_relay/${S1_RUN_NAME}"
  export S2_DIR="outputs_relay/${S2_RUN_NAME}"
  mkdir -p "$EVIDENCE_DIR"
}

production_env() {
  set -euo pipefail
  local default_run=0
  local default_evidence=0
  case "${RUN_PREFIX:-}" in
    smoke-traj64x64-13.3B-h200|integrated-traj64x64-13.3B-adaptive-*gpu) default_run=1 ;;
  esac
  case "${EVIDENCE_DIR:-}" in
    .omo/evidence/13b-h200-4096-smoke|.omo/evidence/13b-h200-4096-integrated) default_evidence=1 ;;
  esac
  if [ -z "${H200_S0_STEPS+x}" ]; then export H200_S0_STEPS=50000; fi
  if [ -z "${H200_S1_STEPS+x}" ]; then export H200_S1_STEPS=50000; fi
  if [ -z "${H200_S2_STEPS+x}" ]; then export H200_S2_STEPS=150000; fi
  if [ -z "${H200_GATE_SAMPLE_STEPS+x}" ]; then export H200_GATE_SAMPLE_STEPS=100; fi
  if [ -z "${H200_GATE_MAX_NEW_TOKENS+x}" ]; then export H200_GATE_MAX_NEW_TOKENS=4096; fi
  if [ -z "${DIFFRWKV_4096_DATA_DIR_WAS_SET:-}" ]; then
    export DIFFRWKV_4096_DATA_DIR="${DIFFRWKV_PRODUCTION_4096_DATA_DIR:-preprocessed_data/fineweb_4096_packed_full}"
  fi
  integrated_env
  if [ "$default_run" -eq 1 ]; then
    export RUN_PREFIX="production-traj64x64-13.3B-adaptive-${H200_NPROC_PER_NODE}gpu"
  fi
  if [ "$default_evidence" -eq 1 ]; then
    export EVIDENCE_DIR=".omo/evidence/13b-h200-4096-production"
  fi
  export S0_RUN_NAME="${RUN_PREFIX}-s0"
  export S1_RUN_NAME="${RUN_PREFIX}-s1"
  export S2_RUN_NAME="${RUN_PREFIX}-s2"
  export S0_DIR="outputs_relay/${S0_RUN_NAME}"
  export S1_DIR="outputs_relay/${S1_RUN_NAME}"
  export S2_DIR="outputs_relay/${S2_RUN_NAME}"
  mkdir -p "$EVIDENCE_DIR"
}

require_adaptive_gpus() {
  set -euo pipefail
  integrated_env
  local hardware_log="$EVIDENCE_DIR/integrated-hardware.txt"
  local fail_log="$EVIDENCE_DIR/integrated-hardware-fail.txt"
  set +e
python - <<'PY' 2>&1 | tee "$hardware_log"
import os, sys, torch

required = int(os.environ['H200_NPROC_PER_NODE'])
min_memory_gib = float(os.environ.get('DIFFRWKV_MIN_GPU_MEMORY_GIB', '139'))
required_name = os.environ.get('DIFFRWKV_REQUIRED_GPU_NAME', '')
print('phase h200-integrated adaptive hardware validation')
print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
print('H200_NPROC_PER_NODE', required)
print('H200_MASTER_PORT', os.environ.get('H200_MASTER_PORT', ''))
print('DIFFRWKV_MIN_GPU_MEMORY_GIB', min_memory_gib)
print('DIFFRWKV_REQUIRED_GPU_NAME', required_name or 'ANY')
print('cuda_available', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; refusing to launch torchrun')

count = torch.cuda.device_count()
print('cuda_device_count', count)
failures = []
if count < 1:
    failures.append('expected at least 1 visible CUDA device')
if required > count:
    failures.append(f'H200_NPROC_PER_NODE={required} exceeds visible CUDA devices={count}')
if required < 1:
    failures.append(f'H200_NPROC_PER_NODE={required} must be at least 1')

validated_count = min(required, count)
print('validated_cuda_device_count', validated_count)
for idx in range(validated_count):
    props = torch.cuda.get_device_properties(idx)
    total_gib = props.total_memory / (1024 ** 3)
    print(idx, props.name, f'{total_gib:.2f}', 'GiB')
    if required_name and required_name not in props.name:
        failures.append(f'gpu {idx} name {props.name!r} does not contain {required_name!r}')
    if total_gib < min_memory_gib:
        failures.append(f'gpu {idx} memory {total_gib:.2f} GiB is below {min_memory_gib:.2f} GiB')

if failures:
    print('H200_HARDWARE_VERDICT: FAIL')
    for failure in failures:
        print('failure:', failure)
    raise SystemExit('hardware validation failed before torchrun')

print('H200_HARDWARE_VERDICT: PASS')
PY
  local status=${PIPESTATUS[0]}
  set -e
  if [ "$status" -ne 0 ]; then
    cp "$hardware_log" "$fail_log"
    return "$status"
  fi
  rm -f "$fail_log"
}

run_integrated_s0() {
  set -euo pipefail
  integrated_env
  test ! -e "$S0_DIR"
  (while true; do nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader; sleep 5; done) > "$EVIDENCE_DIR/integrated-s0-memory.txt" & MON_PID=$!
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nnodes=1 --nproc_per_node="$H200_NPROC_PER_NODE" --node_rank=0 --master_addr=localhost --master_port="$H200_MASTER_PORT" train_state_hijacking_dit.py \
    --config-name rwkv_relay_13.3B_state_hijack_dit_vae32_traj32x16 \
    model.rwkv_local_path="$DIFFRWKV_13B_MODEL_PATH" \
    data.token_dir="$DIFFRWKV_4096_DATA_DIR" data.max_length=4096 \
    model.trajectory_chunk_size=64 model.trajectory_horizon=64 \
    training.stage=3 training.num_train_steps="$H200_S0_STEPS" training.save_every_n_steps="$H200_S0_STEPS" logging.log_freq=1 \
    training.train_batch_size=1 data.num_workers=0 \
    logging.run_name="$S0_RUN_NAME" \
    2>&1 | tee "$EVIDENCE_DIR/integrated-s0-train.txt"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  if [ "$TRAIN_STATUS" -eq 0 ]; then
    assert_clean_log "$EVIDENCE_DIR/integrated-s0-train.txt" "S0 shell" || TRAIN_STATUS=$?
  fi
  if [ "$TRAIN_STATUS" -eq 0 ]; then
    assert_clean_log "$S0_DIR/train.log" "S0 train.log" || TRAIN_STATUS=$?
  fi
  return "$TRAIN_STATUS"
}

run_integrated_s1() {
  set -euo pipefail
  integrated_env
  test -s "$(checkpoint_path "$S0_DIR" "$H200_S0_STEPS")"
  test ! -e "$S1_DIR"
  (while true; do nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader; sleep 5; done) > "$EVIDENCE_DIR/integrated-s1-memory.txt" & MON_PID=$!
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nnodes=1 --nproc_per_node="$H200_NPROC_PER_NODE" --node_rank=0 --master_addr=localhost --master_port="$H200_MASTER_PORT" train_state_hijacking_dit.py \
    --config-name rwkv_relay_13.3B_state_hijack_dit_vae32_traj32x16 \
    model.rwkv_local_path="$DIFFRWKV_13B_MODEL_PATH" \
    data.token_dir="$DIFFRWKV_4096_DATA_DIR" data.max_length=4096 \
    model.trajectory_chunk_size=64 model.trajectory_horizon=64 \
    training.stage=1 training.num_train_steps="$H200_S1_STEPS" training.save_every_n_steps="$H200_S1_STEPS" logging.log_freq=1 \
    training.resume="$(step_dir "$S0_DIR" "$H200_S0_STEPS")" \
    model.trajectory_s1_mode=transformer model.trajectory_state_blend=0.7 \
    training.train_batch_size=1 data.num_workers=0 \
    logging.run_name="$S1_RUN_NAME" \
    2>&1 | tee "$EVIDENCE_DIR/integrated-s1-train.txt"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  if [ "$TRAIN_STATUS" -eq 0 ]; then
    assert_clean_log "$EVIDENCE_DIR/integrated-s1-train.txt" "S1 shell" || TRAIN_STATUS=$?
  fi
  if [ "$TRAIN_STATUS" -eq 0 ]; then
    assert_clean_log "$S1_DIR/train.log" "S1 train.log" || TRAIN_STATUS=$?
  fi
  return "$TRAIN_STATUS"
}

run_integrated_s2() {
  set -euo pipefail
  integrated_env
  test -s "$(checkpoint_path "$S1_DIR" "$H200_S1_STEPS")"
  test ! -e "$S2_DIR"
  (while true; do nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader; sleep 5; done) > "$EVIDENCE_DIR/integrated-s2-memory.txt" & MON_PID=$!
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nnodes=1 --nproc_per_node="$H200_NPROC_PER_NODE" --node_rank=0 --master_addr=localhost --master_port="$H200_MASTER_PORT" train_state_hijacking_dit.py \
    --config-name rwkv_relay_13.3B_state_hijack_dit_vae32_traj32x16 \
    model.rwkv_local_path="$DIFFRWKV_13B_MODEL_PATH" \
    data.token_dir="$DIFFRWKV_4096_DATA_DIR" data.max_length=4096 \
    model.trajectory_chunk_size=64 model.trajectory_horizon=64 \
    training.stage=2 training.gen_type=rf training.num_train_steps="$H200_S2_STEPS" training.save_every_n_steps="$H200_S2_STEPS" logging.log_freq=1 \
    training.resume="$(step_dir "$S1_DIR" "$H200_S1_STEPS")" \
    loss.trajectory_delta_loss_weight=0.02 \
    model.trajectory_s1_mode=transformer model.trajectory_denoiser_type=dit model.trajectory_state_blend=0.7 \
    model.dit_hidden=768 model.dit_depth=8 model.dit_num_heads=8 \
    model.trajectory_state_hidden=256 model.trajectory_state_depth=4 model.trajectory_state_num_heads=4 \
    training.train_batch_size=1 data.num_workers=0 \
    logging.run_name="$S2_RUN_NAME" \
    2>&1 | tee "$EVIDENCE_DIR/integrated-s2-train.txt"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  if [ "$TRAIN_STATUS" -eq 0 ]; then
    assert_clean_log "$EVIDENCE_DIR/integrated-s2-train.txt" "S2 shell" || TRAIN_STATUS=$?
  fi
  if [ "$TRAIN_STATUS" -eq 0 ]; then
    assert_clean_log "$S2_DIR/train.log" "S2 train.log" || TRAIN_STATUS=$?
  fi
  return "$TRAIN_STATUS"
}

run_integrated_generation_gate() {
  set -euo pipefail
  integrated_env
  local s2_ckpt_dir
  s2_ckpt_dir="$(step_dir "$S2_DIR" "$H200_S2_STEPS")"
  test -s "$s2_ckpt_dir/model.pt"
  local output_json="$EVIDENCE_DIR/integrated-s2-generation-grid.json"
  local texts_dir="$EVIDENCE_DIR/integrated-s2-generation-texts"
  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/eval/eval_trajectory_generation_grid.py \
    --ckpt_dir "$s2_ckpt_dir" \
    --output "$output_json" \
    --texts_dir "$texts_dir" \
    --sample_steps "$H200_GATE_SAMPLE_STEPS" \
    --max_new_tokens "$H200_GATE_MAX_NEW_TOKENS" \
    --temperatures 0.5 \
    --repetition_penalties 1.2 \
    --seeds 42 \
    --top_k 5 \
    --top_p 0.7 \
    --trajectory_s1_mode transformer \
    --trajectory_state_blend 0.7 \
    --trajectory_sampler rf_heun \
    2>&1 | tee "$EVIDENCE_DIR/integrated-s2-generation.txt"
  local gate_status=${PIPESTATUS[0]}
  set -e
  if [ "$gate_status" -ne 0 ]; then
    return "$gate_status"
  fi
  GATE_JSON="$output_json" GATE_TEXTS_DIR="$texts_dir" EXPECTED_STEP="$H200_S2_STEPS" python - <<'PY' 2>&1 | tee "$EVIDENCE_DIR/integrated-generation-acceptance.txt"
import json
import math
import os
from pathlib import Path

path = Path(os.environ["GATE_JSON"])
texts_dir = Path(os.environ["GATE_TEXTS_DIR"])
expected_step = int(os.environ["EXPECTED_STEP"])
if not path.is_file() or path.stat().st_size <= 0:
    raise SystemExit("missing generation JSON")
data = json.loads(path.read_text())
checks = {
    "checkpoint_step": int(data.get("checkpoint_step", -1)) == expected_step,
    "checkpoint_gen_type": data.get("checkpoint_gen_type") == "rf",
    "configs_nonempty": bool(data.get("configs")),
    "texts_dir": texts_dir.is_dir(),
}
for key, value in checks.items():
    print(f"{key}: {'PASS' if value else 'FAIL'}")
    if not value:
        raise SystemExit(f"generation gate check failed: {key}")
for name, cfg in data["configs"].items():
    text_path = Path(cfg.get("texts", ""))
    values = {
        "ref_ppl_rwkv": cfg.get("ref_ppl_rwkv"),
        "ref_avg_nll_rwkv": cfg.get("ref_avg_nll_rwkv"),
        "z_norm_mean": cfg.get("z_norm_mean"),
        "repeat_4": cfg.get("repeat_4"),
        "repeat_8": cfg.get("repeat_8"),
    }
    for metric, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SystemExit(f"{name}: non-finite metric {metric}={value!r}")
    if float(values["z_norm_mean"]) <= 0.0:
        raise SystemExit(f"{name}: non-positive z_norm_mean")
    if not (0.0 <= float(values["repeat_4"]) <= 1.0 and 0.0 <= float(values["repeat_8"]) <= 1.0):
        raise SystemExit(f"{name}: repetition metric out of range")
    if int(cfg.get("ref_tokens", 0)) <= 0:
        raise SystemExit(f"{name}: no reference tokens")
    if not text_path.is_file() or not text_path.read_text(errors="ignore").strip():
        raise SystemExit(f"{name}: missing generated text")
    print(f"{name}: PASS")
print("GENERATION_GATE: PASS")
PY
  return "${PIPESTATUS[0]}"
}

integrated_acceptance() {
  set -euo pipefail
  integrated_env
  python - <<'PY' 2>&1 | tee "$EVIDENCE_DIR/integrated-acceptance.txt"
import os
from pathlib import Path

evidence = Path(os.environ['EVIDENCE_DIR'])
stage_dirs = {
    'S0': (Path(os.environ['S0_DIR']), int(os.environ['H200_S0_STEPS'])),
    'S1': (Path(os.environ['S1_DIR']), int(os.environ['H200_S1_STEPS'])),
    'S2': (Path(os.environ['S2_DIR']), int(os.environ['H200_S2_STEPS'])),
}

def nonempty(path):
    return path.is_file() and path.stat().st_size > 0

def has_bad_log(path):
    if not path.is_file():
        return True
    text = path.read_text(errors='ignore')
    needles = ['nan', 'NaN', 'inf', 'OOM', 'out of memory', 'size mismatch']
    return any(needle in text for needle in needles)

criteria = {}
traj_data_log = evidence / 'trajectory-data-check.txt'
criteria['Trajectory data check'] = nonempty(traj_data_log) and 'TRAJECTORY_DATA_CHECK: PASS' in traj_data_log.read_text(errors='ignore')
for stage, (stage_dir, step) in stage_dirs.items():
    low = stage.lower()
    checkpoint = stage_dir / f'step_{step:08d}' / 'model.pt'
    train_log = stage_dir / 'train.log'
    shell_log = evidence / f'integrated-{low}-train.txt'
    memory_log = evidence / f'integrated-{low}-memory.txt'
    ckpt_log = evidence / f'integrated-{low}-checkpoint.txt'
    criteria[f'{stage} checkpoint'] = checkpoint.is_file() and checkpoint.stat().st_size > 0
    criteria[f'{stage} checkpoint integrity'] = nonempty(ckpt_log) and 'CHECKPOINT_INTEGRITY: PASS' in ckpt_log.read_text(errors='ignore')
    criteria[f'{stage} train.log'] = train_log.is_file() and not has_bad_log(train_log)
    criteria[f'{stage} shell log'] = nonempty(shell_log) and not has_bad_log(shell_log)
    criteria[f'{stage} memory'] = nonempty(memory_log)
for stage in ('s1', 's2'):
    resume_log = evidence / f'integrated-{stage}-resume.txt'
    criteria[f'{stage.upper()} resume validation'] = nonempty(resume_log) and 'RESUME_VALIDATION: PASS' in resume_log.read_text(errors='ignore')
gate_log = evidence / 'integrated-generation-acceptance.txt'
criteria['S2 generation gate'] = nonempty(gate_log) and 'GENERATION_GATE: PASS' in gate_log.read_text(errors='ignore')

for name, ok in criteria.items():
    print(f'{name}: {"PASS" if ok else "FAIL"}')

all_pass = all(criteria.values())
print('INTEGRATED_ACCEPTANCE:', 'PASS' if all_pass else 'FAIL')
raise SystemExit(0 if all_pass else 1)
PY
}

integrated_report() {
  set -euo pipefail
  integrated_env
  export INTEGRATED_TRAIN_STATUS="${1:-0}"
  export INTEGRATED_ACCEPTANCE_STATUS="${2:-0}"
  export INTEGRATED_GATE_STATUS="${3:-0}"
  python - <<'PY'
import os
from pathlib import Path

evidence = Path(os.environ['EVIDENCE_DIR'])
run_prefix = os.environ['RUN_PREFIX']
train_status = int(os.environ['INTEGRATED_TRAIN_STATUS'])
acceptance_status = int(os.environ['INTEGRATED_ACCEPTANCE_STATUS'])
gate_status = int(os.environ['INTEGRATED_GATE_STATUS'])
stage_dirs = {
    'S0': (Path(os.environ['S0_DIR']), int(os.environ['H200_S0_STEPS'])),
    'S1': (Path(os.environ['S1_DIR']), int(os.environ['H200_S1_STEPS'])),
    'S2': (Path(os.environ['S2_DIR']), int(os.environ['H200_S2_STEPS'])),
}
report_path = evidence / 'integrated-final-report.md'

def nonempty(path):
    return path.is_file() and path.stat().st_size > 0

def has_bad_log(path):
    if not path.is_file():
        return True
    text = path.read_text(errors='ignore')
    needles = ['nan', 'NaN', 'inf', 'OOM', 'out of memory', 'size mismatch']
    return any(needle in text for needle in needles)

criteria = {
    'Hardware validation': (evidence / 'integrated-hardware.txt').is_file() and not (evidence / 'integrated-hardware-fail.txt').is_file(),
    'Preflight': (evidence / 'preflight.txt').is_file(),
    'Config check': (evidence / 'config-check.txt').is_file(),
    'Tokenizer check': (evidence / 'tokenizer_decode.txt').is_file(),
    'Trajectory data check': (evidence / 'trajectory-data-check.txt').is_file() and 'TRAJECTORY_DATA_CHECK: PASS' in (evidence / 'trajectory-data-check.txt').read_text(errors='ignore'),
    'Training command status': train_status == 0,
    'Acceptance command status': acceptance_status == 0,
    'Generation gate command status': gate_status == 0,
}
for stage, (stage_dir, step) in stage_dirs.items():
    low = stage.lower()
    checkpoint = stage_dir / f'step_{step:08d}' / 'model.pt'
    train_log = stage_dir / 'train.log'
    shell_log = evidence / f'integrated-{low}-train.txt'
    memory_log = evidence / f'integrated-{low}-memory.txt'
    ckpt_log = evidence / f'integrated-{low}-checkpoint.txt'
    criteria[f'{stage} checkpoint'] = checkpoint.is_file() and checkpoint.stat().st_size > 0
    criteria[f'{stage} checkpoint integrity'] = nonempty(ckpt_log) and 'CHECKPOINT_INTEGRITY: PASS' in ckpt_log.read_text(errors='ignore')
    criteria[f'{stage} train.log clean'] = train_log.is_file() and not has_bad_log(train_log)
    criteria[f'{stage} shell log clean'] = nonempty(shell_log) and not has_bad_log(shell_log)
    criteria[f'{stage} memory evidence'] = nonempty(memory_log)
for stage in ('s1', 's2'):
    resume_log = evidence / f'integrated-{stage}-resume.txt'
    criteria[f'{stage.upper()} resume validation'] = nonempty(resume_log) and 'RESUME_VALIDATION: PASS' in resume_log.read_text(errors='ignore')
gate_log = evidence / 'integrated-generation-acceptance.txt'
criteria['S2 generation gate'] = nonempty(gate_log) and 'GENERATION_GATE: PASS' in gate_log.read_text(errors='ignore')

all_pass = all(criteria.values())
verdict = 'PASS' if all_pass else 'FAIL'

lines = [
    '# 13.3B Adaptive GPU 4096 Integrated Train+Test Report',
    '',
    f'INTEGRATED_VERDICT: {verdict}',
    '',
    f'Run prefix: `{run_prefix}`',
    f'Evidence directory: `{evidence}`',
    f'H200_NPROC_PER_NODE: `{os.environ["H200_NPROC_PER_NODE"]}`',
    f'H200_MASTER_PORT: `{os.environ["H200_MASTER_PORT"]}`',
    f'DIFFRWKV_MIN_GPU_MEMORY_GIB: `{os.environ["DIFFRWKV_MIN_GPU_MEMORY_GIB"]}`',
    f'DIFFRWKV_REQUIRED_GPU_NAME: `{os.environ.get("DIFFRWKV_REQUIRED_GPU_NAME", "") or "ANY"}`',
    f'H200_S0_STEPS: `{os.environ["H200_S0_STEPS"]}`',
    f'H200_S1_STEPS: `{os.environ["H200_S1_STEPS"]}`',
    f'H200_S2_STEPS: `{os.environ["H200_S2_STEPS"]}`',
    f'H200_GATE_SAMPLE_STEPS: `{os.environ["H200_GATE_SAMPLE_STEPS"]}`',
    f'H200_GATE_MAX_NEW_TOKENS: `{os.environ["H200_GATE_MAX_NEW_TOKENS"]}`',
    '',
    'This integrated run validates train/resume/checkpoint/generation plumbing. It still does not prove convergence or benchmark quality.',
    '',
    '## Criteria',
]
for name, ok in criteria.items():
    lines.append(f'- {name}: {"PASS" if ok else "FAIL"}')
lines.extend(['', '## Checkpoints'])
for stage, (stage_dir, step) in stage_dirs.items():
    checkpoint = stage_dir / f'step_{step:08d}' / 'model.pt'
    lines.append(f'- {stage}: `{checkpoint}` exists={checkpoint.is_file()}')
lines.extend(['', '## Evidence Files'])
for rel in [
    'integrated-hardware.txt',
    'preflight.txt',
    'config-check.txt',
    'tokenizer_decode.txt',
    'trajectory-data-check.txt',
    'integrated-s0-train.txt',
    'integrated-s0-memory.txt',
    'integrated-s1-train.txt',
    'integrated-s1-memory.txt',
    'integrated-s2-train.txt',
    'integrated-s2-memory.txt',
    'integrated-s0-checkpoint.txt',
    'integrated-s1-checkpoint.txt',
    'integrated-s2-checkpoint.txt',
    'integrated-s1-resume.txt',
    'integrated-s2-resume.txt',
    'integrated-s2-generation.txt',
    'integrated-s2-generation-grid.json',
    'integrated-generation-acceptance.txt',
    'integrated-acceptance.txt',
]:
    path = evidence / rel
    lines.append(f'- `{path}` exists={path.is_file()} nonempty={nonempty(path)}')
lines.extend(['', '## Next Recommendation'])
if all_pass:
    lines.append('Integrated H200 train/test gate passed. Keep evidence with the corresponding checkpoint paths before promoting the run.')
else:
    lines.append('Do not mark the run production-ready until every failed criterion above is fixed and the phase is rerun.')
lines.append('')

report_path.write_text('\n'.join(lines))
print(f'wrote {report_path}')
print(f'INTEGRATED_VERDICT: {verdict}')
raise SystemExit(0 if all_pass else 1)
PY
}

h200-integrated() {
  set -euo pipefail
  integrated_env
  trajectory-data-check
  require_adaptive_gpus
  preflight
  config-check
  tokenizer-check

  local train_status=0
  run_integrated_s0 || train_status=$?
  if [ "$train_status" -eq 0 ]; then
    checkpoint_integrity "S0" "$S0_DIR" "$H200_S0_STEPS" 3 "ddpm" "$EVIDENCE_DIR/integrated-s0-checkpoint.txt" || train_status=$?
  fi
  if [ "$train_status" -eq 0 ]; then
    run_integrated_s1 || train_status=$?
  fi
  if [ "$train_status" -eq 0 ]; then
    checkpoint_integrity "S1" "$S1_DIR" "$H200_S1_STEPS" 1 "ddpm" "$EVIDENCE_DIR/integrated-s1-checkpoint.txt" || train_status=$?
  fi
  if [ "$train_status" -eq 0 ]; then
    resume_log_check "S1" "$EVIDENCE_DIR/integrated-s1-train.txt" "$(checkpoint_path "$S0_DIR" "$H200_S0_STEPS")" "$H200_S0_STEPS" "$EVIDENCE_DIR/integrated-s1-resume.txt" || train_status=$?
  fi
  if [ "$train_status" -eq 0 ]; then
    run_integrated_s2 || train_status=$?
  fi
  if [ "$train_status" -eq 0 ]; then
    checkpoint_integrity "S2" "$S2_DIR" "$H200_S2_STEPS" 2 "rf" "$EVIDENCE_DIR/integrated-s2-checkpoint.txt" || train_status=$?
  fi
  if [ "$train_status" -eq 0 ]; then
    resume_log_check "S2" "$EVIDENCE_DIR/integrated-s2-train.txt" "$(checkpoint_path "$S1_DIR" "$H200_S1_STEPS")" "$H200_S1_STEPS" "$EVIDENCE_DIR/integrated-s2-resume.txt" || train_status=$?
  fi

  local acceptance_status=0
  local gate_status=0
  local report_status=0
  if [ "$train_status" -eq 0 ]; then
    run_integrated_generation_gate || gate_status=$?
  else
    gate_status="$train_status"
  fi
  integrated_acceptance || acceptance_status=$?
  integrated_report "$train_status" "$acceptance_status" "$gate_status" || report_status=$?

  if [ "$train_status" -ne 0 ]; then
    return "$train_status"
  fi
  if [ "$acceptance_status" -ne 0 ]; then
    return "$acceptance_status"
  fi
  if [ "$gate_status" -ne 0 ]; then
    return "$gate_status"
  fi
  return "$report_status"
}

h200-production() {
  set -euo pipefail
  production_env
  h200-integrated
}

h200-hardware-check() {
  require_adaptive_gpus
}

h200-dispatch-check() {
  integrated_env
  print_context
  echo "H200_NPROC_PER_NODE=$H200_NPROC_PER_NODE"
  echo "H200_MASTER_PORT=$H200_MASTER_PORT"
  echo "DIFFRWKV_MIN_GPU_MEMORY_GIB=$DIFFRWKV_MIN_GPU_MEMORY_GIB"
  echo "DIFFRWKV_REQUIRED_GPU_NAME=${DIFFRWKV_REQUIRED_GPU_NAME:-ANY}"
}

h200-production-dispatch-check() {
  production_env
  print_context
  echo "H200_NPROC_PER_NODE=$H200_NPROC_PER_NODE"
  echo "H200_MASTER_PORT=$H200_MASTER_PORT"
  echo "DIFFRWKV_MIN_GPU_MEMORY_GIB=$DIFFRWKV_MIN_GPU_MEMORY_GIB"
  echo "DIFFRWKV_REQUIRED_GPU_NAME=${DIFFRWKV_REQUIRED_GPU_NAME:-ANY}"
}

single-gpu-all() {
  preflight || return $?
  config-check || return $?
  tokenizer-check || return $?
  trajectory-data-check || return $?
  s0 || return $?
  s1 || return $?
  s2 || return $?
  report
}

all() {
  h200-integrated
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_13b_h200_4096_smoke.sh [preflight|config-check|tokenizer-check|trajectory-data-check|s0|s1|s2|report|single-gpu-all|all|h200-dispatch-check|h200-production-dispatch-check|h200-hardware-check|h200-integrated|h200-production]

Default phase is h200-integrated: adaptive local train+test on the visible real GPUs.
`all` is an alias for h200-integrated. Use `h200-production` for full-step production train+test defaults.
Use `single-gpu-all` explicitly for the legacy single-process smoke path. Environment overrides:
  DIFFRWKV_13B_MODEL_PATH
  DIFFRWKV_4096_DATA_DIR (trajectory 64x64 requires packed fixed-length samples)
  DIFFRWKV_PRODUCTION_4096_DATA_DIR (h200-production default: preprocessed_data/fineweb_4096_packed_full)
  CUDA_VISIBLE_DEVICES
  RUN_PREFIX
  EVIDENCE_DIR
  H200_NPROC_PER_NODE (h200-integrated default: visible CUDA device count)
  H200_MASTER_PORT (h200-integrated default: 29517)
  DIFFRWKV_MIN_GPU_MEMORY_GIB (h200-integrated default: 139)
  DIFFRWKV_REQUIRED_GPU_NAME (h200-integrated default: ANY)
  H200_DATA_CHECK_SAMPLES (trajectory-data-check default: 2048)
  H200_S0_STEPS/H200_S1_STEPS/H200_S2_STEPS (h200-integrated default: 10/10/10; h200-production default: 50000/50000/150000)
  H200_GATE_SAMPLE_STEPS/H200_GATE_MAX_NEW_TOKENS (h200-integrated default: 4/64; h200-production default: 100/4096)

Trajectory 64x64 cannot use per-row preprocessed_data/fineweb_4096 or fineweb_4096_full. Create packed full data with:
  python scripts/preprocess/preprocess_fineweb.py --model RWKV/RWKV7-Goose-World3-1.5B-HF --cache_dir ./data/huggingface --local_files_only --max_length 4096 --max_samples 0 --shard_size 100000 --pack_sequences --out_dir preprocessed_data/fineweb_4096_packed_full --out_file fineweb_4096_packed.pkl

Evidence goes to EVIDENCE_DIR. The integrated report is integrated-final-report.md.
Production readiness requires: hardware/config/tokenizer/trajectory-data checks, S0/S1/S2 exit 0, clean logs, checkpoint integrity, resume validation, and S2 trajectory generation gate PASS.
EOF
}

main() {
  local phase="${1:-h200-integrated}"
  case "$phase" in
    preflight) preflight ;;
    config-check) config-check ;;
    tokenizer-check) tokenizer-check ;;
    trajectory-data-check) trajectory-data-check ;;
    s0) s0 ;;
    s1) s1 ;;
    s2) s2 ;;
    report) report ;;
    single-gpu-all) single-gpu-all ;;
    all) all ;;
    h200-dispatch-check) h200-dispatch-check ;;
    h200-production-dispatch-check) h200-production-dispatch-check ;;
    h200-hardware-check) h200-hardware-check ;;
    h200-integrated) h200-integrated ;;
    h200-production) h200-production ;;
    -h|--help|help) usage ;;
    *) usage; return 2 ;;
  esac
}

main "$@"
