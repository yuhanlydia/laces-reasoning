#!/usr/bin/env python3
"""Clean, self-contained OpenWebText tokenization for RELAY training.

Only tokens (no Qwen latents) — RELAY's identity-encoder pipeline doesn't
need pre-computed latents. Pulls OpenWebText from HuggingFace; if you
already have a local jsonl, set OWT_JSONL.

Output: preprocessed_data/owt/tokens/train/{idx:08d}_tokens.npz
  - input_ids: int32 [SEQ_LEN]
  - attention_mask: bool [SEQ_LEN]

Run:
    python scripts/preprocess/preprocess_owt_clean.py
"""
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# ─── Config ────────────────────────────────────────────────────────────
MODEL_NAME = "RWKV/RWKV7-Goose-World3-2.9B-HF"  # tokenizer source — must
                                                 # match the model you train against
SEQ_LEN = 512                # length of each chunk
STRIDE = 256                 # overlap between chunks (set = SEQ_LEN for no overlap)
OUT_DIR = Path("preprocessed_data/owt/tokens/train")

# Source: either local jsonl (one doc per line, "text" field), or HF dataset.
OWT_JSONL = None  # e.g. "/path/to/openwebtext_train.jsonl"
HF_DATASET = "Skylion007/openwebtext"  # used when OWT_JSONL is None
HF_SPLIT = "train"
MAX_DOCS = 200_000  # cap for sanity; bump up for full-scale runs
# ──────────────────────────────────────────────────────────────────────


def docs_from_jsonl(path):
    import json
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                yield d.get("text", "")
            except json.JSONDecodeError:
                continue


def docs_from_hf():
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)
    for i, ex in enumerate(ds):
        if i >= MAX_DOCS:
            break
        yield ex["text"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    docs = docs_from_jsonl(OWT_JSONL) if OWT_JSONL else docs_from_hf()

    chunk_idx = 0
    pbar = tqdm(docs, total=MAX_DOCS, desc="Tokenizing")
    for text in pbar:
        if not text or len(text) < 100:  # skip tiny docs
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < SEQ_LEN // 2:
            continue  # skip docs too short to fill half a chunk

        # Sliding-window chunks. If doc is longer than SEQ_LEN, multiple chunks.
        for start in range(0, max(1, len(ids) - SEQ_LEN + 1), STRIDE):
            chunk = ids[start:start + SEQ_LEN]
            if len(chunk) < SEQ_LEN:
                # Pad short chunks with zeros + zero attention.
                attn = np.zeros(SEQ_LEN, dtype=bool)
                attn[:len(chunk)] = True
                pad = np.zeros(SEQ_LEN, dtype=np.int32)
                pad[:len(chunk)] = chunk
                input_ids = pad
            else:
                attn = np.ones(SEQ_LEN, dtype=bool)
                input_ids = np.array(chunk, dtype=np.int32)

            np.savez(
                OUT_DIR / f"{chunk_idx:08d}_tokens.npz",
                input_ids=input_ids, attention_mask=attn,
            )
            chunk_idx += 1
            if len(chunk) < SEQ_LEN:
                break

        if chunk_idx % 10000 == 0:
            pbar.set_postfix(chunks=chunk_idx)

    print(f"\nDone. Saved {chunk_idx} chunks to {OUT_DIR}")


if __name__ == "__main__":
    main()
