#!/usr/bin/env python3
"""Preprocess WikiText-103 with sliding window tokenization.

Creates overlapping 512-token chunks with stride 256.
Output: preprocessed_data/wikitext103/tokens/{train,val}/
"""
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

SEQ_LEN = 512
STRIDE = 256
BASE_DIR = Path(os.environ.get("DIFFRWKV_DATA_DIR", "preprocessed_data")) / "wikitext103"
MODEL_NAME = "RWKV/RWKV7-Goose-World3-1.5B-HF"

def process_split(split_name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split_name)

    full_text = "\n\n".join(x["text"] for x in ds if x["text"].strip())
    print(f"Tokenizing {split_name} ({len(full_text):,} chars)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    all_ids = tokenizer.encode(full_text)
    print(f"  Total tokens: {len(all_ids):,}")

    chunks = []
    for start in range(0, len(all_ids) - SEQ_LEN + 1, STRIDE):
        chunk = all_ids[start:start + SEQ_LEN]
        if len(chunk) == SEQ_LEN:
            chunks.append(chunk)

    print(f"  Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        ids = np.array(chunk, dtype=np.int32)
        mask = np.ones(SEQ_LEN, dtype=bool)
        np.savez(out_dir / f"chunk_{i:06d}_tokens.npz",
                 input_ids=ids, attention_mask=mask)

    print(f"  Saved {len(chunks)} files to {out_dir}")
    return len(chunks)

if __name__ == "__main__":
    n_train = process_split("train", BASE_DIR / "tokens" / "train")
    n_val = process_split("validation", BASE_DIR / "tokens" / "val")
    print(f"\nDone: {n_train} train, {n_val} val chunks")
