#!/usr/bin/env python3
"""Preprocess PG19 (long-form books) for RELAY long-context PPL eval.

Same chunking as wikitext103: SEQ_LEN=512, STRIDE=256.

Output: preprocessed_data/pg19/tokens/{train,val}/
"""
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

SEQ_LEN = 512
STRIDE = 256
BASE_DIR = Path("preprocessed_data/pg19")
MODEL_NAME = "RWKV/RWKV7-Goose-World3-2.9B-HF"
MAX_DOCS = 200  # PG19 docs are full books — 200 books is plenty


def process_split(split_name, out_dir, max_docs):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Original `pg19` is script-based and unsupported in newer datasets lib.
    # emozilla/pg19 is the parquet-mirror.
    ds = load_dataset("emozilla/pg19", split=split_name, streaming=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    chunk_idx = 0
    for i, ex in enumerate(ds):
        if i >= max_docs:
            break
        text = ex["text"]
        if not text or len(text) < 1000:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        for start in range(0, max(1, len(ids) - SEQ_LEN + 1), STRIDE):
            chunk = ids[start:start + SEQ_LEN]
            if len(chunk) < SEQ_LEN:
                break
            np.savez(
                out_dir / f"chunk_{chunk_idx:08d}_tokens.npz",
                input_ids=np.array(chunk, dtype=np.int32),
                attention_mask=np.ones(SEQ_LEN, dtype=bool),
            )
            chunk_idx += 1
    print(f"  {split_name}: {chunk_idx} chunks → {out_dir}")
    return chunk_idx


if __name__ == "__main__":
    process_split("train", BASE_DIR / "tokens" / "train", MAX_DOCS)
    process_split("validation", BASE_DIR / "tokens" / "val", 50)
    process_split("test", BASE_DIR / "tokens" / "test", 50)
