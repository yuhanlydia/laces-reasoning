import json, os, sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.environ.get("DIFFRWKV_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))))
from transformers import AutoTokenizer

OWT_JSONL = os.environ.get("OWT_JSONL", "data/openwebtext_train.jsonl")
LATENT_SHARDS_DIR = os.environ.get("LATENT_SHARDS_DIR", "data/latent_shards")
OUTPUT_TOKENS_DIR = "preprocessed_data/owt_32d/tokens/train"
OUTPUT_LATENTS_DIR = "preprocessed_data/owt_32d/latents/train"

MAX_SAMPLES = 800000
MAX_LENGTH = 512

from transformers import AutoTokenizer
MODEL_NAME = "RWKV/RWKV7-Goose-World3-1.5B-HF"

print(f"Loading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True,
    cache_dir="./data/huggingface", local_files_only=True,
)

os.makedirs(OUTPUT_TOKENS_DIR, exist_ok=True)
os.makedirs(OUTPUT_LATENTS_DIR, exist_ok=True)

latent_shards = {}
for shard_file in sorted(os.listdir(LATENT_SHARDS_DIR)):
    shard_num = int(shard_file.replace("shard_", "").replace(".npy", ""))
    latent_shards[shard_num] = np.load(os.path.join(LATENT_SHARDS_DIR, shard_file))
    print(f"  Loaded shard {shard_num}: {latent_shards[shard_num].shape}")

print(f"Processing {MAX_SAMPLES} samples...")
with open(OWT_JSONL) as f:
    for i in tqdm(range(MAX_SAMPLES), desc="Tokenizing"):
        line = f.readline()
        if not line:
            break
        entry = json.loads(line)
        text = entry["text"]
        shard_idx = entry["shard"]
        sample_idx = entry["idx"]

        encoded = tokenizer(
            text, truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors="np",
        )
        input_ids = encoded["input_ids"].squeeze(0).astype(np.int32)
        attention_mask = encoded["attention_mask"].squeeze(0).astype(bool)

        token_path = os.path.join(OUTPUT_TOKENS_DIR, f"{i:08d}_tokens.npz")
        np.savez(token_path, input_ids=input_ids, attention_mask=attention_mask)

        if shard_idx in latent_shards:
            latent = latent_shards[shard_idx][sample_idx % len(latent_shards[shard_idx])]
            latent_path = os.path.join(OUTPUT_LATENTS_DIR, f"{i:08d}.npy")
            np.save(latent_path, latent.astype(np.float32))

print(f"Done. Saved to {OUTPUT_TOKENS_DIR} and {OUTPUT_LATENTS_DIR}")
