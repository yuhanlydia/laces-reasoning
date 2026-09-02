"""Encode ALL 8M OpenWebText with Qwen3-Embedding-8B → 32-dim latents.
Output format compatible with both DiffRwkv (.npy per sample) and latentDLM_mmdit (shards of 100K).
"""
import json, os
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

OWT_JSONL = os.environ.get("OWT_JSONL", "data/openwebtext_train.jsonl")
OUT_DIR = "preprocessed_data/owt_qwen_8M"
OUT_LATENTS_PER_SAMPLE = os.path.join(OUT_DIR, "latents")
OUT_LATENTS_SHARDS = os.path.join(OUT_DIR, "latent_shards")
SHARD_SIZE = 100000
BATCH_SIZE = 16
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"

os.makedirs(OUT_LATENTS_PER_SAMPLE, exist_ok=True)
os.makedirs(OUT_LATENTS_SHARDS, exist_ok=True)

print(f"Loading Qwen encoder {QWEN_MODEL} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(QWEN_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE)
model.eval()

with open(OWT_JSONL) as f:
    lines = [json.loads(line)["text"] for line in f]
total = len(lines)
n_shards = total // SHARD_SIZE + 1
print(f"Total lines: {total}  →  {n_shards} shards × {SHARD_SIZE}")

@torch.no_grad()
def encode_batch(texts):
    enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt").to(DEVICE)
    emb = model(**enc).last_hidden_state.mean(dim=1)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float().cpu().numpy().astype(np.float32)

for shard_num in range(0, total, SHARD_SIZE):
    shard_end = min(shard_num + SHARD_SIZE, total)
    embs = []
    for i in tqdm(range(shard_num, shard_end, BATCH_SIZE), desc=f"Shard {shard_num//SHARD_SIZE:04d}"):
        emb = encode_batch(lines[i:min(i+BATCH_SIZE, shard_end)])
        embs.append(emb)
    shard_embs = np.concatenate(embs, axis=0)
    shard_path = os.path.join(OUT_LATENTS_SHARDS, f"shard_{shard_num//SHARD_SIZE:04d}.npy")
    np.save(shard_path, shard_embs)
    for local_idx in range(shard_embs.shape[0]):
        global_idx = shard_num + local_idx
        np.save(os.path.join(OUT_LATENTS_PER_SAMPLE, f"{global_idx:08d}.npy"), shard_embs[local_idx])

print(f"Done:\n  Per-sample: {OUT_LATENTS_PER_SAMPLE}/\n  Shards:     {OUT_LATENTS_SHARDS}/shard_*.npy")
