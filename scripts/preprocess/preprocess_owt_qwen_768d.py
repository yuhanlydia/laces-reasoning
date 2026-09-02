import json, os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

OWT_JSONL = os.environ.get("OWT_JSONL", "data/openwebtext_train.jsonl")
OUT_TOKENS = "preprocessed_data/owt_qwen_768d/tokens/train"
OUT_LATENTS = "preprocessed_data/owt_qwen_768d/latents/train"
MODEL = "RWKV/RWKV7-Goose-World3-1.5B-HF"
QWEN = "Qwen/Qwen3-Embedding-8B"
MAX_LENGTH = 512
CACHE_DIR = "./data/huggingface"
LATENT_DIM = 768
BATCH_SIZE = 64
DEVICE = "cuda"

os.makedirs(OUT_TOKENS, exist_ok=True)
os.makedirs(OUT_LATENTS, exist_ok=True)

print("Loading RWKV tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=CACHE_DIR, local_files_only=True)
print("Loading Qwen encoder...")
qw_tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True)
qw_model = AutoModel.from_pretrained(QWEN, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE)
qw_model.eval()

with open(OWT_JSONL) as f:
    total = sum(1 for _ in f)
print(f"Total lines: {total}")

idx = 0
with open(OWT_JSONL) as f:
    with tqdm(total=total, desc="Encoding") as pbar:
        while True:
            batch_texts = []
            for _ in range(BATCH_SIZE):
                line = f.readline()
                if not line:
                    break
                batch_texts.append(json.loads(line)["text"])
            if not batch_texts:
                break

            enc = tok(batch_texts, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="np")
            for j in range(len(batch_texts)):
                np.savez(os.path.join(OUT_TOKENS, f"{idx:08d}_tokens.npz"),
                         input_ids=enc["input_ids"][j].astype(np.int32),
                         attention_mask=enc["attention_mask"][j].astype(bool))
                idx += 1

            qw_enc = qw_tok(batch_texts, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                emb = qw_model(**qw_enc).last_hidden_state.mean(dim=1)
            emb = (emb[:, :LATENT_DIM] / emb[:, :LATENT_DIM].norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
            for j in range(len(batch_texts)):
                np.save(os.path.join(OUT_LATENTS, f"{idx - len(batch_texts) + j:08d}.npy"), emb[j])

            pbar.update(len(batch_texts))

print(f"Done → {idx} samples saved to {OUT_TOKENS}, {OUT_LATENTS}")
