#!/usr/bin/env python3
"""Continuous prompt baseline (efficient): 13.3B Z -> 0.4B soft prefix.

Preloads paired (tokens, latent) data into memory. Compares input-side soft
prompting vs state-side S1 injection.
"""
import sys, glob, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from transformers import AutoModelForCausalLM

device = "cuda"; dtype = torch.bfloat16

rwkv = AutoModelForCausalLM.from_pretrained(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B-world",
    trust_remote_code=True, torch_dtype=dtype, local_files_only=True
).to(device).eval()
for p in rwkv.parameters():
    p.requires_grad = False

hidden_dim = rwkv.config.hidden_size
vocab_size = rwkv.config.vocab_size
prefix_length = 16
latent_dim = 32

class PrefixAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        h = 2048
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, prefix_length * hidden_dim),
        )
    def forward(self, z):
        return self.net(z).view(-1, prefix_length, hidden_dim)

adapter = PrefixAdapter().to(device, dtype)
print(f"PrefixAdapter params: {sum(p.numel() for p in adapter.parameters())/1e6:.1f}M", flush=True)

print("Preloading data...", flush=True)
token_files = {Path(f).stem.replace("_tokens", ""): f
               for f in sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))}
latent_files = sorted(glob.glob("preprocessed_data/owt_13b_s2_denoised_z/train/*.npy"))[:1000]

pairs = []
for lf in latent_files:
    stem = Path(lf).stem.replace("_tokens", "").replace("_latent", "")
    tf = token_files.get(stem)
    if tf is None:
        continue
    d = np.load(tf); lat = np.load(lf)
    ids = d["input_ids"][:256].astype(np.int64)
    if len(ids) < 256:
        continue
    pairs.append((ids, lat.mean(0).astype(np.float32)))
    if len(pairs) >= 500:
        break
print(f"Loaded {len(pairs)} pairs", flush=True)

ids_all = torch.tensor(np.stack([p[0] for p in pairs]), device=device, dtype=torch.long)
z_all = torch.tensor(np.stack([p[1] for p in pairs]), device=device, dtype=dtype)

opt = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
save_dir = "outputs_relay/prefix_baseline_0.4B"
Path(save_dir).mkdir(exist_ok=True)

n = ids_all.shape[0]; split = int(n * 0.85)

@torch.no_grad()
def eval_ppl():
    ppls = []
    for i in range(split, n):
        ids = ids_all[i:i+1]; z = z_all[i:i+1]
        prefix = adapter(z)
        tok_embeds = rwkv.get_input_embeddings()(ids)
        full = torch.cat([prefix.to(dtype), tok_embeds.to(dtype)], dim=1)
        out = rwkv(inputs_embeds=full, use_cache=True, return_dict=True)
        logits = out.logits[:, prefix_length:, :].float()[:, :-1, :]
        targets = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        ppls.append(float(torch.exp(loss).item()))
    return float(np.mean(ppls))

t0 = time.time()
for step in range(1, 3001):
    idx = np.random.randint(0, split, size=4)
    bl = 0.0
    for i in idx:
        ids = ids_all[i:i+1]; z = z_all[i:i+1]
        prefix = adapter(z)
        tok_embeds = rwkv.get_input_embeddings()(ids)
        full = torch.cat([prefix.to(dtype), tok_embeds.to(dtype)], dim=1)
        out = rwkv(inputs_embeds=full, use_cache=True, return_dict=True)
        logits = out.logits[:, prefix_length:, :].float()[:, :-1, :]
        targets = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        bl += loss.item()
    if step % 200 == 0:
        ev = eval_ppl()
        print(f"[step {step:5d}] train_loss={bl/4:.3f} eval_ppl={ev:.1f} "
              f"step/s={step/max(time.time()-t0,0.01):.1f}", flush=True)
        torch.save({"state": adapter.state_dict(), "step": step, "eval_ppl": ev},
                   f"{save_dir}/prefix_step{step}.pt")

final_ppl = eval_ppl()
print(f"\nDone! Prefix baseline eval PPL: {final_ppl:.1f}", flush=True)
print(f"(compare: raw 0.4B=54.3, state-inject cross-S1=44.7)", flush=True)
with open("outputs_eval/prefix_baseline_result.json", "w") as f:
    json.dump({"prefix_ppl": final_ppl, "raw_ppl": 54.3, "cross_s1_ppl": 44.7}, f, indent=2)
