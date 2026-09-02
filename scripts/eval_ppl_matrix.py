#!/usr/bin/env python3
"""Cross-model PPL matrix: raw + cross-source for all available backbones."""
import sys, glob, json, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BACKBONES = {
    "0.4B": "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B-world",
    "1.5B": "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-1.5B-HF",
    "2.9B": "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF",
    "7.2B": "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-7.2B-g0",
    "13.3B": "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-G1f-13.3B-HF",
}

CROSS_S1 = {
    "0.4B": "outputs_relay/cross-source-0.4B-s1-denoised-v2/cross_s1_final.pt",
    "2.9B": None,  # Not ready yet
}

TEMPLATE = {
    "0.4B": "outputs_relay/drafter-04b-traj32x16-joint-scratch-coadapt/step_00006000",
    "2.9B": "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
}

from scripts.eval.diag_loop1_common import build_model
from transformers import AutoModelForCausalLM

device = "cuda"; dtype = torch.bfloat16
files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))[:30]
latent_files = sorted(glob.glob("preprocessed_data/owt_13b_s2_denoised_z/train/*.npy"))

results = {}

@torch.no_grad()
def compute_raw_ppl(rwkv, ids, am):
    out = rwkv(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    loss = torch.nn.functional.cross_entropy(
        out.logits[:, :-1].reshape(-1, 65536).float(), ids[:, 1:].reshape(-1), reduction="mean"
    )
    return float(torch.exp(loss).item())

@torch.no_grad()
def compute_cross_ppl(drafter, rwkv, Z, ids, am):
    H = int(drafter.trajectory_horizon)
    Z = Z[:, :H, :]
    states = drafter.predict_trajectory_states(Z)
    out = rwkv(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    pkv = drafter.inject_into_cache(out.past_key_values, states)
    out2 = rwkv(input_ids=ids, attention_mask=am.bool(), past_key_values=pkv, use_cache=True, return_dict=True)
    loss = torch.nn.functional.cross_entropy(
        out2.logits[:, :-1].reshape(-1, 65536).float(), ids[:, 1:].reshape(-1), reduction="mean"
    )
    return float(torch.exp(loss).item())

t0 = time.time()

for name, path in BACKBONES.items():
    print(f"\n=== {name} raw ===", flush=True)
    rwkv = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    
    ppls = []
    for f in files:
        d = np.load(f)
        ids = torch.tensor([d["input_ids"][:512]], device=device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)
        ppls.append(compute_raw_ppl(rwkv, ids, am))
    results[f"{name}_raw"] = float(np.mean(ppls))
    print(f"  raw PPL: {results[f'{name}_raw']:.2f}", flush=True)
    
    del rwkv; torch.cuda.empty_cache()

# Cross-source (0.4B reads 13.3B Z)
print(f"\n=== 0.4B cross-source ===", flush=True)
drafter, _, _, _ = build_model(TEMPLATE["0.4B"], device)
s1 = torch.load(CROSS_S1["0.4B"], map_location=device)
drafter.load_state_dict(s1["trainable_state"], strict=False); drafter.eval()

ppls_clean = []; ppls_shuffle = []
for i, (tf, lf) in enumerate(zip(files, latent_files)):
    d = np.load(tf); lat = np.load(lf)
    ids = torch.tensor([d["input_ids"][:512]], device=device, dtype=torch.long)
    am = torch.ones_like(ids, dtype=torch.float32)
    Z = torch.tensor(lat, device=device, dtype=dtype).unsqueeze(0)
    
    ppls_clean.append(compute_cross_ppl(drafter, drafter.rwkv_model, Z, ids, am))
    
    # Shuffled Z control
    Z_shuffle = Z[:, torch.randperm(Z.shape[1]), :]
    ppls_shuffle.append(compute_cross_ppl(drafter, drafter.rwkv_model, Z_shuffle, ids, am))
    
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/30] clean={np.mean(ppls_clean):.1f} shuffle={np.mean(ppls_shuffle):.1f}", flush=True)

results["0.4B_cross_13.3B_clean"] = float(np.mean(ppls_clean))
results["0.4B_cross_13.3B_shuffle"] = float(np.mean(ppls_shuffle))
print(f"  cross clean: {results['0.4B_cross_13.3B_clean']:.2f}")
print(f"  cross shuffle: {results['0.4B_cross_13.3B_shuffle']:.2f}")

del drafter; torch.cuda.empty_cache()

# Print matrix
print(f"\n{'='*60}")
print(f"  CROSS-MODEL PPL MATRIX")
print(f"{'='*60}")
sizes = list(BACKBONES.keys())
print(f"{'receiver':<8s} {'raw':>8s} {'cross(13.3B)':>12s} {'shuffle':>10s}")
for s in sizes:
    raw = results.get(f"{s}_raw", 0)
    cross = results.get(f"{s}_cross_13.3B_clean", 0) if s in CROSS_S1 else 0
    shuffle = results.get(f"{s}_cross_13.3B_shuffle", 0) if s in CROSS_S1 else 0
    cross_s = f"{cross:.1f}" if cross else "--"
    shuffle_s = f"{shuffle:.1f}" if shuffle else "--"
    print(f"{s:<8s} {raw:8.1f} {cross_s:>12s} {shuffle_s:>10s}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s")

with open("outputs_eval/ppl_matrix.json", "w") as f:
    json.dump({"results": results, "elapsed": elapsed}, f, indent=2)
print("Saved: outputs_eval/ppl_matrix.json")
