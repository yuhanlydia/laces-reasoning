"""Compute latent μ/σ buffers for the identity-encoder pipeline.

Identity encoder skips the variational MLP; the latent IS rwkv_pool's
mean-pooled hidden state, normalized by dataset-level (μ, σ). This script
runs once, before training, to compute these statistics from a sample of
training docs and saves them to a .pt file that the trainer loads via
`model.latent_stats_path` config.

Usage:
  python scripts/tools/compute_latent_stats.py \\
      --rwkv_path /inspire/.../models/RWKV7-Goose-World3-2.9B-HF \\
      --token_dir preprocessed_data/owt_qwen_32d/tokens/train \\
      --out_path preprocessed_data/latent_stats_2.9B.pt \\
      --n_docs 1024

The output .pt contains {'mu': [hidden_size], 'sigma': [hidden_size]}.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rwkv_path", required=True,
                   help="Local path or HF id of the pretrained RWKV-7 model")
    p.add_argument("--token_dir", required=True,
                   help="Dir of training .npz files (input_ids + attention_mask)")
    p.add_argument("--out_path", required=True,
                   help="Where to write the .pt with {'mu', 'sigma'}")
    p.add_argument("--n_docs", type=int, default=1024,
                   help="How many docs to use for the statistics")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_doc(npz_path):
    d = np.load(npz_path)
    ids = d["input_ids"].astype(np.int64)
    mask = d["attention_mask"].astype(np.int64) if "attention_mask" in d else np.ones_like(ids)
    return ids, mask


def pad_batch(ids_list, mask_list):
    L = max(ids.shape[0] for ids in ids_list)
    B = len(ids_list)
    ids_pad = np.zeros((B, L), dtype=np.int64)
    mask_pad = np.zeros((B, L), dtype=np.int64)
    for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
        n = ids.shape[0]
        ids_pad[i, :n] = ids
        mask_pad[i, :n] = mask
    return ids_pad, mask_pad


@torch.no_grad()
def main():
    args = parse_args()

    print(f"Loading RWKV from {args.rwkv_path}")
    # local_files_only=True forces HF to treat rwkv_path as a local dir and
    # skip Hub repo_id validation (which barfs on absolute paths containing /)
    rwkv = AutoModelForCausalLM.from_pretrained(
        args.rwkv_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(args.device).eval()
    hidden_size = rwkv.config.hidden_size
    print(f"hidden_size = {hidden_size}")

    files = sorted(glob.glob(os.path.join(args.token_dir, "*.npz")))[: args.n_docs]
    print(f"Computing μ/σ over {len(files)} docs from {args.token_dir}")

    # Welford's online algorithm for stable mean/variance
    n = 0
    mean = torch.zeros(hidden_size, dtype=torch.float64, device=args.device)
    M2 = torch.zeros(hidden_size, dtype=torch.float64, device=args.device)

    t0 = time.time()
    for i in range(0, len(files), args.batch_size):
        batch_files = files[i:i + args.batch_size]
        ids_list, mask_list = [], []
        for fp in batch_files:
            ids, mask = load_doc(fp)
            ids_list.append(ids)
            mask_list.append(mask)
        ids_pad, mask_pad = pad_batch(ids_list, mask_list)
        ids_t = torch.from_numpy(ids_pad).long().to(args.device)
        mask_t = torch.from_numpy(mask_pad).long().to(args.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = rwkv(input_ids=ids_t, attention_mask=mask_t.bool(),
                       output_hidden_states=True, return_dict=True)
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
        else:
            hidden = out.last_hidden_state

        # Mean pool with attention mask
        mask_f = mask_t.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        pooled = pooled.to(dtype=torch.float64)  # for numerical stability

        # Welford update per sample
        for vec in pooled:
            n += 1
            delta = vec - mean
            mean = mean + delta / n
            delta2 = vec - mean
            M2 = M2 + delta * delta2

        if (i // args.batch_size) % 16 == 0:
            elapsed = time.time() - t0
            print(f"  [{n}/{len(files)}] {n / elapsed:.1f} docs/s ({elapsed:.1f}s)")

    var = M2 / max(n - 1, 1)
    sigma = var.sqrt()

    mu_cpu = mean.float().cpu()
    sigma_cpu = sigma.float().cpu()

    print(f"\nDone. n={n} docs, elapsed {time.time() - t0:.1f}s")
    print(f"μ stats:    norm={mu_cpu.norm():.4f}, mean={mu_cpu.mean():+.6f}, std={mu_cpu.std():.6f}")
    print(f"σ stats:    norm={sigma_cpu.norm():.4f}, mean={sigma_cpu.mean():.6f}, std={sigma_cpu.std():.6f}")
    print(f"σ range:    min={sigma_cpu.min():.6f}, max={sigma_cpu.max():.6f}")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save({"mu": mu_cpu, "sigma": sigma_cpu, "n_docs": n,
                "hidden_size": hidden_size}, args.out_path)
    print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    main()
