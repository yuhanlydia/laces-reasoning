"""Train 2.9B S1 from scratch to read 13.3B S0 shared latents.

Cross-source S1: 13.3B S0 encodes text → z_shared → 2.9B S1 maps z → 2.9B state.
S1 trained from scratch (random init) on external 13.3B latents.

This enables multi-agent latent sharing: same latent space, different backbones.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path: sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as C
from eval.sample_prefix_suffix_cfg import encode_prefix


def load_batch(token_dir, latent_dir, batch_size, seq_len, device):
    latent_files = sorted(glob.glob(f"{latent_dir}/*.npy"))
    np.random.shuffle(latent_files)

    batch = []
    for lf in latent_files:
        stem = Path(lf).stem.replace("_tokens", "")
        tf = Path(token_dir) / f"{stem}_tokens.npz"
        if not tf.exists():
            continue
        d = np.load(str(tf))
        lat = np.load(str(lf))
        ids = d["input_ids"][:seq_len]
        am = d["attention_mask"][:seq_len]
        batch.append((ids, am, lat))
        if len(batch) >= batch_size:
            break

    ids = torch.tensor([b[0] for b in batch], device=device, dtype=torch.long)
    am = torch.tensor([b[1] for b in batch], device=device, dtype=torch.float32)
    latents = [b[2] for b in batch]
    return ids, am, latents


def train_cross_source_s1(ckpt_dir, token_dir, latent_dir, save_dir, device,
                          num_steps=30000, batch_size=8, lr=1e-4, save_every=3000):
    model, tok, dtype, pad = C.build_model(ckpt_dir, device)
    model.eval()

    for name, p in model.named_parameters():
        if "alpha_heads" in name or "state_basis" in name or "state_scale" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_trainable/1e6:.1f}M (S1 only, S0/RWKV frozen)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for step in range(1, num_steps + 1):
        ids, am, raw_lats = load_batch(token_dir, latent_dir, batch_size, 512, device)

        lats = []
        for rl in raw_lats:
            z = torch.tensor(rl, device=device, dtype=dtype)
            if z.dim() == 2 and z.shape[0] > 1:
                z = z.mean(dim=0, keepdim=True)
            lats.append(z[0])
        z_batch = torch.stack(lats)

        states = model.predict_states(z_batch)

        with torch.no_grad():
            out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                   use_cache=True, return_dict=True)
        past_kv = model.inject_into_cache(out.past_key_values, states)

        with torch.no_grad():
            pass

        out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                past_key_values=past_kv, use_cache=True, return_dict=True)
        logits = out2.logits[:, :-1]
        targets = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, 65536).float(), targets.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                ppl = float(torch.exp(loss.clone().detach()).item())
            print(f"[step {step}] loss={loss.item():.3f} PPL={ppl:.1f} | "
                  f"z_norm={z_batch.float().norm(dim=-1).mean():.2f} | lr={lr:.1e}", flush=True)

        if step % save_every == 0:
            torch.save({
                "trainable_state": {k: v for k, v in model.state_dict().items()
                                    if "alpha_heads" in k or "state_basis" in k or "state_scale" in k},
                "step": step,
            }, f"{save_dir}/cross_s1_step{step}.pt")
            print(f"  saved step{step}", flush=True)

    torch.save({
        "trainable_state": {k: v for k, v in model.state_dict().items()
                            if "alpha_heads" in k or "state_basis" in k or "state_scale" in k},
        "step": num_steps,
    }, f"{save_dir}/cross_s1_final.pt")
    print("Cross-source S1 training done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--latent_dir", default="preprocessed_data/owt_13b_s0_latents/train")
    ap.add_argument("--save_dir", default="outputs_relay/cross-source-s1-2.9B")
    ap.add_argument("--num_steps", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    train_cross_source_s1(a.ckpt_dir, a.token_dir, a.latent_dir, a.save_dir,
                          a.device, a.num_steps, a.batch_size, a.lr, a.save_every)
