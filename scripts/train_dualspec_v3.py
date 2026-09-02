"""DualSpec v3: extend standard Medusa with trajectory latent conditioning.

Standard Medusa:  hidden(t) → head_k → predict token t+k+1  
DualSpec:         [hidden(t), z_{chunk(t)+k}] → head_k → predict token t+k+1

Key difference from v1/v2: single RWKV forward per batch (like Medusa),
not per-chunk forwards. All frozen except DualSpec heads (~10M params).
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path: sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag


class DualSpecHeads(nn.Module):
    def __init__(self, hidden_dim=2560, latent_dim=32, vocab_size=65536,
                 num_heads=4, fusion_dim=384):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_dim, fusion_dim)
        self.latent_proj = nn.Linear(latent_dim, fusion_dim)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.SiLU(),
                          nn.Linear(fusion_dim, vocab_size))
            for _ in range(num_heads)
        ])

    def forward(self, hidden, z_cond):
        h = self.hidden_proj(hidden)
        z = self.latent_proj(z_cond)
        return [head(torch.cat([h, z], dim=-1)) for head in self.heads]


def train(args):
    model, _, dtype, _ = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    H, C = model.trajectory_horizon, model.trajectory_chunk_size
    seq_len = H * C
    hidden_dim = model.rwkv_model.config.hidden_size

    dualspec = DualSpecHeads(hidden_dim, model.latent_dim, 65536, args.num_heads)
    dualspec = dualspec.to(args.device, dtype)
    opt = torch.optim.AdamW(dualspec.parameters(), lr=args.lr)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    n = sum(p.numel() for p in dualspec.parameters()) / 1e6
    print(f"DualSpec v3: {n:.1f}M params, {args.num_heads} heads", flush=True)

    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))

    for step in range(1, args.num_steps + 1):
        idxs = np.random.choice(len(files), args.batch_size, replace=False)
        batch_ids = [np.load(files[i])["input_ids"][:seq_len] for i in idxs]
        batch_am = [np.load(files[i])["attention_mask"][:seq_len] for i in idxs]
        ids = torch.tensor(np.stack(batch_ids), device=args.device, dtype=torch.long)
        am = torch.tensor(np.stack(batch_am), device=args.device, dtype=torch.float32)
        B = ids.shape[0]

        with torch.no_grad():
            Z_flat = model._encode_trajectory_chunks(ids, am.bool())[0]
            Z = Z_flat.reshape(B, H, -1)
            states = model.predict_trajectory_states(Z)
            out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                   output_hidden_states=True, use_cache=True, return_dict=True)
            past_kv = model.inject_into_cache(out.past_key_values, states)
            out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                    past_key_values=past_kv, output_hidden_states=True,
                                    use_cache=True, return_dict=True)
            H_all = out2.hidden_states[-1]

        total_loss = 0.0
        for head_idx in range(args.num_heads):
            offset = head_idx + 1
            for t in range(seq_len - offset):
                chunk_t = t // C
                chunk_future = chunk_t + offset
                z_cond = Z[:, chunk_future, :] if chunk_future < H else torch.zeros(
                    B, model.latent_dim, device=args.device, dtype=dtype)
                logits = dualspec.heads[head_idx](torch.cat([
                    dualspec.hidden_proj(H_all[:, t, :]),
                    dualspec.latent_proj(z_cond)], dim=-1))
                total_loss += F.cross_entropy(logits, ids[:, t + offset])

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(dualspec.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"[step {step:5d}] loss={total_loss.item():.1f} | lr={args.lr:.1e}", flush=True)
        if step % args.save_every == 0:
            torch.save({"dualspec_state": dualspec.state_dict(), "step": step},
                       f"{args.save_dir}/dualspec_step{step}.pt")

    torch.save({"dualspec_state": dualspec.state_dict(), "step": args.num_steps},
               f"{args.save_dir}/dualspec_final.pt")
    print("DualSpec v3 done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_dir", default="outputs_relay/dualspec-2.9B-v3")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    train(ap.parse_args())
