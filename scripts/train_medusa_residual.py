"""Medusa + Trajectory Residual (simplified forward).

  base_logits = MedusaHead(hidden)
  Z → Transformer → per-chunk bias
  final = base_logits + gate * expand_bias_to_positions(bias)

Simple: compute bias per chunk, expand to per-token positions.
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


class MedusaResidual(nn.Module):
    def __init__(self, hidden_dim=2560, latent_dim=32, vocab_size=65536, num_heads=4, cond_dim=512):
        super().__init__()
        self.z_proj = nn.Linear(latent_dim, cond_dim)
        enc = nn.TransformerEncoderLayer(cond_dim, 8, cond_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, 2)
        self.base_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                          nn.Linear(hidden_dim // 2, vocab_size))
            for _ in range(num_heads)
        ])
        self.bias_head = nn.Linear(cond_dim, vocab_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h, Z):
        """h: [B, S, D], Z: [B, H, 32] → list of [B, S, V]"""
        B, S, D = h.shape
        H = Z.shape[1]
        x = self.z_proj(Z)
        cond = self.encoder(x)  # [B, H, C]
        bias = self.bias_head(cond)  # [B, H, V]
        chunk_per_pos = (torch.arange(S, device=h.device) // (S // H)).clamp(max=H - 1)
        bias_per_pos = bias[:, chunk_per_pos, :]  # [B, S, V]
        base_logits = [head(h) for head in self.base_heads]
        g = self.gate.sigmoid()
        return [bl + g * bias_per_pos for bl in base_logits]


def train(args):
    model, _, dtype, _ = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    H, C = model.trajectory_horizon, model.trajectory_chunk_size
    S = H * C
    hidden_dim = model.rwkv_model.config.hidden_size

    mr = MedusaResidual(hidden_dim, model.latent_dim, 65536, args.num_heads).to(args.device, dtype)
    opt = torch.optim.AdamW(mr.parameters(), lr=args.lr)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    n = sum(p.numel() for p in mr.parameters()) / 1e6
    print(f"Medusa+Res: {n:.1f}M params", flush=True)

    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))

    for step in range(1, args.num_steps + 1):
        idxs = np.random.choice(len(files), args.batch_size, replace=False)
        b_ids = [np.load(files[i])["input_ids"][:S] for i in idxs]
        b_am = [np.load(files[i])["attention_mask"][:S] for i in idxs]
        ids = torch.tensor(np.stack(b_ids), device=args.device, dtype=torch.long)
        am = torch.tensor(np.stack(b_am), device=args.device, dtype=torch.float32)
        B = ids.shape[0]

        with torch.no_grad():
            Z = model._encode_trajectory_chunks(ids, am.bool())[0].reshape(B, H, -1)
            states = model.predict_trajectory_states(Z)
            out = model.rwkv_model(ids, am.bool(), output_hidden_states=True, use_cache=True, return_dict=True)
            pkv = model.inject_into_cache(out.past_key_values, states)
            out2 = model.rwkv_model(ids, am.bool(), past_key_values=pkv, output_hidden_states=True, use_cache=True, return_dict=True)
            H_all = out2.hidden_states[-1]

        logits_list = mr(H_all, Z)
        total_loss = torch.tensor(0.0, device=args.device)
        for k in range(args.num_heads):
            offset = k + 1
            p = logits_list[k][:, :S - offset]
            tgt = ids[:, offset:offset + p.shape[1]]
            total_loss += F.cross_entropy(p.reshape(-1, 65536), tgt.reshape(-1))

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mr.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            g = mr.gate.sigmoid().item()
            print(f"[step {step:5d}] loss={total_loss.item()/args.num_heads:.2f} gate={g:.3f} | lr={args.lr:.1e}", flush=True)
        if step % args.save_every == 0:
            torch.save({"state": mr.state_dict(), "step": step}, f"{args.save_dir}/mr_step{step}.pt")

    torch.save({"state": mr.state_dict(), "step": args.num_steps}, f"{args.save_dir}/mr_final.pt")
    print("Done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_dir", default="outputs_relay/medusa-residual-2.9B")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    train(ap.parse_args())
