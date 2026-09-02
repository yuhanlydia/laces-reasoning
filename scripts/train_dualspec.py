"""DualSpec v7: Transformer trajectory encoder + adaLN + residual.

  Z [H,32] → TransformerEncoder(2 layers) → cond [H, D]
  cond → adaLN modulate hidden → trajectory logits
  final = base_logits + gate * trajectory_logits
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


class TrajEncoder(nn.Module):
    """Lightweight Transformer over Z trajectory."""
    def __init__(self, latent_dim=32, cond_dim=512, depth=2):
        super().__init__()
        self.proj_in = nn.Linear(latent_dim, cond_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cond_dim, nhead=8, dim_feedforward=cond_dim * 2,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj_out = nn.Linear(cond_dim, cond_dim)

    def forward(self, Z):
        x = self.proj_in(Z)  # [B, H, cond_dim]
        x = self.transformer(x)
        return self.proj_out(x)


class DualSpecHeads(nn.Module):
    def __init__(self, hidden_dim=2560, vocab_size=65536, num_heads=4,
                 latent_dim=32, cond_dim=512):
        super().__init__()
        self.traj_encoder = TrajEncoder(latent_dim, cond_dim)
        self.adaln_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 3), nn.Tanh())
        self.base_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), nn.SiLU(),
                          nn.Linear(hidden_dim // 4, vocab_size))
            for _ in range(num_heads)
        ])
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8), nn.SiLU(),
            nn.Linear(hidden_dim // 8, vocab_size))

    def forward(self, hidden, Z_full, chunk_idx):
        B = Z_full.shape[0]
        cond = self.traj_encoder(Z_full)  # [B, H, cond_dim]
        c = cond[0, chunk_idx] if B == 1 else cond.reshape(B * cond.shape[1], -1)[
            torch.arange(len(chunk_idx), device=chunk_idx.device) % (B * cond.shape[1]),
            :cond.shape[-1]]
        c = c.reshape(-1, cond.shape[-1])[:len(chunk_idx)]
        mod = self.adaln_proj(c)
        shift, scale, gate = mod.chunk(3, dim=-1)
        modulated = hidden * (1.0 + scale) + shift
        g = gate.sigmoid().mean(dim=-1, keepdim=True)
        base_logits = [h(hidden) for h in self.base_heads]
        traj_logits = self.traj_head(modulated)
        return [b + g * traj_logits for b in base_logits]


def train(args):
    model, _, dtype, _ = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    H, C = model.trajectory_horizon, model.trajectory_chunk_size
    S = H * C
    hidden_dim = model.rwkv_model.config.hidden_size

    dualspec = DualSpecHeads(hidden_dim, 65536, args.num_heads).to(args.device, dtype)
    opt = torch.optim.AdamW(dualspec.parameters(), lr=args.lr)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    n = sum(p.numel() for p in dualspec.parameters()) / 1e6
    print(f"DualSpec v7: {n:.1f}M params", flush=True)

    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))

    for step in range(1, args.num_steps + 1):
        idxs = np.random.choice(len(files), args.batch_size, replace=False)
        batch_ids = [np.load(files[i])["input_ids"][:S] for i in idxs]
        batch_am = [np.load(files[i])["attention_mask"][:S] for i in idxs]
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

        total_loss = torch.tensor(0.0, device=args.device)
        for head_idx in range(args.num_heads):
            offset = head_idx + 1
            T = S - offset
            h_in = H_all[:, :T, :].reshape(B * T, hidden_dim)
            c_idx = (torch.arange(T, device=args.device) // C + offset).clamp(max=H - 1)
            c_idx = c_idx.unsqueeze(0).expand(B, -1).reshape(-1)

            logits_list = dualspec(h_in, Z, c_idx)
            targets = ids[:, offset:].reshape(-1)
            total_loss += F.cross_entropy(logits_list[head_idx], targets)

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(dualspec.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"[step {step:5d}] loss={total_loss.item():.2f} | lr={args.lr:.1e}", flush=True)
        if step % args.save_every == 0:
            torch.save({"dualspec_state": dualspec.state_dict(), "step": step},
                       f"{args.save_dir}/dualspec_step{step}.pt")

    torch.save({"dualspec_state": dualspec.state_dict(), "step": args.num_steps},
               f"{args.save_dir}/dualspec_final.pt")
    print("DualSpec v7 done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_dir", default="outputs_relay/dualspec-2.9B-v7")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    train(ap.parse_args())
