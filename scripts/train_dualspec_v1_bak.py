"""Train DualSpec heads: trajectory-conditioned speculative decoding.

DualSpec = dual-stream draft heads that condition on BOTH:
  1. Token-level hidden state (from autoregressive generation)
  2. Trajectory-level latent z_{k+1} (next chunk's planned latent from S2)

Unlike standard Medusa (blind last-hidden-state guess), DualSpec sees the
"future plan" z_{k+1} that S2 already jointly sampled via bidirectional scan.
This is uniquely possible in StateDiffRWKV because S2 samples all 16 z's at
once before any token generation begins.

Usage:
  python scripts/train_dualspec.py --ckpt_dir <relay_ckpt> --state_injection
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

import eval.diag_loop1_common as C


# ═══════════════════════════════════════════════════════════════════════
#  DualSpec Head
# ═══════════════════════════════════════════════════════════════════════
class DualSpecHeads(nn.Module):
    """Dual-stream speculative heads: hidden-state + trajectory latent → vocab.

    Architecture:
      hidden (2560) → Linear → 384
      z_next (32)   → Linear → 384
      concat(768) → fusion MLP → 384 → k parallel heads → vocab
    """

    def __init__(self, hidden_dim: int = 2560, latent_dim: int = 32,
                 vocab_size: int = 65536, num_heads: int = 4,
                 fusion_dim: int = 384):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_dim, fusion_dim)
        self.latent_proj = nn.Linear(latent_dim, fusion_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.SiLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.SiLU(),
                          nn.Linear(fusion_dim, vocab_size))
            for _ in range(num_heads)
        ])

    def forward(self, hidden: torch.Tensor, z_next: torch.Tensor):
        """hidden: [B, hidden_dim], z_next: [B, latent_dim] → list of [B, vocab_size]"""
        h = self.hidden_proj(hidden)
        z = self.latent_proj(z_next)
        fused = self.fusion(torch.cat([h, z], dim=-1))
        return [head(fused) for head in self.heads]


# ═══════════════════════════════════════════════════════════════════════
#  Data: loads trajectory latents + provides per-chunk hidden states
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def load_trajectory_batch(token_dir, batch_size, seq_len, device, dtype):
    files = sorted(glob.glob(f"{token_dir}/*.npz"))
    idxs = np.random.randint(0, len(files), batch_size)
    batch_ids, batch_am = [], []
    for i in idxs:
        d = np.load(files[i])
        batch_ids.append(d["input_ids"][:seq_len])
        batch_am.append(d["attention_mask"][:seq_len])
    return (torch.tensor(np.stack(batch_ids), device=device, dtype=torch.long),
            torch.tensor(np.stack(batch_am), device=device, dtype=torch.float32))


@torch.no_grad()
def get_trajectory_hidden_and_z(model, ids, am, horizon, chunk_sz):
    """Run RWKV once with trajectory state injection, return:
       - per-chunk last-hidden-state [B, H, hidden_dim]
       - trajectory latents Z [B, H, latent_dim]
    """
    B = ids.shape[0]
    H, C = horizon, chunk_sz

    # 1. Encode trajectory latents
    z_flat = model._encode_trajectory_chunks(ids, am.bool())[0]  # [B*H, D]
    Z = z_flat.reshape(B, H, -1)                                   # [B, H, D]

    # 2. Predict states from Z, inject into cache
    states_list = model.predict_trajectory_states(Z)

    # 3. Forward pass: chunk by chunk to get per-chunk hidden states
    #    We want hidden states BEFORE injection for each chunk position
    hiddens = []
    for h in range(H):
        chunk_ids = ids[:, h * C:(h + 1) * C]
        chunk_am = am[:, h * C:(h + 1) * C]

        out = model.rwkv_model(input_ids=chunk_ids, attention_mask=chunk_am.bool(),
                               output_hidden_states=True, use_cache=True, return_dict=True)

        # Inject planned states for THIS chunk
        states_chunk = [s[:, h:h + 1, :, :, :] for s in states_list]
        past_kv = model.inject_into_cache(out.past_key_values, states_chunk)

        # Forward with injected cache to get meaningful hidden states
        out2 = model.rwkv_model(input_ids=chunk_ids, attention_mask=chunk_am.bool(),
                                past_key_values=past_kv,
                                output_hidden_states=True, use_cache=True, return_dict=True)

        last_hidden = out2.hidden_states[-1][:, -1, :]  # last token hidden
        hiddens.append(last_hidden)

    hiddens_tensor = torch.stack(hiddens, dim=1)  # [B, H, hidden_dim]
    return hiddens_tensor, Z


# ═══════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════
def train(args):
    model, tok, dtype, pad_id = C.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    horizon = model.trajectory_horizon
    chunk_sz = model.trajectory_chunk_size
    hidden_dim = model.rwkv_model.config.hidden_size
    latent_dim = model.latent_dim
    vocab_size = 65536

    dualspec = DualSpecHeads(hidden_dim, latent_dim, vocab_size, args.num_heads).to(args.device, dtype)
    opt = torch.optim.AdamW(dualspec.parameters(), lr=args.lr)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in dualspec.parameters()) / 1e6
    print(f"DualSpec: {n_params:.1f}M params, {args.num_heads} heads, "
          f"horizon={horizon}, chunk={chunk_sz}", flush=True)

    for step in range(1, args.num_steps + 1):
        ids, am = load_trajectory_batch(args.token_dir, args.batch_size,
                                        horizon * chunk_sz, args.device, dtype)

        hiddens, Z = get_trajectory_hidden_and_z(model, ids, am, horizon, chunk_sz)

        total_loss = 0.0
        for head_idx, head in enumerate(dualspec.heads):
            offset = head_idx + 1
            for h in range(horizon):
                # For chunk h, use z_{h+offset} as "future plan"
                future_h = h + offset
                if future_h >= horizon:
                    continue

                h_in = hiddens[:, h, :]     # hidden from chunk h
                z_in = Z[:, future_h, :]    # latent for chunk h+offset
                logits = head(dualspec.fusion(torch.cat([dualspec.hidden_proj(h_in),
                                                          dualspec.latent_proj(z_in)], dim=-1)))

                # Target: next token after chunk h's last position
                target_position = (h + 1) * chunk_sz + offset - 1
                if target_position >= ids.shape[1]:
                    continue
                targets = ids[:, target_position]
                total_loss += F.cross_entropy(logits, targets)

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"[step {step:5d}] loss={total_loss.item():.4f} | lr={args.lr:.1e}", flush=True)

        if step % args.save_every == 0:
            ckpt = {"dualspec_state": dualspec.state_dict(), "step": step}
            torch.save(ckpt, f"{args.save_dir}/dualspec_step{step}.pt")
            print(f"  saved step {step}", flush=True)

    torch.save({"dualspec_state": dualspec.state_dict(), "step": args.num_steps},
               f"{args.save_dir}/dualspec_final.pt")
    print("DualSpec training done.", flush=True)


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_dir", default="outputs_relay/dualspec-2.9B")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    train(ap.parse_args())
