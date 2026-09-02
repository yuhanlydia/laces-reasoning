"""ChunkDraft v2: bi-directional trajectory → token prediction.

  Z_full [H,32] → TransformerEncoder(4 layers) → cond [H, D]
  cond_h → MLP → first K tokens of chunk h

The transformer sees the FULL trajectory (bi-directional),
so each chunk's token prediction is context-aware.
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


class BiTrajDecoder(nn.Module):
    """Transformer over Z trajectory → per-chunk token prediction."""

    def __init__(self, latent_dim=32, vocab_size=65536, num_tokens=4, hidden=512, depth=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.z_proj = nn.Linear(latent_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=hidden * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_embed = nn.Embedding(32, hidden)  # max 32 chunks
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, vocab_size * num_tokens))

    def forward(self, Z):
        """Z: [B, H, 32] → logits: [B, H, K, vocab]"""
        B, H, D = Z.shape
        x = self.z_proj(Z)  # [B, H, hidden]
        x = x + self.pos_embed(torch.arange(H, device=Z.device)).unsqueeze(0)
        x = self.transformer(x)  # [B, H, hidden]
        logits = self.head(x)  # [B, H, K * vocab]
        return logits.reshape(B, H, self.num_tokens, -1)


def train(args):
    model, _, dtype, _ = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    H, C = model.trajectory_horizon, model.trajectory_chunk_size
    S = H * C

    decoder = BiTrajDecoder(latent_dim=model.latent_dim, vocab_size=65536,
                            num_tokens=args.num_tokens).to(args.device, dtype)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    n = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"BiTrajDecoder: {n:.1f}M params, {args.num_tokens} tokens/chunk", flush=True)

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

        logits = decoder(Z)  # [B, H, K, vocab]
        # Targets: first K tokens of each chunk
        targets = ids.reshape(B, H, C)[:, :, :args.num_tokens]  # [B, H, K]
        loss = F.cross_entropy(logits.reshape(-1, 65536), targets.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            ppl = torch.exp(loss / H).item()
            print(f"[step {step:5d}] loss={loss.item():.2f} ppl={ppl:.1f}", flush=True)
        if step % args.save_every == 0:
            torch.save({"decoder_state": decoder.state_dict(), "step": step},
                       f"{args.save_dir}/bitraj_step{step}.pt")

    torch.save({"decoder_state": decoder.state_dict(), "step": args.num_steps},
               f"{args.save_dir}/bitraj_final.pt")
    print("BiTraj done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_tokens", type=int, default=4)
    ap.add_argument("--num_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_dir", default="outputs_relay/bitraj-2.9B")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    train(ap.parse_args())
