"""Fine-tune S2 on condboundary to widen conditional diffusion space with VICReg.

Resumes from condboundary checkpoint (best diversity: cos=0.455 but spread=129).
Adds VICReg(var+cov) on z_0_pred to stabilize latent space while keeping diversity.
Only trains S2 trajectory_dit; S0, S1, RWKV frozen.

Loss = diff_loss + λ_var * VICReg_var(z_0_pred) + λ_cov * VICReg_cov(z_0_pred)

z_0_pred is the denoiser's one-step prediction of clean Z from noisy z_t.
VICReg_var keeps per-dimension variance above 1.0 (anti-collapse).
VICReg_cov decorrelates latent dimensions (anti-redundancy).

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/train_s2_diversity.py \
    --ckpt_dir outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary/step_00026000 \
    --num_steps 1000 --lr 1e-5 --lambda_var 1e-3 --lambda_cov 1e-3
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag
from models.state_hijacking_dit import cosine_alpha_bar


def vicreg_loss(z):
    z = z.float()
    std = z.std(dim=0)
    l_var = F.relu(1.0 - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(1, z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    l_cov = (off ** 2).sum() / z.shape[1]
    return l_var, l_cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary/step_00026000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lambda_var", type=float, default=1e-3)
    ap.add_argument("--lambda_cov", type=float, default=1e-3)
    ap.add_argument("--cfg_drop_prob", type=float, default=0.1)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--save_dir", default="outputs_relay/s2-diversity-vicreg")
    ap.add_argument("--out", default="outputs_eval/s2_diversity_vicreg.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trajectory_dit.parameters():
        p.requires_grad = True
    trainable = [p for p in model.trajectory_dit.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    # Override CFG dropout
    if hasattr(model, "_cfg_drop_prob"):
        model._cfg_drop_prob = float(args.cfg_drop_prob)

    H = int(model.trajectory_horizon)
    C = int(model.trajectory_chunk_size)
    S = H * C
    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(0)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    hist_diff, hist_var, hist_cov = [], [], []

    for step in range(1, args.num_steps + 1):
        # Load batch
        batch_ids = []
        batch_am = []
        for _ in range(args.batch_size):
            f = files[np.random.randint(len(files))]
            ids_np = np.load(f)["input_ids"][:S]
            if len(ids_np) < S:
                continue
            batch_ids.append(torch.tensor(ids_np, device=args.device, dtype=torch.long))
            batch_am.append(torch.ones(S, device=args.device, dtype=torch.float32))
        if len(batch_ids) < args.batch_size:
            continue
        ids = torch.stack(batch_ids)
        am = torch.stack(batch_am)

        # Forward: use the model's forward_trajectory_diffusion
        # We need to manually run the diffusion forward
        z_0 = model._encode_trajectory_chunks(ids, am.bool())[0]
        B = z_0.shape[0]

        # DDPM noise schedule
        t = torch.rand(B, device=args.device, dtype=dtype)
        ab = cosine_alpha_bar(t).to(dtype).clamp(min=1e-4)
        view = (B,) + (1,) * (z_0.dim() - 1)
        noise = torch.randn_like(z_0)
        z_t = ab.sqrt().view(view) * z_0 + (1.0 - ab).sqrt().view(view) * noise

        # Denoiser forward (no condition for unconditional diversity training)
        cond = torch.zeros(B, model.latent_dim, device=args.device, dtype=dtype)
        eps_pred = model.trajectory_dit(z_t, t, cond=cond)

        # Diffusion loss
        diff_loss = F.mse_loss(eps_pred, noise)

        # z_0_pred for VICReg
        z_0_pred = (z_t - (1.0 - ab).sqrt().view(view) * eps_pred) / ab.sqrt().view(view)

        # VICReg on predicted Z (per-sample, then average)
        l_var_total = torch.zeros((), device=args.device)
        l_cov_total = torch.zeros((), device=args.device)
        for b in range(B):
            lv, lc = vicreg_loss(z_0_pred[b])
            l_var_total = l_var_total + lv
            l_cov_total = l_cov_total + lc
        l_var = l_var_total / B
        l_cov = l_cov_total / B

        loss = diff_loss + args.lambda_var * l_var + args.lambda_cov * l_cov

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        hist_diff.append(float(diff_loss.item()))
        hist_var.append(float(l_var.item()))
        hist_cov.append(float(l_cov.item()))

        if step % 20 == 0:
            r20 = lambda h: np.mean(h[-20:]) if len(h) >= 20 else np.mean(h)
            print(f"[step {step}] diff={r20(hist_diff):.4f} var={r20(hist_var):.4f} cov={r20(hist_cov):.4f} loss={loss.item():.4f}", flush=True)

        if step % args.save_every == 0:
            torch.save({"trajectory_dit": model.trajectory_dit.state_dict(), "step": step},
                       f"{args.save_dir}/diversity_step{step}.pt")

    torch.save({"trajectory_dit": model.trajectory_dit.state_dict(), "step": args.num_steps},
               f"{args.save_dir}/diversity_final.pt")

    def _avg(xs, a, b):
        seg = xs[a:b]
        return float(np.mean(seg)) if seg else 0.0

    res = {
        "ckpt": args.ckpt_dir, "num_steps": len(hist_diff),
        "first20_diff": _avg(hist_diff, 0, 20), "last20_diff": _avg(hist_diff, -20, None),
        "first20_var": _avg(hist_var, 0, 20), "last20_var": _avg(hist_var, -20, None),
        "first20_cov": _avg(hist_cov, 0, 20), "last20_cov": _avg(hist_cov, -20, None),
        "hist_diff": hist_diff, "hist_var": hist_var, "hist_cov": hist_cov,
        "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== S2 Diversity VICReg ===")
    print(f"  diff: {res['first20_diff']:.4f} -> {res['last20_diff']:.4f}")
    print(f"  var:  {res['first20_var']:.4f} -> {res['last20_var']:.4f}")
    print(f"  cov:  {res['first20_cov']:.4f} -> {res['last20_cov']:.4f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
