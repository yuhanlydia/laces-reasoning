"""Method 5/10 (advisor): Mode-conditioned S2 for MCQ-style tasks.

Different framing from Method 1/2. Instead of forcing generic diversity on a
single conditional p(Z|prefix), attach a discrete MODE signal so that different
modes deterministically steer S2 to different answers. Motivation: for MCQ, we
do not want "wide fuzzy diversity"; we want K distinct, controllable behaviors
(one per candidate answer), so a downstream selector can pick the best.

Mechanism (no from-scratch retrain): a small learnable mode table
  mode_embed: [K, latent_dim]
is ADDED to z_prefix before it enters the denoiser's cond path. Mode k =>
cond = z_prefix + mode_embed[k]. Trains mode_embed + trajectory_dit; S0/S1/RWKV
frozen. Objective:
  - per-mode quality: each mode's best CE stays low (mode still generates sane text)
  - cross-mode separation: modes should produce DIFFERENT logits (top-k TVD high)
So the K modes carve the narrow S2 space into K controllable directions.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/train_s2_mode_cond.py \
    --ckpt_dir outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --num_steps 400 --num_modes 4 --lambda_sep 1.0 --lr 1e-5
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag
from models.state_hijacking_dit import cosine_alpha_bar
from scripts.train_s2_logit_js import group_suffix_logits_and_ce, pairwise_js


def sample_one_differentiable(model, cond, steps, cfg_scale, device, dtype, eta=0.3):
    H = int(model.trajectory_horizon)
    B = cond.shape[0]
    z = torch.randn(B, H, model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(B)
        eps_c = model.trajectory_dit(z, t_batch, cond=cond)
        if cfg_scale == 1.0:
            eps = eps_c
        else:
            eps_u = model.trajectory_dit(z, t_batch, cond=uncond)
            eps = eps_u + cfg_scale * (eps_c - eps_u)
        z0 = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        sigma = eta * ((1 - ab_nxt) / (1 - ab_cur)).clamp(min=1e-6).sqrt() * (1 - ab_cur / ab_nxt).clamp(min=0).sqrt()
        coef = (1 - ab_nxt - sigma ** 2).clamp(min=0).sqrt()
        mean = ab_nxt.sqrt() * z0 + coef * eps
        z = mean + sigma * torch.randn_like(z) if i < steps - 1 else mean
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_steps", type=int, default=400)
    ap.add_argument("--num_modes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--mode_lr", type=float, default=1e-3)
    ap.add_argument("--lambda_sep", type=float, default=1.0)
    ap.add_argument("--mode_init_std", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--save_dir", default="outputs_relay/s2-mode-cond")
    ap.add_argument("--out", default="outputs_eval/s2_mode_cond_probe.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model._prefix_suffix_trajectory_s2 = True
    model.train()  # FOOTGUN: .eval() breaks grad through the RWKV rollout.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trajectory_dit.parameters():
        p.requires_grad = True

    mode_embed = torch.nn.Parameter(
        args.mode_init_std * torch.randn(args.num_modes, model.latent_dim, device=args.device, dtype=dtype)
    )
    dit_params = [p for p in model.trajectory_dit.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": dit_params, "lr": args.lr},
        {"params": [mode_embed], "lr": args.mode_lr},
    ])

    from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix
    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)
    S = H * C
    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(0)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    hist_ce, hist_sep = [], []
    for step in range(1, args.num_steps + 1):
        f = files[np.random.randint(len(files))]
        ids_np = np.load(f)["input_ids"][:S]
        if len(ids_np) < S:
            continue
        prefix_len = (H // 2) * C
        pre_ids = torch.tensor(np.array([ids_np[:prefix_len]]), device=args.device, dtype=torch.long)
        pre_am = torch.ones_like(pre_ids, dtype=torch.float32)
        suffix_ids = ids_np[prefix_len:]
        with torch.no_grad():
            z_prefix, _, _ = encode_prefix(model, pre_ids, pre_am)

        # One Z per mode: cond_k = z_prefix + mode_embed[k]
        cond = z_prefix.detach() + mode_embed  # [K, latent_dim]
        z_group = sample_one_differentiable(model, cond, args.steps, args.cfg_scale,
                                            args.device, dtype, eta=args.eta)
        logits, ce = group_suffix_logits_and_ce(model, z_group, suffix_ids, args.device)

        # Per-mode quality: mean CE across modes (each mode must stay sane).
        quality = ce.mean()
        # Cross-mode separation: modes should behave differently (JS over logits).
        sep = pairwise_js(logits) if args.num_modes >= 2 else torch.zeros((), device=args.device)

        loss = quality - args.lambda_sep * sep
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dit_params + [mode_embed], 1.0)
        opt.step()

        hist_ce.append(float(ce.mean().item()))
        hist_sep.append(float(sep.item()))
        if step % 10 == 0:
            r = lambda h: float(np.mean(h[-20:])) if len(h) >= 20 else float(np.mean(h))
            mnorm = float(mode_embed.detach().norm(dim=-1).mean().item())
            print(f"[step {step}] ce={r(hist_ce):.4f} sep={r(hist_sep):.5f} "
                  f"mode_norm={mnorm:.3f} loss={loss.item():.4f}", flush=True)
        if step % args.save_every == 0:
            torch.save({"trajectory_dit": model.trajectory_dit.state_dict(),
                        "mode_embed": mode_embed.detach().cpu(), "step": step},
                       f"{args.save_dir}/mode_cond_step{step}.pt")

    torch.save({"trajectory_dit": model.trajectory_dit.state_dict(),
                "mode_embed": mode_embed.detach().cpu(), "step": args.num_steps},
               f"{args.save_dir}/mode_cond_final.pt")
    _a = lambda xs, a, b: float(np.mean(xs[a:b])) if xs[a:b] else 0.0
    res = {"ckpt": args.ckpt_dir, "num_steps": len(hist_ce), "num_modes": args.num_modes,
           "first20_ce": _a(hist_ce, 0, 20), "last20_ce": _a(hist_ce, -20, None),
           "first20_sep": _a(hist_sep, 0, 20), "last20_sep": _a(hist_sep, -20, None),
           "hist_ce": hist_ce, "hist_sep": hist_sep, "config": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== Method 5/10: mode-conditioned S2 ({args.num_modes} modes) ===")
    print(f"  ce:  {res['first20_ce']:.4f} -> {res['last20_ce']:.4f}")
    print(f"  sep: {res['first20_sep']:.5f} -> {res['last20_sep']:.5f}  (higher=modes more distinct)")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
