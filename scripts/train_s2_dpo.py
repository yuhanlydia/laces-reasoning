"""DPO + VICReg self-distillation for S2 trajectory diffusion.

ONLY trains S2 trajectory_dit (S0, S1, RWKV frozen). No Z_clean reference.
Reward = -suffix_CE (RWKV judges its own state quality).
Loss = DPO(-log σ(β·(log_p(Z_good) - log_p(Z_bad)))) + VICReg(var+cov).

Based on joint-scratch champion checkpoint. Uses stochastic ancestral DDiM
sampler (same as GRPO) for differentiable log-prob.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/train_s2_dpo.py \
    --ckpt_dir outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --num_steps 1000 --steps 20 --cfg_scale 3.0 --lr 1e-5 --beta 1.0 \
    --device cuda:0
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as Diag
from models.state_hijacking_dit import cosine_alpha_bar
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix


# ---- Stochastic ancestral DDiM sampler (reused from train_s2_grpo.py) ----


def _denoise_eps(model, z, t_batch, cond, uncond, cfg_scale):
    eps_cond = model.trajectory_dit(z, t_batch, cond=cond)
    if cfg_scale == 1.0:
        return eps_cond
    eps_uncond = model.trajectory_dit(z, t_batch, cond=uncond)
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


def stochastic_sample_with_logprob(model, cond, steps, cfg_scale, device, dtype, eta=0.3):
    """Ancestral DDiM with injected Gaussian noise, returning (z0, sum_logprob).

    Only returns logprob (no KL to reference). For DPO we need pairwise comparison,
    not reference regularization.
    """
    H = int(model.trajectory_horizon)
    B = cond.shape[0]
    z = torch.randn(B, H, model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    total_logprob = torch.zeros(B, device=device, dtype=torch.float32)

    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(B)

        eps = _denoise_eps(model, z, t_batch, cond, uncond, cfg_scale)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()

        sigma = (
            eta
            * ((1 - ab_nxt) / (1 - ab_cur)).clamp(min=1e-6).sqrt()
            * (1 - ab_cur / ab_nxt).clamp(min=0).sqrt()
        )
        coef = (1 - ab_nxt - sigma**2).clamp(min=0).sqrt()
        mean = ab_nxt.sqrt() * z0_pred + coef * eps

        if i < steps - 1:
            noise = torch.randn_like(z)
            z_next = mean + sigma * noise
            std = sigma.clamp(min=1e-6)
            var = (std**2).clamp(min=1e-8)
            lp = -0.5 * (((z_next - mean) ** 2) / var + torch.log(2 * math.pi * var))
            total_logprob = total_logprob + lp.float().flatten(1).sum(dim=1)
            z = z_next
        else:
            z = mean

    return z, total_logprob


# ---- CE reward ----


@torch.no_grad()
def suffix_ce(model, z_traj, suffix_ids, blend, device):
    """Teacher-forced CE of RWKV predicting suffix under injected states. Lower = better."""
    C = int(model.trajectory_chunk_size)
    H = z_traj.shape[1]
    B = z_traj.shape[0]
    rwkv = model.rwkv_model
    layer_states = model.predict_trajectory_states(z_traj)

    suffix = torch.tensor(suffix_ids[: H * C], device=device, dtype=torch.long)
    if suffix.numel() < H * C:
        pad = torch.zeros(H * C - suffix.numel(), device=device, dtype=torch.long)
        suffix = torch.cat([suffix, pad])
    chunks = suffix.view(H, C)

    ce_per_member = torch.zeros(B, device=device, dtype=torch.float32)
    for h in range(H):
        chunk_h = chunks[h].unsqueeze(0).expand(B, C)
        seed = chunk_h[:, :1]
        out = rwkv(input_ids=seed, use_cache=True, return_dict=True)
        states_h = [ls[:, h] for ls in layer_states]
        cache = model.inject_into_cache(out.past_key_values, states_h)
        out2 = rwkv(input_ids=chunk_h, past_key_values=cache, use_cache=False, return_dict=True)
        logits = out2.logits[:, :-1, :]
        tgt = chunk_h[:, 1:]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            tgt.reshape(-1),
            reduction="none",
        ).view(B, -1)
        ce_per_member = ce_per_member + ce.mean(dim=1)

    return ce_per_member / max(1, H)


# ---- VICReg ----


def vicreg_loss(z):
    """Per-chunk VICReg: variance regularization + covariance decorrelation."""
    z = z.float()
    std = z.std(dim=0)
    l_var = F.relu(1.0 - std).mean()

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(1, z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    l_cov = (off**2).sum() / z.shape[1]

    return l_var, l_cov


# ---- Main ----


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt_dir",
        default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
    )
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_steps", type=int, default=500)
    ap.add_argument("--steps", type=int, default=20, help="Diffusion steps per sample")
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--eta", type=float, default=0.3, help="Ancestral noise level")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--blend", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=1.0, help="DPO temperature")
    ap.add_argument("--lambda_var", type=float, default=1e-4)
    ap.add_argument("--lambda_cov", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--save_dir", default="outputs_relay/s2-dpo-vicreg")
    ap.add_argument("--out", default="outputs_eval/s2_dpo_vicreg.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trajectory_dit.parameters():
        p.requires_grad = True
    trainable = [p for p in model.trajectory_dit.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)
    S = H * C
    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(0)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    prefix_len = (H // 2) * C
    num_pairs = 0
    num_good_swaps = 0

    hist_ce_good = []
    hist_ce_bad = []
    hist_dpo = []
    hist_var = []
    hist_cov = []

    for step in range(1, args.num_steps + 1):
        f = files[np.random.randint(len(files))]
        d = np.load(f)
        ids_np = d["input_ids"][:S]
        if len(ids_np) < S:
            continue

        ids = torch.tensor([ids_np], device=args.device, dtype=torch.long)
        pre_ids = ids[:, :prefix_len]
        pre_am = torch.ones_like(pre_ids, dtype=torch.float32)
        suffix_ids = ids_np[prefix_len:]

        z_prefix, _c, _l = encode_prefix(model, pre_ids, pre_am)
        cond = z_prefix.detach().expand(2, -1)  # 2 samples for DPO pair

        # Sample 2 Z's with stochastic ancestral sampler
        z_pair, logprob = stochastic_sample_with_logprob(
            model, cond, args.steps, args.cfg_scale, args.device, dtype, eta=args.eta,
        )

        # Compute CE for both
        ces = suffix_ce(model, z_pair.detach(), suffix_ids, args.blend, args.device)

        # DPO: lower CE = better
        ce_0, ce_1 = float(ces[0].item()), float(ces[1].item())
        if ce_0 < ce_1:
            lp_good, lp_bad = logprob[0], logprob[1]
            z_good, z_bad = z_pair[0:1], z_pair[1:2]
        else:
            lp_good, lp_bad = logprob[1], logprob[0]
            z_good, z_bad = z_pair[1:2], z_pair[0:1]
        num_pairs += 1
        if ce_0 < ce_1:
            num_good_swaps += 1  # tracking random baseline ~50%

        # DPO loss
        log_ratio = args.beta * (lp_good - lp_bad)
        dpo_loss = -F.logsigmoid(log_ratio).mean()

        # VICReg on the good Z (anti-collapse)
        l_var, l_cov = vicreg_loss(z_good[0])

        loss = dpo_loss + args.lambda_var * l_var + args.lambda_cov * l_cov

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        hist_ce_good.append(min(ce_0, ce_1))
        hist_ce_bad.append(max(ce_0, ce_1))
        hist_dpo.append(float(dpo_loss.item()))
        hist_var.append(float(l_var.item()))
        hist_cov.append(float(l_cov.item()))

        if step % 10 == 0:
            r20 = np.mean(hist_dpo[-20:]) if len(hist_dpo) >= 20 else np.mean(hist_dpo)
            ce_good_r20 = np.mean(hist_ce_good[-20:])
            ce_bad_r20 = np.mean(hist_ce_bad[-20:])
            good_pct = 100.0 * num_good_swaps / max(1, num_pairs)
            print(
                f"[step {step}] dpo={dpo_loss.item():.4f} running20={r20:.4f} "
                f"CE_good={ce_good_r20:.3f} CE_bad={ce_bad_r20:.3f} "
                f"gap={ce_bad_r20 - ce_good_r20:.3f} "
                f"var={l_var.item():.3f} cov={l_cov.item():.3f} "
                f"swap%={good_pct:.1f}%",
                flush=True,
            )

        if step % args.save_every == 0:
            torch.save(
                {"trajectory_dit": model.trajectory_dit.state_dict(), "step": step},
                f"{args.save_dir}/dpo_step{step}.pt",
            )

    # Save final
    torch.save(
        {"trajectory_dit": model.trajectory_dit.state_dict(), "step": args.num_steps},
        f"{args.save_dir}/dpo_final.pt",
    )

    # Report
    def _avg(xs, a, b):
        seg = xs[a:b]
        return float(np.mean(seg)) if seg else 0.0

    res = {
        "ckpt": args.ckpt_dir,
        "num_steps": len(hist_dpo),
        "beta": args.beta,
        "eta": args.eta,
        "first20_dpo": _avg(hist_dpo, 0, 20),
        "last20_dpo": _avg(hist_dpo, -20, None),
        "first20_ce_good": _avg(hist_ce_good, 0, 20),
        "last20_ce_good": _avg(hist_ce_good, -20, None),
        "first20_ce_bad": _avg(hist_ce_bad, 0, 20),
        "last20_ce_bad": _avg(hist_ce_bad, -20, None),
        "ce_gap_first": _avg(hist_ce_bad, 0, 20) - _avg(hist_ce_good, 0, 20),
        "ce_gap_last": _avg(hist_ce_bad, -20, None) - _avg(hist_ce_good, -20, None),
        "good_swap_pct": 100.0 * num_good_swaps / max(1, num_pairs),
        "first20_var": _avg(hist_var, 0, 20),
        "last20_var": _avg(hist_var, -20, None),
        "first20_cov": _avg(hist_cov, 0, 20),
        "last20_cov": _avg(hist_cov, -20, None),
        "hist_dpo": hist_dpo,
        "hist_ce_good": hist_ce_good,
        "hist_ce_bad": hist_ce_bad,
        "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    dpo_down = res["first20_dpo"] - res["last20_dpo"]
    ce_gap_up = res["ce_gap_last"] - res["ce_gap_first"]
    verdict = (
        "DPO DOWN + CE_GAP UP -> DPO works"
        if dpo_down > 0.01 and ce_gap_up > 0.01
        else "flat/negative -> tune beta/lr"
    )
    print(f"\n=== DPO+VICReg ===")
    print(f"  DPO loss: {res['first20_dpo']:.4f} -> {res['last20_dpo']:.4f}")
    print(f"  CE(good): {res['first20_ce_good']:.3f} -> {res['last20_ce_good']:.3f}")
    print(f"  CE(bad):  {res['first20_ce_bad']:.3f} -> {res['last20_ce_bad']:.3f}")
    print(f"  CE gap:   {res['ce_gap_first']:.3f} -> {res['ce_gap_last']:.3f}")
    print(f"  swap%:    {res['good_swap_pct']:.1f}%")
    print(f"  var:      {res['first20_var']:.3f} -> {res['last20_var']:.3f}")
    print(f"  cov:      {res['first20_cov']:.3f} -> {res['last20_cov']:.3f}")
    print(f"  verdict:  {verdict}")
    print(f"saved: {args.out}; ckpt: {args.save_dir}/dpo_final.pt")


if __name__ == "__main__":
    main()
