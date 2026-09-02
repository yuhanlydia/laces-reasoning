#!/usr/bin/env python3
"""Four-diagnostic check of whether the champion S2 conditional diffusion has
collapsed to a near-deterministic map  H(Z|x) ~ 0.

Diagnostics (all on the trajectory joint-scratch champion, zero-training unless
stated):

  [1] Conditional diversity  D_Z = E_{i!=j}[1 - cos(Z_i, Z_j)]
      for 32 samples under the SAME prefix. Reported on raw vectors AND on
      de-meaned residuals (the champion is known to carry a huge shared mean
      ||z_bar||~9.19, which makes raw cos ~1.0 trivially; de-meaning isolates
      whether the samples are *truly* identical or only share a common offset).

  [2] Noise sensitivity  Var_eps(Z) / Var_x(Z)
      Var_eps = variance across the 32 noise draws (same x), Var_x = variance of
      the per-prompt mean across prompts. Ratio -> 0 means I(Z;eps|x) ~ 0.

  [3] Output diversity  D_y = unique decoded strings across the 32 samples.
      If D_Z ~ 0 AND D_y ~ 0 -> strong deterministic collapse.

  [4] Multimodal-futures capacity (the decisive one): train the S2 denoiser
      from scratch on synthetic data where a SINGLE condition c maps to K>=2
      distinct valid targets (sampled uniformly). If the trained denoiser can
      still separate them (D_Z >> 0), the original determinism is single-modal
      DATA, not a diffusion-capacity limit. If it collapses to one point even
      here, it is a true model collapse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix, sample_trajectory_cfg, generate,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from models.state_hijacking_dit import cosine_alpha_bar  # noqa: E402

PROMPTS = [
    "The history of artificial intelligence",
    "In a shocking turn of events, scientists discovered",
    "The recipe calls for two cups of flour and",
    "Climate change is one of the most pressing",
    "Once upon a time in a distant kingdom",
    "The stock market fell sharply today after",
    "To solve this equation, first isolate the variable",
    "The human immune system defends the body by",
    "Ancient Rome was founded according to legend by",
    "Machine learning models require large amounts of",
    "The novel opens with a description of the",
    "Photosynthesis converts sunlight into chemical energy",
    "The Supreme Court ruled today that the law",
    "Basketball is a sport played by two teams",
    "The chemical formula for water is composed of",
    "During the Renaissance, artists began to explore",
]


def _pairwise_cos_sq(mat: torch.Tensor) -> tuple[float, float, float]:
    """Return (mean 1-cos raw, mean 1-cos de-meaned, mean pairwise cos raw)."""
    x = mat.float()
    n = x.shape[0]
    xn = F.normalize(x, dim=-1)
    sim = xn @ xn.t()
    off = (sim.sum() - sim.diagonal().sum()) / max(1, n * (n - 1))
    # de-meaned
    xc = x - x.mean(dim=0, keepdim=True)
    xcn = F.normalize(xc, dim=-1)
    simc = xcn @ xcn.t()
    offc = (simc.sum() - simc.diagonal().sum()) / max(1, n * (n - 1))
    return float(1.0 - off), float(1.0 - offc), float(off)


@torch.no_grad()
def sample_n(model, cond, n, steps, cfg_scale, device, dtype):
    """Sample n trajectory latents Z under the same condition, distinct seeds."""
    zs = []
    for k in range(n):
        torch.manual_seed(1000 + k)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(1000 + k)
        Z = sample_trajectory_cfg(model, cond, steps, cfg_scale, device, dtype)
        zs.append(Z[0].float().cpu())  # [H, D]
    return torch.stack(zs, dim=0)  # [n, H, D]


def diag_1_2_3(args):
    device = args.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    model._training_stage = 2
    model._cfg_drop_prob = float(getattr(model, "_cfg_drop_prob", 0.0))
    dtype = next(model.alpha_heads.parameters()).dtype

    per_prompt = {}
    flat_means = []
    for pi, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        am = torch.ones_like(ids)
        cond = encode_prefix(model, ids, am)[0]  # [1, D]
        cond = cond.to(dtype)
        Z = sample_n(model, cond, args.n_samples, args.steps, args.cfg_scale, device, dtype)
        # [n, H, D]
        flat = Z.reshape(args.n_samples, -1)  # [n, H*D]
        d_raw, d_demean, cos_raw = _pairwise_cos_sq(flat)

        # variance decomposition
        var_eps = flat.var(dim=0).mean().item()  # across noise draws (same x)
        mean_z = flat.mean(dim=0)
        flat_means.append(mean_z.double())
        per_prompt[prompt] = {
            "D_Z_raw": d_raw,
            "D_Z_demeaned": d_demean,
            "cos_raw": cos_raw,
            "var_eps": var_eps,
            "mean_norm": float(mean_z.norm().item()),
            "residual_norm_frac": float(
                (flat - mean_z).norm(dim=-1).mean().item() / max(mean_z.norm().item(), 1e-9)
            ),
        }

    mean_z_stack = torch.stack(flat_means, dim=0)
    var_x = mean_z_stack.var(dim=0).mean().item()
    var_eps_mean = float(sum(v["var_eps"] for v in per_prompt.values()) / len(per_prompt))

    ratio = var_eps_mean / max(var_x, 1e-12)

    print("\n=== [1] Conditional diversity D_Z (same prefix, 32 samples) ===")
    for p, v in per_prompt.items():
        print(f"  {p[:44]:44s}  D_Z_raw={v['D_Z_raw']:.4f}  D_Z_demeaned={v['D_Z_demeaned']:.4f}  cos_raw={v['cos_raw']:.4f}")
    print(f"\n  MEAN D_Z_raw     = {sum(v['D_Z_raw'] for v in per_prompt.values())/len(per_prompt):.4f}")
    print(f"  MEAN D_Z_demeaned= {sum(v['D_Z_demeaned'] for v in per_prompt.values())/len(per_prompt):.4f}")
    print(f"  MEAN residual_norm_frac = {sum(v['residual_norm_frac'] for v in per_prompt.values())/len(per_prompt):.4f}")

    print("\n=== [2] Noise sensitivity Var_eps(Z)/Var_x(Z) ===")
    print(f"  Var_eps (across noise, same x) = {var_eps_mean:.6e}")
    print(f"  Var_x   (across prompts)       = {var_x:.6e}")
    print(f"  ratio Var_eps/Var_x            = {ratio:.6f}")

    return {
        "per_prompt": per_prompt,
        "mean_D_Z_raw": sum(v["D_Z_raw"] for v in per_prompt.values()) / len(per_prompt),
        "mean_D_Z_demeaned": sum(v["D_Z_demeaned"] for v in per_prompt.values()) / len(per_prompt),
        "mean_residual_norm_frac": sum(v["residual_norm_frac"] for v in per_prompt.values()) / len(per_prompt),
        "var_eps": var_eps_mean,
        "var_x": var_x,
        "ratio_var_eps_over_var_x": ratio,
    }


@torch.no_grad()
def diag_3_output(args):
    """Decode 32 samples under a few prompts; count unique outputs (D_y)."""
    device = args.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    model._training_stage = 2
    model._cfg_drop_prob = float(getattr(model, "_cfg_drop_prob", 0.0))
    dtype = next(model.alpha_heads.parameters()).dtype

    gen_args = SimpleNamespace(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
    )
    out = {}
    for prompt in args.decode_prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        am = torch.ones_like(ids)
        z_prefix, prefix_cache, prefix_logits = encode_prefix(model, ids, am)
        cond = z_prefix.to(dtype)
        texts = []
        for k in range(args.n_samples):
            torch.manual_seed(1000 + k)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(1000 + k)
            Z = sample_trajectory_cfg(model, cond, args.steps, args.cfg_scale, device, dtype)
            text, _ = generate(model, tokenizer, ids, am, prefix_cache, prefix_logits, Z, gen_args)
            texts.append(text)
        uniq = sorted(set(texts))
        out[prompt] = {
            "n_samples": args.n_samples,
            "n_unique": len(uniq),
            "unique_rate": len(uniq) / args.n_samples,
            "examples": [t.replace("\n", "\\n")[:160] for t in uniq[:4]],
        }
        print(f"\n[prompt] {prompt}\n  n_unique={len(uniq)}/{args.n_samples}  unique_rate={len(uniq)/args.n_samples:.3f}")
        for t in uniq[:4]:
            print(f"    - {t.replace(chr(10),' ')[:120]}")
    return out


def diag_4_multimodal(args):
    """Train a fresh S2 denoiser on synthetic multi-modal conditional data.

    Same architecture family as the champion (TrajectoryLatentRWKV, birwkv
    bidirectional, condboundary), but the target distribution is explicitly
    multimodal: condition c -> one of K distinct valid latent targets, uniform.
    If D_Z >> 0 after training, determinism = single-modal data, not capacity.
    """
    from models.state_hijacking_dit import TrajectoryLatentRWKV

    torch.manual_seed(0)
    device = args.device
    D, H = args.mm_latent_dim, args.mm_horizon
    K = args.mm_n_modes
    hidden = args.mm_hidden
    depth = args.mm_depth

    denoiser = TrajectoryLatentRWKV(
        latent_dim=D, horizon=H, hidden_size=hidden, depth=depth,
        bidirectional=True, use_cond_adaln=False, use_cond_boundary=True,
    ).to(device).train()

    # synthetic data: N conditions; each has K valid targets spread far apart
    N = args.mm_n_cond
    conds = torch.randn(N, D, device=device)
    conds = conds / conds.norm(dim=-1, keepdim=True)
    targets = {}
    for i in range(N):
        # K orthonormal directions per condition -> clearly distinct modes
        basis = torch.randn(K, D, device=device)
        basis = basis / basis.norm(dim=-1, keepdim=True)
        modes = []
        for k in range(K):
            Zk = torch.randn(H, D, device=device)
            Zk = Zk + args.mm_mode_sep * basis[k]  # offset along a distinct dir
            modes.append(Zk)
        targets[i] = torch.stack(modes, dim=0)  # [K, H, D]

    opt = torch.optim.AdamW(denoiser.parameters(), lr=1e-3)
    steps = args.mm_train_steps
    for it in range(steps):
        i = torch.randint(0, N, (1,)).item()
        k = torch.randint(0, K, (1,)).item()
        z0 = targets[i][k].unsqueeze(0)  # [1,H,D]
        c = conds[i].unsqueeze(0)
        t = torch.rand(1, device=device)
        ab = cosine_alpha_bar(t).clamp(min=1e-4)
        eps = torch.randn_like(z0)
        zt = ab.sqrt() * z0 + (1 - ab).sqrt() * eps
        eps_pred = denoiser(zt, t, cond=c)
        loss = F.mse_loss(eps_pred, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 200 == 0:
            print(f"  [mm train] step {it+1}/{steps} loss={loss.item():.4f}")

    # sample: for each condition, 32 draws; measure D_Z
    denoiser.eval()
    ts = torch.linspace(1.0, 0.0, args.steps + 1, device=device)
    dz_cond = []
    with torch.no_grad():
        for i in range(N):
            c = conds[i].unsqueeze(0)
            samples = []
            for s in range(32):
                torch.manual_seed(5000 + s)
                z = torch.randn(1, H, D, device=device)
                for st in range(args.steps):
                    t_cur, t_nxt = ts[st], ts[st + 1]
                    ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).clamp(min=1e-4)
                    ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).clamp(min=1e-4)
                    t_b = t_cur.expand(1)
                    ep = denoiser(z, t_b, cond=c)
                    z0_pred = (z - (1 - ab_cur).sqrt() * ep) / ab_cur.sqrt()
                    z = ab_nxt.sqrt() * z0_pred + (1 - ab_nxt).sqrt() * ep
                samples.append(z[0].reshape(-1).float().cpu())
            S = torch.stack(samples, dim=0)
            d_raw, d_demean, _cos = _pairwise_cos_sq(S)
            dz_cond.append((d_raw, d_demean))

    mean_d_raw = sum(a for a, b in dz_cond) / N
    mean_d_demean = sum(b for a, b in dz_cond) / N
    print("\n=== [4] Multimodal-futures capacity (synthetic, K=%d modes) ===" % K)
    print(f"  D_Z_raw     = {mean_d_raw:.4f}   (0=collapsed to one point, 1=fully separated)")
    print(f"  D_Z_demeaned= {mean_d_demean:.4f}")
    return {
        "K_modes": K, "n_cond": N, "latent_dim": D, "horizon": H,
        "mode_sep": args.mm_mode_sep, "train_steps": steps,
        "mean_D_Z_raw": mean_d_raw, "mean_D_Z_demeaned": mean_d_demean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--decode_prompts", nargs="*",
                    default=["The capital of France is", "The chemical formula for water is",
                             "Basketball is a sport played by"])
    ap.add_argument("--skip_output", action="store_true")
    ap.add_argument("--skip_multimodal", action="store_true")
    # multimodal synthetic args
    ap.add_argument("--mm_latent_dim", type=int, default=32)
    ap.add_argument("--mm_horizon", type=int, default=16)
    ap.add_argument("--mm_n_modes", type=int, default=4)
    ap.add_argument("--mm_n_cond", type=int, default=8)
    ap.add_argument("--mm_mode_sep", type=float, default=6.0)
    ap.add_argument("--mm_hidden", type=int, default=256)
    ap.add_argument("--mm_depth", type=int, default=4)
    ap.add_argument("--mm_train_steps", type=int, default=1500)
    ap.add_argument("--output", default="outputs_eval/diag_conditional_determinism.json")
    ap.add_argument("--only", choices=("all", "1_2", "3", "4"), default="all")
    args = ap.parse_args()

    result = {"ckpt_dir": args.ckpt_dir, "cfg_scale": args.cfg_scale, "n_samples": args.n_samples}

    if args.only in ("all", "1_2"):
        result["diag1_2_conditional_noise"] = diag_1_2_3(args)
    if args.only in ("all", "3") and not args.skip_output:
        result["diag3_output_diversity"] = diag_3_output(args)
    if args.only in ("all", "4") and not args.skip_multimodal:
        result["diag4_multimodal_capacity"] = diag_4_multimodal(args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
