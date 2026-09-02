"""GLSSD: General Latent State Self-Distillation (advisory's recommended first run).

Supervised on-policy self-distillation of the S2 trajectory sampler. NOT RL.
Trains S2 so its SAMPLED trajectory Z_hat (a) matches the clean encoder Z_clean
in effective-rank band and per-axis variance (anti-collapse), and (b) behaves
like Z_clean after S1 + frozen RWKV (logit KL + layer-gated state match). S0/S1
and RWKV stay frozen so any gain is attributable to a better sampler.

Loss = L_diffusion + rank-band + VICReg(var/cov) + logit-KL(D-OPSD) + state + CE.
"""
import argparse
import glob
import json
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
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix, sample_trajectory_cfg

USEFUL_LAYERS = {0, 2, 7, 11, 13, 19}
HARMFUL_LAYERS = set(range(27, 31))


def differentiable_effective_rank(z):
    z = z.float()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(1, z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=1e-8)
    return (ev.sum() ** 2) / (ev * ev).sum().clamp(min=1e-8)


def rank_band_loss(z_hat, z_clean):
    r_hat = differentiable_effective_rank(z_hat)
    r_clean = differentiable_effective_rank(z_clean).detach()
    return (torch.log(r_hat) - torch.log(r_clean)) ** 2


def vicreg_loss(z, gamma=1.0):
    z = z.float()
    std = z.std(dim=0)
    l_var = F.relu(gamma - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(1, z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    l_cov = (off ** 2).sum() / z.shape[1]
    return l_var, l_cov


def sample_z_hat_with_grad(model, cond, steps, cfg_scale, device, dtype):
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
        eps_cond = model.trajectory_dit(z, t_batch, cond=cond)
        eps = eps_cond if cfg_scale == 1.0 else \
            model.trajectory_dit(z, t_batch, cond=uncond) + cfg_scale * (eps_cond - model.trajectory_dit(z, t_batch, cond=uncond))
        z0 = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        z = ab_nxt.sqrt() * z0 + (1 - ab_nxt).sqrt() * eps
    return z


def chunk_logits(model, ids, am, z_traj, blend):
    states = model.predict_trajectory_states(z_traj)
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                           output_hidden_states=False, use_cache=True, return_dict=True)
    pkv = model.blend_into_cache(out.past_key_values, [s[:, 0] for s in states], blend)
    out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(), past_key_values=pkv,
                            use_cache=False, return_dict=True)
    return out2.logits, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_steps", type=int, default=300)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--blend", type=float, default=0.7)
    ap.add_argument("--lambda_rank", type=float, default=1e-3)
    ap.add_argument("--lambda_var", type=float, default=1e-4)
    ap.add_argument("--lambda_cov", type=float, default=1e-4)
    ap.add_argument("--lambda_logit", type=float, default=1e-3)
    ap.add_argument("--lambda_state", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs_eval/glssd_probe.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trajectory_dit.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.trajectory_dit.parameters() if p.requires_grad], lr=args.lr)

    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)
    S = H * C
    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(0)

    hist = {"rank_hat": [], "rank_clean": [], "logit_kl": [], "loss": []}
    for step in range(1, args.num_steps + 1):
        f = files[np.random.randint(len(files))]
        ids_np = np.load(f)["input_ids"][:S]
        if len(ids_np) < S:
            continue
        ids = torch.tensor([ids_np], device=args.device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)
        prefix_len = (H // 2) * C
        pre_ids = ids[:, :prefix_len]
        pre_am = am[:, :prefix_len]

        with torch.no_grad():
            z_clean = model._encode_trajectory_chunks(ids, am.bool())[0].reshape(1, H, -1)
        z_prefix, _c, _l = encode_prefix(model, pre_ids, pre_am)
        z_hat = sample_z_hat_with_grad(model, z_prefix, args.steps, args.cfg_scale, args.device, dtype)

        l_rank = rank_band_loss(z_hat[0], z_clean[0])
        l_var, l_cov = vicreg_loss(z_hat[0])

        with torch.no_grad():
            logits_clean, states_clean = chunk_logits(model, ids, am, z_clean, args.blend)
        logits_hat, states_hat = chunk_logits(model, ids, am, z_hat, args.blend)
        lp_hat = F.log_softmax(logits_hat.float(), dim=-1)
        p_clean = F.softmax(logits_clean.float(), dim=-1)
        l_logit = F.kl_div(lp_hat.reshape(-1, lp_hat.shape[-1]),
                           p_clean.reshape(-1, p_clean.shape[-1]),
                           reduction="batchmean")

        l_state = torch.zeros((), device=args.device)
        for li in range(len(states_hat)):
            if li in HARMFUL_LAYERS:
                continue
            gate = 1.0 if li in USEFUL_LAYERS else 0.2
            l_state = l_state + gate * F.mse_loss(states_hat[li].float(), states_clean[li].float().detach())

        loss = (args.lambda_rank * l_rank + args.lambda_var * l_var + args.lambda_cov * l_cov
                + args.lambda_logit * l_logit + args.lambda_state * l_state)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.trajectory_dit.parameters() if p.requires_grad], 1.0)
        opt.step()

        with torch.no_grad():
            rh = differentiable_effective_rank(z_hat[0]).item()
            rc = differentiable_effective_rank(z_clean[0]).item()
        hist["rank_hat"].append(rh)
        hist["rank_clean"].append(rc)
        hist["logit_kl"].append(float(l_logit.item()))
        hist["loss"].append(float(loss.item()))
        if step % 10 == 0:
            print(f"[step {step}] rank_hat={rh:.3f} rank_clean={rc:.3f} "
                  f"logit_kl={l_logit.item():.4f} l_rank={l_rank.item():.4f} "
                  f"l_state={l_state.item():.2f} loss={loss.item():.4f}", flush=True)

    def _avg(xs, a, b):
        seg = xs[a:b]
        return float(np.mean(seg)) if seg else 0.0
    res = {
        "ckpt": args.ckpt_dir, "num_steps": len(hist["loss"]),
        "first20_rank_hat": _avg(hist["rank_hat"], 0, 20),
        "last20_rank_hat": _avg(hist["rank_hat"], -20, None),
        "mean_rank_clean": float(np.mean(hist["rank_clean"])) if hist["rank_clean"] else 0.0,
        "first20_logit_kl": _avg(hist["logit_kl"], 0, 20),
        "last20_logit_kl": _avg(hist["logit_kl"], -20, None),
        "hist": hist, "config": vars(args),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\n=== GLSSD ===")
    print(f"  rank_hat: {res['first20_rank_hat']:.3f} -> {res['last20_rank_hat']:.3f} "
          f"(clean band {res['mean_rank_clean']:.3f})")
    print(f"  logit_kl: {res['first20_logit_kl']:.4f} -> {res['last20_logit_kl']:.4f}")
    rank_up = res['last20_rank_hat'] - res['first20_rank_hat']
    kl_down = res['first20_logit_kl'] - res['last20_logit_kl']
    verdict = ("rank UP + KL DOWN -> GLSSD works" if rank_up > 0.2 and kl_down > 0
               else "partial/flat -> tune lambdas")
    print(f"  verdict: {verdict}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
