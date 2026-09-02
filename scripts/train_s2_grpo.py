"""GRPO feasibility probe for S2 trajectory diffusion (latent self-distillation).

Idea: instead of a teacher-z, let RL optimize the S2 denoiser to sample better
16-latent trajectories. Reward = quality of the text the frozen LM produces from
those latents (negative suffix CE, dense + free). GRPO: for each prefix, sample a
GROUP of N latent trajectories, score each, use group-relative advantage, and do
policy-gradient on the S2 denoiser only (RWKV + S1 stay frozen).

To get a stochastic policy with a computable log-prob, we replace the
deterministic DDIM update with ancestral (DDPM-style) sampling: each denoising
step adds Gaussian noise with known mean/std, so log pi(z_{t-1} | z_t) is a
diagonal Gaussian log-density. The trajectory log-prob is the sum over steps.

This is a FEASIBILITY probe: run a few hundred steps and watch whether mean
reward rises. If it does, DDPO/GRPO on S2 is viable and worth a full run.
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


def _denoise_eps(model, z, t_batch, cond, uncond, cfg_scale):
    eps_cond = model.trajectory_dit(z, t_batch, cond=cond)
    if cfg_scale == 1.0:
        return eps_cond
    eps_uncond = model.trajectory_dit(z, t_batch, cond=uncond)
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


def stochastic_sample_with_logprob(model, cond, steps, cfg_scale, device, dtype,
                                   eta=0.3, ref_model=None):
    """Ancestral DDIM sampler with injected Gaussian noise (eta>0) so each step is
    a stochastic action. Returns (z0, sum_logprob, sum_kl). sum_kl is the
    per-step KL(policy_step || reference_step) summed over the trajectory, using
    that both steps are Gaussians with identical sigma so KL reduces to the
    squared mean gap over 2*variance. Gradients flow through the policy denoiser."""
    H = int(model.trajectory_horizon)
    B = cond.shape[0]
    z = torch.randn(B, H, model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    total_logprob = torch.zeros(B, device=device, dtype=torch.float32)
    total_kl = torch.zeros(B, device=device, dtype=torch.float32)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(B)
        eps = _denoise_eps(model, z, t_batch, cond, uncond, cfg_scale)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        sigma = eta * ((1 - ab_nxt) / (1 - ab_cur)).clamp(min=1e-6).sqrt() * (1 - ab_cur / ab_nxt).clamp(min=0).sqrt()
        coef = (1 - ab_nxt - sigma ** 2).clamp(min=0).sqrt()
        mean = ab_nxt.sqrt() * z0_pred + coef * eps
        if ref_model is not None:
            with torch.no_grad():
                eps_ref = _denoise_eps(ref_model, z, t_batch, cond, uncond, cfg_scale)
                z0_ref = (z - (1 - ab_cur).sqrt() * eps_ref) / ab_cur.sqrt()
                mean_ref = ab_nxt.sqrt() * z0_ref + coef * eps_ref
        if i < steps - 1:
            noise = torch.randn_like(z)
            z_next = mean + sigma * noise
            std = sigma.clamp(min=1e-6)
            var = (std ** 2).clamp(min=1e-8)
            lp = -0.5 * (((z_next - mean) ** 2) / var + torch.log(2 * math.pi * var))
            total_logprob = total_logprob + lp.float().flatten(1).sum(dim=1)
            if ref_model is not None:
                kl_step = ((mean - mean_ref) ** 2) / (2 * var)
                total_kl = total_kl + kl_step.float().flatten(1).sum(dim=1)
            z = z_next
        else:
            z = mean
    return z, total_logprob, total_kl


@torch.no_grad()
def suffix_ce_reward(model, prefix_cache, prefix_logits, z_traj, suffix_ids, blend, device):
    """Dense reward = -mean teacher-forced CE of frozen LM predicting the real
    suffix under the injected planned states from z_traj. Higher = better.

    FAST: fully batched. The entire group (B members) x each chunk is processed
    as ONE forward per chunk (teacher-forced, whole chunk at once), instead of
    token-by-token AR. Independent per-chunk injection (blend into a fresh cache)
    matches the parallel-decoding semantics and avoids a cross-chunk serial chain.
    """
    C = int(model.trajectory_chunk_size)
    H = z_traj.shape[1]
    B = z_traj.shape[0]
    rwkv = model.rwkv_model
    layer_states = model.predict_trajectory_states(z_traj)
    suffix = torch.tensor(suffix_ids[:H * C], device=device, dtype=torch.long)
    if suffix.numel() < H * C:
        pad = torch.zeros(H * C - suffix.numel(), device=device, dtype=torch.long)
        suffix = torch.cat([suffix, pad])
    chunks = suffix.view(H, C)

    ce_per_member = torch.zeros(B, device=device, dtype=torch.float32)
    ce_count = 0
    for h in range(H):
        chunk_h = chunks[h].unsqueeze(0).expand(B, C)
        seed = chunk_h[:, :1]
        out = rwkv(input_ids=seed, use_cache=True, return_dict=True)
        states_h = [ls[:, h] for ls in layer_states]
        cache = model.inject_into_cache(out.past_key_values, states_h)
        out2 = rwkv(input_ids=chunk_h, past_key_values=cache, use_cache=False, return_dict=True)
        logits = out2.logits[:, :-1, :]
        tgt = chunk_h[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             tgt.reshape(-1), reduction="none").view(B, -1)
        ce_per_member = ce_per_member + ce.mean(dim=1)
        ce_count += 1
    return -(ce_per_member / max(1, ce_count))


def vicreg_loss(z):
    """Per-chunk VICReg: variance regularization + covariance decorrelation."""
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
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_steps", type=int, default=300)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--blend", type=float, default=0.7)
    ap.add_argument("--kl_coef", type=float, default=0.1)
    ap.add_argument("--lambda_var", type=float, default=1e-4)
    ap.add_argument("--lambda_cov", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs_eval/s2_grpo_vicreg_probe.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import copy
    from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix
    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trajectory_dit.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.trajectory_dit.parameters() if p.requires_grad], lr=args.lr)

    ref_denoiser = copy.deepcopy(model.trajectory_dit).eval()
    for p in ref_denoiser.parameters():
        p.requires_grad = False

    class _RefShim:
        def __init__(self, base, denoiser):
            self._base = base
            self.trajectory_dit = denoiser
            self.trajectory_horizon = base.trajectory_horizon
            self.latent_dim = base.latent_dim
    ref_model = _RefShim(model, ref_denoiser)

    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)
    S = H * C
    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(0)

    reward_hist = []
    for step in range(1, args.num_steps + 1):
        f = files[np.random.randint(len(files))]
        d = np.load(f)
        ids_np = d["input_ids"][:S]
        if len(ids_np) < S:
            continue
        prefix_len = (H // 2) * C
        pre_ids = torch.tensor([ids_np[:prefix_len]], device=args.device, dtype=torch.long)
        pre_am = torch.ones_like(pre_ids, dtype=torch.float32)
        suffix_ids = ids_np[prefix_len:]
        z_prefix, prefix_cache, prefix_logits = encode_prefix(model, pre_ids, pre_am)
        cond = z_prefix.detach().expand(args.group_size, -1)

        z_group, logprob, kl = stochastic_sample_with_logprob(
            model, cond, args.steps, args.cfg_scale, args.device, dtype,
            eta=args.eta, ref_model=ref_model if args.kl_coef > 0 else None)

        rewards = suffix_ce_reward(model, prefix_cache, prefix_logits, z_group.detach(),
                                   suffix_ids, args.blend, args.device)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        pg_loss = -(adv.detach() * logprob).mean()
        kl_loss = kl.mean() if args.kl_coef > 0 else torch.zeros((), device=args.device)

        # VICReg anti-collapse on the best Z in the group
        best_idx = int(rewards.argmax().item())
        l_var, l_cov = vicreg_loss(z_group[best_idx])
        loss = pg_loss + args.kl_coef * kl_loss + args.lambda_var * l_var + args.lambda_cov * l_cov
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.trajectory_dit.parameters() if p.requires_grad], 1.0)
        opt.step()

        reward_hist.append(float(rewards.mean().item()))
        if step % 10 == 0:
            recent = np.mean(reward_hist[-20:])
            spread = float(rewards.max().item() - rewards.min().item())
            print(f"[step {step}] mean_reward={rewards.mean().item():.4f} "
                  f"best={rewards.max().item():.4f} worst={rewards.min().item():.4f} "
                  f"spread={spread:.4f} kl={float(kl_loss.item()):.4f} "
                  f"var={l_var.item():.3f} cov={l_cov.item():.3f} "
                  f"running20={recent:.4f} loss={loss.item():.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    first20 = float(np.mean(reward_hist[:20])) if len(reward_hist) >= 20 else float(np.mean(reward_hist))
    last20 = float(np.mean(reward_hist[-20:]))
    res = {
        "ckpt": args.ckpt_dir, "num_steps": len(reward_hist), "group_size": args.group_size,
        "diffusion_steps": args.steps, "eta": args.eta, "lr": args.lr, "kl_coef": args.kl_coef,
        "first20_mean_reward": first20, "last20_mean_reward": last20,
        "reward_improvement": last20 - first20, "reward_hist": reward_hist,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== GRPO feasibility ===")
    print(f"  first20 mean reward: {first20:.4f}")
    print(f"  last20  mean reward: {last20:.4f}")
    print(f"  improvement: {last20 - first20:+.4f}")
    verdict = ("REWARD RISES -> GRPO on S2 viable" if last20 - first20 > 0.02
               else "flat/negative -> not viable as-is")
    print(f"  verdict: {verdict}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
