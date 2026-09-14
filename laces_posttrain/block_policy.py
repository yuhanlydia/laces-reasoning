"""Block-factor credit assignment for the original LACES continuous S2 policy."""
from __future__ import annotations

import math
import torch

from .policy import PolicyConfig, transition_mean, policy_mode


def block_log_ratio(action: torch.Tensor, new_mean: torch.Tensor, old_mean: torch.Tensor, std) -> torch.Tensor:
    """Joint Gaussian log-ratio within each latent block, shape ``[B,H]``.

    Only latent coordinates are summed. Blocks stay separate for credit assignment.
    Samples and old means are detached score-function observations.
    """
    if action.shape != new_mean.shape or old_mean.shape != new_mean.shape or new_mean.ndim != 3:
        raise ValueError("Expected action/means with identical [B,H,D] shapes")
    a = action.detach().double()
    old = old_mean.detach().to(device=new_mean.device, dtype=torch.float64)
    new = new_mean.double()
    s = torch.as_tensor(std, device=new.device, dtype=torch.float64)
    if not torch.isfinite(s).all() or (s <= 0).any():
        raise ValueError("Nonpositive/nonfinite Gaussian std")
    return (-.5 * (((a - new) / s).square() - ((a - old) / s).square())).sum(-1)


def potential_rewards(phi: torch.Tensor, exact: torch.Tensor, format_ok: torch.Tensor,
                      active_mask: torch.Tensor, *, exact_weight: float = 1., format_weight: float = .1) -> torch.Tensor:
    """Potential-difference rewards plus terminal verifier bonuses.

    ``phi`` is ``[G,H+1]``: pre-generation potential followed by one value per block.
    Terminal bonuses are attached to each rollout's last active block.
    """
    if phi.ndim != 2 or active_mask.ndim != 2 or phi.shape[0] != active_mask.shape[0] or phi.shape[1] != active_mask.shape[1] + 1:
        raise ValueError("phi must be [G,H+1] and active_mask [G,H]")
    if exact.shape != (phi.shape[0],) or format_ok.shape != (phi.shape[0],):
        raise ValueError("exact and format_ok must be [G]")
    if not all(math.isfinite(float(v)) for v in (exact_weight, format_weight)):
        raise ValueError("Nonfinite reward weights")
    mask = active_mask.bool()
    rewards = (phi[:, 1:] - phi[:, :-1]).float().masked_fill(~mask, 0.)
    for g in range(mask.shape[0]):
        active = torch.nonzero(mask[g], as_tuple=False).flatten()
        if active.numel():
            last = int(active[-1])
            rewards[g, last] = rewards[g, last] + exact_weight * exact[g].float() + format_weight * format_ok[g].float()
    if not torch.isfinite(rewards).all():
        raise FloatingPointError("Nonfinite block reward")
    return rewards


def returns_to_go(rewards: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    if rewards.shape != active_mask.shape or rewards.ndim != 2:
        raise ValueError("rewards and active_mask must share [G,H]")
    mask = active_mask.bool()
    masked = rewards.float().masked_fill(~mask, 0.)
    returns = torch.flip(torch.cumsum(torch.flip(masked, dims=[1]), dim=1), dims=[1])
    return returns.masked_fill(~mask, 0.)


def block_advantages(returns: torch.Tensor, active_mask: torch.Tensor, *, mode: str = "center", min_std: float = 1e-7) -> torch.Tensor:
    """Center within rollout group independently for each block.

    ``center`` is the Dr.GRPO-style control used by default. ``normalized`` retains
    the previous standard-deviation normalized GRPO as an explicit ablation.
    """
    if mode not in {"center", "normalized"}:
        raise ValueError("advantage mode must be center or normalized")
    if returns.shape != active_mask.shape or returns.ndim != 2:
        raise ValueError("returns and active_mask must share [G,H]")
    if not torch.isfinite(returns).all():
        raise ValueError("Nonfinite returns")
    mask = active_mask.bool()
    out = torch.zeros_like(returns, dtype=torch.float32)
    for h in range(returns.shape[1]):
        valid = mask[:, h]
        values = returns[valid, h].float()
        if values.numel() < 2:
            continue
        centered = values - values.mean()
        if mode == "normalized":
            std = values.std(unbiased=False)
            if std <= min_std:
                continue
            centered = centered / std
        out[valid, h] = centered
    return out.detach()


def block_grpo_update(model, reference, optimizer, cond: torch.Tensor, group, returns: torch.Tensor,
                      active_mask: torch.Tensor, config: PolicyConfig, *, advantage_mode: str = "center",
                      clip_range: float = .2, kl_coef: float = .01, inner_epochs: int = 1,
                      max_grad_norm: float = 1., max_log_ratio: float = 20.) -> dict:
    if len(group) != returns.shape[0] or returns.shape != active_mask.shape:
        raise ValueError("group and [G,H] credit tensors disagree")
    if inner_epochs < 1 or kl_coef < 0 or not 0 < clip_range < 1:
        raise ValueError("Invalid block GRPO arguments")
    advantages = block_advantages(returns, active_mask, mode=advantage_mode).to(cond.device)
    mask = active_mask.to(device=cond.device, dtype=torch.bool)
    if not advantages.abs().sum():
        optimizer.zero_grad(set_to_none=True)
        return dict(skipped_flat=True, grad_norm=0., pg_loss=0., reference_kl=0.,
                    clip_fraction=0., ratio_mean=1., ratio_max=1., updates=0,
                    advantage_std=0., active_blocks=int(mask.sum()))
    policy_mode(model); policy_mode(reference)
    active_per_group = mask.sum(1)
    if (active_per_group == 0).any():
        raise ValueError("Every rollout must have at least one active block")
    denom = int(mask.sum()) * config.steps
    report = dict(skipped_flat=False, grad_norm=0., pg_loss=0., reference_kl=0.,
                  clip_fraction=0., ratio_mean=0., ratio_max=0., updates=0,
                  advantage_std=float(advantages[mask].std(unbiased=False)), active_blocks=int(mask.sum()))
    for _ in range(inner_epochs):
        optimizer.zero_grad(set_to_none=True)
        pg_total = kl_total = clip_total = ratio_total = 0.
        ratio_max_seen = 0.
        for g, trajectory in enumerate(group):
            if len(trajectory.transitions) != config.steps:
                raise ValueError("Rollout/replay transition count mismatch")
            block_mask = mask[g]
            adv = advantages[g]
            for tr in trajectory.transitions:
                state = tr.state.to(cond.device)
                action = tr.action.to(cond.device)
                old_mean = tr.old_mean.to(cond.device)
                mean, std = transition_mean(model, state, cond, tr.index, config)
                if abs(float(std) - float(tr.std)) > 1e-7:
                    raise ValueError("Rollout/replay variance mismatch")
                lr = block_log_ratio(action, mean, old_mean, std)[0]
                active_lr = lr[block_mask]
                if not torch.isfinite(active_lr).all() or active_lr.abs().max() > max_log_ratio:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Block transition ratio overflow; reduce LR or update epochs")
                ratio = active_lr.exp()
                active_adv = adv[block_mask]
                clipped_ratio = ratio.clamp(1 - clip_range, 1 + clip_range)
                pg_vec = -torch.minimum(ratio * active_adv, clipped_ratio * active_adv)
                with torch.no_grad():
                    ref_mean, _ = transition_mean(reference, state, cond, tr.index, config)
                block_kl = ((mean - ref_mean).square() / (2 * std.square())).mean(-1)[0][block_mask]
                if not torch.isfinite(block_kl).all():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Nonfinite block KL")
                ((pg_vec.sum() + kl_coef * block_kl.sum()) / denom).backward()
                pg_total += float(pg_vec.detach().sum())
                kl_total += float(block_kl.detach().sum())
                clip_total += float(((ratio - 1).abs() > clip_range).float().sum())
                ratio_total += float(ratio.detach().sum())
                ratio_max_seen = max(ratio_max_seen, float(ratio.detach().max()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite S2 gradient")
        if norm == 0 and kl_coef == 0:
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("Nonflat block advantages but zero score-function gradient")
        optimizer.step()
        report.update(grad_norm=float(norm), pg_loss=pg_total / denom, reference_kl=kl_total / denom,
                      clip_fraction=clip_total / denom, ratio_mean=ratio_total / denom,
                      ratio_max=ratio_max_seen, updates=report["updates"] + 1)
    optimizer.zero_grad(set_to_none=True)
    return report
