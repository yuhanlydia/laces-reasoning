"""Zero-training diagnostic: does iterated single-z latent relay refine or collapse?

Core question for the multi-agent latent-relay direction: if agent_{t+1} takes ONLY
the previous latent z_t as its diffusion condition and samples z_{t+1}=p(.|z_t), does
the latent sequence stay diverse/meaningful, or collapse to a low-rank fixed point
(the same homogenization failure seen in the trajectory-collapse diagnostic)?

Measures, over K relay steps starting from real prefix-encoded conditions:
  - effective rank of {z_t} across prompts at each step (collapse => rank drops)
  - mean pairwise cosine across prompts at each step (collapse => cosine -> 1)
  - step-to-step drift ||z_{t+1}-z_t|| and cosine(z_{t+1}, z_t) (fixed point => drift->0)
  - norm trajectory (blow-up or decay)

Uses the existing 2.9B single-z prefix/suffix checkpoint; NO training.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg  # noqa: E402
from scripts.eval import diag_loop1_common as C  # noqa: E402


def _effective_rank(mat: torch.Tensor) -> float:
    x = mat - mat.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    ent = -(p * p.log()).sum()
    return float(ent.exp())


def _mean_pairwise_cosine(mat: torch.Tensor) -> float:
    x = torch.nn.functional.normalize(mat.float(), dim=-1)
    g = x @ x.t()
    n = g.shape[0]
    off = (g.sum() - g.diagonal().sum()) / (n * (n - 1))
    return float(off)


@torch.no_grad()
def run(ckpt_dir, device, num_prompts, steps, cfg_scale, k_relay, output,
        anchor=0.0, renorm=0.0):
    model, tokenizer, _dtype, pad_id = C.build_model(ckpt_dir, device)
    model.eval()
    model._prefix_suffix_s2 = True
    dtype = next(model.latent_dit.parameters()).dtype

    passages = C.PASSAGES[:num_prompts]
    # step 0: encode each prompt's prefix -> initial condition z_0 (a real intent)
    conds = []
    for p in passages:
        prefix_ids, _suffix_ids, _ln = C.split_prefix_suffix(tokenizer, p, model, pad_id)
        ids = torch.tensor([prefix_ids], device=device)
        am = torch.ones_like(ids)
        z0 = encode_prefix(model, ids, am).to(dtype)
        conds.append(z0[0])
    z_cur = torch.stack(conds, dim=0)  # [P, D]

    per_step = []

    def log_step(t, z, z_prev):
        er = _effective_rank(z)
        cos = _mean_pairwise_cosine(z)
        norm = float(z.float().norm(dim=-1).mean())
        rec = {"step": t, "effective_rank": er, "pairwise_cos": cos, "mean_norm": norm}
        if z_prev is not None:
            drift = float((z - z_prev).float().norm(dim=-1).mean())
            selfcos = float(
                torch.nn.functional.cosine_similarity(z.float(), z_prev.float(), dim=-1).mean()
            )
            rec["drift_from_prev"] = drift
            rec["cos_with_prev"] = selfcos
        per_step.append(rec)
        extra = ""
        if "drift_from_prev" in rec:
            extra = f" drift={rec['drift_from_prev']:.3f} cos_prev={rec['cos_with_prev']:.3f}"
        print(
            f"[relay {t}] eff_rank={er:.2f} pair_cos={cos:.3f} norm={norm:.2f}{extra}",
            flush=True,
        )

    z_problem = z_cur.clone()  # the anchor = each prompt's initial real intent
    target_norm = float(z_cur.float().norm(dim=-1).mean())  # ~10, training-dist norm
    log_step(0, z_cur, None)
    # relay: z_{t+1} = sample(cond=blend(z_problem, z_t)); optional renorm to keep in-dist
    for t in range(1, k_relay + 1):
        cond = z_cur
        if anchor > 0.0:
            cond = anchor * z_problem + (1.0 - anchor) * z_cur
        z_next = sample_ddim_cfg(model, cond, steps, cfg_scale, device, dtype)
        if renorm > 0.0:
            cur = z_next.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            z_next = (z_next.float() * (renorm / cur)).to(dtype)
        log_step(t, z_next, z_cur)
        z_cur = z_next

    result = {
        "ckpt_dir": ckpt_dir,
        "num_prompts": len(passages),
        "steps": steps,
        "cfg_scale": cfg_scale,
        "k_relay": k_relay,
        "anchor": anchor,
        "renorm": renorm,
        "target_norm": target_norm,
        "latent_dim": int(model.latent_dim),
        "per_step": per_step,
        "verdict_hint": (
            "COLLAPSE if effective_rank drops sharply / pairwise_cos->1 / drift->0 over relay; "
            "EXPLODE if norm blows up; STABLE if effective_rank and norm persist across steps."
        ),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    print(f"written: {output}", flush=True)
    print(
        f"gpu peak MiB: {torch.cuda.max_memory_allocated()//(1024*1024) if torch.cuda.is_available() else 0}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Iterated single-z latent relay collapse diagnostic.")
    ap.add_argument(
        "--ckpt_dir",
        default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_prompts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--k_relay", type=int, default=8)
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--renorm", type=float, default=0.0)
    ap.add_argument("--output", default="/tmp/diag_latent_relay.json")
    a = ap.parse_args()
    run(a.ckpt_dir, a.device, a.num_prompts, a.steps, a.cfg_scale, a.k_relay, a.output,
        anchor=a.anchor, renorm=a.renorm)


if __name__ == "__main__":
    main()
