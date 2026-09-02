"""Method 2 (advisor): Behavioral rank loss on top-k logits for S2.

Same motivation as Method 1 (diag_behavioral_nullspace: D_s=1.49 >> D_l=0.0147,
readout flattens diversity). Method 1 pushed full-distribution JS. Method 2 pushes
a coarser, more decode-relevant target: different Z's should produce different
top-k token RANKINGS at each position, since greedy decode only cares about the
argmax / top ordering, not the full simplex.

Behavioral rank diversity = mean pairwise (1 - soft top-k overlap) across Z's.
We use a differentiable soft-rank proxy: for the union of each pair's top-k token
ids, compare their softmax mass. Low overlap of high-mass tokens = behaviorally
diverse. Quality gate identical to Method 1 (keep only Z's within ce_margin).

Loss = quality_CE(best Z) - lambda_rank * mean_pairwise_rank_divergence

ONLY trains S2 trajectory_dit (S0, S1, RWKV frozen).

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/train_s2_behavioral_rank.py \
    --ckpt_dir outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --num_steps 400 --group_size 4 --lambda_rank 1.0 --ce_margin 0.5 --topk 20 --lr 1e-5
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag
from scripts.train_s2_logit_js import (
    sample_group_differentiable,
    group_suffix_logits_and_ce,
)


def pairwise_rank_divergence(logits, topk=20):
    """Mean pairwise behavioral-rank divergence over top-k tokens.

    For each pair (i,j) and each position, take the union of their top-k token
    ids and compare softmax mass on that union. Divergence = 0.5 * L1(mass_i,
    mass_j) restricted to the union (a TVD on the behaviorally relevant tokens).
    Higher = the two Z's put their probability on different tokens.
    logits: [G, T, V]. Returns scalar.
    """
    G, T, V = logits.shape
    p = F.softmax(logits.float(), dim=-1)  # [G, T, V]
    div_total = torch.zeros((), device=logits.device)
    n_pairs = 0
    for i in range(G):
        for j in range(i + 1, G):
            _, idx_i = torch.topk(logits[i].float(), topk, dim=-1)  # [T, k]
            _, idx_j = torch.topk(logits[j].float(), topk, dim=-1)
            # gather mass on each other's top-k (behavioral overlap probe)
            mi_on_i = p[i].gather(-1, idx_i)          # i's mass on i's top-k
            mj_on_i = p[j].gather(-1, idx_i)          # j's mass on i's top-k
            mi_on_j = p[i].gather(-1, idx_j)
            mj_on_j = p[j].gather(-1, idx_j)
            # TVD over the union (approx): average of the two directed gaps
            d_i = 0.5 * (mi_on_i - mj_on_i).abs().sum(dim=-1)  # [T]
            d_j = 0.5 * (mj_on_j - mi_on_j).abs().sum(dim=-1)
            div_total = div_total + (0.5 * (d_i + d_j)).mean()
            n_pairs += 1
    return div_total / max(1, n_pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_steps", type=int, default=400)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lambda_rank", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--ce_margin", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--save_dir", default="outputs_relay/s2-behavioral-rank")
    ap.add_argument("--out", default="outputs_eval/s2_behavioral_rank_probe.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model._prefix_suffix_trajectory_s2 = True
    model.train()  # FOOTGUN: .eval() breaks grad through the RWKV rollout.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trajectory_dit.parameters():
        p.requires_grad = True
    trainable = [p for p in model.trajectory_dit.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix
    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)
    S = H * C
    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(0)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    hist_ce, hist_rank, hist_kept = [], [], []
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
        cond = z_prefix.detach().expand(args.group_size, -1)

        z_group = sample_group_differentiable(model, cond, args.steps, args.cfg_scale,
                                              args.device, dtype, eta=args.eta)
        logits, ce = group_suffix_logits_and_ce(model, z_group, suffix_ids, args.device)

        best_ce = ce.min().detach()
        keep_mask = (ce <= best_ce + args.ce_margin)
        n_kept = int(keep_mask.sum().item())
        quality = ce.min()
        if n_kept >= 2:
            kept_idx = keep_mask.nonzero(as_tuple=True)[0]
            rank_div = pairwise_rank_divergence(logits[kept_idx], topk=args.topk)
        else:
            rank_div = torch.zeros((), device=args.device)

        loss = quality - args.lambda_rank * rank_div
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        hist_ce.append(float(ce.mean().item()))
        hist_rank.append(float(rank_div.item()))
        hist_kept.append(n_kept)
        if step % 10 == 0:
            r = lambda h: float(np.mean(h[-20:])) if len(h) >= 20 else float(np.mean(h))
            print(f"[step {step}] ce={r(hist_ce):.4f} rank_div={r(hist_rank):.5f} "
                  f"kept={r(hist_kept):.1f}/{args.group_size} loss={loss.item():.4f}", flush=True)
        if step % args.save_every == 0:
            torch.save({"trajectory_dit": model.trajectory_dit.state_dict(), "step": step},
                       f"{args.save_dir}/behavioral_rank_step{step}.pt")

    torch.save({"trajectory_dit": model.trajectory_dit.state_dict(), "step": args.num_steps},
               f"{args.save_dir}/behavioral_rank_final.pt")
    _a = lambda xs, a, b: float(np.mean(xs[a:b])) if xs[a:b] else 0.0
    res = {"ckpt": args.ckpt_dir, "num_steps": len(hist_ce),
           "first20_ce": _a(hist_ce, 0, 20), "last20_ce": _a(hist_ce, -20, None),
           "first20_rank": _a(hist_rank, 0, 20), "last20_rank": _a(hist_rank, -20, None),
           "hist_ce": hist_ce, "hist_rank": hist_rank, "hist_kept": hist_kept,
           "config": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== Method 2: behavioral rank diversity ===")
    print(f"  ce:       {res['first20_ce']:.4f} -> {res['last20_ce']:.4f}")
    print(f"  rank_div: {res['first20_rank']:.5f} -> {res['last20_rank']:.5f}  (higher=more diverse)")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
