"""Method 1 (advisor): Quality-constrained logit JS diversity loss for S2.

Motivation (from diag_behavioral_nullspace.py on champion):
  D_z=0.60  D_s=1.49  D_l=0.0147  D_y=0.33  unique=2.2/8
  => States DO differ (D_s > D_z), but the state->logit map FLATTENS the
     difference (D_l tiny). The bottleneck is the READOUT, not S1.

So instead of latent-space diversity (VICReg, already shown useless), we push
diversity DIRECTLY at the logit distribution, where it collapses. To avoid the
LaDiR failure mode (diversity destroys quality -> garbage), we add a hard
quality gate: each Z's mean suffix CE must stay within a margin of the best Z's
CE, otherwise its diversity gradient is masked out.

Loss = quality_CE(best Z) - lambda_js * mean_pairwise_JS(logits over quality-OK Z's)

ONLY trains S2 trajectory_dit (S0, S1, RWKV frozen). Group sampled with the
same differentiable ancestral DDiM sampler used by GRPO.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/train_s2_logit_js.py \
    --ckpt_dir outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --num_steps 400 --group_size 4 --lambda_js 1.0 --ce_margin 0.5 --lr 1e-5
"""
import argparse, glob, json, math, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag
from models.state_hijacking_dit import cosine_alpha_bar


def _denoise_eps(model, z, t_batch, cond, uncond, cfg_scale):
    eps_cond = model.trajectory_dit(z, t_batch, cond=cond)
    if cfg_scale == 1.0:
        return eps_cond
    eps_uncond = model.trajectory_dit(z, t_batch, cond=uncond)
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


def sample_group_differentiable(model, cond, steps, cfg_scale, device, dtype, eta=0.3):
    """Ancestral DDiM sampler, gradients flow through denoiser. Returns z0 [G,H,D]."""
    H = int(model.trajectory_horizon)
    G = cond.shape[0]
    z = torch.randn(G, H, model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(G)
        eps = _denoise_eps(model, z, t_batch, cond, uncond, cfg_scale)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        sigma = eta * ((1 - ab_nxt) / (1 - ab_cur)).clamp(min=1e-6).sqrt() * (1 - ab_cur / ab_nxt).clamp(min=0).sqrt()
        coef = (1 - ab_nxt - sigma ** 2).clamp(min=0).sqrt()
        mean = ab_nxt.sqrt() * z0_pred + coef * eps
        if i < steps - 1:
            z = mean + sigma * torch.randn_like(z)
        else:
            z = mean
    return z


def group_suffix_logits_and_ce(model, z_group, suffix_ids, device):
    """For each group member, per-chunk teacher-forced logits + mean CE.
    Returns (logits [G, T, V], ce_per_member [G]). Independent per-chunk inject
    (matches parallel decoding). Grad flows through predict_trajectory_states.
    """
    C = int(model.trajectory_chunk_size)
    H = z_group.shape[1]
    G = z_group.shape[0]
    rwkv = model.rwkv_model
    layer_states = model.predict_trajectory_states(z_group)
    suffix = torch.tensor(suffix_ids[:H * C], device=device, dtype=torch.long)
    if suffix.numel() < H * C:
        suffix = torch.cat([suffix, torch.zeros(H * C - suffix.numel(), device=device, dtype=torch.long)])
    chunks = suffix.view(H, C)
    all_logits = []
    ce_per_member = torch.zeros(G, device=device, dtype=torch.float32)
    ce_count = 0
    for h in range(H):
        chunk_h = chunks[h].unsqueeze(0).expand(G, C)
        seed = chunk_h[:, :1]
        with torch.no_grad():
            out = rwkv(input_ids=seed, use_cache=True, return_dict=True)
        states_h = [ls[:, h] for ls in layer_states]
        cache = model.inject_into_cache(out.past_key_values, states_h)
        out2 = rwkv(input_ids=chunk_h, past_key_values=cache, use_cache=False, return_dict=True)
        logits = out2.logits[:, :-1, :]  # [G, C-1, V]
        tgt = chunk_h[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             tgt.reshape(-1), reduction="none").view(G, -1)
        ce_per_member = ce_per_member + ce.mean(dim=1)
        ce_count += 1
        all_logits.append(logits)
    logits_cat = torch.cat(all_logits, dim=1)  # [G, T, V]
    return logits_cat, ce_per_member / max(1, ce_count)


def pairwise_js(logits, topk=100):
    """Mean pairwise Jensen-Shannon divergence over top-k logit positions.
    logits: [G, T, V]. Returns scalar (higher = more diverse behavior)."""
    G, T, V = logits.shape
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()  # [G, T, V]
    js_total = torch.zeros((), device=logits.device)
    n_pairs = 0
    for i in range(G):
        for j in range(i + 1, G):
            m = 0.5 * (p[i] + p[j]).clamp(min=1e-9)
            logm = m.log()
            # KL(p_i || m) + KL(p_j || m), both summed over vocab, mean over T
            kl_i = (p[i] * (logp[i] - logm)).sum(dim=-1)
            kl_j = (p[j] * (logp[j] - logm)).sum(dim=-1)
            js_total = js_total + (0.5 * (kl_i + kl_j)).mean()
            n_pairs += 1
    return js_total / max(1, n_pairs)


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
    ap.add_argument("--lambda_js", type=float, default=1.0)
    ap.add_argument("--ce_margin", type=float, default=0.5,
                    help="A Z's diversity grad is kept only if its CE <= best_CE + margin")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--save_dir", default="outputs_relay/s2-logit-js")
    ap.add_argument("--out", default="outputs_eval/s2_logit_js_probe.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model._prefix_suffix_trajectory_s2 = True
    # FOOTGUN: .eval() silently breaks grad through the RWKV rollout (states->dit).
    model.train()
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

    hist_ce, hist_js, hist_kept = [], [], []
    for step in range(1, args.num_steps + 1):
        f = files[np.random.randint(len(files))]
        ids_np = np.load(f)["input_ids"][:S]
        if len(ids_np) < S:
            continue
        prefix_len = (H // 2) * C
        pre_ids = torch.tensor([ids_np[:prefix_len]], device=args.device, dtype=torch.long)
        pre_am = torch.ones_like(pre_ids, dtype=torch.float32)
        suffix_ids = ids_np[prefix_len:]
        with torch.no_grad():
            z_prefix, _, _ = encode_prefix(model, pre_ids, pre_am)
        cond = z_prefix.detach().expand(args.group_size, -1)

        z_group = sample_group_differentiable(model, cond, args.steps, args.cfg_scale,
                                              args.device, dtype, eta=args.eta)
        logits, ce = group_suffix_logits_and_ce(model, z_group, suffix_ids, args.device)

        # Quality gate: keep only Z's within ce_margin of the best member.
        best_ce = ce.min().detach()
        keep_mask = (ce <= best_ce + args.ce_margin)
        n_kept = int(keep_mask.sum().item())

        # Quality term: minimize best-member CE (pull whole group toward good region)
        quality = ce.min()

        # Diversity term: JS over the kept (quality-OK) members only
        if n_kept >= 2:
            kept_idx = keep_mask.nonzero(as_tuple=True)[0]
            js = pairwise_js(logits[kept_idx])
        else:
            js = torch.zeros((), device=args.device)

        loss = quality - args.lambda_js * js
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        hist_ce.append(float(ce.mean().item()))
        hist_js.append(float(js.item()))
        hist_kept.append(n_kept)
        if step % 10 == 0:
            r = lambda h: float(np.mean(h[-20:])) if len(h) >= 20 else float(np.mean(h))
            print(f"[step {step}] ce={r(hist_ce):.4f} js={r(hist_js):.5f} "
                  f"kept={r(hist_kept):.1f}/{args.group_size} loss={loss.item():.4f}", flush=True)
        if step % args.save_every == 0:
            torch.save({"trajectory_dit": model.trajectory_dit.state_dict(), "step": step},
                       f"{args.save_dir}/logit_js_step{step}.pt")

    torch.save({"trajectory_dit": model.trajectory_dit.state_dict(), "step": args.num_steps},
               f"{args.save_dir}/logit_js_final.pt")
    _a = lambda xs, a, b: float(np.mean(xs[a:b])) if xs[a:b] else 0.0
    res = {"ckpt": args.ckpt_dir, "num_steps": len(hist_ce),
           "first20_ce": _a(hist_ce, 0, 20), "last20_ce": _a(hist_ce, -20, None),
           "first20_js": _a(hist_js, 0, 20), "last20_js": _a(hist_js, -20, None),
           "hist_ce": hist_ce, "hist_js": hist_js, "hist_kept": hist_kept,
           "config": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== Method 1: logit JS diversity ===")
    print(f"  ce: {res['first20_ce']:.4f} -> {res['last20_ce']:.4f}")
    print(f"  js: {res['first20_js']:.5f} -> {res['last20_js']:.5f}  (higher=more diverse)")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
