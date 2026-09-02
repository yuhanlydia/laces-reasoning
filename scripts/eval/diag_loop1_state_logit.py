# pyright: reportAny=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""Loop-1 diagnostics #1, #3, #4: clean-Z vs sampled-Z divergence.

#1 STATE L2: distance between S1(z_clean) and S1(z_sampled) recurrent states,
   per layer and overall, absolute and relative (||clean-sampled|| / ||clean||).
#3 LOGIT KL: teacher-forced decode with each z injected into the frozen RWKV;
   KL between the clean-injected and sampled-injected next-token distributions.
#4 LOGIT SUBSPACE DRIFT: SVD of the logit matrices; principal-angle / cosine
   agreement of the top-k right singular subspaces (clean vs sampled).

All operate on existing checkpoints. No retraining.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval.diag_loop1_common as C  # noqa: E402

DEFAULT_CKPT = "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary/step_00026000"


def _state_l2(model, z_clean, z_sampled):
    sc = C.per_layer_states_flat(model, z_clean)
    ss = C.per_layer_states_flat(model, z_sampled)
    per_layer_abs = []
    per_layer_rel = []
    for a, b in zip(sc, ss):
        diff = (a - b).norm()
        base = a.norm().clamp(min=1e-8)
        per_layer_abs.append(float(diff.item()))
        per_layer_rel.append(float((diff / base).item()))
    return per_layer_abs, per_layer_rel


def _logits_from_z(model, prefix_ids, suffix_ids, z, device):
    pre = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    pre_mask = torch.ones_like(pre)
    suf = torch.tensor([suffix_ids], device=device, dtype=torch.long)
    suf_mask = torch.ones_like(suf)
    logits, _ = model._prefix_suffix_trajectory_logits_from_z(
        pre, pre_mask, suf, suf_mask, z, max_chunks=0
    )
    return logits  # [B*H, C, vocab]


def _logit_kl(clean_logits, sampled_logits):
    t = clean_logits[:, :-1, :].float()
    s = sampled_logits[:, :-1, :].float()
    tp = F.softmax(t, dim=-1)
    slp = F.log_softmax(s, dim=-1)
    tlp = F.log_softmax(t, dim=-1)
    sp = F.softmax(s, dim=-1)
    kl_fwd = F.kl_div(slp, tp, reduction="none").sum(-1).mean()   # KL(clean||sampled)
    kl_rev = F.kl_div(tlp, sp, reduction="none").sum(-1).mean()   # KL(sampled||clean)
    return float(kl_fwd.item()), float(kl_rev.item())


def _subspace_drift(clean_logits, sampled_logits, topk=8, max_pos=256):
    # Flatten to [positions, vocab]; subsample positions & top vocab dims for memory.
    cl = clean_logits.reshape(-1, clean_logits.shape[-1]).float()
    sl = sampled_logits.reshape(-1, sampled_logits.shape[-1]).float()
    n = min(cl.shape[0], sl.shape[0], max_pos)
    cl, sl = cl[:n], sl[:n]
    cl = cl - cl.mean(0, keepdim=True)
    sl = sl - sl.mean(0, keepdim=True)
    # right singular vectors span the logit-direction subspace
    _, _, vc = torch.linalg.svd(cl, full_matrices=False)
    _, _, vs = torch.linalg.svd(sl, full_matrices=False)
    k = min(topk, vc.shape[0], vs.shape[0])
    a = vc[:k]  # [k, vocab]
    b = vs[:k]
    m = a @ b.t()  # [k,k] cosines between subspace bases
    sv = torch.linalg.svdvals(m).clamp(-1, 1)  # cos of principal angles
    return float(sv.mean().item()), float(sv.min().item())


@torch.no_grad()
def run(ckpt_dir: str, device: str, num_samples: int, steps: int, cfg_scale: float, output: str):
    model, tokenizer, dtype, pad_id = C.build_model(ckpt_dir, device)
    passages = C.PASSAGES[:num_samples]

    abs_all, rel_all, klf_all, klr_all, cos_all, cmin_all = [], [], [], [], [], []
    per_layer_rel_acc = None

    for i, passage in enumerate(passages):
        pre, suf, _ln = C.split_prefix_suffix(tokenizer, passage, model, pad_id)
        z_clean = C.get_z_clean(model, suf, device)
        z_sampled = C.get_z_sampled(model, pre, device, dtype, steps, cfg_scale)

        pl_abs, pl_rel = _state_l2(model, z_clean, z_sampled)
        abs_all.append(sum(pl_abs) / len(pl_abs))
        rel_all.append(sum(pl_rel) / len(pl_rel))
        if per_layer_rel_acc is None:
            per_layer_rel_acc = [[] for _ in pl_rel]
        for li, v in enumerate(pl_rel):
            per_layer_rel_acc[li].append(v)

        cl = _logits_from_z(model, pre, suf, z_clean, device)
        sl = _logits_from_z(model, pre, suf, z_sampled, device)
        kf, kr = _logit_kl(cl, sl)
        klf_all.append(kf)
        klr_all.append(kr)
        cmean, cmin = _subspace_drift(cl, sl)
        cos_all.append(cmean)
        cmin_all.append(cmin)
        print(f"[{i+1}/{len(passages)}] state_rel={rel_all[-1]:.3f} "
              f"KL(c||s)={kf:.3f} KL(s||c)={kr:.3f} subspace_cos={cmean:.3f}")

    def ms(x):
        return {"mean": statistics.mean(x), "std": statistics.pstdev(x) if len(x) > 1 else 0.0}

    per_layer_rel_mean = [statistics.mean(v) for v in per_layer_rel_acc] if per_layer_rel_acc else []
    result = {
        "diagnostic": "loop1_state_logit(#1,#3,#4)",
        "ckpt": ckpt_dir,
        "num_samples": len(passages),
        "steps": steps,
        "cfg_scale": cfg_scale,
        "state_l2_abs_mean_over_layers": ms(abs_all),
        "state_l2_rel_mean_over_layers": ms(rel_all),
        "state_l2_rel_per_layer_mean": per_layer_rel_mean,
        "logit_kl_clean_given_sampled": ms(klf_all),
        "logit_kl_sampled_given_clean": ms(klr_all),
        "logit_subspace_cos_mean": ms(cos_all),
        "logit_subspace_cos_min": ms(cmin_all),
        "interpretation": (
            "High state_l2_rel + high logit_kl + low subspace_cos => the sampled "
            "latent decodes to a substantially different (potentially off-manifold) "
            "state/behavior than the clean latent, explaining the clean->sampled gap."
        ),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== SUMMARY (state/logit) ===")
    print(f"state L2 rel (mean over layers): {result['state_l2_rel_mean_over_layers']['mean']:.4f}")
    print(f"logit KL(clean||sampled):        {result['logit_kl_clean_given_sampled']['mean']:.4f}")
    print(f"logit KL(sampled||clean):        {result['logit_kl_sampled_given_clean']['mean']:.4f}")
    print(f"logit subspace cos (top-8):      {result['logit_subspace_cos_mean']['mean']:.4f}")
    print(f"written: {output}")
    print(f"gpu peak MiB: {int(torch.cuda.max_memory_allocated()/1024/1024) if device.startswith('cuda') else 0}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--output", default="/tmp/diag_loop1_state_logit.json")
    args = ap.parse_args()
    run(args.ckpt_dir, args.device, args.num_samples, args.steps, args.cfg_scale, args.output)


if __name__ == "__main__":
    main()
