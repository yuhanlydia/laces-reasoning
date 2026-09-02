"""Go/no-go: do single-z latents of consecutive REASONING STEPS carry signal?

For the Coconut-style single-z multi-step chain we split each openr1_math answer's
<think> into reasoning steps (by blank lines), encode EACH step into one single-z
latent z_k via the existing 2.9B single-z S0, and ask: is z_{k+1} predictable from
z_{<=k} (and the problem latent)? If yes, a chained latent-reasoning S2 is worth
building/finetuning; if the z-sequence is noise, it is not.

NOTE: reasoning-step slicing (by \n\n, semantic units) is DIFFERENT from trajectory
chunking (fixed 32-token windows): each z_k here is one complete reasoning step.

Zero training. Metrics per relay position:
  - linear-AR predictability: MSE of predicting z_{k+1} from z_k vs predicting the mean
    (skill score = 1 - mse_ar/mse_mean; >0 means real signal beyond the global mean)
  - cosine(z_{k+1}, z_k): step-to-step continuity
  - anchor effect: does adding the problem latent improve predictability
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.sample_prefix_suffix_cfg import encode_prefix  # noqa: E402
from scripts.eval import diag_loop1_common as C  # noqa: E402

DATA = "preprocessed_data/s2_traj_sft_512_openr1_math_220k_default"


@torch.no_grad()
def encode_text_to_z(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.shape[-1] < 2:
        return None
    am = torch.ones_like(ids)
    return encode_prefix(model, ids, am)[0].float().cpu()


@torch.no_grad()
def run(ckpt_dir, device, num_samples, min_steps, output):
    model, tokenizer, _dtype, _pad = C.build_model(ckpt_dir, device)
    model.eval()

    files = sorted(glob.glob(f"{DATA}/openr1_math_*_tokens.npz"))[: num_samples * 3]
    seqs = []  # list of (z_problem, [z_1..z_N])
    for f in files:
        d = np.load(f)
        ids = d["input_ids"]
        pl = int(d["prompt_lengths"])
        rm = int(d["response_mask"].sum())
        problem = tokenizer.decode(ids[:pl])
        resp = tokenizer.decode(ids[pl : pl + rm])
        steps = [s.strip() for s in resp.split("\n\n") if len(s.strip()) > 10]
        if len(steps) < min_steps:
            continue
        zp = encode_text_to_z(model, tokenizer, problem, device)
        zs = [encode_text_to_z(model, tokenizer, s, device) for s in steps]
        zs = [z for z in zs if z is not None]
        if zp is None or len(zs) < min_steps:
            continue
        seqs.append((zp, torch.stack(zs, dim=0)))
        if len(seqs) >= num_samples:
            break
        print(f"encoded {len(seqs)}/{num_samples} ({len(zs)} steps)", flush=True)

    # Collect (z_k, z_{k+1}, z_problem) triples across all sequences.
    prev, nxt, prob = [], [], []
    for zp, zs in seqs:
        for k in range(zs.shape[0] - 1):
            prev.append(zs[k])
            nxt.append(zs[k + 1])
            prob.append(zp)
    prev = torch.stack(prev)
    nxt = torch.stack(nxt)
    prob = torch.stack(prob)

    mean_next = nxt.mean(dim=0, keepdim=True)
    mse_mean = ((nxt - mean_next) ** 2).mean().item()

    def ridge_predict(X, Y, lam=1.0):
        Xa = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
        A = Xa.t() @ Xa + lam * torch.eye(Xa.shape[1])
        W = torch.linalg.solve(A, Xa.t() @ Y)
        return Xa @ W

    pred_ar = ridge_predict(prev, nxt)
    mse_ar = ((nxt - pred_ar) ** 2).mean().item()
    pred_anch = ridge_predict(torch.cat([prev, prob], dim=1), nxt)
    mse_anch = ((nxt - pred_anch) ** 2).mean().item()

    cos_step = torch.nn.functional.cosine_similarity(prev, nxt, dim=-1).mean().item()

    result = {
        "ckpt_dir": ckpt_dir,
        "num_sequences": len(seqs),
        "num_step_pairs": int(prev.shape[0]),
        "mse_predict_mean_baseline": mse_mean,
        "mse_linear_AR_from_prev": mse_ar,
        "mse_linear_AR_prev_plus_problem": mse_anch,
        "skill_AR": 1.0 - mse_ar / mse_mean,
        "skill_AR_with_anchor": 1.0 - mse_anch / mse_mean,
        "cos_step_to_step": cos_step,
        "verdict_hint": (
            "SIGNAL if skill_AR > ~0.1 (z_{k+1} predictable from z_k beyond mean); "
            "ANCHOR HELPS if skill_AR_with_anchor > skill_AR; "
            "NOISE if skill_AR ~ 0 (reasoning-step z-sequence is unpredictable)."
        ),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s0/step_00050000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_samples", type=int, default=60)
    ap.add_argument("--min_steps", type=int, default=4)
    ap.add_argument("--output", default="outputs_eval/diag_reasoning_step_latents.json")
    a = ap.parse_args()
    run(a.ckpt_dir, a.device, a.num_samples, a.min_steps, a.output)


if __name__ == "__main__":
    main()
