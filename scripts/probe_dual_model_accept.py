"""Base two-model speculative acceptance probe (NO training, NO RELAY state).

Question this answers (the #1 risk for the "small drafts, large verifies" plan):
    When an independent small RWKV (0.4B) drafts tokens and an independent large
    RWKV (2.9B champion backbone) verifies them, what is the raw greedy acceptance
    rate = P(drafter argmax == verifier argmax at the same position)?

This is the FLOOR of any speculative-decoding acceptance between these two models.
RELAY plan-alignment can only improve on this; if it is already very low, the whole
small-for-large direction needs a different alignment strategy before training.

Method (lossless-greedy, standard block speculative decoding acceptance):
  - Take an OWT sample, use first `prefix_len` tokens as context.
  - Drafter (0.4B) autoregressively proposes k tokens from the prefix.
  - Verifier (2.9B) does ONE forward over [prefix + k drafts]; at each drafted
    position i it produces the token it would itself emit (argmax).
  - Greedy accept: walk i=0..k-1; accept while draft[i] == verifier_argmax[i];
    stop at first mismatch. This is exactly the accept rule real spec-decoding uses.
  - Aggregate: mean accepted tokens per block, per-position accept rate,
    and the implied 1+avg_accept acceptance-estimated speedup (NOT wall-clock).

Only the base RWKV backbones are loaded; no RELAY checkpoint is required.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def load_backbone(path, device, dtype):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def draft_k(drafter, prefix_ids, k, device):
    """0.4B autoregressively proposes k tokens (greedy). Returns list[int]."""
    out = drafter(input_ids=prefix_ids, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    next_tok = out.logits[0, -1].argmax().item()
    drafts = [next_tok]
    cur = torch.tensor([[next_tok]], device=device, dtype=torch.long)
    for _ in range(k - 1):
        out = drafter(input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        next_tok = out.logits[0, -1].argmax().item()
        drafts.append(next_tok)
        cur = torch.tensor([[next_tok]], device=device, dtype=torch.long)
    return drafts


@torch.no_grad()
def verify_block(verifier, prefix_ids, drafts, device):
    """2.9B ONE forward over prefix+drafts. Return verifier argmax at each draft slot.

    Verifier argmax for draft position i is taken from the logits at the token
    that PRECEDES draft[i]: position (prefix_len-1+i) predicts draft slot i.
    """
    draft_t = torch.tensor([drafts], device=device, dtype=torch.long)
    full = torch.cat([prefix_ids, draft_t], dim=1)
    out = verifier(input_ids=full, use_cache=False, return_dict=True)
    plen = prefix_ids.shape[1]
    k = len(drafts)
    # logits at index (plen-1 + i) predict the (i-th) drafted token slot
    verifier_argmax = [out.logits[0, plen - 1 + i].argmax().item() for i in range(k)]
    return verifier_argmax


@torch.no_grad()
def run(args):
    device = args.device
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    print(f"loading drafter (0.4B): {args.drafter}", flush=True)
    drafter = load_backbone(args.drafter, device, dtype)
    print(f"loading verifier (2.9B): {args.verifier}", flush=True)
    verifier = load_backbone(args.verifier, device, dtype)

    files = sorted(glob.glob(f"{args.token_dir}/*_tokens.npz"))
    if not files:
        files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    rng = np.random.RandomState(42)
    chosen = [files[i] for i in rng.choice(len(files), args.num_samples, replace=False)]

    k = args.block
    per_pos_accept = [0] * k
    per_pos_total = [0] * k
    total_accepted = 0
    total_blocks = 0
    first_tok_agree = 0  # position-0 agreement = ceiling on any accept

    for si, path in enumerate(chosen):
        d = np.load(path)
        ids = torch.tensor([d["input_ids"][: args.prefix_len + args.gen_len]],
                           device=device, dtype=torch.long)
        # slide over the generation region in blocks of k
        for start in range(args.prefix_len, args.prefix_len + args.gen_len - k, k):
            prefix_ids = ids[:, :start]
            drafts = draft_k(drafter, prefix_ids, k, device)
            vargmax = verify_block(verifier, prefix_ids, drafts, device)
            total_blocks += 1
            # greedy accept walk
            accepted = 0
            for i in range(k):
                per_pos_total[i] += 1
                if drafts[i] == vargmax[i]:
                    per_pos_accept[i] += 1
                    if i == accepted:  # still contiguous from start
                        accepted += 1
                else:
                    pass
            # contiguous accept length (stop at first mismatch)
            contig = 0
            for i in range(k):
                if drafts[i] == vargmax[i]:
                    contig += 1
                else:
                    break
            total_accepted += contig
            if drafts[0] == vargmax[0]:
                first_tok_agree += 1
        if (si + 1) % 5 == 0:
            avg = total_accepted / max(1, total_blocks)
            print(f"[{si+1}/{args.num_samples}] avg_accept={avg:.3f} "
                  f"pos0_agree={first_tok_agree/max(1,total_blocks):.3f}", flush=True)

    avg_accept = total_accepted / max(1, total_blocks)
    result = {
        "drafter": args.drafter,
        "verifier": args.verifier,
        "num_samples": args.num_samples,
        "block_k": k,
        "prefix_len": args.prefix_len,
        "gen_len": args.gen_len,
        "total_blocks": total_blocks,
        "avg_accepted_per_block": avg_accept,
        "acceptance_estimated_speedup": 1.0 + avg_accept,
        "position0_agreement": first_tok_agree / max(1, total_blocks),
        "per_position_accept_rate": [
            per_pos_accept[i] / max(1, per_pos_total[i]) for i in range(k)
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== BASE TWO-MODEL ACCEPTANCE (no RELAY, no training) ===")
    print(f"drafter 0.4B -> verifier 2.9B")
    print(f"position-0 agreement (per-token argmax match): {result['position0_agreement']:.3f}")
    print(f"per-position accept: " +
          " ".join(f"p{i}={r:.3f}" for i, r in enumerate(result['per_position_accept_rate'])))
    print(f"avg accepted / block (k={k}): {avg_accept:.3f}")
    print(f"acceptance-estimated speedup: ~{1+avg_accept:.2f}x  (NOT wall-clock)")
    print(f"\nsaved: {args.out}")

    p0 = result["position0_agreement"]
    if p0 < 0.20:
        print(f"\n[DECISION GATE] pos-0 agreement {p0:.3f} < 0.20 -> base gap too large; "
              "need distillation/alignment BEFORE training a joint-scratch drafter.")
    elif p0 < 0.40:
        print(f"\n[DECISION GATE] pos-0 agreement {p0:.3f} in [0.20,0.40) -> marginal; "
              "RELAY plan-alignment must lift this to be worthwhile.")
    else:
        print(f"\n[DECISION GATE] pos-0 agreement {p0:.3f} >= 0.40 -> base gap workable; "
              "training a plan-aligned 0.4B drafter is justified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", default="/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B-world")
    ap.add_argument("--verifier", default="/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--out", default="outputs_eval/probe_dual_model_accept_04b_2p9b.json")
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--prefix_len", type=int, default=128)
    ap.add_argument("--gen_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
