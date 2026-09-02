"""Cross-source decode diagnostic (CORRECT pairing).

Takes ONE shared 13.3B-S0 latent Z (from the dumped 13.3B latents), injects it
into (a) the 0.4B base with its trained cross-source S1 patched in, and (b) the
13.3B model with its own S1, then greedily decodes from each and compares.

This is the decoder-paper setup: a UNIFIED 13.3B latent read by per-model S1
adapters, not each model encoding its own latent. Answers: does the SAME shared
latent drive the drafter (0.4B) and verifier (13.3B) to the same tokens?

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/diag_crosssource_decode.py \
    --small_base outputs_relay/test-v6-0.4B-s1-prefix-suffix/step_00050000 \
    --small_s1 outputs_relay/cross-source-0.4B-s1-v2/cross_s1_final.pt \
    --large outputs_relay/traj32x16-13.3B-singlez-bridge-s1-merged/step_00000000 \
    --latent_dir preprocessed_data/owt_13b_s0_latents/train \
    --data_dir preprocessed_data/owt_rwkv_tokens/train \
    --n_new 48 --num_prompts 4
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag


@torch.no_grad()
def greedy_from_injected(model, ids, z_global, n_new, device):
    out = model.rwkv_model(input_ids=ids, use_cache=True, return_dict=True)
    states = model.predict_states(z_global)
    pkv = model.inject_into_cache(out.past_key_values, states)
    cur = ids[:, -1:]
    toks = []
    for _ in range(n_new):
        o = model.rwkv_model(input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        toks.append(nxt)
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small_base", default="outputs_relay/test-v6-0.4B-s1-prefix-suffix/step_00050000")
    ap.add_argument("--small_s1", default="outputs_relay/cross-source-0.4B-s1-v2/cross_s1_final.pt")
    ap.add_argument("--large", default="outputs_relay/traj32x16-13.3B-singlez-bridge-s1-merged/step_00000000")
    ap.add_argument("--latent_dir", default="preprocessed_data/owt_13b_s0_latents/train")
    ap.add_argument("--data_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--n_new", type=int, default=48)
    ap.add_argument("--num_prompts", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"loading small base (0.4B): {args.small_base}", flush=True)
    ms, tok, dtype, _ = Diag.build_model(args.small_base, args.device)
    ms.eval(); ms._prefix_suffix_s2 = True
    trained = torch.load(args.small_s1, map_location=args.device, weights_only=False)
    inc = ms.load_state_dict(trained["trainable_state"], strict=False)
    n_load = len(trained["trainable_state"]) - len(
        [k for k in trained["trainable_state"] if k in inc.missing_keys])
    print(f"  patched cross-S1: {n_load}/{len(trained['trainable_state'])} tensors, step={trained.get('step')}", flush=True)

    print(f"loading large (13.3B): {args.large}", flush=True)
    ml, tokl, _, _ = Diag.build_model(args.large, args.device)
    ml.eval()

    # match 13.3B latent .npy to token files
    latent_files = sorted(glob.glob(f"{args.latent_dir}/*.npy"))
    latent_files = [f for f in latent_files if "_tokens" not in f]
    pairs = []
    for lf in latent_files:
        stem = Path(lf).stem
        tf = Path(args.data_dir) / f"{stem}_tokens.npz"
        if tf.exists():
            pairs.append((lf, str(tf)))
        if len(pairs) >= args.num_prompts:
            break

    agree_total = n_total = 0
    for pi, (lf, tf) in enumerate(pairs):
        d = np.load(tf)
        ids = torch.tensor([d["input_ids"][:256]], device=args.device, dtype=torch.long)
        z = torch.tensor(np.load(lf), device=args.device, dtype=dtype).unsqueeze(0)
        z_global = z.mean(dim=1) if z.dim() == 3 else z  # [1,32]

        toks_s = greedy_from_injected(ms, ids, z_global, args.n_new, args.device)
        toks_l = greedy_from_injected(ml, ids, z_global, args.n_new, args.device)
        agree = sum(1 for a, b in zip(toks_s, toks_l) if a == b)
        agree_total += agree
        n_total += len(toks_s)

        print(f"\n{'='*70}\nSAMPLE {pi} (shared 13.3B latent)\n{'='*70}")
        print(f"[0.4B+cross-S1 | shared Z]: {tok.decode(toks_s)!r}")
        print(f"[13.3B+own-S1  | shared Z]: {tokl.decode(toks_l)!r}")
        print(f"token agreement: {agree}/{len(toks_s)} = {100.0*agree/max(1,len(toks_s)):.1f}%")

    print(f"\n{'='*70}\nOVERALL token agreement (shared 13.3B Z, 0.4B-cross-S1 vs 13.3B-own-S1): "
          f"{agree_total}/{n_total} = {100.0*agree_total/max(1,n_total):.1f}%")


if __name__ == "__main__":
    main()
