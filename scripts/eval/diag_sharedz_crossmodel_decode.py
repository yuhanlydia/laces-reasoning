"""Diagnostic: inject the SAME single-z latent into two different-size frozen
RWKV models and greedily decode from each. Shows directly whether one shared Z
produces divergent behavior across model sizes -- the mechanistic root of why
cross-model speculative decoding via shared-Z injection loses to bare.

For each prompt:
  1. Encode prefix -> z_prefix via the verifier (13.3B) S0.
  2. Inject predict_states(z) into BOTH models' caches (single-z, one global write).
  3. Greedy-decode N tokens from each; print side by side.
  4. Report token-level agreement between the two continuations (the quantity
     speculative acceptance depends on).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/diag_sharedz_crossmodel_decode.py \
    --model_a outputs_relay/test-v6-0.4B-s1-prefix-suffix/step_00050000 \
    --model_b outputs_relay/traj32x16-13.3B-singlez-bridge-s1-merged/step_00000000 \
    --n_new 48 --num_prompts 4
"""
import argparse, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag


@torch.no_grad()
def encode_prefix_z(model, ids, am):
    out = model.rwkv_model(
        input_ids=ids, attention_mask=am.bool(),
        output_hidden_states=True, use_cache=True, return_dict=True,
    )
    pooled = model._pool_hidden(out.hidden_states[-1], am)
    z, _ = model._encode_pooled(pooled)
    return z


@torch.no_grad()
def greedy_from_injected(model, tok, ids, z_global, n_new, device):
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
    ap.add_argument("--model_a", default="outputs_relay/test-v6-0.4B-s1-prefix-suffix/step_00050000")
    ap.add_argument("--model_b", default="outputs_relay/traj32x16-13.3B-singlez-bridge-s1-merged/step_00000000")
    ap.add_argument("--n_new", type=int, default=48)
    ap.add_argument("--num_prompts", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"loading model_a (small): {args.model_a}", flush=True)
    ma, tok, _, _ = Diag.build_model(args.model_a, args.device)
    ma.eval()
    print(f"loading model_b (large): {args.model_b}", flush=True)
    mb, tokb, _, _ = Diag.build_model(args.model_b, args.device)
    mb.eval()

    prompts = Diag.PASSAGES[: args.num_prompts]
    agree_total = n_total = 0
    for pi, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        am = torch.ones_like(ids)
        # shared Z from the LARGE model (the source in cross-model speculation)
        z = encode_prefix_z(mb, ids, am)

        toks_a = greedy_from_injected(ma, tok, ids, z, args.n_new, args.device)
        toks_b = greedy_from_injected(mb, tokb, ids, z, args.n_new, args.device)

        agree = sum(1 for a, b in zip(toks_a, toks_b) if a == b)
        agree_total += agree
        n_total += len(toks_a)

        print(f"\n{'='*70}\nPROMPT {pi}: {prompt[:60]}...\n{'='*70}")
        print(f"[0.4B  | same Z]: {tok.decode(toks_a)!r}")
        print(f"[13.3B | same Z]: {tokb.decode(toks_b)!r}")
        print(f"token agreement: {agree}/{len(toks_a)} = {100.0*agree/max(1,len(toks_a)):.1f}%")

    print(f"\n{'='*70}\nOVERALL token agreement (same Z, 0.4B vs 13.3B): "
          f"{agree_total}/{n_total} = {100.0*agree_total/max(1,n_total):.1f}%")
    print("This is the quantity speculative acceptance depends on; low agreement "
          "= the same Z drives the two sizes to different tokens = injection loses to bare.")


if __name__ == "__main__":
    main()
