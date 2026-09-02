"""Confidence-gated speculative decoding.

From Day 1 dashboard: accepted h0 conf=0.48, rejected conf=0.12.
Threshold ~0.25 separates them well.

Strategy:
1. If neural h0 confidence > threshold: use neural draft
2. Else: use n-gram draft (fallback)
3. Else: skip speculation (no draft, direct AR)

This raises CONDITIONAL acceptance by filtering low-confidence neural predictions.
Combined with state injection + n-gram gate, targets 2x.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path: sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as C
from scripts.train_medusa import MedusaHeads, get_hidden_states
from scripts.eval_combined_specdec import ngram_lookup


@torch.no_grad()
def run(ckpt_dir, medusa_path, device, num_samples, num_heads,
        use_state_injection, thresholds):
    model, tok, dtype, pad = C.build_model(ckpt_dir, device)
    model.eval()
    ckpt = torch.load(medusa_path, map_location=device, weights_only=False)
    medusa = MedusaHeads(2560, 65536, num_heads).to(device, dtype)
    medusa.load_state_dict(ckpt["medusa_state"])
    medusa.eval()

    files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))
    np.random.seed(42)
    test_files = [files[i] for i in np.random.choice(len(files), num_samples, replace=False)]

    for threshold in thresholds:
        h0_accept = 0
        h1_accept = 0
        spec_rounds = 0
        skip_rounds = 0
        total = 0
        ngram_used = 0
        ngram_accept = 0

        for si, f in enumerate(test_files):
            d = np.load(f)
            ids = torch.tensor([d["input_ids"][:256]], device=device, dtype=torch.long)
            am = torch.ones_like(ids, dtype=torch.float32)
            token_list = ids[0].tolist()
            h = get_hidden_states(model, ids, am, use_state_injection=use_state_injection).to(dtype)

            for t in range(128, 250, num_heads + 1):
                prefix_ids = ids[:, :t + 1]
                prefix_am = am[:, :t + 1]
                prefix_out = model.rwkv_model(
                    input_ids=prefix_ids, attention_mask=prefix_am.bool(),
                    use_cache=True, return_dict=True)
                prefix_state = prefix_out.past_key_values
                target_tok_0 = prefix_out.logits[0, -1].argmax().item()

                h_t = h[:, t:t + 1, :]
                head_logits = medusa(h_t)
                h0_probs = torch.softmax(head_logits[0][0, 0].float(), dim=-1)
                neural_tok = h0_probs.argmax().item()
                neural_conf = h0_probs[neural_tok].item()

                total += 1

                if neural_conf >= threshold:
                    draft_tok = neural_tok
                    spec_rounds += 1
                    ng_tok = -1
                else:
                    ng_tok = ngram_lookup(token_list, t)
                    if ng_tok >= 0:
                        draft_tok = ng_tok
                        ngram_used += 1
                        spec_rounds += 1
                    else:
                        skip_rounds += 1
                        continue

                accepted_0 = (draft_tok == target_tok_0)
                if accepted_0:
                    h0_accept += 1
                    if draft_tok == ng_tok and ng_tok >= 0:
                        ngram_accept += 1
                    if num_heads > 1:
                        draft_1 = head_logits[1][0, 0].argmax().item()
                        verify_ids = torch.tensor([[draft_tok]], device=device, dtype=torch.long)
                        v_out = model.rwkv_model(
                            input_ids=verify_ids,
                            attention_mask=torch.ones_like(verify_ids, dtype=torch.float32).bool(),
                            past_key_values=prefix_state, use_cache=False, return_dict=True)
                        if draft_1 == v_out.logits[0, 0].argmax().item():
                            h1_accept += 1

            if (si + 1) % 5 == 0:
                print(f"  [thr={threshold}] [{si+1}/{num_samples}] "
                      f"h0={h0_accept}/{spec_rounds} skip={skip_rounds} "
                      f"ng={ngram_used}/{spec_rounds}", flush=True)

        cond_h0 = h0_accept / max(1, spec_rounds)
        avg_accept = (h0_accept + h1_accept) / max(1, total)
        effective_speedup = 1 + avg_accept
        spec_rate = spec_rounds / max(1, total)
        print(f"\n=== THRESHOLD={threshold} ===", flush=True)
        print(f"  spec rate: {spec_rate:.3f} ({spec_rounds}/{total} rounds speculated)", flush=True)
        print(f"  conditional h0 (when speculate): {cond_h0:.3f}", flush=True)
        print(f"  unconditional h0 (all rounds): {h0_accept/max(1,total):.3f}", flush=True)
        print(f"  ngram used: {ngram_used} ngram accepted: {ngram_accept}", flush=True)
        print(f"  avg accept/round: {avg_accept:.3f}", flush=True)
        print(f"  EFFECTIVE SPEEDUP: ~{effective_speedup:.2f}x", flush=True)
        print(f"  (skip rounds cost 1 forward but produce 1 token = no gain/loss)", flush=True)
        print(flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--medusa_path", default="outputs_relay/medusa-rwkv-2.9B-stateprimed/medusa_final.pt")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--state_injection", action="store_true")
    ap.add_argument("--thresholds", default="0,0.1,0.2,0.3,0.4")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run(a.ckpt_dir, a.medusa_path, a.device, a.num_samples, a.num_heads,
        a.state_injection, [float(x) for x in a.thresholds.split(",")])
