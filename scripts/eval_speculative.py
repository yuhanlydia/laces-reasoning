"""Speculative decoding eval: Medusa draft + RWKV verify on fixed test set.

Measures acceptance rate and estimated speedup. Uses fixed OWT samples
(not random training batches) for reliable comparison.
"""

import argparse, glob, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as C
from eval.sample_prefix_suffix_cfg import encode_prefix
from scripts.train_medusa import MedusaHeads, get_hidden_states


@torch.no_grad()
def run(ckpt_dir, medusa_path, device, num_samples, num_heads, use_state_injection):
    model, tokenizer, dtype, pad_id = C.build_model(ckpt_dir, device)
    model.eval()

    ckpt = torch.load(medusa_path, map_location=device, weights_only=False)
    medusa = MedusaHeads(2560, 65536, num_heads).to(device, dtype)
    medusa.load_state_dict(ckpt["medusa_state"])
    medusa.eval()

    files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))
    np.random.seed(42)
    test_files = [files[i] for i in np.random.choice(len(files), num_samples, replace=False)]

    per_head_accept = [0] * num_heads
    total_rounds = 0

    for si, f in enumerate(test_files):
        d = np.load(f)
        ids = torch.tensor([d["input_ids"][:256]], device=device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)

        h = get_hidden_states(model, ids, am, use_state_injection)
        h = h.to(dtype)

        for t in range(128, 250, num_heads + 1):
            prefix_ids = ids[:, :t+1]
            prefix_am = am[:, :t+1]
            prefix_out = model.rwkv_model(input_ids=prefix_ids,
                                          attention_mask=prefix_am.bool(),
                                          use_cache=True, return_dict=True)
            prefix_state = prefix_out.past_key_values

            h_t = h[:, t:t+1, :]
            head_logits = medusa(h_t)
            draft_tokens = [head_logits[k].argmax(-1)[0, 0].item() for k in range(num_heads)]

            verify_ids = torch.tensor([draft_tokens], device=device, dtype=torch.long)
            verify_am = torch.ones_like(verify_ids, dtype=torch.float32)
            verify_out = model.rwkv_model(input_ids=verify_ids,
                                          attention_mask=verify_am.bool(),
                                          past_key_values=prefix_state,
                                          use_cache=False, return_dict=True)
            target_logits = verify_out.logits[0]

            total_rounds += 1
            target_tok_0 = prefix_out.logits[0, -1].argmax().item()
            if draft_tokens[0] == target_tok_0:
                per_head_accept[0] += 1
                for k in range(1, num_heads):
                    target_tok = target_logits[k - 1].argmax().item()
                    if draft_tokens[k] == target_tok:
                        per_head_accept[k] += 1
                    else:
                        break
            else:
                pass

        if (si + 1) % 5 == 0:
            avg = sum(per_head_accept) / max(1, total_rounds)
            print(f"[{si+1}/{num_samples}] avg_accept={avg:.2f} rounds={total_rounds}", flush=True)

    avg_accept = sum(per_head_accept) / max(1, total_rounds)
    speedup_est = 1 + avg_accept
    per_head_rate = [per_head_accept[k] / max(1, total_rounds) for k in range(num_heads)]
    print(f"\n=== RESULTS ({'state-primed' if use_state_injection else 'raw'}) ===", flush=True)
    print(f"per-head accept: {' '.join(f'h{k}={r:.3f}' for k, r in enumerate(per_head_rate))}", flush=True)
    print(f"avg accepted/round: {avg_accept:.2f}", flush=True)
    print(f"est speedup: ~{speedup_est:.2f}x", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--medusa_path", required=True)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--state_injection", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run(a.ckpt_dir, a.medusa_path, a.device, a.num_samples, a.num_heads, a.state_injection)


if __name__ == "__main__":
    main()
