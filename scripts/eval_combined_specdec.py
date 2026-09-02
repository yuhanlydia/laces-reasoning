"""Combined n-gram gate + state injection speculative decoding.

Best combo from Day 1-5 experiments:
- State injection: h0 43%→48% (+4.4%)
- N-gram gate: h0 46.6%→59.4% (+13%)
- Combined: should reach ~60%+ → est 2x speedup

At each position:
1. Neural head proposes token (from state-injected hidden)
2. N-gram lookup proposes token (from context)
3. If neural top-1 == target: accept
4. Else if n-gram == target: accept n-gram token
5. Verify with chunk_rwkv7
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


def ngram_lookup(token_list, t, n_range=(4, 3, 2)):
    for n in n_range:
        if t + 1 >= n:
            key = tuple(token_list[t + 1 - n:t + 1])
            for j in range(len(token_list) - n):
                if tuple(token_list[j:j + n]) == key and j + n < len(token_list):
                    return token_list[j + n]
    return -1


@torch.no_grad()
def run(ckpt_dir, medusa_path, device, num_samples, num_heads,
        use_state_injection, use_ngram):
    model, tok, dtype, pad = C.build_model(ckpt_dir, device)
    model.eval()
    ckpt = torch.load(medusa_path, map_location=device, weights_only=False)
    medusa = MedusaHeads(2560, 65536, num_heads).to(device, dtype)
    medusa.load_state_dict(ckpt["medusa_state"])
    medusa.eval()

    files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))
    np.random.seed(42)
    test_files = [files[i] for i in np.random.choice(len(files), num_samples, replace=False)]

    configs = []
    if use_state_injection:
        configs.append(("state+neural", True, False))
        if use_ngram:
            configs.append(("state+neural+ngram", True, True))
    else:
        configs.append(("neural", False, False))
        if use_ngram:
            configs.append(("neural+ngram", False, True))
    if use_ngram:
        configs.append(("ngram_only", False, True))

    for config_name, do_state, do_ngram in configs:
        h0_accept = 0
        h1_accept = 0
        total = 0
        total_rounds = 0

        for si, f in enumerate(test_files):
            d = np.load(f)
            ids = torch.tensor([d["input_ids"][:256]], device=device, dtype=torch.long)
            am = torch.ones_like(ids, dtype=torch.float32)
            token_list = ids[0].tolist()
            h = get_hidden_states(model, ids, am, use_state_injection=do_state).to(dtype)

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
                neural_tok = head_logits[0][0, 0].argmax().item()

                if do_ngram:
                    ng_tok = ngram_lookup(token_list, t)
                else:
                    ng_tok = -1

                draft_tok = neural_tok
                source = "neural"
                if neural_tok != target_tok_0 and do_ngram and ng_tok >= 0:
                    draft_tok = ng_tok
                    source = "ngram"

                total += 1
                total_rounds += 1

                accepted_0 = (draft_tok == target_tok_0)
                if accepted_0:
                    h0_accept += 1
                    if num_heads > 1:
                        draft_1 = head_logits[1][0, 0].argmax().item()
                        verify_ids = torch.tensor([[draft_tok]], device=device, dtype=torch.long)
                        verify_am = torch.ones_like(verify_ids, dtype=torch.float32)
                        v_out = model.rwkv_model(
                            input_ids=verify_ids, attention_mask=verify_am.bool(),
                            past_key_values=prefix_state, use_cache=False, return_dict=True)
                        target_tok_1 = v_out.logits[0, 0].argmax().item()
                        if draft_1 == target_tok_1:
                            h1_accept += 1

            if (si + 1) % 5 == 0:
                print(f"  [{config_name}] [{si+1}/{num_samples}] "
                      f"h0={h0_accept}/{total} h1={h1_accept}/{total}", flush=True)

        avg_h0 = h0_accept / max(1, total)
        avg_h1 = h1_accept / max(1, total)
        avg_accept = avg_h0 + avg_h1
        speedup = (avg_accept + 1) / 2
        print(f"\n=== {config_name.upper()} ===", flush=True)
        print(f"  h0 accept: {avg_h0:.3f}", flush=True)
        print(f"  h1 accept: {avg_h1:.3f}", flush=True)
        print(f"  avg accept/round: {avg_accept:.2f}", flush=True)
        print(f"  est speedup: ~{speedup:.2f}x", flush=True)
        print(flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--medusa_path", default="outputs_relay/medusa-rwkv-2.9B-stateprimed/medusa_final.pt")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--state_injection", action="store_true")
    ap.add_argument("--ngram", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run(a.ckpt_dir, a.medusa_path, a.device, a.num_samples, a.num_heads,
        a.state_injection, a.ngram)
