"""Tree speculation + dashboard for RWKV speculative decoding.

Key insight (智库): P(target in top-B0) >> P(argmax=target).
B0=4 may give 80%+ vs 43% for argmax.

Measures:
1. Coverage: P(target in top-B0) for B0={1,2,4,8,16}
2. Tree acceptance: accepted tokens per verification with top-B tree
3. Dashboard: h0 confidence vs acceptance, calibration, entropy buckets
"""

import argparse, glob, sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path: sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as C
from scripts.train_medusa import MedusaHeads, get_hidden_states


@torch.no_grad()
def run(ckpt_dir, medusa_path, device, num_samples, num_heads, use_state_injection):
    model, tok, dtype, pad = C.build_model(ckpt_dir, device)
    model.eval()
    ckpt = torch.load(medusa_path, map_location=device, weights_only=False)
    medusa = MedusaHeads(2560, 65536, num_heads).to(device, dtype)
    medusa.load_state_dict(ckpt["medusa_state"])
    medusa.eval()

    files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))
    np.random.seed(42)
    test_files = [files[i] for i in np.random.choice(len(files), num_samples, replace=False)]

    coverage_counts = {b: 0 for b in [1, 2, 4, 8, 16]}
    coverage_total = 0
    h0_conf_accepted = []
    h0_conf_rejected = []
    tree_accept_lengths = []
    total_tree_rounds = 0

    B0 = 4
    B1 = 2

    for si, f in enumerate(test_files):
        d = np.load(f)
        ids = torch.tensor([d["input_ids"][:256]], device=device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)
        h = get_hidden_states(model, ids, am, use_state_injection).to(dtype)

        for t in range(128, 250, 3):
            prefix_ids = ids[:, :t + 1]
            prefix_am = am[:, :t + 1]
            prefix_out = model.rwkv_model(input_ids=prefix_ids,
                                          attention_mask=prefix_am.bool(),
                                          use_cache=True, return_dict=True)
            prefix_state = prefix_out.past_key_values
            target_token = prefix_out.logits[0, -1].argmax().item()

            h_t = h[:, t:t + 1, :]
            head_logits = medusa(h_t)

            h0_probs = torch.softmax(head_logits[0][0, 0].float(), dim=-1)
            h0_topk = h0_probs.topk(16).indices.tolist()
            h0_conf = h0_probs[h0_probs.topk(16).indices].tolist()

            for b in [1, 2, 4, 8, 16]:
                if target_token in h0_topk[:b]:
                    coverage_counts[b] += 1
            coverage_total += 1

            if target_token == h0_topk[0]:
                h0_conf_accepted.append(float(h0_conf[0]))
            else:
                h0_conf_rejected.append(float(h0_conf[0]))

            h0_candidates = h0_topk[:B0]
            tree_paths = []
            for h0_tok in h0_candidates:
                h0_emb = medusa.heads[0][0](h_t)
                tree_paths.append([h0_tok])
                if num_heads > 1:
                    prev_emb = medusa_embed_if_markov(medusa, h0_tok, device, dtype)
                    inp2 = torch.cat([h_t, prev_emb], dim=-1) if prev_emb is not None else h_t
                    h1_logits = head_logits[1][0, 0]
                    h1_topk = h1_logits.topk(B1).indices.tolist()
                    for h1_tok in h1_topk:
                        tree_paths.append([h0_tok, h1_tok])

            accepted_len = 0
            for pi, path in enumerate(tree_paths):
                path_tensor = torch.tensor([path], device=device, dtype=torch.long)
                v_out = model.rwkv_model(input_ids=path_tensor,
                                         past_key_values=prefix_state,
                                         use_cache=False, return_dict=True)
                v_logits = v_out.logits[0]
                p_out = prefix_out.logits[0, -1]

                match_len = 0
                if path[0] == p_out.argmax().item():
                    match_len = 1
                    if len(path) > 1:
                        if path[1] == v_logits[0].argmax().item():
                            match_len = 2
                if match_len > accepted_len:
                    accepted_len = match_len

            tree_accept_lengths.append(accepted_len)
            total_tree_rounds += 1

        if (si + 1) % 5 == 0:
            print(f"[{si+1}/{num_samples}]", flush=True)

    print(f"\n=== DASHBOARD ===", flush=True)
    print(f"\n--- Coverage: P(target in top-B0) ---", flush=True)
    for b in [1, 2, 4, 8, 16]:
        rate = coverage_counts[b] / max(1, coverage_total)
        bar = "#" * int(rate * 40)
        print(f"  B0={b:2d}: {rate:.3f} {bar}", flush=True)

    print(f"\n--- h0 Confidence: accepted vs rejected ---", flush=True)
    if h0_conf_accepted:
        print(f"  accepted mean_conf={np.mean(h0_conf_accepted):.3f} std={np.std(h0_conf_accepted):.3f}", flush=True)
    if h0_conf_rejected:
        print(f"  rejected mean_conf={np.mean(h0_conf_rejected):.3f} std={np.std(h0_conf_rejected):.3f}", flush=True)

    print(f"\n--- Tree Acceptance (B0={B0}, B1={B1}) ---", flush=True)
    avg_accept = np.mean(tree_accept_lengths)
    print(f"  avg accepted tokens/round: {avg_accept:.2f}", flush=True)
    print(f"  distribution: 0={tree_accept_lengths.count(0)} 1={tree_accept_lengths.count(1)} 2={tree_accept_lengths.count(2)}", flush=True)
    est_speedup = (avg_accept + 1) / 2
    print(f"  est speedup: ~{est_speedup:.2f}x", flush=True)
    print(f"  (compare: linear Medusa h0-only = 1.54x)", flush=True)


def medusa_embed_if_markov(medusa, token, device, dtype):
    if hasattr(medusa, 'embed'):
        return medusa.embed(torch.tensor([token], device=device)).unsqueeze(0).unsqueeze(0).to(dtype)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--medusa_path", required=True)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--state_injection", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run(a.ckpt_dir, a.medusa_path, a.device, a.num_samples, a.num_heads, a.state_injection)
