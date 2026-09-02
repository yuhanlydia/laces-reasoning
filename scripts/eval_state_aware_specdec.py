"""State-aware chunk-level speculative decoding for RWKV.

Core innovation: RWKV state injection creates periodic freshness — chunk start
has high hidden-state quality, chunk end has decayed. We dynamically adjust
draft block size based on position-in-chunk, exploiting a signal unique to RNN.

Measures acceptance rate at each position within a chunk to validate the
hypothesis that chunk-start positions have higher draft acceptance.
Then runs full state-aware speculative decoding with adaptive block size.
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

CHUNK_SIZE = 32


def get_adaptive_block(pos_in_chunk, max_heads=8):
    freshness = 1.0 - pos_in_chunk / CHUNK_SIZE
    if freshness > 0.75:
        return min(max_heads, 8)
    elif freshness > 0.50:
        return min(max_heads, 4)
    elif freshness > 0.25:
        return min(max_heads, 2)
    return 1


@torch.no_grad()
def measure_position_acceptance(model, medusa, ids, am, device, dtype, num_heads):
    h = get_hidden_states(model, ids, am).to(dtype)
    B, T, D = h.shape
    per_pos_accept = [[] for _ in range(num_heads)]
    positions_recorded = []

    for t in range(128, min(T - num_heads - 2, 250)):
        prefix_ids = ids[:, :t + 1]
        prefix_am = am[:, :t + 1]
        prefix_out = model.rwkv_model(
            input_ids=prefix_ids, attention_mask=prefix_am.bool(),
            use_cache=True, return_dict=True)
        prefix_state = prefix_out.past_key_values
        h_t = h[:, t:t + 1, :]

        head_logits = medusa(h_t)
        draft_tokens = [head_logits[k].argmax(-1)[0, 0].item() for k in range(num_heads)]

        verify_ids = torch.tensor([draft_tokens], device=device, dtype=torch.long)
        verify_am = torch.ones_like(verify_ids, dtype=torch.float32)
        verify_out = model.rwkv_model(
            input_ids=verify_ids, attention_mask=verify_am.bool(),
            past_key_values=prefix_state, use_cache=False, return_dict=True)
        target_logits = verify_out.logits[0]

        target_tok_0 = prefix_out.logits[0, -1].argmax().item()
        positions_recorded.append(t % CHUNK_SIZE)

        if draft_tokens[0] == target_tok_0:
            per_pos_accept[0].append(t % CHUNK_SIZE)
            for k in range(1, num_heads):
                tk = target_logits[k - 1].argmax().item()
                if draft_tokens[k] == tk:
                    per_pos_accept[k].append(t % CHUNK_SIZE)
                else:
                    break

    return per_pos_accept, positions_recorded


@torch.no_grad()
def run_adaptive_specdec(model, medusa, ids, am, device, dtype, num_heads, max_new):
    h = get_hidden_states(model, ids, am).to(dtype)
    B, T, D = h.shape
    prompt_len = min(128, T - max_new - num_heads - 2)
    generated = ids[0, :prompt_len].tolist()

    total_tokens = 0
    total_rounds = 0
    total_draft_calls = 0
    total_accepted = 0

    while total_tokens < max_new:
        current_len = prompt_len + total_tokens
        pos_in_chunk = current_len % CHUNK_SIZE
        block = get_adaptive_block(pos_in_chunk, max_heads=num_heads)

        prefix_ids = ids[:, :current_len + 1]
        prefix_am = am[:, :current_len + 1]
        prefix_out = model.rwkv_model(
            input_ids=prefix_ids, attention_mask=prefix_am.bool(),
            use_cache=True, return_dict=True)
        prefix_state = prefix_out.past_key_values

        h_t = h[:, current_len:current_len + 1, :]
        head_logits = medusa(h_t)
        draft_tokens = [head_logits[k].argmax(-1)[0, 0].item() for k in range(block)]

        verify_ids = torch.tensor([draft_tokens], device=device, dtype=torch.long)
        verify_am = torch.ones_like(verify_ids, dtype=torch.float32)
        verify_out = model.rwkv_model(
            input_ids=verify_ids, attention_mask=verify_am.bool(),
            past_key_values=prefix_state, use_cache=False, return_dict=True)
        target_logits = verify_out.logits[0]

        total_rounds += 1
        total_draft_calls += block

        target_tok_0 = prefix_out.logits[0, -1].argmax().item()
        accepted = 0
        if draft_tokens[0] == target_tok_0:
            accepted += 1
            generated.append(draft_tokens[0])
            for k in range(1, block):
                tk = target_logits[k - 1].argmax().item()
                if draft_tokens[k] == tk:
                    accepted += 1
                    generated.append(draft_tokens[k])
                else:
                    generated.append(tk)
                    break
            else:
                pass
        else:
            generated.append(target_tok_0)

        total_accepted += accepted
        total_tokens += accepted + 1

    return total_tokens, total_rounds, total_accepted, total_draft_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--medusa_path", required=True)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--state_injection", action="store_true")
    ap.add_argument("--mode", choices=["measure", "adaptive", "both"], default="both")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    model, tokenizer, dtype, pad_id = C.build_model(a.ckpt_dir, a.device)
    model.eval()
    ckpt = torch.load(a.medusa_path, map_location=a.device, weights_only=False)
    medusa = MedusaHeads(2560, 65536, a.num_heads).to(a.device, dtype)
    medusa.load_state_dict(ckpt["medusa_state"])
    medusa.eval()

    files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))
    np.random.seed(42)
    test_files = [files[i] for i in np.random.choice(len(files), a.num_samples, replace=False)]

    if a.mode in ("measure", "both"):
        print("=== PHASE 1: Position-in-chunk acceptance measurement ===", flush=True)
        all_pos = []
        all_h0 = []
        for si, f in enumerate(test_files[:min(10, a.num_samples)]):
            d = np.load(f)
            ids = torch.tensor([d["input_ids"][:256]], device=a.device, dtype=torch.long)
            am = torch.ones_like(ids, dtype=torch.float32)
            h0_pos, positions = measure_position_acceptance(
                model, medusa, ids, am, a.device, dtype, a.num_heads)
            all_h0.extend(h0_pos[0])
            if (si + 1) % 5 == 0:
                print(f"  [{si+1}/10] measured", flush=True)

        bins = [(0, 8, "0-7"), (8, 16, "8-15"), (16, 24, "16-23"), (24, 32, "24-31")]
        print("\n--- h0 acceptance by position-in-chunk ---", flush=True)
        for lo, hi, label in bins:
            count = sum(1 for p in all_h0 if lo <= p < hi)
            total = sum(1 for p in positions if lo <= p < hi)
            rate = count / max(1, total)
            bar = "#" * int(rate * 40)
            print(f"  pos {label:>6s}: {rate:.3f} ({count}/{total}) {bar}", flush=True)

    if a.mode in ("adaptive", "both"):
        print("\n=== PHASE 2: State-aware adaptive speculative decoding ===", flush=True)
        tot_tokens = tot_rounds = tot_accepted = tot_drafts = 0
        for si, f in enumerate(test_files):
            d = np.load(f)
            ids = torch.tensor([d["input_ids"][:256]], device=a.device, dtype=torch.long)
            am = torch.ones_like(ids, dtype=torch.float32)
            nt, nr, na, nd = run_adaptive_specdec(
                model, medusa, ids, am, a.device, dtype, a.num_heads, a.max_new)
            tot_tokens += nt; tot_rounds += nr; tot_accepted += na; tot_drafts += nd
            if (si + 1) % 5 == 0:
                avg_a = tot_accepted / max(1, tot_rounds)
                print(f"  [{si+1}/{a.num_samples}] avg_accept={avg_a:.2f} "
                      f"rounds={tot_rounds} tokens={tot_tokens}", flush=True)

        avg_accept = tot_accepted / max(1, tot_rounds)
        avg_draft = tot_drafts / max(1, tot_rounds)
        speedup_fixed = (tot_accepted + tot_rounds) / tot_rounds
        speedup_adaptive = tot_tokens / (tot_rounds * 2)
        print(f"\n=== RESULTS (state-aware adaptive) ===", flush=True)
        print(f"state_injection: {a.state_injection}", flush=True)
        print(f"total tokens: {tot_tokens}", flush=True)
        print(f"total rounds: {tot_rounds}", flush=True)
        print(f"avg accepted/round: {avg_accept:.2f}", flush=True)
        print(f"avg draft block used: {avg_draft:.1f} (adaptive, max={a.num_heads})", flush=True)
        print(f"speedup vs pure autoregressive: ~{tot_tokens / tot_rounds:.2f}x", flush=True)
        print(f"(compare: fixed Medusa-4 was 1.54x, state-primed was 1.59x)", flush=True)


if __name__ == "__main__":
    main()
