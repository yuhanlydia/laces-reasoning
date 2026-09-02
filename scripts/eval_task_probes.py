#!/usr/bin/env python3
"""Task probes: format-following + short-arithmetic to test 13.3B plan → 0.4B ability transfer."""
import sys, json, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.eval.diag_loop1_common import build_model

device = "cuda"; dtype = torch.bfloat16

# Sampling config (project-standard non-degenerate params; overridable via env)
import os
GREEDY = os.environ.get("GREEDY", "0") == "1"
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.5"))
TOP_K = int(os.environ.get("TOP_K", "10"))
TOP_P = float(os.environ.get("TOP_P", "0.75"))
REP_PEN = float(os.environ.get("REP_PEN", "1.3"))
SEED = int(os.environ.get("SEED", "42"))
torch.manual_seed(SEED)

def _sample_next(logits, prev_tokens):
    if GREEDY:
        return int(logits.argmax().item())
    logits = logits.float().clone()
    if REP_PEN != 1.0 and prev_tokens:
        uniq = torch.tensor(list(set(prev_tokens)), device=logits.device, dtype=torch.long)
        vals = logits[uniq]
        vals = torch.where(vals > 0, vals / REP_PEN, vals * REP_PEN)
        logits[uniq] = vals
    logits = logits / max(TEMPERATURE, 1e-6)
    if TOP_K > 0:
        kth = torch.topk(logits, min(TOP_K, logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if TOP_P < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sp, dim=-1)
        mask = cdf - sp > TOP_P
        sp[mask] = 0.0
        sp = sp / sp.sum()
        idx = torch.multinomial(sp, 1)
        return int(si[idx].item())
    return int(torch.multinomial(probs, 1).item())

# Load models
drafter, _, _, _ = build_model(
    "outputs_relay/drafter-04b-traj32x16-joint-scratch-coadapt/step_00006000", device
)
s1 = torch.load("outputs_relay/cross-source-0.4B-s1-denoised-v2/cross_s1_final.pt", map_location=device)
drafter.load_state_dict(s1["trainable_state"], strict=False); drafter.eval()
rwkv_d = drafter.rwkv_model

verifier, _, _, _ = build_model(
    "outputs_relay/owt512-traj32x16-13.3B-basis32-prefix-suffix-blend0p5-s2-rwkv-rf/step_00150000", device
)
verifier.eval(); rwkv_v = verifier.rwkv_model

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B-world",
    trust_remote_code=True, local_files_only=True
)

# Task definitions
FORMAT_TASKS = [
    ("Repeat after me: hello world", "hello world"),
    ("Repeat after me: the cat sat on the mat", "the cat sat on the mat"),
    ("Repeat after me: artificial intelligence", "artificial intelligence"),
    ("Say exactly: YES", "YES"),
    ("Say exactly: NO THANKS", "NO THANKS"),
    ("Output only the word: COMPLETE", "COMPLETE"),
]

ARITH_TASKS = [
    ("2 + 3 =", "5"),
    ("10 - 4 =", "6"),
    ("3 * 4 =", "12"),
    ("20 / 5 =", "4"),
    ("7 + 8 =", "15"),
    ("100 - 37 =", "63"),
]

@torch.no_grad()
def generate(model_rwkv, ids, max_new=32):
    out = model_rwkv(input_ids=ids, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    nxt = _sample_next(out.logits[0, -1], [])
    tokens = [nxt]
    cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    for _ in range(max_new - 1):
        out = model_rwkv(input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        nxt = _sample_next(out.logits[0, -1], tokens)
        tokens.append(nxt)
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    return tokenizer.decode(tokens)

@torch.no_grad()
def generate_with_plan(model_rwkv, relay, ids, max_new=32):
    am = torch.ones_like(ids, dtype=torch.float32)
    # Get Z from 13.3B
    out_v = rwkv_v(input_ids=ids, attention_mask=am.bool(), output_hidden_states=True, use_cache=True, return_dict=True)
    pooled = verifier._pool_hidden(out_v.hidden_states[-1], am)
    z, _ = verifier._encode_pooled(pooled)
    # Inject
    states = relay.predict_states(z)
    out_d = model_rwkv(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    pkv = relay.inject_into_cache(out_d.past_key_values, states)
    out = model_rwkv(input_ids=ids, attention_mask=am.bool(), past_key_values=pkv, use_cache=True, return_dict=True)
    nxt = _sample_next(out.logits[0, -1], [])
    tokens = [nxt]
    cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    for _ in range(max_new - 1):
        out = model_rwkv(input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        nxt = _sample_next(out.logits[0, -1], tokens)
        tokens.append(nxt)
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    return tokenizer.decode(tokens)

results = {"format": [], "arithmetic": []}

print("=== FORMAT-FOLLOWING ===")
for prompt, expected in FORMAT_TASKS:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    raw_out = generate(rwkv_d, ids)
    plan_out = generate_with_plan(rwkv_d, drafter, ids)
    ref_out = generate(rwkv_v, ids)

    raw_ok = expected.lower() in raw_out.lower()
    plan_ok = expected.lower() in plan_out.lower()
    ref_ok = expected.lower() in ref_out.lower()

    results["format"].append({"prompt": prompt, "expected": expected,
        "raw_ok": raw_ok, "plan_ok": plan_ok, "ref_ok": ref_ok,
        "raw": raw_out[:100], "plan": plan_out[:100], "ref": ref_out[:100]})
    print(f"  {prompt[:40]:<40s} raw={'✓' if raw_ok else '✗'} plan={'✓' if plan_ok else '✗'} ref={'✓' if ref_ok else '✗'}")

print("\n=== SHORT ARITHMETIC ===")
for prompt, expected in ARITH_TASKS:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    raw_out = generate(rwkv_d, ids)
    plan_out = generate_with_plan(rwkv_d, drafter, ids)
    ref_out = generate(rwkv_v, ids)

    raw_ok = expected in raw_out[:50]
    plan_ok = expected in plan_out[:50]
    ref_ok = expected in ref_out[:50]

    results["arithmetic"].append({"prompt": prompt, "expected": expected,
        "raw_ok": raw_ok, "plan_ok": plan_ok, "ref_ok": ref_ok,
        "raw": raw_out[:100], "plan": plan_out[:100], "ref": ref_out[:100]})
    print(f"  {prompt:<15s} raw={'✓' if raw_ok else '✗'} plan={'✓' if plan_ok else '✗'} ref={'✓' if ref_ok else '✗'}")

# Summary
fmt_raw = sum(1 for r in results["format"] if r["raw_ok"])
fmt_plan = sum(1 for r in results["format"] if r["plan_ok"])
fmt_ref = sum(1 for r in results["format"] if r["ref_ok"])
ari_raw = sum(1 for r in results["arithmetic"] if r["raw_ok"])
ari_plan = sum(1 for r in results["arithmetic"] if r["plan_ok"])
ari_ref = sum(1 for r in results["arithmetic"] if r["ref_ok"])

print(f"\n=== SUMMARY ===")
print(f"Format:   raw={fmt_raw}/{len(FORMAT_TASKS)}  plan={fmt_plan}/{len(FORMAT_TASKS)}  ref={fmt_ref}/{len(FORMAT_TASKS)}")
print(f"Arithmetic: raw={ari_raw}/{len(ARITH_TASKS)}  plan={ari_plan}/{len(ARITH_TASKS)}  ref={ari_ref}/{len(ARITH_TASKS)}")

with open("outputs_eval/task_probes.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: outputs_eval/task_probes.json")
