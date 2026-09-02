#!/usr/bin/env python3
"""Test: Does 13.3B's latent plan help 0.4B generate better content?

Compares bare 0.4B vs plan-injected 0.4B on math/reasoning prompts.
"""
import sys, json
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.eval.diag_loop1_common import build_model
from transformers import AutoTokenizer

device = "cuda"; dtype = torch.bfloat16

# Models
drafter, tok_d, _, _ = build_model(
    "outputs_relay/drafter-04b-traj32x16-joint-scratch-coadapt/step_00006000", device
)
s1 = torch.load(
    "outputs_relay/cross-source-0.4B-s1-denoised-v2/cross_s1_final.pt", map_location=device
)
drafter.load_state_dict(s1["trainable_state"], strict=False); drafter.eval()
rwkv_d = drafter.rwkv_model

verifier, tok_v, _, _ = build_model(
    "outputs_relay/owt512-traj32x16-13.3B-basis32-prefix-suffix-blend0p5-s2-rwkv-rf/step_00150000", device
)
verifier.eval(); rwkv_v = verifier.rwkv_model

tokenizer = tok_v  # Both use same RWKV tokenizer

prompts = [
    "Solve step by step: if 2x + 5 = 15, then x =",
    "What is 15% of 200? Show your work.",
    "The history of artificial intelligence began in",
]

@torch.no_grad()
def generate_04b_bare(ids, max_new=64):
    out = rwkv_d(input_ids=ids, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    nxt = out.logits[0, -1].argmax().item()
    tokens = [nxt]
    cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    for _ in range(max_new - 1):
        out = rwkv_d(input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        nxt = out.logits[0, -1].argmax().item()
        tokens.append(nxt)
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    return tokens

@torch.no_grad()
def generate_04b_plan(ids, max_new=64):
    am = torch.ones_like(ids, dtype=torch.float32)
    # Get Z from 13.3B
    out_v = rwkv_v(input_ids=ids, attention_mask=am.bool(), output_hidden_states=True, use_cache=True, return_dict=True)
    pooled = verifier._pool_hidden(out_v.hidden_states[-1], am)
    z, _ = verifier._encode_pooled(pooled)
    # Inject into 0.4B
    states = drafter.predict_states(z)
    out_d = rwkv_d(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    pkv = drafter.inject_into_cache(out_d.past_key_values, states)
    # Generate
    out = rwkv_d(input_ids=ids, attention_mask=am.bool(), past_key_values=pkv, use_cache=True, return_dict=True)
    nxt = out.logits[0, -1].argmax().item()
    tokens = [nxt]
    cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    for _ in range(max_new - 1):
        out = rwkv_d(input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        nxt = out.logits[0, -1].argmax().item()
        tokens.append(nxt)
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    return tokens

@torch.no_grad()
def generate_13b(ids, max_new=64):
    out = rwkv_v(input_ids=ids, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    nxt = out.logits[0, -1].argmax().item()
    tokens = [nxt]
    cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    for _ in range(max_new - 1):
        out = rwkv_v(input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        nxt = out.logits[0, -1].argmax().item()
        tokens.append(nxt)
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    return tokens

for prompt in prompts:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*70}")
    
    t_04b = generate_04b_bare(ids)
    t_plan = generate_04b_plan(ids)
    t_13b = generate_13b(ids)
    
    print(f"\n[13.3B reference]")
    print(tokenizer.decode(t_13b))
    print(f"\n[0.4B bare]")
    print(tokenizer.decode(t_04b))
    print(f"\n[0.4B + 13.3B plan]")
    print(tokenizer.decode(t_plan))
