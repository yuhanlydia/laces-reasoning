#!/usr/bin/env python3
"""Collect offline data for P0 (confidence gate) and P1 (behavior-alignment S1).

One pass through OWT data, loading 0.4B (cross-S1) + 13.3B simultaneously.

Output:
  P0: per-sample features + labels for gate classifier training
      {z_norm, raw_accept, inj_accept, top1_raw, top1_inj, top5_raw, top5_inj, ...}
  P1: verifier argmax tokens (used as labels for behavior-alignment S1 training)
      {verifier_argmax: [int], ...} per position
"""
import sys, glob, json, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.eval.diag_loop1_common import build_model

device = "cuda"; dtype = torch.bfloat16

# ── Load models ──
print("Loading 0.4B + cross-S1...", flush=True)
drafter, _, _, _ = build_model(
    "outputs_relay/drafter-04b-traj32x16-joint-scratch-coadapt/step_00006000", device
)
s1 = torch.load("outputs_relay/cross-source-0.4B-s1-denoised-v2/cross_s1_final.pt", map_location=device)
drafter.load_state_dict(s1["trainable_state"], strict=False); drafter.eval()
rwkv_d = drafter.rwkv_model

print("Loading 13.3B verifier...", flush=True)
verifier, _, _, _ = build_model(
    "outputs_relay/owt512-traj32x16-13.3B-basis32-prefix-suffix-blend0p5-s2-rwkv-rf/step_00150000", device
)
verifier.eval(); rwkv_v = verifier.rwkv_model

files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))[:200]
gate_data = []      # P0: per-sample features + labels
verifier_logits = []  # P1: verifier argmax tokens per position

t0 = time.time()
for fi, f in enumerate(files):
    d = np.load(f)
    ids = torch.tensor([d["input_ids"][:256]], device=device, dtype=torch.long)
    am = torch.ones_like(ids, dtype=torch.float32)

    # ── RAW (bare) ──
    out_d = rwkv_d(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    out_v = rwkv_v(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)

    # ── Shared Z from 13.3B S0 ──
    out_vh = rwkv_v(input_ids=ids, attention_mask=am.bool(), output_hidden_states=True, use_cache=True, return_dict=True)
    pooled = verifier._pool_hidden(out_vh.hidden_states[-1], am)
    z_prefix, _ = verifier._encode_pooled(pooled)

    # ── INJECTED ──
    s_d = drafter.predict_states(z_prefix)
    out_d_pre = rwkv_d(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    pkv_d = drafter.inject_into_cache(out_d_pre.past_key_values, s_d)
    out_d_inj = rwkv_d(input_ids=ids, attention_mask=am.bool(), past_key_values=pkv_d, use_cache=True, return_dict=True)

    # ── P0 Features ──
    z_norm = float(z_prefix.norm(dim=-1).item())

    # Top-k overlap over suffix [128:255]
    raw_t1, raw_t5, inj_t1, inj_t5 = [], [], [], []
    for pos in range(128, 255):
        v_raw = out_v.logits[0, pos].argmax().item()
        v_inj = out_v.logits[0, pos].argmax().item()
        d_raw_top5 = out_d.logits[0, pos].topk(5).indices.tolist()
        d_inj_top5 = out_d_inj.logits[0, pos].topk(5).indices.tolist()
        raw_t1.append(v_raw == out_d.logits[0, pos].argmax().item())
        raw_t5.append(v_raw in d_raw_top5)
        inj_t1.append(v_inj == out_d_inj.logits[0, pos].argmax().item())
        inj_t5.append(v_inj in d_inj_top5)

    raw_top1 = float(np.mean(raw_t1))
    inj_top1 = float(np.mean(inj_t1))
    raw_top5 = float(np.mean(raw_t5))
    inj_top5 = float(np.mean(inj_t5))

    # Simple accept/round simulation (block k=4 on suffix)
    def sim_accept(d_logits, v_logits, k=4):
        accepted = 0; rounds = 0
        pos = 128
        while pos < 255:
            rounds += 1
            drafts = [d_logits[0, pos + j].argmax().item() for j in range(min(k, 255 - pos))]
            v_preds = [v_logits[0, pos + j].argmax().item() for j in range(min(k, 255 - pos))]
            for j in range(len(drafts)):
                if drafts[j] == v_preds[j]:
                    accepted += 1
                else:
                    break
            pos += len(drafts)
        return accepted / max(1, rounds)

    raw_acc = sim_accept(out_d.logits, out_v.logits)
    inj_acc = sim_accept(out_d_inj.logits, out_v.logits)

    # P0 label: does injection improve?
    gate_label = 1 if inj_acc > raw_acc else 0
    catastrophic = 1 if inj_top1 < 0.15 else 0  # injected top-1 < 15% = catastrophic

    gate_data.append({
        "z_norm": z_norm,
        "raw_top1": raw_top1, "inj_top1": inj_top1,
        "raw_top5": raw_top5, "inj_top5": inj_top5,
        "raw_acc": raw_acc, "inj_acc": inj_acc,
        "gate_label": gate_label,
        "catastrophic": catastrophic,
    })

    # ── P1: Verifier argmax tokens (labels for behavior-alignment S1) ──
    verifier_argmax = out_v.logits[0, 128:255].argmax(-1).cpu().tolist()
    verifier_logits.append({"sample": fi, "verifier_argmax": verifier_argmax})

    if (fi + 1) % 20 == 0:
        elapsed = time.time() - t0
        pos_rate = sum(1 for g in gate_data if g["gate_label"] == 1) / len(gate_data)
        cat_rate = sum(1 for g in gate_data if g["catastrophic"] == 1) / len(gate_data)
        print(f"[{fi+1}/200] pos_rate={pos_rate:.2f} cat_rate={cat_rate:.2f} "
              f"raw_t1={np.mean([g['raw_top1'] for g in gate_data]):.3f} "
              f"inj_t1={np.mean([g['inj_top1'] for g in gate_data]):.3f} "
              f"speed={elapsed/(fi+1):.1f}s/sample", flush=True)

# ── Save ──
Path("outputs_eval").mkdir(exist_ok=True)

# P0
with open("outputs_eval/gate_training_data.json", "w") as f:
    json.dump(gate_data, f, indent=1)
print(f"\nP0 saved: {len(gate_data)} gate samples", flush=True)

# P1
with open("outputs_eval/verifier_argmax_labels.json", "w") as f:
    json.dump(verifier_logits, f, indent=1)
print(f"P1 saved: {len(verifier_logits)} verifier label sets", flush=True)

# Summary stats
pos_rate = sum(1 for g in gate_data if g["gate_label"] == 1) / len(gate_data)
cat_rate = sum(1 for g in gate_data if g["catastrophic"] == 1) / len(gate_data)
print(f"\n=== SUMMARY ===")
print(f"injection helps: {pos_rate:.1%}")
print(f"catastrophic:    {cat_rate:.1%}")
print(f"raw top-1 mean:  {np.mean([g['raw_top1'] for g in gate_data]):.3f}")
print(f"inj top-1 mean:  {np.mean([g['inj_top1'] for g in gate_data]):.3f}")
print(f"raw top-5 mean:  {np.mean([g['raw_top5'] for g in gate_data]):.3f}")
print(f"inj top-5 mean:  {np.mean([g['inj_top5'] for g in gate_data]):.3f}")
