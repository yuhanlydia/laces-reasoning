#!/usr/bin/env python3
"""Diagnostic: Does shared Z improve top-k overlap between 0.4B and 13.3B?

Key question: given the SAME Z (from 13.3B S0), does injecting it through
per-model S1 adapters make the two models agree MORE or LESS than raw?
"""
import sys, glob, json
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.eval.diag_loop1_common import build_model

device = "cuda"; dtype = torch.bfloat16

# Load drafter + patch denoised S1
drafter, _, _, _ = build_model(
    "outputs_relay/drafter-04b-traj32x16-joint-scratch-coadapt/step_00006000", device
)
s1 = torch.load(
    "outputs_relay/cross-source-0.4B-s1-denoised-v2/cross_s1_final.pt",
    map_location=device,
)
drafter.load_state_dict(s1["trainable_state"], strict=False)
drafter.eval()
rwkv_d = drafter.rwkv_model

# Load verifier (13.3B with trained S2)
verifier, _, _, _ = build_model(
    "outputs_relay/owt512-traj32x16-13.3B-basis32-prefix-suffix-blend0p5-s2-rwkv-rf/step_00150000",
    device,
)
verifier.eval()
rwkv_v = verifier.rwkv_model

files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))[:15]

raw_top1, raw_top5, raw_top10 = [], [], []
inj_top1, inj_top5, inj_top10 = [], [], []
z_norms = []

for fi, f in enumerate(files):
    d = np.load(f)
    ids = torch.tensor([d["input_ids"][:256]], device=device, dtype=torch.long)
    am = torch.ones_like(ids, dtype=torch.float32)

    # ── RAW (no injection) ──
    out_d = rwkv_d(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    out_v = rwkv_v(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)

    # ── Get shared Z from 13.3B S0 ──
    # Manually: rwkv forward → _pool_hidden → _encode_pooled
    out_vh = rwkv_v(
        input_ids=ids, attention_mask=am.bool(),
        output_hidden_states=True, use_cache=True, return_dict=True,
    )
    pooled = verifier._pool_hidden(out_vh.hidden_states[-1], am)
    z_prefix, _ = verifier._encode_pooled(pooled)
    z_norms.append(z_prefix.norm(dim=-1).item())

    # ── INJECTED: same Z, per-model S1 ──
    # 0.4B
    s_d = drafter.predict_states(z_prefix)
    out_d_pre = rwkv_d(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    pkv_d = drafter.inject_into_cache(out_d_pre.past_key_values, s_d)
    out_d_inj = rwkv_d(input_ids=ids, attention_mask=am.bool(), past_key_values=pkv_d, use_cache=True, return_dict=True)

    # 13.3B
    s_v = verifier.predict_states(z_prefix)
    out_v_pre = rwkv_v(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    pkv_v = verifier.inject_into_cache(out_v_pre.past_key_values, s_v)
    out_v_inj = rwkv_v(input_ids=ids, attention_mask=am.bool(), past_key_values=pkv_v, use_cache=True, return_dict=True)

    # ── Compare over suffix ──
    for pos in range(128, 255):
        v_argmax_raw = out_v.logits[0, pos].argmax().item()
        v_argmax_inj = out_v_inj.logits[0, pos].argmax().item()

        d_topk_raw = out_d.logits[0, pos].topk(10).indices.tolist()
        d_topk_inj = out_d_inj.logits[0, pos].topk(10).indices.tolist()

        raw_top1.append(v_argmax_raw == out_d.logits[0, pos].argmax().item())
        raw_top5.append(v_argmax_raw in out_d.logits[0, pos].topk(5).indices.tolist())
        raw_top10.append(v_argmax_raw in d_topk_raw)

        inj_top1.append(v_argmax_inj == out_d_inj.logits[0, pos].argmax().item())
        inj_top5.append(v_argmax_inj in out_d_inj.logits[0, pos].topk(5).indices.tolist())
        inj_top10.append(v_argmax_inj in d_topk_inj)

    n = len(raw_top1)
    print(
        f"[{fi+1:2d}] raw top1={np.mean(raw_top1[-128:]):.3f} top5={np.mean(raw_top5[-128:]):.3f} | "
        f"inj top1={np.mean(inj_top1[-128:]):.3f} top5={np.mean(inj_top5[-128:]):.3f} | "
        f"z_norm={z_norms[-1]:.1f}",
        flush=True,
    )

print(f"\n{'='*60}")
print(f"  0.4B top-k contains 13.3B argmax? (shared Z from 13.3B S0)")
print(f"{'='*60}")
print(f"  z_norm: {np.mean(z_norms):.1f} ± {np.std(z_norms):.1f}")
print()
for name, rl, il in [
    ("top-1  (exact match)", raw_top1, inj_top1),
    ("top-5                ", raw_top5, inj_top5),
    ("top-10               ", raw_top10, inj_top10),
]:
    rm, im = np.mean(rl), np.mean(il)
    print(f"  {name}:  raw={rm:.4f}  injected={im:.4f}  delta={im-rm:+.4f}")
print()

# Bonus: correlation between z_norm and agreement
print(f"  Corr(z_norm, raw_top1):  {np.corrcoef(z_norms, [np.mean(raw_top1[i*127:(i+1)*127]) for i in range(len(z_norms))])[0,1]:.3f}")

result = {
    "z_norm_mean": float(np.mean(z_norms)),
    "raw_top1": float(np.mean(raw_top1)),
    "raw_top5": float(np.mean(raw_top5)),
    "raw_top10": float(np.mean(raw_top10)),
    "inj_top1": float(np.mean(inj_top1)),
    "inj_top5": float(np.mean(inj_top5)),
    "inj_top10": float(np.mean(inj_top10)),
}
Path("outputs_eval/diag_shared_z_topk.json").parent.mkdir(parents=True, exist_ok=True)
with open("outputs_eval/diag_shared_z_topk.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nsaved: outputs_eval/diag_shared_z_topk.json")
