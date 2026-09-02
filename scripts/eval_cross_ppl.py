"""Cross-source S1 PPL evaluation: measure state bridge quality."""
import sys, glob
import numpy as np
import torch
import torch.nn.functional as F

REPO = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")
from eval.diag_loop1_common import build_model

def eval_cross_ppl(template_ckpt, s1_ckpt, device, num_samples=200):
    m, _, dtype, _ = build_model(template_ckpt, device)
    m.eval()
    s1 = torch.load(s1_ckpt, map_location=device)
    m.load_state_dict(s1["trainable_state"], strict=False)
    
    files = sorted(glob.glob(f"{REPO}/preprocessed_data/owt_13b_s0_latents/train/*.npy"))[:num_samples]
    total_loss, total_tokens = 0.0, 0
    
    for lf in files:
        stem = lf.split("/")[-1].replace(".npy", "")
        tf = f"{REPO}/preprocessed_data/owt_rwkv_tokens/train/{stem}.npz"
        try:
            d = np.load(tf)
            lat = np.load(lf)
            ids = torch.tensor(d["input_ids"][:512], device=device, dtype=torch.long).unsqueeze(0)
            am = torch.tensor(d["attention_mask"][:512], device=device, dtype=torch.float32).unsqueeze(0)
            Z = torch.tensor(lat, device=device, dtype=dtype).unsqueeze(0)
            states = m.predict_trajectory_states(Z)
            out = m.rwkv_model(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
            past_kv = m.inject_into_cache(out.past_key_values, states)
            out2 = m.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                past_key_values=past_kv, use_cache=True, return_dict=True)
            logits = out2.logits[:, :-1].float()
            targets = ids[:, 1:]
            mask = am[:, 1:].bool()
            loss = F.cross_entropy(logits[mask], targets[mask], reduction="sum")
            total_loss += loss.item()
            total_tokens += mask.sum().item()
        except Exception as e:
            pass
    
    ppl = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
    return ppl, total_tokens

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_ckpt", required=True)
    ap.add_argument("--s1_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_samples", type=int, default=200)
    args = ap.parse_args()
    ppl, tokens = eval_cross_ppl(args.template_ckpt, args.s1_ckpt, args.device, args.num_samples)
    print(f"PPL: {ppl:.2f} ({int(tokens)} tokens)")
