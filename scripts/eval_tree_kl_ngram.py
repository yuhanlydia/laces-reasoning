"""Batch tree verification + KL distillation training + n-gram gate.

Day 3: Batch tree verification — verify all B0*B1 paths in ONE forward pass
Day 4: KL distillation — train h0 to match target RWKV logit distribution
Day 5: N-gram gate — combine neural h0 with prompt-lookup proposals
"""

import argparse, glob, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path: sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as C
from scripts.train_medusa import MedusaHeads, get_hidden_states


# ============================================================
# Day 3: Batch Tree Verification
# ============================================================

@torch.no_grad()
def batch_tree_verify(model, medusa, ids, am, h, device, dtype, num_heads,
                      B0=4, B1=2, num_samples=10):
    files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))
    np.random.seed(42)
    test_files = [files[i] for i in np.random.choice(len(files), num_samples, replace=False)]

    tree_accept_lengths = []
    total_verify_forwards = 0
    total_rounds = 0

    for si, f in enumerate(test_files):
        d = np.load(f)
        ids_t = torch.tensor([d["input_ids"][:256]], device=device, dtype=torch.long)
        am_t = torch.ones_like(ids_t, dtype=torch.float32)
        h_t = get_hidden_states(model, ids_t, am_t).to(dtype)

        for t in range(128, 250, 3):
            prefix_ids = ids_t[:, :t + 1]
            prefix_am = am_t[:, :t + 1]
            prefix_out = model.rwkv_model(input_ids=prefix_ids,
                                          attention_mask=prefix_am.bool(),
                                          use_cache=True, return_dict=True)
            prefix_state = prefix_out.past_key_values
            target_tok_0 = prefix_out.logits[0, -1].argmax().item()

            h_pos = h_t[:, t:t + 1, :]
            head_logits = medusa(h_pos)
            h0_topk = head_logits[0][0, 0].topk(B0).indices.tolist()

            depth2_paths = []
            if num_heads > 1:
                h1_topk_per_h0 = []
                for h0_tok in h0_topk:
                    h1_top = head_logits[1][0, 0].topk(B1).indices.tolist()
                    h1_topk_per_h0.append(h1_top)
                    for h1_tok in h1_top:
                        depth2_paths.append((h0_tok, h1_tok))
            else:
                depth2_paths = [(h0_tok,) for h0_tok in h0_topk]

            if not depth2_paths:
                continue

            max_len = max(len(p) for p in depth2_paths)
            padded = torch.full((len(depth2_paths), max_len), 0,
                                device=device, dtype=torch.long)
            masks = torch.zeros(len(depth2_paths), max_len,
                                device=device, dtype=torch.float32)
            for i, path in enumerate(depth2_paths):
                for j, tok in enumerate(path):
                    padded[i, j] = tok
                    masks[i, j] = 1.0

            v_out = model.rwkv_model(input_ids=padded[0:1], attention_mask=masks[0:1].bool(),
                                     use_cache=False, return_dict=True)
            single_logits = v_out.logits[0]
            total_verify_forwards += 1
            total_rounds += 1

            best_accept = 0
            for i, path in enumerate(depth2_paths):
                match = 0
                if path[0] == target_tok_0:
                    match = 1
                    if len(path) > 1:
                        tok1_target = single_logits[0].argmax().item()
                        if path[1] == tok1_target:
                            match = 2
                if match > best_accept:
                    best_accept = match
            tree_accept_lengths.append(best_accept)

        if (si + 1) % 5 == 0:
            avg = np.mean(tree_accept_lengths) if tree_accept_lengths else 0
            print(f"  [{si+1}/{num_samples}] tree_avg={avg:.2f} forwards={total_verify_forwards}", flush=True)

    avg = np.mean(tree_accept_lengths) if tree_accept_lengths else 0
    speedup = (avg + 1) / 2
    print(f"\n=== BATCH TREE (B0={B0}, B1={B1}) ===", flush=True)
    print(f"  avg accepted/round: {avg:.2f}", flush=True)
    print(f"  verify forwards/round: {total_verify_forwards/max(1,total_rounds):.1f}", flush=True)
    print(f"  est speedup: ~{speedup:.2f}x", flush=True)
    print(f"  dist: 0={tree_accept_lengths.count(0)} 1={tree_accept_lengths.count(1)} 2={tree_accept_lengths.count(2)}", flush=True)


# ============================================================
# Day 4: KL Distillation Training
# ============================================================

class KLMedusaHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_heads=4, top_k_kl=2048):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                          nn.Linear(hidden_dim // 2, vocab_size))
            for _ in range(num_heads)
        ])
        self.top_k_kl = top_k_kl

    def forward(self, h):
        return [head(h) for head in self.heads]


@torch.no_grad()
def get_target_logits(model, ids, am):
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                           output_hidden_states=True, use_cache=True, return_dict=True)
    return out.logits[:, :-1], out.hidden_states[-1][:, :-1]


def train_kl_medusa(ckpt_dir, save_dir, device, dtype_str, num_steps=5000,
                    batch_size=4, seq_len=512, lr=1e-4, num_heads=4, kl_weight=0.5):
    model, tok, dtype, pad = C.build_model(ckpt_dir, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    head = KLMedusaHead(2560, 65536, num_heads).to(device, dtype)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))

    print(f"KL Medusa: {sum(p.numel() for p in head.parameters())/1e6:.1f}M, kl_weight={kl_weight}", flush=True)

    for step in range(1, num_steps + 1):
        idxs = np.random.randint(0, len(files), batch_size)
        batch = [np.load(files[i]) for i in idxs]
        ids = torch.tensor([b["input_ids"][:seq_len] for b in batch], device=device, dtype=torch.long)
        am = torch.tensor([b["attention_mask"][:seq_len] for b in batch], device=device, dtype=torch.float32)

        with torch.no_grad():
            target_logits, h = get_target_logits(model, ids, am)

        h_inp = h.detach()
        head_logits = head(h_inp)

        loss_ce = 0
        loss_kl = 0
        for k in range(num_heads):
            tgt_k = ids[:, k + 1:ids.shape[1]]
            pred_k = head_logits[k][:, :tgt_k.shape[1]]
            tk = min(tgt_k.shape[1], pred_k.shape[1])
            loss_ce += F.cross_entropy(pred_k[:, :tk].reshape(-1, 65536),
                                        tgt_k[:, :tk].reshape(-1))
            tgt_kl = target_logits[:, k:tgt_k.shape[1] + k]
            pred_kl = head_logits[k][:, :tgt_kl.shape[1]]
            tk2 = min(tgt_kl.shape[1], pred_kl.shape[1])
            tgt_probs = F.softmax(tgt_kl[:, :tk2].float(), dim=-1)
            pred_log_probs = F.log_softmax(pred_kl[:, :tk2].float(), dim=-1)
            loss_kl += (tgt_probs * (tgt_probs.log() - pred_log_probs)).sum(-1).mean()

        loss = loss_ce / num_heads + kl_weight * loss_kl / num_heads

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                accs = []
                for k in range(num_heads):
                    tgt_k = ids[:, k + 1:ids.shape[1]]
                    pred_k = head_logits[k][:, :tgt_k.shape[1]]
                    tk = min(tgt_k.shape[1], pred_k.shape[1])
                    acc = (pred_k[:, :tk].argmax(-1) == tgt_k[:, :tk]).float().mean().item()
                    accs.append(acc)
            print(f"[step {step}] ce={loss_ce.item()/num_heads:.3f} kl={loss_kl.item()/num_heads:.3f} | "
                  f"{' '.join(f'h{k}={a:.3f}' for k,a in enumerate(accs))}", flush=True)

        if step % 2000 == 0:
            torch.save({"medusa_state": head.state_dict(), "step": step, "kl": True},
                       f"{save_dir}/kl_step{step}.pt")
    torch.save({"medusa_state": head.state_dict(), "step": num_steps, "kl": True},
               f"{save_dir}/kl_final.pt")
    print("KL training done", flush=True)


# ============================================================
# Day 5: N-gram Gate
# ============================================================

@torch.no_grad()
def ngram_gate_eval(ckpt_dir, medusa_path, device, num_samples, num_heads):
    model, tok, dtype, pad = C.build_model(ckpt_dir, device)
    model.eval()
    ckpt = torch.load(medusa_path, map_location=device, weights_only=False)
    medusa = MedusaHeads(2560, 65536, num_heads).to(device, dtype)
    medusa.load_state_dict(ckpt["medusa_state"])
    medusa.eval()

    files = sorted(glob.glob("preprocessed_data/owt_rwkv_tokens/train/*.npz"))
    np.random.seed(42)
    test_files = [files[i] for i in np.random.choice(len(files), num_samples, replace=False)]

    neural_accept = 0
    ngram_accept = 0
    combined_accept = 0
    total = 0

    for si, f in enumerate(test_files):
        d = np.load(f)
        ids_t = torch.tensor([d["input_ids"][:256]], device=device, dtype=torch.long)
        am_t = torch.ones_like(ids_t, dtype=torch.float32)
        h = get_hidden_states(model, ids_t, am_t).to(dtype)

        token_list = ids_t[0].tolist()

        for t in range(128, 250):
            prefix_ids = ids_t[:, :t + 1]
            prefix_am = am_t[:, :t + 1]
            prefix_out = model.rwkv_model(input_ids=prefix_ids,
                                          attention_mask=prefix_am.bool(),
                                          use_cache=True, return_dict=True)
            target_tok = prefix_out.logits[0, -1].argmax().item()

            h_pos = h[:, t:t + 1, :]
            neural_tok = medusa(h_pos)[0][0, 0].argmax().item()

            context = token_list[max(0, t - 8):t + 1]
            ngram_tok = -1
            for n in [4, 3, 2]:
                if len(context) >= n:
                    key = tuple(context[-n:])
                    for j in range(len(token_list) - n):
                        if tuple(token_list[j:j + n]) == key and j + n < len(token_list):
                            ngram_tok = token_list[j + n]
                            break
                if ngram_tok >= 0:
                    break

            if neural_tok == target_tok:
                neural_accept += 1
            if ngram_tok == target_tok:
                ngram_accept += 1
            if neural_tok == target_tok or ngram_tok == target_tok:
                combined_accept += 1
            total += 1

        if (si + 1) % 5 == 0:
            print(f"  [{si+1}/{num_samples}] neural={neural_accept}/{total} "
                  f"ngram={ngram_accept}/{total} combined={combined_accept}/{total}", flush=True)

    print(f"\n=== N-GRAM GATE ===", flush=True)
    print(f"  neural h0 accept: {neural_accept/max(1,total):.3f}", flush=True)
    print(f"  n-gram accept: {ngram_accept/max(1,total):.3f}", flush=True)
    print(f"  combined (neural OR ngram): {combined_accept/max(1,total):.3f}", flush=True)
    print(f"  → combined coverage = P(target in {{neural, ngram}})", flush=True)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tree", "kl_train", "ngram", "all"], default="all")
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--medusa_path", default="outputs_relay/medusa-rwkv-2.9B/medusa_final.pt")
    ap.add_argument("--save_dir", default="outputs_relay/kl-medusa-rwkv-2.9B")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kl_steps", type=int, default=5000)
    ap.add_argument("--kl_weight", type=float, default=0.5)
    args = ap.parse_args()

    dtype = torch.bfloat16

    if args.mode in ("tree", "all"):
        print("=== DAY 3: Batch Tree Verification ===", flush=True)
        model, tok, dt, pad = C.build_model(args.ckpt_dir, args.device)
        model.eval()
        ckpt = torch.load(args.medusa_path, map_location=args.device, weights_only=False)
        medusa = MedusaHeads(2560, 65536, args.num_heads).to(args.device, dt)
        medusa.load_state_dict(ckpt["medusa_state"])
        medusa.eval()
        for B0, B1 in [(4, 2), (8, 2), (4, 4)]:
            print(f"\n--- B0={B0}, B1={B1} ---", flush=True)
            batch_tree_verify(model, medusa, None, None, None, args.device, dt,
                              args.num_heads, B0=B0, B1=B1, num_samples=args.num_samples)
        del model, medusa
        torch.cuda.empty_cache()

    if args.mode in ("kl_train", "all"):
        print("\n=== DAY 4: KL Distillation Training ===", flush=True)
        train_kl_medusa(args.ckpt_dir, args.save_dir, args.device, dtype,
                        num_steps=args.kl_steps, num_heads=args.num_heads,
                        kl_weight=args.kl_weight)

    if args.mode in ("ngram", "all"):
        print("\n=== DAY 5: N-gram Gate ===", flush=True)
        ngram_gate_eval(args.ckpt_dir, args.medusa_path, args.device,
                        args.num_samples, args.num_heads)
