"""Train Medusa heads on frozen RWKV-7 for speculative decoding.

Loads frozen RWKV + optional state injection (S0/S1/S2), does teacher-forced
forward on OWT, trains lightweight Medusa heads (h_t -> token_{t+1}, t+2, ...).
Target fully frozen; only heads update.
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


class MedusaHeads(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_heads=4, num_layers=3):
        super().__init__()
        input_dim = hidden_dim * num_layers
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, vocab_size))
            for _ in range(num_heads)
        ])

    def forward(self, h):
        return [head(h) for head in self.heads]


@torch.no_grad()
def get_hidden_states(model, ids, am, use_state_injection=False):
    if use_state_injection:
        z = encode_prefix(model, ids, am)
        states = model.predict_states(z)
        out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                               output_hidden_states=True, use_cache=True, return_dict=True)
        past_kv = model.inject_into_cache(out.past_key_values, states)
        out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                past_key_values=past_kv, output_hidden_states=True,
                                use_cache=True, return_dict=True)
        return out2.hidden_states[-1]
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                           output_hidden_states=True, use_cache=True, return_dict=True)
    h = torch.cat(out.hidden_states[-3:], dim=-1)
    return h


def load_owt_batch(token_dir, batch_size, seq_len, device):
    files = sorted(glob.glob(f"{token_dir}/*.npz"))
    idxs = np.random.randint(0, len(files), batch_size)
    batch = []
    for i in idxs:
        d = np.load(files[i])
        ids = d["input_ids"][:seq_len]
        am = d["attention_mask"][:seq_len]
        batch.append((ids, am))
    ids = torch.tensor([b[0] for b in batch], device=device)
    am = torch.tensor([b[1] for b in batch], device=device, dtype=torch.float32)
    return ids, am


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_dir", default="outputs_relay/eagle-rwkv-2.9B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--state_injection", action="store_true")
    args = ap.parse_args()

    model, tokenizer, dtype, pad_id = C.build_model(args.ckpt_dir, args.device)
    model.eval()
    hidden_dim = 2560
    vocab_size = 65536
    eagle_layers = 3
    input_dim = hidden_dim * eagle_layers

    for p in model.parameters():
        p.requires_grad = False

    medusa = MedusaHeads(hidden_dim, vocab_size, args.num_heads, num_layers=eagle_layers).to(args.device, dtype)
    opt = torch.optim.AdamW(medusa.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print(f"Medusa heads: {sum(p.numel() for p in medusa.parameters())/1e6:.1f}M params, {args.num_heads} heads", flush=True)

    for step in range(1, args.num_steps + 1):
        ids, am = load_owt_batch(args.token_dir, args.batch_size, args.seq_len, args.device)
        ids = ids.to(dtype=torch.long)

        with torch.no_grad():
            h = get_hidden_states(model, ids, am, use_state_injection=args.state_injection)

        targets = []
        for k in range(args.num_heads):
            targets.append(ids[:, k + 1:])

        h_input = h[:, :-args.num_heads].float()
        h_input.requires_grad = False
        preds = medusa(h_input.to(dtype))

        loss = 0
        for k in range(args.num_heads):
            tk = min(targets[k].shape[1], preds[k].shape[1])
            loss += nn.functional.cross_entropy(
                preds[k][:, :tk].reshape(-1, vocab_size),
                targets[k][:, :tk].reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                accs = []
                for k in range(args.num_heads):
                    tk = min(targets[k].shape[1], preds[k].shape[1])
                    pred_tok = preds[k][:, :tk].argmax(-1)
                    acc = (pred_tok == targets[k][:, :tk]).float().mean().item()
                    accs.append(acc)
                acc_str = " ".join(f"h{k}={a:.3f}" for k, a in enumerate(accs))
            print(f"[step {step}] loss={loss.item()/(args.num_heads):.3f} | {acc_str} | lr={opt.param_groups[0]['lr']:.1e}", flush=True)

        if step % 2000 == 0:
            torch.save({"medusa_state": medusa.state_dict(), "step": step, "config": vars(args)},
                       f"{args.save_dir}/medusa_step{step}.pt")

    torch.save({"medusa_state": medusa.state_dict(), "step": args.num_steps, "config": vars(args)},
               f"{args.save_dir}/medusa_final.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
