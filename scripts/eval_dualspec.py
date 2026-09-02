"""Evaluate DualSpec: trajectory-conditioned speculative decoding.

DualSpec = dual-stream draft heads: read BOTH hidden(t) AND z_{k+1} (next chunk
latent). S2 jointly samples all 16 z's before any token generation, so z_{k+1}
is known before chunk k finishes — giving DualSpec a "future plan" advantage
that standard Medusa lacks.

Usage:
  python scripts/eval_dualspec.py \
    --ckpt_dir <relay_ckpt> --dualspec_path <dualspec_ckpt> --state_injection
"""

import argparse, glob, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path: sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as C
from scripts.train_dualspec import DualSpecHeads


def chunk_forward_verify(model, prefix_ids, draft_tokens, device, dtype):
    """Verify draft tokens with chunk_rwkv7. Returns accepted count."""
    with torch.no_grad():
        full_ids = torch.cat([prefix_ids,
                              torch.tensor(draft_tokens, device=device).unsqueeze(0)], dim=1)
        out = model.rwkv_model(input_ids=full_ids, use_cache=True, return_dict=True)
        logits = out.logits[0]  # [seq_len, vocab]

        accepted = 0
        for i, draft in enumerate(draft_tokens):
            pos = prefix_ids.shape[1] + i - 1
            if pos >= logits.shape[0]:
                break
            if draft == logits[pos].argmax().item():
                accepted += 1
            else:
                break
    return accepted


def ngram_lookup(token_list, t, n_range=(4, 3, 2)):
    for n in n_range:
        if t + 1 >= n:
            key = tuple(token_list[t + 1 - n:t + 1])
            for j in range(len(token_list) - n):
                if tuple(token_list[j:j + n]) == key and j + n < len(token_list):
                    return token_list[j + n]
    return -1


@torch.no_grad()
def run(args):
    model, tok, dtype, pad_id = C.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    ckpt = torch.load(args.dualspec_path, map_location=args.device, weights_only=False)
    dualspec = DualSpecHeads(2560, model.latent_dim, 65536, args.num_heads).to(args.device, dtype)
    dualspec.load_state_dict(ckpt["dualspec_state"])
    dualspec.eval()

    horizon = model.trajectory_horizon
    chunk_sz = model.trajectory_chunk_size

    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(42)
    idxs = np.random.choice(len(files), args.num_samples, replace=False)

    total_accept, total_rounds, total_tokens = 0, 0, 0
    t0 = time.time()

    for sample_idx, idx in enumerate(idxs):
        d = np.load(files[idx])
        ids = d["input_ids"][:horizon * chunk_sz]
        am = d["attention_mask"][:horizon * chunk_sz]

        # ── S2: sample full trajectory Z ──
        z_clean = model._encode_trajectory_chunks(
            torch.tensor([ids], device=args.device, dtype=torch.long),
            torch.tensor([am], device=args.device, dtype=torch.float32).bool()
        )[0].reshape(1, horizon, model.latent_dim)

        z_prefix = z_clean[:, :horizon // 2, :]
        z_sampled = z_clean  # placeholder: in real CFG mode, S2 samples from noise
        # TODO: real S2 CFG sampling:
        # z_sampled = sample_trajectory_cfg(model, z_prefix, steps=100, cfg_scale=3, ...)

        Z = z_sampled[0]  # [H, D]

        # ── Per-chunk speculative generation ──
        all_tokens = list(ids[:horizon // 2 * chunk_sz])  # prefix tokens
        past_kv_chunks = [None] * horizon
        hidden_per_chunk = [None] * horizon

        for h in range(horizon // 2, horizon):
            # Build current context
            ctx_len = min(len(all_tokens), (h + 1) * chunk_sz)
            ctx_ids = torch.tensor([all_tokens[:ctx_len]], device=args.device, dtype=torch.long)

            # Get hidden state from current chunk context
            out = model.rwkv_model(input_ids=ctx_ids, output_hidden_states=True,
                                   use_cache=True, return_dict=True)
            last_hidden = out.hidden_states[-1][:, -1, :]  # [1, hidden_dim]

            # Inject planned state for this chunk
            z_h = Z[h:h + 1, :]  # [1, D]
            states_h = model.predict_states(z_h)
            past_kv = model.inject_into_cache(out.past_key_values, states_h)

            # ── DualSpec draft ──
            draft_tokens = []
            if h + 1 < horizon:
                z_next = Z[h + 1:h + 2, :]  # next chunk's latent
                logits_list = dualspec(last_hidden, z_next)
                for head_logits in logits_list:
                    draft_tokens.append(head_logits[0].argmax().item())

            # N-gram supplement
            ngram_tok = ngram_lookup(all_tokens, len(all_tokens) - 1)
            if ngram_tok >= 0 and ngram_tok not in draft_tokens:
                draft_tokens.append(ngram_tok)

            # ── Verify ──
            prefix_tensor = torch.tensor([all_tokens], device=args.device, dtype=torch.long)
            accepted = chunk_forward_verify(model, prefix_tensor, draft_tokens, args.device, dtype)

            total_accept += accepted
            total_rounds += 1
            total_tokens += 1 + accepted

            # Append accepted tokens
            all_tokens.extend(draft_tokens[:accepted])
            # Add ground-truth next token
            gt_pos = h * chunk_sz
            if gt_pos < len(ids):
                all_tokens.append(int(ids[gt_pos]))

        if sample_idx % 10 == 0:
            avg_acc = total_accept / max(1, total_rounds)
            print(f"[{sample_idx}/{args.num_samples}] avg_accept={avg_acc:.2f} "
                  f"speedup_est={1 + avg_acc:.2f}x | elapsed={time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    avg_accept = total_accept / max(1, total_rounds)
    speedup_est = 1 + avg_accept
    print(f"\n=== DualSpec Results ===")
    print(f"  Samples: {args.num_samples}")
    print(f"  Avg accept/round: {avg_accept:.3f}")
    print(f"  Est speedup: {speedup_est:.2f}x")
    print(f"  Wall time: {elapsed:.1f}s ({elapsed / args.num_samples:.1f}s/sample)")
    print(f"  Total tokens: {total_tokens}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dualspec_path", required=True)
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    run(ap.parse_args())
