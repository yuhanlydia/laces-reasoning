"""End-to-end speculative decoding test: BiTraj vs Medusa vs DualSpec.

Measures per-chunk draft accuracy and estimated speedup on a fixed test set.
"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path: sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag
from scripts.train_dualspec import DualSpecHeads as DualSpecV7
from scripts.train_chunk_draft import BiTrajDecoder
from scripts.train_medusa import MedusaHeads, get_hidden_states


@torch.no_grad()
def test_bitraj(model, bitraj, ids, am, num_tokens=4):
    """BiTraj: Z_full → per-chunk draft tokens."""
    B = ids.shape[0]
    H, C = model.trajectory_horizon, model.trajectory_chunk_size
    Z = model._encode_trajectory_chunks(ids, am.bool())[0].reshape(B, H, -1)
    logits = bitraj(Z)  # [B, H, K, vocab]
    preds = logits.argmax(-1)  # [B, H, K]
    targets = ids.reshape(B, H, C)[:, :, :num_tokens]
    correct = (preds == targets).float()
    per_chunk_acc = correct.mean(dim=-1)  # [B, H]
    per_pos_acc = correct.float().mean(dim=(0, 1))  # [K]
    return per_chunk_acc, per_pos_acc


@torch.no_grad()
def test_medusa(model, medusa, ids, am, num_heads=4):
    """Standard Medusa: hidden → draft tokens."""
    h = get_hidden_states(model, ids, am, use_state_injection=True)
    preds = medusa(h.to(dtype=medusa.heads[0][0].weight.dtype))  # list of [B, T, vocab]
    accs = []
    for k in range(num_heads):
        offset = k + 1
        p = preds[k]  # [B, T, vocab]
        targets = ids[:, offset:offset + p.shape[1]]
        min_len = min(p.shape[1], targets.shape[1])
        correct = (p[:, :min_len].argmax(-1) == targets[:, :min_len]).float().mean().item()
        accs.append(correct)
    return accs


@torch.no_grad()
def test_medusa_boundary(model, medusa, ids, am, num_heads=4):
    """Medusa accuracy specifically at chunk boundaries (first K tokens per chunk)."""
    H, C = model.trajectory_horizon, model.trajectory_chunk_size
    h = get_hidden_states(model, ids, am, use_state_injection=True)
    preds = medusa(h.to(dtype=medusa.heads[0][0].weight.dtype))
    accs = []
    for k in range(num_heads):
        offset = k + 1
        correct_list = []
        for h_idx in range(H):
            pos = h_idx * C
            if pos < preds[k].shape[1] and pos + offset < ids.shape[1]:
                pred_tok = preds[k][:, pos].argmax(-1)
                target = ids[:, pos + offset]
                correct_list.append((pred_tok == target).float().mean().item())
        if correct_list:
            accs.append(sum(correct_list) / len(correct_list))
    return accs


@torch.no_grad()
def test_dualspec(model, dualspec, ids, am, num_heads=4):
    """DualSpec v7: hidden + Z_full → draft tokens."""
    B = ids.shape[0]
    H, C = model.trajectory_horizon, model.trajectory_chunk_size
    S = H * C
    Z = model._encode_trajectory_chunks(ids, am.bool())[0].reshape(B, H, -1)
    states = model.predict_trajectory_states(Z)
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                           output_hidden_states=True, use_cache=True, return_dict=True)
    past_kv = model.inject_into_cache(out.past_key_values, states)
    out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                            past_key_values=past_kv, output_hidden_states=True,
                            use_cache=True, return_dict=True)
    H_all = out2.hidden_states[-1]
    hidden_dim = model.rwkv_model.config.hidden_size
    hd = next(dualspec.parameters()).dtype

    accs = []
    for k in range(num_heads):
        offset = k + 1
        T = S - offset
        h_in = H_all[:, :T, :].reshape(B * T, hidden_dim)
        c_idx = (torch.arange(T, device=ids.device) // C + offset).clamp(max=H - 1)
        c_idx = c_idx.unsqueeze(0).expand(B, -1).reshape(-1)
        logits_list = dualspec(h_in.to(hd), Z.to(hd), c_idx)
        pred_tokens = logits_list[k].argmax(-1)
        targets = ids[:, offset:].reshape(-1)
        acc = (pred_tokens == targets).float().mean().item()
        accs.append(acc)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--bitraj_path", default="")
    ap.add_argument("--medusa_path", default="")
    ap.add_argument("--dualspec_path", default="")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--num_tokens", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, _, dtype, _ = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    H, C = model.trajectory_horizon, model.trajectory_chunk_size
    S = H * C

    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(42)
    idxs = np.random.choice(len(files), args.num_samples, replace=False)

    # Load models
    bitraj = medusa = dualspec = None
    if args.bitraj_path and Path(args.bitraj_path).exists():
        bitraj = BiTrajDecoder(latent_dim=model.latent_dim, vocab_size=65536,
                                num_tokens=args.num_tokens).to(args.device, dtype)
        ckpt = torch.load(args.bitraj_path, map_location=args.device, weights_only=False)
        bitraj.load_state_dict(ckpt["decoder_state"])
        bitraj.eval()
        print(f"Loaded BiTraj from {args.bitraj_path}")

    if args.medusa_path and Path(args.medusa_path).exists():
        hidden_dim = model.rwkv_model.config.hidden_size
        medusa = MedusaHeads(hidden_dim, 65536, 4).to(args.device, torch.bfloat16)
        ckpt = torch.load(args.medusa_path, map_location=args.device, weights_only=False)
        medusa.load_state_dict(ckpt["medusa_state"])
        medusa.eval()
        print(f"Loaded Medusa from {args.medusa_path}")

    if args.dualspec_path and Path(args.dualspec_path).exists():
        hidden_dim = model.rwkv_model.config.hidden_size
        dualspec = DualSpecV7(hidden_dim, 65536, 4).to(args.device, dtype)
        ckpt = torch.load(args.dualspec_path, map_location=args.device, weights_only=False)
        dualspec.load_state_dict(ckpt["dualspec_state"])
        dualspec.eval()
        print(f"Loaded DualSpec v7 from {args.dualspec_path}")

    # Test
    t0 = time.time()
    bitraj_chunk_accs, bitraj_pos_accs = [], []
    medusa_accs, medusa_boundary_accs = [], []
    dualspec_accs = []

    for idx in idxs:
        d = np.load(files[idx])
        ids = torch.tensor(d["input_ids"][:S], device=args.device, dtype=torch.long).unsqueeze(0)
        am = torch.tensor(d["attention_mask"][:S], device=args.device, dtype=torch.float32).unsqueeze(0)

        if bitraj:
            ca, pa = test_bitraj(model, bitraj, ids, am, args.num_tokens)
            bitraj_chunk_accs.append(ca)
            bitraj_pos_accs.append(pa)
        if medusa:
            ma = test_medusa(model, medusa, ids, am)
            medusa_accs.append(ma)
            mb = test_medusa_boundary(model, medusa, ids, am)
            medusa_boundary_accs.append(mb)
        if dualspec:
            da = test_dualspec(model, dualspec, ids, am)
            dualspec_accs.append(da)

    elapsed = time.time() - t0
    print(f"\n=== Results ({args.num_samples} samples, {elapsed:.1f}s) ===\n")

    if bitraj:
        chunk_avg = torch.cat(bitraj_chunk_accs, dim=0).mean().item()
        pos_avg = torch.stack(bitraj_pos_accs).mean(dim=0)
        print("BiTraj (Z_full → draft tokens):")
        print(f"  Per-chunk accuracy (first {args.num_tokens} tokens): {chunk_avg:.3f}")
        print(f"  Per-position: " + " ".join(f"pos{k}={pos_avg[k].item():.3f}" for k in range(args.num_tokens)))

    if medusa:
        avg_ma = np.array(medusa_accs).mean(axis=0)
        avg_mb = np.array(medusa_boundary_accs).mean(axis=0)
        print("\nMedusa (hidden → draft tokens):")
        print("  Overall: " + " ".join(f"h{k}={avg_ma[k]:.3f}" for k in range(4)))
        print("  Chunk boundary only: " + " ".join(f"h{k}={avg_mb[k]:.3f}" for k in range(len(avg_mb))))

    if dualspec:
        avg_da = np.array(dualspec_accs).mean(axis=0)
        print("\nDualSpec v7 (hidden+Z → draft tokens):")
        print("  " + " ".join(f"h{k}={avg_da[k]:.3f}" for k in range(len(avg_da))))

    # Head-to-head at chunk boundaries
    if bitraj and medusa:
        bitraj_h0 = pos_avg[0].item() if len(pos_avg) > 0 else 0
        medusa_boundary_h0 = avg_mb[0] if len(avg_mb) > 0 else 0
        print(f"\n=== Chunk boundary head-to-head ===")
        print(f"  BiTraj pos0 (z→token):      {bitraj_h0:.3f}")
        print(f"  Medusa h0 at boundaries:     {medusa_boundary_h0:.3f}")
        if bitraj_h0 > medusa_boundary_h0:
            print(f"  → BiTraj wins by +{bitraj_h0 - medusa_boundary_h0:.3f}")
        else:
            print(f"  → Medusa wins by +{medusa_boundary_h0 - bitraj_h0:.3f}")


if __name__ == "__main__":
    main()
