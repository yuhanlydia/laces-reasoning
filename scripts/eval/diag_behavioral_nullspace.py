"""Diagnostic: where does Z diversity get lost?

Measures four pairwise distances + Jacobian sensitivity for the champion model:
  D_z: ||Z_i - Z_j||
  D_s: ||S1(Z_i) - S1(Z_j)|| (layer-gated)
  D_l: TVD of top-k logits
  D_y: 1[y_i != y_j]
  rho: Jacobian sensitivity to random perturbation

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/eval/diag_behavioral_nullspace.py \
    --ckpt_dir outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --num_seeds 8 --steps 20 --cfg_scale 3.0
"""
import argparse, json, sys
from pathlib import Path
import torch, numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag
from eval.sample_prefix_suffix_trajectory_cfg import encode_prefix, sample_trajectory_ddim_cfg


def topk_tvd(p_log: torch.Tensor, q_log: torch.Tensor, k=100) -> float:
    """Total variation distance on top-k logits."""
    _, p_idx = torch.topk(p_log.float(), k)
    _, q_idx = torch.topk(q_log.float(), k)
    p_mass = torch.softmax(p_log.float()[p_idx], dim=-1)
    q_mass = torch.softmax(q_log.float()[q_idx], dim=-1)
    return float(0.5 * (p_mass - q_mass).abs().sum())


def decode_greedy(model, tokenizer, input_ids, z_traj, max_tokens=32):
    """Decode text from Z trajectory (greedy). Returns token list."""
    C = int(model.trajectory_chunk_size)
    H = z_traj.shape[1]
    states = model.predict_trajectory_states(z_traj)
    out = model.rwkv_model(input_ids=input_ids, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    gen_ids = []
    for h in range(min(H, 4)):
        states_h = [s[:, h] for s in states]
        pkv = model.inject_into_cache(pkv, states_h)
        for _ in range(min(C, max_tokens - len(gen_ids))):
            last_id = gen_ids[-1] if gen_ids else input_ids[0, -1].item()
            out2 = model.rwkv_model(
                input_ids=torch.tensor([[last_id]], device=input_ids.device),
                past_key_values=pkv, use_cache=True, return_dict=True,
            )
            pkv = out2.past_key_values
            logits = out2.logits[0, -1]
            next_id = int(logits.argmax())
            gen_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
    return gen_ids


def get_logits(model, input_ids, z_traj, position=0):
    """Get logits at first generated position from Z trajectory."""
    C = int(model.trajectory_chunk_size)
    states = model.predict_trajectory_states(z_traj)
    out = model.rwkv_model(input_ids=input_ids, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    # inject chunk 0 state
    states_h = [s[:, 0] for s in states]
    pkv = model.inject_into_cache(pkv, states_h)
    # forward one token
    last_id = input_ids[0, -1].item()
    out2 = model.rwkv_model(
        input_ids=torch.tensor([[last_id]], device=input_ids.device),
        past_key_values=pkv, use_cache=True, return_dict=True,
    )
    return out2.logits[0, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--num_seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs_eval/diag_behavioral_nullspace.json")
    args = ap.parse_args()

    model, tok, dtype, _ = Diag.build_model(args.ckpt_dir, args.device)
    model._prefix_suffix_trajectory_s2 = True
    model.eval()
    H = int(model.trajectory_horizon)

    # Use a diverse set of prompts
    prompts = Diag.PASSAGES[:4]
    all_results = []

    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        am = torch.ones_like(ids)
        z_prefix, _, _ = encode_prefix(model, ids, am)

        # Sample Z's
        Z_list, L_list = [], []
        for s in range(args.num_seeds):
            torch.manual_seed(42 + s)
            z = sample_trajectory_ddim_cfg(model, z_prefix, args.steps, args.cfg_scale, args.device, dtype)
            Z_list.append(z[0].cpu())
            # Get logits at first position
            l = get_logits(model, ids, z)
            L_list.append(l.cpu())

        Z_stack = torch.stack(Z_list)  # [N, H, D]
        L_stack = torch.stack(L_list)  # [N, V]

        # D_z: pairwise Z distances
        z_flat = Z_stack.reshape(args.num_seeds, -1)
        dz_list = [(z_flat[i] - z_flat[j]).float().norm().item() for i in range(args.num_seeds) for j in range(i+1, args.num_seeds)]

        # D_s: state distances (layer-gated, use useful layers)
        USEFUL = {0, 2, 7, 11, 13, 19}
        ds_list = []
        for i in range(args.num_seeds):
            for j in range(i+1, args.num_seeds):
                si = model.predict_trajectory_states(Z_stack[i:i+1].to(args.device))
                sj = model.predict_trajectory_states(Z_stack[j:j+1].to(args.device))
                d = 0.0
                for li in USEFUL:
                    d += float((si[li].cpu().float() - sj[li].cpu().float()).norm())
                ds_list.append(d)

        # D_l: top-k TVD
        dl_list = [topk_tvd(L_stack[i], L_stack[j]) for i in range(args.num_seeds) for j in range(i+1, args.num_seeds)]

        # D_y: decoded text difference
        Y_list = []
        for s in range(args.num_seeds):
            torch.manual_seed(42 + s)
            z = sample_trajectory_ddim_cfg(model, z_prefix, args.steps, args.cfg_scale, args.device, dtype)
            y = decode_greedy(model, tok, ids, z, max_tokens=32)
            Y_list.append(tuple(y))
        dy_list = [1.0 if Y_list[i] != Y_list[j] else 0.0 for i in range(args.num_seeds) for j in range(i+1, args.num_seeds)]

        # Jacobian sensitivity rho
        torch.manual_seed(99)
        z0 = sample_trajectory_ddim_cfg(model, z_prefix, args.steps, args.cfg_scale, args.device, dtype)
        v = torch.randn_like(z0)
        v = v / v.float().norm() * 0.1  # sigma=0.1 direction

        l0 = get_logits(model, ids, z0)
        l1 = get_logits(model, ids, z0 + v)
        rho = topk_tvd(l0, l1) / 0.1  # TVD per unit sigma

        result = {
            "prompt": prompt[:80],
            "D_z_mean": float(np.mean(dz_list)), "D_z_std": float(np.std(dz_list)),
            "D_s_mean": float(np.mean(ds_list)), "D_s_std": float(np.std(ds_list)),
            "D_l_mean": float(np.mean(dl_list)), "D_l_std": float(np.std(dl_list)),
            "D_y_mean": float(np.mean(dy_list)),
            "rho_rand": float(rho),
            "unique_answers": len(set(Y_list)),
            "num_seeds": args.num_seeds,
        }
        all_results.append(result)
        print(f"prompt: {prompt[:60]}...", flush=True)
        print(f"  D_z={result['D_z_mean']:.2f}±{result['D_z_std']:.2f}  "
              f"D_s={result['D_s_mean']:.2f}±{result['D_s_std']:.2f}  "
              f"D_l={result['D_l_mean']:.4f}±{result['D_l_std']:.4f}  "
              f"D_y={result['D_y_mean']:.2f}  rho={rho:.4f}  "
              f"unique={result['unique_answers']}/{args.num_seeds}", flush=True)

    # Summary
    print("\n=== DIAGNOSIS ===")
    avg = lambda k: float(np.mean([r[k] for r in all_results]))
    print(f"  D_z (Z distance):         {avg('D_z_mean'):.2f}")
    print(f"  D_s (state distance):     {avg('D_s_mean'):.2f}")
    print(f"  D_l (logit TVD):          {avg('D_l_mean'):.4f}")
    print(f"  D_y (text diff rate):     {avg('D_y_mean'):.3f}")
    print(f"  rho (Jacobian sensitivity): {avg('rho_rand'):.4f}")
    print(f"  unique answers:           {avg('unique_answers'):.1f}/{args.num_seeds}")

    # Diagnosis
    dz, ds, dl, dy = avg('D_z_mean'), avg('D_s_mean'), avg('D_l_mean'), avg('D_y_mean')
    if dz > 1 and ds < 0.5 * dz:
        print("  → D_z high, D_s low: S1 is smoothing/collapsing Z differences")
    if ds > 1 and dl < 0.01:
        print("  → D_s high, D_l low: RWKV state dynamics insensitive to injection differences")
    if dl > 0.01 and dy < 0.1:
        print("  → D_l high, D_y low: logits differ but greedy decoding hides it")
    if dy < 0.05:
        print("  → D_y ≈ 0: all Z decode to same text — NO preference signal possible")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
