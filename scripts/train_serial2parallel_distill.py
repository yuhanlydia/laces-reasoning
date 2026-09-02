"""Serial->Parallel self-distillation for trajectory decoding.

The serial rollout (blend=0.7, chunk h sees the previous chunks' real cache) is
the TEACHER (frozen, no grad). The parallel rollout (blend=1, chunk h starts only
from its own planned state, chunks independent) is the STUDENT. We distill the
teacher's per-chunk next-token distribution into the student so the parallel path
recovers the ~5.3-point quality it loses by dropping cross-chunk context. Only
S2 (trajectory denoiser) and S1 are trainable; RWKV stays frozen.

teacher: serial chunk logits (has prior-chunk context)
student: parallel chunk logits (independent per-chunk planned state)
loss   : KL(teacher || student), teacher-forced over the real suffix
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as Diag
from models.state_hijacking_dit import cosine_alpha_bar
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix


def sample_z_with_grad(model, cond, steps, cfg_scale, device, dtype):
    H = int(model.trajectory_horizon)
    B = cond.shape[0]
    z = torch.randn(B, H, model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(B)
        eps_c = model.trajectory_dit(z, t_batch, cond=cond)
        if cfg_scale == 1.0:
            eps = eps_c
        else:
            eps_u = model.trajectory_dit(z, t_batch, cond=uncond)
            eps = eps_u + cfg_scale * (eps_c - eps_u)
        z0 = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        z = ab_nxt.sqrt() * z0 + (1 - ab_nxt).sqrt() * eps
    return z


@torch.no_grad()
def teacher_serial_logits(model, ids, z_traj, blend):
    C = int(model.trajectory_chunk_size)
    H = z_traj.shape[1]
    layer_states = model.predict_trajectory_states(z_traj)
    out = model.rwkv_model(input_ids=ids[:, :1], use_cache=True, return_dict=True)
    cache = out.past_key_values
    chunks = ids[0].reshape(H, C)
    per_chunk = []
    for h in range(H):
        states_h = [ls[:, h] for ls in layer_states]
        cache = model.blend_into_cache(cache, states_h, blend)
        oh = model.rwkv_model(input_ids=chunks[h:h + 1], past_key_values=cache,
                              use_cache=True, return_dict=True)
        cache = oh.past_key_values
        per_chunk.append(oh.logits[0])
    return torch.stack(per_chunk, dim=0)


def student_parallel_logits(model, ids, z_traj):
    """Parallel student: each chunk independently rolls out from ONLY its own
    planned state (blend=1, no cross-chunk cache). Uses the train-mode chunk_rwkv7
    path (differentiable) by batching the H chunks and injecting each chunk's state
    into a fresh per-chunk cache. Requires model.train() so the backbone takes the
    differentiable chunk_rwkv7 branch instead of the inference cache path.
    """
    C = int(model.trajectory_chunk_size)
    H = z_traj.shape[1]
    layer_states = model.predict_trajectory_states(z_traj)
    chunks = ids[0].reshape(H, C)
    out = model.rwkv_model(input_ids=chunks, use_cache=True, return_dict=True)
    cache = model.blend_into_cache(out.past_key_values, [ls[0] for ls in layer_states], 1.0)
    out2 = model.rwkv_model(input_ids=chunks, past_key_values=cache,
                            use_cache=True, return_dict=True)
    return out2.logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_steps", type=int, default=300)
    ap.add_argument("--steps_diff", type=int, default=20)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--teacher_blend", type=float, default=0.7)
    ap.add_argument("--train_s1", action="store_true")
    ap.add_argument("--save_dir", default="outputs_relay/serial2parallel-distill")
    ap.add_argument("--out", default="outputs_eval/serial2parallel_distill.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model.train()  # backbone takes the differentiable chunk_rwkv7 branch only in train mode
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trajectory_dit.parameters():
        p.requires_grad = True
    trainable = list(model.trajectory_dit.parameters())
    if args.train_s1:
        for name, p in model.named_parameters():
            if any(k in name for k in ("state_basis", "alpha_heads", "state_scale")):
                p.requires_grad = True
                trainable.append(p)
    opt = torch.optim.AdamW([p for p in trainable if p.requires_grad], lr=args.lr)

    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)
    S = H * C
    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(0)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    prefix_len = (H // 2) * C
    kl_hist = []
    for step in range(1, args.num_steps + 1):
        ids_np = np.load(files[np.random.randint(len(files))])["input_ids"][:S]
        if len(ids_np) < S:
            continue
        ids = torch.tensor([ids_np], device=args.device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)

        with torch.no_grad():
            z_prefix, _c, _l = encode_prefix(model, ids[:, :prefix_len], am[:, :prefix_len])
        z_student = sample_z_with_grad(model, z_prefix, args.steps_diff, args.cfg_scale,
                                       args.device, dtype)

        teacher = teacher_serial_logits(model, ids, z_student.detach(), args.teacher_blend)
        student = student_parallel_logits(model, ids, z_student)

        t_logits = teacher[:, :-1, :].reshape(-1, teacher.shape[-1]).float()
        s_logits = student[:, :-1, :].reshape(-1, student.shape[-1]).float()
        kl = F.kl_div(F.log_softmax(s_logits, dim=-1),
                      F.softmax(t_logits, dim=-1), reduction="batchmean")

        opt.zero_grad()
        kl.backward()
        torch.nn.utils.clip_grad_norm_([p for p in trainable if p.requires_grad], 1.0)
        opt.step()
        kl_hist.append(float(kl.item()))
        if step % 10 == 0:
            print(f"[step {step}] KL(serial||parallel)={kl.item():.4f} "
                  f"running20={np.mean(kl_hist[-20:]):.4f}", flush=True)
        if step % 100 == 0:
            torch.save({"trainable": model.trajectory_dit.state_dict(), "step": step},
                       f"{args.save_dir}/distill_step{step}.pt")

    torch.save({"trainable": model.trajectory_dit.state_dict(), "step": args.num_steps},
               f"{args.save_dir}/distill_final.pt")
    first20 = float(np.mean(kl_hist[:20])) if len(kl_hist) >= 20 else float(np.mean(kl_hist))
    last20 = float(np.mean(kl_hist[-20:]))
    res = {"ckpt": args.ckpt_dir, "num_steps": len(kl_hist),
           "first20_kl": first20, "last20_kl": last20, "kl_reduction": first20 - last20,
           "kl_hist": kl_hist, "config": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\n=== serial->parallel distill ===")
    print(f"  KL: {first20:.4f} -> {last20:.4f} (reduction {first20 - last20:+.4f})")
    print(f"  verdict: {'KL DOWN -> parallel learning serial behavior' if first20 - last20 > 0.05 else 'flat -> tune'}")
    print(f"saved: {args.out}; ckpt: {args.save_dir}/distill_final.pt")


if __name__ == "__main__":
    main()
