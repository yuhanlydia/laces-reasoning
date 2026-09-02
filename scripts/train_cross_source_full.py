"""Train cross-source S1 + S2 on shared 13.3B S0 latent space.

Cross-source pipeline (shared VAE):
  13.3B S0 encodes text → z_traj [H,32] → 2.9B/0.4B S1 maps z → backbone WKV states
  → frozen backbone generates from injected states.

S1 (--stage 1): train alpha_heads + state_basis + state_scale
S2 (--stage 2): train birwkv denoiser + condboundary on 13.3B latent distribution

Usage:
  # 2.9B S1
  CUDA_VISIBLE_DEVICES=0 python scripts/train_cross_source_full.py \
    --backbone 2.9B --stage 1 \
    --latent_dir preprocessed_data/owt_13b_s0_latents/train \
    --token_dir preprocessed_data/owt_rwkv_tokens/train \
    --save_dir outputs_relay/cross-source-2.9B-s1 \
    --num_steps 30000 --batch_size 8 --lr 1e-4

  # 0.4B S1
  CUDA_VISIBLE_DEVICES=1 python scripts/train_cross_source_full.py \
    --backbone 0.4B --stage 1 \
    --latent_dir preprocessed_data/owt_13b_s0_latents/train \
    --token_dir preprocessed_data/owt_rwkv_tokens/train \
    --save_dir outputs_relay/cross-source-0.4B-s1 \
    --num_steps 30000 --batch_size 8 --lr 1e-4

  # 2.9B S2 (after S1 done)
  CUDA_VISIBLE_DEVICES=0 python scripts/train_cross_source_full.py \
    --backbone 2.9B --stage 2 \
    --s1_ckpt outputs_relay/cross-source-2.9B-s1/cross_s1_final.pt \
    --latent_dir preprocessed_data/owt_13b_s0_latents/train \
    --token_dir preprocessed_data/owt_rwkv_tokens/train \
    --save_dir outputs_relay/cross-source-2.9B-s2 \
    --num_steps 50000 --batch_size 6
"""
import argparse, glob, sys, os, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path: sys.path.insert(0, str(REPO / "scripts"))

from eval.diag_loop1_common import build_model


# ── backbone-specific settings ──────────────────────────────────────────
BACKBONE_CFG = {
    "2.9B": {
        "template_ckpt": "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
    },
    "0.4B": {
        "template_ckpt": "outputs_relay/traj32x16-0.4B-s0/step_00050000",
    },
    "1.5B": {
        "template_ckpt": "outputs_relay/traj32x16-1.5B-s0/step_00050000",
    },
}


def load_batch_trajectory(token_dir, latent_dir, batch_size, seq_len, device):
    """Load paired (token, trajectory-latent) batches."""
    latent_files = sorted(glob.glob(f"{latent_dir}/*.npy"))
    np.random.shuffle(latent_files)
    token_roots = [Path(p.strip()) for p in str(token_dir).split(",") if p.strip()]

    batch_tokens, batch_ams, batch_lats = [], [], []
    for lf in latent_files:
        stem = Path(lf).stem.replace("_tokens", "")
        tf = None
        for root in token_roots:
            cand = root / f"{stem}_tokens.npz"
            if cand.exists():
                tf = cand
                break
        if tf is None:
            continue
        d = np.load(tf)
        lat = np.load(lf)
        ids = d["input_ids"][:seq_len]
        am  = d["attention_mask"][:seq_len]
        batch_tokens.append(ids)
        batch_ams.append(am)
        batch_lats.append(lat)
        if len(batch_tokens) >= batch_size:
            break

    ids = torch.tensor(np.stack(batch_tokens), device=device, dtype=torch.long)
    am  = torch.tensor(np.stack(batch_ams), device=device, dtype=torch.float32)
    lats_t = [torch.tensor(l, device=device, dtype=torch.float32) for l in batch_lats]
    lats_stacked = torch.stack(lats_t)  # [B, H, D]
    return ids, am, lats_stacked


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1: S1 training
# ═══════════════════════════════════════════════════════════════════════
def _resize_s1_basis(model, new_k, device, dtype):
    """Rebuild state_basis and alpha_heads at K=new_k (fresh init). Lets us test
    wider S1 capacity than the checkpoint's K."""
    old = model.state_basis  # [L, K_old, heads, hd, hd]
    L, _, heads, hd, hd2 = old.shape
    d_z = model.alpha_heads[0].in_features
    model.state_basis = nn.Parameter(
        torch.empty(L, new_k, heads, hd, hd2, device=device, dtype=old.dtype)
    )
    new_heads = nn.ModuleList([
        nn.Linear(d_z, new_k).to(device=device, dtype=model.alpha_heads[0].weight.dtype)
        for _ in range(len(model.alpha_heads))
    ])
    model.alpha_heads = new_heads
    print(f"Resized S1: state_basis {tuple(old.shape)} -> {tuple(model.state_basis.shape)}, "
          f"alpha_heads out {d_z}->{new_k}", flush=True)


def train_s1(ckpt_dir, token_dir, latent_dir, save_dir, device,
             num_steps=30000, batch_size=8, lr=1e-4, save_every=3000, n_basis=0):
    model, tok, dtype, pad_id = build_model(ckpt_dir, device)
    model.eval()

    # ── force independent S1 mode (no learned trajectory_state_decoder) ──
    if getattr(model, "trajectory_s1_mode", "independent") != "independent":
        print(f"Overriding trajectory_s1_mode: {model.trajectory_s1_mode} → independent")
        model.trajectory_s1_mode = "independent"
        model.trajectory_state_decoder = None

    if n_basis and n_basis > 0:
        _resize_s1_basis(model, n_basis, device, dtype)

    # ── reinitialize S1 weights ──
    s1_param_names = {"alpha_heads", "state_basis", "state_scale", "state_norm"}
    for name, p in model.named_parameters():
        base = name.split(".")[0]
        if base in s1_param_names:
            p.requires_grad = True
            if hasattr(p, "data") and p.dim() >= 2:
                nn.init.xavier_uniform_(p.data)
            elif hasattr(p, "data") and p.dim() == 1:
                nn.init.zeros_(p.data)
        else:
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_trainable/1e6:.1f}M params (S1 only)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    horizon   = int(model.trajectory_horizon)
    chunk_sz  = int(model.trajectory_chunk_size)
    seq_len   = horizon * chunk_sz

    for step in range(1, num_steps + 1):
        ids, am, Z_0 = load_batch_trajectory(token_dir, latent_dir, batch_size, seq_len, device)
        B, H, D = Z_0.shape
        Z_0 = Z_0.to(dtype=dtype)  # match model dtype (bfloat16 for 0.4B)

        # ── predict per-chunk states from 13.3B S0 latents ──
        states_list = model.predict_trajectory_states(Z_0)  # list[layer] of [B,H,heads,hd,hd]

        # ── inject trajectory states into RWKV cache ──
        with torch.no_grad():
            out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                   use_cache=True, return_dict=True)
        past_kv = model.inject_into_cache(out.past_key_values, states_list)

        # ── forward with injected states, compute CE (need gradients!) ──
        out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                past_key_values=past_kv, use_cache=True, return_dict=True)
        logits = out2.logits[:, :-1].float()
        targets = ids[:, 1:]

        ce_mask = am[:, 1:].bool()
        loss = F.cross_entropy(logits[ce_mask], targets[ce_mask])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                ppl = float(torch.exp(loss.detach()).item())
            print(f"[S1 step {step:5d}] loss={loss.item():.3f} PPL={ppl:.1f} | "
                  f"z_norm={Z_0.float().norm(dim=-1).mean():.2f} | lr={lr:.1e}", flush=True)

        if step % save_every == 0:
            ckpt = {
                "trainable_state": {k: v for k, v in model.state_dict().items()
                                    if any(p in k for p in s1_param_names)},
                "step": step,
            }
            torch.save(ckpt, f"{save_dir}/cross_s1_step{step}.pt")
            print(f"  saved step {step}", flush=True)

    torch.save({
        "trainable_state": {k: v for k, v in model.state_dict().items()
                            if any(p in k for p in s1_param_names)},
        "step": num_steps,
    }, f"{save_dir}/cross_s1_final.pt")
    print("Cross-source S1 training done.", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1b: CO-ADAPT S1 — train S1 on S2-SAMPLED Z (not clean latent)
# ═══════════════════════════════════════════════════════════════════════
def train_s1_coadapt(ckpt_dir, s2_ckpt_path, token_dir, latent_dir, save_dir, device,
                     num_steps=10000, batch_size=8, lr=2e-5, save_every=5000, n_basis=0,
                     s1_mode="independent"):
    """JOINT-SCRATCH on external latent: train S2 denoiser AND S1 together from
    scratch on the external (e.g. 13.3B) latent. loss = 1.0*diff + 0.5*CE(z_0_pred).
    s1_mode: 'independent' = linear projection (proven weak on foreign latent);
    'birwkv' = non-linear recurrent TrajectoryStateDecoder (tests whether a
    stronger S1 can read a foreign latent the linear one cannot).
    """
    model, tok, dtype, pad_id = build_model(ckpt_dir, device)
    model.eval()

    if s1_mode == "birwkv":
        from models.state_hijacking_dit import TrajectoryStateDecoder
        model.trajectory_s1_mode = "birwkv"
        model.trajectory_state_decoder = TrajectoryStateDecoder(
            latent_dim=model.latent_dim, horizon=model.trajectory_horizon,
            num_layers=model.num_layers, n_basis=int(model.state_basis.shape[1]),
            hidden_size=768, block_type="birwkv",
        ).to(device, dtype=dtype)
        print("S1: birwkv non-linear TrajectoryStateDecoder", flush=True)
    else:
        if getattr(model, "trajectory_s1_mode", "independent") != "independent":
            model.trajectory_s1_mode = "independent"
            model.trajectory_state_decoder = None
        print("S1: linear independent projection", flush=True)

    # build S2 denoiser FROM SCRATCH (trained jointly, not loaded/frozen)
    from models.state_hijacking_dit import TrajectoryLatentRWKV
    model.trajectory_dit = TrajectoryLatentRWKV(
        latent_dim=model.latent_dim, horizon=model.trajectory_horizon,
        hidden_size=768, depth=8, bidirectional=True,
        use_cond_adaln=False, use_cond_boundary=True,
    ).to(device, dtype=dtype)
    model.use_cond_boundary = True
    print("S2 denoiser: from scratch, jointly trained (NOT frozen)", flush=True)

    if n_basis and n_basis > 0:
        _resize_s1_basis(model, n_basis, device, dtype)

    # linear-S1 tensors get manual re-init; birwkv decoder keeps its own module init
    reinit_names = {"alpha_heads", "state_basis", "state_scale", "state_norm"}
    s1_param_names = reinit_names | {"trajectory_state_decoder"}
    s2_patterns = ("trajectory_dit", "cond_boundary")
    for name, p in model.named_parameters():
        base = name.split(".")[0]
        if base in s1_param_names:
            p.requires_grad = True
            if base in reinit_names:
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p.data)
                elif p.dim() == 1:
                    nn.init.zeros_(p.data)
        elif any(pat in name for pat in s2_patterns):
            p.requires_grad = True  # S2 trains jointly
        else:
            p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_trainable/1e6:.1f}M params (S1 + S2 joint-scratch)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    horizon = int(model.trajectory_horizon)
    chunk_sz = int(model.trajectory_chunk_size)
    seq_len = horizon * chunk_sz

    for step in range(1, num_steps + 1):
        ids, am, Z_0 = load_batch_trajectory(token_dir, latent_dir, batch_size, seq_len, device)
        Z_0 = Z_0.to(dtype=dtype)
        B, H, D = Z_0.shape

        # ── S2 forward WITH grad (S2 trains): diff loss + z_0_pred ──
        t = torch.rand(B, device=device, dtype=Z_0.dtype)
        ab = cosine_alpha_bar(t)
        view = (B,) + (1,) * (Z_0.dim() - 1)
        sqrt_ab = ab.sqrt().view(view)
        sqrt_1mab = (1.0 - ab).sqrt().view(view)
        noise = torch.randn_like(Z_0)
        Z_t = sqrt_ab * Z_0 + sqrt_1mab * noise
        H_half = max(1, H // 2)
        z_prefix = Z_t[:, :H_half, :].mean(dim=1)
        cond = z_prefix if getattr(model, "use_cond_boundary", False) else None
        eps_pred = model.trajectory_dit(Z_t, t.unsqueeze(-1), cond=cond)
        diff_loss = F.mse_loss(eps_pred, noise)
        z_0_pred = (Z_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-4)

        # ── S1 on sampled z_0_pred -> inject -> CE ──
        states_list = model.predict_trajectory_states(z_0_pred)
        with torch.no_grad():
            out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                   use_cache=True, return_dict=True)
        past_kv = model.inject_into_cache(out.past_key_values, states_list)
        out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                past_key_values=past_kv, use_cache=True, return_dict=True)
        logits = out2.logits[:, :-1].float()
        targets = ids[:, 1:]
        ce_mask = am[:, 1:].bool()
        ce = F.cross_entropy(logits[ce_mask], targets[ce_mask])
        loss = 1.0 * diff_loss + 0.5 * ce  # champion recipe: diff_w=1.0, coadapt_w=0.5

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"[joint-scratch step {step:5d}] loss={loss.item():.3f} | diff={diff_loss.item():.4f} "
                  f"ce={ce.item():.3f} | z0pred_norm={z_0_pred.float().norm(dim=-1).mean():.2f} | lr={lr:.1e}",
                  flush=True)
        save_pats = tuple(s1_param_names) + ("trajectory_dit", "cond_boundary")
        if step % save_every == 0:
            torch.save({"trainable_state": {k: v for k, v in model.state_dict().items()
                        if any(p in k for p in save_pats)}, "step": step},
                       f"{save_dir}/cross_s1_coadapt_step{step}.pt")
            print(f"  saved step {step}", flush=True)

    torch.save({"trainable_state": {k: v for k, v in model.state_dict().items()
                if any(p in k for p in (tuple(s1_param_names) + ('trajectory_dit', 'cond_boundary')))},
                "step": num_steps},
               f"{save_dir}/cross_s1_coadapt_final.pt")
    print("Joint-scratch (S1+S2) cross-source training done.", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2: S2 diffusion denoiser training
# ═══════════════════════════════════════════════════════════════════════
def cosine_alpha_bar(t):
    """DDPM cosine schedule: ᾱ_t = cos²(π/2 · t) / cos²(π/2 · s) with s=0.008."""
    s = 0.008
    return (torch.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2) / (np.cos(s / (1.0 + s) * np.pi / 2.0) ** 2)


def train_s2(ckpt_dir, s1_ckpt_path, token_dir, latent_dir, save_dir, device,
             num_steps=50000, batch_size=6, lr=1e-4, save_every=5000):
    model, tok, dtype, pad_id = build_model(ckpt_dir, device)

    # ── load trained cross-source S1 weights ──
    s1_state = torch.load(s1_ckpt_path, map_location=device)["trainable_state"]
    model.load_state_dict(s1_state, strict=False)
    print(f"Loaded cross-source S1 from {s1_ckpt_path}", flush=True)

    # ── ensure birwkv denoiser exists (0.4B S0 template may lack it) ──
    if not hasattr(model, "trajectory_dit") or model.trajectory_dit is None:
        from models.state_hijacking_dit import TrajectoryLatentRWKV
        dit_hidden = 768
        print(f"Creating birwkv trajectory denoiser (hidden={dit_hidden})", flush=True)
        model.trajectory_dit = TrajectoryLatentRWKV(
            latent_dim=model.latent_dim,
            horizon=model.trajectory_horizon,
            hidden_size=dit_hidden,
            depth=8,
            bidirectional=True,
            use_cond_adaln=False,
            use_cond_boundary=True,
        )
        model.use_cond_boundary = True
    model.trajectory_dit = model.trajectory_dit.to(device, dtype=dtype)

    # ── freeze everything except S2 denoiser ──
    s2_param_patterns = ("trajectory_dit", "trajectory_denoiser", "cond_boundary", "cond_boundary_proj",
                         "trajectory_pos_embed")
    for name, p in model.named_parameters():
        if any(pat in name for pat in s2_param_patterns):
            p.requires_grad = True
        else:
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_trainable/1e6:.1f}M params (S2 only)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    horizon  = int(model.trajectory_horizon)
    chunk_sz = int(model.trajectory_chunk_size)
    seq_len  = horizon * chunk_sz

    for step in range(1, num_steps + 1):
        ids, am, Z_0 = load_batch_trajectory(token_dir, latent_dir, batch_size, seq_len, device)
        Z_0 = Z_0.to(dtype=dtype)
        B, H, D = Z_0.shape

        # ── diffusion forward: add noise ──
        t = torch.rand(B, device=device, dtype=Z_0.dtype)
        ab = cosine_alpha_bar(t)
        view = (B,) + (1,) * (Z_0.dim() - 1)
        sqrt_ab = ab.sqrt().view(view)
        sqrt_1mab = (1.0 - ab).sqrt().view(view)
        noise = torch.randn_like(Z_0)
        Z_t = sqrt_ab * Z_0 + sqrt_1mab * noise

        # ── prefix conditioning: mean-pool first half latents as z_prefix ──
        H_half = max(1, H // 2)
        z_prefix = Z_t[:, :H_half, :].mean(dim=1)  # [B, D]

        # ── denoiser forward ──
        eps_pred = model.trajectory_dit(Z_t, t.unsqueeze(-1),
                                         cond=z_prefix if hasattr(model, 'use_cond_boundary') and model.use_cond_boundary else None)

        # ── diffusion MSE loss ──
        diff_loss = F.mse_loss(eps_pred, noise)

        opt.zero_grad()
        diff_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"[S2 step {step:5d}] diff_loss={diff_loss.item():.4f} | "
                  f"z_norm={Z_0.float().norm(dim=-1).mean():.2f} | "
                  f"eps_norm={eps_pred.detach().float().norm(dim=-1).mean():.2f} | lr={lr:.1e}", flush=True)

        if step % save_every == 0:
            ckpt = {
                "trainable_state": {k: v for k, v in model.state_dict().items()
                                    if any(pat in k for pat in s2_param_patterns)},
                "step": step,
            }
            torch.save(ckpt, f"{save_dir}/cross_s2_step{step}.pt")
            print(f"  saved step {step}", flush=True)

    torch.save({
        "trainable_state": {k: v for k, v in model.state_dict().items()
                            if any(pat in k for pat in s2_param_patterns)},
        "step": num_steps,
    }, f"{save_dir}/cross_s2_final.pt")
    print("Cross-source S2 training done.", flush=True)


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["2.9B", "0.4B"])
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--latent_dir", default="preprocessed_data/owt_13b_s0_latents/train")
    ap.add_argument("--token_dir",  default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--s1_ckpt", default="", help="Path to cross_s1_final.pt (for stage 2)")
    ap.add_argument("--s2_ckpt", default="", help="Frozen S2 denoiser ckpt; if set with --stage 1, "
                    "runs co-adapt S1 training on S2-sampled Z instead of clean latent.")
    ap.add_argument("--num_steps", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=3000)
    ap.add_argument("--n_basis", type=int, default=0,
                    help="If >0, resize S1 state_basis/alpha_heads to this K before training "
                         "(0 = keep checkpoint's K). Used to test wider S1 capacity.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--joint_scratch", action="store_true",
                    help="Stage 1: train S2(from scratch)+S1 jointly (champion co-adapt) "
                         "on the external latent, instead of clean-CE S1 training.")
    ap.add_argument("--s1_mode", choices=("independent", "birwkv"), default="independent",
                    help="S1 type for joint-scratch: independent=linear, birwkv=non-linear recurrent.")
    a = ap.parse_args()

    cfg = BACKBONE_CFG[a.backbone]
    ckpt_dir = cfg["template_ckpt"]

    if not os.path.isdir(ckpt_dir):
        print(f"ERROR: template checkpoint not found: {ckpt_dir}", flush=True)
        sys.exit(1)
    if not os.path.isdir(a.latent_dir):
        print(f"ERROR: latent dir not found: {a.latent_dir}", flush=True)
        sys.exit(1)

    print(f"Backbone: {a.backbone} | Stage: {a.stage} | Template: {ckpt_dir}", flush=True)

    if a.stage == 1 and a.joint_scratch:
        train_s1_coadapt(ckpt_dir, "", a.token_dir, a.latent_dir, a.save_dir, a.device,
                         a.num_steps, a.batch_size, a.lr, a.save_every, a.n_basis, a.s1_mode)
    elif a.stage == 1:
        train_s1(ckpt_dir, a.token_dir, a.latent_dir, a.save_dir, a.device,
                 a.num_steps, a.batch_size, a.lr, a.save_every, a.n_basis)
    else:
        if not a.s1_ckpt or not os.path.exists(a.s1_ckpt):
            print(f"ERROR: --s1_ckpt required for stage 2, not found: {a.s1_ckpt}", flush=True)
            sys.exit(1)
        train_s2(ckpt_dir, a.s1_ckpt, a.token_dir, a.latent_dir, a.save_dir, a.device,
                 a.num_steps, a.batch_size, a.lr, a.save_every)
