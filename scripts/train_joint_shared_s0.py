"""Joint-scratch multi-backbone training with a SHARED S0 latent space.

Trains 0.4B and 2.9B together from scratch so the latent space is defined by
BOTH models, not one imposed on the other. This is the joint-scratch analogue
of the single-backbone recipe, extended across two backbones.

Shared (tied) across both models:
  - encoder_trunk, mu_head, logvar_head   (the S0 latent space)
Per-backbone (own):
  - s0_input_adapter  (maps each backbone's hidden -> canonical width)
  - state_basis, alpha_heads, state_scale (the S1 state bridge)

Each step (champion co-adapt recipe, diff_w=1.0 coadapt_w=0.5):
  z0 = shared_S0(hidden_A, hidden_B)            (averaged shared clean latent)
  z0_pred, diff = shared_S2.sample(z0)          (S2-sampled latent + MSE)
  loss = 1.0*diff + 0.5*[CE_A(z0_pred) + CE_B(z0_pred)]
  so BOTH S1s learn to decode the SAME S2-sampled shared latent (co-adapt).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_joint_shared_s0.py \
    --backbone_a outputs_relay/traj32x16-0.4B-s0/step_00050000 \
    --backbone_b outputs_relay/traj32x16-2.9B-s0/step_00050000 \
    --token_dir preprocessed_data/owt_rwkv_tokens/train \
    --canonical 512 --num_steps 8000 --batch_size 4 --lr 1e-4 \
    --lambda_cross 1.0 --save_dir outputs_relay/joint-shared-s0-04b-2p9b
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
from eval.diag_loop1_common import build_model


def tie_shared_s0(model_a, model_b):
    """Tie S0 latent modules (trunk/mu/logvar) so both backbones share ONE latent
    space. Adapters stay per-backbone. Returns the shared module names."""
    model_b.encoder_trunk = model_a.encoder_trunk
    model_b.mu_head = model_a.mu_head
    model_b.logvar_head = model_a.logvar_head
    return ["encoder_trunk", "mu_head", "logvar_head"]


def encode_z(model, ids, am):
    """Shared-S0 encode: backbone hidden -> per-backbone adapter -> shared trunk -> z."""
    with torch.no_grad():
        out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                               output_hidden_states=True, use_cache=True, return_dict=True)
        h = out.hidden_states[-1]
        m = am.to(h.dtype).unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    x = pooled.to(next(model.encoder_trunk.parameters()).dtype)
    if getattr(model, "s0_input_adapter", None) is not None:
        x = model.s0_input_adapter(x)
    h2 = model.encoder_trunk(x)
    return model.mu_head(h2)  # deterministic mean latent [B, d_z]


def cosine_alpha_bar(t):
    s = 0.008
    return (torch.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2) / (np.cos(s / (1.0 + s) * np.pi / 2.0) ** 2)


def s2_sample_z0pred(shared_s2, z_0, use_cond_boundary):
    """Champion co-adapt: add noise to z_0, run shared S2 denoiser, recover the
    SAMPLED clean-latent proxy z_0_pred (what S1 must decode at inference).

    The DDPM cosine schedule assumes ~unit-scale latents; raw latents here have
    norm ~35-46, which makes eps MSE explode. Normalize per element to ~unit RMS
    before diffusion, then un-scale z_0_pred back so S1 sees the original scale.
    """
    B, H, D = z_0.shape
    rms = z_0.detach().pow(2).mean().clamp(min=1e-8).sqrt()  # scalar scale
    z_n = z_0 / rms
    t = torch.rand(B, device=z_0.device, dtype=z_0.dtype)
    ab = cosine_alpha_bar(t)
    view = (B,) + (1,) * (z_0.dim() - 1)
    sqrt_ab = ab.sqrt().view(view)
    sqrt_1mab = (1.0 - ab).sqrt().view(view)
    noise = torch.randn_like(z_n)
    z_t = sqrt_ab * z_n + sqrt_1mab * noise
    cond = z_t[:, :max(1, H // 2), :].mean(dim=1) if use_cond_boundary else None
    eps_pred = shared_s2(z_t, t.unsqueeze(-1), cond=cond)
    z0_pred_n = (z_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-4)
    diff_loss = F.mse_loss(eps_pred, noise)
    return z0_pred_n * rms, diff_loss  # un-scale back to original latent scale


def ce_from_ztraj(model, ids, am, z_traj):
    """z_traj [B,H,d_z] -> per-chunk states -> inject -> CE on real tokens (grad)."""
    states = model.predict_trajectory_states(z_traj)
    with torch.no_grad():
        out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                               use_cache=True, return_dict=True)
    pkv = model.inject_into_cache(out.past_key_values, states)
    out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                            past_key_values=pkv, use_cache=True, return_dict=True)
    logits = out2.logits[:, :-1].float()
    targets = ids[:, 1:]
    mask = am[:, 1:].bool()
    return F.cross_entropy(logits[mask], targets[mask])


def load_batch(token_dir, batch_size, seq_len, device):
    files = sorted(glob.glob(f"{token_dir}/*_tokens.npz")) or sorted(glob.glob(f"{token_dir}/*.npz"))
    np.random.shuffle(files)
    ids_l, am_l = [], []
    for f in files:
        d = np.load(f)
        if len(d["input_ids"]) < seq_len:
            continue
        ids_l.append(d["input_ids"][:seq_len])
        am_l.append(d["attention_mask"][:seq_len])
        if len(ids_l) >= batch_size:
            break
    ids = torch.tensor(np.stack(ids_l), device=device, dtype=torch.long)
    am = torch.tensor(np.stack(am_l), device=device, dtype=torch.float32)
    return ids, am


def _init_s1(model):
    names = {"alpha_heads", "state_basis", "state_scale", "state_norm"}
    for name, p in model.named_parameters():
        if name.split(".")[0] in names:
            p.requires_grad = True
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p.data)
            elif p.dim() == 1:
                nn.init.zeros_(p.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_a", default="outputs_relay/traj32x16-0.4B-s0/step_00050000")
    ap.add_argument("--backbone_b", default="outputs_relay/traj32x16-2.9B-s0/step_00050000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--canonical", type=int, default=512)
    ap.add_argument("--num_steps", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_cross", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=4000)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = a.device
    print(f"loading backbone A: {a.backbone_a}", flush=True)
    ma, _, dtype, _ = build_model(a.backbone_a, dev)
    print(f"loading backbone B: {a.backbone_b}", flush=True)
    mb, _, _, _ = build_model(a.backbone_b, dev)

    # build shared-S0 modules at canonical width on both, then tie the trunk
    for m in (ma, mb):
        m.s0_input_adapter = nn.Linear(m.hidden_size, a.canonical).to(dev, dtype=dtype)
        mid = a.canonical * 4
        m.encoder_trunk = nn.Sequential(
            nn.Linear(a.canonical, mid), nn.LayerNorm(mid), nn.GELU(),
            nn.Linear(mid, mid), nn.LayerNorm(mid), nn.GELU(),
        ).to(dev, dtype=dtype)
        m.mu_head = nn.Linear(mid, m.latent_dim).to(dev, dtype=dtype)
        m.logvar_head = nn.Linear(mid, m.latent_dim).to(dev, dtype=dtype)
    tie_shared_s0(ma, mb)  # B now shares A's trunk/mu/logvar

    # shared S2 denoiser (one prior over the shared latent, used by BOTH backbones)
    from models.state_hijacking_dit import TrajectoryLatentRWKV
    shared_s2 = TrajectoryLatentRWKV(
        latent_dim=ma.latent_dim, horizon=ma.trajectory_horizon,
        hidden_size=768, depth=8, bidirectional=True,
        use_cond_adaln=False, use_cond_boundary=True,
    ).to(dev, dtype=dtype)
    use_cond = True

    # freeze both backbones; train adapters + shared S0 + shared S2 + both S1
    for m in (ma, mb):
        for p in m.parameters():
            p.requires_grad = False
    _init_s1(ma); _init_s1(mb)
    shared_and_adapter = [ma.s0_input_adapter, mb.s0_input_adapter,
                          ma.encoder_trunk, ma.mu_head, ma.logvar_head, shared_s2]
    for mod in shared_and_adapter:
        for p in mod.parameters():
            p.requires_grad = True

    params = [p for m in (ma, mb) for p in m.parameters() if p.requires_grad]
    # dedup (tied modules appear in both)
    seen = set(); uniq = []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p)); uniq.append(p)
    print(f"Trainable (unique): {sum(p.numel() for p in uniq)/1e6:.1f}M "
          f"(shared S0 + 2 adapters + 2 S1)", flush=True)
    opt = torch.optim.AdamW(uniq, lr=a.lr)

    Hc = int(ma.trajectory_horizon) * int(ma.trajectory_chunk_size)
    Path(a.save_dir).mkdir(parents=True, exist_ok=True)

    diff_w, coadapt_w = 1.0, 0.5  # champion recipe weights
    H = int(ma.trajectory_horizon)
    for step in range(1, a.num_steps + 1):
        ids, am = load_batch(a.token_dir, a.batch_size, Hc, dev)
        # shared S0 encode -> per-chunk clean latent z_0 (B,H,d_z)
        z0_a = encode_z(ma, ids, am).unsqueeze(1).expand(-1, H, -1).contiguous()
        z0_b = encode_z(mb, ids, am).unsqueeze(1).expand(-1, H, -1).contiguous()
        z0 = 0.5 * (z0_a + z0_b)  # shared latent (both backbones' encode averaged)
        # shared S2: sample z_0_pred + diffusion loss (co-adapt uses SAMPLED z)
        z0_pred, diff_loss = s2_sample_z0pred(shared_s2, z0, use_cond)
        # ONE shared co-adapt CE, decoded by the large backbone (mb=2.9B), so the
        # shared latent is anchored to 2.9B behavior; 0.4B shapes it via shared S0/S2.
        ce = ce_from_ztraj(mb, ids, am, z0_pred)
        loss = diff_w * diff_loss + coadapt_w * ce

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(uniq, 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"[joint step {step:5d}] loss={loss.item():.3f} | diff={diff_loss.item():.4f} "
                  f"ce={ce.item():.3f} | lr={a.lr:.1e}", flush=True)

        if step % a.save_every == 0:
            names = {"alpha_heads", "state_basis", "state_scale", "state_norm",
                     "s0_input_adapter", "encoder_trunk", "mu_head", "logvar_head"}
            for tag, m in [("A", ma), ("B", mb)]:
                sd = {k: v for k, v in m.state_dict().items() if any(n in k for n in names)}
                torch.save({"trainable_state": sd, "step": step},
                           f"{a.save_dir}/joint_{tag}_step{step}.pt")
            torch.save({"trainable_state": shared_s2.state_dict(), "step": step},
                       f"{a.save_dir}/joint_S2_step{step}.pt")
            print(f"  saved step {step}", flush=True)

    print("Joint shared-S0 training done.", flush=True)


if __name__ == "__main__":
    main()
