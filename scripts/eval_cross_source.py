"""Cross-source latent sharing pilot: 13.3B encodes, 2.9B reads.

Naive test: feed 13.3B S0 latents to 2.9B's S1 → inject → generate.
Expected: OOD (latent spaces misaligned), but measure HOW bad.

Also test: 2.9B's own latents as control (should work well).

Metrics: PPL on WikiText (teacher-forced, state-injected).
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path: sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as C
from eval.sample_prefix_suffix_cfg import encode_prefix
from scripts.train_medusa import get_hidden_states


@torch.no_grad()
def compute_ppl_with_latent(model, ids, am, z_external=None, use_own_encoder=True):
    """Compute teacher-forced PPL with optional external latent."""
    if z_external is not None:
        states = model.predict_states(z_external)
    elif use_own_encoder:
        z = encode_prefix(model, ids, am)
        states = model.predict_states(z)
    else:
        return float('inf')

    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                           use_cache=True, return_dict=True)
    past_kv = model.inject_into_cache(out.past_key_values, states)
    out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                            past_key_values=past_kv, use_cache=True, return_dict=True)
    logits = out2.logits[:, :-1]
    targets = ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 65536).float(),
                                              targets.reshape(-1), reduction='mean')
    return float(torch.exp(loss).item())


@torch.no_grad()
def run(ckpt_dir, latent_dir, data_dir, device, num_samples, patch_s1=None):
    model, tok, dtype, pad = C.build_model(ckpt_dir, device)
    model.eval()
    model._prefix_suffix_s2 = True

    if patch_s1:
        sd = torch.load(patch_s1, map_location=device)
        trainable = sd.get("trainable_state", sd)
        missing, unexpected = model.load_state_dict(trainable, strict=False)
        print(f"Loaded cross-S1 patch: {patch_s1} "
              f"(applied={len(trainable)} missing={len(missing)} unexpected={len(unexpected)})",
              flush=True)

    latent_files = sorted(glob.glob(f"{latent_dir}/*.npy"))
    token_files = sorted(glob.glob(f"{data_dir}/*.npz"))

    matched = []
    for lf in latent_files:
        stem = Path(lf).stem.replace("_tokens", "").replace("_latent", "")
        tf = Path(data_dir) / f"{stem}_tokens.npz"
        if tf.exists():
            matched.append((lf, str(tf)))
        if len(matched) >= num_samples:
            break

    print(f"Matched {len(matched)} latent-token pairs", flush=True)

    own_ppls = []
    cross_ppls = []
    raw_ppls = []

    for i, (lf, tf) in enumerate(matched):
        d = np.load(tf)
        ids = torch.tensor([d["input_ids"][:512]], device=device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)

        raw_ppl = compute_ppl_with_latent(model, ids, am, use_own_encoder=False)
        if raw_ppl == float('inf'):
            out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                   use_cache=True, return_dict=True)
            logits = out.logits[:, :-1]
            targets = ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 65536).float(),
                                                      targets.reshape(-1), reduction='mean')
            raw_ppl = float(torch.exp(loss).item())
        raw_ppls.append(raw_ppl)

        own_ppl = compute_ppl_with_latent(model, ids, am, use_own_encoder=True)
        own_ppls.append(own_ppl)

        cross_latent = np.load(lf)
        z_cross = torch.tensor(cross_latent, device=device, dtype=dtype).unsqueeze(0)
        if z_cross.shape[-1] != 32:
            print(f"  latent shape {z_cross.shape} != expected last dim 32, skip", flush=True)
            cross_ppls.append(float('nan'))
            continue
        z_flat = z_cross.mean(dim=1) if z_cross.dim() == 3 else z_cross
        cross_ppl = compute_ppl_with_latent(model, ids, am, z_external=z_flat)
        cross_ppls.append(cross_ppl)

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(matched)}] raw={np.nanmean(raw_ppls):.1f} "
                  f"own={np.nanmean(own_ppls):.1f} cross={np.nanmean(cross_ppls):.1f}", flush=True)

    print(f"\n=== CROSS-SOURCE PILOT ({len(matched)} samples) ===", flush=True)
    print(f"  raw RWKV (no injection):      PPL={np.nanmean(raw_ppls):.2f}", flush=True)
    print(f"  2.9B own S0→S1 (self-source): PPL={np.nanmean(own_ppls):.2f}", flush=True)
    print(f"  13.3B S0→2.9B S1 (cross):     PPL={np.nanmean(cross_ppls):.2f}", flush=True)
    print(f"  cross/own ratio: {np.nanmean(cross_ppls)/max(0.01,np.nanmean(own_ppls)):.1f}x", flush=True)
    print(f"  (ratio=1.0 = perfect cross-source; ratio>5 = latent space misaligned)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--latent_dir", default="preprocessed_data/owt_13b_s0_latents/train")
    ap.add_argument("--data_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--patch_s1", default=None)
    a = ap.parse_args()
    run(a.ckpt_dir, a.latent_dir, a.data_dir, a.device, a.num_samples, a.patch_s1)
