"""Inspect a trained RELAY checkpoint to figure out the actual architecture
that was used during training. Specifically, did the variational latent
encoder (TextLatentEncoder MLP) actually exist, or is the path direct
pooled-hidden -> projector?

Prints all top-level state keys with shapes, grouped by category.
"""
import argparse
import os

import torch
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/relay-2.9B-32d-4090/step_050000")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(args.ckpt_dir, "config.yaml")
    if os.path.isfile(config_path):
        cfg = OmegaConf.load(config_path)
        print("=== config.yaml relevant fields ===")
        print(f"  model.latent_dim = {cfg.model.get('latent_dim')}")
        print(f"  model.latent_encoder_type = {cfg.model.get('latent_encoder_type', 'NOT SET')}")
        print(f"  model.use_variational = {cfg.model.get('use_variational', 'NOT SET')}")
        print(f"  data.use_external_latents = {cfg.data.get('use_external_latents')}")
        print(f"  loss.kl_weight = {cfg.loss.get('kl_weight', 'NOT SET')}")
        print(f"  loss.latent_loss_weight = {cfg.loss.get('latent_loss_weight', 'NOT SET')}")
        print(f"  model.cfg_dropout_prob = {cfg.model.get('cfg_dropout_prob', 'NOT SET')}")
        print()

    # 1. Inspect model.pt (denoiser weights only)
    model_path = os.path.join(args.ckpt_dir, "model.pt")
    print(f"=== model.pt ===")
    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        print(f"  total keys: {len(state)}")

        # Group by prefix
        groups = {}
        for k in state.keys():
            top = k.split(".")[0]
            groups.setdefault(top, []).append(k)
        print(f"  top-level groups: {list(groups.keys())}")

        # Check for relevant keys
        print(f"\n  Relevant keys in model.pt:")
        relevant = ["null_token", "latent_projector", "latent_pos_embed", "latent_scale",
                    "token_type_embed", "latent_head", "text_head", "conditioning",
                    "time_film", "time_adaln"]
        for prefix in relevant:
            matches = [k for k in state.keys() if k.startswith(prefix)]
            if matches:
                for k in matches[:3]:
                    print(f"    {k}: {tuple(state[k].shape)}")
                if len(matches) > 3:
                    print(f"    ... ({len(matches)} total)")
    else:
        print("  NOT FOUND")

    # 2. Inspect trainer_state.pt (everything trainer owned, including latent_encoder)
    print(f"\n=== trainer_state.pt ===")
    trainer_path = os.path.join(args.ckpt_dir, "trainer_state.pt")
    if os.path.isfile(trainer_path):
        ts = torch.load(trainer_path, map_location="cpu", weights_only=False)
        if isinstance(ts, dict):
            print(f"  total keys: {len(ts)}")

            # Group by prefix
            groups = {}
            for k in ts.keys():
                top = k.split(".")[0]
                groups.setdefault(top, []).append(k)
            print(f"  top-level groups: {list(groups.keys())}")

            # Most important: does latent_encoder exist?
            enc_keys = [k for k in ts.keys() if k.startswith("latent_encoder.")]
            print(f"\n  latent_encoder.* keys: {len(enc_keys)}")
            for k in enc_keys:
                print(f"    {k}: {tuple(ts[k].shape) if hasattr(ts[k], 'shape') else type(ts[k])}")

            # Check latent_prior too
            prior_keys = [k for k in ts.keys() if k.startswith("latent_prior.")]
            print(f"\n  latent_prior.* keys: {len(prior_keys)}")
            for k in prior_keys:
                print(f"    {k}: {tuple(ts[k].shape) if hasattr(ts[k], 'shape') else type(ts[k])}")
        else:
            print(f"  type: {type(ts)} (not a dict — might be a different format)")
    else:
        print(f"  NOT FOUND — your training may not have a variational encoder")
        print(f"  This means the latent path is: pooled_hidden -> projector (no MLP)")

    print(f"\n=== Verdict ===")
    if not os.path.isfile(trainer_path):
        print("  No trainer_state.pt → variational encoder was NOT used")
        print("  → latent flow is: rwkv_pool (2560-d) -> projector directly")
        print("  → WAIT, but latent_dim is 32 and pooled is 2560 — there must be SOMETHING")
        print("  → Suggests you may have used a non-standard path or external latents")


if __name__ == "__main__":
    main()
