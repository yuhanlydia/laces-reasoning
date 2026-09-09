"""Shared utilities for RELAY (StateInjectionDiTRELAY) evaluation scripts.

Provides:
  - load_relay_model(): Load StateInjectionDiTRELAY from checkpoint
  - get_repo_root(): Resolve repo root from env or relative path
  - list_npz_files(): List .npz files in a directory
"""
import os
import sys
import glob
from typing import Tuple, Optional

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_repo_root() -> str:
    """Get repo root from DIFFRWKV_ROOT env var or relative to this file."""
    env_root = os.environ.get("DIFFRWKV_ROOT")
    if env_root and os.path.isdir(env_root):
        return env_root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def ensure_repo_in_path():
    """Add repo root to sys.path if not already present."""
    root = get_repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def load_relay_model(
    ckpt_dir: str,
    device: str = "cuda",
    rwkv_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, object, dict, object]:
    """Load StateInjectionDiTRELAY from checkpoint.

    Args:
        ckpt_dir: Path to checkpoint directory containing model.pt
        device: Device to load model on
        rwkv_path: Optional local frozen-backbone override. This is useful when
            the checkpoint stores a path relative to a different checkout.

    Returns:
        (model, rwkv, tokenizer, ckpt, cfg)
    """
    ensure_repo_in_path()
    from models.state_hijacking_dit import StateInjectionDiTRELAY

    dtype = torch.bfloat16
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    rwkv_path = rwkv_path or cfg.model.rwkv_local_path
    rwkv = AutoModelForCausalLM.from_pretrained(
        rwkv_path, trust_remote_code=True, torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        rwkv_path, trust_remote_code=True, local_files_only=True
    )

    model = StateInjectionDiTRELAY(
        config=cfg.model,
        rwkv_model=rwkv,
        vocab_size=len(tokenizer),
        latent_dim=int(cfg.model.latent_dim),
        n_basis=int(cfg.model.get("n_basis", 16)),
        dit_hidden=int(cfg.model.get("dit_hidden", 256)),
        dit_depth=int(cfg.model.get("dit_depth", 4)),
        dit_num_heads=int(cfg.model.get("dit_num_heads", 4)),
        dit_num_tokens=int(cfg.model.get("dit_num_tokens", 4)),
        encoder_type=str(cfg.model.get("encoder_type", "mlp")),
        alpha_type=str(cfg.model.get("alpha_type", "linear")),
        alpha_hidden=int(cfg.model.get("alpha_hidden", 256)),
        latent_stats_path=cfg.model.get("latent_stats_path", None),
    ).to(device)

    model.load_state_dict(ckpt["trainable_state"], strict=False)
    model._gen_type = str(cfg.training.get("gen_type", "ddpm"))
    model._training_stage = int(cfg.training.get("stage", 1))
    for p in model.parameters():
        if p.requires_grad and p.dtype != dtype and p.dtype.is_floating_point:
            p.data = p.data.to(dtype)
    model.eval()

    return model, rwkv, tokenizer, ckpt, cfg


def list_npz_files(data_dir: str, max_files: Optional[int] = None) -> list:
    """List .npz files in directory, sorted."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if max_files is not None:
        files = files[:max_files]
    return files


def get_data_dir(env_var: str, default_subpath: str) -> str:
    """Resolve data directory from env var or default path under repo root."""
    env_val = os.environ.get(env_var)
    if env_val and os.path.isdir(env_val):
        return env_val
    return os.path.join(get_repo_root(), default_subpath)
