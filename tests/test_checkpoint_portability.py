"""Verify checkpoints are portable across machines.

Tests that configs don't contain machine-specific paths, that models
load when the RWKV backbone is at a different path, and that env vars
correctly override hardcoded paths.

    pytest tests/test_checkpoint_portability.py -v
"""
import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "eval"))


class TestCheckpointPortability:
    """Verify checkpoints are portable across machines."""

    def test_config_no_absolute_paths(self, test_checkpoint_dir: str):
        """Config in checkpoint has no machine-specific /inspire/ paths."""
        ckpt_path = os.path.join(test_checkpoint_dir, "model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_str = ckpt["config"]

        assert "/inspire/hdd" not in cfg_str, (
            "Config contains machine-specific /inspire/hdd path. "
            "Use relative paths or env-var-based resolution."
        )

        cfg = OmegaConf.create(cfg_str)
        rwkv_path = cfg.model.get("rwkv_local_path", "")
        if rwkv_path:
            assert not rwkv_path.startswith("/inspire"), (
                f"rwkv_local_path={rwkv_path} is machine-specific"
            )

    def test_checkpoint_load_different_path(
        self, test_checkpoint_dir: str, rwkv_model_path: str, device: str
    ):
        """Checkpoint loads when rwkv_local_path is overridden to a different path."""
        from models.state_hijacking_dit import StateInjectionDiTRELAY
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ckpt_path = os.path.join(test_checkpoint_dir, "model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(ckpt["config"])

        cfg.model.rwkv_local_path = rwkv_model_path

        dtype = torch.bfloat16
        rwkv = AutoModelForCausalLM.from_pretrained(
            rwkv_model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            rwkv_model_path,
            trust_remote_code=True,
            local_files_only=True,
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
        ).to(device)

        model.load_state_dict(ckpt["trainable_state"], strict=False)
        model.eval()

        assert model is not None
        assert model.latent_dim == int(cfg.model.latent_dim)

    def test_env_var_override(self, repo_root: str, tmp_path):
        """Environment variables override config paths in relay_utils."""
        from relay_utils import get_repo_root, get_data_dir

        original_root = os.environ.get("DIFFRWKV_ROOT")
        original_data = os.environ.get("DIFFRWKV_DATA_DIR")
        original_model = os.environ.get("DIFFRWKV_MODEL_DIR")

        try:
            fake_root = str(tmp_path / "fake_repo")
            os.makedirs(fake_root, exist_ok=True)
            os.environ["DIFFRWKV_ROOT"] = fake_root
            assert get_repo_root() == fake_root

            fake_data = str(tmp_path / "fake_data")
            os.makedirs(fake_data, exist_ok=True)
            os.environ["DIFFRWKV_DATA_DIR"] = fake_data
            resolved = get_data_dir("DIFFRWKV_DATA_DIR", "preprocessed_data/default")
            assert resolved == fake_data

            os.environ.pop("DIFFRWKV_DATA_DIR", None)
            resolved_default = get_data_dir(
                "DIFFRWKV_DATA_DIR_NONEXISTENT", "preprocessed_data/default"
            )
            assert resolved_default == os.path.join(fake_root, "preprocessed_data/default")

        finally:
            if original_root is not None:
                os.environ["DIFFRWKV_ROOT"] = original_root
            else:
                os.environ.pop("DIFFRWKV_ROOT", None)
            if original_data is not None:
                os.environ["DIFFRWKV_DATA_DIR"] = original_data
            else:
                os.environ.pop("DIFFRWKV_DATA_DIR", None)
            if original_model is not None:
                os.environ["DIFFRWKV_MODEL_DIR"] = original_model
            else:
                os.environ.pop("DIFFRWKV_MODEL_DIR", None)
