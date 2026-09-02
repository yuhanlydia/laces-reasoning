import os
from collections.abc import Sequence
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from models.albatross_rwkv7 import AlbatrossRWKV7Block
from models.state_hijacking_dit import (
    LatentRWKV7Direction,
    TrajectoryLatentRWKV,
    resolve_latent_rwkv_variant,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
TRAJECTORY_CONFIG = "rwkv_relay_2.9B_state_hijack_dit_vae32_traj32x16"


def _compose_trajectory_config(overrides: Sequence[str] = ()) -> DictConfig:
    with initialize_config_dir(
        version_base=None,
        config_dir=CONFIG_DIR,
        job_name="test_albatross_integration",
    ):
        return compose(config_name=TRAJECTORY_CONFIG, overrides=list(overrides))


def _cfg_int(cfg: DictConfig, path: str) -> int:
    value = cast(object, OmegaConf.select(cfg, path))
    assert isinstance(value, int)
    return value


def _cfg_str(cfg: DictConfig, path: str, default: str | None = None) -> str:
    value = cast(object, OmegaConf.select(cfg, path, default=default))
    assert isinstance(value, str)
    return value


class TestAlbatrossLatentRwkvHydraContract:
    def test_default_config_preserves_existing_fused_rwkv7_behavior(self) -> None:
        cfg = _compose_trajectory_config()

        assert _cfg_str(cfg, "config_name") == TRAJECTORY_CONFIG
        assert _cfg_int(cfg, "data.max_length") == 512
        assert _cfg_int(cfg, "model.n_basis") == 16
        assert _cfg_int(cfg, "model.trajectory_chunk_size") == 32
        assert _cfg_int(cfg, "model.trajectory_horizon") == 16
        assert _cfg_str(cfg, "model.trajectory_denoiser_type") == "dit"
        assert _cfg_str(cfg, "model.latent_rwkv_variant", "fused_rwkv7") == "fused_rwkv7"

    def test_opt_in_accepts_albatross_goose_for_rwkv_trajectory_denoiser(self) -> None:
        cfg = _compose_trajectory_config(
            [
                "model.trajectory_denoiser_type=rwkv",
                "+model.latent_rwkv_variant=albatross_goose",
            ]
        )

        assert _cfg_str(cfg, "model.trajectory_denoiser_type") == "rwkv"
        assert _cfg_str(cfg, "model.latent_rwkv_variant") == "albatross_goose"

    def test_invalid_albatross_cuda_variant_fails_loudly_at_model_validation_time(self) -> None:
        cfg = _compose_trajectory_config(
            [
                "model.trajectory_denoiser_type=rwkv",
                "+model.latent_rwkv_variant=albatross_cuda",
            ]
        )

        with pytest.raises(Exception, match="latent_rwkv_variant|albatross_cuda|unsupported"):
            _ = resolve_latent_rwkv_variant(cast(object, cfg.model))

    def test_fineweb4096_rwkv_denoiser_overrides_compose_with_albatross_goose(self) -> None:
        cfg = _compose_trajectory_config(
            [
                "data.max_length=4096",
                "model.trajectory_chunk_size=64",
                "model.trajectory_horizon=64",
                "model.n_basis=32",
                "model.trajectory_denoiser_type=rwkv",
                "+model.latent_rwkv_variant=albatross_goose",
            ]
        )

        assert _cfg_int(cfg, "data.max_length") == 4096
        assert _cfg_int(cfg, "model.trajectory_chunk_size") == 64
        assert _cfg_int(cfg, "model.trajectory_horizon") == 64
        assert _cfg_int(cfg, "model.n_basis") == 32
        assert _cfg_str(cfg, "model.trajectory_denoiser_type") == "rwkv"
        assert _cfg_str(cfg, "model.latent_rwkv_variant") == "albatross_goose"

    def test_default_rwkv_module_stays_fused_latent_rwkv7_direction(self) -> None:
        denoiser = TrajectoryLatentRWKV(latent_dim=32, horizon=16, hidden_size=16, depth=1)

        block = denoiser.blocks[0]
        assert denoiser.latent_rwkv_evidence == "latent_rwkv_variant=fused_rwkv7 module=LatentRWKV7Direction"
        assert isinstance(block.forward_direction, LatentRWKV7Direction)

    def test_albatross_goose_uses_native_albatross_block_and_evidence_string(self) -> None:
        denoiser = TrajectoryLatentRWKV(
            latent_dim=32,
            horizon=16,
            hidden_size=16,
            depth=1,
            variant="albatross_goose",
        )

        block = denoiser.blocks[0]
        assert denoiser.latent_rwkv_evidence == "latent_rwkv_variant=albatross_goose module=AlbatrossRWKV7Block"
        assert isinstance(block.forward_direction, AlbatrossRWKV7Block)
