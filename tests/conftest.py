"""Shared pytest fixtures for DiffRwkv test suite."""
import os

import pytest
import torch


def _repo_root() -> str:
    env_root = os.environ.get("DIFFRWKV_ROOT")
    if env_root and os.path.isdir(env_root):
        return env_root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def repo_root() -> str:
    return _repo_root()


@pytest.fixture(scope="session")
def test_checkpoint_dir(repo_root: str) -> str:
    ckpt = os.environ.get(
        "DIFFRWKV_TEST_CKPT",
        os.path.join(repo_root, "outputs_relay", "test-v5-s2-ddpm", "step_00001000"),
    )
    if not os.path.isdir(ckpt):
        pytest.skip(f"Test checkpoint not found: {ckpt}")
    return ckpt


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def dtype():
    return torch.bfloat16


@pytest.fixture(scope="session")
def rwkv_model_path(repo_root: str) -> str:
    env_path = os.environ.get("DIFFRWKV_RWKV_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(repo_root)), "models", "rwkv7-0.4B"),
        os.path.join(os.path.dirname(repo_root), "models", "rwkv7-0.4B"),
        os.path.join(repo_root, "models", "rwkv7-0.4B-world"),
        os.path.join(repo_root, "models", "rwkv7-0.4B"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    pytest.skip(f"RWKV model not found. Tried: {candidates}")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires CUDA GPU")
