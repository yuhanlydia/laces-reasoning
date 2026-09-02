"""GPU-dependent sanity tests for 2xH200 offline machine.

Run FIRST on the H200 machine to validate the environment before training:
    pytest tests/test_h200_sanity.py -v -m gpu
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "eval"))


@pytest.mark.gpu
class TestH200Sanity:
    """GPU-dependent sanity tests for 2xH200 offline machine."""

    def test_fla_sanity(self, rwkv_model_path: str):
        """FLA 0.5.0 produces CE < 5 on frozen RWKV-7 0.4B backbone."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            rwkv_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(
            rwkv_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        ids = tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
        ).input_ids.cuda()

        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()

        ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        )
        assert ce.item() < 5.0, f"CE={ce.item():.2f} (FLA 0.4.2 produces ~15)"

    def test_model_load(self, test_checkpoint_dir: str, device: str):
        """StateInjectionDiTRELAY loads from checkpoint without error."""
        from relay_utils import load_relay_model

        model, rwkv, tokenizer, ckpt, cfg = load_relay_model(
            test_checkpoint_dir, device
        )

        assert model is not None
        assert hasattr(model, "latent_dit")
        assert hasattr(model, "predict_states")
        assert hasattr(model, "inject_into_cache")
        assert hasattr(model, "ddim_sample")
        assert model.latent_dim == int(cfg.model.latent_dim)
        assert ckpt["step"] == 1000

    def test_forward_pass(self, test_checkpoint_dir: str, device: str):
        """Forward pass produces non-NaN logits with correct shape."""
        from relay_utils import load_relay_model

        model, rwkv, tokenizer, ckpt, cfg = load_relay_model(
            test_checkpoint_dir, device
        )

        B, L = 2, 32
        input_ids = torch.randint(0, len(tokenizer), (B, L), device=device)
        attention_mask = torch.ones(B, L, dtype=torch.long, device=device)

        with torch.no_grad():
            text_logits, eps_pred, eps_target, extras = model(
                text_tokens=input_ids, attention_mask=attention_mask
            )

        assert text_logits.shape[0] == B
        assert text_logits.shape[1] == L
        assert text_logits.shape[2] >= len(tokenizer)
        assert not torch.isnan(text_logits).any(), "logits contain NaN"
        assert eps_pred.shape == (B, model.latent_dim)
        assert not torch.isnan(eps_pred).any(), "eps_pred contains NaN"

    def test_vae_reconstruction(self, test_checkpoint_dir: str, device: str):
        """VAE encodes then decodes with reasonable MSE."""
        from relay_utils import load_relay_model

        model, rwkv, tokenizer, ckpt, cfg = load_relay_model(
            test_checkpoint_dir, device
        )

        if model.encoder_type != "variational":
            pytest.skip(f"VAE test requires variational encoder, got {model.encoder_type}")

        B = 2
        input_ids = torch.randint(0, len(tokenizer), (B, 64), device=device)
        attention_mask = torch.ones(B, 64, dtype=torch.long, device=device)

        with torch.no_grad():
            out = rwkv(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            h_last = out.hidden_states[-1]
            m = attention_mask.to(h_last.dtype).unsqueeze(-1)
            pooled = (h_last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

        dtype = next(model.encoder_trunk.parameters()).dtype
        pooled_enc = pooled.to(dtype)
        h = model.encoder_trunk(pooled_enc)
        mu = model.mu_head(h)
        logvar = model.logvar_head(h).clamp(min=-10.0, max=10.0)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)

        dec_dtype = next(model.aux_decoder.parameters()).dtype
        recon_h = model.aux_decoder(z.to(dec_dtype))

        mse = F.mse_loss(recon_h, pooled.to(recon_h.dtype))
        assert mse.item() < 10.0, f"VAE reconstruction MSE={mse.item():.4f} (expected < 10)"

    def test_ddp_init(self):
        """2-GPU DDP initializes correctly with NCCL backend."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip(f"Need 2 GPUs, found {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        os.environ["WORLD_SIZE"] = "2"
        os.environ["RANK"] = "0"

        import torch.distributed as dist

        try:
            dist.init_process_group(backend="nccl", rank=0, world_size=2)
            assert dist.get_world_size() == 2
            assert dist.is_initialized()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            for key in ("WORLD_SIZE", "RANK"):
                os.environ.pop(key, None)
