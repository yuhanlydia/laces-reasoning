#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import transformers
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from relay_utils import load_relay_model


@register_model("relay")
class RELAYLM(HFLM):
    def __init__(
        self,
        ckpt_dir: str,
        mode: str = "single",
        gen_type: str = None,
        sample_steps: int = 100,
        cfg_scale: float = 2.0,
        trajectory_state_blend: float = None,
        trajectory_sampler: str = None,
        device: str = "cuda",
        batch_size=1,
        max_length: int = 1792,
        **kwargs,
    ):
        relay, rwkv, tokenizer, ckpt, cfg = load_relay_model(ckpt_dir, device)
        if gen_type is not None:
            relay._gen_type = str(gen_type)
        if trajectory_sampler is not None:
            relay._trajectory_sampler = str(trajectory_sampler)
        if trajectory_state_blend is not None:
            relay.trajectory_state_blend = float(trajectory_state_blend)
        self._relay = relay
        self._mode = str(mode)
        self._sample_steps = int(sample_steps)
        self._cfg_scale = float(cfg_scale)
        self._relay_device = device
        if self._mode in ("trajectory", "trajectory_cfg"):
            self._sample_dtype_src = relay.trajectory_dit
        else:
            self._sample_dtype_src = relay.latent_dit
        super().__init__(
            pretrained=rwkv,
            tokenizer=tokenizer,
            backend="causal",
            batch_size=batch_size,
            max_length=max_length,
            trust_remote_code=True,
            **kwargs,
        )

    @torch.no_grad()
    def _encode_cond(self, inps):
        attn = torch.ones_like(inps)
        out = self.model(input_ids=inps, attention_mask=attn.bool(),
                         output_hidden_states=True, use_cache=True, return_dict=True)
        pooled = self._relay._pool_hidden(out.hidden_states[-1], attn)
        z_prefix, _ = self._relay._encode_pooled(pooled)
        return z_prefix, out.past_key_values

    @torch.no_grad()
    def _cfg_single_states(self, cond, dtype):
        m = self._relay
        steps = self._sample_steps
        z = torch.randn(cond.shape[0], m.latent_dim, device=self._relay_device, dtype=dtype)
        from models.state_hijacking_dit import cosine_alpha_bar
        ts = torch.linspace(1.0, 0.0, steps + 1, device=self._relay_device, dtype=dtype)
        uncond = torch.zeros_like(cond)
        for i in range(steps):
            ab_cur = cosine_alpha_bar(ts[i].unsqueeze(0)).to(dtype).clamp(min=1e-4)
            ab_nxt = cosine_alpha_bar(ts[i + 1].unsqueeze(0)).to(dtype).clamp(min=1e-4)
            tb = ts[i].expand(cond.shape[0])
            eps_c = m.latent_dit(z, tb, cond=cond)
            eps = eps_c if self._cfg_scale == 1.0 else (
                m.latent_dit(z, tb, cond=uncond) + self._cfg_scale * (eps_c - m.latent_dit(z, tb, cond=uncond)))
            z0 = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
            z = ab_nxt.sqrt() * z0 + (1 - ab_nxt).sqrt() * eps
        return m.predict_states(z)

    @torch.no_grad()
    def _cfg_traj_z(self, cond, dtype):
        m = self._relay
        steps = self._sample_steps
        H = int(m.trajectory_horizon)
        z = torch.randn(cond.shape[0], H, m.latent_dim, device=self._relay_device, dtype=dtype)
        uncond = torch.zeros_like(cond)
        gen = str(getattr(m, "_gen_type", "ddpm"))
        if gen in ("rf", "flow"):
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((cond.shape[0],), i * dt, device=self._relay_device, dtype=dtype)
                v_c = m.trajectory_dit(z, t, cond=cond)
                v = v_c if self._cfg_scale == 1.0 else (
                    m.trajectory_dit(z, t, cond=uncond) + self._cfg_scale * (v_c - m.trajectory_dit(z, t, cond=uncond)))
                z = z + dt * v
        else:
            from models.state_hijacking_dit import cosine_alpha_bar
            ts = torch.linspace(1.0, 0.0, steps + 1, device=self._relay_device, dtype=dtype)
            for i in range(steps):
                ab_cur = cosine_alpha_bar(ts[i].unsqueeze(0)).to(dtype).clamp(min=1e-4)
                ab_nxt = cosine_alpha_bar(ts[i + 1].unsqueeze(0)).to(dtype).clamp(min=1e-4)
                tb = ts[i].expand(cond.shape[0])
                eps_c = m.trajectory_dit(z, tb, cond=cond)
                eps = eps_c if self._cfg_scale == 1.0 else (
                    m.trajectory_dit(z, tb, cond=uncond) + self._cfg_scale * (eps_c - m.trajectory_dit(z, tb, cond=uncond)))
                z0 = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
                z = ab_nxt.sqrt() * z0 + (1 - ab_nxt).sqrt() * eps
        return z

    @torch.no_grad()
    def _inject(self, batch_size: int, dtype):
        if self._mode == "trajectory":
            z = self._relay.trajectory_sample(
                batch_size, num_steps=self._sample_steps,
                device=self._relay_device, dtype=dtype,
            )
            layer_states = self._relay.predict_trajectory_states(z)
            return [ls[:, 0] for ls in layer_states]
        z = self._relay.ddpm_sample(
            batch_size, num_steps=self._sample_steps,
            device=self._relay_device, dtype=dtype,
        )
        return self._relay.predict_states(z)

    @torch.no_grad()
    def _build_cache(self, inps):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        dtype = next(self._sample_dtype_src.parameters()).dtype
        if self._mode in ("single_cfg", "trajectory_cfg"):
            # Encode condition from full input (for CFG guidance only),
            # but build cache from just the first token to avoid double-processing.
            cond, _ = self._encode_cond(inps)
            out = self.model(input_ids=inps[:, :1], use_cache=True, return_dict=True)
            cache = out.past_key_values
            if self._mode == "trajectory_cfg":
                z_traj = self._cfg_traj_z(cond, dtype)
                states = [ls[:, 0] for ls in self._relay.predict_trajectory_states(z_traj)]
            else:
                states = self._cfg_single_states(cond, dtype)
            return self._relay.inject_into_cache(cache, states)
        out = self.model(input_ids=inps[:, :1], use_cache=True, return_dict=True)
        cache = out.past_key_values
        states = self._inject(inps.shape[0], dtype)
        return self._relay.inject_into_cache(cache, states)

    @torch.no_grad()
    def _model_call(self, inps, attn_mask=None, labels=None):
        assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
        cache = self._build_cache(inps)
        out = self.model(
            input_ids=inps,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )
        return out.logits

    @torch.no_grad()
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        cache = self._build_cache(context)
        generation_kwargs.pop("past_key_values", None)
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            past_key_values=cache,
            use_cache=True,
            **{k: v for k, v in generation_kwargs.items() if k != "use_cache"},
        )
