"""Refine pretrained LACES latents; never instantiate a second state writer.

R counts computation iterations; H is the existing S2 trajectory horizon. Each
coordinate remains in the original S0/S2 latent units. All S0/S1/S2/backbone
parameters stay frozen. Only LatentTrajectoryRefiner is optimized/checkpointed.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


def validate_pretrained_laces(model, checkpoint: dict, *, expected_step: int | None = None) -> dict:
    """Fail on missing/misloaded active modules, not on intentionally absent backbone keys."""
    if expected_step is not None and checkpoint.get('step') != expected_step:
        raise ValueError(f"checkpoint step={checkpoint.get('step')} != required step={expected_step}")
    if getattr(model, 'trajectory_s1_mode', None) != 'independent':
        raise ValueError('This integration requires the independent S1 path of the 30k checkpoint')
    if getattr(model, 's1_writer_type', None) != 'dynlowrank':
        raise ValueError('Expected pretrained dynlowrank S1; refusing a silent fixed/random-writer fallback')
    if getattr(model, 'trajectory_dit', None) is None:
        raise ValueError('Pretrained trajectory S2 is missing')
    s0 = {'mlp': ['encoder'], 'variational': ['encoder_trunk', 'mu_head', 'logvar_head'],
          'identity': ['latent_mu', 'latent_sigma']}.get(model.encoder_type)
    if s0 is None:
        raise ValueError(f'Unsupported S0 encoder: {model.encoder_type}')
    if getattr(model, 's0_input_adapter', None) is not None:
        s0 = s0 + ['s0_input_adapter']
    groups = {'S0': s0, 'S1': ['s1_trunk', 's1_u_head', 's1_v_head', 'state_scale'],
              'S2': ['trajectory_dit']}
    if getattr(model, 'use_learnable_blend', False):
        groups['S1'].append('blend_gate_logit')
    saved = checkpoint.get('trainable_state', {})
    current = model.state_dict()
    coverage = {}
    fingerprint = sha256()
    for group, prefixes in groups.items():
        keys = []
        for prefix in prefixes:
            matching = [key for key in current if key == prefix or key.startswith(prefix + '.')]
            if not matching:
                raise ValueError(f'Missing active {group} module/buffer: {prefix}')
            keys.extend(matching)
        for key in keys:
            if key not in saved:
                raise ValueError(f'Checkpoint is missing required {group} key: {key}')
            actual = current[key].detach().cpu()
            expected = saved[key].detach().to(device='cpu', dtype=actual.dtype)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError(f'Loaded {group} tensor differs from checkpoint: {key}')
            fingerprint.update(key.encode())
            fingerprint.update(str((tuple(actual.shape), str(actual.dtype))).encode())
            fingerprint.update(memoryview(actual.contiguous().view(torch.uint8).numpy()))
        coverage[group] = len(keys)
    return {'step': checkpoint.get('step'), 'writer_type': model.s1_writer_type,
            'latent_dim': model.latent_dim, 's1_rank': model.s1_rank,
            'active_keys_verified': coverage, 'active_fingerprint': fingerprint.hexdigest(),
            'backbone_path': str(getattr(model.rwkv_model.config, '_name_or_path',
                                  model.config.get('rwkv_local_path', 'unknown')))}


class LatentTrajectoryRefiner(nn.Module):
    """Shared recurrent refiner with an exactly identity, zero-initialized residual head.

    There is no z0 encoder, state basis, state scale, or U/V generator here. Normalization
    is used ONLY inside the update network, never on the pretrained latent itself.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, width: int = 128,
                 attention_heads: int = 4, max_step: float = .05):
        super().__init__()
        if min(latent_dim, hidden_dim, width, attention_heads) <= 0 or width % attention_heads:
            raise ValueError('Positive dimensions and width divisible by attention_heads are required')
        if not 0 < max_step <= 1:
            raise ValueError('max_step must be in (0, 1]')
        self.config = dict(latent_dim=latent_dim, hidden_dim=hidden_dim, width=width,
                           attention_heads=attention_heads, max_step=max_step)
        self.max_step = float(max_step)
        self.feature_projection = nn.Linear(hidden_dim, width)
        self.query_projection = nn.Linear(3 * latent_dim, width)
        self.attention = nn.MultiheadAttention(width, attention_heads, batch_first=True, dropout=0.)
        self.norm = nn.LayerNorm(latent_dim)
        self.cell = nn.GRUCell(width, latent_dim)
        self.delta_head = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, z0: torch.Tensor, prefix_hidden: torch.Tensor, z_prefix: torch.Tensor,
                *, steps: int, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if steps < 0 or z0.ndim != 3 or z0.shape[-1] != self.config['latent_dim']:
            raise ValueError('Expected nonnegative steps and native trajectory [B,H,latent_dim]')
        if steps == 0:
            return z0
        z = z0.float()
        base = z0.detach().float()
        features = self.feature_projection(prefix_hidden.detach().float())
        prefix = z_prefix.detach().float().unsqueeze(1).expand_as(base)
        scale = base.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
        batch, horizon, dim = base.shape
        for _ in range(steps):
            query = self.query_projection(torch.cat((z, base, prefix), -1))
            context, _ = self.attention(query, features, features,
                                        key_padding_mask=padding_mask, need_weights=False)
            work = self.cell(context.reshape(-1, context.shape[-1]),
                             self.norm(z).reshape(-1, dim)).reshape(batch, horizon, dim)
            z = z + self.max_step * scale * torch.tanh(self.delta_head(work))
        return z


def answer_policy_loss(mean: torch.Tensor, actions: torch.Tensor, sigma: torch.Tensor,
                       rewards: torch.Tensor, *, min_reward_std: float = 1e-7) -> torch.Tensor:
    """Score-function gradient in latent space. Renderer rewards need no autograd.

    actions are [G,B,H,D]. Critically, samples and rewards are detached; retaining
    reparameterized samples here would cancel the intended score-function gradient.
    """
    if actions.ndim != mean.ndim + 1 or actions.shape[1:] != mean.shape or actions.shape[0] < 2:
        raise ValueError('Need at least two actions with shape [G,*mean.shape]')
    reward = rewards.detach().float().reshape(-1)
    if reward.numel() != actions.shape[0] or not torch.isfinite(reward).all():
        raise ValueError('Invalid/nonfinite reward vector')
    std = reward.std(unbiased=False)
    if std <= min_reward_std:
        raise ValueError('Flat answer reward: no usable policy signal; inspect the wiring/precision/task')
    advantage = ((reward - reward.mean()) / std.clamp_min(min_reward_std)).detach()
    distribution = torch.distributions.Normal(mean, sigma.detach())
    log_prob = distribution.log_prob(actions.detach()).flatten(1).sum(1)
    return -(advantage * log_prob).mean()


class LACESRuntime:
    """Single full-LACES scoring/generation path with explicit cache-boundary protocols.

    aligned: write before consuming the boundary anchor token, so even the FIRST
      answer-token distribution depends on S1. No question token is consumed twice.
    legacy: reproduce the historical sampler timing, whose first token in a chunk
      uses logits computed before its write. Kept for replay, not answer training.
    """
    def __init__(self, model, checkpoint: dict, *, expected_step: int | None = 30000,
                 plan_steps: int = 1000, cfg_scale: float = 2., blend: float | None = None,
                 diffusion_sampler: str = 'ddim', protocol: str = 'aligned'):
        self.audit = validate_pretrained_laces(model, checkpoint, expected_step=expected_step)
        if protocol not in ('aligned', 'legacy') or plan_steps < 1:
            raise ValueError('Invalid protocol or plan_steps')
        if diffusion_sampler not in ('ddim', 'ddpm'):
            raise ValueError('diffusion_sampler must be ddim or ddpm')
        self.model = model.eval().requires_grad_(False)
        self.protocol = protocol
        self.plan_steps = int(plan_steps)
        self.cfg_scale = float(cfg_scale)
        self.diffusion_sampler = diffusion_sampler
        self.blend = float(model.trajectory_state_blend if blend is None else blend)
        if not 0 <= self.blend <= 1:
            raise ValueError('blend must be in [0,1]')
        self.dtype = next(model.trajectory_dit.parameters()).dtype
        self.calls = Counter(S0=0, S1=0, S2=0)
        self._hooks = []
        # Hooks record actual pretrained module execution, not merely loaded names.
        s0 = model.mu_head if model.encoder_type == 'variational' else getattr(model, 'encoder', None)
        for group, module in [('S0', s0), ('S1', model.s1_trunk), ('S2', model.trajectory_dit)]:
            if module is not None:
                def count(_module, _inputs, _output, group=group):
                    self.calls[group] += 1
                self._hooks.append(module.register_forward_hook(count))
        model._prefix_suffix_trajectory_s2 = True
        model._training_stage = 2

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def prepare(self, prefix_ids: torch.Tensor, *, seed: int):
        """Use the EXISTING stochastic S0 and S2. No labels enter this function."""
        from scripts.eval.sample_prefix_suffix_trajectory_cfg import sample_trajectory_cfg
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] < 2:
            raise ValueError('Expected a single prefix with at least two tokens')
        devices = [prefix_ids.device.index or 0] if prefix_ids.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            out = self.model.rwkv_model(input_ids=prefix_ids, attention_mask=torch.ones_like(prefix_ids).bool(),
                                        output_hidden_states=True, use_cache=True, return_dict=True)
            hidden = out.hidden_states[-1]
            pooled = self.model._pool_hidden(hidden, torch.ones_like(prefix_ids))
            zp, _ = self.model._encode_pooled(pooled)
            if self.model.encoder_type == 'identity':
                self.calls['S0'] += 1  # identity S0 has buffers, not a forward-hookable module
            z0 = sample_trajectory_cfg(self.model, zp, self.plan_steps, self.cfg_scale,
                                      prefix_ids.device, self.dtype, diffusion_sampler=self.diffusion_sampler)
        if not torch.isfinite(z0).all():
            raise ValueError('S2 produced nonfinite latents')
        return z0.detach().float(), hidden.detach().float(), zp.detach().float()

    def _stream(self, prefix_ids, z, length: int, *, raw: bool = False) -> Iterator[torch.Tensor]:
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] < 2:
            raise ValueError('Expected a single prefix with at least two tokens')
        if z.ndim != 3 or z.shape[0] != 1 or z.shape[-1] != self.model.latent_dim:
            raise ValueError('Invalid native latent trajectory')
        chunk = int(self.model.trajectory_chunk_size)
        if length < 1 or length > z.shape[1] * chunk:
            raise ValueError('Requested answer exceeds the S2 horizon; do not truncate gold labels')
        inputs = prefix_ids if self.protocol == 'legacy' else prefix_ids[:, :-1]
        with torch.no_grad():
            out = self.model.rwkv_model(input_ids=inputs, attention_mask=torch.ones_like(inputs).bool(),
                                        use_cache=True, return_dict=True)
        cache, logits = out.past_key_values, out.logits[0, -1]
        pending = prefix_ids[:, -1:]
        for index in range(length):
            if index % chunk == 0 and not raw and self.blend != 0:
                # Use the checkpoint's original S1, state_scale and blend/replace operator.
                states = self.model.predict_states(z[:, index // chunk].to(self.dtype))
                if self.blend < 1:
                    cache = self.model.blend_into_cache(cache, states, self.blend)
                else:
                    cache = self.model.inject_into_cache(cache, states)
            if self.protocol == 'aligned':
                out = self.model.rwkv_model(input_ids=pending, past_key_values=cache,
                                            use_cache=True, return_dict=True)
                cache, logits = out.past_key_values, out.logits[0, -1]
            token = yield logits
            pending = torch.tensor([[int(token)]], device=prefix_ids.device, dtype=torch.long)
            if self.protocol == 'legacy':
                out = self.model.rwkv_model(input_ids=pending, past_key_values=cache,
                                            use_cache=True, return_dict=True)
                cache, logits = out.past_key_values, out.logits[0, -1]

    def score(self, prefix_ids, answer_ids, z, *, raw: bool = False) -> torch.Tensor:
        """Mean gold-token log probability, using exactly the generation timing.

        Gradients are permitted here; callers choose no_grad for policy rewards. A
        backend lacking state-input gradients MUST be caught by the trainer's gate.
        """
        if answer_ids.ndim != 2 or answer_ids.shape[0] != 1:
            raise ValueError('Expected one answer token sequence')
        stream = self._stream(prefix_ids, z, answer_ids.shape[1], raw=raw)
        logits = next(stream)
        values = []
        for index, token in enumerate(answer_ids[0]):
            values.append(F.log_softmax(logits.float(), -1)[token])
            if index + 1 < answer_ids.shape[1]:
                logits = stream.send(int(token))
        stream.close()
        return torch.stack(values).mean()

    @torch.no_grad()
    def generate(self, prefix_ids, z, *, max_new_tokens: int, raw: bool = False,
                 temperature: float = 0., top_k: int = 0, top_p: float = 1.,
                 repetition_penalty: float = 1., eos_id: int | None = None):
        from scripts.eval.sample_prefix_suffix_trajectory_cfg import apply_repetition_penalty, apply_top_p
        stream = self._stream(prefix_ids, z, max_new_tokens, raw=raw)
        logits = next(stream)
        ids, raw_logps = [], []
        history = prefix_ids[0].tolist()
        for index in range(max_new_tokens):
            sample_logits = apply_repetition_penalty(logits.float(), history, repetition_penalty)
            if temperature <= 0:
                token = int(sample_logits.argmax())
            else:
                probabilities = torch.softmax(sample_logits / temperature, -1)
                if top_k > 0:
                    vals, indices = torch.topk(probabilities, min(top_k, probabilities.numel()))
                    probabilities = torch.zeros_like(probabilities).scatter(-1, indices, vals)
                    probabilities /= probabilities.sum()
                probabilities = apply_top_p(probabilities, top_p)
                token = int(torch.multinomial(probabilities, 1))
            if eos_id is not None and token == eos_id:
                break
            ids.append(token)
            history.append(token)
            raw_logps.append(float(F.log_softmax(logits.float(), -1)[token]))
            if index + 1 < max_new_tokens:
                logits = stream.send(token)
        stream.close()
        return ids, raw_logps
