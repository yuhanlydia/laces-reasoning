"""Direct S2 diffusion policy: detached rollouts and clipped transition GRPO.

No reward gradient through S1/RWKV is needed. This is diffusion-transition GRPO,
not a token-TRL wrapper, and the positive terminal variance is part of the policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PolicyConfig:
    steps: int=32
    eta: float=.3
    min_std: float=.02
    cfg_scale: float=2.
    alpha_floor: float=1e-4
    def __post_init__(self):
        if self.steps<1 or not 0<self.eta<=1 or self.min_std<=0 or self.cfg_scale<0 or not 0<self.alpha_floor<1:
            raise ValueError('Need positive steps, eta in (0,1], min_std>0, cfg>=0, alpha_floor in (0,1)')
        if any(not math.isfinite(v) for v in (self.eta,self.min_std,self.cfg_scale,self.alpha_floor)):
            raise ValueError('Nonfinite policy config')


def alpha_bar(t: torch.Tensor) -> torch.Tensor:
    s=.008
    return (torch.cos((t+s)/(1+s)*math.pi/2).square()/math.cos(s/(1+s)*math.pi/2)**2).clamp(1e-6,1.)


def policy_mode(model: nn.Module):
    # The parent BiRWKV switches CUDA kernels on self.training. Use its training
    # kernel for BOTH rollout and replay, but disable stochastic layers. Merely
    # model.eval() can route replay into the fused inference-only implementation.
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm)):
            module.eval()
    return model


def denoise(model: nn.Module, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, cfg_scale: float):
    policy_mode(model)
    dtype=next(model.parameters()).dtype
    # Keep optimizer parameters FP32; CUDA linear kernels run BF16 so the existing
    # FLA training path receives its supported activation dtype. Mean/logp stay FP32/64.
    context=torch.autocast('cuda',dtype=torch.bfloat16) if z.is_cuda else nullcontext()
    with context:
        eps=model(z.to(dtype),t.to(dtype),cond=cond.to(dtype))
        if cfg_scale!=1:
            uncond=model(z.to(dtype),t.to(dtype),cond=torch.zeros_like(cond,dtype=dtype))
            eps=uncond+cfg_scale*(eps-uncond)
    return eps.float()


def transition_mean(model, state, cond, index: int, config: PolicyConfig):
    if not 0<=index<config.steps: raise ValueError('Transition index outside schedule')
    state=state.detach().float()  # observed rollout state, never reparameterization/BPTT
    t=torch.full((state.shape[0],),1-index/config.steps,device=state.device)
    nxt=torch.tensor(1-(index+1)/config.steps,device=state.device)
    ac=alpha_bar(t[0]).clamp_min(config.alpha_floor); an=alpha_bar(nxt).clamp_min(config.alpha_floor)
    eps=denoise(model,state,t,cond.detach(),config.cfg_scale)
    clean=(state-(1-ac).sqrt()*eps)/ac.sqrt()
    schedule_sigma=config.eta*(((1-an)/(1-ac).clamp_min(1e-12))*(1-ac/an).clamp_min(0)).clamp_min(0).sqrt()
    mean=an.sqrt()*clean+(1-an-schedule_sigma.square()).clamp_min(0).sqrt()*eps
    # A deterministic terminal map depending on trainable parameters would not be
    # represented by a score-function estimator. Make EVERY transition stochastic.
    std=schedule_sigma.clamp_min(config.min_std)
    if not torch.isfinite(mean).all(): raise FloatingPointError('Nonfinite diffusion transition')
    return mean,std


def gaussian_log_prob(action,mean,std):
    std=torch.as_tensor(std,device=mean.device,dtype=torch.float64)
    if not torch.isfinite(std).all() or (std<=0).any(): raise ValueError('Nonpositive Gaussian std')
    a=action.detach().double(); m=mean.double()
    return (-.5*((a-m)/std).square()-std.log()-.5*math.log(2*math.pi)).flatten(1).sum(1)


def log_ratio(action, new_mean, old_mean, std):
    # Difference of quadratic terms avoids subtracting two huge log densities.
    a=action.detach().double(); old=old_mean.detach().double(); new=new_mean.double()
    s=torch.as_tensor(std,device=new.device,dtype=torch.float64)
    return (-.5*(((a-new)/s).square()-((a-old)/s).square())).flatten(1).sum(1)


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    old_mean: torch.Tensor
    std: float
    index: int


@dataclass
class Rollout:
    final: torch.Tensor
    transitions: list[Transition]


@torch.no_grad()
def rollout(model, cond, horizon: int, config: PolicyConfig, generator=None) -> Rollout:
    policy_mode(model)
    z=torch.randn(cond.shape[0],horizon,cond.shape[-1],device=cond.device,generator=generator)
    transitions=[]
    for index in range(config.steps):
        mean,std=transition_mean(model,z,cond,index,config)
        action=(mean+std*torch.randn(z.shape,device=z.device,generator=generator)).detach()
        transitions.append(Transition(z.detach().cpu(),action.cpu(),mean.detach().cpu(),float(std),index))
        z=action
    return Rollout(z.detach().cpu(),transitions)


@torch.no_grad()
def ddim_sample(model,cond,horizon: int,config: PolicyConfig,generator=None):
    """Deterministic DDIM control; never given a fictitious stochastic log-prob."""
    policy_mode(model); z=torch.randn(cond.shape[0],horizon,cond.shape[-1],device=cond.device,generator=generator)
    for index in range(config.steps):
        t=torch.full((z.shape[0],),1-index/config.steps,device=z.device)
        an=alpha_bar(torch.tensor(1-(index+1)/config.steps,device=z.device)).clamp_min(config.alpha_floor)
        ac=alpha_bar(t[0]).clamp_min(config.alpha_floor)
        eps=denoise(model,z,t,cond,config.cfg_scale)
        z=an.sqrt()*(z-(1-ac).sqrt()*eps)/ac.sqrt()+(1-an).sqrt()*eps
    return z.detach().cpu()


def group_advantages(rewards,min_std=1e-7):
    r=rewards.detach().float()
    if r.ndim!=1 or r.numel()<2 or not torch.isfinite(r).all(): raise ValueError('Need >=2 finite rewards')
    std=r.std(unbiased=False)
    if std<=min_std: return torch.zeros_like(r)
    return ((r-r.mean())/std).detach()


def grpo_update(model, reference, optimizer, cond, group, rewards, config, *,
                clip_range=.2,kl_coef=.01,inner_epochs=1,max_grad_norm=1.,max_log_ratio=20.):
    if len(group)!=len(rewards) or inner_epochs<1 or kl_coef<0 or not 0<clip_range<1:
        raise ValueError('Invalid GRPO arguments')
    advantages=group_advantages(rewards).to(cond.device)
    if not advantages.any():
        optimizer.zero_grad(set_to_none=True)
        return dict(skipped_flat=True,grad_norm=0.,pg_loss=0.,reference_kl=0.,clip_fraction=0.,updates=0)
    policy_mode(model); policy_mode(reference); total=len(group)*config.steps
    report=dict(skipped_flat=False,grad_norm=0.,pg_loss=0.,reference_kl=0.,clip_fraction=0.,updates=0)
    for _ in range(inner_epochs):
        optimizer.zero_grad(set_to_none=True); pg_value=kl_value=clipped=0.
        for g,adv in zip(group,advantages):
            if len(g.transitions)!=config.steps: raise ValueError('Rollout/replay step mismatch')
            for tr in g.transitions:
                state=tr.state.to(cond.device); action=tr.action.to(cond.device)
                mean,std=transition_mean(model,state,cond,tr.index,config)
                if abs(float(std)-tr.std)>1e-7: raise ValueError('Rollout/replay variance mismatch')
                lr=log_ratio(action,mean,tr.old_mean.to(cond.device),std)
                if not torch.isfinite(lr).all() or lr.abs().max()>max_log_ratio:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError('Joint transition ratio overflow; reduce LR/inner_epochs (no silent logp averaging)')
                ratio=lr.exp(); clipped+=float(((ratio-1).abs()>clip_range).float().mean())
                pg=-torch.minimum(ratio*adv,ratio.clamp(1-clip_range,1+clip_range)*adv).mean()
                with torch.no_grad(): ref_mean,_=transition_mean(reference,state,cond,tr.index,config)
                # KL reported/regularized per latent coordinate. PPO ratio above is
                # the exact JOINT transition density, not the geometric mean ratio.
                kl=((mean-ref_mean).square()/(2*std.square())).mean()
                ((pg+kl_coef*kl)/total).backward()
                pg_value+=float(pg.detach()); kl_value+=float(kl.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True); raise FloatingPointError('Nonfinite S2 gradient')
        if norm==0 and kl_coef==0:
            raise RuntimeError('Nonflat rewards but exactly zero policy gradient; inspect score-function graph')
        optimizer.step()
        report.update(grad_norm=float(norm),pg_loss=pg_value/total,reference_kl=kl_value/total,
                      clip_fraction=clipped/total,updates=report['updates']+1)
    optimizer.zero_grad(set_to_none=True)
    return report


def denoising_loss(model, cond, target, *, generator=None,mask=None):
    target=target.detach().float().to(cond.device)
    t=torch.rand(target.shape[0],device=target.device,generator=generator).clamp(.001,.999)
    eps=torch.randn(target.shape,device=target.device,generator=generator)
    ab=alpha_bar(t).view(-1,1,1)
    noisy=ab.sqrt()*target+(1-ab).sqrt()*eps
    pred=denoise(model,noisy,t,cond.detach(),1.)
    per=(pred-eps).square().mean(-1)
    if mask is None: return per.mean()
    weights=mask.to(device=per.device,dtype=per.dtype)
    if weights.shape!=per.shape or weights.sum()<=0: raise ValueError('Bad teacher latent mask')
    return (per*weights).sum()/weights.sum()


def distill_update(model,optimizer,cond,targets,*,rewards=None,temperature=.5,
                   masks=None,generator=None,max_grad_norm=1.):
    """Native S0 teacher-trace regression, or reward-weighted candidate self-distillation."""
    if not targets or temperature<=0: raise ValueError('Empty targets / bad temperature')
    policy_mode(model); optimizer.zero_grad(set_to_none=True)
    if rewards is None: weights=torch.ones(len(targets))/len(targets)
    else:
        if len(rewards)!=len(targets) or not torch.isfinite(rewards).all(): raise ValueError('Invalid rewards')
        weights=torch.softmax((rewards.detach().float()-rewards.max())/temperature,0)
    value=0.
    for i,(target,weight) in enumerate(zip(targets,weights)):
        loss=denoising_loss(model,cond,target,generator=generator,mask=None if masks is None else masks[i])
        (weight.to(loss.device)*loss).backward(); value+=float(weight)*float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
    if not torch.isfinite(norm) or norm==0: raise RuntimeError('Missing/nonfinite distillation gradient')
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return dict(distill_loss=value,grad_norm=float(norm),effective_candidates=float(1/weights.square().sum()))
