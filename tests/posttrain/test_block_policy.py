import torch
from torch import nn

from laces_posttrain.block_policy import (
    block_advantages, block_grpo_update, block_log_ratio, potential_rewards, returns_to_go,
)
from laces_posttrain.policy import PolicyConfig, rollout


class Denoiser(nn.Module):
    def __init__(self, value=.07):
        super().__init__(); self.w=nn.Parameter(torch.tensor(value))
    def forward(self,z,t,cond=None):
        return self.w*z + .1*cond[:,None,:]


def test_block_log_ratio_sums_only_latent_dimension():
    action=torch.tensor([[[.3,.4],[.1,-.2]]]); old=torch.zeros_like(action); new=old+.05; std=torch.tensor(.4)
    got=block_log_ratio(action,new,old,std)
    manual=(torch.distributions.Normal(new,std).log_prob(action)-torch.distributions.Normal(old,std).log_prob(action)).sum(-1)
    assert got.shape==(1,2); torch.testing.assert_close(got.float(),manual)


def test_potential_rewards_telescope_and_terminal_bonus_hits_last_active_block():
    phi=torch.tensor([[1.,2.,4.,7.],[3.,5.,9.,9.]])
    active=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
    reward=potential_rewards(phi,exact=torch.tensor([1.,0.]),format_ok=torch.tensor([1.,1.]),active_mask=active,exact_weight=2.,format_weight=.5)
    torch.testing.assert_close(reward[0],torch.tensor([1.,2.,5.5])); torch.testing.assert_close(reward[1],torch.tensor([2.,4.5,0.]))
    assert torch.allclose(reward.sum(1),torch.tensor([8.5,6.5]))


def test_returns_to_go_reverse_cumsum_ignores_inactive_blocks():
    rewards=torch.tensor([[1.,2.,3.],[4.,5.,99.]]); active=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
    torch.testing.assert_close(returns_to_go(rewards,active),torch.tensor([[6.,5.,3.],[9.,5.,0.]]))


def test_center_only_block_advantages_and_normalized_control():
    returns=torch.tensor([[1.,10.],[3.,10.],[5.,0.]]); active=torch.tensor([[1,1],[1,1],[1,0]],dtype=torch.bool)
    center=block_advantages(returns,active,mode='center'); torch.testing.assert_close(center[:,0],torch.tensor([-2.,0.,2.]))
    assert torch.equal(center[:,1],torch.zeros(3))
    norm=block_advantages(returns,active,mode='normalized')
    assert abs(float(norm[:,0].mean()))<1e-6 and float(norm[:,0].std(unbiased=False))>0.99


def test_block_grpo_updates_policy_without_touching_reference():
    m=Denoiser(); ref=Denoiser(); ref.load_state_dict(m.state_dict()); cfg=PolicyConfig(steps=3,min_std=.04,cfg_scale=1.)
    cond=torch.ones(1,3); group=[rollout(m,cond,2,cfg,torch.Generator().manual_seed(i+1)) for i in range(4)]
    returns=torch.tensor([[0.,1.],[1.,0.],[2.,-1.],[3.,-2.]]); active=torch.ones(4,2,dtype=torch.bool)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4); before=m.w.detach().clone(); ref_before=ref.w.detach().clone()
    report=block_grpo_update(m,ref,opt,cond,group,returns,active,cfg,advantage_mode='center',kl_coef=0.)
    assert report['updates']==1 and report['grad_norm']>0 and not torch.equal(m.w,before)
    assert torch.equal(ref.w,ref_before) and ref.w.grad is None


def test_flat_block_credit_skips_update_including_weight_decay():
    m=Denoiser(); ref=Denoiser(); cfg=PolicyConfig(steps=2,cfg_scale=1.); cond=torch.ones(1,2)
    group=[rollout(m,cond,2,cfg,torch.Generator().manual_seed(i+1)) for i in range(3)]
    returns=torch.ones(3,2); active=torch.ones(3,2,dtype=torch.bool)
    opt=torch.optim.AdamW(m.parameters(),lr=.1,weight_decay=.1); before=m.w.detach().clone()
    report=block_grpo_update(m,ref,opt,cond,group,returns,active,cfg)
    assert report['skipped_flat'] and torch.equal(m.w,before)
