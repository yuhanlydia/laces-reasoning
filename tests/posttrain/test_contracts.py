"""Data and probability contracts; no downloads or checkpoints."""
import importlib
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn


def module(name):
    name = 'laces_posttrain.' + name
    assert importlib.util.find_spec(name) is not None, f'{name} is not implemented'
    return importlib.import_module(name)


def row(i, category='math', n=10, **extra):
    return dict(question_id=i, question=f'Question {i}?', options=[f'Option {i}-{j}' for j in range(n)],
                answer='B', answer_index=1, category=category, cot_content='A worked explanation.', **extra)


def test_normalized_question_blocks_permuted_options():
    d=module('data')
    a=d.canonicalize(row(1),source='official_validation')
    b=d.canonicalize(dict(row(1),question='  QUESTION   1? ',options=list(reversed(row(1)['options']))), source='official_validation')
    assert d.question_key(a)==d.question_key(b)


def test_pilot_reserves_one_dev_per_category_and_all_test(tmp_path):
    d=module('data')
    val=[row(i + 10*j,c) for j,c in enumerate(['math','law']) for i in range(5)]
    test=[row(100),row(101)]
    d.prepare_bundle(val,test,tmp_path,seed=7)
    train,dev,manifest=d.read_training_bundle(tmp_path)
    assert (len(train),len(dev))==(8,2)
    assert manifest['protocol']=='mmlu_pro_validation_adaptation_v1'
    assert set(d.question_key(x) for x in train).isdisjoint(d.question_key(x) for x in dev)
    assert all('teacher_text' not in x for x in d.read_jsonl(tmp_path/'test.jsonl'))
    with pytest.raises(ValueError,match='acknowledge'):
        d.read_evaluation_bundle(tmp_path,'test',acknowledge_test=False)


def test_test_questions_cannot_enter_external_train(tmp_path):
    d=module('data')
    external=[row(88)]
    with pytest.raises(ValueError,match='overlap'):
        d.prepare_bundle([row(i) for i in range(5)],[row(88)],tmp_path,external_train=external,source_id='local-v1')


def test_manifest_tamper_is_rejected(tmp_path):
    d=module('data')
    d.prepare_bundle([row(i) for i in range(5)],[row(50)],tmp_path)
    with (tmp_path/'train.jsonl').open('a') as f: f.write('{}\n')
    with pytest.raises(ValueError,match='hash'):
        d.read_training_bundle(tmp_path)


def test_actual_options_not_hardcoded_four():
    d=module('data'); x=d.canonicalize(row(2,n=10),source='x')
    p=d.format_prompt(x,mode='direct')
    assert 'J. Option 2-9' in p
    assert 'worked explanation' not in p.lower()
    assert d.parse_final_choice('The answer is (J).',10)==9
    assert d.parse_final_choice('A B C',10) is None
    assert d.parse_final_choice('The answer is (J).',4) is None


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__(); self.w=nn.Parameter(torch.tensor(.07))
    def forward(self,z,t,cond=None):
        return self.w*z + .1*cond[:,None,:]


def test_detached_action_has_nonzero_score_gradient_and_old_bug_cancels():
    p=module('policy'); mean=torch.tensor([[.2,-.5]],requires_grad=True); eps=torch.tensor([[.7,-.4]])
    # Regression reproduction: old implementation differentiates through the sample.
    action=mean+.3*eps
    old=-.5*((action-mean)/.3).square().sum()
    assert torch.equal(torch.autograd.grad(old,mean,retain_graph=True)[0],torch.zeros_like(mean))
    lp=p.gaussian_log_prob(action,mean,torch.tensor(.3)).sum()
    grad=torch.autograd.grad(lp,mean)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    torch.testing.assert_close(grad, eps/.3)


def test_rollout_and_replay_match_all_steps_including_terminal():
    p=module('policy'); m=Denoiser(); cfg=p.PolicyConfig(steps=4,eta=.3,min_std=.02,cfg_scale=1.)
    cond=torch.ones(1,3); traj=p.rollout(m,cond,horizon=2,config=cfg,generator=torch.Generator().manual_seed(3))
    assert len(traj.transitions)==4
    for tr in traj.transitions:
        assert tr.std>0 and not tr.action.requires_grad and not tr.state.requires_grad
        mean,std=p.transition_mean(m,tr.state,cond,tr.index,cfg)
        torch.testing.assert_close(mean,tr.old_mean)
        ratio=p.log_ratio(tr.action,mean,tr.old_mean,std)
        torch.testing.assert_close(ratio,torch.zeros_like(ratio))
    loss=-sum(p.gaussian_log_prob(t.action,t.old_mean.clone().requires_grad_(),torch.tensor(t.std)).mean() for t in traj.transitions)
    assert loss.requires_grad


def test_joint_log_density_ratio_is_not_geometric_mean():
    p=module('policy')
    action=torch.tensor([[[.3,.4]]]); old=torch.zeros_like(action); new=old+.05; std=torch.tensor(.4)
    value=p.log_ratio(action,new,old,std)
    expected=(torch.distributions.Normal(new,std).log_prob(action)-torch.distributions.Normal(old,std).log_prob(action)).flatten(1).sum(1)
    torch.testing.assert_close(value.float(),expected)


def test_group_advantages_flat_and_nonflat():
    p=module('policy')
    a=p.group_advantages(torch.tensor([1.,1.,1.]))
    assert torch.equal(a,torch.zeros_like(a))
    b=p.group_advantages(torch.tensor([0.,1.,2.]))
    assert b[0]<0 and b[2]>0 and abs(b.mean())<1e-6


def test_grpo_reward_term_changes_original_denoiser():
    p=module('policy'); m=Denoiser(); ref=Denoiser(); cfg=p.PolicyConfig(steps=3,min_std=.04,cfg_scale=1.)
    cond=torch.ones(1,3)
    group=[p.rollout(m,cond,2,cfg,torch.Generator().manual_seed(i+1)) for i in range(4)]
    rewards=torch.tensor([float(g.final.sum()) for g in group])
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    before=m.w.detach().clone()
    report=p.grpo_update(m,ref,opt,cond,group,rewards,cfg,kl_coef=0.)
    assert report['grad_norm']>0 and not torch.equal(m.w,before)


def test_flat_rewards_skip_optimizer_including_weight_decay():
    p=module('policy'); m=Denoiser(); cfg=p.PolicyConfig(steps=2,cfg_scale=1.)
    c=torch.ones(1,2); groups=[p.rollout(m,c,2,cfg,torch.Generator().manual_seed(i)) for i in range(2)]
    opt=torch.optim.AdamW(m.parameters(),lr=.1,weight_decay=.1); before=m.w.detach().clone()
    report=p.grpo_update(m,Denoiser(),opt,c,groups,torch.ones(2),cfg)
    assert report['skipped_flat'] and torch.equal(m.w,before)


def test_deterministic_endpoint_is_rejected_for_grpo():
    p=module('policy')
    with pytest.raises(ValueError): p.PolicyConfig(min_std=0.)
    with pytest.raises(ValueError): p.PolicyConfig(steps=0)


def test_s2_rollout_and_replay_use_training_kernel_with_dropout_disabled():
    p=module('policy')
    class ModeDenoiser(Denoiser):
        def __init__(self): super().__init__(); self.drop=nn.Dropout(.8); self.seen=[]
        def forward(self,z,t,cond=None):
            self.seen.append((self.training,self.drop.training,torch.is_grad_enabled()))
            return self.drop(super().forward(z,t,cond))
    m=ModeDenoiser(); cfg=p.PolicyConfig(steps=2,cfg_scale=1.)
    cond=torch.ones(1,2); trace=p.rollout(m,cond,2,cfg)
    p.transition_mean(m,trace.transitions[0].state,cond,0,cfg)
    assert all(training and not dropout for training,dropout,_ in m.seen)
    assert m.seen[-1][-1] and not m.seen[0][-1]


def test_explicit_answer_parser_does_not_turn_words_into_letters():
    from laces_posttrain.data import parse_final_choice
    assert parse_final_choice('The answer is Apple.',10) is None
    assert parse_final_choice('Final answer: correct',10) is None
    assert parse_final_choice('The answer is (J).',10)==9
    assert parse_final_choice('The answer is (K).',10) is None


def test_generic_question_stems_with_different_choices_are_distinct_problems():
    d=module('data')
    a=dict(row(1),question='Which statement is correct?')
    b=dict(row(2),question='Which statement is correct?')
    assert d.question_key(a)!=d.question_key(b)
