from types import SimpleNamespace
import json

import pytest
import torch
from torch import nn

from laces_posttrain.block_runtime import BlockGeneration
from laces_posttrain.prepare_math import prepare_bundle
from laces_posttrain.run_math_block_grpo import evaluation_seed, parse_args, run


class TinyS2(nn.Module):
    def __init__(self):
        super().__init__(); self.w=nn.Parameter(torch.tensor(.04))
    def forward(self,z,t,cond=None):
        return self.w*z + .05*cond[:,None,:]


class FakeNative:
    def __init__(self):
        self.device=torch.device('cpu'); self.horizon=4; self.chunk=2; self.calls={'S0':0,'S1':0,'S2':0}
        self.s2=TinyS2(); self.parent_s2_state={k:v.detach().clone() for k,v in self.s2.state_dict().items()}; self.replay_calls=0
        self.audit={'step':30000,'writer_type':'dynlowrank','latent_dim':2,'horizon':4,'chunk_size':2}
        self.model=SimpleNamespace(named_parameters=lambda: [('trajectory_dit.w',self.s2.w)])
    def enable_s2_training(self): self.s2.requires_grad_(True)
    def close(self): pass
    def _ids(self,text): return torch.tensor([[1,2]]) if 'Final answer' in text else torch.tensor([[3,4,5]])
    def encode_prompt(self,text,seed):
        self.calls['S0']+=1; return torch.tensor([[3,4,5]]),torch.tensor([[.2,.4]])
    def score_tokens(self,prefix,answer,z,raw=False): return torch.tensor(-1.0)
    def generate_blocks(self,prefix,z,*,tokens_per_block,max_blocks,eos_id=None,raw=False,potential_fn=None):
        if not raw: self.calls['S1']+=max_blocks
        total=float(z[:,:max_blocks].sum()); answer='1' if total>=0 else '2'
        ids=list(range(10,10+max_blocks*tokens_per_block)); ends=[tokens_per_block*(i+1) for i in range(max_blocks)]
        potentials=[-.9 + .1*(h+1)*float(z[:,:h+1].sum()) for h in range(max_blocks)]
        return BlockGeneration(ids,f'reasoning\nFinal answer: {answer}',ends,[True]*max_blocks,potentials)
    def score_continuation_after_blocks(self,prefix,z,generated_ids,block_end_offsets,answer_ids,*,upto_block,raw=False):
        self.replay_calls += 1; return torch.tensor(float(z[:,:upto_block+1].sum())*.1)


def gsm(i): return {'question':f'What is {i}+0?','answer':'reasoning\n#### 1'}


def bundle(tmp_path):
    out=tmp_path/'data'
    prepare_bundle([gsm(i) for i in range(12)],[gsm(100),gsm(101)],out,task='gsm8k',source='fixture',revision='r1',seed=2,dev_fraction=.25)
    return out


def args(data,out,*extra):
    base=['--data',str(data),'--ckpt-dir','unused','--output',str(out),'--device','cpu','--diffusion-steps','2','--group-size','3','--max-blocks','4','--tokens-per-block','2','--eval-samples','1','--block-budgets','1','2','4','--save-every','1','--eval-every','1']
    return parse_args([*base,*extra])


def test_preflight_exercises_all_blocks_and_finds_nonflat_credit(tmp_path):
    native=FakeNative(); result=run(args(bundle(tmp_path),tmp_path/'pre','--mode','preflight'),runtime=native)
    assert result['active_blocks']==4 and result['reward_std']>0 and result['block_gradient_norm']>0 and result['ready_for_pilot']
    assert native.replay_calls==0


def test_one_training_update_saves_s2_only_checkpoint_and_three_eval_arms(tmp_path):
    out=tmp_path/'train'; data=bundle(tmp_path); summary=run(args(data,out,'--mode','train','--steps','1'),runtime=FakeNative())
    assert summary['global_step']==1
    saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False); assert set(saved['s2'])=={'w'} and 'optimizer' in saved
    report=json.loads((out/'dev_step_000001.json').read_text()); assert set(report['metrics'])=={'raw_rwkv','parent_laces','current'}
    assert set(report['metrics']['current'])=={'1','2','4'}


def test_resume_continues_same_run_contract(tmp_path):
    data=bundle(tmp_path); out=tmp_path/'resume'
    run(args(data,out,'--mode','train','--steps','1'),runtime=FakeNative())
    summary=run(args(data,out,'--mode','train','--steps','2','--resume',str(out/'latest.pt')),runtime=FakeNative()); assert summary['global_step']==2
    with pytest.raises(ValueError,match='contract'):
        run(args(data,out,'--mode','train','--steps','3','--resume',str(out/'latest.pt'),'--group-size','4'),runtime=FakeNative())


def test_test_split_requires_explicit_acknowledgement(tmp_path):
    data=bundle(tmp_path)
    with pytest.raises(ValueError,match='acknowledge'):
        run(args(data,tmp_path/'eval','--mode','eval','--split','test'),runtime=FakeNative())


def test_eval_reports_reasoning_scaling_budgets_without_using_test_for_selection(tmp_path):
    data=bundle(tmp_path); result=run(args(data,tmp_path/'eval','--mode','eval','--split','dev'),runtime=FakeNative())
    assert result['split']=='dev'
    assert result['selection_policy']=='development checkpoints use fixed max_blocks only; block budgets are reported, not test-selected'
    assert set(result['metrics']['current'])=={'1','2','4'}


def test_eval_samples_one_plan_per_row_and_reuses_it_for_all_budgets(tmp_path, monkeypatch):
    calls=[]
    def sample_once(model,cond,native,cfg,seed):
        calls.append(seed)
        return torch.ones(1,native.horizon,native.audit['latent_dim'])
    monkeypatch.setattr('laces_posttrain.run_math_block_grpo._sample_plan',sample_once)
    result=run(args(bundle(tmp_path),tmp_path/'eval','--mode','eval','--split','dev'),runtime=FakeNative())
    rows=result['metrics']['current']['1']['n']
    assert len(calls)==2*rows  # parent and current; raw RWKV has no latent plan
    assert result['metrics']['current']['1']['n']==result['metrics']['current']['4']['n']


def test_evaluation_seed_reuses_one_trajectory_across_depth_budgets():
    assert evaluation_seed(42,'abc',8,0)==evaluation_seed(42,'abc',8,0)
    assert evaluation_seed(42,'abc',8,0)==evaluation_seed(42,'abc',4,0)
    assert evaluation_seed(42,'abc',8,0)!=evaluation_seed(42,'abc',8,1)
