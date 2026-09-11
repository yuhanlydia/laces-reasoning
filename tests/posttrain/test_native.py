import importlib
import importlib.util
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from models.state_hijacking_dit import StateInjectionDiTRELAY


def api():
    assert importlib.util.find_spec('laces_posttrain.native') is not None, 'native pretrained runtime is missing'
    return importlib.import_module('laces_posttrain.native')


class TinyRWKV(nn.Module):
    def __init__(self):
        super().__init__(); self.config=SimpleNamespace(num_hidden_layers=2,hidden_size=8,head_dim=4)
        self.emb=nn.Embedding(128,8); self.head=nn.Linear(8,128)
    def forward(self,input_ids,past_key_values=None,**kwargs):
        batch=input_ids.shape[0]
        if past_key_values is None:
            past_key_values=SimpleNamespace(states=[dict(recurrent_state=torch.zeros(batch,2,4,4),
                conv_state=torch.zeros(batch,8),ffn_state=torch.zeros(batch,8)) for _ in range(2)],_seen_tokens=0)
        hidden=[]
        for token in input_ids.unbind(1):
            x=self.emb(token)
            for st in past_key_values.states:
                h=x.reshape(batch,2,4)
                st['recurrent_state']=.7*st['recurrent_state']+.05*h.unsqueeze(-1)*h.unsqueeze(-2)
                x=x+.1*st['recurrent_state'].mean(-1).reshape(batch,8)+.03*st['conv_state']
                st['conv_state']=x; st['ffn_state']=.1*x
            hidden.append(x); past_key_values._seen_tokens+=1
        h=torch.stack(hidden,1)
        return SimpleNamespace(logits=self.head(h),hidden_states=(h,),past_key_values=past_key_values)


class CharTokenizer:
    eos_token_id=0
    def __call__(self,text,return_tensors=None,add_special_tokens=False):
        ids=[ord(x)%128 for x in text]
        return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)
    def decode(self,ids,**kwargs): return ''.join(chr(int(x)) for x in ids)


@pytest.fixture(params=[("dit","mlp"),("birwkv","variational")])
def model(request):
    torch.manual_seed(9)
    cfg=OmegaConf.create(dict(s1_writer_type='dynlowrank',s1_rank=2,s1_dyn_hidden=12,
        trajectory_mode=True,trajectory_horizon=3,trajectory_chunk_size=4,
        trajectory_s1_mode='independent',trajectory_state_blend=.7,trajectory_denoiser_type=request.param[0]))
    m=StateInjectionDiTRELAY(cfg,TinyRWKV(),vocab_size=128,latent_dim=4,n_basis=2,
        dit_hidden=8,dit_depth=1,dit_num_heads=2,dit_num_tokens=2,encoder_type=request.param[1])
    m.state_scale.data.fill_(1.); m._gen_type='ddpm'
    return m.eval()


def checkpoint(m):
    return {'step':30000,'trainable_state':{k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('rwkv_model.')}}


def test_exact_s0_s1_s2_checkpoint_coverage(model):
    n=api(); ck=checkpoint(model); del ck['trainable_state']['s1_u_head.weight']
    with pytest.raises(ValueError,match='s1_u_head.weight'): n.NativeLACES(model,CharTokenizer(),ck)


def test_only_original_s2_unfrozen_and_existing_writer_used(model):
    n=api(); run=n.NativeLACES(model,CharTokenizer(),checkpoint(model)); run.enable_s2_training()
    names={k for k,p in model.named_parameters() if p.requires_grad}
    assert names and all(k.startswith('trajectory_dit.') for k in names)
    assert run.s2 is model.trajectory_dit
    ids,c=run.encode_prompt('Q? Answer:',seed=1)
    assert c.shape==(1,4)
    from laces_posttrain.policy import PolicyConfig, rollout
    z=rollout(run.s2,c,run.horizon,PolicyConfig(steps=2,cfg_scale=1.)).final
    s=run.score_options(ids,z,10)
    assert s.shape==(10,) and torch.isfinite(s).all()
    assert all(run.calls[k]>0 for k in ('S0','S1','S2'))
    assert all(p.grad is None for k,p in model.named_parameters() if not k.startswith('trajectory_dit.'))
    run.close()


def test_blend_zero_is_exact_raw_control(model):
    n=api(); run=n.NativeLACES(model,CharTokenizer(),checkpoint(model),blend=0)
    ids,c=run.encode_prompt('Q? Answer:',seed=1); z=torch.randn(1,3,4)
    torch.testing.assert_close(run.score_options(ids,z,4),run.score_options(ids,z,4,raw=True),rtol=0,atol=0)


def test_first_answer_token_sees_write_and_input_is_consumed_once(model):
    n=api(); run=n.NativeLACES(model,CharTokenizer(),checkpoint(model),blend=1)
    ids,c=run.encode_prompt('Q? Answer:',seed=1)
    z=torch.randn(1,3,4); calls=[]
    h=model.rwkv_model.register_forward_pre_hook(lambda m,a,k:calls.extend(k['input_ids'][0].tolist()),with_kwargs=True)
    p1=run.score_tokens(ids,torch.tensor([[65]]),z)
    assert calls==ids[0].tolist(); h.remove()
    p2=run.score_tokens(ids,torch.tensor([[65]]),z*5)
    assert not torch.equal(p1,p2)


def test_multitoken_option_score_uses_full_continuation(model):
    n=api(); run=n.NativeLACES(model,CharTokenizer(),checkpoint(model))
    ids,c=run.encode_prompt('Q? Answer:',seed=1); z=torch.randn(1,3,4)
    scores=run.score_options(ids,z,4)
    expected=run.score_tokens(ids,torch.tensor([[32,65]]),z)
    torch.testing.assert_close(scores[0],expected)


def test_long_teacher_not_silently_truncated(model):
    n=api(); run=n.NativeLACES(model,CharTokenizer(),checkpoint(model))
    with pytest.raises(ValueError,match='horizon'): run.encode_teacher('a'*20,seed=1)
    z,mask=run.encode_teacher('abcde',seed=1)
    assert z.shape==(1,3,4) and mask.tolist()==[[1.,1.,0.]]


def test_distill_changes_s2_but_never_frozen_modules(model):
    from laces_posttrain.policy import distill_update
    n=api(); run=n.NativeLACES(model,CharTokenizer(),checkpoint(model)); run.enable_s2_training()
    before={k:p.detach().clone() for k,p in model.named_parameters()}
    ids,c=run.encode_prompt('Q? Answer:',seed=1)
    opt=torch.optim.AdamW(run.s2.parameters(),lr=.001)
    distill_update(run.s2,opt,c,[torch.randn(1,3,4)])
    changed={k for k,p in model.named_parameters() if not torch.equal(before[k],p)}
    assert changed and all(k.startswith('trajectory_dit.') for k in changed)
