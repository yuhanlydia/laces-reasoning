from types import SimpleNamespace

import torch

from laces_posttrain.block_runtime import generate_blocks, score_answer_from_cache, score_continuation_after_blocks
from laces_posttrain.native import NativeLACES


class ToyTokenizer:
    eos_token_id = 9
    def decode(self, ids, skip_special_tokens=False): return ' '.join(map(str, ids))


class ToyRWKV:
    def __init__(self, events, scripted): self.events=events; self.scripted=list(scripted); self.pos=0
    def __call__(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is None: past_key_values=SimpleNamespace(counter=0)
        for tok in input_ids[0].tolist():
            self.events.append(('consume', int(tok), past_key_values.counter)); past_key_values.counter += 1
        logits=torch.full((1,input_ids.shape[1],12),-1000.)
        next_id=self.scripted[min(self.pos,len(self.scripted)-1)]
        if input_ids.shape[1]==1: self.pos+=1
        logits[0,-1,next_id]=10.
        return SimpleNamespace(past_key_values=past_key_values,logits=logits)


class ToyModel:
    latent_dim=1; s1_writer_type='dynlowrank'
    def __init__(self, events, scripted):
        self.events=events; self.rwkv_model=ToyRWKV(events,scripted); self.s1_trunk=torch.nn.Linear(1,1)
    def predict_states(self,z):
        value=float(z[0,0]); self.events.append(('predict',value)); return [torch.tensor([value])]
    def blend_into_cache(self,cache,states,blend):
        self.events.append(('inject',float(states[0][0]),cache.counter,blend)); return cache
    def inject_into_cache(self,cache,states):
        self.events.append(('inject',float(states[0][0]),cache.counter,1.0)); return cache


def runtime(scripted):
    events=[]; native=object.__new__(NativeLACES)
    native.model=ToyModel(events,scripted); native.tokenizer=ToyTokenizer(); native.device=torch.device('cpu')
    native.horizon=3; native.chunk=2; native.blend=.7
    return native,events


def test_block_h_is_injected_before_its_boundary_anchor_and_cache_is_carried():
    native,events=runtime([4,5,6,7,8,8,8]); prefix=torch.tensor([[1,2,3]]); z=torch.tensor([[[1.],[2.],[3.]]])
    result=generate_blocks(native,prefix,z,tokens_per_block=2,max_blocks=3,eos_id=99)
    assert result.token_ids==[4,5,6,7,8,8] and result.block_end_offsets==[2,4,6] and result.active_block_mask==[True,True,True]
    injects=[e for e in events if e[0]=='inject']; assert [e[1] for e in injects]==[1.,2.,3.]
    first_inject=events.index(injects[0]); first_boundary_consume=next(i for i,e in enumerate(events) if e[:2]==('consume',3))
    assert first_inject < first_boundary_consume and [e[2] for e in injects]==[2,4,6]


def test_eos_stops_generation_and_masks_later_blocks():
    native,events=runtime([4,5,6,9,8,8])
    result=generate_blocks(native,torch.tensor([[1,2,3]]),torch.tensor([[[1.],[2.],[3.]]]),tokens_per_block=2,max_blocks=3,eos_id=9)
    assert result.token_ids==[4,5,6,9] and result.block_end_offsets==[2,4] and result.active_block_mask==[True,True,False]
    assert [e[1] for e in events if e[0]=='inject']==[1.,2.]


def test_raw_block_generation_never_injects_latent_state():
    native,events=runtime([4,5,6,7])
    result=generate_blocks(native,torch.tensor([[1,2,3]]),torch.tensor([[[1.],[2.],[3.]]]),tokens_per_block=2,max_blocks=2,eos_id=99,raw=True)
    assert result.token_ids==[4,5,6,7] and not any(e[0]=='inject' for e in events)


def test_potential_callback_runs_after_each_active_block():
    native,_=runtime([4,5,6,7]); seen=[]
    def potential(cache, pending, tokens, block_index):
        seen.append((cache.counter, int(pending.item()), list(tokens), block_index)); return float(sum(tokens))
    result=generate_blocks(native,torch.tensor([[1,2,3]]),torch.tensor([[[1.],[2.],[3.]]]),tokens_per_block=2,max_blocks=2,eos_id=99,potential_fn=potential)
    assert seen==[(4,5,[4,5],0),(6,7,[4,5,6,7],1)] and result.potentials==[9.,22.]


def test_score_continuation_after_blocks_replays_only_completed_blocks():
    native,events=runtime([4,5,6,7,7,7,7,7,7,7]); prefix=torch.tensor([[1,2,3]]); z=torch.tensor([[[1.],[2.],[3.]]])
    score=score_continuation_after_blocks(native,prefix,z,[4,5],[2],torch.tensor([[7]]),upto_block=0)
    assert torch.isfinite(score) and [e[1] for e in events if e[0]=='inject']==[1.]


def test_score_answer_from_boundary_cache_consumes_pending_once():
    native,events=runtime([4,5,7,8,8]); prefix=torch.tensor([[1,2,3]]); z=torch.tensor([[[1.],[2.],[3.]]]); captured=[]
    def potential(cache,pending,tokens,block_index):
        captured.append(float(score_answer_from_cache(native,cache,pending,torch.tensor([[7]])))); return captured[-1]
    generate_blocks(native,prefix,z,tokens_per_block=2,max_blocks=1,eos_id=99,potential_fn=potential)
    assert len(captured)==1 and torch.isfinite(torch.tensor(captured[0])) and any(e[:2]==('consume',5) for e in events)
