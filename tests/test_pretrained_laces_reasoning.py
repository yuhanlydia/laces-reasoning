"""CPU contract tests using the real LACES S0, S1, S2 and a tiny recurrent renderer."""
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from models.state_hijacking_dit import StateInjectionDiTRELAY


def api():
    name = 'models.laces_latent_refiner'
    assert importlib.util.find_spec(name) is not None, 'pretrained LACES reasoning integration is missing'
    return importlib.import_module(name)


class TinyRWKV(nn.Module):
    """Small differentiable recurrent LM; no FLA/CUDA or network downloads."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=8, head_dim=4)
        self.embedding = nn.Embedding(19, 8)
        self.lm_head = nn.Linear(8, 19)

    def forward(self, input_ids, past_key_values=None, **kwargs):
        batch = input_ids.shape[0]
        if past_key_values is None:
            past_key_values = SimpleNamespace(states=[{
                'recurrent_state': torch.zeros(batch, 2, 4, 4),
                'conv_state': torch.zeros(batch, 8),
                'ffn_state': torch.zeros(batch, 8),
            } for _ in range(2)], _seen_tokens=0)
        hidden = []
        for token in input_ids.unbind(1):
            x = self.embedding(token)
            for state in past_key_values.states:
                h = x.reshape(batch, 2, 4)
                state['recurrent_state'] = .7 * state['recurrent_state'] + h.unsqueeze(-1) * h.unsqueeze(-2) * .05
                x = x + .1 * state['recurrent_state'].mean(-1).reshape(batch, 8) + .03 * state['conv_state']
                state['conv_state'] = x
                state['ffn_state'] = .1 * x
            hidden.append(x)
            past_key_values._seen_tokens += 1
        hidden = torch.stack(hidden, 1)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,), past_key_values=past_key_values)


@pytest.fixture
def laces():
    torch.manual_seed(9)
    config = OmegaConf.create(dict(s1_writer_type='dynlowrank', s1_rank=2, s1_dyn_hidden=12,
        trajectory_mode=True, trajectory_horizon=3, trajectory_chunk_size=2,
        trajectory_s1_mode='independent', trajectory_state_blend=.7, trajectory_denoiser_type='dit'))
    model = StateInjectionDiTRELAY(config, TinyRWKV(), vocab_size=19, latent_dim=4,
        n_basis=2, dit_hidden=8, dit_depth=1, dit_num_heads=2, dit_num_tokens=2, encoder_type='mlp')
    model.state_scale.data.fill_(1.)
    model._gen_type = 'ddpm'
    return model.eval()


def checkpoint(model):
    return {'step': 30000, 'trainable_state': {k: v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('rwkv_model.')}}


def runtime(model, **kwargs):
    return api().LACESRuntime(model, checkpoint(model), expected_step=30000,
        plan_steps=2, cfg_scale=1., **kwargs)


def refiner():
    return api().LatentTrajectoryRefiner(latent_dim=4, hidden_dim=8, width=8, attention_heads=2)


PREFIX = torch.tensor([[2, 3, 4, 5]])
ANSWER = torch.tensor([[6, 7, 8]])


def test_loader_is_importable_without_optional_transformers():
    # Sampling helpers should not require transformers until a real checkpoint is loaded.
    import scripts.eval.relay_utils


def test_strict_component_coverage_detects_missing_s1(laces):
    ck = checkpoint(laces)
    del ck['trainable_state']['s1_u_head.weight']
    with pytest.raises(ValueError, match='s1_u_head.weight'):
        api().validate_pretrained_laces(laces, ck, expected_step=30000)


def test_strict_component_validation_detects_wrong_loaded_weights(laces):
    ck = checkpoint(laces)
    ck['trainable_state']['s1_v_head.bias'] += 1
    with pytest.raises(ValueError, match='s1_v_head.bias'):
        api().validate_pretrained_laces(laces, ck, expected_step=30000)


def test_wrong_training_step_is_not_silently_accepted(laces):
    with pytest.raises(ValueError, match='step'):
        api().validate_pretrained_laces(laces, checkpoint(laces), expected_step=10000)


def test_prepare_really_calls_s0_s2_and_writer_is_existing_s1(laces):
    run = runtime(laces)
    z0, features, zp = run.prepare(PREFIX, seed=12)
    assert z0.shape == (1, 3, 4)
    assert features.shape == (1, 4, 8)
    assert zp.shape == (1, 4)
    run.score(PREFIX, ANSWER, z0)
    assert all(run.calls[group] > 0 for group in ('S0', 'S1', 'S2'))
    assert run.model is laces
    assert run.audit['writer_type'] == 'dynlowrank'
    assert all(not p.requires_grad for p in laces.parameters())


def test_zero_steps_and_untrained_refiner_preserve_native_coordinates():
    model = refiner()
    z = torch.randn(1,3,4) * 9 + 4
    h = torch.randn(1,5,8)
    zp = torch.randn(1,4)
    assert torch.equal(model(z,h,zp,steps=0), z)
    assert torch.equal(model(z,h,zp,steps=4), z)
    assert not any('writer' in name or 'z0_head' in name for name,_ in model.named_parameters())


def test_prepare_seed_reproducible_and_does_not_pollute_training_rng(laces):
    run = runtime(laces)
    before = torch.get_rng_state().clone()
    z1,_,_ = run.prepare(PREFIX, seed=12)
    assert torch.equal(before, torch.get_rng_state())
    z2,_,_ = run.prepare(PREFIX, seed=12)
    assert torch.equal(z1,z2)


def test_zero_refinement_matches_full_laces_logits(laces):
    run = runtime(laces)
    z0,h,zp = run.prepare(PREFIX,seed=7)
    z = refiner()(z0,h,zp,steps=0)
    direct = run.score(PREFIX,ANSWER,z0)
    refined = run.score(PREFIX,ANSWER,z)
    assert torch.equal(direct,refined)


def test_aligned_first_answer_token_has_latent_gradient(laces):
    run = runtime(laces)
    z,_,_ = run.prepare(PREFIX,seed=8)
    z = z.detach().requires_grad_()
    loss = -run.score(PREFIX,ANSWER[:,:1],z)
    grad = torch.autograd.grad(loss,z)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0


def test_legacy_first_token_blindness_is_explicit(laces):
    run = runtime(laces,protocol='legacy')
    z,_,_ = run.prepare(PREFIX,seed=8)
    assert torch.equal(run.score(PREFIX,ANSWER[:,:1],z),run.score(PREFIX,ANSWER[:,:1],z+5))


def test_zero_blend_matches_raw_without_auxiliary_cache_resets(laces):
    run = runtime(laces,blend=0.)
    z,_,_ = run.prepare(PREFIX,seed=8)
    score = run.score(PREFIX,ANSWER,z)
    raw = run.score(PREFIX,ANSWER,z,raw=True)
    assert torch.equal(score,raw)


def test_teacher_forcing_agrees_with_generation_token_scores(laces):
    run = runtime(laces)
    z,_,_ = run.prepare(PREFIX,seed=8)
    ids,logps = run.generate(PREFIX,z,max_new_tokens=5)
    score = run.score(PREFIX,torch.tensor([ids]),z)
    assert score.item() == pytest.approx(sum(logps)/len(logps),abs=1e-6)


def test_refiner_receives_ce_gradient_but_pretrained_weights_stay_fixed(laces):
    run = runtime(laces)
    z0,h,zp = run.prepare(PREFIX,seed=8)
    before = {k:v.clone() for k,v in laces.state_dict().items()}
    f = refiner()
    opt = torch.optim.AdamW(f.parameters(),lr=.01)
    z = f(z0,h,zp,steps=2)
    loss = -run.score(PREFIX,ANSWER,z)
    loss.backward()
    assert f.delta_head.weight.grad.norm() > 0
    assert all(p.grad is None for p in laces.parameters())
    opt.step()
    assert all(torch.equal(v,laces.state_dict()[k]) for k,v in before.items())


def test_incomplete_answer_not_silently_truncated(laces):
    run = runtime(laces)
    z,_,_ = run.prepare(PREFIX,seed=8)
    with pytest.raises(ValueError,match='horizon'):
        run.score(PREFIX,torch.tensor([[6]*7]),z)


def test_policy_loss_uses_detached_actions_and_reward_signal():
    mean = torch.zeros(1,1,2,requires_grad=True)
    actions = torch.tensor([[[[1.,0.]]],[[[-1.,0.]]]])
    rewards = torch.tensor([1.,-1.])
    loss = api().answer_policy_loss(mean,actions,torch.ones_like(mean),rewards)
    loss.backward()
    assert mean.grad[0,0,0] < 0  # gradient descent moves mean toward the better action
    assert mean.grad[0,0,1] == 0


def test_flat_policy_reward_fails_instead_of_claiming_training():
    with pytest.raises(ValueError,match='reward'):
        api().answer_policy_loss(torch.zeros(1,1,2,requires_grad=True),
            torch.randn(4,1,1,2),torch.ones(1,1,2),torch.zeros(4))


def trainer_api():
    name = 'scripts.eval.train_laces_reasoner'
    assert importlib.util.find_spec(name) is not None, 'full-LACES training entry is missing'
    return importlib.import_module(name)


def test_new_trainer_rejects_legacy_protocol_for_training():
    trainer = trainer_api()
    with pytest.raises(ValueError, match='legacy'):
        trainer.validate_args(trainer.parse_args(['--ckpt_dir','unused','--protocol','legacy','--mode','train']))


def test_split_overlap_is_rejected():
    trainer = trainer_api()
    record = {'prefix':'same evidence and question', 'answer':'alpha'}
    with pytest.raises(ValueError, match='overlap'):
        trainer.validate_splits({'train':[record],'validation':[record], 'test':[]})


def test_standalone_checkpoint_cannot_be_resumed_silently(tmp_path):
    trainer = trainer_api()
    path = tmp_path / 'old.pt'
    torch.save({'reasoner':{},'model_config':{}},path)
    with pytest.raises(ValueError,match='standalone'):
        trainer.load_refiner_checkpoint(path, {'active_fingerprint':'new'})


def test_resume_rejects_other_pretrained_writer(tmp_path):
    trainer = trainer_api()
    path = tmp_path / 'different.pt'
    torch.save({'format':'laces_pretrained_refiner_v1', 'base':{'active_fingerprint':'old'}},path)
    with pytest.raises(ValueError,match='fingerprint'):
        trainer.load_refiner_checkpoint(path, {'active_fingerprint':'new'})


def test_legacy_runtime_replays_existing_full_laces_sampler(laces):
    from scripts.eval.sample_prefix_suffix_trajectory_cfg import generate, encode_prefix
    run = runtime(laces,protocol='legacy',blend=.7)
    z,_,_ = run.prepare(PREFIX,seed=8)
    class Tokenizer:
        all_special_tokens = []
        def decode(self, ids, **kwargs):
            return ','.join(map(str, ids))
    tok = Tokenizer()
    args = SimpleNamespace(max_new_tokens=5,temperature=0.,top_k=0,top_p=1.,repetition_penalty=1.)
    _,cache,logits = encode_prefix(laces,PREFIX,torch.ones_like(PREFIX))
    text,_ = generate(laces,tok,PREFIX,torch.ones_like(PREFIX),cache,logits,z,args)
    ours,_ = run.generate(PREFIX,z,max_new_tokens=5)
    assert text == tok.decode(PREFIX[0].tolist()+ours)


@pytest.mark.parametrize('objective', ['answer_pg', 'answer_ce'])
def test_real_training_entry_updates_refiner_and_can_resume_eval(laces, tmp_path, monkeypatch, objective):
    import json
    trainer = trainer_api()
    class Tokenizer:
        eos_token_id = None
        def __call__(self, text, **kwargs):
            ids = [6,7] if kwargs.get('add_special_tokens') is False else [2,3,4,5]
            return SimpleNamespace(input_ids=torch.tensor([ids]))
        def decode(self, ids, **kwargs):
            return ' '.join(f'word{x}' for x in ids)
    initial = {k:v.clone() for k,v in laces.state_dict().items()}
    ck = checkpoint(laces)
    monkeypatch.setattr(trainer,'load_relay_model',lambda *a,**kw:(laces,laces.rwkv_model,Tokenizer(),ck,None))
    paths = []
    for split in ['train','validation','test']:
        path = tmp_path / (split+'.jsonl')
        path.write_text(json.dumps({'prefix':split+' evidence. Question: answer?', 'answer':'word6'})+'\n')
        paths += ['--'+split+'_jsonl',str(path)]
    common = ['--objective',objective,'--ckpt_dir','unused','--device','cpu','--plan_steps','2','--max_new_tokens','3',
        '--train_depths','1','--eval_depths','0','1','--width','8','--epochs','1',
        '--output_dir',str(tmp_path/'out'),'--cache_dir',str(tmp_path/'cache'),*paths]
    summary = trainer.run(trainer.parse_args([*common,'--mode','train']))
    saved = tmp_path/'out'/'refiner_last.pt'
    payload = torch.load(saved,weights_only=True)
    assert payload['epochs_completed'] == 1 and payload['global_step'] == 1
    assert payload['refiner']['delta_head.weight'].abs().sum() > 0
    assert not any('writer' in key for key in payload['refiner'])
    assert all(torch.equal(v,laces.state_dict()[k]) for k,v in initial.items())
    assert set(summary['test']['metrics']) == {'raw_rwkv','laces_R0','laces_R1'}
    replay = trainer.run(trainer.parse_args([*common,'--mode','eval','--resume',str(saved)]))
    assert replay['test']['metrics'] == summary['test']['metrics']
