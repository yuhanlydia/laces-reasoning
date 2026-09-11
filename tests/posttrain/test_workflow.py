import importlib
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from test_native import model, checkpoint, CharTokenizer  # local pytest fixture
from test_contracts import row


def cli():
    assert importlib.util.find_spec('laces_posttrain.run') is not None,'MMLU-Pro runnable workflow missing'
    return importlib.import_module('laces_posttrain.run')


def bundle(tmp_path):
    from laces_posttrain.data import prepare_bundle
    p=tmp_path/'data'; prepare_bundle([row(i,n=4) for i in range(5)],[row(99,n=4)],p)
    return p


def test_cpu_preflight_train_resume_and_eval_reuses_s2(model,tmp_path):
    c=cli()
    from laces_posttrain.native import NativeLACES
    data=bundle(tmp_path); output=tmp_path/'run'
    common=['--data',str(data),'--ckpt-dir','unused','--device','cpu','--output',str(output),
            '--diffusion-steps','2','--cfg-scale','1','--group-size','2','--native-control-steps','2',
            '--train-limit','2','--dev-limit','1','--save-every','1','--eval-every','1']
    runtime=NativeLACES(model,CharTokenizer(),checkpoint(model))
    args=c.parse_args(['--mode','train','--objective','distill','--steps','1',*common])
    out=c.run(args,runtime=runtime)
    assert out['global_step']==1
    saved=torch.load(output/'latest.pt',weights_only=False)
    assert saved['schema']=='laces_mmlu_s2_v1'
    assert not any('rwkv_model' in k or 's1_' in k for k in saved['s2'])
    assert saved['objective']=='distill' and (output/'metrics.jsonl').is_file()
    # The resume contract is checked independently of reloading the base in this injected test.
    args2=c.parse_args(['--mode','train','--objective','distill','--steps','2','--resume',str(output/'latest.pt'),*common])
    out2=c.run(args2,runtime=runtime)
    assert out2['global_step']==2
    test_args=c.parse_args(['--mode','eval','--split','test','--s2-checkpoint',str(output/'latest.pt'),*common])
    with pytest.raises(ValueError,match='acknowledge'): c.run(test_args,runtime=runtime)


def test_warmstart_rejects_legacy_reasoner_checkpoint(model,tmp_path):
    c=cli(); from laces_posttrain.native import NativeLACES
    data=bundle(tmp_path); bad=tmp_path/'old.pt'; torch.save({'reasoner':{}},bad)
    args=c.parse_args(['--mode','train','--data',str(data),'--ckpt-dir','unused','--output',str(tmp_path/'run'),
        '--device','cpu','--init-s2',str(bad),'--steps','1'])
    with pytest.raises(ValueError,match='S2 checkpoint'):
        c.run(args,runtime=NativeLACES(model,CharTokenizer(),checkpoint(model)))


def test_train_cannot_choose_test_split(tmp_path):
    c=cli()
    with pytest.raises(ValueError,match='test'):
        c.parse_args(['--mode','train','--split','test','--data','x','--ckpt-dir','x'])


def test_distillation_is_not_claimed_teacher_when_using_candidates():
    c=cli(); a=c.parse_args(['--data','x','--ckpt-dir','x','--objective','distill'])
    assert a.distill_source=='candidates'


def test_evaluation_reports_original_parent_and_rejects_sampler_mismatch(model,tmp_path):
    import copy
    c=cli(); from laces_posttrain.native import NativeLACES
    data=bundle(tmp_path); output=tmp_path/'run'; original=copy.deepcopy(model)
    parent_ck=checkpoint(model)
    common=['--data',str(data),'--ckpt-dir','unused','--device','cpu',
            '--diffusion-steps','2','--cfg-scale','1','--group-size','2',
            '--dev-limit','1','--train-limit','1','--lr','0.0002']
    c.run(c.parse_args(['--mode','train','--steps','1','--output',str(output),*common]),
          runtime=NativeLACES(model,CharTokenizer(),parent_ck))
    result=c.run(c.parse_args(['--mode','eval','--output',str(tmp_path/'eval'),
                             '--s2-checkpoint',str(output/'latest.pt'),*common]),
                 runtime=NativeLACES(copy.deepcopy(original),CharTokenizer(),parent_ck))
    assert set(result['metrics'])=={'current','raw_rwkv','parent_matched'}
    assert all('encoding_seconds' in r for r in result['samples']['current'])
    bad=c.parse_args(['--mode','eval','--output',str(tmp_path/'bad'),
                     '--s2-checkpoint',str(output/'latest.pt'),*common,'--diffusion-steps','3'])
    with pytest.raises(ValueError,match='evaluation settings'):
        c.run(bad,runtime=NativeLACES(copy.deepcopy(original),CharTokenizer(),parent_ck))


def test_fresh_training_refuses_to_overwrite_existing_run(model,tmp_path):
    c=cli(); from laces_posttrain.native import NativeLACES
    data=bundle(tmp_path); output=tmp_path/'run'; output.mkdir()
    (output/'latest.pt').write_bytes(b'previous experiment')
    a=c.parse_args(['--data',str(data),'--ckpt-dir','unused','--output',str(output),
                  '--mode','train','--steps','1','--device','cpu'])
    with pytest.raises(ValueError,match='already contains'):
        c.run(a,runtime=NativeLACES(model,CharTokenizer(),checkpoint(model)))
    assert (output/'latest.pt').read_bytes()==b'previous experiment'


def test_split_resume_equals_continuous_training(model,tmp_path):
    import copy
    c=cli(); from laces_posttrain.native import NativeLACES
    data=bundle(tmp_path); initial=copy.deepcopy(model); ck=checkpoint(model)
    common=['--data',str(data),'--ckpt-dir','unused','--device','cpu',
            '--diffusion-steps','2','--cfg-scale','1','--group-size','2',
            '--train-limit','2','--dev-limit','1','--eval-every','1','--save-every','1']
    for name,steps in [('continuous',2),('split',1)]:
        a=c.parse_args(['--mode','train','--steps',str(steps),'--output',str(tmp_path/name),*common])
        c.run(a,runtime=NativeLACES(copy.deepcopy(initial),CharTokenizer(),ck))
    c.run(c.parse_args(['--mode','train','--steps','2','--resume',str(tmp_path/'split/latest.pt'),
            '--output',str(tmp_path/'split'),*common]),
          runtime=NativeLACES(copy.deepcopy(initial),CharTokenizer(),ck))
    full=torch.load(tmp_path/'continuous/latest.pt',weights_only=False)
    split=torch.load(tmp_path/'split/latest.pt',weights_only=False)
    assert all(torch.equal(full['s2'][k],split['s2'][k]) for k in full['s2'])
    assert full['python_rng']==split['python_rng']
    assert torch.equal(full['torch_rng'],split['torch_rng'])


def test_native_mmlu_grpo_updates_only_existing_s2(model,tmp_path):
    c=cli(); from laces_posttrain.native import NativeLACES
    data=bundle(tmp_path); out=tmp_path/'grpo'; ck=checkpoint(model)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    a=c.parse_args(['--mode','train','--objective','grpo','--steps','1','--data',str(data),
        '--ckpt-dir','unused','--device','cpu','--output',str(out),'--diffusion-steps','2',
        '--group-size','4','--cfg-scale','1','--dev-limit','1','--train-limit','2','--kl-coef','0'])
    c.run(a,runtime=NativeLACES(model,CharTokenizer(),ck))
    metrics=json.loads((out/'metrics.jsonl').read_text().splitlines()[0])
    assert metrics['reward_std']>0 and metrics['grad_norm']>0 and not metrics['skipped_flat']
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
    assert changed and all(k.startswith('trajectory_dit.') for k in changed)


def test_rationale_mode_validates_all_teacher_records_before_training(model,tmp_path):
    c=cli(); from laces_posttrain.native import NativeLACES
    data=bundle(tmp_path); out=tmp_path/'rationale'
    args=c.parse_args(['--data',str(data),'--ckpt-dir','unused','--mode','train',
        '--objective','distill','--distill-source','rationale','--steps','1','--output',str(out),'--device','cpu'])
    native=NativeLACES(model,CharTokenizer(),checkpoint(model))
    with pytest.raises(ValueError,match='[Tt]eacher'):
        c.run(args,runtime=native)
    assert native.calls['S0']==0, 'teacher records must be validated before encoding/training starts'
    assert not (out/'metrics.jsonl').exists()


def test_real_tiny_preflight_exercises_s0_s1_s2_and_score_gradient(model,tmp_path):
    c=cli(); from laces_posttrain.native import NativeLACES
    data=bundle(tmp_path); out=tmp_path/'preflight'
    a=c.parse_args(['--data',str(data),'--ckpt-dir','unused','--mode','preflight',
        '--device','cpu','--output',str(out),'--diffusion-steps','2','--native-control-steps','2',
        '--group-size','4','--cfg-scale','1'])
    result=c.run(a,runtime=NativeLACES(model,CharTokenizer(),checkpoint(model)))
    assert all(result['calls'][k]>0 for k in ('S0','S1','S2'))
    assert result['s2_score_gradient_norm']>0
    assert result['repeat_score_max_diff']==0
    assert (out/'preflight.json').is_file()
    assert not (out/'latest.pt').exists()
