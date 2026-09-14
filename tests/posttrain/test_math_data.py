import pytest

from laces_posttrain.prepare_math import (
    canonicalize_record,
    prepare_bundle,
    read_evaluation_bundle,
    read_training_bundle,
)


def gsm(i, answer=None):
    return {'question': f'A store has {i} boxes. How many?', 'answer': answer or f'work\n#### {i}'}


def math_row(i, category='algebra'):
    return {'problem': f'Compute {i}+0.', 'solution': rf'Work. \\boxed{{{i}}}', 'type': category, 'level': 'Level 1'}


def test_canonical_rows_retain_rationale_and_verified_answer():
    g=canonicalize_record(gsm(12),'gsm8k',source='openai/gsm8k',revision='abc')
    assert g['answer']=='12' and '#### 12' in g['reference_rationale']
    m=canonicalize_record(math_row(3),'math',source='EleutherAI/hendrycks_math',revision='def')
    assert m['answer']=='3' and m['category']=='algebra' and m['reference_rationale'].endswith(r'\\boxed{3}')
    assert len(g['problem_hash'])==64 and g['source_revision']=='abc'


def test_bundle_partition_is_deterministic_and_test_is_sealed(tmp_path):
    train=[gsm(i) for i in range(20)]; test=[gsm(100+i) for i in range(3)]
    a=prepare_bundle(train,test,tmp_path/'a',task='gsm8k',source='openai/gsm8k',revision='sha1',seed=7,dev_fraction=.2)
    b=prepare_bundle(train,test,tmp_path/'b',task='gsm8k',source='openai/gsm8k',revision='sha1',seed=7,dev_fraction=.2)
    assert a['counts']==b['counts']=={'train':16,'dev':4,'test':3}
    assert (tmp_path/'a'/'train.jsonl').read_text()==(tmp_path/'b'/'train.jsonl').read_text()
    tr,dev,m=read_training_bundle(tmp_path/'a')
    assert {r['problem_hash'] for r in tr}.isdisjoint(r['problem_hash'] for r in dev)
    with pytest.raises(ValueError,match='acknowledge'):
        read_evaluation_bundle(tmp_path/'a','test')
    sealed,_=read_evaluation_bundle(tmp_path/'a','test',acknowledge_test=True)
    assert len(sealed)==3 and m['source_revision']=='sha1'


def test_cross_split_duplicate_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='overlap'):
        prepare_bundle([gsm(1),gsm(2)],[gsm(2)],tmp_path,task='gsm8k',source='x',revision='r')


def test_math_stratified_dev_keeps_each_category(tmp_path):
    train=[math_row(i,'algebra') for i in range(10)]+[math_row(100+i,'geometry') for i in range(10)]
    test=[math_row(1000,'algebra')]
    manifest=prepare_bundle(train,test,tmp_path,task='math',source='math',revision='r',seed=3,dev_fraction=.2)
    _,dev,_=read_training_bundle(tmp_path)
    assert {r['category'] for r in dev}=={'algebra','geometry'}
    assert manifest['counts']['dev']==4


def test_manifest_tampering_is_rejected(tmp_path):
    prepare_bundle([gsm(i) for i in range(6)],[gsm(20)],tmp_path,task='gsm8k',source='x',revision='r')
    with (tmp_path/'train.jsonl').open('a') as f: f.write('{}\n')
    with pytest.raises(ValueError,match='hash'):
        read_training_bundle(tmp_path)


def test_unverifiable_math_solution_is_rejected():
    row={'problem':'prove something','solution':'There is no explicitly marked final result.','type':'geometry'}
    with pytest.raises(ValueError,match='answer'):
        canonicalize_record(row,'math',source='math',revision='r')
