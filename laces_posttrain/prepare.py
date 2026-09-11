"""Prepare a leakage-guarded MMLU-Pro bundle; never train on official test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .data import prepare_bundle, read_jsonl


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True); p.add_argument('--revision',default='main')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--official-validation-jsonl'); p.add_argument('--official-test-jsonl')
    p.add_argument('--external-train-jsonl'); p.add_argument('--source-id')
    a=p.parse_args(argv)
    if bool(a.official_validation_jsonl)!=bool(a.official_test_jsonl):
        p.error('Provide both local official split files, or neither')
    if a.official_validation_jsonl:
        validation=read_jsonl(a.official_validation_jsonl); test=read_jsonl(a.official_test_jsonl)
        revision='local:'+a.revision
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision=HfApi().dataset_info('TIGER-Lab/MMLU-Pro',revision=a.revision).sha
        ds=load_dataset('TIGER-Lab/MMLU-Pro',revision=revision)
        if 'validation' not in ds or 'test' not in ds: raise ValueError('Unexpected official splits')
        validation=list(ds['validation']); test=list(ds['test'])
    result=prepare_bundle(validation,test,a.output,seed=a.seed,revision=revision,
        external_train=read_jsonl(a.external_train_jsonl) if a.external_train_jsonl else None,source_id=a.source_id)
    print(json.dumps(result,indent=2))
    print('Official test is sealed. Default pilot adapts validation and is NOT standard 5-shot evaluation.')


if __name__=='__main__': main()
