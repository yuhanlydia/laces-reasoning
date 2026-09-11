"""MMLU-Pro data contracts. The official test is never a training split."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

LETTERS = 'ABCDEFGHIJ'
SCHEMA = 'laces_mmlu_pro_v1'


def _norm(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def question_key(row: dict) -> str:
    # Option order and gold answer do not change identity. Choices ARE part of a
    # problem: generic stems such as 'Which statement is correct?' can repeat.
    canonical=[_norm(row['question']),sorted(_norm(x) for x in row['options'])]
    return hashlib.sha256(json.dumps(canonical,ensure_ascii=False).encode()).hexdigest()


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path | str) -> list[dict]:
    result=[]
    with Path(path).open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if line.strip():
                obj=json.loads(line)
                if not isinstance(obj,dict): raise ValueError(f'{path}:{line_no}: expected object')
                result.append(obj)
    return result


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')


def canonicalize(row: dict, *, source: str, keep_teacher: bool=True) -> dict:
    question=row.get('question'); options=row.get('options')
    if not isinstance(question,str) or not question.strip(): raise ValueError('Missing question')
    if not isinstance(options,list) or not 2<=len(options)<=10 or any(not isinstance(v,str) or not v.strip() for v in options):
        raise ValueError('Expected 2..10 nonempty options, not a hardcoded four-choice schema')
    raw_index=row.get('answer_index')
    answer=str(row.get('answer','')).strip().upper()
    if raw_index is None:
        if answer not in LETTERS[:len(options)]: raise ValueError('Missing valid answer/index')
        index=LETTERS.index(answer)
    else:
        if isinstance(raw_index,bool) or not isinstance(raw_index,int): raise ValueError('answer_index must be integer')
        index=raw_index
    if not 0<=index<len(options): raise ValueError('answer_index outside available options')
    if answer and answer!=LETTERS[index]: raise ValueError('answer and answer_index disagree')
    out=dict(question_id=str(row.get('question_id',question_key(row)[:16])),question=question.strip(),
             options=[v.strip() for v in options],answer_index=index,answer=LETTERS[index],
             category=str(row.get('category','unknown')),source=source)
    if keep_teacher:
        text=row.get('teacher_text',row.get('cot_content',''))
        if isinstance(text,str) and text.strip() and text.strip().lower() not in ('none','nan','n/a'):
            out['teacher_text']=text.strip()
            out['teacher_source']=str(row.get('teacher_source',source+':provided_rationale'))
    out['question_hash']=question_key(out)
    return out


def format_prompt(row: dict, *, mode: str='direct') -> str:
    if mode not in ('direct','cot'): raise ValueError('mode must be direct or cot')
    instruction=('Choose the correct option. Reply with only its letter.' if mode=='direct' else
                 'Solve the problem step by step. End with "The answer is (X)." using one option letter.')
    choices='\n'.join(f'{LETTERS[i]}. {text}' for i,text in enumerate(row['options']))
    return f'{instruction}\n\nQuestion: {row["question"]}\nOptions:\n{choices}\nAnswer:'


def parse_final_choice(text: str, n_options: int) -> int | None:
    if not 2<=n_options<=10: raise ValueError('Invalid number of options')
    # No search for arbitrary A..J in the rationale; explicit final answer or bare letter only.
    valid=LETTERS[:n_options]
    matches=re.findall(r'(?:the\s+answer\s+is|final\s+answer\s*:)\s*\(?([A-Z])\)?(?![A-Za-z0-9_])',text,re.I)
    if matches:
        letter=matches[-1].upper()
    else:
        match=re.fullmatch(r'\s*\(?([A-J])\)?[.\s]*',text,re.I)
        if not match: return None
        letter=match.group(1).upper()
    return valid.index(letter) if letter in valid else None


def _unique(rows: list[dict], label: str) -> set[str]:
    keys=[question_key(x) for x in rows]
    if len(set(keys))!=len(keys): raise ValueError(f'Duplicate normalized questions in {label}')
    return set(keys)


def prepare_bundle(validation: list[dict], test: list[dict], output: Path | str, *, seed: int=42,
                   external_train: list[dict] | None=None, source_id: str | None=None,
                   revision: str='local-official-files') -> dict:
    output=Path(output)
    if (output/'manifest.json').exists(): raise ValueError('Output bundle already exists; use a new directory')
    val=[canonicalize(x,source='official_validation') for x in validation]
    sealed=[canonicalize(x,source='official_test',keep_teacher=False) for x in test]
    if not val or not sealed: raise ValueError('Official validation and test must both be present')
    if external_train is None:
        groups=defaultdict(list)
        for x in val: groups[x['category']].append(x)
        train=[]; dev=[]; rng=random.Random(seed)
        for category in sorted(groups):
            group=sorted(groups[category],key=question_key); rng.shuffle(group)
            if len(group)<2: raise ValueError('Need at least two validation items in each category')
            dev.extend(group[:1]); train.extend(group[1:])
        protocol='mmlu_pro_validation_adaptation_v1'
    else:
        if not source_id or source_id.casefold() in ('test','official_test','mmlu-pro-test'):
            raise ValueError('External training requires a non-test provenance source_id')
        train=[canonicalize(x,source=source_id) for x in external_train]
        dev=val
        protocol='mmlu_pro_external_train_v1'
    sets={name:_unique(rows,name) for name,rows in [('train',train),('dev',dev),('test',sealed)]}
    if not train or not dev: raise ValueError('Empty train/dev split')
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        if sets[a]&sets[b]: raise ValueError(f'Question overlap between {a} and {b}; reject permutations and duplicates')
    output.mkdir(parents=True,exist_ok=True)
    parts={'train':train,'dev':dev,'test':sealed}
    for name,rows in parts.items(): write_jsonl(output/f'{name}.jsonl',rows)
    manifest=dict(schema=SCHEMA,protocol=protocol,seed=seed,dataset='TIGER-Lab/MMLU-Pro',revision=revision,
        source_id=source_id,counts={name:len(rows) for name,rows in parts.items()},
        files={name:dict(path=f'{name}.jsonl',sha256=digest(output/f'{name}.jsonl')) for name in parts},
        test_policy='test labels/rationales never used for training, reward, or checkpoint selection',
        few_shot=0,notes='Validation-adaptation is not the standard untrained 5-shot protocol; report it explicitly.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def _manifest(directory: Path) -> dict:
    m=json.loads((directory/'manifest.json').read_text())
    if m.get('schema')!=SCHEMA: raise ValueError('Wrong data manifest schema')
    for name in ('train','dev','test'):
        entry=m['files'][name]
        if entry['path']!=f'{name}.jsonl': raise ValueError('Manifest path must be a local split filename')
        if digest(directory/entry['path'])!=entry['sha256']: raise ValueError(f'{name} file hash changed')
    return m


def read_training_bundle(directory: Path | str):
    directory=Path(directory); m=_manifest(directory)
    train=read_jsonl(directory/'train.jsonl'); dev=read_jsonl(directory/'dev.jsonl')
    if _unique(train,'train')&_unique(dev,'dev'): raise ValueError('train/dev overlap')
    if any(x.get('source')=='official_test' for x in train+dev): raise ValueError('Test split in training bundle')
    return train,dev,m


def read_evaluation_bundle(directory: Path | str, split: str, *, acknowledge_test: bool=False):
    if split not in ('dev','test'): raise ValueError('Evaluation split must be dev or test')
    if split=='test' and not acknowledge_test: raise ValueError('Must explicitly acknowledge final test evaluation')
    directory=Path(directory); m=_manifest(directory)
    return read_jsonl(directory/f'{split}.jsonl'),m
