"""User-authorized custom repartition; NOT the official MMLU-Pro benchmark."""
import json,random,unicodedata
from collections import defaultdict,Counter
from pathlib import Path
from laces_posttrain.data import read_jsonl,write_jsonl,question_key,digest,SCHEMA,read_training_bundle
root=Path(__file__).resolve().parents[3]
source=root/'data/mmlu_pro_pilot'; dest=root/'data/mmlu_pro_custom_s42'
assert not dest.exists(), 'Refuse to overwrite a split'
norm=lambda x:' '.join(unicodedata.normalize('NFKC',x).casefold().split())
groups=defaultdict(list)
for part in ['train','dev','test']:
 for row in read_jsonl(source/f'{part}.jsonl'):
  groups[question_key(row)].append(row)
unique=[]; conflicts=[]
for key,rows in sorted(groups.items()):
 if len({norm(r['options'][r['answer_index']]) for r in rows})>1:
  conflicts.append(key);continue
 row=dict(rows[0]);row['original_sources']=sorted({r['source'] for r in rows});row['source']='mmlu_pro_custom_repartition';row.pop('teacher_text',None);row.pop('teacher_source',None);unique.append(row)
bycat=defaultdict(list)
for r in unique:bycat[r['category']].append(r)
rng=random.Random(42);parts={s:[] for s in ['train','dev','test']}
for cat,rows in sorted(bycat.items()):
 rng.shuffle(rows);nt=max(1,round(len(rows)*.10));nd=max(1,round((len(rows)-nt)*.05))
 parts['test']+=rows[:nt];parts['dev']+=rows[nt:nt+nd];parts['train']+=rows[nt+nd:]
sets={s:{question_key(r) for r in rows} for s,rows in parts.items()}
assert all(len(sets[s])==len(parts[s]) for s in parts)
assert not (sets['train']&sets['dev'] or sets['train']&sets['test'] or sets['dev']&sets['test'])
assert sum(map(len,parts.values()))==len(unique)
dest.mkdir(parents=True)
for s,rows in parts.items():
 rng.shuffle(rows);write_jsonl(dest/f'{s}.jsonl',rows)
m=dict(schema=SCHEMA,protocol='mmlu_pro_custom_repartition_v1',seed=42,dataset='TIGER-Lab/MMLU-Pro',revision=json.loads((source/'manifest.json').read_text())['revision'],source_id='official_validation_and_test_user_authorized_repartition',counts={s:len(r) for s,r in parts.items()},category_counts={s:dict(Counter(r['category'] for r in rows)) for s,rows in parts.items()},input_rows=sum(map(len,groups.values())),unique_question_groups=len(groups),conflicting_answer_groups_excluded=conflicts,files={s:dict(path=f'{s}.jsonl',sha256=digest(dest/f'{s}.jsonl')) for s in parts},test_policy='Only custom held-out test is sealed; original official test is explicitly repartitioned by user authorization.',notes='Custom 90/10 partition, 5% of training allocation reserved for dev; not official benchmark. Normalized question+unordered-options identities deduplicated; no semantic near-duplicate guarantee.')
(dest/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
train,dev,loaded=read_training_bundle(dest)
assert len(train)==m['counts']['train'] and len(dev)==m['counts']['dev']
print(json.dumps(m,indent=2))
