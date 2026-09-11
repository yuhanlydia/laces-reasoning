"""Train/evaluate the ORIGINAL LACES S2 on MMLU-Pro: no new writer or GRU."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import time
import torch

from .data import (digest, format_prompt, parse_final_choice, question_key, read_evaluation_bundle,
                   read_training_bundle, LETTERS)
from .native import load_native
from .policy import (PolicyConfig, ddim_sample, rollout, distill_update, grpo_update, policy_mode)

SCHEMA='laces_mmlu_s2_v1'
PROTOCOL='pending-token-direct-s2-v1'


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--ckpt-dir',required=True)
    p.add_argument('--output',default='results/mmlu_pro/s2'); p.add_argument('--device',default='cuda:0')
    p.add_argument('--rwkv-path'); p.add_argument('--expected-step',type=int,default=30000)
    p.add_argument('--expected-writer',choices=['dynlowrank','fixed'],default='dynlowrank')
    p.add_argument('--mode',choices=['preflight','train','eval'],default='preflight')
    p.add_argument('--objective',choices=['distill','grpo'],default='distill')
    p.add_argument('--distill-source',choices=['candidates','rationale'],default='candidates')
    p.add_argument('--reward',choices=['gold_logprob','accuracy'],default='gold_logprob')
    p.add_argument('--steps',type=int,default=200); p.add_argument('--diffusion-steps',type=int,default=32)
    p.add_argument('--native-control-steps',type=int,default=1000)
    p.add_argument('--eta',type=float,default=.3); p.add_argument('--min-std',type=float,default=.02)
    p.add_argument('--cfg-scale',type=float,default=2.); p.add_argument('--blend',type=float,default=.7)
    p.add_argument('--lr',type=float,default=1e-6); p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--group-size',type=int,default=4); p.add_argument('--inner-epochs',type=int,default=1)
    p.add_argument('--kl-coef',type=float,default=.01); p.add_argument('--clip-range',type=float,default=.2)
    p.add_argument('--distill-temperature',type=float,default=.5)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--eval-samples',type=int,default=1)
    p.add_argument('--train-limit',type=int,default=0); p.add_argument('--dev-limit',type=int,default=0)
    p.add_argument('--test-limit',type=int,default=0)
    p.add_argument('--eval-every',type=int,default=25); p.add_argument('--save-every',type=int,default=25)
    p.add_argument('--resume'); p.add_argument('--init-s2'); p.add_argument('--s2-checkpoint')
    p.add_argument('--split',choices=['dev','test'],default='dev'); p.add_argument('--acknowledge-test',action='store_true')
    p.add_argument('--answer-mode',choices=['direct','cot'],default='direct')
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--eval-sampler',choices=['stochastic','ddim'],default='stochastic')
    a=p.parse_args(argv)
    if a.mode!='eval' and a.split=='test': raise ValueError('test may only be read in explicit final evaluation mode')
    if a.mode=='eval' and a.split=='test' and not a.acknowledge_test:
        # Checked again by data loader before any model activity.
        pass
    if sum(bool(x) for x in (a.resume,a.init_s2,a.s2_checkpoint))>1: raise ValueError('Choose resume OR init-s2 OR s2-checkpoint')
    if a.mode=='train' and a.s2_checkpoint: raise ValueError('For training use init-s2 or resume')
    if a.mode=='eval' and (a.init_s2 or a.resume): raise ValueError('For evaluation use s2-checkpoint')
    if min(a.steps,a.diffusion_steps,a.native_control_steps,a.eval_samples,a.save_every,a.eval_every)<1 or a.group_size<2:
        raise ValueError('Positive budgets and group_size>=2 required')
    if a.lr<=0 or a.weight_decay<0 or a.inner_epochs<1: raise ValueError('Invalid optimizer settings')
    if any(x<0 for x in [a.train_limit,a.dev_limit,a.test_limit]): raise ValueError('Limits must be >=0')
    PolicyConfig(a.diffusion_steps,a.eta,a.min_std,a.cfg_scale)
    if a.objective=='grpo' and a.distill_source!='candidates': raise ValueError('Rationale source applies to distill only')
    return a


def _seed(seed,key): return int(sha256(f'{seed}:{key}'.encode()).hexdigest()[:8],16)

def _rng(device,seed): return torch.Generator(device=device).manual_seed(seed)

def _sync(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


def _json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)); tmp.replace(path)


def _cpu_state(module): return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def _training_contract(a,native,manifest):
    return dict(schema=SCHEMA,protocol=PROTOCOL,parent=native.audit['active_fingerprint'],
        data_files=manifest['files'],objective=a.objective,distill_source=a.distill_source,
        seed=a.seed,lr=a.lr,weight_decay=a.weight_decay,group_size=a.group_size,inner_epochs=a.inner_epochs,
        diffusion=asdict(PolicyConfig(a.diffusion_steps,a.eta,a.min_std,a.cfg_scale)),blend=a.blend,
        reward=a.reward,answer_mode=a.answer_mode,max_new_tokens=a.max_new_tokens,
        kl_coef=a.kl_coef,clip_range=a.clip_range,distill_temperature=a.distill_temperature,
        train_limit=a.train_limit,dev_limit=a.dev_limit,eval_samples=a.eval_samples,eval_sampler=a.eval_sampler)


def _evaluation_contract(a):
    return dict(diffusion=asdict(PolicyConfig(a.diffusion_steps,a.eta,a.min_std,a.cfg_scale)),
                blend=a.blend,answer_mode=a.answer_mode,max_new_tokens=a.max_new_tokens,eval_sampler=a.eval_sampler)


def _load_s2(path,native,manifest,*,contract=None,evaluation_contract=None):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('schema')!=SCHEMA or 's2' not in saved:
        raise ValueError('Expected direct S2 checkpoint, not a standalone writer/refiner checkpoint')
    if saved['parent_fingerprint']!=native.audit['active_fingerprint']:
        raise ValueError('S2 checkpoint belongs to a different pretrained parent')
    if saved['data_files']!=manifest['files']: raise ValueError('Checkpoint/data manifest mismatch')
    if saved['protocol']!=PROTOCOL: raise ValueError('Decoder protocol mismatch')
    if contract is not None and saved['training_contract']!=contract:
        raise ValueError('Resume settings differ; use init-s2 for an explicit new training stage')
    if evaluation_contract is not None:
        previous={k:saved['training_contract'][k] for k in evaluation_contract}
        if previous!=evaluation_contract:
            raise ValueError('Checkpoint evaluation settings differ; pass the trained sampler/blend/answer configuration explicitly')
    native.s2.load_state_dict(saved['s2'],strict=True)
    return saved


def _save(path,native,reference,optimizer,step,contract,manifest,a,rng,best):
    payload=dict(schema=SCHEMA,protocol=PROTOCOL,parent_fingerprint=native.audit['active_fingerprint'],
        parent_audit=native.audit,objective=a.objective,global_step=step,training_contract=contract,
        data_files=manifest['files'],s2=_cpu_state(native.s2),reference_s2=_cpu_state(reference),
        optimizer=optimizer.state_dict(),args=vars(a),python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,best_dev=best)
    path=Path(path); tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)


def _metrics(rows):
    categories={}
    for row in rows:
        group=categories.setdefault(row['category'],dict(n=0,correct=0,ensemble_correct=0))
        group['n']+=1; group['correct']+=row['correct']; group['ensemble_correct']+=row['ensemble_correct']
    n=len(rows)
    for g in categories.values():
        g['accuracy']=g['correct']/g['n']; g['ensemble_accuracy']=g['ensemble_correct']/g['n']
    return dict(n=n,accuracy=sum(x['correct'] for x in rows)/n,
        ensemble_accuracy=sum(x['ensemble_correct'] for x in rows)/n,
        macro_accuracy=sum(x['accuracy'] for x in categories.values())/len(categories),
        invalid=sum(x['invalid'] for x in rows),mean_seconds=sum(x['seconds'] for x in rows)/n,
        categories=categories)


def _teacher_text(rec):
    text=rec.get('teacher_text','')
    if not text:
        raise ValueError(f'Missing teacher text for training item {rec["question_id"]}')
    last=parse_final_choice(text,len(rec['options']))
    if last is not None and last!=rec['answer_index']:
        raise ValueError(f'Teacher final answer disagrees with gold for {rec["question_id"]}')
    return text if last is not None else text+f'\nThe answer is ({rec["answer"]}).'


def _sample(denoiser,cond,native,config,seed,sampler):
    gen=_rng(cond.device,seed)
    if sampler=='ddim': return ddim_sample(denoiser,cond,native.horizon,config,gen)
    return rollout(denoiser,cond,native.horizon,config,gen).final


@torch.no_grad()
def _outcome(native,ids,z,record,a,prompt):
    if a.answer_mode=='direct':
        scores=native.score_options(ids,z,len(record['options']),prompt_text=prompt)
        choice=int(scores.argmax()); logq=scores-torch.logsumexp(scores,0)
        reward=float(logq[record['answer_index']]) if a.reward=='gold_logprob' else float(choice==record['answer_index'])
        return reward,choice,dict(option_logprobs=scores.cpu().tolist(),text=LETTERS[choice])
    token_ids,text=native.generate(ids,z,a.max_new_tokens)
    choice=parse_final_choice(text,len(record['options']))
    if a.reward=='accuracy': reward=float(choice==record['answer_index'])
    else:
        # A dense auxiliary on a separate direct-answer prompt. Never call this the
        # probability of the generated CoT. Report it explicitly in the run contract.
        raise ValueError('CoT mode currently requires --reward accuracy; no fabricated rationale reward')
    return reward,choice,dict(text=text,token_ids=token_ids)


@torch.no_grad()
def evaluate(native,records,a,config,*,reference=None):
    outputs={'current':[],'raw_rwkv':[]}
    if reference is not None: outputs['parent_matched']=[]
    for rec in records:
        prompt=format_prompt(rec,mode=a.answer_mode)
        _sync(native.device); encoding_start=time.perf_counter()
        ids,cond=native.encode_prompt(prompt,seed=_seed(a.seed,question_key(rec)))
        _sync(native.device); encoding_seconds=time.perf_counter()-encoding_start
        for arm,denoiser in [('current',native.s2),('parent_matched',reference),('raw_rwkv',None)]:
            if arm=='parent_matched' and reference is None: continue
            predictions=[]; candidates=[]; vectors=[]
            _sync(native.device); start=time.perf_counter()
            for k in range(a.eval_samples if arm!='raw_rwkv' else 1):
                z=(torch.zeros(1,native.horizon,native.model.latent_dim) if arm=='raw_rwkv' else
                   _sample(denoiser,cond,native,config,_seed(a.seed,f'{question_key(rec)}:eval:{k}'),a.eval_sampler))
                if a.answer_mode=='direct':
                    scores=native.score_options(ids,z,len(rec['options']),raw=arm=='raw_rwkv',prompt_text=prompt)
                    pred=int(scores.argmax()); vectors.append(torch.log_softmax(scores.float(),0))
                    candidates.append(dict(choice=pred,option_logprobs=scores.cpu().tolist()))
                else:
                    tokens,text=native.generate(ids,z,a.max_new_tokens,raw=arm=='raw_rwkv')
                    pred=parse_final_choice(text,len(rec['options']))
                    candidates.append(dict(choice=pred,text=text,token_ids=tokens))
                predictions.append(pred)
            if vectors:
                ensemble=int(torch.logsumexp(torch.stack(vectors),0).argmax())
            else:
                from collections import Counter
                valid=[p for p in predictions if p is not None]
                counts=Counter(valid)
                ensemble=min(counts,key=lambda k:(-counts[k],k)) if counts else None
            _sync(native.device)
            outputs[arm].append(dict(question_id=rec['question_id'],question_hash=question_key(rec),category=rec['category'],
                gold=rec['answer_index'],choice=predictions[0],ensemble_choice=ensemble,
                correct=predictions[0]==rec['answer_index'],ensemble_correct=ensemble==rec['answer_index'],
                invalid=predictions[0] is None,candidates=candidates,encoding_seconds=encoding_seconds,
                seconds=time.perf_counter()-start+encoding_seconds))
    return dict(metrics={k:_metrics(v) for k,v in outputs.items()},samples=outputs,
                definition='zero-shot direct option joint likelihood' if a.answer_mode=='direct' else 'greedy CoT, explicit final-letter parse',
                note='Not the official five-shot CoT setting. Ensemble averages probabilities/votes, never oracle best-of-N.')


def run(a,*,runtime=None):
    if a.answer_mode=='cot' and a.reward!='accuracy': raise ValueError('CoT uses --reward accuracy')
    if a.mode=='eval':
        eval_rows,manifest=read_evaluation_bundle(a.data,a.split,acknowledge_test=a.acknowledge_test)
        limit=a.test_limit if a.split=='test' else a.dev_limit
        if limit: eval_rows=eval_rows[:limit]
        train=dev=None
    else:
        train,dev,manifest=read_training_bundle(a.data)
        if a.train_limit: train=train[:a.train_limit]
        if a.dev_limit: dev=dev[:a.dev_limit]
    out=Path(a.output)
    if a.mode=='train' and not a.resume and any((out/k).exists() for k in ('latest.pt','best_dev.pt','metrics.jsonl','summary.json')):
        raise ValueError('Output already contains a training run; use a new output directory or strict resume')
    torch.manual_seed(a.seed); rng=random.Random(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    owned=runtime is None
    native=runtime or load_native(a.ckpt_dir,a.device,rwkv_path=a.rwkv_path,
                  expected_step=a.expected_step,expected_writer=a.expected_writer,blend=a.blend)
    if a.blend==0 and a.mode=='train': raise ValueError('blend=0 disconnects S2 from the answer; control only')
    if a.mode!='eval' and a.objective=='distill' and a.distill_source=='rationale':
        for rec in train:
            text=_teacher_text(rec)
            length=native._ids(text).shape[1]
            if length>native.horizon*native.chunk:
                raise ValueError(f'Teacher {rec["question_id"]} has {length} tokens, beyond native horizon {native.horizon*native.chunk}; provide shorter verified rationales, not silent truncation')
    native.enable_s2_training()
    cfg=PolicyConfig(a.diffusion_steps,a.eta,a.min_std,a.cfg_scale)
    out.mkdir(parents=True,exist_ok=True)
    _json(out/'run_config.json',dict(args=vars(a),parent=native.audit,data=manifest,protocol=PROTOCOL,
        trainable_parameters=sum(p.numel() for p in native.s2.parameters()),
        trainable_names=[k for k,p in native.model.named_parameters() if p.requires_grad]))
    contract=_training_contract(a,native,manifest)
    parent=copy.deepcopy(native.s2).requires_grad_(False)
    parent.load_state_dict(native.parent_s2_state,strict=True)
    loaded=None
    if a.resume: loaded=_load_s2(a.resume,native,manifest,contract=contract)
    if a.init_s2: loaded=_load_s2(a.init_s2,native,manifest)
    if a.s2_checkpoint: loaded=_load_s2(a.s2_checkpoint,native,manifest,evaluation_contract=_evaluation_contract(a))
    reference=copy.deepcopy(native.s2).requires_grad_(False)
    if a.resume: reference.load_state_dict(loaded['reference_s2'],strict=True)
    if a.mode=='eval':
        # Parent matched control must be the original checkpoint, not a copy of the
        # already-loaded trained S2. The saved run's KL anchor may be distilled.
        result=evaluate(native,eval_rows,a,cfg,reference=parent)
        result.update(parent=native.audit,split=a.split,eval_samples=a.eval_samples,sampler=a.eval_sampler,
                      manifest_protocol=manifest['protocol'],checkpoint=a.s2_checkpoint)
        _json(out/f'evaluation_{a.split}.json',result)
        if owned: native.close()
        return result
    # CPU/disk cache only S0 conditioning keyed by prompt; S2 samples are regenerated
    # after every optimizer update, so stale plans cannot masquerade as on-policy RL.
    cache={}
    def prepared(rec):
        key=question_key(rec)
        if key not in cache:
            prompt=format_prompt(rec,mode=a.answer_mode)
            ids,cond=native.encode_prompt(prompt,seed=_seed(a.seed,key))
            cache[key]=(ids.cpu(),cond.cpu(),prompt)
        ids,cond,prompt=cache[key]
        return ids.to(native.device),cond.to(native.device),prompt
    if a.mode=='preflight':
        rec=train[0]; ids,cond,prompt=prepared(rec)
        group=[rollout(native.s2,cond,native.horizon,cfg,_rng(native.device,a.seed+i)) for i in range(a.group_size)]
        rewards=torch.tensor([_outcome(native,ids,g.final,rec,a,prompt)[0] for g in group])
        first=group[0].final
        before=native.score_options(ids,first,len(rec['options']),prompt_text=prompt)
        after=native.score_options(ids,first,len(rec['options']),prompt_text=prompt)
        from .policy import gaussian_log_prob,transition_mean
        tr=group[0].transitions[0]
        mean,std=transition_mean(native.s2,tr.state.to(native.device),cond,tr.index,cfg)
        surrogate=gaussian_log_prob(tr.action.to(native.device),mean,std).sum()
        gradients=torch.autograd.grad(surrogate,tuple(native.s2.parameters()),allow_unused=True)
        gradient_norm=sum(float(g.detach().float().square().sum()) for g in gradients if g is not None)**.5
        if not math.isfinite(gradient_norm) or gradient_norm==0: raise RuntimeError('S2 policy gradient preflight failed')
        control_cfg=PolicyConfig(a.native_control_steps,a.eta,a.min_std,a.cfg_scale)
        control=ddim_sample(native.s2,cond,native.horizon,control_cfg,_rng(native.device,a.seed))
        control_scores=native.score_options(ids,control,len(rec['options']),prompt_text=prompt)
        raw_scores=native.score_options(ids,control,len(rec['options']),raw=True,prompt_text=prompt)
        result=dict(parent=native.audit,calls=dict(native.calls),s2_score_gradient_norm=gradient_norm,
            reward_mean=float(rewards.mean()),reward_std=float(rewards.std(unbiased=False)),
            repeat_score_max_diff=float((before-after).abs().max()),matched_ddim_control_steps=a.native_control_steps,
            parent_ddim_option_logprobs=control_scores.tolist(),raw_option_logprobs=raw_scores.tolist(),
            current_policy_option_logprobs=before.tolist(),ready_for_signal_pilot=float(rewards.std(unbiased=False))>1e-7,
            note='Contracts and reward variance only; not a reasoning-accuracy result.')
        if any(native.calls[k]==0 for k in ('S0','S1','S2')): raise RuntimeError('A pretrained component was bypassed')
        _json(out/'preflight.json',result)
        if owned: native.close()
        return result
    optimizer=torch.optim.AdamW(native.s2.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    step=0; best=-1.
    if a.resume:
        optimizer.load_state_dict(loaded['optimizer']); step=loaded['global_step']; best=loaded['best_dev']
        rng.setstate(loaded['python_rng']); torch.set_rng_state(loaded['torch_rng'])
        if torch.cuda.is_available() and loaded['cuda_rng'] is not None: torch.cuda.set_rng_state_all(loaded['cuda_rng'])
    history_path=out/'metrics.jsonl'
    while step<a.steps:
        rec=train[rng.randrange(len(train))]; ids,cond,prompt=prepared(rec)
        _sync(native.device); started=time.perf_counter(); rewards=None
        if a.objective=='distill' and a.distill_source=='rationale':
            text=_teacher_text(rec)
            target,mask=native.encode_teacher(text,seed=_seed(a.seed,question_key(rec)+':teacher'))
            metrics=distill_update(native.s2,optimizer,cond,[target],masks=[mask])
        else:
            group=[rollout(native.s2,cond,native.horizon,cfg) for _ in range(a.group_size)]
            outcomes=[_outcome(native,ids,g.final,rec,a,prompt) for g in group]
            rewards=torch.tensor([x[0] for x in outcomes]); choices=[x[1] for x in outcomes]
            if a.objective=='distill':
                metrics=distill_update(native.s2,optimizer,cond,[g.final for g in group],rewards=rewards,temperature=a.distill_temperature)
            else:
                metrics=grpo_update(native.s2,reference,optimizer,cond,group,rewards,cfg,
                          clip_range=a.clip_range,kl_coef=a.kl_coef,inner_epochs=a.inner_epochs)
            metrics.update(reward_mean=float(rewards.mean()),reward_std=float(rewards.std(unbiased=False)),
                           group_accuracy=sum(p==rec['answer_index'] for p in choices)/len(choices))
        if any(p.grad is not None for name,p in native.model.named_parameters() if not name.startswith('trajectory_dit.')):
            raise RuntimeError('Frozen parent module received a gradient')
        step+=1; _sync(native.device)
        metrics.update(step=step,seconds=time.perf_counter()-started,objective=a.objective,question_hash=question_key(rec))
        with history_path.open('a') as f: f.write(json.dumps(metrics,allow_nan=False)+'\n')
        print(json.dumps(metrics),flush=True)
        if step%a.save_every==0 or step==a.steps:
            _save(out/'latest.pt',native,reference,optimizer,step,contract,manifest,a,rng,best)
        if step%a.eval_every==0 or step==a.steps:
            # Model selection ONLY uses the held-out development split.
            report=evaluate(native,dev,a,cfg,reference=parent)
            _json(out/f'dev_step_{step:06d}.json',report)
            score=report['metrics']['current']['ensemble_accuracy']
            if score>best:
                best=score
                _save(out/'best_dev.pt',native,reference,optimizer,step,contract,manifest,a,rng,best)
    _save(out/'latest.pt',native,reference,optimizer,step,contract,manifest,a,rng,best)
    summary=dict(global_step=step,best_dev_accuracy=best,checkpoint=str(out/'latest.pt'),parent=native.audit,
                 objective=a.objective,distill_source=a.distill_source,data_protocol=manifest['protocol'],
                 verification='training completed; test was not evaluated')
    _json(out/'summary.json',summary)
    if owned: native.close()
    return summary


def main(argv=None):
    result=run(parse_args(argv)); print(json.dumps({k:v for k,v in result.items() if k not in ('samples','parent')},indent=2))


if __name__=='__main__': main()
