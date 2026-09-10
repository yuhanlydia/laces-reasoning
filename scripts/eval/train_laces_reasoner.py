#!/usr/bin/env python3
"""Answer-supervised latent refinement through the *pretrained* 30k LACES interface.

Default mode is preflight, not a long training job. No fresh writer, S0 encoder, full
state-MSE target, or oracle-hop allocation is used. answer_pg needs forward rewards
only; answer_ce explicitly requires a verified state-input gradient through RWKV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.laces_latent_refiner import LACESRuntime, LatentTrajectoryRefiner, answer_policy_loss
from scripts.eval.relay_utils import load_relay_model

DEFAULT_CKPT = 'outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000'
FORMAT = 'laces_pretrained_refiner_v1'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt_dir', default=DEFAULT_CKPT)
    p.add_argument('--rwkv_path', default=None, help='Local backbone AND tokenizer directory override')
    p.add_argument('--expected_step', type=int, default=30000)
    p.add_argument('--mode', choices=['preflight','train','eval'], default='preflight')
    p.add_argument('--objective', choices=['answer_pg','answer_ce'], default='answer_pg')
    p.add_argument('--protocol', choices=['aligned','legacy'], default='aligned')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output_dir', default='outputs_eval/laces_pretrained_reasoner')
    p.add_argument('--cache_dir', default='outputs_cache/laces_pretrained_reasoner', help='Disk cache of prefix-only S0/S2 outputs and token features')
    p.add_argument('--resume', default=None, help='New refiner checkpoint, never an old standalone-writer checkpoint')
    p.add_argument('--train_jsonl', default=None)
    p.add_argument('--validation_jsonl', default=None)
    p.add_argument('--test_jsonl', default=None)
    p.add_argument('--n_train', type=int, default=4, help='Per task for built-in synthetic smoke data')
    p.add_argument('--n_validation', type=int, default=2)
    p.add_argument('--n_test', type=int, default=2)
    p.add_argument('--epochs', type=int, default=1, help='Total epochs, including resumed epochs')
    p.add_argument('--train_depths', type=int, nargs='+', default=[1,2,4,8])
    p.add_argument('--eval_depths', type=int, nargs='+', default=[0,1,2,4,8])
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--max_step', type=float, default=.05)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--max_grad_norm', type=float, default=1.)
    p.add_argument('--group_size', type=int, default=4)
    p.add_argument('--noise_scale', type=float, default=.05)
    p.add_argument('--drift_weight', type=float, default=.01)
    p.add_argument('--plan_steps', type=int, default=1000)
    p.add_argument('--cfg_scale', type=float, default=2.)
    p.add_argument('--blend', type=float, default=.7)
    p.add_argument('--diffusion_sampler', choices=['ddim','ddpm'], default='ddim')
    p.add_argument('--max_new_tokens', type=int, default=32)
    p.add_argument('--max_prefix_tokens', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args(argv)


def validate_args(args):
    if args.mode == 'train' and args.protocol == 'legacy':
        raise ValueError('legacy timing has a blind first-token write; use aligned for answer training')
    if min(args.train_depths) < 1 or min(args.eval_depths) < 0:
        raise ValueError('Training depths must be positive; evaluation depths may include zero')
    if not all([args.train_jsonl, args.validation_jsonl, args.test_jsonl]) and any(
            [args.train_jsonl, args.validation_jsonl, args.test_jsonl]):
        raise ValueError('Supply all three JSONL splits, or use the built-in smoke generator')
    if args.group_size < 2 or args.noise_scale <= 0 or args.plan_steps < 1 or args.epochs < 0:
        raise ValueError('Invalid group_size, noise_scale, plan_steps or epochs')
    if min(args.n_train, args.n_validation, args.n_test, args.max_new_tokens, args.max_prefix_tokens) < 1:
        raise ValueError('Counts and token limits must be positive')
    if args.mode == 'eval' and not args.resume:
        raise ValueError('eval requires --resume; use preflight for the identity-initialized model')


def validate_splits(splits):
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            prefix = row['prefix']
            if prefix in seen:
                raise ValueError(f'Duplicate/overlap in {seen[prefix]} and {split}: split by prompt before training')
            seen[prefix] = split


def load_refiner_checkpoint(path, base):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Old standalone reasoner checkpoints are not compatible with pretrained LACES coordinates')
    if payload.get('base', {}).get('active_fingerprint') != base['active_fingerprint']:
        raise ValueError('Pretrained S0/S1/S2 fingerprint differs from the refiner checkpoint')
    previous_path = payload.get('base', {}).get('backbone_path')
    if previous_path is not None and previous_path != base.get('backbone_path'):
        raise ValueError('Backbone/tokenizer path differs from the refiner checkpoint; verify provenance first')
    return payload


def read_jsonl(path):
    rows = []
    for line_number,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip():
            continue
        item = json.loads(line)
        # Accept an exact prefix OR facts/question; never insert the gold answer into prefix.
        if 'prefix' in item:
            prefix = item['prefix']
        else:
            facts = item.get('facts', [])
            if not isinstance(facts, list) or not all(isinstance(f,str) for f in facts):
                raise ValueError(f'{path}:{line_number}: facts must be a list of strings')
            prefix = '\n'.join(facts + [item['question']])
        answer = item['answer']
        if not isinstance(prefix,str) or not isinstance(answer,str) or not prefix.strip() or not answer.strip():
            raise ValueError(f'{path}:{line_number}: nonempty string prefix/answer required')
        rows.append(dict(prefix=prefix,answer=answer))
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def build_splits(args):
    if args.train_jsonl:
        splits = {name:read_jsonl(getattr(args,name+'_jsonl')) for name in ('train','validation','test')}
    else:
        from scripts.eval.diag_hard_problems_v2 import gen_2agent, gen_3agent, gen_4agent
        splits = {}
        for name, offset, count in [('train',0,args.n_train),('validation',10000,args.n_validation),
                                    ('test',20000,args.n_test)]:
            rows = []
            for gen in (gen_2agent,gen_3agent,gen_4agent):
                for item in gen(count,args.seed+offset):
                    rows.append({'prefix':'\n'.join(item['agents']+[item['q']]),'answer':item['gold']})
            splits[name] = rows
    validate_splits(splits)
    return splits


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))


class Examples:
    def __init__(self, runtime, tokenizer, args):
        self.runtime, self.tokenizer, self.args = runtime,tokenizer,args
        self.config = dict(base=runtime.audit['active_fingerprint'],backbone=runtime.audit['backbone_path'],plan_steps=args.plan_steps,
            cfg=args.cfg_scale,sampler=args.diffusion_sampler,seed=args.seed)

    def get(self, row, *, refresh=False):
        args = self.args
        prefix = self.tokenizer(row['prefix'],return_tensors='pt').input_ids.to(args.device)
        suffix = row['answer']
        if not row['prefix'][-1:].isspace() and not suffix[:1].isspace():
            suffix = ' ' + suffix
        answer = self.tokenizer(suffix,add_special_tokens=False,return_tensors='pt').input_ids.to(args.device)
        if prefix.shape[1] > args.max_prefix_tokens:
            raise ValueError('Prefix exceeds max_prefix_tokens; no silent evidence truncation is allowed')
        if not 0 < answer.shape[1] <= self.runtime.model.trajectory_horizon * self.runtime.model.trajectory_chunk_size:
            raise ValueError('Gold answer exceeds trajectory horizon; no silent label truncation is allowed')
        key = hashlib.sha256(json.dumps([self.config,prefix.tolist()],sort_keys=True).encode()).hexdigest()
        cached = Path(args.cache_dir)/f'{key}.pt' if args.cache_dir else None
        if cached is not None and cached.exists() and not refresh:
            values = torch.load(cached,map_location='cpu',weights_only=True)
            if values['key'] != key:
                raise ValueError('Prefix cache provenance mismatch')
            z,h,zp = [values[k].to(args.device) for k in ('z','h','zp')]
        else:
            # Prefix-dependent seed; identical S0/S2 realization across R and candidate comparisons.
            seed = (args.seed + int(key[:8],16)) % (2**31)
            z,h,zp = self.runtime.prepare(prefix,seed=seed)
            if cached is not None:
                cached.parent.mkdir(parents=True,exist_ok=True)
                tmp = cached.with_suffix('.tmp')
                torch.save(dict(key=key,z=z.cpu(),h=h.cpu(),zp=zp.cpu()),tmp)
                tmp.replace(cached)
        return prefix,answer,z,h,zp


def policy_candidates(mean, answer_length, runtime, args, *, seed):
    active = math.ceil(answer_length/runtime.model.trajectory_chunk_size)
    active_mean = mean[:,:active]
    sigma = active_mean.detach().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)*args.noise_scale
    generator = torch.Generator(device=mean.device).manual_seed(seed)
    noise = torch.randn((args.group_size,*active_mean.shape),generator=generator,device=mean.device)
    actions = (active_mean.unsqueeze(0) + sigma.unsqueeze(0)*noise).detach()
    candidates = mean.detach().unsqueeze(0).expand(args.group_size,*mean.shape).clone()
    candidates[:,:,:active] = actions
    return active_mean,actions,sigma,candidates


def preflight(runtime, refiner, examples, row, args):
    prefix,answer,z0,h,zp = examples.get(row,refresh=True)
    with torch.no_grad():
        identity = refiner(z0,h,zp,steps=0)
        baseline_lp = runtime.score(prefix,answer,z0)
        r0_lp = runtime.score(prefix,answer,identity)
        ids,_ = runtime.generate(prefix,z0,max_new_tokens=args.max_new_tokens,
                                 eos_id=getattr(examples.tokenizer,'eos_token_id',None))
        raw_ids,_ = runtime.generate(prefix,z0,max_new_tokens=args.max_new_tokens,raw=True,
                                     eos_id=getattr(examples.tokenizer,'eos_token_id',None))
    identity_ok = torch.equal(z0,identity) and torch.equal(baseline_lp,r0_lp)
    if not identity_ok or not all(runtime.calls[g] for g in ('S0','S1','S2')):
        raise RuntimeError('Pretrained-module / R=0 identity gate failed')
    gradient_ok = None
    if args.objective == 'answer_ce':
        zprobe = z0.detach().requires_grad_(True)
        lp = runtime.score(prefix,answer[:, :2],zprobe)
        gradient = torch.autograd.grad(lp,zprobe,allow_unused=True)[0] if lp.requires_grad else None
        gradient_ok = gradient is not None and torch.isfinite(gradient).all().item() and gradient.norm().item()>0
    with torch.no_grad():
        mean = refiner(z0,h,zp,steps=min(args.train_depths))
        _,_,_,candidates = policy_candidates(mean,answer.shape[1],runtime,args,seed=args.seed+1)
        rewards = torch.stack([runtime.score(prefix,answer,z) for z in candidates])
    reward_std = rewards.std(unbiased=False).item()
    report = dict(base=runtime.audit,protocol=args.protocol,R0_identity=identity_ok,
        actual_pretrained_calls=dict(runtime.calls),answer_state_gradient=gradient_ok,
        candidate_reward_std=reward_std,baseline_answer_log_probability=float(baseline_lp),
        raw_text=examples.tokenizer.decode(raw_ids,skip_special_tokens=False),
        laces_R0_text=examples.tokenizer.decode(ids,skip_special_tokens=False),
        learned_writer_created=False,objective=args.objective,
        ready_for_selected_objective=bool(gradient_ok if args.objective=='answer_ce' else reward_std>1e-7))
    write_json(Path(args.output_dir)/'preflight.json',report)
    return report


def evaluate(runtime,refiner,examples,rows,args):
    keys = ['raw_rwkv']+[f'laces_R{r}' for r in sorted(set([0]+args.eval_depths))]
    totals = {k:dict(exact_match=0,contains_answer=0,answer_log_probability=0.) for k in keys}
    samples = []
    with torch.no_grad():
        for row in rows:
            prefix,answer,z0,h,zp = examples.get(row)
            for key in keys:
                raw = key == 'raw_rwkv'
                depth = 0 if raw else int(key.removeprefix('laces_R'))
                z = refiner(z0,h,zp,steps=depth)
                lp = float(runtime.score(prefix,answer,z,raw=raw))
                ids,_ = runtime.generate(prefix,z,max_new_tokens=args.max_new_tokens,raw=raw,
                                          eos_id=getattr(examples.tokenizer,'eos_token_id',None))
                text = examples.tokenizer.decode(ids,skip_special_tokens=True).strip()
                raw_text = examples.tokenizer.decode(ids,skip_special_tokens=False)
                gold = row['answer'].strip().casefold()
                exact = text.casefold() == gold
                contains = re.search(r'(?<!\w)'+re.escape(gold)+r'(?!\w)',text.casefold()) is not None
                totals[key]['exact_match'] += int(exact)
                totals[key]['contains_answer'] += int(contains)
                totals[key]['answer_log_probability'] += lp
                samples.append(dict(prefix=row['prefix'],gold=row['answer'],method=key,token_ids=ids,
                                    text=text,raw_text=raw_text,answer_log_probability=lp))
    return dict(n=len(rows),metrics={k:{m:v/len(rows) for m,v in values.items()} for k,values in totals.items()},
                samples=samples,protocol=args.protocol)


def run(args):
    validate_args(args)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    splits = build_splits(args)
    model,_,tokenizer,checkpoint,_ = load_relay_model(args.ckpt_dir,args.device,rwkv_path_override=args.rwkv_path)
    runtime = LACESRuntime(model,checkpoint,expected_step=args.expected_step,plan_steps=args.plan_steps,
        cfg_scale=args.cfg_scale,blend=args.blend,diffusion_sampler=args.diffusion_sampler,protocol=args.protocol)
    # All required checkpoint tensors were compared against the loaded model before discarding payload.
    del checkpoint
    resume = load_refiner_checkpoint(args.resume,runtime.audit) if args.resume else None
    config = resume['refiner_config'] if resume else dict(latent_dim=model.latent_dim,
        hidden_dim=model.hidden_size,width=args.width,max_step=args.max_step)
    refiner = LatentTrajectoryRefiner(**config).to(args.device)
    if resume:
        refiner.load_state_dict(resume['refiner'],strict=True)
        if resume.get('protocol') != args.protocol:
            raise ValueError('Resume protocol mismatch; do not silently change state/token timing')
    examples = Examples(runtime,tokenizer,args)
    print('[base]',json.dumps(runtime.audit),flush=True)
    print('[trainable] only latent refiner:',sum(p.numel() for p in refiner.parameters()),flush=True)
    report = preflight(runtime,refiner,examples,splits['validation'][0],args)
    print('[preflight]',json.dumps({k:v for k,v in report.items() if not k.endswith('_text')}),flush=True)
    if args.mode == 'preflight':
        runtime.close()
        return report
    if args.mode == 'train' and not report['ready_for_selected_objective']:
        raise RuntimeError('Preflight has no usable answer signal for this objective; see preflight.json')

    if args.mode == 'train':
        optimizer = torch.optim.AdamW(refiner.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        if resume:
            optimizer.load_state_dict(resume['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr
                group['weight_decay'] = args.weight_decay
        first_epoch = resume['epochs_completed'] if resume else 0
        global_step = resume.get('global_step',0) if resume else 0
        for epoch in range(first_epoch,args.epochs):
            rng = random.Random(args.seed+epoch)
            order = list(range(len(splits['train'])))
            rng.shuffle(order)
            updates,flat = 0,0
            for row_index in order:
                prefix,answer,z0,h,zp = examples.get(splits['train'][row_index])
                depth = rng.choice(args.train_depths)
                optimizer.zero_grad(set_to_none=True)
                mean = refiner(z0,h,zp,steps=depth)
                scale = z0.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
                drift = ((mean-z0)/scale).square().mean()
                reward_std = None
                if args.objective == 'answer_ce':
                    task_loss = -runtime.score(prefix,answer,mean)
                    if not task_loss.requires_grad:
                        raise RuntimeError('Renderer answer loss detached; use explicit answer_pg, not a fake CE loss')
                else:
                    active,actions,sigma,candidates = policy_candidates(mean,answer.shape[1],runtime,args,
                        seed=args.seed+epoch*1000003+row_index)
                    with torch.no_grad():
                        rewards = torch.stack([runtime.score(prefix,answer,z) for z in candidates])
                    reward_std = rewards.std(unbiased=False).item()
                    if reward_std <= 1e-7:
                        flat += 1
                        continue  # do not label drift-only updates as answer training
                    task_loss = answer_policy_loss(active,actions,sigma,rewards)
                loss = task_loss + args.drift_weight*drift
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training objective')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(),args.max_grad_norm)
                if not torch.isfinite(norm) or norm.item() == 0:
                    raise RuntimeError('No finite nonzero refiner gradient from the selected objective')
                if any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Frozen LACES unexpectedly received parameter gradients')
                optimizer.step()
                global_step += 1
                updates += 1
                record = dict(epoch=epoch+1,step=global_step,R=depth,loss=float(loss.detach()),
                              task_loss=float(task_loss.detach()),drift=float(drift.detach()),
                              grad_norm=float(norm),reward_std=reward_std)
                with (out/'training_history.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
            if updates == 0:
                raise RuntimeError('Every training example had flat reward; no reasoning update was performed')
            payload = dict(format=FORMAT,refiner=refiner.state_dict(),refiner_config=refiner.config,
                optimizer=optimizer.state_dict(),base=runtime.audit,base_checkpoint=str(Path(args.ckpt_dir).resolve()),
                protocol=args.protocol,epochs_completed=epoch+1,global_step=global_step,training_args=vars(args))
            tmp = out/'refiner_last.tmp'
            torch.save(payload,tmp)
            tmp.replace(out/'refiner_last.pt')
            print(f'[epoch {epoch+1}] updates={updates} flat_reward_examples={flat}',flush=True)

    refiner.eval()
    validation = evaluate(runtime,refiner,examples,splits['validation'],args)
    # Budget selection uses VALIDATION only. The chosen budget is then frozen for test reporting.
    budget_scores = {int(k.removeprefix('laces_R')):v['contains_answer']
                     for k,v in validation['metrics'].items() if k.startswith('laces_R')}
    best = max(budget_scores.values())
    chosen = min(r for r,score in budget_scores.items() if score >= best-.01)
    test = evaluate(runtime,refiner,examples,splits['test'],args)
    summary = dict(base=runtime.audit,args=vars(args),validation=validation,test=test,
        validation_selected_R=chosen,selection_metric='contains_answer; smallest within 1pp of validation best',
        actual_pretrained_calls=dict(runtime.calls),note='No claim of reasoning improvement follows from successful execution')
    write_json(out/'metrics.json',summary)
    runtime.close()
    return summary


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
