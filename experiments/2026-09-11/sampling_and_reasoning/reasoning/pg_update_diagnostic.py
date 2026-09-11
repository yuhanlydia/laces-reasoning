import sys,json,statistics
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()))
import scripts.eval.train_laces_reasoner as t
out=Path('outputs_eval/reasoning_review_2026-09-11')
a=t.parse_args(['--cache_dir','outputs_cache/reasoning_review_2026-09-11'])
torch.manual_seed(a.seed)
m,_,tok,c,_=t.load_relay_model(a.ckpt_dir,a.device)
r=t.LACESRuntime(m,c,plan_steps=a.plan_steps,cfg_scale=a.cfg_scale,blend=a.blend);del c
f=t.LatentTrajectoryRefiner(m.latent_dim,m.hidden_size,width=a.width,max_step=a.max_step).to(a.device)
e=t.Examples(r,tok,a);row=t.build_splits(a)['validation'][0]
p,ans,z,h,zp=e.get(row)
report={'row':row,'prefix_tokens':p.shape[1],'answer_tokens':ans.tolist(),'scores':{}}
with torch.no_grad():
 for label,raw in [('laces',False),('raw',True)]:
  vals=[float(r.score(p,ans,z,raw=raw)) for _ in range(8)]
  report['scores'][label]={'values':vals,'std':statistics.pstdev(vals),'range':max(vals)-min(vals)}
  print(label,report['scores'][label],flush=True)
 _,_,_,cs=t.policy_candidates(z,ans.shape[1],r,a,seed=a.seed+1)
 rewards=[float(r.score(p,ans,v)) for v in cs]
 report['candidate_rewards']=rewards;report['candidate_reward_std']=statistics.pstdev(rewards)
 for label,raw in [('laces',False),('raw',True)]:
  ids,_=r.generate(p,z,max_new_tokens=32,raw=raw,eos_id=tok.eos_token_id)
  report[label+'_text']=tok.decode(ids,skip_special_tokens=False)
before={k:v.detach().clone() for k,v in f.state_dict().items()}
opt=torch.optim.AdamW(f.parameters(),lr=1e-4,weight_decay=1e-4)
mean=f(z,h,zp,steps=1)
active,actions,sigma,candidates=t.policy_candidates(mean,ans.shape[1],r,a,seed=a.seed+2)
with torch.no_grad():
 rewards=torch.stack([r.score(p,ans,v) for v in candidates])
loss=t.answer_policy_loss(active,actions,sigma,rewards)
loss.backward()
grad=torch.nn.utils.clip_grad_norm_(f.parameters(),1.)
report['pg_update']={'loss':float(loss.detach()),'reward_std':float(rewards.std(unbiased=False)),
 'grad_norm':float(grad),'frozen_base_has_grad':any(v.grad is not None for v in m.parameters())}
assert torch.isfinite(grad) and grad>0
assert not report['pg_update']['frozen_base_has_grad']
opt.step()
report['pg_update']['changed_refiner_tensors']=[k for k,v in f.state_dict().items() if not torch.equal(v,before[k])]
assert report['pg_update']['changed_refiner_tensors']
print('PG_UPDATE',report['pg_update'],flush=True)
try:
 probe=z.detach().requires_grad_();lp=r.score(p,ans[:,:2],probe)
 g=torch.autograd.grad(lp,probe,allow_unused=True)[0] if lp.requires_grad else None
 report['ce_gradient']={'loss_requires_grad':lp.requires_grad,'norm':None if g is None else float(g.norm()),'finite':None if g is None else bool(torch.isfinite(g).all())}
except Exception as ex: report['ce_gradient']={'error':type(ex).__name__+': '+str(ex)}
report['actual_calls']=dict(r.calls)
(out/'pg_update_diagnostic.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
