import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()))
import scripts.eval.train_laces_reasoner as t
out=Path('outputs_eval/reasoning_review_2026-09-11')
observed={'scores':[],'identities':[]}
original_score=t.LACESRuntime.score
original_forward=t.LatentTrajectoryRefiner.forward
original_preflight=t.preflight

def score(self,*a,**kw):
 v=original_score(self,*a,**kw)
 observed['scores'].append({'value':float(v.detach()),'raw':kw.get('raw',False),'calls':dict(self.calls)})
 print('TRACE_SCORE',observed['scores'][-1],flush=True)
 return v

def forward(self,z0,*a,**kw):
 v=original_forward(self,z0,*a,**kw)
 if kw.get('steps')==0:
  observed['identities'].append({'steps':0,'equal':torch.equal(z0,v),'same_object':z0 is v})
 return v

def preflight(runtime,refiner,examples,row,args):
 try:
  r=original_preflight(runtime,refiner,examples,row,args)
  observed['report']=r
  return r
 except Exception as e:
  observed['error']=str(e)
  raise
 finally:
  observed['calls']=dict(runtime.calls)
  (out/'preflight_trace.json').write_text(json.dumps(observed,indent=2)+'\n')

t.LACESRuntime.score=score
t.LatentTrajectoryRefiner.forward=forward
t.preflight=preflight
t.main(['--mode','preflight','--output_dir',str(out/'traced'),'--cache_dir','outputs_cache/reasoning_review_2026-09-11'])
