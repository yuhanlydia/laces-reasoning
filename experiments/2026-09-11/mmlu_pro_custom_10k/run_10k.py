import json,os,subprocess,time
from pathlib import Path
root=Path('/root/laces-reasoning'); out=root/'outputs_eval/mmlu_pro_custom_10k'; out.mkdir(parents=True,exist_ok=True)
base=dict(os.environ,PYTHON=str(root/'.venv/bin/python'),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',GPU='0',MODE='train',NUM_STEPS='10000',GROUP_SIZE='4',LR='1e-6',DIFFUSION_STEPS='32',NATIVE_CONTROL_STEPS='1000',EVAL_EVERY='500',SAVE_EVERY='100',DEV_LIMIT='0',DATA_DIR=str(root/'data/mmlu_pro_custom_s42'),CKPT_DIR=str(root/'outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000'))
runs={'distill':{'OBJECTIVE':'distill','OUTPUT_DIR':'results/mmlu_pro/custom_10k_distill_s42'},'grpo_direct':{'OBJECTIVE':'grpo','OUTPUT_DIR':'results/mmlu_pro/custom_10k_grpo_direct_s42'},'distill_grpo':{'OBJECTIVE':'grpo','OUTPUT_DIR':'results/mmlu_pro/custom_10k_distill_grpo_s42','INIT_S2':'results/mmlu_pro/custom_10k_distill_s42/best_dev.pt'}}
active={};done={};started={};peak=0

def start(name):
 env=dict(base,**runs[name]);log=(out/(name+'.log')).open('w')
 p=subprocess.Popen(['bash','training/run_mmlu_pro_s2.sh'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 active[name]=(p,log);started[name]=time.monotonic();print('START',name,'pid',p.pid,flush=True)
start('distill');start('grpo_direct')
with (out/'gpu_utilization.csv').open('w') as monitor:
 monitor.write('timestamp, memory_used_mib, utilization_percent, power_w\n')
 while active:
  try:
   row=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
   monitor.write(row+'\n');monitor.flush();peak=max(peak,int(row.split(',')[1]))
  except Exception: pass
  for name,(p,log) in list(active.items()):
   rc=p.poll()
   if rc is None:continue
   log.close();del active[name];done[name]={'exit_code':rc,'seconds':round(time.monotonic()-started[name],1),'output_dir':runs[name]['OUTPUT_DIR']}
   print('FINISH',name,done[name],flush=True)
   if rc: print((out/(name+'.log')).read_text()[-3500:],flush=True)
  if 'distill' in done and 'distill_grpo' not in active and 'distill_grpo' not in done:
   if done['distill']['exit_code']==0:start('distill_grpo')
   else:done['distill_grpo']={'skipped':'distillation did not complete'}
  (out/'status.json').write_text(json.dumps({'runs':done,'active':{n:{'pid':p.pid,'output_dir':runs[n]['OUTPUT_DIR']} for n,(p,log) in active.items()},'queued':(['distill_grpo'] if 'distill_grpo' not in active and 'distill_grpo' not in done else []),'peak_gpu_memory_mib':peak},indent=2)+'\n')
  if active:time.sleep(5)
print('ALL_DONE',json.dumps(done), 'peak_gpu_memory_mib',peak,flush=True)
