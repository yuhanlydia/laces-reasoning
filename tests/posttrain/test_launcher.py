import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def test_launcher_passes_paths_and_warmstart_without_word_splitting(tmp_path):
    script=ROOT/'training/run_mmlu_pro_s2.sh'
    assert script.is_file(),'MMLU-Pro direct-S2 launcher is missing'
    capture=tmp_path/'args.json'; fake=tmp_path/'fake-python'
    fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env={**os.environ,'PYTHON':str(fake),'CAPTURE':str(capture),'CKPT_DIR':'/path with spaces/step_00030000',
         'DATA_DIR':'/data bundle','OUTPUT_DIR':'/output here','MODE':'train','OBJECTIVE':'grpo',
         'INIT_S2':'/distill/best_dev.pt','NUM_STEPS':'2','DIFFUSION_STEPS':'4'}
    subprocess.run(['bash',str(script)],check=True,env=env,capture_output=True,text=True)
    args=json.loads(capture.read_text())
    assert args[:2]==['-m','laces_posttrain.run']
    assert args[args.index('--ckpt-dir')+1]=='/path with spaces/step_00030000'
    assert args[args.index('--init-s2')+1]=='/distill/best_dev.pt'
    assert args[args.index('--objective')+1]=='grpo'
    assert '--acknowledge-test' not in args
