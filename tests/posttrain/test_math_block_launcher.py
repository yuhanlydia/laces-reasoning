import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def test_math_block_launcher_passes_gsm8k_and_math_controls_without_word_splitting(tmp_path):
    script=ROOT/'training/run_math_block_grpo.sh'
    assert script.is_file(),'math block-GRPO launcher is missing'
    capture=tmp_path/'args.json'; fake=tmp_path/'fake-python'
    fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nopen(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env={**os.environ,'PYTHON':str(fake),'CAPTURE':str(capture),
         'CKPT_DIR':'/model path/step_00030000','DATA_DIR':'/gsm data',
         'OUTPUT_DIR':'/output path','MODE':'train','GROUP_SIZE':'8','MAX_BLOCKS':'16',
         'TOKENS_PER_BLOCK':'32','ADVANTAGE_MODE':'center','LR':'3e-7','KL_COEF':'0.05'}
    subprocess.run(['bash',str(script)],check=True,env=env,capture_output=True,text=True)
    args=json.loads(capture.read_text())
    assert args[:2]==['-m','laces_posttrain.run_math_block_grpo']
    assert args[args.index('--ckpt-dir')+1]=='/model path/step_00030000'
    assert args[args.index('--data')+1]=='/gsm data'
    assert args[args.index('--group-size')+1]=='8'
    assert args[args.index('--max-blocks')+1]=='16'
    assert args[args.index('--advantage-mode')+1]=='center'
    assert '--acknowledge-test' not in args
