# AAAI-27 Multi-Agent Experiments Framework

## Overview

统一框架，用于比较 4 种多智能体通信方法在 3 个任务上的表现。

### Tasks

1. **GSM8K** (200 samples): Sequential Latent Reasoning
   - 4 agents: Planner → Critic → Refiner → Solver
   - 数学推理任务

2. **HiddenBench** (65 tasks): Distributed Evidence Fusion
   - 4 agents, each receives shared + private evidence
   - 需要融合分布式信息

3. **StepGame** (400 samples): Online Collaborative Reasoning
   - N agents (2/4/6/8 hops), each processes one spatial relation
   - 多跳空间推理

### Methods

1. **Raw Qwen** (Qwen2.5-3B): Full context baseline
   - 单模型看所有文本
   - 最快，payload 最小

2. **TextMAS-Qwen** (Qwen2.5-3B): Text sequential passing
   - Agents 顺序处理，传递文本输出
   - 中等速度

3. **LatentMAS** (Qwen2.5-3B): Latent thoughts + cumulative KV
   - Agents 生成 latent thinking tokens
   - KV cache 累积传递
   - payload ~130 MB

4. **LatentWeave** (RWKV-7 2.9B): Plan + recurrent state
   - Agents 编码成 latent plan
   - Recurrent state 固定大小传递
   - payload 固定 20 MB

## Usage

### Run all tasks and methods

```bash
python -m scripts.eval.aaai27.run_all --all
```

### Run specific tasks

```bash
python -m scripts.eval.aaai27.run_all \
  --tasks gsm8k hiddenbench \
  --methods latentweave raw_qwen \
  --n_samples 50 \
  --device cuda:0 \
  --output_dir outputs_eval/aaai27
```

### Run specific task

```bash
# GSM8K only
python -m scripts.eval.aaai27.run_all --tasks gsm8k --n_samples 200

# HiddenBench only
python -m scripts.eval.aaai27.run_all --tasks hiddenbench

# StepGame only
python -m scripts.eval.aaai27.run_all --tasks stepgame --n_samples 400
```

### Run specific methods

```bash
# Only our method vs baseline
python -m scripts.eval.aaai27.run_all \
  --methods latentweave raw_qwen \
  --tasks gsm8k hiddenbench stepgame
```

## Output Format

Results saved to `outputs_eval/aaai27/results_final.json`:

```json
[
  {
    "task_id": "gsm8k_0",
    "task_name": "gsm8k",
    "gold_answer": "18",
    "results": {
      "raw_qwen": {
        "answer": "18",
        "is_correct": true,
        "time_ms": 427,
        "payload_bytes": 1700,
        "metadata": {}
      },
      "latentweave": {
        "answer": "18",
        "is_correct": true,
        "time_ms": 22718,
        "payload_bytes": 20971520,
        "metadata": {"state_bytes": 20971520}
      }
    }
  }
]
```

## Key Metrics

- **Accuracy**: 正确答案比例
- **Time**: 平均推理时间 (ms)
- **Payload**: 通信量 (MB)
  - Raw Qwen: 文本大小
  - TextMAS: 累积文本大小
  - LatentMAS: KV cache 大小 (~130 MB)
  - LatentWeave: 固定 state 大小 (20 MB)

## Expected Results

基于 smoke test (5 GSM8K samples):

| Method | Accuracy | Time | Payload |
|--------|----------|------|---------|
| raw_qwen | 20% | 427ms | 0 MB |
| textmas_qwen | 20% | 10.9s | 0 MB |
| latentmas | 20% | 5.0s | 133 MB |
| latentweave | 20% | 22.7s | 20 MB |

**Key insight**: LatentWeave 的 payload 固定 20 MB，不随 agent 数量增长，而 LatentMAS 的 KV cache 会累积。

## File Structure

```
scripts/eval/aaai27/
├── common.py           # 共享接口
├── run_all.py          # 主入口
├── methods/
│   ├── raw_qwen.py     # Raw Qwen 方法
│   ├── textmas_qwen.py # TextMAS-Qwen 方法
│   ├── latentmas.py    # LatentMAS 方法
│   └── latentweave.py  # LatentWeave 方法
└── tasks/
    ├── gsm8k.py        # GSM8K 任务
    ├── hiddenbench.py  # HiddenBench 任务
    └── stepgame.py     # StepGame 任务
```

## Requirements

- Qwen2.5-3B: `/inspire/hdd/global_user/zhangjiaquan-253108540222/models/Qwen2.5-3B`
- RWKV-7 2.9B: `outputs_relay/C-LDLM-4096-coadapt-8h200-b5-rnn-20260708/step_00022500`
- HiddenBench: `data/hiddenbench/benchmark.json`
- GSM8K: HuggingFace `gsm8k` dataset
- StepGame: HuggingFace `michaelszx/StepGame` dataset

## Notes

- GSM8K 准确率较低（20%），因为 3B 模型在数学推理上能力有限
- LatentWeave 时间较长（22s），因为需要扩散采样（100 steps）
- 建议先用 `--n_samples 10` 做 smoke test，再跑完整实验
- 完整实验（200 GSM8K + 65 HiddenBench + 400 StepGame = 665 tasks）预计需要 4+ 小时
