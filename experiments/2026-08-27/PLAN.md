# 2026-08-27 — LACES Writable-State Capacity Audit + BDH Reasoner Plan

## 1. 今天跑的实验（capacity audit，零训练）

脚本：`experiments/2026-08-27/capacity_audit/diag_capacity_audit.py`
结果：`experiments/2026-08-27/capacity_audit/audit_2a3a.json`

**Champion ckpt**: `outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000`
**任务**: 2-agent 两跳 + 3-agent 三跳 fact-chain，各 20 题（40 题）。

### 定义
- `S_context` = 冻结 RWKV 只读 question 后的 recurrent state
- `S_gold` = 冻结 RWKV 读 (facts + question) 后的 recurrent state
- `dS* = S_gold - S_context` = 要答对所需的 **残差修正**

### 三层 capacity audit（都做残差写 `S = S_context + dS`，匹配 dS\*）
| Level | 可写对象 | 参数化 |
|---|---|---|
| L1 (z) | latent → head → basis | `dS = basis @ alpha_heads(z)` |
| L2 (α) | 直接 basis 系数 | `dS = basis @ alpha` |
| L3 (ΔS) | 无限制 low-rank delta | `dS_l = U_l C_l V_l^T` (rank r=4) |

### 结果（40 题）
| 条件 | acc | 说明 |
|---|---|---|
| text_concat | **100%** | oracle（backbone 会做） |
| inject_gold | **92.5%** (37/40) | 注入真 S_gold = 状态注入天花板 |
| R0_dual | **100%** | 现有方法（残差 plan + seq carryover） |
| L1_z | **0%** | 优化 z 无法改变答案 |
| L2_alpha | **0%** | 优化 α 无法改变答案 |
| L3_delta | **0%** | 优化 low-rank ΔS 无法改变答案 |

**匹配 MSE（越低越好，L=32 表示"完全没匹配上，优化态≈0"）**
| Level | MSE |
|---|---|
| L1_z | 32.01 |
| L2_alpha | 31.998 |
| L3_delta | **24.09** |

**basis 投影误差**
- `e16_full = 0.9999`：16-basis 张成的子空间对完整 S_gold 只捕获 ~0.01% 能量
- `e16_delta = 0.9999`：对残差 dS\* 同样只捕获 ~0.01%

### 结论（对应顾问第 5–18 节的判断，全部被证实）
1. **16 个固定 basis 是硬瓶颈**。`e16_delta≈1` = basis span 与"facts 修正方向"几乎正交。
2. **瓶颈在 S1（z→state writer），不在 RWKV attractor，也不在 32-D latent**。
   - L1 ≈ L2 ≈ 32（两者都经过 basis，都 ~0% 捕获）→ 不是 32-D 太窄，是 basis span 太小。
   - L3（绕开 basis 的 low-rank delta）= 24（捕获 ~25%）> L2 → 证明 **basis span 才是瓶颈**。
3. **但 L3 仍 0% acc**：rank-4 delta 捕获 25% 能量仍不足以翻转答案。需要更高 rank，或
   state-matching(MSE) 目标与"答案正确性"不完全对齐（25% 可能落在与答案无关的方向）。
4. **反传验证**：FLA 的 RWKV-7 核**不反传 recurrent state 梯度**（logits 无 grad_fn，即使去掉
   `@no_grad`）。所以 `∂ℓ/∂S`、answer-loss 优化、J3=∂ℓ/∂S 的谱都无法用反传做 → 只能：
   (a) state-matching 目标（已用），(b) 零阶/有限差分行为敏感度（下一步）。

### 对旧失败的统一解释（顾问第 6 节）
GRPO / diversity / SIM-CoT 全失败，因为 `∂y/∂Z ≈ 0`：Z 的多样性（cos 1.0→0.64）经
16-basis writer 后被压平（`R_ω(Z1)≈R_ω(Z2)`），advantage 恒为 0。reward 设计没错，是
writable behavioral rank ≈ 0。

---

## 2. 统一模型（顾问第 18 节，最终方向）

不再 `W_R → Z → αB → RWKV`，而是分开两个 writer：

```
W_R →  {  Z_plan → α B                          (stable future-memory planning / steering)
          ΔS_reason = U(W_R) V(W_R)^T           (behaviorally expressive reasoning write)
       }
S' = S_memory + λ_p · S_plan + λ_r · ΔS_reason
```

- `S_memory` = evidence（seq carryover，不动）
- `W_R` = thinking（高维 workspace，BDH 式迭代）
- `α B` = stable steering（保留现有 LACES，λ_r=0 时退化为原 LACES，nested formulation）
- `U V^T` = reasoning intervention（dynamic low-rank，方向随 W 变，不像 basis 方向固定）

**核心 claim（论文 observation，不是工程 bug）**：
> Low-rank *fixed* state bases are sufficient for language steering but insufficient for
> latent reasoning, because reasoning requires high *behavioral rank* at the writable memory
> interface. (`latent capacity ≠ writable state capacity ≠ behavioral capacity`)

---

## 3. 下一步实验（按优先级，全部零训练 / 前向，可在剩余 GPU 并行）

### E2: Low-rank delta 的 rank sweep（GPU1）
对 L3 做 `r ∈ {4, 8, 16, 32, 64}` 的 sweep，回答"dynamic low-rank writer 到底需要多少 rank
才能翻转答案"。（r=4 已证明不够，25% 能量 / 0% acc。）

### E3: basis 方向的行为敏感度（GPU2）
对每个 basis 方向 `B_k`（逐层或聚合），注入 `S_context + ε·B_k`，测 gold-token logprob 变化：
`g_k = (logprob(+ε) - logprob(-ε)) / (2ε)`（零阶，前向）。若 16 个方向 `g_k ≈ 0`，则
basis 是 behaviorally inert（r_B ≈ 0），直接坐实顾问第 16 节"basis 学成了 LM-steering
subspace 而非 capability subspace"。（paper figure 素材：basis 方向 ‖B_k‖_F 大但 ∂ℓ/∂S·B_k ≈ 0）

### E4: 4agent / conflict capacity audit（GPU3）
更长链（4 跳）+ 冲突（2 agent 矛盾），看 basis 瓶颈是否随事实数量/冲突恶化。

### E5（训练，最后做）: BDH reasoner + dynamic low-rank writer
训练 `F_ρ`（workspace updater）+ `P_ξ`（W→Z）+ `U/V`（dynamic writer），λ_r 从 0 起
（nested 初始化为原 LACES）。这是"long goal"的主体。

---

## 4. 文件布局（今天全部归档到 experiments/2026-08-27/）
```
experiments/2026-08-27/
  PLAN.md                      本文件
  NOTES.md                     原始实验笔记/日志
  capacity_audit/              实验 1（已完成）
  rank_sweep/                  实验 2
  behavioral_sensitivity/      实验 3
```
