# 2026-08-27 — 实验结果笔记

## Champion
`outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000`
（frozen RWKV-7 2.9B + S0(32-d latent) + S1(linear bridge, 16 basis/layer) + S2(BiRWKV ddpm)）
`state_scale = 0.1387`（plan 是"小 steering signal"，非 full memory 写）

## 关键机制约束
- **FLA 的 RWKV-7 核不反传 recurrent state 梯度**（去掉 @no_grad 后 logits 仍无 grad_fn）。
  → answer-loss 优化、∂ℓ/∂S、J3 谱都不可用反传做。只能用 (a) state-matching，(b) 零阶行为敏感度。

---

## E1 — Capacity audit（2agent+3agent, n=40, 3-level 优化，残差写）
| 条件 | acc | 含义 |
|---|---|---|
| text_concat | 100% | oracle |
| inject_gold | 92.5% | 注入真 S_gold = 状态注入天花板 |
| R0_dual | 100% | 现有方法（plan + seq carryover） |
| L1_z / L2_alpha / L3_delta | 0% | 优化 z / α / low-rank ΔS 都无法翻转答案 |

- MSE（匹配 dS*，L=32 = 完全没匹配）：L1=32.01, L2=32.00, L3=**24.09**
- **e16_full = 0.9999, e16_delta = 0.9999**：16-basis 张成子空间对完整 S_gold 和残差 dS* 都只捕获 ~0.01%。

**结论**：L1≈L2≈32（都过 basis）<< L3=24（绕开 basis）→ **固定 basis span 是硬瓶颈**，不是 32-D latent。

## E2 — Rank sweep（2agent+3agent, n=20，dynamic low-rank writer）
| rank | svd_err(能量残差) | mse | acc |
|---|---|---|---|
| 4 | 0.740 | — | 0% |
| 8 | 0.598 | 19.36 | 0% |
| 16 | 0.405 | 13.19 | 0% |
| 32 | 0.184 | 6.39 | **60%** |
| 64(=full) | 0.000 | 0.00 | **100%** |

**结论**：dynamic low-rank writer 方向正确，但需要 **rank ≈ 32+**（捕获 82% 能量 → 60% acc），
rank 64（full）才 100%。r=4/8（顾问最初建议）远不够。

## E3 — Behavioral sensitivity（2agent, n=20，零阶，固定幅值 ‖dS*‖）
| 方向 | gold-lp |
|---|---|
| base（floor） | -18.01 |
| gold 方向 dS*（ceiling） | **-8.32**（+9.7 提升） |
| random（对照） | -17.24 |
| 16 个 basis 方向 | -19.8 … -13.6（均值 ≈ -17） |

- 11/16 basis 方向略优于 random，但 **0/16 接近 gold 方向**。
- 最佳 basis 方向（k=13, lp=-13.6）也只恢复 gold 提升的 ~45%，且是孤点。

**结论**：basis 是 **behaviorally inert** 的 language-steering subspace（‖B_k‖_F≈48 大，
但对答案的 ∂ℓ/∂S·B_k ≈ 0），不是 reasoning writable subspace。

## E4 — Capacity audit（4agent+conflict, n=40）
| 条件 | acc |
|---|---|
| text_concat | 77.5% |
| inject_gold | 67.5% |
| R0_dual | **82.5%** |
| L1/L2/L3 | 0% |

- 更长链（4 跳）/冲突任务：状态注入天花板下降（92.5%→67.5%），但 R0_dual 仍 82.5%（甚至超 text_concat）。
- e16_full=e16_delta=0.9999，L3 mse=24.05（同 E1 模式）。

---

## E5 — Trained dynamic low-rank writer（BDH workspace, rank=32, pooled-hidden input）
`scripts/eval/train_dynamic_writer.py`，n_train=400 / n_test=80，600 epoch full-batch，lr=2e-3。

参数化：每层 r=32 个 learned rank-1 方向 `U_l [r,H*D], V_l [r,D]`，workspace(BDH 4 步 recurrent)
→ 系数 `c [B,L,r]`，`ΔS_l = Σ_m c_m (u_m v_m^T)`。监督 = state-matching（dS*）。

| 指标 | 值 |
|---|---|
| 训练 loss | 32 → **10.72**（480/540/600 epoch 都 ~10.7，已 plateau） |
| 测试 MSE | 10.72（train≈test，无过拟合 gap） |
| dynamic_writer acc | **0%** |
| 对照 | text_concat 100% / inject_gold 100% |

**关键**：loss plateau 在 **10.7**，高于 rank-32 的表示地板（E2 SVD floor = **6.4**）。
→ 瓶颈**不是** writer 的 rank（32 足够），而是 **pooled-hidden 输入携带的信息不足以
重建 dS***（均值池化丢实体级信息）。MSE 10.7 对应 0% acc（与 E2：13.19→0%、6.39→60% 一致）。

**结论深化**：bottleneck 不止 basis rank，而是「输入 → state correction」映射本身难学。
即便把固定 basis 换成 rank-32 动态 writer，从 compact 输入（pooled hidden）也无法把
behavioral rank 拉起来 → 印证顾问第 17 节 `latent capacity ≠ writable state capacity ≠
behavioral capacity`，且**输入信息容量**是另一层限制。下一步应试更 rich 的输入
（全序列 hidden states / 真实 recurrent state S_facts），而非只加 rank。

## 综合结论（核心 claim）
1. **16 个固定 basis 是 hard bottleneck**：e16_delta≈1.0，L1≈L2≈32。basis span 与 reasoning
   修正方向几乎正交，无法承载多跳/冲突所需的信息。
2. **basis 是 language-steering subspace，不是 capability/reasoning subspace**（E3 实证：
   behaviorally inert）。这解释了 GRPO/diversity/SIM-CoT 全失败（∂y/∂Z≈0，advantage 恒 0）。
3. **dynamic low-rank writer 可行但需要 rank≈32+**（E2：r32=60%, r64=100%），非 r=4/8。
4. **统一模型验证**：`S = S_memory + λ_p·(αB) + λ_r·(U(W)V(W)^T)`，其中 αB 保持 steering
   （λ_r=0 → 原 LACES，nested），U V^T 提供 behavioral rank，且 rank 要 ≥32。

### 论文 observation（不是 bug）
> Low-rank *fixed* state bases are sufficient for language steering but insufficient for
> latent reasoning: reasoning requires high *behavioral rank* at the writable memory interface
> (`latent capacity ≠ writable state capacity ≠ behavioral capacity`)，且 behavioral rank 需要
> 由 dynamic（方向随输入变的）writer 提供，rank 至少 ~32。

---

## 文件清单（今天）
| 文件 | 说明 |
|---|---|
| `scripts/eval/diag_capacity_audit.py` | E1/E4 三层容量审计 |
| `scripts/eval/diag_rank_sweep.py` | E2 rank sweep |
| `scripts/eval/diag_behavioral_sensitivity.py` | E3 行为敏感度 |
| `results/capacity_audit/audit_2a3a.json` | E1 结果 |
| `results/capacity_audit/audit_4a_conflict.json` | E4 结果 |
| `results/capacity_audit/rank_sweep.json` | E2 结果 |
| `results/capacity_audit/behavioral_sensitivity.json` | E3 结果 |
