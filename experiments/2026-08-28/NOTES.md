# 2026-08-28 — E6: BDH dynamic reasoning writer（第 15 节"第一版"）

## 目标（用户 5-15 节的重训计划第 15 节）
把「表达」和「思考」拆开。不再让 32-D latent / 16-basis 承载 reasoning，而是：
```
H_{1:T} --cross-attn--> W_0 ∈ R^512
W_{r+1} = GRU(W_r, CrossAttn(W_r, H)),  R=4
U_l(W_R), V_l(W_R)  (dynamic per-layer directions)
ΔS_l = U_l V_l^T    (rank 32)
```
只训 `L = ||ΔS − ΔS*||²`（state-matching，FLA RWKV 核不反传 recurrent state）。
**Gate**：held-out 2-hop/3-hop 上 MSE<7 且 acc>50%。

## 实现与调试过程（关键教训）

### 教训 1：跨层共享方向是 bug（E5 的错误沿用）
第一版用 `U_core @ M_l`（U_core 跨层共享，M_l 每层）。**连 4 个任务都 overfit 不到**
（MSE 卡 28.5）。根因：dS\* 各层方向独立，共享 U_core 只有 32 个方向，装不下。

### 教训 2：RWKV state 是 per-head，应做 per-head 分解
state = [H=40, D=64, D=64]。改为 per-head rank-r_h：`ΔS_{l,h} = Σ_m u_{l,h,m} v_{l,h,m}^T`，
u/v 由 head 从 W_R 动态产生（真·每层每头 input-dependent 方向）。

### 教训 3：raw MSE 尺度不匹配 → 加方向余弦 loss
U@V 乘积使输出起始 ~1e-4（对比 dS\* 的 O(1)），梯度极微。改 combined loss：
`(1 − mean cos) + λ·rel_mse/L`（尺度不变）。

### Overfit 诊断（4 任务，300 epoch，token-level 输入）
| r_h | params | overfit MSE |
|---|---|---|
| 1 | 43M | 18.85 |
| 2 | 85M | 12.95 |
| 4 | 170M | 8.17 |
（E2 flattened rank-32 地板 = 6.4；E5 mean-pool 输入 = 10.7）
→ per-head 分解对，且 token 输入有用；要过 MSE<7 需 r_h≥8。

## 全量训练（已完成）
`--r_h 8 --ws_dim 512 --n_steps 4 --lr 2e-3 --n_train 120 --n_test 40 --epochs 500`
writer params = 676M。

**结果**：test MSE = **12.27**（overfit 8.17 → 泛化 12.27，明显过拟合），acc = **0%**。
**Gate 未过**（MSE<7=False, acc>50%=False）。

### 结论（重要 negative result）
1. 架构对（per-head 动态 + token 输入能 overfit 到 8.17，逼近 6.4 地板）。
2. 但**不泛化**：676M 参数在 240 合成任务上过拟合，held-out 12.27（甚至差于 E5 的 10.7）。
3. 根因（印证用户第 10/17 节）：state-matching 逼 writer 复现**完整** state change（高秩），
   而答案只依赖**低秩 behavioral 子空间**，state-matching 无法定位它。

### 下一步（按用户 10/11 节）
- **零阶 behavioral loss**（第 10 节）：对候选 ΔS 做前向 gold-lp 评测，训 surrogate/value head，
  直接优化"改答案"而非"复现 state"。
- **curriculum 数据**（第 11 节）：单事实→两跳→三跳→冲突，而非直接合成任务。
- 降低过拟合：weight decay / 更少参数 / 更多样数据。

---

## E7 — 零阶 behavioral loss（NES 随机方向）❌ 失败（08-30）

`scripts/eval/train_behavioral_writer.py`：state-matching + NES 零阶行为梯度（K 个随机方向估 ∇_ΔS r）。

| 配置 | MSE | acc |
|---|---|---|
| r_h=4 K=4 λ_b=0.3 | 23.11 | 0% |
| r_h=4 K=4 λ_b=1.0 | 23.13 | 0% |
| r_h=8 K=4 λ_b=0.3 | 27.02 | 0% |

**关键发现**：所有 run 的 `loss ≈ state`（behavioral 项贡献 ≈0）。λ_b 从 0.3→1.0 无变化（23.11 vs 23.13）。
**根因**：reward r(ΔS)=gold-lp 在**随机方向上是平坦的**（E3 已证：行为敏感度集中在 gold 方向，
随机方向 g≈0）。所以随机方向 NES 梯度 ≈0，反而只加了噪声（MSE 23 比 E6 的 12.27 更差）。

### 结论：随机方向 NES 是第 10 节的错误实现
第 10 节正确的方向（用户原文）是：**scale search along predicted ΔS**（η∈{0,0.5,1} 沿预测方向），
或 **value head V(W,ΔS)≈r 的 surrogate**，而非随机方向 NES。

### 下一步（修正）
- 用 **gold 方向 dS\***（或残差 dS\*−ΔS）作为零阶搜索方向（reward 沿它非零）。
- 或 scale-search + value head surrogate。
- 或 E3 的逐层行为敏感度加权 state-matching（第 9 节 layer weighting）。

---

## E8 — gold 方向 behavioral + curriculum + layer weighting（08-30）

`scripts/eval/train_behavioral_golddir.py`。4 卡并行变体：

| 变体 | tasks | layer-w | MSE | acc |
|---|---|---|---|---|
| e8_base | 2a+3a | ✗ | 32.08 | 0% |
| e8_curric | 1a+2a+3a | ✗ | 32.13 | 0% |
| e8_layerw | 1a+2a+3a | ✓ | **19.70** | 0% |
| e8_fullcurric | 1a+2a+3a+4a+conflict | ✓ | 19.97 | 0% |

**两个发现**：
1. **gold 方向行为信号也 ≈0**（loss≈state，同 E7）→ 零阶 behavioral loss 从根本上无效：
   reward r(ΔS)=gold-lp 在离答案阈值远处**平坦/尖锐**，任何方向的零阶梯度 ≈0。
2. **layer weighting 治好了训练不稳定**（32→19.7）：无权重时 MSE 项(~32)主导 loss 导致振荡不学；
   按逐层行为敏感度归一化后 loss 平衡（~2）才正常训练。但**仍 0% acc**。
3. curriculum（更多任务类型）没帮助（32.13 vs 32.08；19.97 vs 19.70）。

⚠️ bug 记录：无 layer-w 时 state-loss 的 MSE 项没除以 L（应 rel_mse/L），导致尺度失衡、训练不学。
layer-w 版因权重归一化到和=1，天然规避了此 bug。

## E5–E8 全链路结论（训练 generalizing writer 全线未过 gate）

| 实验 | 目标 | MSE | acc | 判读 |
|---|---|---|---|---|
| E5 mean-pool+fixed dirs | state-match | 10.7 | 0% | 过拟合，不泛化 |
| E6 token+per-head dynamic | state-match | 12.27 (overfit 8.17) | 0% | 泛化 gap 4.1 |
| E7 NES 随机方向 | behavioral | 23 | 0% | reward 平坦，信号≈0 |
| E8 gold方向+curriculum+layerw | behavioral+数据 | 19.7 | 0% | 行为仍≈0，curriculum 无效 |

**根因（两层）**：
1. **state-matching 错位**：逼 writer 复现完整高秩 dS\*，但答案只依赖低秩 behavioral 子空间（第 17 节）。
2. **零阶 behavioral 无效**：reward 尖锐/平坦，零阶梯度 ≈0（唯一非零信号是"答案对/错"的二值，不可导）。

**诚实结论**：瓶颈定位清楚了（16-basis，E1-E4 solid）；rank≥32 writer 在 oracle 能承载（E2 60%），
但**训练一个能泛化的 writer 是开放难题**——合成实体池太小 + "facts→dS\*" 映射本质是 RWKV 自身
recurrent 计算的复现。E5-E8 的 state-matching / zero-order behavioral 都过不了 gate。
