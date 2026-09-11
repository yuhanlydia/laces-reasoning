# GRPO strategy review for direct latent S2

This note records the strategy review after the custom MMLU-Pro pilot on 2026-09-11. It applies to the diffusion-transition policy in `laces_posttrain/policy.py`, not to token-level TRL GRPO.

## Evidence from this run

The direct GRPO run used group size 4, one inner epoch, learning rate `1e-6`, `kl_coef=0.01`, and `clip_range=0.2`. At the stop point the policy-gradient loss was still near zero, while the gradient norm was nonzero. The reported reference KL had grown from roughly `5e-4` in the early window to roughly `9e-2` in the last 100 updates. Development accuracy stayed around 14.3–14.5%. This means `pg_loss` alone is not a stopping signal: it is evaluated close to the old-policy rollout, while the policy can still drift through its score-function gradient.

The implementation has several sound properties: rollout actions are detached, the replay ratio uses the joint Gaussian transition log probability, each transition remains stochastic, the parent S0/S1/RWKV path is frozen, and the update is on-policy with one inner epoch. The consistently zero clip fraction says that the chosen update is not reaching the PPO clip boundary; it does not prove that the objective is effective.

## Recommended ablations

1. **Dr.GRPO advantage.** The original GRPO normalizes group rewards by their standard deviation. The Dr.GRPO analysis reports that response-length and standard-deviation normalization introduce optimization bias, and removes both terms. Our latent horizon is fixed, so the immediate ablation is `A_i = R_i - mean(R)` without division by group standard deviation. This matters here because reward standard deviation is often only about `0.03–0.04`; dividing by it amplifies small score noise. Keep the original normalized form as the control.

2. **Larger groups.** A group of four gives a noisy relative baseline and frequently produces little diversity. Compare group sizes 8 and 16, recording the fraction of flat groups, reward quantiles, unique option choices, and group accuracy. Increase group size before increasing the learning rate.

3. **Adaptive KL.** Use a target range for the measured reference KL rather than a permanently weak coefficient. Start with a lower learning rate (`3e-7`) and an ablation with a stronger KL coefficient; keep the run if KL remains bounded and held-out accuracy improves. The current joint PPO ratio and per-coordinate KL are not on the same aggregation scale, so log both scales before tuning the coefficient.

4. **Reward alignment.** `gold_logprob` is dense, but it can reward confidence changes without changing the discrete answer. Compare it with exact `accuracy` using a larger group, and optionally an answer-margin reward (`gold score - best non-gold score`) that is still tied to the verified answer. Do not use training reward as a benchmark; report fixed development and sealed test accuracy.

5. **Diagnostics and stopping.** Add ratio mean/max, advantage standard deviation, flat-group rate, entropy/action standard deviation, reward quantiles, and current-vs-parent accuracy. Stop on a held-out plateau together with bounded KL and no increase in invalid or collapsed-choice rates. Do not stop because `pg_loss` is near zero alone.

The current implementation already has dynamic skipping for flat groups and one on-policy epoch. DAPO-style decoupled clipping and dynamic sampling are reasonable later experiments, but should follow the smaller ablations above so a gain can be attributed to one change.

## Primary references

- DeepSeekMath introduced GRPO as a PPO variant using group-relative rewards and no separate value model: <https://arxiv.org/abs/2402.03300>.
- The Dr.GRPO analysis identifies length and standard-deviation normalization bias and proposes removing those terms: <https://arxiv.org/abs/2503.20783>.
- DAPO describes decoupled clipping and dynamic sampling for large-scale RL: <https://arxiv.org/abs/2503.14476>.

These references motivate ablations; they do not establish that any variant will improve this LACES latent policy.
