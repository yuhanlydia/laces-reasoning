# Block Diffusion GRPO for LACES S2

## Goal

Train only the original LACES S2 diffusion model with verifiable reasoning rewards while keeping S0, dynamic S1, and RWKV frozen. MMLU-Pro remains an evaluation task. GSM8K and MATH provide multi-block reasoning rollouts that can exercise all 16 latent blocks.

This is a latent block diffusion policy. It is related to block diffusion language models, but it is not a masked-token diffusion model: S2 generates a continuous latent trajectory, dynamic S1 maps each latent block into RWKV cache states, and RWKV generates tokens causally.

## Axes and credit assignment

The checkpoint has 16 latent blocks, 32 downstream tokens per block, and a configurable number of reverse diffusion transitions (32 in the full configuration). These axes must remain distinct.

For rollout group member `g`, reverse transition `t`, and latent block `h`, the diagonal Gaussian policy supplies a block-factor log probability

`log pi_theta(z[t+1,h] | z[t], condition)`.

The denoiser may couple all blocks through its mean prediction; the action density still factors over coordinates given the current state. Replay stores detached state, action, old mean, standard deviation, transition index, and block index. PPO ratios are calculated per block by summing over latent dimensions, not over all blocks at once.

## Causal block rollout

For each problem, S2 first samples one 16-block latent plan. Frozen dynamic S1 and RWKV then execute it causally:

1. Inject latent block `h` through dynamic S1.
2. Generate at most 32 new RWKV tokens.
3. Preserve the resulting cache for block `h+1`.
4. Stop on EOS or after 16 blocks.

No gold answer appears in the generated prompt or token stream. Generated text, token IDs, block boundaries, and cache-independent audit metrics are recorded.

## Verifiable rewards

Each example has a canonical final answer and a dataset-specific verifier. GSM8K accepts normalized numeric answers. MATH supports conservative normalization for common boxed numeric, fractional, and symbolic answers; unrecognized outputs receive zero rather than heuristic credit.

At each completed output block, compute a frozen-model potential by asking for the canonical answer continuation after the generated prefix:

`phi[h] = log p_frozen(canonical answer | prompt, generated blocks <= h, latent blocks <= h)`.

The dense shaped reward is `r[h] = phi[h] - phi[h-1]`. The last emitted block also receives an exact-answer and valid-format reward. With discount one, potential differences telescope and retain the final potential objective. Potential, exact, and format components are logged separately so reward hacking is visible.

Block return-to-go is `G[h] = sum_{k=h}^{H-1} r[k]`. Advantages are centered within the rollout group independently for each block. The first pilot compares center-only Dr.GRPO advantages with the existing standard-deviation-normalized form; neither result is silently substituted for the other.

The policy loss is averaged across group, reverse diffusion transitions, and active latent blocks. Inactive blocks after EOS are masked. A KL penalty to the frozen parent S2 is computed on the same block-factor scale. Actions remain detached, every trained reverse transition remains stochastic, and only S2 optimizer parameters may receive gradients.

## Data contracts

GSM8K and MATH are fetched at pinned Hugging Face revisions. Official training splits are partitioned by normalized problem identity into training and development data. Official test splits remain sealed and are read only by an explicit final-evaluation command. Cross-split duplicates are rejected. Raw datasets, rationales, and model checkpoints are not committed to Git.

The prepared schema records problem ID, problem text, canonical answer, attributed reference rationale when supplied by the source, task family, category, source revision, split hashes, and verifier version. Reference rationales may be used for diagnostics but do not enter the GRPO rollout prompt.

## Comparisons

Each pilot reports three downstream arms with identical prompts and evaluation seeds:

- raw RWKV without latent state injection;
- frozen parent S2 plus dynamic S1;
- block-GRPO-trained S2 plus dynamic S1.

MMLU-Pro reports direct option accuracy and confirms that its one-token answer primarily exercises latent block zero. GSM8K and MATH report exact-answer accuracy, parse/format validity, generated length, active block count, reward components, blockwise margin/potential changes, and current-versus-parent deltas.

## Pilot scope

Before long training, run deterministic CPU contract tests and one real-model GPU preflight. Then run short, separately labeled GSM8K and MATH pilots with small train/dev limits. A pilot is successful only if all of the following hold:

- all 16 block indices are exercised by at least one nontruncated rollout;
- S0, S1, and S2 call counts are nonzero;
- only original S2 parameters change;
- block rewards or returns are nonflat for at least some groups;
- block-factor score-function gradients are finite and nonzero;
- parent, current, and raw evaluation arms complete without test leakage.

Accuracy improvement is not required to call the plumbing valid. Longer training is justified only by stable KL and improving held-out metrics.

## Failure handling

Flat groups are skipped without optimizer or weight-decay updates. Nonfinite rewards, ratios, KL, or gradients abort the update. Ratio overflow is reported rather than silently averaged away. Unsupported MATH answer forms are counted as verifier failures. Resume requires exact data, verifier, sampler, reward, and policy-contract identity.

## Tests

Implementation begins with failing tests for:

- numeric and conservative symbolic answer verification;
- block boundary generation and EOS masking;
- telescoping potential rewards and block return-to-go;
- per-block Gaussian log-ratio shape and detached actions;
- per-block advantage centering and flat-block handling;
- frozen parent gradient exclusion and S2-only parameter updates;
- pinned dataset manifests, split isolation, and sealed-test acknowledgement;
- end-to-end fake-runtime training/evaluation contracts.

## References

- Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models: <https://proceedings.iclr.cc/paper_files/paper/2025/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf>
- Stabilizing Reinforcement Learning for Diffusion Language Models: <https://arxiv.org/abs/2603.06743>
- Group Diffusion Policy Optimization: <https://arxiv.org/abs/2510.08554>

The papers motivate the factorization and stability checks. Their masked-token likelihood estimators and staircase attention are not copied because LACES uses tractable continuous Gaussian reverse transitions and causal RWKV cache injection.
