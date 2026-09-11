# 30k LACES sampling and reasoning checks — 2026-09-11

**Sampling produces readable openings but unreliable long continuations. The recurrent-refiner CPU tests pass, while its unmodified real-model preflight fails on a first-versus-subsequent score mismatch. A separate one-step PG diagnostic updates the refiner; direct answer CE has no gradient on this backend. No reasoning-accuracy improvement is established.**

## Scope and revisions

- Sampling, recurrent-refiner preflight and diagnostics ran against `82c456765b9f9fa4819ab7a53ebd3e2ae38727a2`.
- Before publishing these results, remote main gained `49ff6c6` (direct S2 distillation/GRPO for MMLU-Pro). That additive commit was preserved. Its 48 CPU tests passed separately; its real-GPU workflow and MMLU-Pro accuracy were **not** tested here.
- No production model, training code, sampling logic, or pretrained weights were modified for these experiments. Probe scripts live beside the evidence. No trained refiner checkpoint is claimed or published.

## Sampling results

Six prompts used the unchanged historical sampler: DDIM 1000 steps, CFG 2.0, independent S1, blend 0.7, temperature 0.2, top-k 50, top-p 0.9, repetition penalty 1.2, seed 42, and 512 output tokens. Each new topic loaded the model afresh and reset the seed.

| Topic | Observed behavior |
|---|---|
| Blue sky | Readable Rayleigh-scattering opening; first 766 generated characters match the historical cfg2 sample. Later repetition and inaccuracies. |
| Photosynthesis | Correct opening, then unrelated news-style text, wood-burning explanations and stove advertising. |
| Printing press | Relevant opening, followed by repeated `Advertisements`, unrelated tags and numbers. |
| Binary search | Confuses search with sorting and incorrectly gives `O(n log n)` complexity. |
| 17 × 23 | Answers 399 and claims `340 + 51 = 501`; correct answer is 391. |
| Autumn leaves, Chinese | Readable Chinese with factual errors and contradictions, then switches to English. |

The historical sampler does not stop on EOS; it generates all 16 × 32 tokens. Interpret long continuations with that behavior in mind. This does not explain the early arithmetic/algorithm errors. These are qualitative single-seed examples, not a benchmark score.

- [All complete raw outputs](sampling/ALL_OUTPUTS.md)
- [Sampling result JSON files and prompts](sampling/)
- [Historical comparison](sampling/sky_historical_comparison.json)

## Real recurrent-refiner checks

| Check | Result |
|---|---|
| Existing CPU contracts | **23 passed**, 2.35 seconds |
| Active pretrained checkpoint audit | **494 tensors verified:** S0 12, S1 9, S2 473; checkpoint step 30000 |
| Actual component execution in traced preflight | S0 1 call, S1 3 calls, S2 2000 calls (conditional + unconditional at 1000 steps) |
| R=0 latent identity | Exact equality, and the very same tensor object |
| Unmodified GPU preflight | **Fails** `Pretrained-module / R=0 identity gate failed` |
| Repeated identical LACES scores | First −4.684530735; next seven −4.665782928; range 0.018747807 |
| Repeated raw RWKV scores | Eight identical scores, −4.709950447 |
| Candidate reward signal | Std 0.05440757 in the repeated-score diagnostic |
| Direct answer CE | `loss_requires_grad=false`; no state-input gradient through this backend |
| Separate one-step PG diagnostic | Finite pre-clipping gradient norm 12.737865; reward std 0.07533138; delta-head weight and bias changed; frozen base received no parameter gradients |

The failing gate is at `scripts/eval/train_laces_reasoner.py:211–213`. It requires bitwise equality between the first and second scores. Both original and instrumented fresh processes reproduced the failure. The diagnostic confirms a first-call effect for this input; its kernel/operator-level cause remains unresolved. It is not evidence that R=0 changes the latent. Do not silently loosen the gate and claim deterministic correctness.

The separate PG probe uses the actual `policy_candidates`, `answer_policy_loss`, native renderer and refiner, after repeated scoring. It bypasses the failing **entrypoint** only by invoking these functions independently for diagnosis. It is **not** a successful end-to-end trainer run, nor evidence of improved accuracy. The unmodified trainer was not allowed past its failed preflight.

On the diagnostic toy question, both raw RWKV and full LACES R0 generated the correct town, Selby, with extra continuation. That single case cannot establish multihop reasoning: the town is explicitly present in the facts.

- [Original preflight failure](reasoning/preflight_failure.txt)
- [Traced equality and module calls](reasoning/preflight_trace.json)
- [Repeated scores, native outputs and CE gradient probe](reasoning/repeat_score.json)
- [Independent PG update evidence](reasoning/pg_update_diagnostic.json)
- [New direct-S2 CPU test output](reasoning/direct_s2_tests.txt)

## Environment and weight loading

NVIDIA A100-SXM4-40GB; Python 3.10; PyTorch 2.8.0; Transformers 4.57.6; FLA/fla-core 0.3.2. [Observed package versions](environment/requirements-observed.txt). `pip check` passed before these tests.

The backbone repository contains both monolithic and sharded safetensors. The sharded representation matches FLA 0.3's parameter names. A local directory at the checkpoint's configured backbone path links to the original sharded files and tokenizer.

An initial Transformers 5.3.0 / FLA 0.3.2 diagnostic showed that loaded `lm_head.weight` differed from the source file despite an empty missing/unexpected-key report. With Transformers 4.57.6, **all 1059 backbone tensors matched their source files exactly**, and coherent text generation returned. [Failure evidence](environment/backbone_transformers5_failure.json), [verified loading](environment/backbone_verified.json).

Model revisions: [model_sources.json](environment/model_sources.json). The 30k adapter SHA-256 verified locally is `7d0f0834a9aca75b577b660340de50f8d4ad0c4c1779342f0bd2e7b373c04d66`. No weight binaries are committed.

## Reproduce the recurrent preflight

Use the recorded environment and downloaded local model files. From the repository root:

```bash
.venv/bin/python -m pytest tests/test_pretrained_laces_reasoning.py -q
mkdir -p outputs_eval/reasoning_review_2026-09-11
PYTHON="$PWD/.venv/bin/python" OMP_NUM_THREADS=4 MODE=preflight \
OUTPUT_DIR=outputs_eval/reasoning_review_2026-09-11 \
CACHE_DIR=outputs_cache/reasoning_review_2026-09-11 \
bash training/run_recurrent_reasoner.sh
```

The default preflight is expected to fail in this recorded environment. To reproduce the additional probes, run the Python files in `reasoning/` from the repository root. They write into the same ignored `outputs_eval` directory. The repeat/PG probes may populate the prefix cache if absent; their zero S0/S2 call counts in saved runs reflect reuse of the audited prefix cache, not a new S0/S2 execution claim.
