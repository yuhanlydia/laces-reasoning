# 10k native LACES text-continuation probe

This is a quick paired diagnostic on 32 deterministic documents from
[`stas/openwebtext-10k`](https://huggingface.co/datasets/stas/openwebtext-10k).
It tests the original text-continuation objective and does not use raw RWKV as the
primary comparator.

For each document, the first 128 tokens are the prefix and the next 32 are the gold
continuation. The same frozen 10k LACES checkpoint scores the gold continuation with:

- `matched`: S2 latent sampled from the correct prefix;
- `mismatched`: a latent sampled from the next document's prefix;
- `teacher`: a native S0 encoding of the gold continuation, used only as a diagnostic.

All arms use dynamic S1, RWKV, blend `0.7`, and the same scoring path. S2 sampling uses
32-step DDIM with CFG `1.5`.

| Arm | Mean NLL/token | Perplexity |
| --- | ---: | ---: |
| Correct-prefix latent | 2.4990 | 12.1703 |
| Mismatched latent | 2.4924 | 12.0909 |
| Teacher continuation latent | 2.4878 | 12.0349 |

The mean paired advantage `mismatched NLL - matched NLL` is `-0.00655` token NLL,
with a 200,000-resample bootstrap 95% interval of `[-0.02838, 0.00985]`. Matched wins
13 of 32 rows (40.6%). This probe therefore finds no evidence that the 10k S2 latent
reliably improves held-out-style OWT continuation. The teacher arm is only `0.01118`
NLL/token better than matched, so this run also provides no strong oracle margin.

This slice is deterministic but was not decontaminated against the original LACES
training corpus. The result is a mechanism diagnostic, not a benchmark claim. FLA emits
its upstream RWKV implementation warning, so publication-grade numbers require an
official-RWKV cross-check.

The complete row-level scores and two qualitative paired generations are in
[`10k_paired_dev32.json`](10k_paired_dev32.json).

Reproduction after downloading and extracting the dataset:

```bash
python experiments/2026-09-15/owt_continuation_10k/eval_owt_continuation_10k.py \
  --docs data/openwebtext_10k/docs \
  --ckpt-dir outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00010000 \
  --rwkv-path models/RWKV7-Goose-World3-2.9B-HF-fla-v2 \
  --output results/owt_continuation/10k_paired_dev32.json \
  --rows 32
```
