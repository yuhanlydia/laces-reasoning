# Open-Context Generative Benchmark Suite Design

## Status

Approved as the next LACES evaluation stage on 2026-09-16. This document defines the
protocol; it does not claim benchmark improvements before the recorded runs finish.

## Research question

The step-30,000 model keeps a frozen 2.9B RWKV backbone and trains an OpenWebText
prefix-to-future latent trajectory with an input-dependent rank-32 state writer. The
evaluation should therefore ask whether dynamic state writing improves use of supplied
natural-language context. It should not primarily test mathematical knowledge, coding,
closed-book recall, or selecting an A/B/C/D label.

The central comparison is full LACES against the exact frozen RWKV backbone on identical
prompts. The historical fixed-basis checkpoint is a secondary reference because its
training run and trajectory decoder differ from the current dynamic checkpoint.

## Benchmark portfolio

### A. Open-ended long-context QA and retrieval

Use the official LongBench v1 data, prompts, and metrics:

| Dataset | Capability | Official metric |
| --- | --- | --- |
| HotpotQA | multi-document QA | token F1 |
| 2WikiMultihopQA | multi-document, multi-hop QA | token F1 |
| MuSiQue | compositional multi-hop QA | token F1 |
| NarrativeQA | long narrative comprehension | token F1 |
| MultiFieldQA-en | long single-document QA | token F1 |
| PassageRetrieval-en | synthetic long-context retrieval | retrieval accuracy |

Source: <https://github.com/THUDM/LongBench>.

### B. Writable-state and tracing stress tests

Use BABILong QA1, QA2, QA3, QA6, and QA9 at the official 0k, 1k, 2k, and
4k context configurations. These cover one-, two-, and three-supporting-fact QA,
yes/no reasoning, and negation without requiring a choice label. Score normalized exact
match and answer containment, preserving the official answer strings.

Use RULER variable tracking (`vt`) and open-ended QA (`qa_1`, `qa_2`) at 512, 1024,
2048, and 4096 RWKV-token budgets. RULER data must be generated from the official code
with the frozen RWKV tokenizer and seed 42.

Sources: <https://github.com/booydar/babilong> and
<https://github.com/hsiehjackson/RULER>.

### C. Long-form natural-language compression

Use the official LongBench GovReport, QMSum, and MultiNews tasks with ROUGE-L. These are
secondary because summary quality depends strongly on instruction following and surface
generation quality in the 2.9B backbone.

## Exclusions

The first evaluation wave excludes mathematical benchmarks, code benchmarks, Chinese
tasks, closed-book factual QA, LongBench v2, QuALITY, MMLU-style choice tasks, and any
benchmark whose primary output is an A/B/C/D label. Passage counting is also excluded
from the first wave because it mixes memory with arithmetic aggregation.

## Data provenance

Pin every source revision and record SHA-256 for every evaluated file. Raw benchmark data
lives outside git under `/root/benchmarks`; the repository stores only manifests,
protocols, commands, and compact result JSON.

Pinned sources for the initial download:

- `THUDM/LongBench` dataset revision
  `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`.
- `RMT-team/babilong` dataset revision
  `ee0d588794c7ac098062ee0d247c733d62e94fe2`.
- LongBench code commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`.
- BABILong code commit `7a6efee29f5cac03c3c410e6799c80fd2ffe3610`.
- RULER code commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.

## Matched evaluation arms

1. `raw`: frozen RWKV from the active dynamic checkpoint, with no S0/S1/S2 state write.
2. `dynamic`: step-30,000 LACES with its native S0, S2 trajectory, rank-32 dynamic S1,
   state scale, and validation-selected blend.
3. `fixed`: historical fixed-basis checkpoint, reported separately and never described as
   a training-controlled ablation of the 30k dynamic run.

Raw and dynamic must share token IDs, prompt truncation, generation budget, greedy decoding,
EOS handling, and scorer. The tested checkpoint fingerprint and backbone path are mandatory
result fields.

## Generation protocol

Use `LACESRuntime` with the `aligned` cache protocol. State is written before consuming the
boundary anchor token, so the first answer-token distribution depends on S1 and no prompt
token is consumed twice. `legacy` timing and the older LongBench runners are not allowed in
reported comparisons.

Run a development-only blend comparison between 0.7 and 1.0, then freeze one value before
the formal evaluation split. Default sampling is deterministic greedy text decoding. The
S2 sampler, step count, CFG scale, seed, and maximum generation length must be recorded.

## Context and capability gates

Report results separately at 512, 1024, 2048, and 4096 input tokens. Never average length
bins without also publishing each bin.

Before comparing adapters, run a raw-backbone gold-context gate. A dataset/bin that the raw
2.9B backbone cannot answer above its stated minimum is marked `backbone-limited`; it may
remain in the report but cannot support a claim about writer quality.

For the mechanism suite, also report:

- fixed-window text truncation;
- sequential raw RWKV state carry;
- fixed-basis latent/state transfer;
- dynamic-basis latent/state transfer;
- text and state communication payload bytes.

This separates long-context reading from the stronger claim that dynamic state writing is
a useful communication substrate.

## Metrics and statistics

Use the official LongBench F1, retrieval accuracy, and ROUGE-L implementations. BABILong
and RULER use normalized exact match plus answer containment where applicable. Save every
raw prediction and token ID sequence.

Report sample count, mean score, per-sample dynamic-minus-raw difference, paired wins/losses,
and a paired bootstrap 95% confidence interval. Formatting-only gains must be identified by
inspecting raw generations rather than attributed to reasoning.

## Execution order and concurrency

1. Run 20-sample smoke tests for groups A, B, and C.
2. Attempt three concurrent GPU processes only after measuring loaded-model peak memory.
3. If three processes cannot maintain a safety margin, run two concurrent processes and
   queue the third. OOM attempts are discarded and never scored.
4. Run dynamic and matched raw before the historical fixed reference.
5. Promote to the official sample counts only after scorer, token-budget, and first-token
   alignment audits pass.

## Required artifacts

Each run writes a configuration/provenance JSON, one JSONL row per sample, an aggregate
metrics JSON, peak memory and elapsed time, and an error log. A final matrix compares
dynamic/raw/fixed by dataset and context length. Repository documentation must distinguish
completed results from planned or interrupted runs.

## Success criterion

The strongest evidence would be a paired dynamic-over-raw improvement on context-contained,
open-ended QA or tracing that grows with context length while the raw-backbone gold-context
gate remains healthy. A result found only in choice parsing, newline handling, or a scorer
mismatch does not count.
