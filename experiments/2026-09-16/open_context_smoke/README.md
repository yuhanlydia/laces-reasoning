# Open-context matched GPU probe

This directory records the first matched raw-versus-dynamic probe for the step-30,000
dynamic-basis checkpoint. It is a benchmark-selection run, not a full benchmark claim.

## Protocol

- checkpoint: `laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000`
- runtime: aligned first-token cache protocol, greedy decoding
- dynamic sampler: DDPM, 100 steps, CFG 2.0, blend 0.7, seed 42
- shared input limit: 512 RWKV tokens, middle truncation
- output limit: 32 tokens for QA, 64 for GovReport
- scorers: official LongBench QA F1 and `rouge==1.0.1`, official BABILong
  task-label-aware answer comparison
- hardware: one A40 46 GB; another process occupied about 18 GB

Three LACES processes were attempted. The third process OOMed during model loading, while
two processes completed with about 7.5 GB peak PyTorch allocation each. Subsequent runs
therefore used two concurrent processes. The failed OOM run is not included below.

## Results

Scores are percentages. Rows are the same and in the same order within each pair.

| Dataset | n | Dynamic | Raw | Delta | Dynamic wins / ties / losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 20 | 5.34 | 2.91 | +2.43 | 6 / 13 / 1 |
| 2WikiMultihopQA | 5 | 0.00 | 0.00 | 0.00 | 0 / 5 / 0 |
| MultiFieldQA-en | 5 | 21.90 | 26.33 | -4.43 | 2 / 0 / 3 |
| BABILong QA1/1k | 5 | 80.00 | 80.00 | 0.00 | 0 / 5 / 0 |
| GovReport | 5 | 11.34 | 15.37 | -4.03 | 3 / 0 / 2 |

For the 20 HotpotQA pairs, a seed-42 paired bootstrap over 20,000 resamples gives a
95% interval of +0.11 to +5.14 F1 for the mean dynamic-minus-raw difference. The rows are
the first dataset rows rather than a preregistered random sample, so this interval is only
a screening statistic.

## Interpretation

HotpotQA is the only promotion candidate from this probe. Dynamic often starts with the
correct entity where raw drifts into unrelated instruction-style continuation, but both
systems continue beyond the short answer and both absolute scores remain low. The gain did
not transfer to the first five 2WikiMultihopQA examples. It therefore supports a narrow
hypothesis—some context-contained multi-document questions benefit from the written
state—not a general multi-hop or long-context advantage.

The initial one-row BABILong pipeline diagnostic used generic normalized exact match and
was discarded. The committed five-row `B_babilong_5_*` files use the official label-aware
scorer.

RULER 512-token generation was stopped after inspection showed that the official template
already exceeds that budget and its retry loop cannot reduce the sample further. Formal
RULER generation starts at 1024 tokens.

## Next gate

Run a randomized, larger HotpotQA sample and a context-length sweep before making a model
claim. Diagnose answer stopping separately because LongBench does not first-line-truncate
HotpotQA predictions. Keep GovReport secondary: it currently measures the frozen 2.9B
backbone's long-form instruction following as much as state use.
