# DDIM1000 decoding sweep

All samples use the 30k checkpoint and the same QA prompt:

```text
Question: Why does the sky appear blue? Answer in one clear short paragraph:
```

Common settings:

- DDIM: 1000 diffusion steps
- CFG scale: 1.5 for the sweep, 3.0 for the Cola-style greedy comparison
- maximum new tokens: 256 for the sweep, 512 for the greedy comparison
- seed: 42

The sweep varies temperature, top-p, and trajectory state blend. `summary.json`
sorts the outputs by a diagnostic score that penalizes Hangul characters,
replacement/special characters, control characters, and repeated four-token
phrases. This is a cleanliness diagnostic, not a language-model quality metric.

The lowest diagnostic score in this run was `temperature=0.2`, `blend=0.5`,
`top_p=1.0`. Cola-DLM-style greedy decoding had no Hangul or special tokens,
but repeated the same short phrase heavily.
