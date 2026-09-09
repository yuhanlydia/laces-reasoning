# LACES 30k generation check

These files record generation from the dynamic-basis checkpoint at training
step 30,000:

- `qa_sky.json` — question answering prompt about why the sky appears blue
- `owt_predictwise.json` — OpenWebText-style continuation
- `owt_hair_breakage.json` — OpenWebText-style continuation

Sampling configuration:

- trajectory sampler: conditional DDPM
- diffusion steps: 100
- classifier-free guidance scale: 1.5
- trajectory state blend: 0.5
- maximum new tokens: 512
- temperature: 0.7
- top-k: 50
- top-p: 0.9
- repetition penalty: 1.2
- seed: 42

The checkpoint initially follows each prompt, but the latter part of the
512-token window can show corpus-boundary artifacts, special-token leakage,
and repetition. The JSON files contain the complete prompt and generated text.
