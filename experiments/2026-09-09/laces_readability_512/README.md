# Readability comparison at 512 tokens

Both files use the 30k checkpoint, the same sky question, DDIM with 1000
diffusion steps, CFG 1.5, temperature 0.2, top-k 50, top-p 0.9,
repetition penalty 1.2, seed 42, and a 512-token output limit.

- `best_t02_b07_p09.json`: state blend 0.7. It keeps the answer on topic and
  explains Rayleigh scattering and sunset colors for most of the window.
- `compare_t02_b05_p09.json`: state blend 0.5. It starts correctly but enters
  repeated claims about the sky being closer to the ground much earlier.

For semantic readability, blend 0.7 is the preferred setting from this pair.
