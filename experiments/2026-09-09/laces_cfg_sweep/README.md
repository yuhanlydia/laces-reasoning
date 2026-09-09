# CFG sweep at 512 tokens

All four samples use the 30k checkpoint with DDIM 1000 steps, temperature
0.2, top-k 50, top-p 0.9, trajectory state blend 0.7, repetition penalty 1.2,
and the same sky question.

- `cfg0_5.json`: readable start, then drifts into moon/blue-moon claims.
- `cfg2.json`: best semantic continuity and the cleanest scientific answer.
- `cfg3.json`: correct and fluent, but later repeats news-style quotations.
- `cfg5.json`: fluent, but later drifts into atmospheric-pressure details.

For this checkpoint, CFG=2 is the preferred inference setting based on
semantic readability, not just character cleanliness.
