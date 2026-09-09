# Closed-loop recurrent latent reasoner

This experiment checks the reasoning path against the 30k frozen RWKV checkpoint:

```bash
/usr/bin/python3 scripts/eval/train_recurrent_reasoner.py \
  --tasks 2agent 3agent 4agent --n_train 6 --n_test 3 \
  --epochs 20 --train_max_steps 8 --eval_max_steps 8 \
  --eval_depths 1 2 4 8 --R 4 --early_stop \
  --output outputs_eval/reasoner_closed_loop_pooled_n6_e20.json
```

The reasoner now reads a 4x4 adaptive-pooled grid from every RWKV layer/head after
each cumulative write.  The next latent update therefore depends on the current
written state, rather than only on static fact features.  The old direct
`cache.layers` access was also replaced with the FLA 0.3/0.4 cache compatibility
helper.

Validation on 2026-09-09:

- `67 passed, 7 skipped` for the full repository test suite.
- Smoke training and generation completed without the previous Cache API crash.
- The medium run reached train loss `1.2934`; test relative state MSE was `0.940925`
  at R=8 (`0.943932` at R=1).
- The synthetic Agent task gave `0/9` answer accuracy for text concat, oracle state
  injection, and learned R=1/2/4/8 writes.  This is a capability failure, not a
  successful BDH-CQ reproduction.  The base 30k checkpoint also produced unreadable
  text for the synthetic prompts, so answer accuracy is not yet a valid proof that
  the recurrent loop has learned the task.
