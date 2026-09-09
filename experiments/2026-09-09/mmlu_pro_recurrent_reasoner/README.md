# MMLU-Pro recurrent latent reasoner

This experiment tests whether recurrent latent compute can improve a frozen RWKV
2.9B multiple-choice baseline through the LACES dynamic state writer.

MMLU-Pro has no official training split. The 12,032 labeled examples from its
public `test` split are therefore divided by category with seed `20260909` into
9,620 train, 1,196 validation, and 1,216 held-out test examples. Results from this
experiment are **not** official zero-shot MMLU-Pro benchmark scores.

The frozen RWKV encodes the question and A-J options. Its next-token answer-letter
logits form the frozen baseline. At recurrent depth R, the dynamic writer's pooled
state correction produces residual choice logits. The model is trained with a
random depth from `{1, 2, 4, 8, 16}` and evaluated separately at every depth.

Run:

```bash
bash training/run_mmlu_pro_recurrent_reasoner.sh
```

The launcher caches all frozen-backbone features, trains for at most six hours,
saves every 10,000 steps, selects checkpoints on validation accuracy, and reads
the held-out test split only after training ends.
