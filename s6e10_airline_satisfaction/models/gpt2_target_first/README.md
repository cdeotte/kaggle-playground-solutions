# GPT-2 target first (the "Storyteller")

1. Start from a GReaT replica (`../gpt2_replica`).
2. Fine-tune it on the training rows of the fold, written label first, like the generator wrote them:
   `satisfaction is True, Age is 36, Class is Eco, Inflight wifi service is 4, ...` (fields in a random order), 8 epochs.
3. Classify with Bayes' rule: score `log p(row | True)` and `log p(row | False)` (the sum of the log-probabilities of all
   field tokens after the label) for 30 random field orders, pool the orders by log-sum-exp, take the difference.

| Command | Pair | OOF AUC |
|---|---|---:|
| `python train.py --size small` | gpt2_small_target_first | 0.962425 |
| `python train.py --size medium` | gpt2_medium_target_first | 0.962370 |
| `python train.py --size large` | gpt2_large_target_first | 0.962360 |

Medium and large score an EMA of the weights (decay 0.9999). Shared row-to-text code: `../../common/gpt_rows.py`.
The script also keeps each fold model (`checkpoints/`) and its per-order log-likelihoods (`parts/*_ll.npy`): the small fold
models are the frozen "B0" teachers of `../gpt_hybrid` and `../gpt_fold_be`.
The writeup walks through the method step by step.
