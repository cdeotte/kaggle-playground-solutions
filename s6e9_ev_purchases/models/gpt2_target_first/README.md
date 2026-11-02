# GPT-2 target first (generative classifier)

1. Start from the e55 GReaT replica (`../gpt2_replica`).
2. Fine-tune it per fold as a causal LM on the training rows of the fold, written label first like the generator wrote them:
   `Will_Buy_EV is Yes, Age is 36, City_Type is Urban, ...` (fields in a fresh random order per row per step), 12 epochs,
   lr 5e-5, batch 32.
3. Classify with Bayes' rule: score `log p(row | Yes)` and `log p(row | No)` (the summed log-probabilities of all field tokens
   after the label) for 6 fixed random field orders in fp32 / TF32, pool the orders by log-sum-exp, take the difference.

```
python train.py                 # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

| Pair | OOF AUC |
|---|---:|
| gpt2_target_first_12ep | 0.946485 |

Needs `../gpt2_replica/checkpoints/e55.pt`. Ran with transformers 5.x and torch 2.11; about 5.5 hours per fold on one GPU
(12 epochs of training, then 6 orders x 2 labels for the held-out fold and the test rows).
Shared row-to-text code: `../../common/gpt_rows.py`.
