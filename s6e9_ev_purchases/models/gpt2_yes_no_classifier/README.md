# GPT-2 Yes / No next-token classifier

Each row is written as GReaT text with the 13 fields first and the label prompt last,
`Age is 36, City_Type is Urban, ..., Range_Anxiety_Level is Low, Will_Buy_EV is`, and the model's own LM head predicts the next
token: score = logit(` Yes`) - logit(` No`) (no classifier head is added).

1. Start from the e55 GReaT replica (`../gpt2_replica`).
2. Fine-tune per fold with a two-way cross-entropy on the ` No` / ` Yes` logits, 4 epochs, batch 256, lr 5e-5. Each epoch, every
   row is written in the competition column order with probability 1/2, otherwise in a random one of 64 fixed field orders.
3. Score the held-out fold and the test rows in 4 field orders (competition, original, 2 random) and average the scores.

```
python train.py                 # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

| Pair | OOF AUC |
|---|---:|
| gpt2_yes_no | 0.945639 |

Test = sigmoid of the mean of the 5 fold models' scores. Needs `../gpt2_replica/checkpoints/e55.pt`. Ran with transformers 4.26.1
and torch 2.5.1 (the replica environment, `../gpt2_replica/requirements.txt`); about 70 minutes per fold on one GPU.
