# GPT features + BatchEnsemble head

A small neural head trained on what a frozen GPT-2 classifier sees in each row. The frozen model ("B0") is the GPT-2 small
target-first model of the same fold from `../gpt2_target_first`, so neither B0 nor the head ever sees the labels of the
held-out fold.

1. **GPT features** (`extract_features.py`). B0 reads every train and test row with both label prefixes, in two field
   orders: the first of its 30 scoring orders and that order reversed. The softmax is restricted to the 1,073 tokens that
   occur in the row texts. Per field: the log-probability of all its tokens, the log-probability of its value tokens, and
   the entropy at its value tokens. Per text: 32-dim random projections of the mean and last hidden states. With the
   True - False differences, the summed log-likelihoods and B0's two-order log-likelihood ratio (LLR) this gives 639
   features.
2. **Head** (`train.py`). A BatchEnsemble MLP (4 members, 3 layers of width 384, SiLU, dropout 0.1). Its inputs are the 639
   GPT features and the 53 numeric transfer features (z-scored on the training rows, clipped to ±12), and the 35
   categorical transfer codes (embedding size 8). B0's two-order LLR is added to the output as a skip connection.
   Training: AdamW lr 4e-4, cosine schedule, batch 2048, 14 epochs. Logit = mean of the 4 members and of the snapshots
   after epochs 10, 12 and 14.
3. **bag2 + alignment.** Two head seeds are averaged in logit space. Each fold is then aligned without labels:
   `sigmoid(3 (z - mu) / sd)`, with `mu` and `sd` the mean and std of that fold model's test logits.

```
python ../gpt2_target_first/train.py --size small    # B0 fold models (writes ../gpt2_target_first/checkpoints/)
python extract_features.py                           # GPT features of every train + test row, per fold (features/)
python train.py                                      # 2 head seeds per fold, bag, align
```

`train.py` also needs the transfer features from `../transfer_features` (`cache/cat.npy`, `cache/num.npy`); build them
first. Feature extraction takes about 17 minutes per fold on one large GPU, and the features take 1.5 GB per fold (`features/`).

Pair `gpt_fold_be_bag2_aligned` (scores in [0, 1]; test = mean of the 5 aligned fold vectors), OOF AUC 0.960561.
For the training rows the GPT features are in-sample, since B0 was fine-tuned on them; for the held-out and test rows they
are not.
