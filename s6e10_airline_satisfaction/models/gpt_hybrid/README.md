# GPT-2 hybrid: one more epoch with a generative + BCE loss

The starting point ("B0") is the GPT-2 small target-first model of each fold from `../gpt2_target_first`. A child model
continues B0 for one epoch on the same training rows (lr 1e-5, batch 64). Its loss is the usual generative loss plus
0.1 x a BCE on B0's own Bayes readout: the log-likelihood ratio `log p(fields | True) - log p(fields | False)` plus the
prior log-odds of the training rows (positive rate 0.4436), at a fixed scale. During training the softmax is restricted to
the 1,073 tokens that occur in the row texts; scoring uses the full softmax.

Scoring the child with all 30 orders would cost as much as B0's own scoring. Instead it is scored with the first 6 orders,
and the change is added to B0's 30-order score:

    z = LLR_B0(30 orders) + LLR_child(6 orders) - LLR_B0(the same 6 orders),    p = sigmoid(z)

```
python ../gpt2_target_first/train.py --size small    # B0 fold models + their 30-order log-likelihoods
python train.py
```

Pair `gpt_hybrid_cv6` (probabilities, not aligned; test = mean of the 5 fold models' probabilities), OOF AUC 0.962378.
The child checkpoints are kept in `checkpoints/`.
