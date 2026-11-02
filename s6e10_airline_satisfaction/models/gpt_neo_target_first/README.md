# GPT-Neo 125M target first

The Storyteller recipe of `../gpt2_target_first` with a different language model: GPT-Neo 125M instead of GPT-2. A second
architecture adds diversity to the GPT family in the stack. GPT-Neo uses GPT-2's tokenizer (identical token ids), so the same
row-to-text code works.

```
python ../gpt2_replica/train_replica.py --size neo    # GReaT replica: GPT-Neo 125M, 60 epochs on data.csv
python train.py                                        # 5 folds: label-first fine-tuning (8 epochs) + 30-order Bayes scoring
```

Pair `gpt_neo_target_first` (log-likelihood ratios), OOF AUC 0.962159. The held-out fold is scored with the same 30 orders
as `../gpt2_target_first`; for the test rows each fold model uses its own 30 orders, so the 5-fold test mean covers 150.
