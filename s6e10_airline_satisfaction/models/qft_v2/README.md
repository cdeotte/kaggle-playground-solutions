# Q-FT v2: an order-free ModernBERT that reads a table row as a set

Three changes from Q-FT (`../qft`, whose feature coding and tokenizer it reuses):
1. **Order-free encoder:** ModernBERT-base (bidirectional) with every token at position 0, so the row is a set.
2. **Side features** (`side_features.py`): a few engineered numbers on every feature token + 2 summary tokens for the row.
3. **Pretraining on the table itself:** replaced-token detection on the features of train + test + original rows (no labels),
   3 epochs, done once and shared by all folds; then 2 epochs of fine-tuning per fold.

```
python side_features.py
python train.py
```
-> pair `qft_v2`, OOF AUC 0.960798 (+1e-3 over Q-FT; it did not improve our stack).
Discussion: https://www.kaggle.com/competitions/playground-series-s6e10/discussion/751334
