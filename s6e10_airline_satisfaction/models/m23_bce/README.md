# M23 + BCE: label-conditional masked-field transformer with a BCE finish

A small transformer reads a row as 23 tokens: the 21 feature values (every distinct value of a column is its own token),
the label and the data source (Kaggle or original). Two stages per outer fold k:
1. **Parent (M23):** masked-field modelling on the Kaggle train rows (fold k's labels hidden), the Kaggle test rows (label
   unknown) and the original `data.csv` rows. 25% of the fields are masked and predicted; a known label is masked half of
   the time, hidden or unknown labels are always masked and never scored. Its zero-shot readout is
   P(label | x, source = Kaggle) at the masked label position. L6, d256, 8 heads, 15 epochs.
2. **BCE continuation:** starts from the fold-k parent and trains 4 epochs on the outer-train rows with the label input
   always masked and BCE on the difference of the two label-token logits (10% feature masking, lr 5e-5 decaying linearly
   to 0). Prediction = mean of the epoch-3 and epoch-4 probabilities.

```
python train.py
```
-> pair `m23_bce`, OOF AUC 0.960796 (probabilities; test = mean of the 5 fold models, no alignment).
The parent checkpoints are kept in `checkpoints/` and reused. Uses only the competition data and `data.csv`; no other
folder is needed.
