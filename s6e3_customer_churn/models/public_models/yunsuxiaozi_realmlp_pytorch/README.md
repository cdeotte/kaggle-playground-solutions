# RealMLP (PyTorch, from scratch) by yunsuxiaozi (public notebook)

yunsuxiaozi's (@yunsuxiaozi) RealMLP written from scratch in PyTorch, from the public notebook
[RealMLP from scratchCV:0.91908](https://www.kaggle.com/code/yunsuxiaozi/realmlp-from-scratchcv-0-91908). Thank you,
yunsuxiaozi!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).
`realmlp.py` is the notebook's model code; `train.py` is its feature engineering and fold loop.

- **Features (label-free; built on train and test separately, bins fitted on train):** tenure, MonthlyCharges and
  TotalCharges rounded to integers as categories; their digits at positions 10^-2 .. 10^3 and every pair of the digits
  at 10^0 .. 10^3 as categories; MonthlyCharges / TotalCharges, TotalCharges / MonthlyCharges, TotalCharges / tenure,
  MonthlyCharges / (TotalCharges / tenure) and tenure^2; the Contract + InternetService + PaymentMethod combo; the
  service count; quantile bins of MonthlyCharges (100, 1,000) and TotalCharges (500, 4,000). Columns that are constant
  on test are dropped. Every categorical (the 15 string columns, rounded values, digits, combo, service count and bins)
  is coded by its frequency rank in train; values seen fewer than 5 times share one code.
- **Encodings per fold:** a scikit-learn `TargetEncoder` (binary, cv 5) of the categoricals with more than 9 codes is
  appended to the 9 numerics; all of them are robust-scaled (median / IQR) and smoothly clipped, fitted on the training
  rows.
- **Model (`realmlp.py`):** RealMLP with 8 vectorized ensemble members: PBLD periodic embeddings of the numerics (16
  hidden, 5 outputs), one-hot encodings of the categoricals with up to 8 levels and 10-d embeddings of the larger ones
  (as in the notebook, these embedding tables are not registered as module parameters, so they keep their random
  initial values), a learnable scaling layer, NTP linear layers 512-64-128 with GELU and dropout 0.05, and a 2-class
  softmax head per member. AdamW (lr 0.003; x10 for the scaling layer and x0.1 for biases, weight decay 1e-2 scaled
  likewise; betas 0.9 / 0.98), learning rate flat for half of the training and then linearly decayed, label smoothing
  0.04 on a cosine schedule, gradient clipping 1; 3 epochs, batch 256 (the first epoch in row order, then shuffled).
  The weights of the epoch with the best validation AUC (above 0.915) are restored and predict the validation and test
  rows; the members' probabilities are averaged. The notebook seeds once (2026) and trains its folds in one loop, so
  `train.py` runs the folds in order in one process. Test = mean of the fold models.

**Changed to run it on our folds:** the notebook splits 10 folds by row index (row i is in fold i mod 10); our run uses
our 5 stratified folds. The notebook's OOF is the last epoch's validation predictions; ours are predicted with the
restored best weights, like the test predictions. The start seed (2026) also seeds torch, and the best weights are kept
in memory instead of a checkpoint file.

```
python train.py            # all folds in order in one process
```

Writes `preds/yunsuxiaozi_realmlp_pytorch_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no
other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages: torch, scikit-learn, pandas,
numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| yunsuxiaozi_realmlp_pytorch | the notebook's from-scratch PyTorch RealMLP (8 members) on our 5 folds | 0.918943 |
