# LightGBM by redamountassir (public notebook)

Reda Mountassir's (@redamountassir) LightGBM from the public notebook
[S6E8: LGBM | LB 0.96965](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965) (version 1). Thank you, Reda!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features** (`../lib/constrained_te.py`, shared with the author's HistGradientBoosting notebook): the 12 raw columns
  (3 native categoricals, NaN kept), missing count, 9 screen-time ratio features, constrained-imputation features from
  the identity daily >= social + gaming + work_study (lower bound of a missing daily total; upper bound, midpoint and
  interval width of a single missing component; exact slack; case id; number of missing screen values), and the
  frequency of every raw value over train + test.
- **Encodings per fold:** a scikit-learn `TargetEncoder` (cv=5, smooth 'auto', shuffle, seed 42) of the 12 raw columns
  as strings, fitted on the fold's training rows (cross-fitted for those rows).
- **Model:** LightGBM, lr 0.0463, 780 rounds (no early stopping), 96 leaves, min_child_samples 267, lambda 5.24,
  max_bin 1023, colsample_bytree 0.473, seed 0. Test = mean of the 5 fold models.

**Changed to run it on our folds:** nothing; the notebook already uses our split and fold-local encodings.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/redamountassir_lgbm_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed. CPU, 24 threads (`THREADS`). Packages: lightgbm, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| redamountassir_lgbm | the notebook's LightGBM on our 5 folds | 0.968259 |
