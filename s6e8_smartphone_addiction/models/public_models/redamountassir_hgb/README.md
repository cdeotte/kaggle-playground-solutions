# HistGradientBoosting by redamountassir (public notebook)

Reda Mountassir's (@redamountassir) HistGradientBoosting model from the public notebook
[S6E8: HistGradientBoosting | LB 0.96945](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945)
(version 2). Thank you, Reda!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features** (`../lib/constrained_te.py`, shared with the author's LightGBM notebook): the 12 raw columns (3 native
  categoricals, NaN kept), missing count, 9 screen-time ratio features, constrained-imputation features from the
  identity daily >= social + gaming + work_study, and the frequency of every raw value over train + test.
- **Encodings per fold:** a scikit-learn `TargetEncoder` (cv=5, smooth 'auto', shuffle, seed 42) of the 12 raw columns
  as strings, fitted on the fold's training rows (cross-fitted for those rows).
- **Model:** scikit-learn `HistGradientBoostingClassifier`, lr 0.0463, 780 iterations (no early stopping), 96 leaf
  nodes, min_samples_leaf 267, l2 5.24, max_bins 255, max_features 0.473, native categoricals, seed 0.
  Test = mean of the 5 fold models.

**Changed to run it on our folds:** nothing; the notebook already uses our split and fold-local encodings.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/redamountassir_hgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed. CPU. Packages: scikit-learn (1.4 or later for `max_features`), pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| redamountassir_hgb | the notebook's HistGradientBoosting on our 5 folds | 0.968018 |
