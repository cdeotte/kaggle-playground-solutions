# XGBoost by mikhailnaumov (public notebook)

Mikhail Naumov's (@mikhailnaumov) XGBoost ("XGB5") from the public notebook
[Smartphone Addiction | XGB](https://www.kaggle.com/code/mikhailnaumov/smartphone-addiction-xgb) (version 3).
Thank you, Mikhail!

This is the notebook's XGB5 recipe and features run on our 5 folds (`StratifiedKFold(5, shuffle=True,
random_state=42)`, the same split the notebook uses). `preprocessing.py` is the notebook's `Preprocessing` class
(unchanged); `train.py` is the fold loop of its `Trainer` for the XGB5 model.

- **Features** (`Preprocessing`, fitted on the fold's training rows): the 12 raw columns; for every raw column the mean
  target of that value in the 7,500-row original dataset; every numeric column also as a string category (`_cat`) and
  rounded to 0 and 1 decimals; 9 screen-time ratio features; constrained-imputation features (bounds of a missing daily
  total / missing component, slack, case id, number missing); missing count; the normalized frequency of every
  categorical and `_cat` column; constant columns dropped.
- **Encodings per fold:** a scikit-learn `TargetEncoder` (cv=5, smooth 'auto', shuffle, seed 42) of the categorical
  columns, fitted on the training rows; the `_cat` columns are then dropped.
- **Model:** XGBoost, lr 0.01, depth 6, min_child_weight 5, subsample 0.894, colsample_bytree 0.184, lambda 4.32,
  alpha 2.21, max_bin 1024, native categoricals, CUDA, seed 42, up to 50,000 rounds with early stopping 100 on the
  validation AUC. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook loads XGB5's saved predictions; we train the recipe. Its `Preprocessing`
was fitted once on all training rows; we refit it on each fold's training rows (frequencies counted on those rows). Its
`TargetEncoder` asks for `target_type='multiclass'` on this binary target; we use `'binary'` (one column per feature).

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/mikhailnaumov_xgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv and the
original `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`); no other repo folder is needed. One GPU.
Packages: xgboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| mikhailnaumov_xgb | the notebook's XGB5 on our 5 folds | 0.968702 |
