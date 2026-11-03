# XGBoost by abdullahsafwan333 (public notebook)

AbdullahSafwan333's (@abdullahsafwan333) XGBoost from the public notebook
[S6E8: XGBoost | SAP](https://www.kaggle.com/code/abdullahsafwan333/s6e8-xgboost-sap) (version 2).
Thank you, Abdullah!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features** (`../lib/sap_features.py`, shared with the author's LightGBM notebook; no target-dependent features): the
  9 numeric predictors, 8 domain features (unexplained screen time, total activity, work / social / gaming share of the
  daily screen time, max and min of daily / weekend / social screen time, awake hours), the ones digit of every numeric
  column and the tenths and hundredths digits of the 6 fractional columns, and the 3 categoricals as native categories.
- **Model:** `xgboost.train`, 4,377 rounds (fixed, no early stopping), eta 0.0515, depth 5, subsample 0.778,
  colsample_bytree 0.649, colsample_bylevel 0.877, min_child_weight 9.24, lambda 0.129, alpha 0.00924, gamma 0.0705,
  max_bin 512, hist on CUDA, native categoricals, seed 42 + fold. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook uses `StratifiedKFold(7, shuffle=True, random_state=42)`; our run uses
our 5 folds (seeds 42 to 46). Nothing else changed.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/abdullahsafwan333_xgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed (shared code: `../lib/`). One GPU. Packages: xgboost, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| abdullahsafwan333_xgb | the notebook's XGBoost on our 5 folds | 0.966692 |
