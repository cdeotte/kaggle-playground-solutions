# LightGBM by abdullahsafwan333 (public notebook)

AbdullahSafwan333's (@abdullahsafwan333) LightGBM from the public notebook
[S6E8: LightGBM | SAP](https://www.kaggle.com/code/abdullahsafwan333/s6e8-lightgbm-sap) (version 1).
Thank you, Abdullah!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features** (`../lib/sap_features.py`, shared with the author's XGBoost notebook; no target-dependent features): the 9
  numeric predictors, 8 domain features (unexplained screen time, total activity, work / social / gaming share of the
  daily screen time, max and min of daily / weekend / social screen time, awake hours), the ones digit of every numeric
  column and the tenths and hundredths digits of the 6 fractional columns, and the 3 categoricals as native categories.
- **Model:** LightGBM (GPU), 8,775 rounds (fixed, no early stopping), lr 0.0375, 39 leaves, min_child_samples 365,
  colsample_bytree 0.865, subsample 0.976 every round, alpha 0.246, lambda 30.6, min_split_gain 0.34, max_bin 255,
  seed 42 + fold. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook uses `StratifiedKFold(7, shuffle=True, random_state=42)`; our run uses
our 5 folds (seeds 42 to 46). Nothing else changed.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/abdullahsafwan333_lgbm_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed (shared code: `../lib/`). Needs a GPU build of LightGBM (`device_type='gpu'`), 24 CPU threads.
Packages: lightgbm, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| abdullahsafwan333_lgbm | the notebook's LightGBM on our 5 folds | 0.966918 |
