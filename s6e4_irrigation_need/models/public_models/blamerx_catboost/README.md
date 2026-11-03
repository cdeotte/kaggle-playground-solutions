# CatBoost by blamerx (public notebook)

The CatBoost from @blamerx's public notebook
[s6e4-catboost-robust-0-978-cv](https://www.kaggle.com/code/blamerx/s6e4-catboost-robust-0-978-cv). Thank you!

This is the notebook's model and features on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`),
without its threshold tuning. The features are the same as in `../blamerx_xgb`.

- **Features (80 + 405 encoded):** the 19 raw columns (categoricals label-encoded over train + test + original rows,
  CatBoost categorical features); 7 formula indicators (Soil_Moisture < 25, Rainfall_mm < 300, Temperature_C > 30,
  Wind_Speed_kmh > 10, Harvest, Sowing, Mulching = Yes), the formula score 2 soil + 2 rain + temp + wind - 2 harvest -
  2 sowing - mulch, its distances to the class boundaries 0 and 3 and the signed distances to the 4 thresholds;
  `logit_Low` / `logit_Medium` / `logit_High`, a fixed logistic regression of those indicators fitted on the original
  data (100% accurate there); the first decimal digit of the 11 numerics; 32 agronomy features (water input, ET /
  evaporation / VPD / heat / dryness / aridity / drought proxies, soil health, salinity, pH deviation, per-hectare
  ratios, mulch effects, logs).
- **Encodings per fold:** a multiclass `TargetEncoder` (cv=5, seed 42) of every pairwise combination of the 19 columns
  that has at most half as many distinct values as there are train + test + original rows (135 pairs, 405 columns),
  fitted on the fold's training rows + original rows.
- **Original rows:** all 10,000 are appended to each fold's training rows (never to its validation rows).
- **Model:** CatBoost on one GPU, MultiClass loss with the notebook's tuned parameters (depth 6, lr 0.0447,
  l2_leaf_reg 4.41, random_strength 2.21, bagging_temperature 0.318), up to 5,000 iterations, balanced class weights,
  early stopping 200 on `TotalF1:average=Macro` of the validation fold (best model kept), seed 42. Test = mean of the
  5 fold models.

**Changed to run it on our folds:** the notebook appends the original rows to the competition rows before a 10-fold
`StratifiedKFold` (seed 42), so original rows are also validated on, and for its submission multiplies the class
probabilities by weights tuned on its OOF (grid + Nelder-Mead). Our run uses our 5 folds of the competition rows,
appends all original rows to every fold's training rows (the target encoder is fitted on those rows) and removes the
threshold tuning: the pair holds the fold models' own probabilities. The notebook version we ported has no
pseudo-labelling step (unlike `../blamerx_xgb` and `../blamerx_lgbm`), and our run has none either.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/blamerx_catboost_oof.npy` and `preds/blamerx_catboost_test.npy` (probabilities in class order Low,
Medium, High). Needs only `data/` (train.csv, test.csv, irrigation_prediction.csv) and `common/` (shared fold loop:
`../lib/runner.py`). One GPU. Packages: catboost, scikit-learn, pandas, numpy.

The pair is a member of stack B (stack A and stack B are the two stacks of `stack/`).

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| blamerx_catboost | B | CatBoost on formula / logit / decimal / agronomy features + 135 pair target encodings, original rows appended | 0.977986 |
