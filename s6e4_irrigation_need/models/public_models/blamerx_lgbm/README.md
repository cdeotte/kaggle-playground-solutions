# LightGBM by blamerx (public notebook)

The LightGBM from @blamerx's public notebook
[S6E4 🍀| LightGBM High Balance | 0.979 CV](https://www.kaggle.com/code/blamerx/s6e4-lightgbm-high-balance-0-979-cv).
Thank you!

This is the notebook's model and features on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`),
without its pseudo-labelling and threshold tuning. The features are the same as in `../blamerx_xgb`.

- **Features (80 + 405 encoded):** the 19 raw columns (categoricals label-encoded over train + test + original rows);
  7 formula indicators (Soil_Moisture < 25, Rainfall_mm < 300, Temperature_C > 30, Wind_Speed_kmh > 10, Harvest,
  Sowing, Mulching = Yes), the formula score 2 soil + 2 rain + temp + wind - 2 harvest - 2 sowing - mulch, its
  distances to the class boundaries 0 and 3 and the signed distances to the 4 thresholds; `logit_Low` /
  `logit_Medium` / `logit_High`, a fixed logistic regression of those indicators fitted on the original data (100%
  accurate there); the first decimal digit of the 11 numerics; 32 agronomy features (water input, ET / evaporation /
  VPD / heat / dryness / aridity / drought proxies, soil health, salinity, pH deviation, per-hectare ratios, mulch
  effects, logs).
- **Encodings per fold:** a multiclass `TargetEncoder` (cv=5, seed 42) of every pairwise combination of the 19 columns
  that has at most half as many distinct values as there are train + test + original rows (135 pairs, 405 columns),
  fitted on the fold's training rows + original rows.
- **Original rows:** all 10,000 are appended to each fold's training rows (never to its validation rows); balanced
  sample weights computed on those training rows.
- **Model:** LightGBM multiclass with the notebook's tuned parameters (num_leaves 33, lr 0.0311, min_child_samples 62,
  feature_fraction 0.685, bagging_fraction 0.877 every round, lambda_l1 0.0003, lambda_l2 7.25), up to 10,000 rounds,
  early stopping 300 on the validation fold's multi_logloss (predictions use the best iteration), seed 42. Test = mean
  of the 5 fold models.

**Changed to run it on our folds:** the notebook appends the original rows to the competition rows before a 10-fold
`StratifiedKFold` (seed 42), so original rows are also validated on, and computes its balanced sample weights once on
all rows; for its submission it then retrains one model on all rows plus the test rows whose top probability is at
least 0.92 (pseudo-labels) and multiplies the class probabilities by weights tuned on its OOF (grid + Nelder-Mead). Our
run uses our 5 folds of the competition rows, appends all original rows to every fold's training rows (the sample
weights and the target encoder are fitted on those rows) and removes the pseudo-labelling and the threshold tuning:
the pair holds the fold models' own probabilities. LightGBM runs on the CPU with 32 threads (the notebook used its GPU
build).

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/blamerx_lgbm_oof.npy` and `preds/blamerx_lgbm_test.npy` (probabilities in class order Low, Medium,
High). Needs only `data/` (train.csv, test.csv, irrigation_prediction.csv) and `common/` (shared fold loop:
`../lib/runner.py`). CPU only. Packages: lightgbm, scikit-learn, pandas, numpy.

The pair is a member of stack B (stack A and stack B are the two stacks of `stack/`).

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| blamerx_lgbm | B | LightGBM on formula / logit / decimal / agronomy features + 135 pair target encodings, original rows appended | 0.975711 |
