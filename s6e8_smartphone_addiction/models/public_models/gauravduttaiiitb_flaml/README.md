# FLAML by gauravduttaiiitb (public notebook)

Gaurav Dutta's (@gauravduttaiiitb) FLAML AutoML model from the public notebook
[Ex51 : S6E8 : FLAML : roc_auc](https://www.kaggle.com/code/gauravduttaiiitb/ex51-s6e8-flaml-roc-auc) (version 1).
Thank you, Gaurav!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features (no target-dependent features):** the 9 numeric predictors and one-hot columns of the 3 categoricals
  (missing -> 'Unknown'), memory-downcast (integers / floats downcast, booleans -> int8).
- **Model:** FLAML `AutoML(task='classification', metric='roc_auc', time_budget=9 hours)` with FLAML's default
  estimator list and its own internal validation; the selected single estimator predicts (no ensemble). In our run FLAML
  selected CatBoost in 3 folds and depth-limited XGBoost in 2. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook runs one search on all training rows; our run runs one search per fold
on the fold's training rows only (the validation rows are not shown to FLAML).

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/gauravduttaiiitb_flaml_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed (shared fold loop: `../lib/runner.py`). CPU, 9 hours per fold (the time budget; folds can run in
parallel processes). Packages: flaml (we used 2.6.0) with its default learners (lightgbm, xgboost, catboost,
scikit-learn), pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| gauravduttaiiitb_flaml | the notebook's FLAML AutoML on our 5 folds | 0.958878 |
