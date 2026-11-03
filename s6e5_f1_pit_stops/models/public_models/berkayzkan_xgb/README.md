# XGBoost by berkayzkan (public notebook)

Berkay Özkan's (@berkayzkan) XGBoost from the public notebook
[xgboost baseline](https://www.kaggle.com/code/berkayzkan/xgboost-baseline) (version 6). Thank you, Berkay!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
split of the notebook's own 5-fold check of its best parameters).

- **Features (label-free; built once on all train rows and once on all test rows):** the 14 competition columns plus 47
  engineered columns: laps since the pit stop (= TyreLife) and its ratios / products with lap number and race progress,
  remaining race, lap time / degradation / lap-time delta per tyre lap or race lap, absolute position change,
  position x progress / tyre life, top-5 / top-10 / back-marker and gained / lost / unchanged position flags, early /
  mid / late race, first / later stint and pit-stop flags, tyre-age quartile flags (quartiles of the frame the features
  are built on), soft / medium / hard flags and their products with tyre life and progress, recent-season and testing
  flags.
- **Preprocessing per fold** (scikit-learn pipeline): median imputation of the numeric columns; most-frequent imputation
  and ordinal encoding (unknown -> -1) of Driver, Compound and Race.
- **Model:** XGBoost on one GPU (`hist`) with the best parameters of the notebook's Optuna search (trial 74): 5,000
  trees, max_depth 10, lr 0.00484, subsample 0.911, colsample_bytree 0.703, min_child_weight 8.16, gamma 1.07,
  alpha 0.697, lambda 3.87, scale_pos_weight 1, seed 42, no early stopping. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook ran a 100-trial Optuna search (fitting on a class-balanced sample of
an 80/20 split, so its scale_pos_weight is 1), checked the best parameters with 5-fold CV and refit them on all training
rows for the test predictions. Our run skips the search and uses the best parameters printed by the notebook's run
(trial 74), trains the 5 fold models and averages them for test; OOF and test are clipped to [1e-7, 1 - 1e-7].

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/berkayzkan_xgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other repo
folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages: xgboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| berkayzkan_xgb | the notebook's XGBoost (Optuna trial 74 parameters) on our 5 folds | 0.949626 |
