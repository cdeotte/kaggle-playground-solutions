# LightGBM by shivamgravity (public notebook)

Shivam's (@shivamgravity) LightGBM from the public notebook
[pgs-s6e5-lgbm-optuna](https://www.kaggle.com/code/shivamgravity/pgs-s6e5-lgbm-optuna) (version 15). Thank you,
Shivam!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features:** the 14 competition columns; Driver, Compound and Race as pandas categoricals (LightGBM's native
  categorical handling).
- **Model:** LightGBM with the best parameters of the notebook's Optuna search (trial 15): lr 0.0188, 177 leaves,
  min_child_samples 60, feature_fraction 0.636, bagging 0.896 every 3 rounds, reg_alpha 0.000149, reg_lambda 0.00457,
  balanced class weights, up to 1,000 rounds with early stopping 50 on the validation AUC, seed 42. Test = mean of the
  5 fold models at their best iteration.

**Changed to run it on our folds:** the notebook ran a 20-trial Optuna search (7-fold CV on a 100,000-row sample) and
trained its final models on `StratifiedKFold(7, shuffle=True, random_state=42)`. Our run skips the search, uses the best
parameters printed by the notebook's run (trial 15) and our 5 folds, with 32 CPU threads (`THREADS`) instead of all
cores; OOF and test are clipped to [1e-7, 1 - 1e-7].

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/shivamgravity_lgbm_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed (shared fold loop: `../lib/runner.py`). CPU. Packages: lightgbm, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| shivamgravity_lgbm | the notebook's LightGBM (Optuna trial 15 parameters) on our 5 folds | 0.945580 |
