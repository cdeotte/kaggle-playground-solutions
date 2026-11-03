# HistGradientBoosting by tamerlanomralinov (public notebook)

Tamerlan Omralinov's (@tamerlanomralinov) HistGradientBoosting baseline from the public notebook
[S6E5 | HistBoost Baseline](https://www.kaggle.com/code/tamerlanomralinov/s6e5-histboost-baseline) (version 2). Thank
you, Tamerlan!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features (label-free):** the 14 competition columns plus `lap_tyre` = LapNumber x TyreLife; Driver, Compound and
  Race label-encoded on the train + test values.
- **Model:** scikit-learn `HistGradientBoostingClassifier`, lr 0.03, max_depth 6, l2_regularization 0.1, up to 3,000
  iterations with its internal early stopping (10% validation split of the training rows, 100 iterations without
  improvement), seed 42. Test = mean of the 5 fold models.

**Changed to run it on our folds:** nothing in the model or features; OOF and test are clipped to [1e-7, 1 - 1e-7].

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/tamerlanomralinov_hgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed (shared fold loop: `../lib/runner.py`). CPU. Packages: scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| tamerlanomralinov_hgb | the notebook's HistGradientBoostingClassifier on our 5 folds | 0.947546 |
