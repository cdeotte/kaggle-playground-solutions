# CatBoost by emanuellcs (public notebook)

Emanuel Lázaro's (@emanuellcs) CatBoost from the public notebook
[Predicting Irrigation Need | XGBoost + CatBoost](https://www.kaggle.com/code/emanuellcs/predicting-irrigation-need-xgboost-catboost).
Thank you, Emanuel!

This is the notebook's CatBoost (we do not use its XGBoost) with its features, original-row handling and parameters,
run in a nested 5 x 5 cross-validation on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features:** the 19 raw columns; `TE_ORIG_<col>` for each of them, the mean of the 0 / 1 / 2-coded target over the
  original rows with that value (0.5 for values the original data does not have); every pairwise string combination
  of the 19 columns that has at most half as many distinct values as there are train + test + original rows (135 of
  the 171 pairs). The 8 categoricals and the pair combinations are CatBoost categorical features.
- **Original rows:** all 10,000 are appended to every inner training fold (never to a validation fold).
- **Model:** CatBoost on one GPU, MultiClass loss, up to 15,000 iterations, lr 0.02, depth 6, balanced class weights,
  early stopping 500 on the inner validation fold's MultiClass loss (best model kept), seed 42 + inner fold (43-47).
- **Nested CV:** for outer fold k, `StratifiedKFold(5, shuffle=True, random_state=1001 + k)` splits the outer training
  rows into 5 inner folds. Each of the 5 inner models predicts its inner validation rows, the outer held-out rows and
  test. Slice k of the OOF (shape `(5, 630000, 3)`) holds the inner out-of-fold predictions for the outer training
  rows and the mean of the 5 inner models for the outer held-out rows; the stacker averages the 5 slices. Test = mean
  of the 25 inner models.

**Changed to run it on our folds:** the nested 5 x 5 scheme around the notebook's CatBoost is ours: every outer fold
also yields out-of-fold predictions for its training rows, so the pair carries 5 complete OOF slices.

```
python train.py            # all 5 outer folds; --folds 0,1 runs a subset (finished outer folds are kept in parts/ and skipped)
```

Writes `preds/emanuellcs_catboost_oof.npy` (`(5, 630000, 3)`) and `preds/emanuellcs_catboost_test.npy` (`(270000, 3)`),
probabilities in class order Low, Medium, High. Needs only `data/` (train.csv, test.csv, irrigation_prediction.csv) and
`common/` (shared fold loop: `../lib/runner.py`). One GPU. Packages: catboost, scikit-learn, pandas, numpy.

The pair is a member of stack B (stack A and stack B are the two stacks of `stack/`). OOF balanced accuracy of the mean
of the 5 slices:

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| emanuellcs_catboost | B | the notebook's CatBoost (TE_ORIG + 135 pair categoricals, original rows appended), nested 5 x 5 | 0.970478 |
