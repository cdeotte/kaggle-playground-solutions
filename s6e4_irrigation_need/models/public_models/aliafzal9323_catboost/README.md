# CatBoost by aliafzal9323 (public notebook)

The CatBoost from @aliafzal9323's public notebook
[s6e4-0-978-xgb-cat-pairwise-te-magic](https://www.kaggle.com/code/aliafzal9323/s6e4-0-978-xgb-cat-pairwise-te-magic)
(the utaazu notebooks in `../utaazu_catboost` and `../utaazu_xgb` credit it for their pairwise target encoding).
Thank you!

This is the notebook's CatBoost (we do not use its XGBoost) with its features and parameters, run in a nested 5 x 5
cross-validation on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features:** the 8 categoricals, factorized over train + test + original rows (CatBoost categorical features), and
  the 11 numerics; `TE_ORIG_<col>` for all 19 columns, the mean of the 0 / 1 / 2-coded target over the original rows
  with that value, 0.5 where there is none (the categoricals are already integer codes when they are mapped, so their
  8 `TE_ORIG` columns are the constant 0.5); per inner fold, a multiclass `TargetEncoder` (cv=5, seed 42) of every
  pairwise combination of the 19 columns that has at most half as many distinct values as there are train + test +
  original rows (135 pairs, 405 columns), fitted on the inner training rows.
- **Original rows:** used only for `TE_ORIG`; none are appended to the training rows.
- **Model:** CatBoost on one GPU, up to 10,000 iterations, lr 0.03, depth 6, balanced class weights, early stopping
  500 on `TotalF1:average=Macro` of the inner validation fold (best model kept), seed 42 + inner fold (43-47).
- **Nested CV:** for outer fold k, `StratifiedKFold(5, shuffle=True, random_state=1001 + k)` splits the outer training
  rows into 5 inner folds. Each of the 5 inner models predicts its inner validation rows, the outer held-out rows and
  test. Slice k of the OOF (shape `(5, 630000, 3)`) holds the inner out-of-fold predictions for the outer training
  rows and the mean of the 5 inner models for the outer held-out rows; the stacker averages the 5 slices. Test = mean
  of the 25 inner models.

**Changed to run it on our folds:** the nested 5 x 5 scheme around the notebook's CatBoost is ours (the same as in
`../emanuellcs_catboost`): every outer fold also yields out-of-fold predictions for its training rows, so the pair
carries 5 complete OOF slices; the pair target encoding is fitted inside each inner fold.

```
python train.py            # all 5 outer folds; --folds 0,1 runs a subset (finished outer folds are kept in parts/ and skipped)
```

Writes `preds/aliafzal9323_catboost_oof.npy` (`(5, 630000, 3)`) and `preds/aliafzal9323_catboost_test.npy`
(`(270000, 3)`), probabilities in class order Low, Medium, High. Needs only `data/` (train.csv, test.csv,
irrigation_prediction.csv) and `common/` (shared fold loop: `../lib/runner.py`). One GPU. Packages: catboost,
scikit-learn, pandas, numpy.

The pair is a member of stack B (stack A and stack B are the two stacks of `stack/`). OOF balanced accuracy of the mean
of the 5 slices:

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| aliafzal9323_catboost | B | the notebook's CatBoost (TE_ORIG + 135 pair target encodings), nested 5 x 5 | 0.977471 |
