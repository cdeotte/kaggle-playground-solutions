# CatBoost by utaazu (public notebook)

The CatBoost from @utaazu's public notebook
[0.979 CV Single CAT: Pairwise TE + Bias Tuning](https://www.kaggle.com/code/utaazu/0-979-cv-single-cat-pairwise-te-bias-tuning).
Thank you!

This is the notebook's model and features on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses), without its bias tuning.

- **Features:** the 19 raw columns (the 8 categoricals as CatBoost categorical features); ET_proxy
  (temperature x wind x sunlight / (humidity + 1)), water_deficit and soil_quality; `TE_ORIG_<col>` for the 19 columns
  from the original data (categoricals: the target mean per category; numerics: the target mean in 10 equal-width bins
  of the original column).
- **Encodings per fold:** a multiclass `TargetEncoder` (cv=3, seed 42) of every pairwise combination of the 19 columns
  that has at most half as many distinct values as there are train + test + original rows (135 pairs, 405 columns),
  fitted on the fold's training rows + original rows; the raw pair codes are dropped.
- **Original rows:** all 10,000 are appended to each fold's training rows. Sample weights: balanced class weights for
  all rows, then x 0.35 for the original rows.
- **Model:** CatBoost on one GPU, SymmetricTree, depth 8, lr 0.03, up to 3,000 iterations, MultiClass loss, early
  stopping 150 on `TotalF1` of the validation fold (best model kept), seed 42. Test = mean of the 5 fold models.

**Changed to run it on our folds:** only the post-processing. The notebook adds per-class offsets to the logits of its
OOF and test probabilities, tuned by coordinate descent (steps 1.0 down to 0.002) to maximize the OOF balanced
accuracy, and saves the shifted probabilities; our pair keeps the model's own probabilities. The notebook's
adversarial-validation check and plots are left out.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/utaazu_catboost_oof.npy` and `preds/utaazu_catboost_test.npy` (probabilities in class order Low,
Medium, High). Needs only `data/` (train.csv, test.csv, irrigation_prediction.csv) and `common/` (shared fold loop:
`../lib/runner.py`). One GPU. Packages: catboost, scikit-learn, pandas, numpy.

The pair is a member of stack B (stack A and stack B are the two stacks of `stack/`).

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| utaazu_catboost | B | SymmetricTree CatBoost, TE_ORIG priors + 135 pair target encodings (cv=3), original rows x 0.35 | 0.977055 |
