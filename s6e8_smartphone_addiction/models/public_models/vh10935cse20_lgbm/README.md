# LightGBM by vh10935cse20 (public notebook)

SARANG's (@vh10935cse20) LightGBM from the public notebook
[Mobile Addiction || LGBM](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm) (version 17).
Thank you, Sarang!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features (label-free):** missing indicators of the 9 numeric predictors, ~30 sleep / weekly-usage / activity-share /
  engagement / interaction / log features, and 3 categorical combinations (stress x impact, gender x stress, all three).
- **Per fold:** normalized frequencies of the 12 raw predictors; differences and z-scores of 5 usage columns against
  their means within 3 categorical groups (statistics from the fold's training rows); smoothed target encodings
  (smoothing 15) of the 3 combinations and the 3 categoricals. The combination strings are then dropped.
- **Model:** LightGBM, lr 0.008, 45 leaves, depth 7, feature_fraction 0.7, bagging 0.85 every round,
  min_child_samples 35, alpha 1.5, lambda 3.5, seed 42, up to 10,000 rounds with early stopping 300 on the validation
  AUC. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook uses `StratifiedKFold(10, shuffle=True, random_state=42)`; our run uses
our 5 folds. The notebook fitted each fold's target encodings on the training rows and applied them to those same
rows; our run gives the training rows 5-fold inner out-of-fold encodings (validation and test rows keep the fold's
training-row map). The notebook averaged rank-transformed fold predictions for its submission; our run averages the
probabilities. The categoricals use the sorted levels of all frames, and the columns come in a slightly different order.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/vh10935cse20_lgbm_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed. CPU, 24 threads (`THREADS`). Packages: lightgbm, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| vh10935cse20_lgbm | the notebook's LightGBM on our 5 folds | 0.966486 |
