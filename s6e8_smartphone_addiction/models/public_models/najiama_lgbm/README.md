# LightGBM by najiama (public notebook)

Naji's (@najiama) LightGBM from the public notebook
[Single LGBM Model LB 0.96988 CV 0.96863](https://www.kaggle.com/code/najiama/single-lgbm-model-lb-0-96988-cv-0-96863)
(version 2). Thank you, Naji!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features (label-free, train + test):** the 9 numeric predictors; stress level mapped to 0/1/2, gender and academic
  impact label-encoded, all three and the case id as categories; missing counts, active / passive screen time, 8 ratios,
  5 threshold flags (sleep < 6 / < 5 h, screen > 8 h, social > 3 h, notifications > 150), constrained-imputation features
  (bound of a missing daily total, bound / midpoint of one missing component, exact slack, case id); the frequency of
  every numeric value over train + test. The 10 columns the notebook found least important are dropped.
- **Encodings per fold:** a smoothed target encoding (smoothing 30) of every numeric predictor.
- **Model:** LightGBM, lr 0.01, 127 leaves, min_child_samples 200, feature_fraction 0.34, bagging 0.75 every 5 rounds,
  alpha 0.1, lambda 1, max_bin 1023, seed 42, up to 10,000 rounds with early stopping 500 on the validation AUC.
  Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook fitted each fold's target encoding on the training rows, applied it to
those same rows and smoothed towards the mean of all training rows; our run gives the training rows 5-fold inner
out-of-fold encodings and smooths towards the fold's own training mean. The categoricals use the sorted levels of
train + test in every frame.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/najiama_lgbm_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other repo
folder is needed. CPU, 24 threads (`THREADS`). Packages: lightgbm, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| najiama_lgbm | the notebook's LightGBM on our 5 folds | 0.968488 |
