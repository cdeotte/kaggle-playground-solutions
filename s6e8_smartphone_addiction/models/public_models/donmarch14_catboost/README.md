# CatBoost by donmarch14 (public notebook)

Don Mani's (@donmarch14) CatBoost from the public notebook
[S6E8: CatBoost](https://www.kaggle.com/code/donmarch14/s6e8-catboost) (version 3). Thank you, Don!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features (label-free):** the 12 raw columns (categoricals as strings, missing -> '__MISSING__'), missing flags and
  count, 8 domain relations (leisure / other / total activity screen time, weekend minus daily, sleep deficit,
  notifications per open, social and gaming share), the ones / tenths / hundredths digits of 10 columns and their
  floor / round(0.1) / mod-10 / 20ths-of-fraction values as CatBoost categoricals, activity mean / std / min / max /
  range / argmax, and 2 floor-pair categoricals.
- **Encodings per fold:** smoothed target encodings (smoothing 20) of the 3 categoricals, their 3 pairs and 20-quantile
  bins (edges from the fold's training rows) of the 9 numeric predictors; inner 4-fold out-of-fold (seed 143) for the
  training rows, the fold's training-row map for validation and test.
- **Model:** CatBoost on GPU, Logloss, up to 4,000 iterations, lr 0.06, depth 8, l2 5, random_strength 0.5,
  border_count 128, Bernoulli subsample 0.8, seed 42, early stopping (300) on the validation AUC.
  Test = mean of the 5 fold models.

**Changed to run it on our folds:** nothing; the notebook already uses our split and fold-local, inner out-of-fold
encodings (we use its GPU branch).

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/donmarch14_catboost_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed. One GPU. Packages: catboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| donmarch14_catboost | the notebook's CatBoost on our 5 folds | 0.966667 |
