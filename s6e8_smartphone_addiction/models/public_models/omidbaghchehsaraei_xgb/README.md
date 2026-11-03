# XGBoost by omidbaghchehsaraei (public notebook)

OMID BAGHCHEH SARAEI's (@omidbaghchehsaraei) XGBoost from the public notebook
[XGBoost for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/xgboost-for-predicting-smartphone-addiction)
(version 1). Thank you, Omid!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features** (`../lib/ratio_te.py`): the 12 raw predictors as string categories (XGBoost native categoricals),
  missing count and 9 screen-time ratio features (NaN kept), and per raw predictor a smoothed target encoding
  (smoothing 10) and a frequency encoding.
- **Model:** XGBoost, lr 0.01, depth 7, min_child_weight 10, subsample 0.8, colsample_bytree 0.5, alpha 0.1, lambda 1,
  max_bin 2000, hist on CUDA, seed 42, up to 20,000 rounds with early stopping 150 on the validation AUC.
  Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook computed the target / frequency encodings once on all training rows
(5-fold out-of-fold) before its CV loop. Our run computes them inside each fold: the training rows get 5-fold inner
out-of-fold target encodings, the validation and test rows the map fitted on all training rows of the fold, and the
frequencies are counted on the fold's training rows.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/omidbaghchehsaraei_xgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other
repo folder is needed (shared code: `../lib/`). One GPU. Packages: xgboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| omidbaghchehsaraei_xgb | the notebook's XGBoost on our 5 folds | 0.967041 |
