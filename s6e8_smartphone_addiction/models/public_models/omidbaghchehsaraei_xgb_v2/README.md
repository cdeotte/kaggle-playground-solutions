# XGBoost v2 by omidbaghchehsaraei (public notebook)

OMID BAGHCHEH SARAEI's (@omidbaghchehsaraei) XGBoost from the public notebook
[XGBoost v2 for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/xgboost-v2-for-predicting-smartphone-addiction)
(version 1). Thank you, Omid!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses). `features.py` is the notebook's feature code (cells 1-5, only the data loading changed);
`train.py` is the fold loop.

- **Features (label-free):** the 9 numeric predictors, missing count, 22 ratio / intensity / log features (screen-time
  shares, unaccounted screen time, productivity and disruption indices, weekly estimate, ...), sin / cos of 6 hour
  columns and 3 ratios, decile bins (edges from train) of 8 columns with their sin / cos, and 5 bin-group mean
  differences / ratios over train + test.
- **Encodings per fold:** the 12 raw predictors and the 8 bins get a smoothed target encoding (smoothing 10) and a
  frequency encoding (counts on the fold's training rows).
- **Model:** XGBoost, lr 0.01, depth 7, colsample_bytree 0.2, max_bin 256, hist, seed 42, up to 100,000 rounds with
  early stopping 200 on the validation AUC (best iterations 5,845 to 7,144 in our run). Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook encoded each fold's training rows with the map fitted on those same
rows; our run gives them 5-fold inner out-of-fold target encodings (`../lib/ratio_te.py`, `nested_encodings`), while the
validation and test rows keep the fold's training-row map. The encoding columns come in a slightly different order
(target encoding before frequency for each column).

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/omidbaghchehsaraei_xgb_v2_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no
other repo folder is needed. CPU, 12 threads (`THREADS`). Packages: xgboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| omidbaghchehsaraei_xgb_v2 | the notebook's XGBoost v2 on our 5 folds | 0.968659 |
