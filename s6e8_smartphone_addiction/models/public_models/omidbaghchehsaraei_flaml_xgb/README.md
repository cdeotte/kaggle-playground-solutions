# FLAML XGBoost by omidbaghchehsaraei (public notebook)

OMID BAGHCHEH SARAEI's (@omidbaghchehsaraei) FLAML-tuned XGBoost from the public notebook
[FLAML[XGBoost] for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/flaml-xgboost-for-predicting-smartphone-addiction)
(version 1). Thank you, Omid!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features** (`../lib/ratio_te.py`): the 12 raw predictors as string categories, missing count and 9 screen-time
  ratio features (NaN kept), and per raw predictor a smoothed target encoding (smoothing 10) and a frequency encoding.
- **Model:** FLAML `AutoML` restricted to `xgb_limitdepth`, metric roc_auc, a 7,000 s budget per fold with the fold's
  validation rows as FLAML's validation set, early stop, no ensemble, seed 42 + fold. In our run the selected models had
  depth 2 to 9 and 40 to 1,777 trees. Test = mean of the 5 fold models.

**Changed to run it on our folds:** the notebook computed the target / frequency encodings once on all training rows
(5-fold out-of-fold) before its CV loop; our run computes them inside each fold (5-fold inner out-of-fold target
encodings for the training rows, the fold's training-row map for validation and test, frequencies counted on the
training rows). The notebook converted each fold's predictions to ranks before averaging; our run keeps the
probabilities.

```
python train.py            # all 5 folds; --folds 0,1 runs a subset (finished folds are kept in parts/ and skipped)
```

Writes `preds/omidbaghchehsaraei_flaml_xgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no
other repo folder is needed (shared code: `../lib/`). CPU, about 2 hours per fold (the time budget).
Packages: flaml (we used 2.6.0), xgboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| omidbaghchehsaraei_flaml_xgb | the notebook's FLAML XGBoost on our 5 folds | 0.966447 |
