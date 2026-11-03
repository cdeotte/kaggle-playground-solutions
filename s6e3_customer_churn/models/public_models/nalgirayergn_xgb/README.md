# XGBoost by nalgirayergn (public notebook)

Ünal Giray Ergün's (@nalgirayergn) XGBoost from the public notebook
[XGB - Predict Customer Churn - CV: 0.91819](https://www.kaggle.com/code/nalgirayergn/xgb-predict-customer-churn-cv-0-91819).
Thank you, Ünal!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, the
same split the notebook uses).

- **Features (label-free, built once on train + test together):** the 16 categorical columns as pandas categories
  (XGBoost's native categorical support), the 3 numeric columns and 13 digit-position features: every integer digit
  position of tenure (2), MonthlyCharges (3) and TotalCharges (4) (as many as the largest value has) and the first two
  decimal digits of MonthlyCharges and TotalCharges. 32 features.
- **Model:** XGBoost on one GPU, depth 3, lr 0.02, subsample 0.8, colsample_bytree 0.8, max_bin 16,000, up to 20,000
  trees with early stopping (200 rounds) on the validation fold's AUC, seed 42. Test = mean of the fold models.

**Changed to run it on our folds:** nothing in the model or features (the notebook uses the same split).

```
python train.py            # finished folds are kept in parts/ and skipped when the script is restarted
```

Writes `preds/nalgirayergn_xgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv); no other repo
folder is needed (shared fold loop: `../lib/runner.py`). One GPU. Packages: xgboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| nalgirayergn_xgb | the notebook's XGBoost (native categoricals + digit positions) on our 5 folds | 0.918210 |
