# CatBoost by blamerx (public notebook)

BlamerX's (@blamerx) CatBoost from the public notebook
[S6E3 | 0.91902 Optimized CatBoost](https://www.kaggle.com/code/blamerx/s6e3-0-91902-optimized-catboost).
Thank you, BlamerX!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features (built once on train / test; the original dataset only supplies statistics, its rows are not appended):**
  frequencies of the 3 numerics over train + original + test; charges deviation, monthly / total and total / tenure
  ratios; service count, has-internet and has-phone flags; the original dataset's churn rate of every categorical and
  numeric value (19 columns, 0.5 if unseen); TotalCharges percentile ranks and z-scores against the original churners,
  non-churners and all original rows, conditional percentile ranks within InternetService and Contract, the
  MonthlyCharges residual from the original InternetService mean and distances to the original churner / non-churner
  TotalCharges quartiles; 35 digit and rounding features of tenure, MonthlyCharges and TotalCharges; the 3 numerics as
  strings; 15 bigrams of Contract, InternetService, PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling and 4
  trigrams of the first four.
- **Encodings per fold** (5 inner stratified folds over the training rows; validation and test rows get the
  statistics of all training rows): std / min / max of the target per value of the 16 categoricals and the 3 numeric
  strings (unseen: 0), the target mean per n-gram value (unseen: 0.5) and a scikit-learn `TargetEncoder` (cv 5,
  smooth 'auto', seed 42) of the 16 categoricals and 3 numeric strings. The raw categoricals, numeric strings and
  n-grams are then dropped, so CatBoost sees 178 numeric columns.
- **Model:** CatBoost on one GPU with the notebook's Optuna-tuned parameters: SymmetricTree, depth 7, lr 0.00984, up
  to 20,000 iterations with early stopping (200) on the validation fold's AUC (best model kept), l2_leaf_reg 5.333,
  random_strength 2.877, bagging_temperature 0.264, border_count 254, min_data_in_leaf 14, seed 42. Test = mean of the
  fold models.

**Changed to run it on our folds:** the notebook trains 20 folds; our run uses our 5 folds (5 inner folds for the
encodings, as in the notebook). TotalCharges is cast to float32 before the features are built.

```
python train.py            # finished folds are kept in parts/ and skipped when the script is restarted
```

Writes `preds/blamerx_catboost_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv,
WA_Fn-UseC_-Telco-Customer-Churn.csv); no other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU.
Packages: catboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| blamerx_catboost | the notebook's CatBoost (original-data statistics, digits, nested target statistics, n-grams) on our 5 folds | 0.918818 |
