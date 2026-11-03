# TabM by blamerx (public notebook)

BlamerX's (@blamerx) pytabkit TabM from the public notebook
[S6E3 | TabM + Advanced Features | 0.91898 CV](https://www.kaggle.com/code/blamerx/s6e3-tabm-advanced-features-0-91898-cv).
Thank you, BlamerX!

This is the notebook's model and features run on our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`).

- **Features (built once on train / test; the original dataset only supplies statistics, its rows are not appended):**
  frequencies of the 3 numerics over train + original + test; charges deviation (TotalCharges - tenure x
  MonthlyCharges), monthly / total and total / tenure ratios; service count, has-internet and has-phone flags; the
  original dataset's churn rate of every categorical and numeric value (19 columns, 0.5 if unseen); TotalCharges
  percentile ranks and z-scores against the original churners, non-churners and all original rows, conditional
  percentile ranks within InternetService and Contract, the MonthlyCharges residual from the original InternetService
  mean, and distances to the original churner / non-churner TotalCharges quartiles; 35 digit and rounding features of
  tenure, MonthlyCharges and TotalCharges (first / second / last digit, mod 10 / 12 / 100, number of digits, multiples,
  rounding deviations, fractional parts, years); 15 bigrams of Contract, InternetService, PaymentMethod,
  OnlineSecurity, TechSupport, OnlineBackup and 4 trigrams of the first four.
- **Encodings per fold:** the 3 numerics as strings and the 16 categoricals are mean-target-encoded with 5 inner
  stratified folds over the training rows (validation and test rows get the means of all training rows), the 19
  n-grams with the training-row means; unseen values get 0.5. The 121 numeric columns are standardized, the 16
  categoricals ordinal-encoded.
- **Model:** pytabkit `TabM_D_Classifier`: `tabm-mini-normal`, k = 32, piecewise-linear embeddings (d 24), 3 blocks of
  256, dropout 0.2, lr 1e-3, weight decay 1e-3, batch 512, up to 50 epochs with patience 10 on the validation fold,
  seed 42. The notebook seeds once (42) and trains its folds in one loop, so `train.py` runs the folds in order
  in one process. Test = mean of the fold models.

**Changed to run it on our folds:** the notebook trains 10 folds; our run uses our 5 folds (5 inner folds for the
target encodings, as in the notebook). The sixth n-gram column is OnlineBackup (the notebook's latest version uses
PaperlessBilling).

```
python train.py            # all folds in order in one process
```

Writes `preds/blamerx_tabm_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv,
WA_Fn-UseC_-Telco-Customer-Churn.csv); no other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU.
Packages: pytabkit (our run: 1.7.3), torch, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| blamerx_tabm | the notebook's TabM (original-data statistics, digits, nested target encodings, n-grams) on our 5 folds | 0.918788 |
