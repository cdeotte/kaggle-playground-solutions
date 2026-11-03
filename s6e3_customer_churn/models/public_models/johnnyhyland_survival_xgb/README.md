# Bayesian survival features + XGBoost, after johnnyhyland (public notebook)

Our fold-local version of the final model of Johnny Hyland's (@johnnyhyland) public notebook
[s6e3 - Reframing as Bayesian Survival - PyMC](https://www.kaggle.com/code/johnnyhyland/s6e3-reframing-as-bayesian-survival-pymc),
which treats churn as a time-to-event problem (tenure = time, churn = event), fits Bayesian survival models with PyMC
and feeds the frailty model's survival features, with other engineered features, into an XGBoost. Thank you, Johnny!

Everything below is fitted inside each of our 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`) on the
fold's training rows only, then applied to the fold's validation rows and to test.

- **Survival model:** a piecewise-exponential proportional-hazards model (Poisson likelihood on monthly tenure
  intervals up to 6 months, quarterly to 24, half-yearly after) with gender-stratified log baseline hazards
  (Normal(-4.5, 0.5)), a shared Gamma frailty per InternetService x Contract group (HalfNormal(1) frailty scale) and
  Normal(0, 0.5) coefficients on 9 billing / demographic predictors (male, senior, partner, dependents, paperless
  billing, credit-card and mailed-check payment, standardized MonthlyCharges and TotalCharges). PyMC's NUTS sampler,
  2 chains x (1,000 tuning + 2,000 draws), target_accept 0.9, seed 43 + fold, on 2,000 rows drawn from a 5,000-row
  sample of the fold's training rows.
- **Survival features** from the posterior means, for every row (the curve is read at interval index = tenure, capped
  at the last interval, as in the notebook): churn probability (1 - survival), frailty, survival velocity (drop to the
  next interval), hazard ratio vs the baseline, survival at 6 and 36 months and their ratio (7 features).
- **Other features:** 22 binary / standardized encodings of the raw columns; 16 engineered numerics (the 3 numerics,
  charge per month, charge ratio, charges deviation, tenure x charges, logs, square roots, internet add-on count and
  its product with tenure, any-security and any-streaming flags); the original dataset's churn rate of each of the 16
  categorical values; ranks of the 3 numerics and frequencies of 10 categoricals among the training rows; 174
  interaction codes (deciles of the 3 numerics and all 171 pairs of the 16 categoricals + 3 deciles; unseen values
  -1) replaced by their scikit-learn `TargetEncoder` values (fitted on the training rows), plus the mean / std / min /
  max of those 174 encodings. 252 features.
- **Model:** XGBoost (GPU `hist`), depth 7, lr 0.005, up to 50,000 trees with early stopping (500 rounds) on the
  validation fold's AUC, colsample_bytree 0.35, subsample 0.9, min_child_weight 6, lambda 5, alpha 0.05, max_bin 1024,
  seed 42. Test = mean of the fold models.

**Differences from the notebook:** the notebook fits its frailty model once (frailty grouped by InternetService; its 10
predictors include the contract type, electronic-check payment, online security and tech support; numpyro's NUTS on a
sample of all training rows), builds its rank,
frequency and interaction features on train + test together and trains its XGBoost on 10 folds. Our version refits the
frailty model inside every fold with a different grouping (InternetService x Contract) and predictors, adds the
survival-curve shape features (velocity, hazard ratio, 6 / 36-month survival and their ratio), fits every statistic on
the fold's training rows and uses our 5 folds.

```
python train.py            # finished folds are kept in parts/ and skipped when the script is restarted
```

Writes `preds/johnnyhyland_survival_xgb_{oof,test}.npy` (probabilities). Reads only `data/` (train.csv, test.csv,
WA_Fn-UseC_-Telco-Customer-Churn.csv); no other repo folder is needed (shared fold loop: `../lib/runner.py`). One GPU
(XGBoost); the PyMC sampling runs on the CPU. Packages: pymc, pytensor, arviz, xgboost, scikit-learn, pandas, numpy.

| Pair | Model | OOF AUC |
|---|---|---:|
| johnnyhyland_survival_xgb | fold-local PyMC frailty survival features + 245 other features -> XGBoost | 0.916741 |
