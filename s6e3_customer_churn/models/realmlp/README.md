# RealMLP: pytabkit RealMLP models

Six pytabkit `RealMLP_TD_Classifier` models in five scripts (shared code in `lib/appended_original.py`). The fold's
held-out rows are passed to `fit` as the validation set (pytabkit keeps the best epoch on them; configurations with early
stopping also stop on them). Test = mean of the fold models.

Folds: the 5 folds of `common/data.py`, except for the three `*_original` models, which append the 7,032 original rows
with a TotalCharges to the 594,194 competition rows before the split: their folds are
`StratifiedKFold(5, shuffle=True, random_state=42)` over the 601,226 rows, a fold trains on the competition and original
rows of the 4 other folds and is validated on all its own rows, and the pair keeps the OOF of the competition rows.

Credits:
- `train_3config_digits_bins_rare_original.py`, `realmlp_3config_crosses_priors_drift_original` and
  `train_sincos_digit_combo_te_stats.py` adapt mahoganybuttstrings's public notebook
  [PG S6E1 RealMLP](https://www.kaggle.com/code/mahoganybuttstrings/pg-s6e1-realmlp-cv-8-58748-lb-8-58006).
- `realmlp_risk_score_crosses_priors_drift_original` takes its RealMLP configuration, risk score and original value rates
  from mahoganybuttstrings's [PG S5E10 RealMLP](https://www.kaggle.com/code/mahoganybuttstrings/pg-s5e10-realmlp-cv-0-055936-lb-0-05549).
- `train_nested_te_stats_ngram.py` runs the features of blamerx's
  [XGBoost bi/tri target encoding](https://www.kaggle.com/code/blamerx/auc-0-91925-xgboost-bi-tri-target-encoding) with a RealMLP.
- `train_pair_te_logit3_kbins_priors.py` adapts include4eto's
  [RealMLP feature engineering](https://www.kaggle.com/code/include4eto/realmlp-feature-engineering).

## Models

- **`train_3config_digits_bins_rare_original.py`** (original rows appended): the 19 raw columns; for tenure,
  MonthlyCharges and TotalCharges a NaN flag, the base-10 digits of the value rounded to its step (1, 0.5, 5; x10 for
  MonthlyCharges) and a sign flag, quantile (4, 5, 10, 20), equal-width (5, 10, 20) and rounded bins; every raw column
  as a (rounded) string category; rare flags (< 25 rows) of the rounded charges. Computed once over the 601,226 rows +
  test. Three tuned RealMLP configurations (mish, PBLD embeddings, n_ens 8, up to 32 epochs, batch 1024; two with early
  stopping), probabilities averaged.
- **`train_crosses_priors_drift_original.py`** (original rows appended, numerics float32), two presets. Features built
  for the 601,226 rows, the test rows and the original rows: frequency and percentile rank of the 3 numerics; log1p /
  sqrt / 1 / (1 + x) / missing flags; TotalCharges - tenure x MonthlyCharges (and its absolute value) and 5 more ratios /
  products; Yes / No / no-service counts over the 8 services, has_internet, has_phone; Yes / No / other flags of 11
  columns; tenure bins, 40 / 60 quantile bins of the charges, 6 categorical pairs and Contract x InternetService x
  PaymentMethod; counts and rare flags (<= 50) of the 26 categories; original churn rates (0.5 if unseen) of the 26
  categories and of the 3 numeric values; drift log1p(train frequency / (original frequency + 1e-6)); plus the
  digit / bin / rounded-category block above.
  - `realmlp_3config_crosses_priors_drift_original`: the three configurations above, averaged.
  - `realmlp_risk_score_crosses_priors_drift_original`: + a handcrafted churn-risk score (weighted flags of contract,
    billing, payment, fiber, missing security / support, short tenure, high charges, large charge deviation) and the
    original churn rate of each of the 19 raw values; one configuration (2 x 512 mish, PBLD, p_drop 0.42, n_ens 8,
    30 epochs, no early stopping).
- **`train_sincos_digit_combo_te_stats.py`** (all 7,043 original rows added to every fold's training rows): sin / cos
  of tenure and MonthlyCharges (periods 12, 14, 20), squares, digits ((x * 10^k) mod 10) of SeniorCitizen, tenure and
  the charges as categories; 8 numeric x categorical pairs and 6 pairs of digit features, encoded per fold by
  `TargetEncoder_` (target mean (smoothing auto) / std / skew; training rows out-of-fold over 5 folds); SeniorCitizen,
  tenure and the charges factorized as categories. RealMLP 3 x 384 mish, n_ens 8, lr 0.03, up to 60 epochs with early
  stopping.
- **`train_nested_te_stats_ngram.py`** (no original rows in training): frequencies, 3 charge ratios, service counts,
  original churn rate of every value, TotalCharges against the original data (percentile ranks among all / churned /
  retained and within InternetService / Contract, z-scores, quartile distances), 35 digit / rounding features, the 3
  numerics as categories, 15 bigrams and 4 trigrams; per fold, nested (5 inner folds) label std / min / max of the 19
  category keys and label means of the 19 n-grams, and a scikit-learn `TargetEncoder` of the 19 keys. RealMLP
  4 x 512 mish, n_ens 1, lr 0.13, up to 64 epochs with early stopping.
- **`train_pair_te_logit3_kbins_priors.py`** (no original rows in training): service counts, 6 charge ratios /
  products, (TotalCharges / 1000) mod 10, KBinsDiscretizer quantile bins of MonthlyCharges (200), TotalCharges (4000)
  and tenure (25), factorized value codes of the 4 numerics, original churn rates of every value and the original
  TotalCharges distribution features; per fold, smoothed (m = 20) target means of all 105 pairs of the 15 string
  categoricals (inner 5 folds, seed 42 + fold) as logit, logit^2, logit^3, standardized. RealMLP 4 x 352 mish,
  n_ens 8, lr 0.075, batch 256, up to 10 epochs with early stopping.

```
python train_3config_digits_bins_rare_original.py
python train_crosses_priors_drift_original.py realmlp_3config_crosses_priors_drift_original
python train_crosses_priors_drift_original.py realmlp_risk_score_crosses_priors_drift_original
python train_sincos_digit_combo_te_stats.py
python train_nested_te_stats_ngram.py
python train_pair_te_logit3_kbins_priors.py
```
Each run trains its folds in one process on one GPU and writes `preds/<pair>_{oof,test}.npy` (probabilities). Only
`data/` is read; no other repo folder has to run first.

| Pair | Model | OOF AUC |
|---|---|---:|
| realmlp_3config_digits_bins_rare_original | 3 RealMLP configurations on digit / bin / rounded-category / rare features, original rows appended | 0.917064 |
| realmlp_3config_crosses_priors_drift_original | 3 RealMLP configurations on rank / frequency / crosses / counts / original priors / drift + digit block, original rows appended | 0.917810 |
| realmlp_risk_score_crosses_priors_drift_original | the same features + risk score and original value rates, one RealMLP configuration | 0.917546 |
| realmlp_sincos_digit_combo_te_stats | RealMLP on sin / cos, digits and mean / std / skew TEs of 14 pair codes, original rows in every fold | 0.916634 |
| realmlp_nested_te_stats_ngram | RealMLP with nested label-stat TEs of categories / numeric values, n-gram means and original distribution features | 0.913508 |
| realmlp_pair_te_logit3_kbins_priors | RealMLP on KBins bins, value codes, original priors and 105 pair TEs as logit powers | 0.918292 |

For the `*_original` models the score is that of the competition rows.

Requirements: pytabkit, torch (CUDA), scikit-learn (>= 1.3 for `TargetEncoder`), pandas 2.x, numpy.
