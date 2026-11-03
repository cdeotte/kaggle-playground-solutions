# Classic models: pair target encodings, logistic regression, random forest, nested KNN stacks

Six models outside the big GBDT / NN families. Each training script writes `preds/<pair>_{oof,test}.npy` (churn
probabilities); the OOF predictions use the shared 5 folds.

**Pair target encodings + logistic regression** (`train_pair_te_lr.py`, 2 presets): all 171 pairs of the 19 columns
(each column label-encoded from its string value, so the numerics by their exact value). Per outer fold and pair, the
cuML `TargetEncoder` mean churn of the pair's value combination (no smoothing; inner 5-fold out-of-fold values for the
outer-training rows, means over all outer-training rows for the held-out and test rows) -> 171 columns -> logit z ->
[z, z^2, z^3] (`lr_pair_te_logit3`, 513 columns) or [z, z^2] (`lr_pair_te_logit2`, 342 columns), standardized ->
cuML `LogisticRegression` (L2, C 0.5). The original rows are not used.

**Logistic regression** (`train_logreg.py`): cuML `LogisticRegression` (L2, C 0.3, max_iter 3000, tol 1e-6) on 4,510
columns. 4,362 one-hot columns (levels over train + test + original): the 16 categoricals, tenure / MonthlyCharges /
TotalCharges in (at most) 100 / 512 / 2,048 and 16 / 16 / 32 bins of consecutive sorted unique original values, the
first decimal digit of the two charges, 21 categorical / coarse-bin crosses. 148 dense columns standardized per fold:
the numerics, 12 interactions (ratios, squares, logs), 13 spline hinges (fixed tenure knots, charge knots at original
quantiles), 30 category-gated numerics, 18 category-gated hinges and value - mean, value / mean and z-score of 6
numerics within Contract, InternetService, PaymentMethod and Contract x InternetService (statistics over train + test +
original). The original rows shape the features but are not training rows.

**Random forest** (`train_rf.py`): cuML `RandomForestClassifier` (1,500 trees, max_depth 16, max_features 0.3,
min_samples_leaf 15, random_state 42) on the raw columns (including `id`) plus MonthlyCharges / TotalCharges snapped to
the nearest original value and value - snap, their first / second decimal digits and a round-value flag,
TotalCharges - tenure x MonthlyCharges, MonthlyCharges / TotalCharges, a count of "Yes" services and the bigram
Contract x InternetService; per fold every string column is replaced by its cuML `TargetEncoder` value (5 interleaved
inner folds, smoothing 20).

**Nested KNN stacks** (`train_knn_nested_xgb.py`, 2 presets; `select_knn_schemes.py`; shared code in
`lib/knn_schemes.py`): GPU brute-force KNN under weighted distances (categorical mismatches, or differences of the
categories' churn rates, + weighted squared standardized numeric differences); 30 "random" weight schemes (10
generators) and 30 "smart" schemes (around the feature importances of an XGBoost fitted on the original rows).
Original-data KNN priors (k = 1 or 10 nearest of the 7,043 original rows) of the train and test rows; 8 random + 8
smart schemes are kept with blend weights. Per outer fold, train-data KNN with the kept schemes (k = 80 or 40;
neighbours = the outer-training rows; for the outer-training rows themselves out-of-fold over 5 inner folds), logistic
regressions (C 0.2) on the logits of both KNN blocks, and an XGBoost (GPU, lr 0.018, depth 4, colsample_bytree 0.32,
up to 6,000 rounds, early stopping 250 on the fold's validation rows) on the 19 raw columns + 10 stack columns (LR
averages, weighted KNN means and their blends).
- `xgb_nested_knn_k1_k80_top_schemes`: original-data k 1, train-data k 80; the 8 + 8 schemes whose priors have the
  largest mean XGBoost gain, weighted by that gain. `select_knn_schemes.py` makes this selection (original-data priors
  with k 80, a 5-fold XGBoost per family on the raw columns + the family's 30 priors) and writes it to `cache/`. In our
  run it kept random schemes 14, 22, 20, 16, 17, 18, 2, 7 and smart schemes 24, 4, 14, 11, 21, 7, 27, 26.
- `xgb_nested_knn_k10_k40_random_schemes`: original-data k 10, train-data k 40; one scheme from each of 8 random
  buckets of scheme indices per family (seed 42 + 1000 x 10 + 10 x 40) with Dirichlet(0.7) blend weights.

```
python train_pair_te_lr.py lr_pair_te_logit3
python train_pair_te_lr.py lr_pair_te_logit2
python train_logreg.py
python train_rf.py
python select_knn_schemes.py
python train_knn_nested_xgb.py xgb_nested_knn_k1_k80_top_schemes
python train_knn_nested_xgb.py xgb_nested_knn_k10_k40_random_schemes
```
No other folder has to run first. `select_knn_schemes.py` must run before the `top_schemes` preset.
`train_knn_nested_xgb.py` takes `--folds 0,1` to run a subset (finished folds are kept in `parts/`; the pair is written
when the last fold is done; the original-data KNN features are rebuilt in every run). Times in our runs (one GPU):
nested KNN stacks about one hour per outer fold (half of it the inner-fold train-data KNN), scheme selection about 12
minutes, logistic regression about 5 minutes. The logistic regression keeps dense float32 one-hot matrices: plan for
about 40 GB of RAM (the fold's training matrix alone is about 9 GB, also on the GPU).

| Pair | Model | OOF AUC |
|---|---|---:|
| lr_pair_te_logit3 | cuML logistic regression on 171 pair target encodings -> [z, z^2, z^3] | 0.916014 |
| lr_pair_te_logit2 | cuML logistic regression on 171 pair target encodings -> [z, z^2] | 0.914806 |
| logreg_ohe_crosses_hinges_group_dense | cuML logistic regression, 4,362 one-hot (bins, digits, crosses) + 148 dense columns | 0.916847 |
| rf_snap_digits_te | cuML random forest, snap / decimal-digit features, per-fold target encodings | 0.914143 |
| xgb_nested_knn_k1_k80_top_schemes | XGBoost on original-data KNN (k 1), nested train-data KNN (k 80), LR meta features; top-gain schemes | 0.915721 |
| xgb_nested_knn_k10_k40_random_schemes | XGBoost on original-data KNN (k 10), nested train-data KNN (k 40), LR meta features; random schemes | 0.915561 |

Packages: RAPIDS cuML / cuDF / CuPy (26.02 in our runs; pair target encodings, logistic regressions, random forest),
PyTorch (GPU KNN), xgboost (CUDA; 3.x in our runs), scikit-learn, pandas (2.3.3; `train_rf.py` finds the string columns
by their `object` dtype), numpy.
