# cuML models on the density and target-encoding features

Six GPU models built with RAPIDS cuML: an SVC, a random-Fourier-feature logistic regression and a random forest on the
per-fold density matrix, an XGBoost on a feature stack built entirely with cuML, and a random forest on that stack plus
nested cuML margins.

## SVC / RFF / RF on the density matrix (`train_cuml.py`)

**Matrix** (`lib/assemble.py`, one per outer fold). 213 columns: the 12 raw columns, 15 missingness flags / counts,
13 budget-geometry columns (sum of social + gaming + work, slack, shares of daily, weekend - daily, region coordinates),
12 value counts, 31 original-data log-likelihood ratios / amplifications, 18 exact-pair PMIs, 6 interval bounds implied by
the budget identity, 11 Monte-Carlo posterior mechanism features, 9 imputation quantiles, 19 imputations / fills / region
probabilities, 26 masked-density log-probabilities and entropies, 12 lattice coordinates (fractional part and first
decimal of 6 hour columns), 15 nested target encodings (exact-value tokens of the 12 columns + 3 pair keys, smoothing 20,
inner-fold noise averaged over the inner seeds 777 / 1778 / 2779), 12 sharp encodings (smoothing 4) and 2 WOE sums.
Nested: outer-train rows are encoded by an inner `StratifiedKFold(5, shuffle=True)`, validation and test rows by the map
of all outer-train rows. The **rich** matrix (346 columns) adds 133 per-fold channels: completion block `cmp4` (23),
lift banks `plift` / `cbank` / `wbank` (10 / 9 / 5), masked-density v4 `denv4` (26), MLM fills `mfill4` (27),
reconstruction residuals `rrd` (19) and quantile fills `qfill` (14).
NaN -> outer-train median; standardised with outer-train mean / sd (not for the forest); inf -> 0.

**Models** (fitted on all outer-train rows, one seed):
- SVC: `cuml.svm.SVC(kernel='rbf', C=0.5, gamma=0.002, cache_size=4000, tol=1e-3)`, decision function.
- RFF: 4096 CuPy random Fourier features of an RBF kernel (gamma 0.002, seed 42) + `cuml LogisticRegression(C=3,
  max_iter=1000)`.
- RF: `cuml RandomForestClassifier(n_estimators=300, max_depth=20, max_features=0.3, min_samples_leaf=20, n_bins=128,
  bootstrap=True, random_state=42)`.

**Probabilities** (`lib/calibrate.py`): each fold model's validation outputs become percentile ranks within the fold and
its test outputs percentile ranks within the test set. OOF: cross-fitted isotonic regression (fold f is mapped by an
isotonic rank -> label fit on the other 4 folds). Test: the 5 test rank vectors are averaged and re-ranked, then mapped by
an isotonic fit on all OOF ranks. Both clipped to [1e-6, 1 - 1e-6].

## Models on the cuML-built feature stack

**Stack** (`build_cuml_features.py`, 99 columns per fold, built with cuML / cuDF / CuPy): raw 12, missingness 9,
geometry 8, log value counts 12, cuML `TargetEncoder` of the 12 exact-value tokens (smoothing 1 and 20) and of 10 token
pairs (smoothing 5), a nested WOE sum, nested LogisticRegression margin, KMeans (k 64 / 256) cluster encodings, nested
kNN(200) response, a Gaussian-kernel label smoother on the (daily, social) plane, 16 PCA components of the masked language
model's 384-d row embedding, and a kNN(100) fitted on the 7,500 original rows. Label-derived blocks are nested with 5
inner folds (seed 777); validation and test rows use a fit on all outer-train rows.

- **`cuml_built_features_xgb`** (`train_cuml_xgb.py`): XGBoost (hist, CUDA, depth 6, eta 0.02, min_child_weight 20,
  subsample 0.7, colsample_bytree 0.7, colsample_bynode 0.6, lambda 4, alpha 0.02, max_bin 1024, num_parallel_tree 6,
  seed 42). Early stopping (patience 300) on a stratified 10% of outer-train, refit on all outer-train rows for 1.11x the
  best round count.
- **`cuml_rf_nested_margins`**: matrix = the 99 columns + 6 masked-language-model channels (`denv4`, `denv5`, `mfill4`,
  `qfill`, `rrd`, `cmp4`; 135 columns) + 6 lift banks (`plift`, `cbank`, `wbank`, `wbank12`, `sbank`, `ebank`; 48) = 282
  (`build_rf_matrix.py`), plus 5 nested margins fitted on it (`build_rf_margins.py`: cuML LogisticRegression(C 1)
  and SVC(rbf, C 0.5, gamma 0.002) decision functions, kNN probabilities at k = 50 / 200 / 1000; inner
  `StratifiedKFold(5, seed 777)`). Per fold the 40 columns with the highest gain of a 600-round XGBoost fitted on the
  outer-train rows are kept, then 3 cuML random forests (1000 trees, depth 20, max_features 0.3, min_samples_leaf 20,
  seeds 43 / 44 / 45) are averaged in logit space (`train_rf_margins.py`).

Test = mean of the 5 fold models (calibrated as above for the first four pairs).

## Run

Run `../density_features` and `../lift_banks` first. Files read from their caches (names in `lib/assemble.py`,
`lib/channels.py` and `build_cuml_features.py`): `static_*`, `tok_*`, `mc_*`, `impq_*`, `den_*`, `row_emb384.npy`,
`den_v4_*`, `den_v5_*`, `mlmfill4_*`, `rrd_*`, `qfill_*`, `cmp4_f<k>_*` (density_features) and `plift_f<k>_*`,
`cbank_f<k>_*`, `wbank_f<k>_*`, `wbank12_f<k>_*`, `sbank_f<k>_*`, `ebank_f<k>_*` (lift_banks).

```
python train_cuml.py cuml_svc_rbf_density
python train_cuml.py cuml_svc_rbf_density_rich
python train_cuml.py cuml_rff_logreg_density
python train_cuml.py cuml_rf_density_rich
python build_cuml_features.py      # cache/cf_f<k>_{tr,te}.parquet
python train_cuml_xgb.py
python build_rf_matrix.py          # cache/M_f<k>.npy (282 columns)
python build_rf_margins.py         # cache/margins_f<k>_{tr,te}.parquet
python train_rf_margins.py
```
Every script takes `--folds` (e.g. `--folds 2`) to run folds separately; finished folds are kept in `parts/` and each
pair is written to `preds/` once all 5 folds exist. A fold takes about 1 minute for the SVC / RFF / RF models (most of it
the matrix assembly), 30 s for the cuML stack and 4 minutes for the margins.

| Pair | Model | OOF AUC |
|---|---|---:|
| `cuml_svc_rbf_density` | cuML SVC (rbf, C 0.5, gamma 0.002), 213 columns, calibrated | 0.964832 |
| `cuml_svc_rbf_density_rich` | the same SVC on the 346-column rich matrix, calibrated | 0.965210 |
| `cuml_rff_logreg_density` | cuML logistic regression (C 3) on 4096 random Fourier features, 213 columns, calibrated | 0.965029 |
| `cuml_rf_density_rich` | cuML random forest (300 trees, depth 20), 346 columns, calibrated | 0.967049 |
| `cuml_built_features_xgb` | XGBoost on the 99 cuML-built columns | 0.968557 |
| `cuml_rf_nested_margins` | 3 cuML random forests (1000 trees) on the top 40 of 282 columns + 5 nested margins | 0.968667 |

Packages: RAPIDS cuML, cuDF and CuPy (run with cuML 26.02), xgboost (CUDA), pandas, pyarrow, scikit-learn, scipy.
