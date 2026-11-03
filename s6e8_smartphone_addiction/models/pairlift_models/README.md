# Exact-pair lift bank on four forensic learners

One fixed 10-column feature bank appended to four strong single-model recipes (TabM, CatBoost, LightGBM, XGBoost). Every
other feature, hyper-parameter and seed of each recipe is unchanged.

**Pair-lift bank** (`lib/pairlift.py`, `build_pairlift.py`): every value is tokenized to its exact two-decimal string
(missing = `__NA__`). For a pair of columns (A, B) the marginal rates P(y=1|A), P(y=1|B) are shrunk to the prior
(smoothing 20) and combined under log-odds independence, `baseline = sigmoid(logit P(y|A) + logit P(y|B) - logit P(y))`;
the feature is the shrunk residual of the exact pair, `(pair_positives - pair_count * baseline) / (pair_count + s)`
(s = 20; 100 for notifications x app opens), so an unseen pair is exactly 0. Nine pairs: app opens x weekend, daily x app
opens, notifications x weekend, daily x weekend, daily x notifications, daily x gaming, notifications x app opens,
sleep x app opens, social x sleep; plus their row mean. Inside each outer fold the outer-training rows are encoded with a
five-way inner cross-fit (`StratifiedKFold(5, shuffle=True, random_state=777)`); outer-validation and test rows use the
state fitted on all outer-training rows. The 10 columns are appended right after each parent's own columns.

**Learners** (one model per outer fold, no early stopping on the outer-validation fold, test = mean of the 5 fold models).
The forensic base margin is `logit(prior) + 0.40 raw_te_woe_sum + 0.10 unsupported-pair response`.
- `train_lgbm.py` - CPU LightGBM on the shared forensic fold cache of `../lgbm_margin`: the 104 `strong__*` forensic
  columns and the 114 `fixed_source__*` fixed-original likelihood channels (margin columns removed), log1p outer-training
  counts of the 20 native and 53 semantic categories and 6 count summaries (305 features with the pair lift). The
  `strong__selected_base_margin` column is the init score (boost_from_average off); 4,400 rounds, lr 0.02, 63 leaves,
  depth 6, min_data_in_leaf 500, max_bin 511, feature_fraction 0.85, bagging 0.9 every round, lambda_l1 0.05, lambda_l2
  24, seeds 42 / 143 / 253 / 349 / 443 + fold, deterministic.
- `train_xgb.py` - GPU XGBoost on the 102-column forensic matrix of `../xgb_forensic` (native categoricals), the
  25-column masked-autoencoder fill view (completed numerics, categorical probabilities if missing, completed screen
  geometry; row-excluded cross-fit on the outer-training rows) and the pair lift (137 features); base margin in fit and
  predict; 12,000 rounds, lr 0.006875, depth 6, max_bin 2048, subsample / colsample 0.8, alpha 0.05, lambda 10,
  max_cat_threshold 64, random_state 42.
- `train_catboost.py` - GPU CatBoost on the forensic CatBoost parent of `../catboost_forensic` (forensic columns as
  float32, 114 fixed-original channels, 20 native source-fragment categories, 53 semantic interaction categories), the
  pair lift, the compact 71-column cross-fitted reconstruction view (XGBoost / masked-autoencoder fills if missing,
  disagreement, uncertainty, source LLR, error aggregates, completed screen geometry), the 10 density-completion
  aggregates and 30 reconstruction categories (`lib/reconstruction_views.py`: missing-pattern codes, 32-quantile bins of
  the hybrid completions, 8-quantile disagreement bins, completed / predicted / agreement levels of the categoricals);
  the margin columns and 120 listed columns are dropped (290 features, 103 categorical). Baseline = base margin. The
  pair-lift columns get CatBoost feature weight 0.1 (with weight 1 they overfit this learner). 24,000 trees, depth 6,
  lr 0.01, l2 30, random strength 0.3, Bernoulli subsample 0.9, 254 borders, one-hot up to 16, max CTR complexity 3,
  NewtonL2, random_seed 42 + fold.
- `train_tabm.py` - pytabkit TabM_D (k = 32) on row-local TabM blocks of the raw row and of the neurally completed row
  (`lib/tabm_features.py`: raw, exact-value tokens, 14 smooth latent coordinates, missing-gated equal-slack completion,
  fixed bins; missing values completed from the hybrid reconstruction), 12 missing flags, the 104 `strong__*` columns,
  the compact reconstruction view, the 30 density-completion responses, the pair lift and 12 fold-standardized pair
  contrasts (354 features, 58 categorical; numeric NaN -> outer-training median). d_block 512, 2 blocks, PWL embeddings
  (48 bins, d 16), quantile_tabr, dropout 0.1, lr 0.002, batch 1024, 40 epochs with patience 8 on 1 - AUC; ten members
  (`n_cv = 10`), each early-stopping on its own 2.5% stratified holdout of the outer-training rows (seed 20260824 + fold);
  random_state 142 + fold.

```
python build_pairlift.py
python train_lgbm.py
python train_xgb.py
python train_catboost.py
python train_tabm.py
```
Every script takes `--folds` (e.g. `--folds 0,1`), keeps finished folds in `parts/` and writes
`preds/<pair>_{oof,test}.npy` (probabilities) once all five folds are done.

Run first (caches located through `lib/paths.py`):
- `models/lgbm_margin` - its fold feature builder (`cache/fold_<k>/` with `manifest.json`) - LightGBM, TabM;
- `models/xgb_forensic` - its feature caches (the forensic matrix is rebuilt per fold by its `lib/strong.py`) -
  XGBoost, CatBoost;
- `models/catboost_forensic` and `models/forensic_features` - source-fragment categories and fixed-original channels
  (read by `catboost_forensic/lib/strong_cat.py`) - CatBoost;
- `models/realmlp_repair` - the cross-fitted reconstruction cache (`cache/reconstruction-production-v1/`, all 189
  columns; the XGBoost fill view is 25 of them) and the density-completion responses
  (`cache/density_completion_response_production_features/`) - XGBoost, CatBoost, TabM.

| Pair | Model | OOF AUC |
|---|---|---:|
| pairlift_xgb | XGBoost on the forensic matrix + autoencoder fill view + pair lift, base margin | 0.969458 |
| pairlift_lgbm | LightGBM on the forensic core + fixed-original channels + category counts + pair lift, init-score margin | 0.969516 |
| pairlift_catboost | CatBoost on the reconstruction parent + pair lift (feature weight 0.1), baseline margin | 0.969954 |
| pairlift_tabm | TabM (10 inner members) on completed-row blocks + forensic core + reconstruction + pair lift | 0.970011 |

Requirements: numpy, pandas, scikit-learn, lightgbm, xgboost (CUDA), catboost (GPU), pytabkit with torch (CUDA), cuML
(target encoder of the forensic matrix).
