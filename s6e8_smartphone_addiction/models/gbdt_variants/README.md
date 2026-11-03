# GBDT variants: LightGBM, XGBoost and CatBoost with their own feature recipes

Three gradient-boosted models built only from the competition data and the original dataset (no other repo folder).
All use our 5 folds; test = mean of the 5 fold models.

**`lgbm_exact_te36`** (`train_lgbm.py`, `lib/features.py`, 86 columns)
- 9 raw numerics (NaN kept) and 23 derived columns that propagate NaN (no imputation): slack = daily - (social + gaming +
  work), comp_sum, utilization = comp_sum / daily, their cents ((x 100).round() % 100), exact-budget flag, sleep + daily
  and its == 20 flag, social / gaming / work / notifications / app opens per daily hour, weekend - daily, weekend / daily,
  notifications x app opens, social x gaming, gaming x work, a repair flag (0.85 <= utilization <= 0.95) and the 3
  categoricals as ordinal codes;
- 18 original-dataset rates per exact value ('%.2f' tokens) of the 9 numerics: label rate and rate of addiction level
  'Severe';
- 36 nested exact-value target encodings: the 12 columns (smoothing 30), slack / comp_sum / sleep + daily (30),
  utilization (50), daily / weekend / social / notifications / app opens again at smoothing 10 and 200, and 10 token pairs
  (100). Outer-train rows are encoded by 5 inner folds (StratifiedKFold(5, shuffle=True, random_state=777)), validation
  and test rows by the map of all outer-train rows.
- LightGBM (CUDA): lr 0.02, 255 leaves, min_data_in_leaf 20, feature / bagging fraction 0.8 (bagging_freq 1), L1 0.1,
  L2 1, max_bin 255, seed 2026. Rounds: early stopping (patience 100, at most 5000) on one third of the outer-train rows
  (first split of StratifiedKFold(3, shuffle=True, random_state=2026 + fold)), trained on the other two thirds; then a
  refit on all outer-train rows for that many rounds.

**`xgb_dual_smooth_te`** (`train_xgb.py`, `lib/xgb_features.py`, 73 columns)
- 49 base columns: 9 raw numerics (NaN kept), 8 geometry columns (comp_sum, slack, utilization, slack / daily,
  sleep + daily, 20 - sleep - daily, weekend / daily, social / daily), 5 conditional-median imputations computed on
  train + test (daily | social, gaming; social | daily, gaming; weekend | daily, social; notifications | app opens;
  app opens | notifications; fallback to the first key, then the global median), 2 block observation counts,
  11 one-hot categoricals (NA = own column), 14 original-dataset per-token label rates (smoothing 3): 9 numerics,
  3 categoricals, notifications x app opens, daily x weekend;
- 24 nested target encodings of the exact-value tokens of 8 numerics (age excluded) and 4 pairs (notifications x app
  opens, daily x weekend, sleep x daily, daily x social), each at smoothing 20 and 2. Outer-train rows are encoded by
  5 inner folds (StratifiedKFold(5, shuffle=True, random_state=2042 + fold)), validation and test rows by the full map.
- XGBoost (hist, CUDA): lr 0.02, max_depth 8, min_child_weight 60, subsample 0.8, colsample_bytree 0.8, lambda 2,
  max_bin 1023, num_parallel_tree 5, seed 42. Rounds = round(mean best iteration) of 5 inner models on the outer-train
  rows (StratifiedKFold(5, shuffle=True, random_state=2042 + fold), early stopping patience 100, at most 8000), then a
  refit on all outer-train rows.

**`catboost_quantile_bins`** (`train_catboost.py`, `lib/cat_features.py`, 25 columns, no target encodings)
- 9 quantile-binned numerics and 3 binned geometry columns (slack, utilization, weekend / daily): 16 bins with edges
  from the outer-train rows, missing values stay NaN; 5 mask / slice columns (missing screen columns, missing numerics,
  6 < daily <= 8 and social <= 4, daily missing, >= 3 screen columns missing); 5 conditional-median imputations (2-level
  fallback, fitted on the outer-train rows); gender, stress_level and academic_work_impact as native CatBoost
  categoricals (NA -> 'NA').
- CatBoost (one GPU): Logloss, depth 8, lr 0.03, l2_leaf_reg 3, Bernoulli bootstrap subsample 0.8, seed 42.
  Rounds = round(mean best iteration) of 5 inner models (StratifiedKFold(5, shuffle=True, random_state=3042 + fold),
  early stopping patience 200, at most 9000), then a refit on all outer-train rows.

```
python train_lgbm.py        # -> preds/lgbm_exact_te36_{oof,test}.npy
python train_xgb.py         # -> preds/xgb_dual_smooth_te_{oof,test}.npy      (about 9 minutes per fold)
python train_catboost.py    # -> preds/catboost_quantile_bins_{oof,test}.npy  (about 13 minutes per fold)
```
Each script takes `--folds` (e.g. `--folds 0,1`); finished folds are kept in `parts/`. No other repo folders are needed.

| Pair | Model | OOF AUC |
|---|---|---:|
| `lgbm_exact_te36` | LightGBM, raw + 23 derived + 18 original-data rates + 36 exact-value target encodings | 0.967857 |
| `xgb_dual_smooth_te` | XGBoost, 49 base columns + 24 exact-value target encodings at smoothing 20 and 2 | 0.968152 |
| `catboost_quantile_bins` | CatBoost on quantile bins, conditional-median imputations and native categoricals | 0.960246 |

Packages: lightgbm (CUDA build), xgboost (CUDA; the run used 3.0.0), catboost (GPU; 1.2.10), pandas, scikit-learn.
