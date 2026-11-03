# XGBoost with a WOE base margin on nested target encodings, MLM density channels and lift banks

Eight GPU XGBoost models on one growing per-fold channel matrix, plus the library `lib/` that builds it (also used by
`../catboost_density` and `../dcn`, so keep the folders side by side).

**Channel matrix** (`lib/assemble.py`, `lib/channels.py`; every label-derived channel is cross-fitted inside the outer-train
rows with 5 inner folds, seed 777, and the outer-validation / test rows get the maps of all outer-train rows):
- **base** (static, target-free; `static_*` of `../density_features`): the 12 raw columns (categoricals as fixed codes),
  12 missing flags, missing counts, budget geometry (`sgw = social + gaming + work`, `slack = daily - sgw`, shares of daily,
  weekend - daily, weekend - sgw, distances to the region borders daily 6 / 8 and social 4, the (daily, social) region),
  exact-value counts (`cnt`), original-data channels (`orig`: smoothed original LLR of the exact and 0.1-rounded values,
  synthetic / original frequency log ratio), 18 observed-only pair PMIs (`pmi`), predictor-only XGBoost screen imputers,
  the filled geometry and a 3-class region posterior (`imp`), fractional part and first decimal of the 6 hour columns
  (`lat`).
- **nested TEs** (`te`) of the 12 exact `%.2f` tokens, of daily|social and daily|weekend at 0.1 and notifications|app opens
  (exact), of region x token for notifications, app opens, age, sleep, gender (`rte`) and of 8 quantile bins of geometry /
  imputer channels (`geo`); a second TE of the 12 tokens (`te2s`, `tes_` columns). Smoothing: `te` 20 and `tes` 4 in the
  first model; from the second model on `te` 1 averaged over 3 inner seeds (777, 1778, 2779; `teavg`) and `tes` 20.
- **base margin** (`mrg`): `logit(prior) + scale * sum of the 12 single-token TE WOEs` (scale 0.5, later 0.55), plus 0.4 x the
  sum of the per-screen completion logits minus `logit(prior)` of the missing screen cells (`cmpmrg`). It is XGBoost's
  `base_margin` in every fit and prediction, so the trees learn a correction to it.
- **MLM channels** (`../density_features`; a masked transformer over the 12 exact-value tokens of train + test, no labels):
  completion response (`cmp3_f*` from the d384 MLM, later `cmp4_f*` from the v4 MLM), per-cell log p and entropy
  (`den384v3`, later `den_v4` and `den_v5`), expected-value fills with recomputed geometry (`mlmfill`, `mlmfill2`, later
  `mlmfill4`), posterior width / skew of the screen cells (`qfill`), 3 PCA components of each column's token embedding
  (`tokemb`).
- **lift banks** (`../lift_banks`, cross-fitted per fold): pair response shrunk toward the log-odds independence base minus
  that base - `plift` (9 exact pairs + mean), `cbank` (coarse-lattice pairs, pair smoothing 1), `wbank` (utilization /
  weekend - daily pairs), `wbank12` (wbank at pair smoothings 0.5 / 1 / 2), `sbank` (notifications / app opens x social
  share, social x notifications); `dsx`: exact-grid (daily, social) residual surfaces at sigma 1 / 3 / 8.
- **sf** (`lib/smooth_fields.py`): Gaussian-blurred 1-D response fields of the 9 numerics and 2-D fields of (daily, social),
  (daily, weekend), (notifications, app opens), as logits; **ares**: `logit(te) - logit(tes)` per column,
  `logit(te) - sf1` per numeric and both row sums.

**Model** (`train.py`): `binary:logistic`, hist on GPU, depth 6, min_child_weight 20, lambda 4, alpha 0.02, max_bin 1024,
subsample 0.7, colsample_bytree 0.8 (0.7 from the third model on), seed 42. Rounds: early stopping (AUC, patience 600) on a
stratified 10% split of the outer-train rows (seed 99), then a refit on all outer-train rows at 1.11x the best iteration.
From the third model on, the refit model's gain importances drop the 120 weakest columns and the model is refit once more
on the rest. One model per fold; test = mean of the 5 fold models (probabilities).

```
python train.py xgb_te_woe_margin
python train.py xgb_te_mlm3_margin
python train.py xgb_te_mlm3_sf_pruned
python train.py xgb_te_mlm3_plift
python train.py xgb_te_mlm3_plift_cbank
python train.py xgb_te_mlm4_plift_cbank
python train.py xgb_te_mlm4_qfill_wbank
python train.py xgb_te_mlm4_qfill_wbank12_sbank
```
Run `../density_features` and `../lift_banks` first (this folder reads their `cache/` files under the names listed in
`lib/channels.py`; `cmp4_f{k}_*.parquet` is the completion block of the v4 MLM). One fold takes 5-35 minutes on one GPU;
`--folds 0,1` runs a subset (e.g. one process per GPU with `CUDA_VISIBLE_DEVICES`), finished folds are kept in `parts/` and
`preds/<pair>_{oof,test}.npy` is written when the last fold is done.

| Pair | Model | OOF AUC |
|---|---|---:|
| xgb_te_woe_margin | base + cnt, orig, pmi, imp, lat + TEs (s 20 / tes s 4) with rte and geo keys; WOE margin 0.5; lr 0.015, 3 parallel trees per round; 172 columns | 0.969484 |
| xgb_te_mlm3_margin | + TE s 1 x 3 inner seeds, tes s 20, cmp3, den384v3, mlmfill, mlmfill2, completion margin; lr 0.02, 6 parallel trees, colsample_bynode 0.6; 263 columns | 0.969753 |
| xgb_te_mlm3_sf_pruned | + sf, ares, tokemb, dsx; colsample_bytree 0.7, margin 0.55; gain pruning 337 -> 217 columns | 0.969987 |
| xgb_te_mlm3_plift | + plift; 347 -> 227 columns | 0.970011 |
| xgb_te_mlm3_plift_cbank | + cbank; 356 -> 236 columns | 0.970055 |
| xgb_te_mlm4_plift_cbank | v4 MLM blocks (cmp4, den_v4, mlmfill4) instead of cmp3, den384v3, mlmfill, mlmfill2; 341 -> 221 columns | 0.970137 |
| xgb_te_mlm4_qfill_wbank | + den_v5, qfill, wbank; 386 -> 266 columns | 0.970186 |
| xgb_te_mlm4_qfill_wbank12_sbank | + wbank12, sbank; 404 -> 284 columns | 0.970203 |

Requirements: xgboost (CUDA), pandas, pyarrow, numpy, scipy, scikit-learn.
