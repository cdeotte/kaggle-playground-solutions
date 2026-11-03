# CatBoost on ordered-CTR value tokens, or on the numeric channel matrix with a WOE baseline

Four GPU CatBoost models (`train.py`), two input paths.

**tokens path** (three models). `build_cat_features.py` and `build_fills.py` build fold-independent, target-free inputs:
- 27 categorical string columns that CatBoost encodes with its own ordered CTRs (no hand-made target encoding): the exact
  `%.2f` tokens of the 12 columns, 0.1-rounded tokens of the 6 hour columns, the (daily, social) region (0 / 1 / 2 /
  undetermined), pair tokens daily x social, daily x weekend, gaming x work (0.1) and notifications x app opens (exact),
  and region x notifications / app opens / age / gender;
- 36 numerics: the 9 raw numerics, 3 categorical codes, 12 missing flags, missing count, budget geometry
  (`sgw`, `slack`, `slack / daily`, the 3 shares of daily, weekend - daily, weekend - sgw, daily - 6, daily - 8, social - 4);
  `build_cat_features.py` also writes original-data channels (kernel-smoothed original LLR, class-conditional original CDF
  gap, frequency ratio, counts) that the models leave out;
- `fill`: XGBoost fills of the 5 screen columns from the other 11 (fit on the observed rows of train + test), geometry on
  the fills and a 3-class region posterior (12 columns).
The MLM blocks of `../density_features` are added as numerics: `mlmfill`, `mlmfill2`, `rrd` (reconstruction residuals of
the observed cells), `cmp3_f*`, `bp2_f*` (budget-mixture posteriors) for `catboost_tokens_mlm3`; `mlmfill4`, `cmp4_f*` (the
completion block of the v4 MLM), `den_v4`, `rrd`, `bp2_f*` for `catboost_tokens_mlm4`.
Parameters: lr 0.02, depth 6, l2_leaf_reg 6, max_ctr_complexity 4, one_hot_max_size 1, random_strength 2.9,
bagging_temperature 0.26, border_count 254, seed 42.

**channels path** (`catboost_mlm4_banks_baseline`): the numeric per-fold matrix of `../xgb_density_margin/lib` (no
categorical features; 348 columns): base + counts, original channels, PMIs, imputers, decimal lattice, nested TEs of the
12 tokens / 3 pairs / 5 region keys / 8 geometry bins (smoothing 20, averaged over 3 inner seeds) and a second TE
(smoothing 4), cmp4, den_v4, den_v5, mlmfill4, qfill, smooth fields, lift banks plift, cbank, wbank, wbank12, sbank and
ebank (weekend / daily / social at a 0.5 lattice x notifications / app opens). The WOE margin (scale 0.4) + completion
margin (0.4) of the XGBoost models is CatBoost's `baseline` in every pool. Parameters: lr 0.005 (up to 100,000
iterations; 39-49k selected), depth 6, l2_leaf_reg 120, Bayesian bootstrap with bagging_temperature 0.26,
random_strength 2.9, auto_class_weights Balanced, border_count 254, seed 42. (The source run's flag list also named the
XGBoost-only blocks ares, dsx and tokemb, but its runner had no code for them, so they are not in this matrix.)

**Protocol** (both paths): early stopping (AUC; od_wait 700 / 700 / 400 / 1200) on a stratified 10% split of the
outer-train rows (seed 99), then a refit on all outer-train rows at 1.11x the best iteration count. One model per fold;
test = mean of the 5 fold models (probabilities).

```
python build_cat_features.py     # cache/cat_*.parquet, cache/num_*.parquet
python build_fills.py            # cache/fill_*.parquet (GPU)
python train.py catboost_tokens_fills
python train.py catboost_tokens_mlm3
python train.py catboost_tokens_mlm4
python train.py catboost_mlm4_banks_baseline
```
Run `../density_features` and `../lift_banks` first; `train.py` imports `../xgb_density_margin/lib` (keep the folders side
by side). One fold takes 15-55 minutes on one GPU; `--folds 0,1` runs a subset, finished folds are kept in `parts/` and
`preds/<pair>_{oof,test}.npy` is written when the last fold is done.

| Pair | Model | OOF AUC |
|---|---|---:|
| catboost_tokens_fills | 27 CTR categoricals + 36 numerics + fills (75 columns), od_wait 700 | 0.969002 |
| catboost_tokens_mlm3 | + mlmfill, mlmfill2, rrd, cmp3, bp2 (185 columns) | 0.969088 |
| catboost_tokens_mlm4 | + mlmfill4, cmp4, den_v4, rrd, bp2 (196 columns), od_wait 400 | 0.969351 |
| catboost_mlm4_banks_baseline | numeric channel matrix with all lift banks (348 columns), WOE + completion baseline, lr 0.005, l2 120, Balanced class weights | 0.969802 |

Requirements: catboost (GPU), xgboost (CUDA, for `build_fills.py`), pandas, pyarrow, numpy, scipy, scikit-learn.
