# Forensic XGBoost: nested TEs, imputed screens, original-pair evidence and a WOE base margin

GPU XGBoost models on one shared 102-column "forensic" matrix, plus the shared library `lib/` (also used by
`../catboost_forensic`, `../lgbm_margin` and `../tabm_forensic`).

**Forensic matrix** (`lib/strong.py`, built per outer fold; folds = `common.data.folds()`):
- parent (42, `lib/parent.py`): the 12 raw columns (categoricals as pandas categories), parts_sum (skip-NaN sum of
  social + gaming + work), social / gaming / work share of daily, weekend - daily, daily - strict component sum, the
  fractional cents of daily and weekend (`rint(100 x) mod 100`), the train + test count of each exact numeric value;
  nested target encodings of the 9 exact numeric values (smooth 1; inside the outer fold 3 inner StratifiedKFold(5)
  views with seeds 42 / 1042 / 2042 + fold + 1, cuML `TargetEncoder(split_method='customize')`, averaged; validation and
  test rows use the full outer-training map) and two WOE sums `sum(logit(TE) - logit(prior))` over all 9 / the 5 screen
  columns; `daily - parts_sum` and a scaled version for rows with a missing component (observed parts scaled up by the
  median component / daily shares of the complete train + test rows).
- screen imputer (10, `build_screen_imputer.py`): label-free LightGBM regressors (L2 objective; fitted per outer fold on
  up to 480k outer-training rows where the field is observed) impute daily / social / gaming / work / weekend:
  imputed value and value-if-missing.
- screen response (6): the strict-slack, partial-slack and component-share logits (q16, q64) of
  `../forensic_features`.
- imputed screen response (6): the same kind of inner-cross-fitted (label-free split `(index * 2654435761 + 8111) mod 5`)
  smoothed target logits on q16 / q64 codes of strict slack, component share and weekend - daily of the imputed fields.
- observed joint (18, `build_observed_joint.py`): for 18 selected numeric pairs (`PMI_PAIRS18` in
  `lib/pair_evidence.py`), the PMI of the exact value pair over all train + test rows, shrunk by count / (count + 5).
- exact copula (18): `log(competition pair frequency) - log(original pair frequency)` of the same pairs where the exact
  pair does not occur in the original dataset (else 0).
- unsupported pair response (2): for those pairs, the smoothed (m = 20) target logit of the exact pair, cross-fitted
  with the label-free split `(index * 2654435761 + 2718281) mod 5`; reliability-weighted mean over the available pairs
  and the available fraction.

**Base margin**: `logit(prior) + 0.40 raw_te_woe_sum + 0.10 unsupported-pair weighted mean`, passed as XGBoost's
`base_margin` for training, validation and test rows.

**xgb_forensic_woe_margin** (`train_forensic_xgb.py`): XGBoost (GPU hist) on the 102 columns with that margin:
12,000 rounds, lr 0.006875, depth 6, max_bin 2048, min_child_weight 1, subsample 0.8, colsample_bytree 0.8,
reg_alpha 0.05, reg_lambda 10, native categoricals, seed 42; no early stopping. Test = mean of the 5 fold models.

**xgb_forensic_ae_imputation** (`train_ae_imputation_xgb.py`, `lib/ae_features.py`): the same model with 25 features of a
masked neural autoencoder fitted on predictors only (from the 189-column reconstruction cache
`reconstruction-production-v1` of `../realmlp_repair`: training rows reconstructed by 3 row-excluded states,
validation / test rows by the outer-training state): the 9 numerics completed by the autoencoder, the 8 categorical
level probabilities where the categorical is missing, and 8 geometry values of the completed row (parts sum, slack,
slack share, social / gaming / work share, weekend - daily, original-rule probability). The block is inserted right
after the screen-response block (127 columns); margin, parameters and protocol as above.

**Frontier matrix** (`lib/frontier_features.py`, 290 columns): the 102 forensic columns with 188 columns inserted after
the screen-response block:
- 24 fixed-original source channels of `../forensic_features` (selected LLRs, pair LLRs / log counts, CDF geometry);
- 134 columns made by `../realmlp_repair`: exact pair-lift bank (10), coarse-lattice lift bank (9), generator
  temperature (39), global train + test density-mixture surprisal / entropy (24) and conditional fills (27),
  derived-coordinate lift bank (5), repair posteriors (20);
- the 10 completion aggregates of its global density mixture;
- 14 source-tempered residual contrasts (`build_residual_ares.py`, `lib/residual_ares.py`): for age, daily, social,
  notifications, opens and weekend, a weighted power law maps the original's per-class exact-token counts to the
  competition's; contrasts of the exact-token posteriors at alpha 5 / 80 and the tempered prior, plus their row sums;
  outer-training rows are encoded from the other 4 splitmix64(id) groups;
- 6 daily x social surfaces (`lib/residual_surfaces.py`): the residual y - sigmoid(completion logit mean) accumulated
  on the exact 0.01-hour (daily, social) grid, Gaussian-smoothed (sigma 1 / 3 / 8 ticks), complement-cross-fitted over
  splitmix64(id) mod 5 owner groups.

Base margin: `logit(prior) + 0.55 raw_te_woe_sum + 0.40 (completion_logit_mean - logit(prior))`. XGBoost (GPU hist):
lr 0.02, depth 6, max_bin 1024, min_child_weight 20, subsample 0.7, colsample_bytree 0.7, colsample_bynode 0.6,
lambda 4, 6 parallel trees per round, seed 42. Rounds (`lib/refit_xgb.py`): a selector is fitted on 9/10 of the
outer-training rows (`StratifiedKFold(10, shuffle, random_state=91027)`, first split) with early stopping (350 rounds,
cap 5,000) on the other tenth, then a refit on all outer-training rows runs `ceil(best x 10 / 9)` rounds.

**xgb_forensic_original_residual_refit** (`train_residual_refit_xgb.py`): that XGBoost on the 290-column frontier
matrix.

**xgb_forensic_periodic_rbf_regions** (`train_realmlp_rbf_xgb.py`, `lib/region_lift.py`, `lib/realmlp_embeddings.py`):
the same plus, fitted per fold on the outer-training rows, 9 strict region x exact-value lifts (region = original rule
region of the raw daily / social hours; smoothed m = 20 logit lift of the (value, region) pair over value and region,
complement-cross-fitted over splitmix64(id) mod 5 groups), 36 periodic coordinates (robust-scaled raw numerics,
2 random frequencies each from `default_rng(250314)`, sin / cos) and 144 RBF coordinates (16 quantile-placed Gaussian
bumps per raw numeric): 479 features. XGBoost interaction constraints: the 42 parent columns are a shared core that
forms one clique with each feature family (imputer, screen response, residual source channels, lift banks,
temperature, global density, fills, repair, completion, ares / dsx, imputed screen response, PMI, copula, unsupported
response) and with each new block (periodic, region lifts, RBF).

```
python ../forensic_features/build_source_channel.py
python ../forensic_features/build_screen_response.py
python build_screen_imputer.py
python build_observed_joint.py
python train_forensic_xgb.py
python train_ae_imputation_xgb.py           # needs reconstruction-production-v1 of ../realmlp_repair
python build_residual_ares.py
python train_residual_refit_xgb.py          # needs the lift / temperature / global density / completion / repair
python train_realmlp_rbf_xgb.py             # caches of ../realmlp_repair (its build_lift_banks.py, build_temperature.py,
                                            # train_global_density.py x 4, build_global_channels.py,
                                            # build_global_completion.py, build_repair.py)
```
The builders with fold-wise state and the training scripts take `--folds 0,1,2,3,4` (finished folds are kept in
`parts/`, the pair is written once all 5 folds exist); a training fold runs on one GPU.

Cache files (`cache/`), the first four are also read by other folders:

| File | Content |
|---|---|
| `screen_imputer_v2/fold_{k}_{train,test}.npy`, `manifest.json` | L1 and L2 LightGBM imputations of the 5 screen fields, fitted on fold k's outer-training rows (all train rows / test rows) |
| `observed_joint_cache_v1/{train_features,test_features}.npy`, `manifest.json` | log pair count and PMI of the 36 numeric pairs over train + test (72 columns); the manifest has `columns` and `feature_audit.complete_pair_counts` |
| `source_knn_v1/{train,test}.npy`, `manifest.json` | `build_source_knn.py`: label geometry of the 4 / 16 / 64 nearest original rows (mask-aware distance; 17 columns) |
| `source_knn_impute_v1/{train,test}.npy`, `manifest.json` | `build_source_knn.py`: inverse-distance KNN imputations of the screen fields from the original rows (15 columns) |
| `ares_production/fold_{k}/{train,valid,test}_features.npy`, `{train,valid}_indices.npy`, `manifest.json` | `build_residual_ares.py`: the 44 source-tempered response columns of fold k |

`python build_source_knn.py` (GPU, cupy) is only needed by `../cuml_models`.

| Pair | Model | OOF AUC |
|---|---|---:|
| xgb_forensic_woe_margin | XGBoost on the 102-column forensic matrix, TE-WOE + unsupported-pair base margin | 0.969415 |
| xgb_forensic_ae_imputation | the same + 25 masked-autoencoder fill features (127 columns) | 0.969421 |
| xgb_forensic_original_residual_refit | XGBoost on the 290-column frontier matrix, completion margin, selector + 10/9 refit, 6 parallel trees | 0.969973 |
| xgb_forensic_periodic_rbf_regions | the frontier XGBoost + region lifts, periodic and RBF coordinates (479 features), semantic interaction constraints | 0.970128 |

Requirements: xgboost (CUDA), cuML (TargetEncoder), lightgbm, cupy (source KNN), scipy, scikit-learn, pandas, numpy.
