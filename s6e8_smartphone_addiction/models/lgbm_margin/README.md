# LightGBM on the forensic fold cache with an analytic base margin

Two CPU LightGBM models on one per-fold feature cache. Their boosting starts from the forensic base margin (passed as
LightGBM `init_score`), so the trees learn a correction to it.

**Fold cache** (`build_fold_features.py`, layout in `lib/fold_cache.py`), built separately for every outer fold:
- **strong** (104): the 102-column forensic matrix of `../xgb_forensic` (parent columns, nested target encodings with
  WOE sums, l2-imputed screen fields, screen-response logits, imputed screen response, shrunk pair PMIs, original
  copula density ratios of unsupported pairs, unsupported-pair response) as float32, plus the base margin
  `logit(prior) + 0.40 raw_te_woe_sum + 0.10 unsupported-pair weighted mean` and its probability. The nested target
  encodings use a CPU exact-key encoder (`lib/target_encoder.py`, same arithmetic as cuML's `TargetEncoder` with
  custom inner folds) instead of cuML.
- **fixed_source** (114): the fixed-original univariate binary LLRs and log counts of `../forensic_features`.
- **cat_numeric** (115): the numeric columns of the label-free source-fragment cache of `../catboost_forensic`.
- **categories** (129): the source-fragment categories (raw categoricals, missing masks, exact values, original rule
  region / allocation state, recurrence and original-pair-row ids of the 18 PMI pairs, nearest original value, tick
  residual, consensus original row) and 53 semantic interaction categories (original q16 / q64 codes, q16 pair codes,
  imputed-screen q16 codes and pairs, original rule region), recoded over the fold's outer-training values; plus each
  code's log1p outer-training count and 6 per-row summaries of those counts.

**Models** (`train_lgbm.py`): features = strong + fixed_source numerics (margin and probability columns removed);
`lgbm_forensic_margin_frequency` adds the log-frequencies of the native base and semantic categories and the 6
summaries (295 features), `lgbm_forensic_margin_wide` uses the numerics only (216 features). `init_score` = the
float32 base margin column, `boost_from_average=False`; prediction = `sigmoid(clip(tree raw score + init_score, -50,
50))`. Common parameters: lr 0.02, max_bin 511, feature_fraction 0.85, bagging 0.9 every iteration, lambda_l1 0.05,
deterministic CPU training with 12 threads, seeds 42 / 143 / 253 / 349 / 443 + fold (seed / bagging / feature
fraction / data / extra); fixed number of rounds, no early stopping. Test = mean of the 5 fold models.

```
python build_fold_features.py            # needs ../xgb_forensic, ../catboost_forensic and ../forensic_features caches
python train_lgbm.py lgbm_forensic_margin_frequency
python train_lgbm.py lgbm_forensic_margin_wide
```
Run first: `../forensic_features` (both builders), `../xgb_forensic` (`build_screen_imputer.py`,
`build_observed_joint.py`) and `../catboost_forensic/build_source_fragments.py`. Both scripts take `--folds 0,1,2,3,4`.

The fold cache is written to `cache/features/fold_<k>/` (`{train,valid,test}_{numeric,categories,category_frequency,
category_summary}.npy`, `{train,valid}_indices.npy`, `manifest.json` with `base_numeric_names` and the column groups;
train = outer-training rows, valid = the rows of the fold, test = all test rows). It is also read by
`../tabm_forensic`, `../realmlp_repair`, `../structural_nets` and `../pairlift_models`.

| Pair | Model | OOF AUC |
|---|---|---:|
| lgbm_forensic_margin_frequency | 63 leaves, depth 6, min leaf 500, lambda_l2 24, 4,400 rounds; numerics + category log-frequencies + summaries | 0.969471 |
| lgbm_forensic_margin_wide | 127 leaves, unlimited depth, min leaf 350, lambda_l2 32, 1,800 rounds; numerics only | 0.969398 |

Requirements: lightgbm, numpy, pandas, scikit-learn, scipy (CPU only; the fold cache uses the CPU target encoder, so
cuML is not needed here).
