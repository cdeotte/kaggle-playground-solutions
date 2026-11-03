# Forensic CatBoost: forensic matrix + source-fragment categories with a native baseline

GPU CatBoost models on the 102-column forensic matrix of `../xgb_forensic` (as float32 `strong__` columns), fixed-original
channels of `../forensic_features` and CatBoost-native categories of a label-free source-fragment cache, plus the
shared library `lib/strong_cat.py` (also used by `../lgbm_margin`).

**Source-fragment cache** (`build_source_fragments.py`, all train + test rows; fitted from the original's predictors /
labels and the unlabeled train + test predictors). Values are tokenized exactly (age, notifications, opens x1;
hours x100). Categories: raw categoricals; 5 missing-pattern bitmasks and the missing count; the exact value tokens;
the original rule region (daily <= 6 and social <= 4 / daily > 8 or social > 4) and the screen allocation state; per
numeric the nearest original value and the clipped tick distance to it; per numeric pair a recurrence id (exact pair
recurring >= 2 / >= 10 times in train + test) and the matching original row (0 missing, 1 not in the original,
2 ambiguous, else 3 + the unique original row); the consensus original row of the pair votes. Numeric companions:
raw values and observed flags, original target rates of the categorical levels and of the exact / nearest values,
distances to the nearest original value, 20 screen-budget coordinates, pair-recurrence summaries.

**catboost_forensic_fragments** (`train_forensic_catboost.py`): features = the 102 forensic columns, the 114 univariate
LLR / log-count source channels, native categories raw_cat / mask / exact_value / semantic_state, and semantic
interaction categories (original q16 / q64 codes of each numeric, q16 x q16 codes of the 18 pairs of
`PMI_PAIRS18`, original q16 codes of the l2-imputed screen fields and their 10 pairs, the original rule region of the raw
and of the imputed daily / social values). The forensic base margin (`logit(prior) + 0.40 raw_te_woe_sum + 0.10
unsupported-pair response`) is CatBoost's `baseline` (not a feature). CatBoost (GPU): 24,000 iterations, depth 6,
lr 0.01, l2_leaf_reg 30, random_strength 0.3, Bernoulli subsample 0.9, border_count 254, one_hot_max_size 16,
max_ctr_complexity 2, score_function NewtonL2, model_size_reg 0, seed 42 + fold; no early stopping. Test = mean of
the 5 fold models.

**catboost_forensic_reconstruction** (`train_reconstruction_catboost.py`, `lib/recon_features.py`): the same matrix plus
predictor-reconstruction evidence made by `../realmlp_repair`:
- 71 numeric reconstruction columns (the first 71 columns of the 189-column reconstruction cache
  `reconstruction-production-v1`: masked-autoencoder / XGBoost completions of every predictor, fitted on predictors
  only, with uncertainties and residuals);
- 10 density-completion aggregates (the `__aggregate__` columns of the per-fold cache `fold_completion`);
- 30 reconstruction categories: missing-pattern codes (all 12 fields, 5 screen fields, count), per numeric the q32 code
  of the hybrid completion and the q8 code of the XGBoost / autoencoder disagreement where missing (edges from the
  outer-training rows), per categorical the completed level, the predicted level where missing and the XGBoost /
  autoencoder agreement.

120 numeric columns are then pruned (`DROP_FEATURE_NAMES`), leaving 280 features (103 categorical). Same baseline and
CatBoost parameters as above except max_ctr_complexity 3.

```
python ../forensic_features/build_source_channel.py
python ../forensic_features/build_screen_response.py
python ../xgb_forensic/build_screen_imputer.py
python ../xgb_forensic/build_observed_joint.py
python build_source_fragments.py
python train_forensic_catboost.py
python train_reconstruction_catboost.py      # after ../realmlp_repair has built its reconstruction and
                                             # density-completion caches (see lib/recon_features.py)
```
The training scripts take `--folds 0,1,2,3,4`; finished folds are kept in `parts/` and the pair is written once all 5
folds exist.

Cache: `cache/cat_source_fragments_v1/{numeric.npy, categorical.npy, manifest.json}` (train rows, then test rows; the
manifest lists `numeric_names`, `categorical_names`, `groups`, `rows`, `train_rows`). `../lgbm_margin` reads it.

| Pair | Model | OOF AUC |
|---|---|---:|
| catboost_forensic_fragments | CatBoost on forensic matrix + source channels + fragment / semantic categories, forensic baseline | 0.969769 |
| catboost_forensic_reconstruction | the same + reconstruction / density-completion evidence, 120 columns pruned, max_ctr_complexity 3 | 0.969941 |

Requirements: catboost (GPU), cuML (TargetEncoder), lightgbm (screen imputer of `../xgb_forensic`), scikit-learn,
scipy, pandas, numpy.
