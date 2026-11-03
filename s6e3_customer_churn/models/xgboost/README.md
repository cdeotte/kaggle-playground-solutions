# XGBoost

Ten GPU XGBoost models in nine training scripts (`train_charges_anchor_te.py` holds 2 presets), plus
`build_tabpfn_original.py`, which prepares the input of `train_tabpfn_original_feature.py`. Every script reads `data/`
through `common/data.py`, cross-validates on the shared 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`,
identical to `common.data.folds`) and writes `preds/<pair>_{oof,test}.npy` (probabilities). Each script runs all of its
folds in one process. Full feature lists and parameters are in each docstring.

Only the competition rows are training rows; the 7,043 original IBM rows ("original data") feed label statistics,
nearest-neighbour lookups and the TabPFN feature. "Snap": a MonthlyCharges / TotalCharges value that does not occur in
the original data replaced by the closest original value (from `../snap_features`). "Nested target encoding": per outer
fold, out-of-fold over an inner split of the fold's training rows for the training rows, and fitted on all training
rows for the validation / test rows. Unless stated otherwise a model early-stops on the AUC of the validation fold.

**Original-data lookups**
- `train_original_knn_distances.py`: the 19 raw columns (strings ordinal-encoded) plus 4 cKDTree lookups into the
  original rows in the standardized (tenure, MonthlyCharges, TotalCharges) space: mean churn label of the 5 nearest
  original rows, distance to the nearest original churner and non-churner, and their ratio; depth 6, lr 0.03, 500
  rounds without early stopping.
- `build_tabpfn_original.py` + `train_tabpfn_original_feature.py`: a TabPFN classifier (32 estimators) is fitted on 4
  of 5 folds of the 7,032 original rows with a TotalCharges value and predicts every train / test row; the mean of the
  5 predictions (`cache/tabpfn_original_pred.npy`) joins the 19 raw columns (strings as native categoricals) of a
  depth-3 model, lr 0.1, early stopping 200.

**Snap features** (run `../snap_features/build.py` first)
- `train_snap_radix.py`: stumps (depth 1, lr 0.1, max_bin 16,000, early stopping 300) on 81 columns: native
  categoricals, numerics, decimal strings of the charges, snaps, snap - value, the original churn rate of each snapped
  value, and 48 radix keys round(100 x snap) + 100,000 x (category code) and tenure + 100,000 x (category code) for
  every categorical. Reads `combo_b.csv`.
- `train_feature_union.py`: a label-free union built on train + test + original rows (snaps, frequency / rank and
  log / sqrt / inverse transforms, charge relations, service counts and Yes / No flags, tenure and rank-quantile bins,
  6 category pairs and a triple, roundings, quantile bins, digits, counts and rare flags, original churn rates / counts /
  seen flags of 10 columns and 4 snap x category pairs, train-vs-original drift ratios) plus nested sklearn
  `TargetEncoder` encodings of 18 columns; depth 5, lr 0.025, colsample_bytree 0.45, max_bin 16,000. Reads
  `combo.csv`.

**Nested target statistics**
- `train_ngram_te.py`: numeric frequencies, 3 charge relations, service counts, the original churn rate of every value
  of the 19 columns, TotalCharges ranks / z-scores / quantile distances against the original churners and
  non-churners, 35 digit features; per fold (inner 5 folds) target std / min / max and `TargetEncoder` means of the 16
  categoricals and the 3 numerics as strings, and target means of 15 bigrams and 4 trigrams of 6 key categoricals;
  depth 5, lr 0.0063, colsample_bytree 0.32, alpha 3.5, gamma 0.79.
- `train_ngram_te_union.py`: the features of `train_ngram_te.py` widened with digit positions, counts, original
  counts and churn rates of the numerics, drift / original churn rate / original count of factorized categoricals,
  numerics, roundings and 2-digit fractions, numeric means per categorical; nested `TargetEncoder` means of 58 string
  keys (n-grams, numerics as strings, round / integer / 5,000-quantile / equal-width / log bins, tenure and charge bins,
  category pairs, decimal-digit pairs) and target std / min / max of the categoricals and numerics; the 16
  categoricals stay native categories. Same model parameters as `train_ngram_te.py`.
- `train_te_stats_original_proba.py`: numerics, 2 charge relations, numeric frequencies, the original churn rate of
  every value of the 19 columns; per fold (inner 10 folds) target std / min / max and `TargetEncoder` means of the 16
  categoricals and the 3 numerics as strings; depth 6, lr 0.03.
- `train_charges_anchor_te.py` (2 presets): all raw columns (strings as integer codes), category x anchor keys,
  6 anchor digits and digit pairs, the 105 category pairs; per fold (inner KFold(5)) target mean / std / count /
  nunique / median / min / max / skew and 8 quantiles per anchor key, target means of the category x anchor keys, digit
  pairs and category pairs; categories and digits as XGBoost categoricals, depth 6, colsample_bynode 0.3, lr 0.01.
  The anchor is MonthlyCharges (`xgb_monthly_anchor_te`) or TotalCharges with the statistics keyed by its value
  rounded to an integer (`xgb_total_round0_anchor_te`).

**Ranking objective**
- `train_pairwise_ranker.py`: the 19 raw columns (strings as integer codes); `XGBRanker` with `rank:pairwise` on
  query groups of 256 consecutive training rows (depth 6, lr 0.03, early stopping 200), its scores mapped to
  probabilities by a logistic regression (Platt scaling) fitted on the fold's training rows.

```
python train_original_knn_distances.py
python build_tabpfn_original.py
python train_tabpfn_original_feature.py
python train_snap_radix.py
python train_feature_union.py
python train_ngram_te.py
python train_ngram_te_union.py
python train_te_stats_original_proba.py
python train_charges_anchor_te.py xgb_monthly_anchor_te
python train_charges_anchor_te.py xgb_total_round0_anchor_te
python train_pairwise_ranker.py
```
Run `../snap_features/build.py` before `train_snap_radix.py` and `train_feature_union.py`, and `build_tabpfn_original.py`
before `train_tabpfn_original_feature.py`. The other scripts read only `data/`.

| Pair | Model | OOF AUC |
|---|---|---:|
| xgb_original_knn5_class_distances | raw columns + 5-nearest-original churn mean + distances to the nearest original churner / non-churner | 0.915798 |
| xgb_native_cat_tabpfn_original | native categoricals + the prediction of TabPFN models fitted on the original rows | 0.916507 |
| xgb_snap_radix_original_priors | stumps on snaps, decimal strings, original churn rates of the snaps, 48 radix keys | 0.915747 |
| xgb_union_snap_priors_drift_te | label-free union (snaps, transforms, bins, counts, original priors, drift) + nested TE of 18 columns | 0.918657 |
| xgb_bigram_trigram_te_stats | original-distribution and digit features, nested TE stats of categoricals / numerics as strings, bigram / trigram TE | 0.919015 |
| xgb_ngram_te_binned_keys_drift | the n-gram TE recipe + binned / rounded / digit-pair keys, counts, drift, original churn rates | 0.919175 |
| xgb_te_stats_original_proba | original churn rates, frequencies, nested TE std / min / max / mean of categoricals and numerics as strings | 0.918067 |
| xgb_monthly_anchor_te | MonthlyCharges-anchor target statistics / quantiles, category x anchor, digit-pair and category-pair TE | 0.915428 |
| xgb_total_round0_anchor_te | the same with a TotalCharges anchor, statistics keyed by round(TotalCharges) | 0.917728 |
| xgb_pairwise_ranker_platt | XGBRanker rank:pairwise on the raw columns, Platt-scaled | 0.914916 |

Credits: `train_ngram_te.py` and `train_ngram_te_union.py` adapt blamerx's public notebook
[auc-0-91925-xgboost-bi-tri-target-encoding](https://www.kaggle.com/code/blamerx/auc-0-91925-xgboost-bi-tri-target-encoding);
`train_te_stats_original_proba.py` adapts the XGBoost model of badalkrsharma's public notebook
[rank-blend-cv-0-91865-xgb-lgb-cat](https://www.kaggle.com/code/badalkrsharma/rank-blend-cv-0-91865-xgb-lgb-cat).

Requirements: xgboost (CUDA), pandas, numpy, scikit-learn (>= 1.3, for `TargetEncoder`), scipy; tabpfn for
`build_tabpfn_original.py`.

## More XGBoost models

Ten more GPU XGBoost models in six training scripts (`train_total_tenure_anchor_te.py` holds 2 presets,
`train_union_pipelines_ab.py` holds 4). Same conventions as README.md: every script reads `data/` through
`common/data.py`, cross-validates on the shared 5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)`), runs all
of its folds in one process and writes `preds/<pair>_{oof,test}.npy` (probabilities). Full feature lists and parameters
are in each docstring. Only the competition rows are training rows, except in `train_pairwise_combos_original.py`,
which appends all 7,043 original rows to the training side of every fold.

**Anchor target statistics**
- `train_total_tenure_anchor_te.py` (2 presets): all raw columns (strings as integer codes), category x anchor keys,
  6 anchor digits and digit pairs, the 105 category pairs; per fold (inner KFold(5)) target mean / std / count /
  nunique / median / min / max / skew and 8 quantiles per anchor key, target means of the category x anchor keys,
  digit pairs and category pairs; categories and digits as XGBoost categoricals, depth 6, colsample_bynode 0.3, lr 0.01.
  The anchor is TotalCharges with the statistics keyed by its value rounded to 0.1 (`xgb_total_round1_anchor_te`) or
  the raw tenure (`xgb_tenure_anchor_te`).

**Original-data models**
- `train_original_self_supervised.py`: 95 auxiliary XGBoost models (5 folds x 19 columns) trained on the 7,043
  original rows predict each column from the 18 others, without the target (regression for tenure / MonthlyCharges /
  TotalCharges; class max-probability, entropy and, for binary columns, P(class 1) otherwise); the final model (depth 3,
  lr 0.1, native categoricals) uses the 19 columns + the 41 predictions.
- `train_pairwise_combos_original.py`: smoothed (10) target means and counts of all 171 pairs of the 19 columns, with
  the original rows appended to the training side of every fold (target means out-of-fold over a random 5-way split of
  that training side); strings as native categoricals, depth 8, lr 0.01, colsample_bytree 0.34.

**Feature unions with nested target encoding**
- `train_union_pipelines_ab.py` (4 presets): two label-free pipelines built on train + test + original rows.
  Pipeline A: frequency / rank / log / sqrt / inverse transforms of the numerics, charge relations, service counts and
  Yes / No flags, tenure and charge quantile bins, 6 category pairs and a triple, counts and rare flags, original churn
  rates; pipeline B: quantile bins, roundings and digits of the charges, 27 factorized keys (categoricals, numerics,
  rounded charges, 2-digit fractions) with train-vs-original drift ratios, original churn rates / counts and numeric
  means per categorical, key pairs with <= 20,000 levels. Per fold (inner StratifiedKFold(5)): target std / min / max
  of the 29 A keys, sklearn `TargetEncoder` means of the A keys, B keys and B pairs (with the row-wise mean / std / min /
  max of the B encodings), B pair counts; depth 6, lr 0.03, colsample_bytree 0.4. Presets: pipelines A + B
  (`xgb_union_ab_te`); + a TotalCharges anchor block (TotalCharges in cents, category x anchor keys, anchor digits and
  digit pairs, category pairs; per-fold anchor target statistics / quantiles and key target means; colsample_bytree
  0.8; `xgb_union_ab_total_anchor_te`); + an LGBM-port block (the numerics as string keys with counts and original
  churn rates / counts, round / integer / quantile / equal-width / log bins, decimal digits of the charges and their
  pairs, all target-encoded per fold; `xgb_union_ab_lgbm_port_te`); the LGBM-port variant with every column except
  MonthlyCharges / TotalCharges pre-binned to 256 quantile bins of the fold's training rows and max_bin 16,000
  (`xgb_union_ab_lgbm_port_prebinned`).
- `train_snap_lgbm_port_radix_union.py`: a wide label-free union over train + test + original rows: the raw columns,
  decimal strings and snaps of the charges (`combo.csv`), transforms, charge relations, service flags, tenure / charge /
  snap bins and category crosses, counts and rare flags, drift ratios, original churn rates, roundings, digit
  positions, an LGBM-port block (every column with > 12 levels as a string key with counts and original churn rates /
  counts, 10-quantile and log bins, string digits and digit pairs), radix keys round(100 x snap) / tenure + 100,000 x
  (category code), and 7 pair keys capped at 20,000 levels; per fold nested `TargetEncoder` means of all string keys,
  the other strings as integer codes; depth 6, lr 0.03, colsample_bytree 0.45, max_bin 4096.
- `train_numcat_pairs_cuml_te.py`: native categoricals, the 3 numerics and their digit positions; per fold (inner
  StratifiedKFold(5)) cuML `TargetEncoder` (smooth 10) means of the 48 numeric x categorical keys (exact value +
  category) and of the 3 numerics as keys, plus value counts of the numerics; depth 3, lr 0.02.

```
python train_total_tenure_anchor_te.py xgb_total_round1_anchor_te
python train_total_tenure_anchor_te.py xgb_tenure_anchor_te
python train_original_self_supervised.py
python train_pairwise_combos_original.py
python train_union_pipelines_ab.py xgb_union_ab_te
python train_union_pipelines_ab.py xgb_union_ab_total_anchor_te
python train_union_pipelines_ab.py xgb_union_ab_lgbm_port_te
python train_union_pipelines_ab.py xgb_union_ab_lgbm_port_prebinned
python train_snap_lgbm_port_radix_union.py
python train_numcat_pairs_cuml_te.py
```
Run `../snap_features/build.py` before `train_snap_lgbm_port_radix_union.py` (it reads `combo.csv`). The other scripts
read only `data/`.

| Pair | Model | OOF AUC |
|---|---|---:|
| xgb_total_round1_anchor_te | TotalCharges-anchor target statistics / quantiles keyed by round(TotalCharges, 1), category x anchor, digit-pair and category-pair TE | 0.917678 |
| xgb_tenure_anchor_te | the same with a raw tenure anchor | 0.915757 |
| xgb_original_self_supervised | raw columns + 41 predictions of each column by models trained on the original rows | 0.915474 |
| xgb_union_ab_te | pipelines A + B, nested target std / min / max and TargetEncoder means of their keys and pairs | 0.918667 |
| xgb_pairwise_combo_te_original | target means + counts of the 171 column pairs; original rows appended to every training fold | 0.917745 |
| xgb_union_ab_total_anchor_te | pipelines A + B + TotalCharges-anchor target statistics and anchor-key target means | 0.918490 |
| xgb_union_ab_lgbm_port_te | pipelines A + B + target-encoded LGBM-port keys (bins, integer / decimal digits, digit pairs) | 0.918798 |
| xgb_union_ab_lgbm_port_prebinned | the same, every column but the raw charges pre-binned to 256 quantile bins, max_bin 16,000 | 0.918804 |
| xgb_snap_lgbm_port_radix_nested_te | snap / decimal-string / LGBM-port / radix / pair-key union with nested TargetEncoder means | 0.918645 |
| xgb_numcat_pair_cuml_te_digits | native categoricals + digit positions + cuML TE of 48 numeric x categorical pairs | 0.917838 |

Credits: pipeline A of `train_union_pipelines_ab.py` follows badalkrsharma's
[cv-0-9163-xgb-lgb-multi-seed-ensemble](https://www.kaggle.com/code/badalkrsharma/cv-0-9163-xgb-lgb-multi-seed-ensemble)
and pipeline B mahoganybuttstrings's
[pg-s5e11-xgb-cv-0-92818-pb-0-92923](https://www.kaggle.com/code/mahoganybuttstrings/pg-s5e11-xgb-cv-0-92818-pb-0-92923);
the LGBM-port blocks follow angelosmar1's
[s5e11-lgbm-cv-0-92813-2nd-place](https://www.kaggle.com/code/angelosmar1/s5e11-lgbm-cv-0-92813-2nd-place);
`train_pairwise_combos_original.py` adapts mahoganybuttstrings's
[pg-s5e8-single-xgb-cv-0-975782-lb-0-97681](https://www.kaggle.com/code/mahoganybuttstrings/pg-s5e8-single-xgb-cv-0-975782-lb-0-97681);
`train_numcat_pairs_cuml_te.py` derives from nalgirayergn's public notebook of this competition
[xgb-predict-customer-churn-cv-0-91819](https://www.kaggle.com/code/nalgirayergn/xgb-predict-customer-churn-cv-0-91819).

Requirements: xgboost (CUDA), pandas, numpy, scikit-learn (>= 1.3, for `TargetEncoder`); cudf and cuml (RAPIDS) for
`train_numcat_pairs_cuml_te.py`.
