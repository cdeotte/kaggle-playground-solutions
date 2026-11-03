# XGBoost

## Stack A: snap / radix, union pipelines, target-encoded keys, original-data statistics

Nineteen GPU XGBoost members of stack A (stack A and stack B are the two stacks of `stack/`) in 13 self-contained
scripts; three scripts hold presets (`train_union_te_a.py` 4, `train_snap_radix_a.py` 3, `train_numeric_pair_te_a.py`
2). Every script reads only `data/`, uses the shared 5 folds (`common.data.folds`) and writes
`preds/<pair>_{oof,test}.npy` (class probabilities in the order Low, Medium, High; test = mean of the 5 fold models).
Each script keeps every finished fold in `parts/` (`--folds 0,1` runs a subset, `--force` reruns finished folds; each
process rebuilds its features) and writes the pair once all 5 folds are done. Full details (feature lists, parameters,
run times) are in each docstring.

Recurring features: "snap": a train / test value of Rainfall_mm or Previous_Irrigation_mm that does not occur in the
original data is replaced by the nearest original value, plus snap - value; "radix" keys: round(100 x value) +
100,000 x (category code), i.e. a numeric value within each category; "original priors": per class, the share of that
class among the original rows with the same value; "nested target encoding": per-class encodings, out-of-fold over an
inner StratifiedKFold(5, shuffle, 42) of the fold's training rows and fitted on all training rows for the validation /
test rows. Unless stated otherwise a model is multi:softprob with early stopping (mlogloss) on the validation fold, and
only the competition rows are training rows (the 10,000 original rows only feed statistics).

**Snap / radix, shallow and self-supervised models**
- `train_snap_radix_a.py` (3 presets): native categoricals, the 11 numerics, rain / irrigation snaps, the 2-decimal
  digit strings of rain / irrigation and radix keys of both snaps x the 8 categoricals (47 columns); depth 1 (stumps), lr 0.1, max_bin
  16,000, subsample / colsample_bytree 0.85, early stopping 300, no class weights. Two presets add a radix block of
  Field_Area_hectare or Sunlight_Hours (55 columns). About a minute of features, 4-10 minutes per fold.
- `train_minimal_fe_a.py`: native categoricals, the 11 numerics and 7 water-balance interactions (moisture - water
  input, its absolute value, water per field area, moisture / water and water / moisture ratios, irrigation x rain,
  temperature x humidity); depth 2, lr 0.1, max_bin 16,000, no class weights. About 6 minutes per fold.
- `train_digit_round_categories_a.py`: native categoricals plus 6 arithmetic features; 6 numerics rounded to 1 and to
  0 decimals, 3 category triples and the integer / 2 decimal digit positions of the same numerics, all as categories
  (67 columns); depth 3, lr 0.03, max_bin 16,000, balanced sample weights.
- `train_original_self_supervised_a.py`: XGBoost models trained on the original rows (5 folds, early stopping on the
  held-out original rows) predict each of the 19 columns from the 18 others; the final model (depth 3, lr 0.1, native
  categoricals, balanced sample weights) uses the 19 raw columns + 28 predictions (value, or max probability / entropy
  of the class). About 2 minutes for the auxiliary models, 1 minute per fold.

**Target-encoded keys**
- `train_numeric_pair_te_a.py` (2 presets): native categoricals, numerics, 44 digit positions, the rule score of the
  original data (2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10] -
  2 [stage Harvest] - 2 [stage Sowing] - [mulch Yes]) with its flags; per fold, cuML `TargetEncoder` (5 random inner
  folds, smooth 0) class rates of the 88 numeric x categorical value pairs and the 11 numeric values, and the value
  counts of the numerics; depth 4, lr 0.03, balanced class weights. The second preset uses max_bin 16,000. About 4
  (max_bin 16,000: 20) minutes per fold.
- `train_pairwise_combo_te_a.py`: the 171 pairs of the 19 columns; per fold, with all original rows appended to the
  training rows, smoothed (10) class rates of every pair (out-of-fold over a random 5-way split of the training side)
  and its count; depth 8, lr 0.01, no class weights. About 2 hours.
- `train_ngram_te_stats_a.py`: numeric frequencies, 4 water / climate interactions, management flags, original priors
  of all 19 columns, Rainfall_mm percentile ranks / z-gaps / quantile distances against the original Low / Medium /
  High rows; per fold (inner 5 folds) the std / min / max of the class id and per-class sklearn `TargetEncoder` means
  of the 8 categoricals and the 11 numerics as 2-decimal categories, class rates of 15 bigrams and 4 trigrams (272
  columns); tuned slow parameters (lr 0.01, depth 5, colsample_bytree 0.32, alpha 3.5, gamma 0.79), no class weights.
- `train_group_priors_te_a.py`: 11 numerics, 8 engineered numerics and roundings, counts and percentile ranks on the
  fold's training rows, original priors of categoricals / a Crop_Type x Irrigation_Type group / 6 crosses, per-class
  z-distance profiles to the original classes, smoothed (20) class rates of the 6 crosses fitted on the training rows;
  depth 7, lr 0.01, balanced class weights. As run, the original labels stay strings in the original-data lookups, so
  those priors reduce to seen-in-original flags and the distance profile is all missing. About 5 minutes per fold.
- `train_rainfall_anchor_union_a.py`: pipelines A (8 interactions, 2 triples) and B (quantile bins, rounding, digits,
  fraction / mod features), original priors, anchor round(10 x Rainfall_mm) with its digits, digit pairs and category
  x anchor keys; per fold: anchor statistics and quantiles, counts of 78 keys, nested class rates of the 78 keys and
  30 count columns (480 columns); depth 6, lr 0.03, no class weights. About 13 minutes per fold.
- `train_numeric_keys_te_a.py`: 18 keys per numeric (roundings, floor, steps, quantile / equal-width / log bins, digit
  characters and pairs), 10 category n-grams, interactions, original counts / priors / z-scores, pair keys (bins
  computed on each frame); per fold, nested class rates of 211 keys, the remaining columns (including id) kept;
  depth 6, lr 0.03, colsample_bytree 0.4, no class weights. About 1 hour per fold.

**Union pipelines**
- `train_snap_priors_drift_a.py`: snaps, decimal parts, management counts, 16 / 32-quantile bins, roundings, digits,
  counts and rare flags, original membership / priors / counts, original priors of 4 category pairs, drift
  log1p(train share / original share); per fold, nested per-class sklearn `TargetEncoder` encodings of 25 keys,
  columns sorted by name; lr 0.02, depth 6, colsample_bytree 0.55, max_bin 4096, balanced class weights. About 45
  minutes of features (the snap step), 4.5 minutes per fold.
- `train_union_te_a.py` (4 presets): a 479-column label-free union on train + test + original rows (snaps, frequency /
  rank, transforms, water-balance relations, management flags, quantile bins and crosses, counts and rare flags,
  drift, original priors, roundings, digit positions, a block of string keys / counts / priors / bins / digit pairs in
  the style of a public LightGBM notebook, radix keys of the snaps and Soil_Moisture, capped pair keys); per fold,
  nested per-class sklearn `TargetEncoder` encodings of 59 `*_te` keys (597 columns); depth 6, lr 0.03, max_bin 4096,
  balanced class weights. As run, the test frame's string columns are integer-coded on the test rows alone (the
  training / validation rows share codes). Presets: colsample_bytree 0.45 with min_child_weight 6, gamma 0.1,
  alpha 0.5, lambda 4; and colsample_bytree 0.80 / 0.60 / 0.40 with min_child_weight 5, gamma 0.05, alpha 0.1,
  lambda 1. 30-40 minutes of features (the snap maps), about 15 minutes per fold.
- `train_union_prebinned_a.py`: pipelines A (7 interactions, a crop x season x irrigation key) and B (quantile bins,
  roundings, digits, 4 category pairs, original means / counts) and string keys / counts / original statistics / fine
  bins / digit keys of the numerics in the style of a public LightGBM notebook; per fold, class rates of the A key,
  multiclass sklearn `TargetEncoder` encodings of 84 keys (row mean / std / min / max of the B and string-key
  encodings), counts of the B keys; every
  column except Rainfall_mm and Previous_Irrigation_mm is pre-binned to 256 quantile bins of the fold's training rows
  (444 columns), max_bin 16,000, depth 6, lr 0.03, no class weights. About 3.5 minutes per fold.

```
python train_union_te_a.py xgb_union_snap_te_a
python train_union_te_a.py xgb_union_snap_te_cs080_a
python train_union_te_a.py xgb_union_snap_te_cs060_a
python train_union_te_a.py xgb_union_snap_te_cs040_a
python train_snap_radix_a.py xgb_snap_radix_depth1_a
python train_snap_radix_a.py xgb_snap_radix_field_area_depth1_a
python train_snap_radix_a.py xgb_snap_radix_sunlight_depth1_a
python train_numeric_pair_te_a.py xgb_numeric_pair_te_a
python train_numeric_pair_te_a.py xgb_numeric_pair_te_maxbin16000_a
python train_snap_priors_drift_a.py
python train_ngram_te_stats_a.py
python train_original_self_supervised_a.py
python train_pairwise_combo_te_a.py
python train_minimal_fe_a.py
python train_group_priors_te_a.py
python train_digit_round_categories_a.py
python train_rainfall_anchor_union_a.py
python train_numeric_keys_te_a.py
python train_union_prebinned_a.py
```
Each script runs on one GPU. No other repo folder has to run first.

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| xgb_union_snap_te_a | A | 479-column union (snaps, bins, drift, priors, digits, string keys, radix, pairs) + nested per-class TE of 59 keys; colsample 0.45, lambda 4 | 0.976857 |
| xgb_union_snap_te_cs080_a | A | the same union, lighter regularization (lambda 1, alpha 0.1, gamma 0.05), colsample 0.80 | 0.976639 |
| xgb_union_snap_te_cs060_a | A | the same, colsample 0.60 | 0.976888 |
| xgb_union_snap_te_cs040_a | A | the same, colsample 0.40 | 0.976827 |
| xgb_snap_radix_depth1_a | A | depth-1 stumps; rain / irrigation snaps, decimal digits, radix keys snap x 8 categoricals | 0.962792 |
| xgb_snap_radix_field_area_depth1_a | A | the same + Field_Area_hectare radix keys | 0.962861 |
| xgb_snap_radix_sunlight_depth1_a | A | the same + Sunlight_Hours radix keys | 0.962780 |
| xgb_numeric_pair_te_a | A | cuML per-class TE of 88 numeric x categorical pairs + 11 numeric values, value counts, digits, rule score | 0.976064 |
| xgb_numeric_pair_te_maxbin16000_a | A | the same, max_bin 16,000 | 0.976075 |
| xgb_snap_priors_drift_te_a | A | snaps, quantile / round / digit keys, counts, original priors, drift + nested per-class TE of 25 keys | 0.974354 |
| xgb_ngram_te_stats_a | A | original distribution features + inner-fold label std / min / max and per-class TE of 19 keys, n-gram class rates | 0.971595 |
| xgb_original_self_supervised_a | A | raw columns + 28 predictions of each column by models trained on the original rows | 0.968615 |
| xgb_pairwise_combo_te_original_a | A | class rates + counts of the 171 column pairs, original rows appended to every training fold | 0.970693 |
| xgb_minimal_fe_depth2_a | A | depth 2; raw columns + 7 water-balance interactions, native categoricals | 0.967529 |
| xgb_group_priors_interaction_te_a | A | engineered numerics, counts / ranks, original group priors, in-fold class rates of 6 crosses | 0.969715 |
| xgb_digit_round_categories_a | A | depth 3; digit positions, rounded numerics and category triples as native categoricals | 0.971352 |
| xgb_rainfall_anchor_union_te_a | A | Rainfall_mm anchor keys / statistics, pipelines A / B, original priors + nested class rates of 108 keys | 0.970961 |
| xgb_numeric_keys_ngram_te_a | A | 18 keys per numeric, n-grams, original statistics + nested class rates of 211 keys | 0.973352 |
| xgb_union_te_prebinned_a | A | pipelines A / B / string keys + multiclass TE of 84 keys, pre-binned to 256 bins except rain / irrigation, max_bin 16,000 | 0.972771 |

Requirements: xgboost (>= 2.0, CUDA), scikit-learn (>= 1.3 for `TargetEncoder`), pandas 2.x (string columns as object
dtype), numpy; cuDF / cuML / CuPy (RAPIDS) for `train_numeric_pair_te_a.py`.

## Stack B: rankers, formula margins, snap / radix, anchor statistics, nested target encodings, union pipelines

Sixteen GPU XGBoost members of stack B (stack A and stack B are the two stacks of `stack/`) in 13 self-contained
scripts; three scripts hold two presets each. Every script reads only `data/`, uses the shared 5 folds
(`common.data.folds`) and writes `preds/<pair>_{oof,test}.npy` (class probabilities in the order Low, Medium, High;
test = mean of the fold models). The per-fold scripts keep each finished fold in `parts/` (`--folds 0,1` runs a
subset, `--force` reruns finished folds; each process rebuilds its features) and write the pair once all 5 folds are
done. `train_rank_platt_b.py` runs in one process. `train_snap_nested_b.py` runs a nested 5x5 cross-validation and
writes an OOF of shape `(5, n_train, 3)`. Full details (feature lists, parameters, run times) are in each docstring.

Recurring features: the rule score of the original data, 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] +
[Temperature_C > 30] + [Wind_Speed_kmh > 10] - 2 [stage Harvest] - 2 [stage Sowing] - [mulch Yes], with is_low
(score <= 0) and is_high (score > 3); "original priors": per class, the share of that class among the original rows
with the same value; "snap": a value replaced by the nearest value of the original data, plus value - snap. Unless
stated otherwise a model is multi:softprob with balanced class weights and early stopping (mlogloss) on the validation
fold, and only the competition rows are training rows.

**Rankers, margins, shallow and nested snap models**
- `train_rank_platt_b.py`: 91 features (raw columns, rule score and thresholds, value frequencies, digits of the 4
  formula numerics, 8 categorical pairs, original priors of the categoricals and 4 pairs). Per fold 4 pairwise rankers
  (`rank:pairwise`, random query groups of 256 rows): one ordinal (Low < Medium < High) and one one-vs-rest per class;
  depth 6, lr 0.03, early stopping 200. Their out-of-fold scores + the 91 features go into a balanced multinomial
  logistic regression (C 1, standardized), fitted per fold on the other 4 folds.
- `train_formula_margin_b.py` (2 presets): the native API boosts from a 3-class base margin set by the rule (train,
  validation and test rows alike) on the 19 raw columns (native categoricals); depth 4, lr 0.05, early stopping 500,
  predictions from all trees of the returned booster. `xgb_rule_formula_margin_b`: logit 2 for the rule's class and -1
  for the others, max_bin 1024. `xgb_logit_formula_margin_b`: 3 linear logits of the 4 threshold flags and the
  growth-stage / mulching indicators.
- `train_snap_radix_b.py` (2 presets): snaps of all 11 numerics, digits of the formula numerics, the rule score,
  native categoricals, original priors and "radix" integers round(100 x snap) + 100,000 x (category code); depth 2,
  lr 0.1, max_bin 16,000, early stopping 300. Radix keys on 5 numerics (40) or on all 11 (88).
- `train_snap_nested_b.py`: the 4 formula numerics snapped to the original grid (+ differences), the other raw columns,
  per-class original target encodings of the 4 snaps. Nested 5x5 (inner split seed 1000 + outer fold number, model seed
  42 + inner fold number); depth 4, lr 0.01, max_bin 1024, early stopping 500 on balanced accuracy (best model kept);
  test = mean of the 25 models.
- `train_digit_positions_b.py`: native categoricals, the numerics, all integer digit positions and 2 decimal digits of
  every numeric, the rule score with its flags and one-hot class; depth 3, lr 0.02, max_bin 16,000, no class weights.

**Target statistics and nested target encodings**
- `train_num_cat_pair_te_b.py`: digits, rule score, original priors and value counts; per fold, cuML `TargetEncoder`
  (smooth 10) one-vs-rest encodings of the 88 numeric x categorical value pairs and the 11 numeric values, out-of-fold
  over an inner StratifiedKFold(5, seed 42 + fold number) (398 columns); depth 3, lr 0.02.
- `train_nums_as_cats_te_b.py`: numerics, rule score and flags, frequencies, original priors of all 19 columns; per
  fold, for the 8 categoricals and the 11 numerics as strings: label std / min / max over 5 inner folds and 3 per-class
  sklearn `TargetEncoder` means; the raw keys are dropped (204 columns); depth 6, lr 0.03.
- `train_cat_ngram_te_b.py`: raw columns, rule score, frequencies, original priors; per fold, per-class nested
  encodings (sklearn `TargetEncoder`, inner 5 folds, seed 42 + fold number) of 15 bigrams, 4 trigrams and the 8
  categoricals; tuned slow parameters (lr 0.0063, depth 5, colsample_bytree 0.32, alpha 3.5, gamma 0.79).
- `train_snap_anchor_te_b.py`: snaps of all numerics; Rainfall_mm and Soil_Moisture snaps as anchors: category x anchor
  keys, 6 decimal digits and 10 digit pairs per anchor, 28 category pairs; per fold over inner KFold(5): per-anchor
  label mean / std / count / nunique / median / min / max / skew, 8 quantiles and 3 class shares, and the class shares
  of every key; depth 6, lr 0.01, colsample_bynode 0.3.
- `train_irrigation_anchor_te_b.py`: the same design around one anchor, round(Previous_Irrigation_mm), without snaps,
  with original priors and the numeric part of the rule.

**Union pipelines**
- `train_union_nested_te_original_b.py`: frequency / rank / log / sqrt transforms, rule score and thresholds, 8 pairs
  and 2 triples, 16 / 32 / 64-quantile bins, digits and rounded values, counts and rare flags, original priors / counts,
  drift log1p(train share / original share), "seen in the original data" flags; per fold, per-class nested encodings of
  26 keys; the 10,000 original rows are appended to the training rows. Depth 5, lr 0.025, max_bin 16,000.
- `train_union_multiclass_te_b.py` (2 presets): pipelines A (rule flags, frequency, rank, transforms, 8 arithmetic
  interactions, priors, all 28 category pairs + 3 triples and their counts), B (digits) and an LGBM-style block
  (round / int / quantile bins, value counts, original mean label and count, drift, 135 value pairs): 441 columns; per
  fold with the original rows appended: label std of 50 keys (inner 5 folds) and multiclass `TargetEncoder` encodings of
  218 keys with row mean / std / min / max (1157 columns); depth 6, lr 0.03, colsample_bytree 0.4. The pre-binned
  preset cuts every column except the 4 formula numerics into 256 quantile bins of the fold's training rows (max_bin
  16,000) and also computes the drift on the original rows.
- `train_mega_union_b.py`: an 882-column label-free union on train + test + original rows (raw columns including the
  row id, snaps, frequency / rank, transforms, rule score and interactions, bins and crosses, counts, drift, original
  priors, rounded values, digit positions, an LGBM-style block of string keys / counts / priors / ratios / bins / string
  digits, radix keys, capped pair keys); per fold, per-class nested encodings of every `*_te` key (1066 columns); depth
  6, lr 0.03, colsample_bytree 0.45, max_bin 4096.

```
python train_rank_platt_b.py
python train_formula_margin_b.py xgb_rule_formula_margin_b
python train_formula_margin_b.py xgb_logit_formula_margin_b
python train_snap_radix_b.py xgb_snap_radix_5num_depth2_b
python train_snap_radix_b.py xgb_snap_radix_11num_depth2_b
python train_snap_nested_b.py
python train_digit_positions_b.py
python train_num_cat_pair_te_b.py
python train_nums_as_cats_te_b.py
python train_cat_ngram_te_b.py
python train_snap_anchor_te_b.py
python train_irrigation_anchor_te_b.py
python train_union_nested_te_original_b.py
python train_union_multiclass_te_b.py xgb_union_multiclass_te_original_b
python train_union_multiclass_te_b.py xgb_union_multiclass_te_original_prebinned_b
python train_mega_union_b.py
```
Each script runs on one GPU. No other repo folder has to run first.

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| xgb_rank_ordinal_ovr_platt_b | B | ordinal + 3 one-vs-rest pairwise rankers, balanced multinomial LR on their scores + 91 features | 0.972911 |
| xgb_rule_formula_margin_b | B | base margin 2 / -1 from the rule's class, 19 raw columns, max_bin 1024 | 0.969886 |
| xgb_logit_formula_margin_b | B | base margin = 3 linear logits of the rule flags, 19 raw columns | 0.969102 |
| xgb_snap_radix_5num_depth2_b | B | depth 2; snaps, digits, priors, radix keys of 5 numerics x 8 categoricals | 0.971374 |
| xgb_snap_radix_11num_depth2_b | B | depth 2; snaps, digits, priors, radix keys of all 11 numerics x 8 categoricals | 0.970539 |
| xgb_snap_original_te_nested_b | B | nested 5x5; formula numerics snapped to the original grid, original TE of the snaps | 0.973015 |
| xgb_native_cat_digit_positions_b | B | native categoricals, integer / decimal digit positions, rule flags, no class weights | 0.966304 |
| xgb_num_cat_pair_te_digits_b | B | cuML nested one-vs-rest TE of 88 numeric x categorical pairs + 11 numeric values, digits | 0.976527 |
| xgb_nums_as_cats_te_stats_b | B | numerics as categories: inner-fold label std / min / max + per-class TE of 19 keys | 0.976824 |
| xgb_cat_ngram_nested_te_b | B | per-class nested TE of bigrams / trigrams / categoricals, tuned slow parameters | 0.968950 |
| xgb_rain_moisture_snap_anchor_te_b | B | Rainfall / Soil_Moisture snap anchors: label statistics, quantiles, class shares of anchor keys | 0.973955 |
| xgb_prev_irrigation_anchor_te_b | B | round(Previous_Irrigation_mm) anchor: label statistics, quantiles, class shares of anchor keys | 0.966615 |
| xgb_union_nested_te_original_b | B | transforms / bins / digits / counts / priors / drift + nested per-class TE of 26 keys, original rows appended | 0.972985 |
| xgb_union_multiclass_te_original_b | B | pipelines A / B / LGBM-style (441 columns) + label std and multiclass TE of 218 keys, original rows appended | 0.976833 |
| xgb_union_multiclass_te_original_prebinned_b | B | the same, pre-binned to 256 quantile bins except the 4 formula numerics, max_bin 16,000 | 0.976740 |
| xgb_mega_union_snap_radix_nested_te_b | B | 882-column union (snaps, bins, drift, priors, digits, string keys, radix) + nested per-class TE | 0.976777 |

The nested model's score is that of the mean of its 5 OOF slices.

Requirements: xgboost (>= 2.0, CUDA), scikit-learn (>= 1.3 for `TargetEncoder`), pandas 2.x (string columns as object
dtype), numpy; cuDF / cuML / CuPy (RAPIDS) for `train_num_cat_pair_te_b.py`.
