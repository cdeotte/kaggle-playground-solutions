# LightGBM: value tokens, original-data priors and nested target encodings

Eleven CPU LightGBM models, one self-contained script each: six are members of stack A and five of stack B (stack A
and stack B are the two stacks of `stack/`). Every script reads only `data/`, uses the shared 5 folds
(`common.data.folds`) and writes `preds/<pair>_{oof,test}.npy` (class probabilities in the order Low, Medium, High;
test = mean of the fold models). Most models turn numerics, digits, bins and category crosses into string keys and
replace them per fold by class encodings computed out-of-fold on the fold's training rows (inner 5-fold split) and
from all training rows for the validation / test rows; the original 10,000 rows are used only for priors, counts,
ranks and drift, never as training rows. `train_nested_5x5.py` runs a nested 5x5 cross-validation and writes an OOF of
shape `(5, n_train, 3)`. Full details are in each script's docstring.

**Stack A**
- `train_highcard_te_digits.py`: every numeric as a key with its count, original count, per-class original frequency
  and count ratio; round(x) // 2, int(x), <= 5000 quantile / equal-width bins of 5 numerics; 3 integer + 3 decimal
  digits of 4 numerics and 4 digit pairs; 6 category pairs; 41 keys encoded by class frequencies smoothed toward the
  prior (weight 10). 7500 rounds, lr 0.01, depth 4, colsample 0.2, L1 10 / L2 15.
- `train_extra_trees_two_stage_te.py`: integer-cast roundings and floor bins of 6 numerics, 3 + 3 digits of 4
  numerics and 14 digit pairs, all 28 category pairs, original priors / counts of the categoricals, roundings, bins
  and digit pairs; stage-1 encoding of the category and digit pairs (dropped), stage-2 encoding of 5 numerics, the
  digits, roundings, bins and priors (kept), smoothing 20. extra_trees, 141 leaves, depth 7, feature_fraction 0.17,
  max_bin 1008, early stopping 500.
- `train_highcard_te_artifacts.py`: the value keys, counts, original priors and numeric tokens of the first model,
  6 pair keys and their counts, 5-quantile bins and digits at 100 / 10 / 1 / 0.1 of 4 numerics as keys, 5 ratios;
  57 keys encoded by plain class frequencies. lr 0.005, depth 4, early stopping 500.
- `train_union_class_freq_te.py`: value keys, counts, add-one smoothed original priors, numeric tokens, up to 4 + 4
  digits of every numeric; frequencies, ranks, transforms and bins of the main numerics; quantile bins, roundings,
  digits, factorized codes, train / original drift, original priors and train group means; previous irrigation
  against the original class distributions; 15 category pairs and 4 triples; 57 keys encoded by add-one smoothed class
  frequencies fitted on the fold's training rows. Balanced class weights, lr 0.01, depth 4, early stopping 500.
- `train_goss_union.py`: the blocks of the previous models for all 11 numerics (tokens, 4 + 4 digits, digit pairs,
  ranks, 9 ratios, bins, priors, counts / rare flags, drift, class distributions of previous irrigation, rounded
  numerics and category n-grams); 158 columns encoded by class frequencies. GOSS, 63 leaves, early stopping 300.
- `train_dart.py`: frequencies, 6 relations, percentile ranks / z-scores of 4 numerics within each original class,
  digits, original class priors of the categoricals and 4 rounded numerics, factorized categoricals; scikit-learn
  `TargetEncoder` (multiclass, cv 5) of 36 keys (categoricals, 2-decimal numerics, quantile bins, 15 pairs, 4
  triples). DART, 3000 rounds, lr 0.03, 127 leaves.

**Stack B**
- `train_class_dist.py`: a moisture / rainfall / temperature / wind / growth-stage / mulching threshold score and
  flags; counts, original counts and original mean class of the numerics; train / original drift; train group means;
  every numeric's percentile rank and z-score within each original class, class-quartile distances, conditional ranks
  within growth stage / mulching; scikit-learn `TargetEncoder` (multiclass, cv 5) of 65 keys (categoricals, 4 numeric
  bins per numeric, 10 pairs, 3 triples). class_weight High = 5, lr 0.01, depth 4, early stopping 500.
- `train_three_block_union.py`: block 1 (all 19 columns as keys, counts, original priors, numeric bins, digits and
  digit pairs), block A (frequency / rank / log / sqrt / inverse of every numeric, the threshold score, quantile bins,
  6 pairs and a triple, original priors, counts), block B (quantile bins, roundings, digits, drift, original priors
  and counts, train group means); 103 columns encoded per class with smoothing 10 over 5 random inner folds. Balanced
  class weights, 15 leaves, depth 4, 7500 rounds, early stopping 500.
- `train_goss_crosses.py`: frequency / rank / log / sqrt / inverse of every numeric, the threshold score, 8 category
  pairs and a triple, 32 / 64 / 128-quantile bins, digits and roundings of 4 numerics, category counts, original
  priors, drift and original group means; per-class binary scikit-learn `TargetEncoder` of 25 columns (inner 5 folds,
  seed 42 + fold number). GOSS, balanced class weights, 63 leaves, early stopping 300.
- `train_extra_trees_binary_te.py`: roundings, 3 + 2 digits of every numeric, units-digit pairs, 32 / 64-quantile
  bins, 8 category pairs, the threshold score, original priors; two-stage per-class binary `TargetEncoder` (stage 1:
  pairs and digit keys, dropped; stage 2: numerics, digits, roundings, bins, priors, kept). extra_trees, balanced class
  weights, 141 leaves, depth 7, max_bin 1008, early stopping 200.
- `train_nested_5x5.py`: the 8 categoricals as native categories, their counts and original class frequencies,
  2 digits of 3 numerics; scikit-learn `TargetEncoder` (multiclass, smooth 10) of the categoricals, their 28 pairs and
  quantile bins of soil moisture / rainfall, fitted per inner fold. Nested 5x5 (inner split seed 1000 + outer fold
  number), balanced class weights, lr 0.01, depth 4, early stopping 500 on the inner validation rows; test = mean of the
  25 models.

Credit: `train_nested_5x5.py` adapts the recipe of angelosmar1's public notebook
[s5e11-lgbm-cv-0-92813-2nd-place](https://www.kaggle.com/code/angelosmar1/s5e11-lgbm-cv-0-92813-2nd-place).

```
python train_highcard_te_digits.py
python train_extra_trees_two_stage_te.py
python train_highcard_te_artifacts.py
python train_union_class_freq_te.py
python train_goss_union.py
python train_dart.py
python train_class_dist.py
python train_three_block_union.py
python train_goss_crosses.py
python train_extra_trees_binary_te.py
python train_nested_5x5.py
```
Each script runs all folds in one process on 32 CPU threads (roughly one to several hours per model). No other repo
folder has to run first.

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| lgbm_highcard_te_counts_digits_a | A | value keys, counts, original priors, numeric tokens, digits; nested smoothed class-frequency TE | 0.973248 |
| lgbm_extra_trees_round_digit_two_stage_te_a | A | roundings, floor bins, digits, 28 category pairs; two-stage nested TE; extra_trees | 0.970390 |
| lgbm_highcard_te_artifact_tokens_a | A | value keys, counts, original priors, quantile / digit tokens; nested class-frequency TE; lr 0.005 | 0.973201 |
| lgbm_union_dist_ngrams_class_freq_te_a | A | union of keys, digits, drift, original priors, class distributions, n-grams; add-one class-frequency TE; balanced weights | 0.973689 |
| lgbm_goss_union_nested_te_a | A | union of keys, digits, ranks, drift, distributions, n-grams; nested class-frequency TE; GOSS | 0.973062 |
| lgbm_dart_priors_ngrams_sklearn_te_a | A | frequencies, original class distributions and priors, digits; TargetEncoder of 36 keys; DART | 0.970240 |
| lgbm_class_pctrank_zscore_te_b | B | percentile ranks / z-scores within the original classes, counts, drift; TargetEncoder; High weight 5 | 0.977026 |
| lgbm_three_block_union_nested_te_b | B | three feature blocks; nested smoothed per-class TE; balanced weights | 0.978524 |
| lgbm_goss_crosses_binary_te_b | B | transforms, crosses, bins, digits, original priors; nested per-class binary TE; GOSS, balanced weights | 0.972563 |
| lgbm_extra_trees_two_stage_binary_te_b | B | roundings, digits, bins, pairs; two-stage nested per-class binary TE; extra_trees, balanced weights | 0.976727 |
| lgbm_nested5x5_cat_pairs_te_b | B | nested 5x5; categoricals, 28 pairs, counts, original priors, bins, digits; TargetEncoder; balanced weights | 0.972146 |

The nested model's score is that of the mean of its 5 OOF slices.

Requirements: lightgbm (4.x), scikit-learn (>= 1.3 for `TargetEncoder`), pandas, numpy (CPU only).
