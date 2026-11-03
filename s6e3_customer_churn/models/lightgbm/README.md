# LightGBM: high-cardinality key encodings, feature unions, decimal strings and original-row references

Thirteen CPU LightGBM models. Seven are presets of one feature recipe (`train_union_te.py` + `lib/highcard_union_features.py`);
six are self-contained scripts. Every model makes its OOF predictions on the shared 5 folds (`common.data.folds`) and
writes `preds/<pair>_{oof,test}.npy` (probabilities).

## High-cardinality key recipe (`train_union_te.py`, `lib/highcard_union_features.py`)

Label-free features of the train + test rows (the original rows give priors, counts and reference distributions):
- **base blocks** (all presets): the 19 columns (string categoricals as native categories); tenure, MonthlyCharges and
  TotalCharges (the columns with more than 7 values) as text keys with their train + test counts, original churn rate
  and count ratio to the original data; round(x) // 2, int(x), <= 5000 quantile bins, equal-width bins and log bins of
  the 3 numerics and SeniorCitizen as keys; up to 4 integer and 4 decimal digits of the two charges and the 6 pairs of
  the TotalCharges decimal digits as keys;
- **union blocks**: frequency / percentile rank / log / sqrt / inverse of the numerics, charge deviation
  (TotalCharges - tenure x MonthlyCharges), ratios and products, service yes / no / other counts, yes / no / other flags,
  tenure bins, 40 / 60-quantile charge bins, 6 category pairs and a triple (keys), original churn rates, counts and
  rare flags of the categoricals and keys; 5-quantile bins, roundings, positional digits, numeric codes and cents
  factorized over train + test + original, train / original drift, original churn rate and count of the categoricals,
  codes and digits, train mean of the numerics per categorical value;
- **distribution / digit / n-gram blocks**: TotalCharges percentile ranks and z-scores against the original churners,
  non-churners and all rows (also within InternetService / Contract), MonthlyCharges minus the original InternetService
  mean, distances to the original churner / non-churner TotalCharges quartiles; first / second / last digits, digit
  counts, modulo 10 / 12 / 100, multiples, roundings and fractional parts of the 3 numerics; the numerics as text, the 15
  pairs of Contract, InternetService, PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling and 4 triples (keys).

Per fold every `_te` column is replaced by its scikit-learn `TargetEncoder` value (binary, shuffled inner 5 folds, seed
42, fitted on the fold's training rows; the training rows are encoded out-of-fold). The gbdt presets use lr 0.01,
max_depth 4, subsample 0.5, colsample_bytree 0.2, lambda_l1 10, lambda_l2 15, seed 42 and a fixed number of rounds; the
GOSS preset (63 leaves, no depth limit) stops early on the validation fold (300 rounds), the DART preset keeps max_depth 4
with 31 leaves (see the table and the docstring of `train_union_te.py`).

Credit: this recipe adapts angelosmar1's public notebook of an earlier Playground competition,
[s5e11-lgbm-cv-0-92813-2nd-place](https://www.kaggle.com/code/angelosmar1/s5e11-lgbm-cv-0-92813-2nd-place).

## Self-contained scripts

- `train_extra_trees_two_stage_te.py`: MonthlyCharges / TotalCharges kept as CSV text; roundings, floor / round bins,
  digits of tenure and 100 x the charges, digit pairs and triples, all 136 pairs of the 17 non-charge base columns,
  35 digit x category crosses, original churn rate and count of the 19 base columns (340 columns); per fold a two-stage
  mean target encoding (inner KFold(5)): stage 1 replaces the 248 pair / digit keys, stage 2 adds encodings of the
  charge texts, tenure, SeniorCitizen, digits, roundings and bins. extra_trees, 141 leaves, depth 7, lr 0.01,
  feature_fraction 0.17, max_bin 1008, early stopping 200. Credit: adapts masayakawamata's public notebook of an earlier
  Playground competition, [s5e11-single-lgbm-tuned](https://www.kaggle.com/code/masayakawamata/s5e11-single-lgbm-tuned).
- `train_union_snap_nested_te.py`: snap values and decimal strings of the two charges from `../snap_features`;
  frequencies, ranks and transforms of the charges and snapped charges, charge relations, service counts and flags,
  tenure / quantile / rounding / integer / decimal-string keys, category pairs, counts, rare flags, original priors and
  drift, original priors of snapped charge x Contract / InternetService, pair keys of 15 columns (<= 20,000 levels),
  <= 5000-quantile / integer / round-to-2 keys; per fold a nested target mean (inner StratifiedKFold(5)) of 154 text
  columns, the other 18 factorized. 63 leaves, depth 6, lr 0.01, colsample_bytree 0.22, early stopping 400.
- `train_decimal_ngram_prototypes.py`: the exact charge texts and snapped charges as tokens (exact strings, character
  2 / 3-grams, decimal strings, quantile bins, fingerprint keys, crosses) hashed into 2^16 columns, TF-IDF and a
  48-component TruncatedSVD; distances and cosine similarities to the centroids of the churned / retained original rows,
  nearest-original-row distances and MiniBatchKMeans prototypes in that embedding; counts of 5 fingerprint keys in train /
  test / original with their original churn rates; target encodings over the shared folds of 3 keys and the prototype
  ids (203 features). 127 leaves, lr 0.028, early stopping 300.
- `train_te_stats_orig_priors.py`: the 3 numerics, TotalCharges / (tenure + 1), tenure x MonthlyCharges, frequencies
  over train + original + test, original churn rate of every value of the 16 categoricals and 3 numerics; per fold the
  target std / min / max (inner StratifiedKFold(10)) and the `TargetEncoder` mean of the 19 columns as keys. 63 leaves,
  depth 7, lr 0.03, early stopping 500. Credit: adapts the LightGBM part of badalkrsharma's public notebook
  [rank-blend-cv-0-91865-xgb-lgb-cat](https://www.kaggle.com/code/badalkrsharma/rank-blend-cv-0-91865-xgb-lgb-cat).
- `train_synthetic_fingerprint.py`: standardized numerics, ordinal categoricals, row statistics and fractional-part
  counts, value / pair / numeric-signature counts over train + test + original, distances to the 1-10 nearest original
  rows, MiniBatchKMeans(128) prototypes of the original rows, target encodings over the shared folds of the cluster id
  and 6 categoricals (110 features). 127 leaves, lr 0.03, early stopping 200.
- `train_nearest_original.py`: the 19 columns plus the churn label of the nearest original row (cKDTree on the
  standardized tenure / MonthlyCharges / TotalCharges of the original rows). 600 rounds, lr 0.03, 31 leaves, depth 6.

Full details are in each script's docstring.

```
python train_union_te.py lgbm_highcard_bins_digits_te
python train_union_te.py lgbm_union_te
python train_union_te.py lgbm_union_dist_ngram_te
python train_union_te.py lgbm_union_dist_ngram_te_slow
python train_union_te.py lgbm_union_dist_ngram_te_20k
python train_union_te.py lgbm_union_dist_ngram_te_goss
python train_union_te.py lgbm_union_dist_ngram_te_dart
python train_extra_trees_two_stage_te.py
python train_union_snap_nested_te.py
python train_decimal_ngram_prototypes.py
python train_te_stats_orig_priors.py
python train_synthetic_fingerprint.py
python train_nearest_original.py
```
`train_union_snap_nested_te.py` and `train_decimal_ngram_prototypes.py` read `../snap_features/cache/combo.csv`: run
`python build.py` in `../snap_features` first. The other scripts read only `data/`. Each command runs all folds in one
process on 32 CPU threads, from a few minutes to about 2.5 hours per model.

| Pair | Model | OOF AUC |
|---|---|---:|
| lgbm_highcard_bins_digits_te | recipe base blocks: high-cardinality keys, counts, original priors, bins, digits and digit pairs (72 columns); TargetEncoder smooth 10; 7,500 rounds | 0.918611 |
| lgbm_union_te | base + union blocks (391 columns: transforms, flags, crosses, original priors, counts, codes, drift, group means); TargetEncoder auto; 7,500 rounds | 0.919025 |
| lgbm_union_dist_ngram_te | base + union + original-distribution, explicit digit and n-gram key blocks (466 columns); 10,000 rounds | 0.919138 |
| lgbm_union_dist_ngram_te_slow | the same features; 30,000 rounds, lr 0.0025 | 0.919164 |
| lgbm_union_dist_ngram_te_20k | the same features; 20,000 rounds | 0.918791 |
| lgbm_union_dist_ngram_te_goss | the same features; GOSS, 63 leaves, early stopping | 0.918806 |
| lgbm_union_dist_ngram_te_dart | the same features; DART, 31 leaves, depth 4 | 0.918742 |
| lgbm_extra_trees_pairs_digits_two_stage_te | category pairs, digit pairs / triples / crosses, roundings, bins, original priors; two-stage mean target encoding; extra_trees | 0.918361 |
| lgbm_union_snap_pairs_nested_te | union with snap features, decimal strings, pair keys, quantile / integer keys, original priors; nested target means | 0.918782 |
| lgbm_decimal_ngram_svd_prototypes | hashed charge-text n-grams in a TF-IDF / SVD embedding, original-row distances, k-means prototypes, fingerprint counts, target encodings | 0.914912 |
| lgbm_te_stats_orig_priors | original churn rates, frequencies, nested target std / min / max and TargetEncoder means of 19 keys | 0.917891 |
| lgbm_synthetic_fingerprint | value / pair / signature counts, nearest-original distances, k-means prototypes, target encodings | 0.915831 |
| lgbm_nearest_original_churn | raw columns + churn label of the nearest original row (cKDTree) | 0.915976 |

Requirements: lightgbm (4.x), scikit-learn (>= 1.3 for `TargetEncoder`), scipy, pandas, numpy (CPU only).
