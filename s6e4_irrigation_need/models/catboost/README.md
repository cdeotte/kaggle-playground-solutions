# CatBoost: original-data priors, value categories and nested target statistics

Six GPU CatBoost models, one self-contained script each: three are members of stack A and three of stack B (stack A
and stack B are the two stacks of `stack/`). Every script reads only `data/`, uses the shared 5 folds
(`common.data.folds`), early-stops on the validation rows (best iteration kept) and writes
`preds/<pair>_{oof,test}.npy` (class probabilities in the order Low, Medium, High; test = mean of the fold models).
`train_nested_5x5.py` runs a nested 5x5 cross-validation and writes an OOF of shape `(5, n_train, 3)`.
Full details are in each script's docstring.

**Stack A**
- `train_numcat_crosses.py`: for every numeric its value, log1p, frequency, integer part mod 10 / 100 and fractional
  part; 6 water / climate relations and a 3-level |rainfall - previous irrigation| flag; category frequencies; 53
  native categories (each numeric rounded to 2 decimals, floored and floored to an even integer, the 8 categoricals,
  a rainfall | irrigation | moisture key, 10 category pairs). All 10,000 original rows are appended to each fold's
  training rows; numerics are ranked by variance and the top 120 kept (all 81 pass). Depth 6, lr 0.03, Bernoulli 0.8,
  one_hot_max_size 4, max_ctr_complexity 1, 5,000 iterations.
- `train_snap_keys.py`: 13 water / climate relations; quantile bins of 5 numerics; 5 numerics snapped to the nearest
  original value (and its first decimal digit); one-hot indicators of 4 categoricals; 12 pair keys, 3 triple keys and
  the rainfall / irrigation snaps crossed with every categorical (54 native categories); original class frequencies,
  counts, rare flags, original counts and train-vs-original drift of all keys; per fold, nested class-frequency
  encodings of 11 keys (inner 5 folds, seed 42 + fold number). Depth 7, lr 0.03, l2 8, Bernoulli 0.8, Cosine score,
  up to 20,000 iterations.
- `train_te_stats.py`: frequencies of the numerics, 2 relations, the original expected class index of every column
  value; per fold, for the 8 categoricals and 3 numerics as 3-decimal strings: std / min / max of the class index over
  an inner 10-fold split plus a scikit-learn `TargetEncoder` mean (continuous, cv 10); the encoded string columns are
  then dropped. Depth 6, lr 0.05, auto_class_weights Balanced, up to 5,000 iterations.

**Stack B**
- `train_class_te_stats.py`: a threshold score of moisture / rainfall / temperature / wind / growth stage / mulching
  and its flags, numeric frequencies, per-class original priors of the categoricals, per-class original percentile
  ranks of 4 numerics, rounded numerics as strings, 15 category pairs and 4 triples; per fold and class: mean / std /
  min / max target statistics of 19 columns and means of the 19 n-grams (inner 5 folds, seed 42). Tuned parameters:
  depth 7, lr 0.00984, l2 5.333, random_strength 2.877, bagging_temperature 0.264, min_data_in_leaf 14, TotalF1
  eval metric, Balanced class weights, up to 20,000 iterations.
- `train_radix_nested_te.py`: the threshold score, digits / frequency / log1p of the numerics, quantile-bin ids,
  3 ratios, per-class original priors, 6 category pairs and 88 "radix" keys (round(10 x numeric) + 50,000 x category
  code, for every numeric x categorical) as native categories; per fold, nested mean-class-index encodings of the 8
  categoricals and 5 bins (inner 5 folds, seed 42 + fold number). Depth 7, lr 0.03, l2 8, Bernoulli 0.8, Cosine
  score, TotalF1, Balanced, up to 20,000 iterations.
- `train_nested_5x5.py`: frequencies, 3 ratios, 3 flags, per-class original priors of all 19 columns, the percentile
  rank / z-score / quartile distances of soil moisture against each original class, numeric values as strings,
  15 pairs and 4 triples. Nested 5x5 (inner split seed 1000 + outer fold number, model seed 42 + inner fold number),
  depth 6, lr 0.02, Balanced, up to 15,000 iterations; test = mean of the 25 models.

```
python train_numcat_crosses.py
python train_snap_keys.py
python train_te_stats.py
python train_class_te_stats.py
python train_radix_nested_te.py
python train_nested_5x5.py
```
Each script runs all folds in one process on one GPU. No other repo folder has to run first.

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| catboost_numcat_crosses_original_rows_a | A | numeric transforms + native value categories and pairs, original rows appended | 0.966253 |
| catboost_snap_bins_keys_nested_te_a | A | snapped / binned values, pair / triple / snap x category keys, original priors, nested class-frequency TE | 0.969425 |
| catboost_freq_priors_te_stats_balanced_a | A | frequencies, original expected-class priors, inner 10-fold TE std / min / max + mean, Balanced | 0.975863 |
| catboost_class_te_stats_ngrams_b | B | per-class TE mean / std / min / max + n-gram TE, tuned parameters, Balanced | 0.975334 |
| catboost_radix_bins_nested_te_b | B | 88 numeric x category radix keys, quantile bins, nested mean-class TE, Balanced | 0.971697 |
| catboost_nested5x5_freq_priors_ngrams_b | B | nested 5x5; frequencies, per-class original priors, soil-moisture class distances, n-grams | 0.971122 |

The nested model's score is that of the mean of its 5 OOF slices.

Requirements: catboost (GPU), scikit-learn (>= 1.3 for `TargetEncoder`), pandas, numpy.
