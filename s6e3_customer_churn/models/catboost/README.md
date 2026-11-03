# CatBoost: numeric artifacts, original-data priors, fold statistics and native categoricals

Fourteen GPU CatBoost models in ten scripts. Three scripts hold two recipes each that differ in how the 7,043-row
original IBM dataset is used: as lookups (original churn rates, counts, nearest original values, drift) or as extra
training rows appended to every fold (`*_orig_rows`, no original lookups). Every script reads the data through
`common.data`, uses the shared 5 folds (`common.data.folds`), early-stops on the validation fold (best iteration kept)
and writes `preds/<pair>_{oof,test}.npy` (probabilities). Full details are in each script's docstring.

- `train_numcat_artifacts.py` (2 recipes): features over train + test (+ original rows): tenure, MonthlyCharges,
  TotalCharges and SeniorCitizen with log / sqrt / square / inverse / percentile rank, 9 charge relations, value
  counts of every column, digit / fractional / mod artifacts, Yes / No service counts, and 65 CatBoost-native
  categoricals (raw columns including id; exact / rounded / integer / 50-quantile / even-floor strings of the
  numerics; 10 bigrams and a trigram; numeric x id / gender / Partner crosses; digit pairs). The lookup recipe adds
  original value counts and churn rates, TotalCharges percentiles among original churners / non-churners,
  train-vs-original drift and original group means / stds. Depth 8, lr 0.03, Bayesian bagging, max_ctr_complexity 2.
- `train_fold_density.py` (2 recipes): row-level features (charge relations, transforms, service flags, string
  length / decimals / trailing zeros / hash / digit positions, TotalCharges vs tenure x MonthlyCharges residuals),
  per-fold features from the fold's training rows (frequencies, ranks, joint counts and PMI of the numeric pairs and
  triple, frequencies given Contract / InternetService / PaymentMethod, counts, KD-tree ball counts and
  nearest-neighbour distances, group means and a smoothed churn rate per categorical value) and 50 native
  categoricals. The lookup recipe adds original churn rates, percentiles / z-scores against original churners and
  non-churners, nearest original values, churn rates and distances of the nearest original rows, and original drift /
  priors of rounded charges. Depth 8, lr 0.03, Bayesian bagging.
- `train_fold_artifacts.py` (2 recipes): the same scheme with string-precision artifacts (string lengths, more than 2
  decimals, integer flags), the TotalCharges / (tenure x MonthlyCharges) ratio, fold frequencies of the decimal
  strings and 40 native categoricals; the lookup recipe adds the match of the 16-categorical profile with the original
  rows and the churn rate of the nearest original rows.
- `train_ranked_orig_rows.py`: a compact frame (73 columns: numerics and log1p, charge deviation and a 3-level
  deviation flag, value counts, mod 100, raw / exact / integer / even-floor strings, the exact
  tenure_MonthlyCharges_TotalCharges triple, 6 bigrams) with the original rows appended; per fold a 300-iteration
  depth-4 CatBoost orders the features by importance and the top 150 are kept (all of them). Depth 6, Bernoulli 0.8,
  max_ctr_complexity 1.
- `train_orig_knn.py`: the raw columns plus the churn rate of the 5 nearest original rows and the distances to the
  nearest original churner / non-churner (cKDTree on standardized numerics). Depth 6, lr 0.03, 1000 iterations.
- `train_snap_radix_nested_te.py`: a 1,770-column union (charges snapped to the nearest original value, decimal
  digits, frequencies / ranks, transforms, charge relations, flags, bins, pair / triple keys, quantile / rounded
  categories, digit positions, radix keys of snapped charges / tenure x the categoricals, counts, original priors of
  every shared column and of 16 pairs, drift) plus per-fold nested target encodings of 19 columns. Depth 7, lr 0.03,
  l2 8, Bernoulli 0.8, Cosine score function.
- `train_native_cats.py` (2 recipes): the 19 raw columns and a missing flag with native categoricals and Balanced
  class weights; the second recipe adds per-fold frequency and target mean / count encodings of all columns and 20
  quantile bins of the numerics. Depth 8, lr 0.03.
- `train_orig_dist_te_stats.py`: an all-numeric frame: frequencies, original churn rate of every value, TotalCharges
  percentiles / z-scores / quartile distances against the original churners and non-churners, digit features, and
  per-fold inner target std / min / max, TargetEncoder means and n-gram target means. Depth 5, Balanced.
- `train_all_categorical.py`: the 3 numerics plus 21 categoricals (label-encoded raw columns, original-value bins of
  the numerics, first decimal digits of the charges). Depth 6, lr 0.03, l2 8.
- `train_snap_combo.py` (feature frames in `lib/snap_combo_frames.py`): the snap / decimal-string columns of
  `models/snap_features` plus bins, pair / triple keys, 32 n-grams, counts, group means, original priors, original
  centroid distances and train-vs-test frequency ratios (498 features, 315 native categoricals). Depth 8, lr 0.02,
  Bernoulli 0.85, max_ctr_complexity 4.

```
python train_numcat_artifacts.py catboost_numcat_digits_orig_priors_drift
python train_numcat_artifacts.py catboost_numcat_digits_orig_rows
python train_fold_density.py catboost_fold_density_knn_orig_priors
python train_fold_density.py catboost_fold_density_knn_orig_rows
python train_fold_artifacts.py catboost_fold_artifacts_orig_priors
python train_fold_artifacts.py catboost_fold_artifacts_orig_rows
python train_ranked_orig_rows.py
python train_orig_knn.py
python train_snap_radix_nested_te.py
python train_native_cats.py catboost_native_cats_balanced
python train_native_cats.py catboost_native_cats_fold_stats_balanced
python train_orig_dist_te_stats.py
python train_all_categorical.py
python train_snap_combo.py
```
`train_snap_combo.py` needs `models/snap_features` to have run first (it reads its `combo.csv`); no other script
needs another folder. Each script runs all its folds in one process. CatBoost trains on every visible GPU, so pick
the GPUs with `CUDA_VISIBLE_DEVICES`: the all-categorical model was trained with two GPUs visible, every other model
with one. A fold takes from under a minute (`train_orig_knn.py`) to about an hour (`train_snap_radix_nested_te.py`).

| Pair | Model | OOF AUC |
|---|---|---:|
| catboost_numcat_digits_orig_priors_drift | numeric transforms, digit / count artifacts, 65 native categoricals, original priors and drift | 0.918957 |
| catboost_numcat_digits_orig_rows | same frame without original lookups, original rows appended | 0.919017 |
| catboost_fold_density_knn_orig_priors | string artifacts, fold-local frequency / PMI / KD-tree density / group statistics, original priors and neighbours | 0.918762 |
| catboost_fold_density_knn_orig_rows | same without original lookups, original rows appended (fold statistics include them) | 0.918723 |
| catboost_fold_artifacts_orig_priors | string-precision artifacts, fold frequencies of decimal strings, original profile match and neighbours | 0.918544 |
| catboost_fold_artifacts_orig_rows | same without original lookups, original rows appended | 0.918565 |
| catboost_triple_cross_ranked_orig_rows | compact frame with the exact numeric triple category, importance-ordered, original rows, depth 6 | 0.918874 |
| catboost_orig_knn_class_distances | raw columns + nearest-original churn rate and churner / non-churner distances | 0.915216 |
| catboost_snap_radix_union_nested_te | 1,770-column snap / radix / bin / count / original-prior union + nested target encodings | 0.918133 |
| catboost_native_cats_balanced | raw columns, native categoricals, Balanced class weights | 0.915667 |
| catboost_native_cats_fold_stats_balanced | + fold frequency / target encodings and quantile-bin categoricals | 0.915854 |
| catboost_orig_dist_te_stats_ngrams_balanced | all-numeric original probabilities / distributions / digits + inner target statistics, depth 5, Balanced | 0.918707 |
| catboost_all_categorical_orig_bins_d1 | label-encoded columns, original-value bins and first decimal digits as categoricals | 0.917963 |
| catboost_snap_combo_native_cats | snap combo frame with keys, n-grams, counts and original priors, max_ctr_complexity 4 | 0.918268 |

Credit: `train_orig_dist_te_stats.py` is adapted from the CatBoost pipeline of Artem (artemevstafyev),
https://www.kaggle.com/code/artemevstafyev/xgb-cb-best-cv-auc-0-91930-blend (blended in his public notebook
https://www.kaggle.com/code/artemevstafyev/cv-auc-0-91930-xgb-cb-blend).

Requirements: catboost 1.2 (GPU), numpy 2.x, pandas 2.x (several scripts detect string columns by their object dtype),
scikit-learn >= 1.3 (`TargetEncoder`), scipy.
