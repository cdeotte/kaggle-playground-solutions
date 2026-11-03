# CatBoost: artifact categories, original-data priors and native categorical crosses

Twelve GPU CatBoost models: four recipes on one feature harness (`train.py`, `lib/cat_harness.py`), two on a per-fold
feature builder (`train_fold_builder.py`, `lib/fold_builder.py`) and six self-contained pipelines (one script each).
Every model uses the shared 5 folds (`common.data.folds`), early-stops on the validation fold with `use_best_model=True`
and averages the 5 fold models on the test rows (probabilities).

**Harness models** (`train.py`). `lib/cat_harness.py` builds one label-free frame each for the train, test and
original rows; counts, group and sequence statistics are taken over train + test + original:
- raw columns and an `IsOrig` flag; domain features (estimated total laps = lap / progress, laps remaining, tyre-age
  ratios, lap minus tyre life, degradation and lap-time delta per tyre lap, absolute values, tyre x progress,
  position x progress, stint x tyre life), race-phase / tyre-life / position bins, 9 crosses (Race x Year,
  Race x Compound x Stint, Driver x Compound, ...), the tyre-age ratio (3 decimals) and total laps (1 decimal) as
  strings;
- categories of the 7 low-cardinality numerics; artifact categories (lap time rounded to 0-2 decimals, delta and
  degradation to 0-3, progress and tyre-age ratio to 2-4, degradation per tyre lap and |delta| to 1-3, floors,
  lap x tyre life, tyre life x position change); 3 integer digits of 19 numerics and 2 decimal digits of the float
  ones; the 4 lowest digits of 100 x value of 10 numerics (categories);
- count / frequency of 12 keys and of the lap-time / degradation artifact categories; mean, std and deviation of
  delta, position change, progress and tyre life per Race x Year, Race x Compound x Stint, Driver x Race;
- optional blocks: previous / next row values, differences and rolling means of the previous 3 / 5 rows of 8 columns
  within Year x Race x Driver (x Stint) sorted by lap (`sequence`); 28 ratio / phase / compound-life formulas of the
  public notebooks; floor categories, per-frame counts and 200 progress quantile bins (`pipeline A`); within-group
  percentile rank, deviation / ratio to the median and robust z of 6 numerics in 4 Year / Race / Stint / Compound
  groups (+ 10-bin rank categories); tyre life as a fraction of a nominal compound life, laps to it, a stress score
  and their bins, lap / tyre / progress / position / delta windows and 19 crosses of them with Driver / Race /
  Compound / Year / Stint / pit stops (`hazard, pit-state, strategy`); rounded / 20-quantile / sign categories of 12
  numerics and 23 more crosses (`native combos`); counts in train, test and original separately and their ratios
  (`per-source counts`); group statistics also per Compound x Stint, Race x Compound, Driver x Compound and of 5 more
  numerics (`wide groups`); min / max / mean / std of tyre life, lap, progress and two pressures in 4 groups and the
  row's distances to them (`stint end points`).

Strings are filled with their mode and numerics with their median (train + test + original); string columns are
CatBoost-native categoricals. Per outer fold, original-label priors `OP_<key>_mean` (smoothed toward the original
mean with weight `prior_smooth`) and `OP_<key>_count` of all original rows are mapped onto every frame, and all
101,371 original rows are appended to the training side with sample weight `original_weight`. CatBoost (GPU, Bayesian
bootstrap): seed 42 + fold number (1-5) + offset; fold predictions clipped to [1e-7, 1 - 1e-7].

| Recipe | Blocks | Priors (smoothing) | Original weight | CatBoost |
|---|---|---|---:|---|
| catboost_artifact_cats_priors_depth9 | base (309 columns, 96 categorical) | 6 keys (25) | 1.0 | depth 9, lr 0.025, l2 10, rs 1.2, bt 0.7, 5000 it, ES 250, Balanced |
| catboost_artifact_cats_next_lap_sequence | + sequence (357 / 96) | 6 keys (25) | 1.0 | depth 8, lr 0.03, l2 7, rs 0.8, bt 0.55, 5000 it, ES 250 |
| catboost_pit_state_cats_original_w115 | + formulas, pipeline A, ranks, hazard / pit-state / strategy (541 / 141) | 6 keys (75) | 1.15 | depth 9, lr 0.02, l2 14, rs 1.35, bt 0.75, 6500 it, ES 350, Balanced |
| catboost_native_combos_sequence_rich_priors | all blocks, sequence with Stint (1562 / 244) | 37 keys (55) | 0.9 | depth 9, lr 0.02, l2 14, rs 1.35, bt 0.75, one_hot_max_size 12, max_ctr_complexity 4, 7600 it, ES 400, Balanced, seed offset 240 |

**Fold-builder models** (`train_fold_builder.py`). Per outer fold k the original subset is the original rows outside
fold k of the same 5-fold split of the original data (`common.data.orig_folds`). `lib/fold_builder.py` is fit on the
fold's training rows + the original subset + the test rows: medians, low-cardinality numerics as categories, tyre /
lap / progress / position bins, rounded (1-3 decimals), fractional-part and 50 / 100-quantile categories of lap time,
delta, degradation and progress, 10 crosses (with the pair grid: all 36 pairs of 9 keys), frequency of every category,
mean / std / count / residual of 7 numerics per 9 anchor keys, original priors (smoothing 40) and log counts of 16 keys
from the original subset, and a source flag. The original subset is appended to the training side. CatBoost (GPU):
Bayesian bootstrap, border_count 254, seed 42 + fold number (1-5); predictions clipped to [1e-7, 1 - 1e-7].

| Recipe | Pair grid | Original weight | CatBoost |
|---|---|---:|---|
| catboost_pair_grid_original_subset_w07 | yes (475 columns, 84 categorical) | 0.7 | depth 8, lr 0.024, l2 10, rs 0.75, bt 0.85, 15000 it, od_wait 750 |
| catboost_anchor_priors_original_subset_w085 | no (403 / 48) | 0.85 | depth 9, lr 0.028, l2 12, rs 0.8, bt 0.75, 13000 it, od_wait 650 |

**Converted pipelines** (one script each; details in each script's docstring). GPU CatBoost; predictions clipped to
[1e-7, 1 - 1e-7] except in `train_dense_priors.py`:
- `train_numeric_artifacts.py`: numeric transforms, pairwise NUMS arithmetic, counts and digit artifacts of every
  column, and native categories of the exact / rounded / floor / 50-quantile values, digit pairs and n-gram crosses of
  the 6 highest-cardinality keys (460 columns, 120 categorical); all original rows appended. Depth 8, lr 0.03, l2 3,
  random_strength 0.5, Bayesian bagging_temperature 0.7, max_ctr_complexity 2, 5000 it, ES 300, seed 42.
- `train_fold_density.py`: row features and native categories plus fold-local features measured against the fold's
  training rows + all original rows (frequency / count / rank, pair PMIs, exact-triple counts, conditional
  frequencies, ball counts and neighbour distances in a standardized 4-numeric space, group means; 417 columns); all
  original rows appended. Same CatBoost parameters as above, seed 42.
- `train_feature_select.py`: 169 transform / digit / native-category / original-prior columns; per fold a
  300-iteration depth-4 CatBoost (lr 0.06) keeps the 150 most important; all original rows appended. Depth 6, lr 0.03,
  l2 3, Bernoulli subsample 0.8, max_ctr_complexity 1, 5000 it, ES 300, seed 42.
- `train_dense_priors.py`: relationships, transforms, digit categories, 12-quantile bins and crosses, counts,
  train-vs-original drift, original-label priors and original class-centroid distances (514 columns, 210
  categorical); original rows are not appended. Depth 8, lr 0.02, l2 6, min_data_in_leaf 20, Bernoulli subsample
  0.85, random_strength 0.6, max_ctr_complexity 4, up to 50,000 it, ES 600, seed 42 + fold number (1-5).
- `train_nearest_original.py`: label mean / std and distance of the 1 / 3 / 5 / 10 nearest original rows (KD-tree on
  20 standardized numerics), original-label priors of 10 keys and 6 crosses, frequency / original-rank features (90
  columns); original rows are not appended. Depth 6, lr 0.03, 1000 it, ES 50, seed 42 + fold number (1-5).
- `train_original_bins.py`: 4 continuous numerics; every other column, original-value bins (2048 / 512 equal-count
  chunks of the original unique values) and first decimal digits as label-encoded categories (31 columns); original
  rows are not appended. Depth 6, lr 0.03, l2 8, up to 50,000 it, ES 200, seed 42.

```
python train.py catboost_artifact_cats_priors_depth9
python train.py catboost_artifact_cats_next_lap_sequence
python train.py catboost_pit_state_cats_original_w115
python train.py catboost_native_combos_sequence_rich_priors
python train_fold_builder.py catboost_pair_grid_original_subset_w07
python train_fold_builder.py catboost_anchor_priors_original_subset_w085
python train_numeric_artifacts.py
python train_fold_density.py
python train_feature_select.py
python train_dense_priors.py
python train_nearest_original.py
python train_original_bins.py
```
No other folder has to run first. Every script takes `--folds 0,1,2,3,4` (and `--force` to redo finished folds);
finished folds are kept in `parts/` and `preds/<pair>_{oof,test}.npy` is written once all 5 folds exist, so folds can
run in separate processes (one GPU each, e.g. with `CUDA_VISIBLE_DEVICES`). One fold takes 1-60 minutes on one GPU
(the 1562-column harness recipe and the pair-grid builder are the slowest).

| Pair | Model | OOF AUC |
|---|---|---:|
| catboost_artifact_cats_priors_depth9 | harness base frame, 6 original priors, all original rows, depth 9, Balanced | 0.953235 |
| catboost_artifact_cats_next_lap_sequence | + previous / next row sequence features, depth 8, no class weights | 0.952805 |
| catboost_pit_state_cats_original_w115 | + formulas, ranks, hazard / pit-state / strategy crosses, original weight 1.15, depth 9, Balanced | 0.953404 |
| catboost_native_combos_sequence_rich_priors | all harness blocks, 37 original priors, original weight 0.9, max_ctr_complexity 4 | 0.952457 |
| catboost_pair_grid_original_subset_w07 | fold builder with the 36-pair grid, fold-matched original subset (weight 0.7), depth 8 | 0.952226 |
| catboost_anchor_priors_original_subset_w085 | fold builder without the pair grid, fold-matched original subset (weight 0.85), depth 9 | 0.952044 |
| catboost_numeric_artifacts_numcat | numeric transforms / digit artifacts + native value categories and n-grams, all original rows | 0.950532 |
| catboost_fold_density_knn | fold-local frequency / PMI / KNN-density features + native categories, all original rows | 0.951296 |
| catboost_top150_select_depth6 | fold-local top-150 feature selection, depth 6 Bernoulli, max_ctr_complexity 1 | 0.950348 |
| catboost_dense_priors_crosses | dense original priors, drift, class centroids, 16 crosses, max_ctr_complexity 4, no original rows | 0.950737 |
| catboost_nearest_original_targets | nearest-original-row label features + original priors, depth 6, 1000 iterations | 0.945827 |
| catboost_original_value_bins | 4 numerics + label-encoded categories and original-value bins, depth 6 | 0.949583 |

Requirements: catboost (GPU), pandas 2.x (the scripts detect string columns by their object dtype), numpy, scipy,
scikit-learn.
