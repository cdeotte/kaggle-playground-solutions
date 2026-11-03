# LightGBM: original-row weighting, original-data reference features and nested target encodings

Eight CPU LightGBM models. Five share one feature recipe (`lib/lgbm_harness.py`, run by `train.py`) and train on the
competition rows plus the weighted original rows; three are self-contained scripts that use the original rows only as a
reference for features.

**Shared recipe** (`train.py`, `lib/lgbm_harness.py`; label-free features built once on train + original + test rows):
- the 14 columns and `lap_tyre = LapNumber * TyreLife`;
- **domain** (11): LapNumber / RaceProgress (clipped to [1, 120]), TyreLife / LapNumber, LapNumber / (TyreLife + 1),
  degradation and lap-time delta per tyre lap (signed and absolute), |LapTime_Delta|, |Position_Change|,
  Position * RaceProgress, Stint * TyreLife;
- **interactions**: Race x Compound, Race x Year, Race x Compound x Stint, Driver x Race, Driver x Compound,
  Compound x Stint as categories;
- **counts**: Year, PitStop, LapNumber, Stint, TyreLife, Position, Position_Change as categories and the train + original +
  test row count of every category value;
- **group stats** (one model): LapTime (s), LapTime_Delta, Cumulative_Degradation, TyreLife minus the group mean and z-score
  within Race, Compound, Driver, Race x Compound;
- **target encodings** (one model, per fold): scikit-learn `TargetEncoder` (smooth 30) of the 16 category columns, out-of-fold
  (inner StratifiedKFold(5), seed 42 + 111 x fold number) on the training rows, fitted on all training rows for the validation
  and test rows.

Categories are label-encoded with their sorted values. Training rows of fold k: the competition rows of the other 4 folds
followed by all 101,371 original rows with sample weight 0.35-1.25. LightGBM (gbdt, 32 threads, bagging every 5 rounds)
with early stopping (300 rounds, AUC) on the validation fold; random_state 2026 + k. Test = mean of the 5 fold models.

**Self-contained scripts** (the original rows are a feature reference only):
- `train_snap_priors_te.py`: columns downcast to float32; values snapped to the nearest original value (snap, difference,
  decimal digits), frequencies / ranks, numeric transforms, tyre / lap-time / degradation relations, race-context flags,
  bins, quantile codes, rounded values, 9 crosses, category counts, original target means and counts of single values and
  5 pairs, train / original drift, all <= 20,000-level pairs of 21 key columns, 5000-quantile / integer keys of the main
  numerics (638 columns); every string key with > 7 levels (345) becomes a nested target-mean encoding per fold (inner 5
  folds, seed 42 + fold number), the rest are factorized. LightGBM lr 0.01, 63 leaves, depth 6, min_child_samples 200,
  subsample 0.72, colsample 0.22, alpha 8, lambda 14, max_bin 511, early stopping 400 (<= 12,000 rounds).
- `train_original_knn.py`: 10 race-state ratios; a cKDTree over the original rows (20 numerics standardized on the original
  rows) gives the mean / std of the 1, 3, 5, 10 nearest original targets and their mean distance; original target mean and
  count of every category, numeric-as-category and 6 crosses; frequency, rank among the original values, 2-decimal rounding
  and the first 3 decimals of the 4 continuous columns (90 features). LightGBM 600 rounds, lr 0.03, 31 leaves, depth 6.
- `train_fingerprint.py`: standardized numerics, ordinal categories, row statistics, value / pair / numeric-signature counts
  over train + test + original, distances to the 1-10 nearest original rows, MiniBatchKMeans(128) prototypes of the original
  rows (cluster id, size, distance, original target rate), smoothed target encodings over the same 5 folds of the cluster id
  and the 10 categoricals (117 features). LightGBM lr 0.03, 127 leaves, min_data_in_leaf 200, early stopping 200.

```
python train.py lgbm_domain_counts_te_original_w035
python train.py lgbm_domain_counts_group_stats_original_w085
python train.py lgbm_lean_original_w105
python train.py lgbm_lean_original_w125
python train.py lgbm_lean_original_w125_slow
python train_snap_priors_te.py
python train_original_knn.py
python train_fingerprint.py
```
`train.py` takes `--folds 0,1` to run a subset of folds (finished folds are kept in `parts/`; the pair is written when all 5
are done); each recipe takes 6-15 minutes for 5 folds on 32 CPU threads. The scripts run all folds: about 40, 11 and
6 minutes. Each run writes `preds/<pair>_{oof,test}.npy` (probabilities). Reads only `data/`; no other repo folder is
needed.

| Pair | Model | OOF AUC |
|---|---|---:|
| lgbm_domain_counts_te_original_w035 | domain + interactions + counts + target encodings (71 features); original weight 0.35; 96 leaves, ff 0.84, bagging 0.82, lambda_l2 0.9, min_child_samples 60, lr 0.018 | 0.951988 |
| lgbm_domain_counts_group_stats_original_w085 | domain + interactions + counts + group stats (87 features); original weight 0.85; 104 leaves, ff 0.84, bagging 0.82, lambda_l2 1.0, min_child_samples 60, lr 0.018 | 0.952319 |
| lgbm_lean_original_w105 | domain + interactions + counts (55 features); original weight 1.05; 104 leaves, ff 0.10, bagging 0.92, lambda_l1 0.1, lambda_l2 1.8, min_child_samples 80, lr 0.018 | 0.952938 |
| lgbm_lean_original_w125 | the lean model with original weight 1.25 | 0.952929 |
| lgbm_lean_original_w125_slow | the lean model with original weight 1.25, lr 0.012, <= 9,500 rounds | 0.953023 |
| lgbm_snap_priors_pairs_nested_te | snap / original-prior / pair-key union with nested target encodings (`train_snap_priors_te.py`) | 0.948752 |
| lgbm_original_knn_target | nearest-original-row target features + original priors (`train_original_knn.py`) | 0.946744 |
| lgbm_fingerprint_knn_kmeans | value fingerprints, original-row distances and k-means prototypes (`train_fingerprint.py`) | 0.950035 |

Requirements: lightgbm, scikit-learn (>= 1.3 for `TargetEncoder`), scipy, pandas, numpy (CPU only).
