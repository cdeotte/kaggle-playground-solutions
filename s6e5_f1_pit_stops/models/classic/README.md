# Classic models: random forest, hazard XGBoost, KNN stack, YDF, logistic regression, snap / radix XGBoost

Seven models outside the big GBDT / NN families. Each script writes `preds/<pair>_{oof,test}.npy` (probabilities) on
the shared 5 folds; test = mean of the 5 fold models.

**Random forest** (`train_rf.py`, a config on the shared harness `../neural_nets/lib/shared_harness.py`): features built
once on train + test + original (encoders fitted on train): the 14 raw columns, 16 domain ratios / pressures (estimated
total laps, laps remaining, tyre-age ratio, degradation and lap-time-delta per tyre lap, pit-window / position / stint
pressure, ...) and race-phase / tyre-age / position bins, LapNumber / RaceProgress and TyreLife / LapNumber, floor-value
codes of every numeric column, count encodings, a 200-bin RaceProgress quantile code, Race x Compound and Race x Year,
and group statistics (count, value - group mean, z-score of 4 numerics within Race, Compound, Driver, Race x Compound).
Per fold the fold-matched 4/5 of the original rows are appended to the training rows, the two combos are
target-encoded (sklearn `TargetEncoder`, 5 inner folds), string / low-cardinality columns become vocabulary codes and
the rest is standardized (113 columns). cuML `RandomForestClassifier`: 950 trees, max_depth 20, max_features 0.45,
min_samples_leaf 8, 128 bins, random_state 42 + fold.

**Hazard / frailty XGBoost** (`train_hazard_xgb.py`): a survival view of "pit next lap". Per fold, on the fold's train
rows + the fold-matched 4/5 of the original rows (sample weights 1.0 / 0.75), the script fits a piecewise-constant
hazard over 11 TyreLife bins (smoothing 45), per-compound hazards, the implied survival curve and its velocity, and
smoothed group "frailties" (Race x Compound, Driver x Compound, Compound x Stint, Race x Driver, tyre-life bin, race
phase) with their ratio / difference to the baseline hazard; plus 32 numeric state / domain features, codes, log counts
and CDF ranks, original-data priors / counts / train-vs-original drift of 31 keys (smoothing 25), and 210 pairwise
interactions of 15 categoricals and 12-quantile bins of 6 numerics, target-encoded (`TargetEncoder`, 5 inner folds) with
their row mean / std / min / max: 472 columns. XGBoost (GPU) lr 0.005, depth 7, min_child_weight 6, subsample 0.9,
colsample_bytree 0.35, lambda 5, alpha 0.05, max_bin 1024, early stopping 500 on the fold's validation rows (best
iterations 3,800-10,700), seed 42 + fold.

**KNN stack** (`train_knn_stack_xgb.py`): a 64-column base matrix (11 numerics with log1p, 2-decimal rounding and the
first 3 decimals, 10 state ratios, codes of 10 categoricals); original-data priors; original-KNN priors (cKDTree on the
standardized original rows: mean original target of the 1 / 3 / 5 / 10 / 20 nearest neighbours and mean / min distance
under 5 column-weight schemes); per fold, train-KNN features (k = 5 / 20 / 40, 3 schemes; neighbours among the
outer-training rows, inner-out-of-fold with 5 inner folds for the training rows themselves), logistic-regression meta
features on the KNN logits (C 0.2) and a final XGBoost (GPU) on the 138 columns: lr 0.018, depth 4, min_child_weight 8,
subsample 0.82, colsample_bytree 0.32, alpha 1, lambda 8, early stopping 250 on the fold's validation rows, seed 42.

**YDF** (`train_ydf.py`): the raw columns (string and < 100-value columns as categoricals, 4 continuous numerics),
`ydf.GradientBoostedTreesLearner` with shrinkage 0.1, max_depth 2, BEST_FIRST_GLOBAL growth, RANDOM categorical
splits, up to 10,000 trees with early stopping (look-ahead 300) on the fold's validation rows. CPU.

**Logistic regression** (`train_logreg.py`): cuML L2 logistic regression (C 0.3, max_iter 3000, tol 1e-6) on 167
standardized numerics (5 continuous columns, interactions, spline hinges at original-data quantiles, category-gated
numerics and hinges, and value - group mean, value / |group mean| and z-score of 7 numerics within Race, Compound,
PitStop, Race x Compound, Compound x Stint) and 8,143 one-hot columns (low-cardinality columns, bins built from the original values, first decimal
digits, 22 categorical crosses; levels with < 30 rows pooled). The original rows shape the features but are not
training rows.

**Snap / radix XGBoost** (`train_snap_radix_xgb.py`): each continuous value that does not occur in the original data is
snapped to the nearest original value; features are the snapped values and snap differences, decimal-digit categoricals,
"radix" integers round(1000 x snap) + 50,000,000 x (category code) for the 10 low-cardinality columns, and smoothed
original target means / counts. Shallow GPU XGBoost with native categoricals (max_bin 16,000, subsample /
colsample_bytree 0.85, early stopping 300 on the fold's validation rows, seed 42 + fold): depth 2 / lr 0.05, or stumps
(depth 1) / lr 0.1 with an extra radix block on LapNumber.

```
python train_rf.py
python train_hazard_xgb.py
python train_knn_stack_xgb.py
python train_ydf.py
python train_logreg.py
python train_snap_radix_xgb.py xgb_snap_radix_depth2
python train_snap_radix_xgb.py xgb_snap_radix_lapnumber_depth1
```
`train_rf.py` imports the harness of `../neural_nets` (keep the folders side by side); no other folder has to run
first. All scripts except `train_rf.py` take `--folds 0,1` to run a subset (finished folds are kept in `parts/`; the
pair is written when the last fold is done). Times in our runs: random forest about 10 minutes in total; hazard XGBoost
about 6 minutes per fold; KNN stack about 2 hours per fold (CPU KD-tree queries); logistic regression a few minutes for
the features and under a minute per fold, but the dense one-hot matrices alone take about 20 GB of RAM (and the fold
matrices about 20 GB of GPU memory); snap / radix 1.5-2 hours of CPU feature building (run all folds in one process),
then 1-2 minutes per fold.

| Pair | Model | OOF AUC |
|---|---|---:|
| rf_domain_group_stats_combo_te | cuML random forest, domain features + group stats + combo TEs, fold-matched original rows | 0.948845 |
| xgb_hazard_frailty_pair_te_original_w075 | XGBoost on tyre-life hazard / frailty, original priors and 210 pair TEs; original rows w 0.75 | 0.946209 |
| xgb_knn_stack_lr_meta | XGBoost on original-KNN and inner-OOF train-KNN target features + LR meta features | 0.947231 |
| ydf_gbt_depth2_raw | YDF GBT, depth 2, raw columns | 0.943788 |
| logreg_ohe_crosses_splines_group_dense | cuML logistic regression, 8,143 one-hot + 167 dense columns | 0.933935 |
| xgb_snap_radix_depth2 | XGBoost depth 2 on snap / digit / radix features | 0.932076 |
| xgb_snap_radix_lapnumber_depth1 | XGBoost stumps on snap / digit / radix features + LapNumber radix block | 0.921717 |

Packages: xgboost (CUDA; 3.x in our runs), RAPIDS cuML + cupy (random forest, logistic regression; 26.02 in our runs),
ydf (0.16.1), scikit-learn (1.6.1; `TargetEncoder` needs >= 1.3), scipy, pandas (2.3.3), numpy; the random forest
also needs what `../neural_nets/lib/shared_harness.py` imports (torch).
