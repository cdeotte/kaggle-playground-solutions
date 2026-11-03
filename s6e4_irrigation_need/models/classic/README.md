# Classic models: random forests, KNN stacks, logistic regressions, pair target encodings, YDF

Ten models outside the big GBDT / NN families. Stack A and stack B are the two stacks of `stack/`; the table says which
stack each pair belongs to. Each script writes `preds/<pair>_{oof,test}.npy` (probabilities of Low, Medium, High) on the
shared 5 folds; test = mean of the 5 fold models.

**Random forests** (`train_rf.py`, 2 presets): cuML `RandomForestClassifier` (1,500 trees, max_depth 16, max_features
0.3, min_samples_leaf 15, random_state 42) on the raw columns (including `id`) plus original-data "snap" features, and
per fold every categorical replaced by 3 one-vs-rest target encodings (cuML `TargetEncoder`, 5 inner folds, smoothing
20).
- `rf_water_snap_digit_te_a`: Rainfall_mm and Previous_Irrigation_mm snapped to the nearest original value and value -
  snap, their first / second decimal digits and a round-value flag, water balance (rain + previous irrigation, moisture
  minus that, per hectare), a managed-water count (mulch, drip / sprinkler, non-rain water source), bigrams Crop x Season
  and Irrigation x Water source.
- `rf_snap_digit_formula_te_b`: all 11 numerics snapped, value - snap and first decimal digit; a threshold score
  2 [moisture < 25] + 2 [rainfall < 300] + [temperature > 30] + [wind > 10] - 2 [Harvest] - 2 [Sowing] - [mulch] with
  the flags score <= 0 and score > 3; 4 bigrams (Crop x Season, Stage x Mulch, Irrigation x Water source, Soil x Region).

**KNN stacks on the original data** (`train_knn_lr_xgb.py`, presets k 10 and k 1): GPU brute-force KNN of every train /
test row among the 10,000 original rows under 30 "random" column-weight schemes (categorical mismatches + squared
standardized numeric differences) and 30 "smart" schemes (weights from an XGBoost fitted on the original rows;
categoricals replaced by their original class rates; inverse-distance weights, train / original prior ratio). 8 + 8
schemes are drawn and give Dirichlet-weighted class-share / distance means (12 columns); per outer fold, multinomial
logistic regressions (C 0.2) on the schemes' logits give out-of-fold meta features (12 columns). Final model: XGBoost
(GPU, lr 0.018, depth 4, colsample_bytree 0.32, early stopping 250 on the fold's validation rows) on these 24 columns
+ the 19 raw columns as codes / numbers.

**Nested KNN stack** (`train_knn_nested_lr_xgb.py`): the same idea with 10 scheme generators / templates and one-hot
neighbour labels; original-data KNN (k 10) plus, per outer fold, train-data KNN (k 40, neighbours = the outer-training
rows) for the held-out and test rows and, for the outer-training rows, out-of-fold over 5 inner folds; logistic
regressions on both KNN blocks (fitted on the outer-training rows); XGBoost (GPU, native categoricals, balanced sample
weights, early stopping 250) on 22 base columns (raw + threshold score) + 30 stack columns.

**Logistic regression** (`train_logreg.py`): 4,518 one-hot columns (the 8 categoricals, Season x Irrigation_Type,
128-512 and 16 bins of every numeric over the sorted unique original values, first decimal digit of 5 numerics, 20
categorical / bin crosses; levels over train + test + original) + 174 dense columns standardized per fold (numerics,
15 interactions, 16 category-gated numerics, value - mean, value / mean and z-score within Season, Irrigation_Type,
Water_Source and Season x Irrigation_Type). Model: multinomial logistic regression in PyTorch (full batch, Adam lr
0.05, 80 epochs, class-weighted cross-entropy, L2 1 / (0.3 n)). `--cuml` tries cuML's `LogisticRegression` first; in
our run it raised a CUDA error on the fold matrices in every fold, so the pair comes from the PyTorch model (default).

**Pair target encodings + logistic regression** (`train_pair_te_lr.py`, 3 presets): all 171 pairs of the 19 columns,
per pair and class the cuML `TargetEncoder` mean of the pair's value combination (no smoothing, inner 5-fold values for
the outer-training rows) = 513 columns -> logit z -> [z, z^2] or [z, z^2, z^3], optionally + 9 threshold-score
columns, standardized -> multinomial logistic regression in PyTorch (full batch, Adam lr 0.01, 1,000 epochs,
class-balanced sample weights, L2 1 / (0.5 n)). The presets differ in the expansion, the extra columns and the inner
TE split (random with seeds per fold / class for stack A, interleaved for stack B); details in the script header.

**YDF** (`train_ydf.py`): the 19 raw columns, `ydf.GradientBoostedTreesLearner` with shrinkage 0.1, max_depth 2,
BEST_FIRST_GLOBAL growth, RANDOM categorical splits, up to 10,000 trees with early stopping (look-ahead 300) on the
fold's validation rows. CPU.

```
python train_rf.py rf_water_snap_digit_te_a
python train_rf.py rf_snap_digit_formula_te_b
python train_knn_lr_xgb.py xgb_original_knn_k10_lr_meta_a
python train_knn_lr_xgb.py xgb_original_knn_k1_lr_meta_a
python train_knn_nested_lr_xgb.py
python train_logreg.py
python train_pair_te_lr.py lr_pair_te_logit3_a
python train_pair_te_lr.py lr_pair_te_logit2_formula_a
python train_pair_te_lr.py lr_pair_te_logit3_formula_b
python train_ydf.py
```
No script needs another folder. Every script takes `--folds 0,1` to run a subset (finished folds are kept in `parts/`;
the pair is written when the last fold is done); the KNN scripts rebuild their original-data KNN features in every run.
Times in our runs (one GPU): nested KNN stack about 6 minutes of original-data KNN, then about 1.5 hours per outer
fold; KNN stacks on the original data about 1 minute per XGBoost fold; pair target encodings about 2 minutes per fold;
logistic regression about 2 minutes of features and under 30 s per fold. The logistic regression keeps dense float32
one-hot matrices: plan for about 40 GB of RAM and about 17 GB of GPU memory per fold.

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| rf_water_snap_digit_te_a | A | cuML random forest, rainfall / irrigation snap + digits + water balance, per-class TEs | 0.959458 |
| rf_snap_digit_formula_te_b | B | cuML random forest, snap + digits of all numerics, threshold score, per-class TEs | 0.960169 |
| xgb_original_knn_k10_lr_meta_a | A | XGBoost on original-data KNN (k 10) + LR meta features | 0.961572 |
| xgb_original_knn_k1_lr_meta_a | A | XGBoost on original-data KNN (k 1) + LR meta features | 0.961310 |
| xgb_nested_knn_lr_meta_b | B | XGBoost on original-data KNN + nested train-data KNN (k 40) + LR meta features | 0.970090 |
| logreg_ohe_crosses_bins_group_dense_a | A | logistic regression, 4,518 one-hot + 174 dense columns | 0.968458 |
| lr_pair_te_logit3_a | A | 171 pair TEs -> [z, z^2, z^3] -> logistic regression | 0.957787 |
| lr_pair_te_logit2_formula_a | A | 171 pair TEs -> [z, z^2] + threshold score -> logistic regression | 0.975519 |
| lr_pair_te_logit3_formula_b | B | 171 pair TEs -> [z, z^2, z^3] + threshold score -> logistic regression | 0.975785 |
| ydf_gbt_depth2_raw_a | A | YDF GBT, depth 2, raw columns | 0.969607 |

Packages: RAPIDS cuML / cuDF / CuPy (26.02 in our runs; random forests, pair target encodings, optional cuML logistic
regression), PyTorch (GPU KNN, logistic regressions), xgboost (CUDA; 3.x in our runs), ydf (0.16.1), scikit-learn (1.6.1;
the KNN scripts pass `multi_class='multinomial'` to `LogisticRegression`, which newer releases no longer accept),
pandas (2.3.3), numpy.
