# cuML models: linear, SVM, landmark, UMAP, random-forest and cluster-posterior pipelines

Six RAPIDS cuML pipelines on one shared feature matrix without competition labels plus a small per-fold response bank.
Every fitted state (scalers, medians, PCA, KMeans, UMAP, bandwidths, models) uses the outer-train rows of the fold only.

**Base matrix** (`build_features.py`, 240 columns, no competition labels): the 9 raw numerics; 9 missing flags and 32
one-hots of the screen-field missingness mask; grid artifacts of the 6 hour fields (`round(100 x) mod 100`, distances to the 0.05
and 0.10 grids); log train + test frequency of every field's value (numerics at 2 decimals); 11 categorical one-hots;
18 derived columns (sums / shares / spread of social, gaming and work/study vs daily, weekend minus / over daily, log
notifications and app opens, notifications per open, missing screen fields); per numeric the CDF in the original data,
the distance to the nearest original value and the log count of exact original matches; and from `../xgb_forensic`
17 original-neighbor (KNN) features, 15 original-neighbor imputations of the screen fields and 72 observed-pair joint
log-count / PMI columns.

**Response bank** (per outer fold, 37 columns): smoothed (m 20) target logits of the 12 raw fields (numeric tokens at 2
decimals, missing as a token) — exact leave-one-row-out for outer-train rows, the full outer-train map for held-out and
test rows — their 12 log counts, 3 aggregates (sum of numeric WoE, sum of screen WoE, mean logit) and the 10 pair-lift
columns of `../pairlift_models` (5-fold inner cross-fit inside the outer-train rows).

## Pipelines

| Pair | Pipeline |
|---|---|
| cuml_original_knn_lr | base without the observed-joint block (168) + response, robust-scaled (median / IQR of 120,000 evenly spaced outer-train rows, clip ±12) -> logistic regression (qn, L2, C 0.12, 1200 iterations) |
| cuml_residual_pca_svm | robust-scaled base -> whitened PCA-64; + robust-scaled response -> LinearSVC (squared hinge, C 0.035, 1500 iterations); probability = sigmoid(decision function) |
| cuml_landmark_rbf_lr | whitened PCA-40 -> 256 KMeans landmarks (180,000 rows, seed 2600 + fold), RBF features at bandwidths 0.65 / 1 / 1.75 x the median nearest-landmark distance; [PCA, RBF, response] -> LR C 0.08 |
| cuml_umap_lr | UMAP-16 fitted on 120,000 evenly spaced outer-train rows (40 neighbors, 240 epochs, min_dist 0.08, nn-descent, seed 3600 + fold), applied to all rows, robust-scaled; + response -> LR C 0.16 |
| cuml_artifact_rf | median-filled 240 base columns, no response bank -> random forest (900 trees, depth 18, max_features 0.55, min leaf 18, min split 36, 256 bins, bootstrap 0.78, seed 4600 + fold) |
| cuml_cluster_posterior_lr | whitened PCA-32 -> KMeans with 64 / 256 / 1024 clusters (220,000 rows, seed 5600 + 10 fold + k); per resolution the cluster's smoothed (m 25) label logit (leave-one-row-out for outer-train rows), log size and RMS distance to the center, robust-scaled; [PCA, clusters, response] -> LR C 0.12 |

Each fold model predicts the held-out fold and the test rows; test = mean of the 5 fold models.

## Inputs from other folders
- `../xgb_forensic`: `cache/source_knn_v1/{train,test}.npy`, `cache/source_knn_impute_v1/{train,test}.npy` (names in
  their `manifest.json`) and `cache/observed_joint_cache_v1/{train,test}_features.npy` (`build_observed_joint.py`).
- `../pairlift_models`: `cache/pairlift/fold_<k>/{train,valid,test}_features.npy` (`build_pairlift.py`).

The paths are in `lib/inputs.py`.

## Steps
```
python build_features.py                  # cache/base_{train,test}.npy, base_columns.json, cache/response/fold_<k>/
python train.py original_knn_lr           # -> preds/cuml_original_knn_lr_{oof,test}.npy
python train.py residual_pca_svm
python train.py landmark_rbf_lr
python train.py umap_lr
python train.py artifact_rf
python train.py cluster_posterior_lr
```
`--folds 0,1` runs a subset; finished folds are kept in `parts/` and skipped. On one A100 a fold takes 8-30 s (1-1.5 min
for the random forest).

## Pairs

| Pair | Model | OOF AUC |
|---|---|---:|
| cuml_original_knn_lr | original-neighbor geometry + response bank -> logistic regression | 0.961365 |
| cuml_residual_pca_svm | whitened PCA-64 + response bank -> linear SVM | 0.917456 |
| cuml_landmark_rbf_lr | PCA-40 + 256 KMeans landmarks x 3 RBF maps + response bank -> logistic regression | 0.952970 |
| cuml_umap_lr | UMAP-16 + response bank -> logistic regression | 0.955080 |
| cuml_artifact_rf | 240 label-free source / generator-artifact columns -> 900-tree random forest | 0.947677 |
| cuml_cluster_posterior_lr | PCA-32 + KMeans 64 / 256 / 1024 cluster posteriors + response bank -> logistic regression | 0.953891 |

Packages: RAPIDS cuML and CuPy (our run: cuML 26.02), numpy, pandas, scipy, scikit-learn.

## Notes
- The random forest uses only the label-free base matrix (no response bank); the other five use both.
- cuML's multi-stream random forest, UMAP and qn solver are not bit-exact between runs.
