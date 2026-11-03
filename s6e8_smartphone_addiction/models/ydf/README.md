# YDF gradient boosted trees with Newton (hessian) split gain

**Inputs**: the 253-column `mlm3` matrix of `../dcn/build_cache.py` (base geometry, original-data channels, PMIs, imputer
fills, nested TEs as logits with smoothing 1 averaged over 3 inner seeds, a second TE with smoothing 20, WOE sums, smooth
fields, MLM completion / density / fills / reconstruction residuals and the 10 pair-lift channels; outer-train medians for
missing values) plus the 9 coarse-lattice pair-lift channels `cbank_f{k}` of `../lift_banks` = 262 columns. No token
columns.

**Model** (`train.py`): `ydf.GradientBoostedTreesLearner`, axis-aligned splits, `use_hessian_gain=True`, shrinkage 0.03,
up to 7,000 trees with early stopping (look-ahead 100) on a stratified 1/7 of the outer-train rows (StratifiedKFold(7,
seed 777), first split; trained on the other 6/7), min_examples 20, l2_regularization 4, num_candidate_attributes_ratio
0.3, max_depth 6 (default), seed 42, 8 threads. 2,300-3,300 trees are kept per fold; no refit. One model per fold; test =
mean of the 5 fold models (probabilities).

```
python ../dcn/build_cache.py mlm3      # once; writes ../dcn/cache/
python train.py ydf_hessian_te_mlm3_cbank
```
Run `../density_features` and `../lift_banks` first (for `../dcn/build_cache.py` and `cbank_f*`). CPU only, about 30
minutes per fold with 8 threads; `--folds 0,1` runs a subset and `--threads` sets the thread count; the pair is written
when the last fold is done.

| Pair | Model | OOF AUC |
|---|---|---:|
| ydf_hessian_te_mlm3_cbank | YDF GBT, hessian gain, shrinkage 0.03, l2 4, 30% candidate attributes, on the 253-column mlm3 matrix + cbank (262 columns) | 0.969298 |

Requirements: ydf, numpy, pandas, pyarrow, scikit-learn.
