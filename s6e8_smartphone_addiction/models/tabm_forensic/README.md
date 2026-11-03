# TabM on the forensic features

Two pytabkit TabM-D models on TabM-oriented row-local views plus the forensic columns of the `../lgbm_margin` fold
cache.

**Row-local views** (`lib/row_features.py`, label-free): `raw` (9 numerics, 3 categoricals), `exact_grid_onehots`
(exact-value tokens: hours x100, others x1, as categoricals), `smooth_latent_geometry` (14 coordinates from fixed
anchors: screen load, leisure vs productive, weekend escalation, recovery pressure, interruption load, log balances,
z-score products, budget slack / balance), `missing_gated_views` (missing counts / masks and an equal-slack completion
of missing screen components), `fixed_bins` (2-3 fixed-origin bin widths per numeric, as categoricals).
**Fold-fitted** (`lib/fold_features.py`): 12 pair contrasts `z(left) - z(right)` with outer-training medians / stds;
every numeric NaN filled with its outer-training median; categoricals as strings.
**Forensic columns**: the 104 `strong__` columns of the `../lgbm_margin` fold cache (the 102-column forensic matrix of
`../xgb_forensic` with nested TEs / WOE sums, imputed screens, screen-response logits, pair PMIs, original copula
ratios and unsupported-pair response, plus the base margin and its probability).

**TabM** (`lib/runner.py`): TabM-D, k = 32, 2 blocks of 512, d_embedding 16, piecewise-linear embeddings with 48 bins
after `quantile_tabr`, dropout 0.1, lr 0.002, batch 1024, AMP, gradient clipping 1.0, weight decay 0, at most 40 epochs,
patience 8 on 1 - AUC, `n_refit=0`. The `n_cv` internal members are each early-stopped on their own stratified holdout
of the outer-training rows (StratifiedShuffleSplit, seed 20260824 + fold); random_state 142 + fold. One model per
outer fold; test = mean of the 5 fold models.

- `tabm_forensic_strong` (`train_tabm_strong.py`): row-local views + strong columns + contrasts (185 features, 35
  categorical); n_cv 5, 5% holdouts.
- `tabm_forensic_imputed` (`train_tabm_imputed.py`): the same views of the raw rows, plus the raw / smooth latent /
  fixed-bin blocks of the completed rows (`nncomplete__`: missing numerics from the hybrid XGBoost + masked-autoencoder
  reconstruction of `../realmlp_repair`, missing categoricals from the mean of the two class-probability vectors), 12
  missing flags, the strong columns, the first 71 reconstruction diagnostics and the 30 density-completion response
  channels of `../realmlp_repair` (344 features, 58 categorical); n_cv 10, 2.5% holdouts.

```
python train_tabm_strong.py
python train_tabm_imputed.py
```
Run first: `../lgbm_margin/build_fold_features.py` (and everything it needs); for `train_tabm_imputed.py` also the
caches of `../realmlp_repair`: the full 189-column reconstruction cache `reconstruction-production-v1/fold_<k>/`
(`{train,valid,test}_features.npy`, `{train,valid}_indices.npy`, `manifest.json` with `feature_names`) and the
density-completion cache `fold_completion/fold_<k>/{train,valid,test}.npy`. Both scripts take `--folds 0,1,2,3,4`;
each fold trains on one GPU.

| Pair | Model | OOF AUC |
|---|---|---:|
| tabm_forensic_strong | TabM-D k 32, row-local views + 104 forensic columns + pair contrasts; 5 internal members | 0.969763 |
| tabm_forensic_imputed | TabM-D k 32, + neurally completed view, missing flags, reconstruction and density-completion channels; 10 internal members | 0.969964 |

Requirements: pytabkit (TabM), torch (CUDA), numpy, pandas, scikit-learn.
