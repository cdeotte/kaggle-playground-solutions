# Factorization machines over exact value tokens

Both models see every row as 12 tokens (`lib/tokens.py`): numerics formatted `%.2f`, categoricals as strings, NaN ->
`__NA__` (per column, no imputation), with one vocabulary per column over train + test. No target encodings: the
embeddings learn the per-token response.

**fm_exact_tokens** (`train_fm.py`): a factorization machine (`lib/fm_model.FM`): global bias + per-token linear
weights + pairwise interactions of k = 16 factor embeddings, dropout 0.5 on the factors.

**afn_tokens_density_heads** (`train_afn.py`): an AFN-style model (`lib/fm_model.AFN`): per-token linear weights + an MLP
on the 66 field-pair products of 32-dim token factors (kept separate, not sum-pooled; 2112 values) and a 50-column
label-free side input; dropout 0.8 on the factors. The MLP is 8 heads trained jointly (shared token factors, each head
`Linear(2162, 512) - ReLU - Dropout 0.3 - Linear(512, 128) - ReLU - Dropout 0.3 - Linear(128, 1)` on its own learned
scaling of the input); the loss is the BCE of every head, the prediction the mean head probability. Only the global bias
and the parameters without "bias" in their name are optimized, so the heads' layer biases keep their initial values.
Side input:
- `lib/side_features.py` (44 columns): 9 raw numerics, 9 fills and 9 fill uncertainties (q90 - q10 where missing) from
  LightGBM quantile imputers fit on train + test, 17 budget-geometry columns recomputed from the fills; standardized;
- `lib/density_model.py` (6 columns): a masked-token transformer (d 192, 6 layers, 8 heads, 30 epochs) trained on the
  train + test tokens without labels; per row: pseudo-log-likelihood of the observed cells, their mean predictive
  entropy, minimum log-probability, mean entropy of the missing cells, numbers of observed and missing cells.

Protocol (both): AdamW lr 1e-3, weight decay 1e-5 (none on the global bias), batch 8192, cosine schedule. Phase 1 trains
on 90% of the outer-train rows (first split of `StratifiedKFold(10, shuffle=True, random_state=2026 + fold)`) for 200
(FM) / 80 (AFN) epochs and picks the epoch with the best AUC on the other 10%; phase 2 trains a fresh model with the same
seed (2026 + fold) on all outer-train rows for that many epochs. Test = mean of the 5 fold models. Each fold is saved in
`parts/` (finished folds are skipped); the pair is written to `preds/<pair>_{oof,test}.npy` (probabilities) when all
5 folds are done.

```
python train_fm.py
python build_density.py     # density model + row channels -> cache/ (train_afn.py also builds them if missing)
python train_afn.py
```
`--folds 0,1` runs a subset of the folds. Token ids, imputers and side inputs are cached in `cache/`.

No other repo folders are needed.

| Pair | Model | OOF AUC |
|---|---|---:|
| fm_exact_tokens | factorization machine (k 16) on the 12 exact value tokens | 0.963130 |
| afn_tokens_density_heads | AFN-style pair-product MLP (k 32, 8 heads) on the tokens + imputation and density-model side input | 0.967969 |

Packages: torch (CUDA), lightgbm, scikit-learn, pandas, numpy.
