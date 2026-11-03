# Neural variants on nested target encodings: NODE, FT-Transformer and a wide linear model

Three models that share one feature builder, `lib/features.py`:
- **raw + derived** (32 columns, label-free): the 9 raw numerics (NaN kept) and 23 derived columns that propagate NaN
  instead of imputing: budget slack `daily - (social + gaming + work)`, component sum, utilization `sum / daily`, the
  cent parts of these three, exact-budget flag, `sleep + daily`, at-20 flag, the 5 per-daily-hour ratios,
  `weekend - daily`, `weekend / daily`, 3 products, repair flag (utilization in [0.85, 0.95]) and ordinal codes of the
  3 categoricals;
- **original rates** (18): per-value label rate and 'Severe' addiction-level rate of each numeric in the 7,500-row
  original dataset;
- **nested target encodings** (36) of exact value tokens (`%.2f`, NaN -> `__NA__`), `(sum + m prior) / (count + m)`:
  the 12 columns (m 30), slack / sum / sleep + daily (m 30) and utilization (m 50), daily / weekend / social /
  notifications / app opens again at m 10 and m 200, and 10 column pairs (m 100). Inside each outer fold the outer-train
  rows get inner 5-fold cross-fitted values (StratifiedKFold, seed 777); held-out and test rows the map of all
  outer-train rows.

**node_oblivious_te** (`train_node.py`): NODE (`lib/node_model.py`, neural oblivious decision ensembles with entmax15
feature selection and entmax15 threshold regions) with 2 layers of 128 trees, depth 6, residual connections and a
linear head, on the 50 label-free columns (median-imputed and standardized on train + test) + the 36 encodings.
Adam lr 3e-3, weight decay 1e-5, batch 1024, warm-up 3 epochs then cosine. The number of epochs is chosen on 10% of the
outer-train rows (up to 60 epochs, patience 10); a fresh model is then trained on all outer-train rows for that many
epochs. Seed 2026 + fold.

**ftt_imputed_tokens_te** (`train_ftt.py`): FT-Transformer (`lib/ftt_model.py`; d_token 128, 3 layers, 8 heads, FFN
factor 4/3, dropout 0.1, linear numerical embedding, CLS head). Inputs (`lib/ftt_features.py`, label-free, train + test):
LightGBM quantile imputers (q10 / q50 / q90 of each numeric from the other columns) give 9 fills, 9 fill uncertainties
(q90 - q10 where missing), 9 raw values (NaN -> fill) and 20 budget-geometry columns from the fills (47 numerics,
standardized), plus the 36 encodings as numerics; categoricals = gender / stress / impact and the exact value token of
each numeric (12 embeddings). AdamW lr 3e-4, weight decay 1e-5, batch 1024, mixed precision, warm-up 5 epochs then
cosine over 40 epochs, early stopping on 10% of the outer-train rows (patience 10); the best epoch predicts. Seed
2026 + fold. `--mc 10` (MC dropout at inference) gives a variant that is not in the stack.

**wide_elasticnet_te** (`train_wide.py`): elastic-net logistic regression (`SGDClassifier`, log loss, alpha 1e-4,
l1_ratio 0.9, max_iter 2000, seed 777) on the 32 raw + derived columns (median-imputed), the 18 original rates, the 36
encodings and the 36 pairwise products of the 9 single-numeric encodings, standardized on the outer-train rows.

The early-stopping / epoch-selection split is the first split of `StratifiedKFold(10, shuffle=True, random_state=seed +
fold)` of the outer-train rows. Test = mean of the 5 fold models. Each fold is saved in `parts/` (finished folds are
skipped); the pair is written to `preds/<pair>_{oof,test}.npy` (probabilities) when all 5 folds are done.

```
python train_node.py
python train_ftt.py      # --lr 3e-4 --mc 1 are the defaults
python train_wide.py
```
`--folds 0,1` runs a subset of the folds. `train_ftt.py` caches its label-free input matrices in `cache/`.

No other repo folders are needed.

| Pair | Model | OOF AUC |
|---|---|---:|
| node_oblivious_te | NODE (2 x 128 oblivious trees, depth 6) on raw + derived + original rates + nested TEs | 0.956847 |
| ftt_imputed_tokens_te | FT-Transformer on imputation features, value-token embeddings and nested TEs | 0.966821 |
| wide_elasticnet_te | elastic-net logistic regression on nested TEs, their products, raw + derived, original rates | 0.959015 |

Packages: torch (CUDA), lightgbm, scikit-learn, pandas, numpy.
