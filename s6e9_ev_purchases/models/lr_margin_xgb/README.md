# XGBoost with nested target encodings and a logistic-regression base margin

Six GPU XGBoost models on one feature recipe, plus the shared library `lib/` (also used by `../te_mlp_lr`).

**Champion matrix** (`lib/features.py`, 45 columns): the 13 raw columns (categoricals as fixed integer codes), log1p of each
column's value count over train + test, 4 flags (income == 30,000, commute == 5 km, income not in the original data, number of
original rows with this income), nested target encodings (TE) of 11 keys and 4 differences of income TE logits. The keys:
income, income // 10, // 100, // 1000, commute (0.1 km and 1 km), age, home stations, work stations, the
(environment, subsidy, anxiety, home charging) cell and the (city, home stations, work stations) cell. Nested TE: inside each
outer fold, the training rows are encoded with an inner 5-fold split and the held-out and test rows with all training rows of
the fold.

**Additions** used by some of the models:
- **LR margin** (`lib/lr_margin.py`): a logistic regression fitted on each fold's training rows (the 6 inputs of the purchase
  formula, the TE logits and differences, the champion matrix with its <= 12-level columns one-hot). Its decision function is
  XGBoost's `base_margin`, so the trees learn a correction to it.
- **rule**: `p_rule = Phi((s - 5.6092) / 0.8345)` with `s = 1.2 income / 1e5 + 0.6 env + 2 subsidy - 1 [anxiety Medium] - 3 [anxiety High]`,
  the purchase score of the script that generated the original data.
- **rte**: residual TEs, the same nested encoding of `y - p_rule` (keys income, income // 100, income // 1000).
- **art**: 12 row-wise artifact columns: income mod 10 / 100 / 1000 / 10000, its last 5 digits, the commute's first decimal and
  integer part, income minus the nearest multiple of 1000.
- **bpe**: the leading GPT-2 BPE piece of the income string (`common/bpe.py`) as an extra TE key.

**Protocol** (`lib/xgb_dress.py`): pass 1 fits each fold with the first seed and early stopping and sets the number of rounds to
1.1 x the median best iteration; pass 2 trains every seed on every fold with that fixed number of rounds. OOF and test (mean of
the 5 fold models) are averaged over the seeds. Fast parameters: lr 0.03, depth 7, colsample 0.3, subsample 0.7,
min_child_weight 5, max_bin 1024; slow parameters: lr 0.015, depth 6, colsample 0.3, lambda 5.

```
python train_te_diffs.py
python train.py xgb_lr_margin
python train.py xgb_lr_margin_rte
python train.py xgb_lr_margin_rte_art
python train.py xgb_rule_margin_art
python train.py xgb_lr_margin_rte_art_bpe
```
Each run writes `preds/<pair>_{oof,test}.npy` (probabilities) and takes 5-15 minutes on one GPU.

| Pair | Model | Seeds | Rounds in our run | OOF AUC |
|---|---|---:|---:|---:|
| xgb_te_diffs | champion matrix, no margin, slow parameters; two batches of 5 seeds, each with its own pass 1 | 10 | 1471, 1422 | 0.946091 |
| xgb_lr_margin | + LR margin (LR max_iter 1000), fast parameters | 10 | 332 | 0.946191 |
| xgb_lr_margin_rte | + rte | 20 | 372 | 0.946211 |
| xgb_lr_margin_rte_art | + art | 20 | 324 | 0.946244 |
| xgb_rule_margin_art | rule margin `logit(p_rule)` instead of the LR, rte, art, slow parameters | 20 | 1446 | 0.946150 |
| xgb_lr_margin_rte_art_bpe | xgb_lr_margin_rte_art + bpe TEs (m 1, 20; trees and LR) + bpe rte (m 5) | 20 | 315 | 0.946462 |

Requirements: xgboost (CUDA), scikit-learn, scipy, pandas; transformers for the GPT-2 tokenizer of the bpe model.
