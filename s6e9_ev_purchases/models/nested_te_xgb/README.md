# XGBoost on nested target encodings

**The library** (`te_library.py`, `build_library.py`): 203 channels per row (the "Q8" set): the 13 raw columns, two atom
flags (income 30,000, commute 5.0), and target encodings, stored as logits with smoothing s = 1 and s = 20, of the exact
values of the 5 numeric columns, of 71 floor binnings of income, commute and age (50 to 30,000 USD, 0.1 to 10 km, 1 to 15
years) and of 4 log-income binnings. It is built per outer fold k: validation and test rows are encoded by the map fitted
on the outer-train rows; outer-train rows get nested encodings from an inner `StratifiedKFold(5)` of the outer-train rows
(seed 777; some families average the logits of 3, 5 or 8 inner seeds). `static_bases.py` makes label-free bases of the
numeric columns; the pairs use the periodic ones (`per_`: sin / cos of the standardised value at 3 frequencies).

**XGBoost** (`train.py`): depth 4, eta 0.02, min_child_weight 5, colsample_bynode 0.6, 6 parallel trees per round, and a
base margin from a logistic regression on the s = 1 / s = 20 encodings of the exact values and coarse income / commute
bins. Rounds are picked on a 10% holdout of the training rows (early stopping 200), then the model is refit on all of them
for 1.11x the rounds; test = mean of the 5 fold models. Each pair is a logit mean of streams of two kinds:
- **noise line**: the library's nested encodings plus Gaussian noise (sd 0.1, logit units) on the label channels of the
  training rows; the early-stopping rows are re-encoded from the fit rows' labels only;
- **pool line**: outer-train is split 50/50 three times, and each half trains a model with every label channel encoded
  from the other half (6 models per fold).

Later pairs add the periodic bases, one interaction clique per source column, gain pruning of 150 columns before the
refit, test rows encoded from all train labels, explicit differences between the income channels (`deltas`), and target
encodings of the GPT-2 BPE leading piece(s) of the income string (`tex_IncBpePre*`, see `common/bpe.py`) and of the
exact income shrunk toward its leading-piece group (`hte_*`).

| Pair | Streams (exact settings in `train.py`) | OOF AUC |
|---|---|---:|
| `xgb_te_dual` | pool line + noise line (2 noise draws) | 0.946157 |
| `xgb_te_periodic` | the same two lines + periodic bases, cliques, pruning, test rows from all train labels | 0.946226 |
| `xgb_te_deltas` | noise line (6 runs: deltas at feature weight 0.2, seeds 42, 7; at 0.5, seeds 11, 13, 42, 7) 50/50 with pool line (seeds 42, 7); periodic bases, cliques, pruning | 0.946275 |
| `xgb_te_bpe` | noise line with deltas + 4 BPE channels (leading piece and first two pieces, s = 1, 20); seeds 42, 7 | 0.946402 |
| `xgb_te_bpe_hier` | `xgb_te_bpe` + 2 hierarchical channels | 0.946415 |
| `xgb_te_bpe_hier_deltas` | 5 BPE + 2 hierarchical channels, deltas also on the BPE channels; seeds 42, 7 | 0.946436 |
| `xgb_te_periodic_bpe` | `xgb_te_periodic` + 2 BPE channels (leading piece, s = 1, 20) | 0.946442 |
| `xgb_te_dual_bpe` | `xgb_te_dual` + the same 2 BPE channels | 0.946377 |

```
python build_library.py                     # features/X_f{k}_{tr,te}.npy, the 203 channels of every fold (~4 GB)
python static_bases.py                      # features/static_{tr,te}.npy
python train.py all                         # or one pair, e.g. python train.py xgb_te_bpe
python train.py --stream q8_pool --folds 3  # one stream and fold (to run folds in parallel)
```
Each stream keeps its finished folds in `runs/parts/` (they are skipped when the command is repeated) and its own OOF /
test pair in `runs/preds/`; the pairs above go to `preds/`. A fold of a stream takes about 5 to 30 minutes on one GPU.
Needs xgboost (GPU), scikit-learn and scipy; transformers for the BPE channels. `../nested_te_realmlp_tabm` uses the same
library.
