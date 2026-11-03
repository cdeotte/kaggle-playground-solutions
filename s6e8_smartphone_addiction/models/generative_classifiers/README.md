# Generative classifiers over exact value tokens

**naive_bayes_exact_tokens** (`train_token_nb.py`): naive Bayes over the 12 columns as exact value tokens (`%.2f` for
numerics, raw strings, NaN -> `__NA__`; vocabulary over train + test). Per column and class
`P(token | y) = (count + 20 prior_y) / (n_y + 20)`; score = sum of the 12 log-likelihood ratios + prior log-odds.
Per outer fold the held-out and test rows are scored with the tables of all outer-train rows and the outer-train rows by a
3-fold inner cross-fit (StratifiedKFold seed 777 + fold); the score array is filled fold by fold (details in the script
docstring). A per-fold logistic regression on the outer-train scores calibrates the held-out and test scores.
Test = mean of the 5 calibrated fold predictions.

**bayes_llr_channels_catboost** (`train_llr_channels.py`): 105 generative channels per row, read by CatBoost.
Channels computed from a set of labelled fitting rows:
- log-likelihood ratios `log P(t | y=1) / P(t | y=0)` of exact value tokens with Dirichlet smoothing toward the token's
  overall frequency, `P(t | y) = (count_y(t) + s p(t)) / (n_y + s)`: the 12 columns at s = 20, 5, 1 (36); 1-decimal
  tokens of the 8 numerics other than age at s = 50, 20 (16); 6 budget quantities (slack, sleep + daily, utilization,
  weekend - daily, weekend - (social + gaming + work), notifications / (app opens + 0.25)) at s = 20 (6); the 12 columns
  in the 7,500-row original dataset at s = 5 (12);
- pair lifts of 16 column pairs: `base = sigmoid(logit p_A + logit p_B - logit prior)` from the single-column rates,
  pair rate `(positives + m mean base) / (count + m)` (m 20, one pair also at m 100), lift = pair rate - base; plus the
  mean lift (17);
- the 9 numerics standardized with the fitting rows' mean / std (NaN -> 0) and their missing flags (18).
Per outer fold the outer-train rows get their channels from a 5-fold inner cross-fit (seed 42), the held-out and test rows
from all outer-train rows. CatBoostClassifier(800 iterations, lr 0.04, depth 6, l2_leaf_reg 5, seed 42, 8 threads) is
fit on the outer-train channels. Test = mean of the 5 fold models. Each fold is saved in `parts/` (finished folds are
skipped).

```
python train_token_nb.py
python train_llr_channels.py     # --folds 0,1 runs a subset of the folds
```
Each script writes `preds/<pair>_{oof,test}.npy` (probabilities).

No other repo folders are needed.

| Pair | Model | OOF AUC |
|---|---|---:|
| naive_bayes_exact_tokens | naive Bayes over the 12 exact value tokens (smoothing 20), logistic calibration | 0.944478 |
| bayes_llr_channels_catboost | CatBoost on 105 generative channels (token LLRs, pair lifts, budget and original-data LLRs) | 0.966292 |

Packages: catboost, scikit-learn, pandas, numpy.
