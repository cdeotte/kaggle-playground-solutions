# Base feature blocks of the forensic models (fixed-original channels, screen response)

No prediction pairs: this folder builds the two shared blocks that the forensic models in `../xgb_forensic`,
`../catboost_forensic`, `../lgbm_margin` and `../tabm_forensic` read from `cache/`.

**Fixed-original source channels** (`build_source_channel.py`, 583 columns for every train and test row). All fitted
state comes from the 7,500-row original dataset (its predictors, its binary label and its 4-level `addiction_level`);
the competition labels are not used.
- univariate (171): keys = each numeric rounded to 2 decimals (only values that occur in the original get a code) and cut
  at the original's q8 / q16 / q32 / q64 / q128 quantile edges, and each categorical's exact level. Per key: binary
  log-likelihood ratio `log P(key | 1) - log P(key | 0)` with additive smoothing toward the key's base rate (alpha 20),
  smoothed posterior, `log1p(original count)`. Missing / unseen keys get LLR 0.
- latent_univariate (48): for the q16 and exact keys, the log-ratio of each addiction level's key likelihood to the base
  rate (None, Mild, Moderate, Severe).
- pair (198): all 66 pairs of the 12 predictors on their q8 / exact codes: LLR (alpha 50), posterior, log count.
- geometry (72): per numeric, empirical CDF of the value among the class-0 and class-1 original values, their
  difference, the distances to the nearest class-0 / class-1 original value (in original std units) and their
  log-difference, observed flag, exact-value original support.
- aggregate (79): naive-Bayes sums of the univariate LLRs per block (all, screen, core, background, categorical) and
  resolution with their probabilities, the sum of the pair LLRs, and per block the softmax posterior of the addiction
  levels (P(Moderate or Severe) and each level).
- missing (15): missing flags and missing counts (all, screen, background).

**Screen response** (`build_screen_response.py`, 84 columns per outer fold). Row-local geometry of the screen fields:
daily, component sum (social + gaming + work), strict slack (daily - sum, NaN unless all three components are
observed), partial slack, slack / daily, sum / daily, the three component shares of daily and of the sum, Herfindahl
index, dominance, spread, weekend - daily, weekend / daily, daily / sleep. Keys: each value cut at its q16 and q64
quantile edges (edges from the fold's outer-training rows, NaN is its own key), five q16 x q16 pairs and the 6-bit
screen missing mask (42 keys). Per key: target rate smoothed toward the prior with m = 20 as a logit, and
`log1p(count)`. The fold's outer-training rows are encoded with an inner `KFold(5, shuffle=True, random_state=742 + fold)`;
the fold's rows and the test rows with all outer-training rows. Folds = `common.data.folds()`.

```
python build_source_channel.py
python build_screen_response.py
```
Both are CPU only and take a few minutes. Run them first; nothing else has to run before.

Cache files (`cache/`):

| File | Content |
|---|---|
| `source_channel_train.npy`, `source_channel_test.npy` | float32 (691,369 / 296,302 x 583), train.csv / test.csv row order |
| `source_channel_manifest.json` | `feature_names`, `groups` (column indices of univariate, latent_univariate, pair, geometry, aggregate, missing, compact) |
| `screen_response/fold_{k}_train.npy` | float32, the outer-training rows of fold k (train.csv order) |
| `screen_response/fold_{k}_valid.npy` | float32, the rows of fold k |
| `screen_response/fold_{k}_test.npy` | float32, all test rows (encoded with fold k's state) |
| `screen_response/fold_{k}_{train,valid}_indices.npy` | train.csv row indices of the two row sets |
| `screen_response/fold_{k}_manifest.json` | `feature_names` (`screen_response_logit__<key>`, `screen_response_count__<key>`) |

`lib/blocks.py` has the loaders (`source_channel(role)`, `screen_response(fold)`) the other folders use.

| Pair | Model | OOF AUC |
|---|---|---:|
| (none) | feature blocks only | - |

Requirements: numpy, pandas, scipy, scikit-learn.
