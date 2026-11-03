# Lift banks: cross-fitted pair-lift and response-surface features

Seven label-derived feature banks read by the density-model family (`../realmlp_density`, `../tabm_density`,
`../xgb_density_margin`, `../catboost_density`, `../dcn`, `../ydf`, `../cuml_density_models`). This folder trains no
classifier and writes no prediction pairs; each script writes one file pair per outer fold to `cache/`.

**Fold discipline (every bank).** In outer fold k, an outer-train row gets values from maps fit on the other 4 inner
groups of `StratifiedKFold(5, shuffle=True, random_state=777)` (on the outer-train rows); validation and test rows get
values from maps fit on all outer-train rows. Labels of the validation fold are never used. Keys are coded on the
unlabeled train + test rows together; a missing or undefined value is its own key.

**Pair lift** (`lib/lift.py`, used by all banks except dsx). On the fit rows (label rate p0):
`pA`, `pB` = label rates of the row's A key and B key, smoothed toward p0 with strength 20 (unseen key -> p0);
`base = sigmoid(logit pA + logit pB - logit p0)` (log-odds independence of A and B);
`pair = (positives of the (A, B) cell + s * base) / (rows of the cell + s)`; **lift = pair - base** (0 for an unseen cell).
The lift measures how far the pair's label rate departs from what the two single keys predict; s sets how strongly a
sparse cell is shrunk toward the independence base. Every bank also writes the row mean of its channels.

Key types: exact = the `'%.2f'` value, r01 = value rounded to 0.1, q025 = floor(value / 0.25), q5 = floor(value / 0.5).

- **dsx** (`build_dsx.py`, 3 columns `dsx_s1`, `dsx_s3`, `dsx_s8`): not a lift but multi-scale response surfaces of
  (daily, social) on the exact 0.01 grid (the distinct values of train + test). Label sums and counts of the fit rows per
  grid cell are blurred with a 2-D Gaussian filter of sigma 1, 3 and 8 grid steps; rate = (blurred sum + 25 p0) /
  (blurred count + 25) with p0 the outer-train label rate; value = logit of the row's cell (clip 1e-4); rows missing daily
  or social get logit(p0).
- **plift** (`build_plift.py`, 9 + mean): exact x exact pairs with s = 20: (app opens, weekend), (daily, app opens),
  (notifications, weekend), (daily, weekend), (daily, notifications), (daily, gaming), (sleep, app opens), (social,
  sleep), and (notifications, app opens) with s = 100. Columns `plift_<A[:5]>_<B[:5]>`, `plift_mean`.
- **cbank** (`build_cbank.py`, 8 + mean): coarse keys with a tiny pair smoothing s = 1: daily q025 x notifications
  exact, app opens exact x weekend q025, notifications exact x weekend q025, daily q025 x app opens exact, sleep q025 x
  app opens exact, social r01 x work r01, daily r01 x sleep r01, social r01 x gaming r01 (`cb_*`, `cb_mean`).
- **wbank** (`build_wbank.py`, 4 + mean): derived keys utilization = (social + gaming + work) / daily (r01) and weekend -
  daily (q025): utilization x social exact (s = 0.5), weekend-daily x daily exact (0.5), weekend-daily x weekend exact
  (2), utilization x gaming exact (2) (`wb_*`, `wb_mean`).
- **wbank12** (`build_wbank12.py`, 12 + mean): the 4 wbank pairs, each at s = 0.5, 1 and 2 (`w12_<pair>_<s>`,
  `w12_mean`).
- **sbank** (`build_sbank.py`, 4 + mean): share_s = social / max(daily, 0.001): notifications exact x share_s r01
  (s = 1), app opens r01 x share_s r01 (5), social q5 x notifications exact (2), notifications q025 x share_s exact (5)
  (`sb_*`, `sb_mean`).
- **ebank** (`build_ebank.py`, 5 + mean): half-hour lattice of the screen columns x exact counts: weekend q5 x
  notifications (s = 2), weekend q5 x app opens (2), daily q5 x app opens (1), daily q5 x notifications (2), social q5 x
  notifications (2) (`eb_*`, `eb_mean`).

## Commands
```
python build_dsx.py
python build_plift.py
python build_cbank.py
python build_wbank.py
python build_wbank12.py
python build_sbank.py
python build_ebank.py
```
Each takes `--folds 0,1,2,3,4` (default: all five) and runs on the CPU (seconds to a few minutes per fold). No other
folder has to run first (only the data in `../../data`).

## Cache files (in run order)
| File | Built by | Contents |
|---|---|---|
| `dsx_f{0..4}_{tr,te}.parquet` | `build_dsx.py` | dsx_s1, dsx_s3, dsx_s8: (daily, social) response-surface logits (flag `dsx`) |
| `plift_f{0..4}_{tr,te}.parquet` | `build_plift.py` | 9 exact-pair lifts + plift_mean (flag `plift`) |
| `cbank_f{0..4}_{tr,te}.parquet` | `build_cbank.py` | 8 coarse-lattice lifts + cb_mean (flag `cbank`) |
| `wbank_f{0..4}_{tr,te}.parquet` | `build_wbank.py` | 4 utilization / weekend-gap lifts + wb_mean (flag `wbank`) |
| `wbank12_f{0..4}_{tr,te}.parquet` | `build_wbank12.py` | the wbank pairs at 3 smoothings (12) + w12_mean (flag `wbank12`) |
| `sbank_f{0..4}_{tr,te}.parquet` | `build_sbank.py` | 4 social-share lifts + sb_mean (flag `sbank`) |
| `ebank_f{0..4}_{tr,te}.parquet` | `build_ebank.py` | 5 half-hour-lattice x count lifts + eb_mean (flag `ebank`) |

`_tr` files hold all 691,369 train rows (inner-cross-fitted values for the outer-train rows of the fold, full-map values
for its validation rows), `_te` files the 296,302 test rows, in train.csv / test.csv order; float32.

## Requirements
numpy, pandas, pyarrow, scipy (`build_dsx.py`), scikit-learn.
