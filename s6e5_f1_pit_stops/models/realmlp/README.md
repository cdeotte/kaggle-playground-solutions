# RealMLP (pytabkit) on Pipeline A, per-fold builder and original-statistics features

Eighteen pytabkit `RealMLP_TD_Classifier` models in four scripts, plus their shared code in `lib/`. Folds: the 5 folds of
`common/data.py`; when a model adds original rows to a fold's training rows, it uses the original rows outside the same
fold of the original data (`orig_folds`), except `realmlp_3config_rank_freq_drift`, which adds all of them. The fold's
validation rows are passed to `fit` as the validation set (pytabkit selects the best epoch on it; configurations with
`use_early_stopping` also stop early on it).

**Pipeline A** (`lib/pipeline_a.py`, 37 columns): the 14 columns; `LapNumber / (RaceProgress + 1e-6)` and
`TyreLife / max(LapNumber, 1)`; every numeric column and the two ratios floored and coded as a category (codes in order of
first appearance in the fitting rows, -1 if unseen); counts of Driver, Compound, Race and of the Year / PitStop codes;
200 quantile bins of RaceProgress; codes of Race x Compound and Race x Year. It is fitted on all train rows (global) or on
each fold's train rows (fold-local) and applied to the other frames. **Extended Pipeline A** (`lib/pipeline_a3.py`, 41
columns) adds LapTime x Cumulative_Degradation, LapTime x |Cumulative_Degradation|,
LapTime / (|Cumulative_Degradation| + 1e-6) and 7 quantile bins of LapTime, and has other RealMLP defaults (6 epochs, lr 0.01, sq_mom 0.99, n_ens 24, p_drop
schedule `invsqrtp1e-3`).

**`train.py`** (14 pairs, `lib/pipeline_a.py` / `lib/pipeline_a3.py`). Common RealMLP settings: PLR embeddings
(`plr_hidden` 32 / 16 unless noted), `lin_cos_log_15` learning-rate schedule, label smoothing with `sqrt_cos` schedule,
transforms one-hot / median-center / robust-scale / smooth-clip / embedding / l2-normalize, batch 512, model seed
`42 + fold + offset` (fold 1-5).
- `realmlp_fold_pipeline_a_te_*`: fold-local Pipeline A + sklearn `TargetEncoder` (smooth auto, 5 inner folds, seed
  42 + fold) of the two combination codes (39 columns); original rows appended; silu, 4 epochs, lr 0.03, PLR sigma 1,
  label smoothing 0.01, one model per fold, offset 100.
- `realmlp_pipeline_a_mish_<n>seed*`: global Pipeline A, no target encoding (37 columns); original rows appended; mish
  with parametric activations, [768, 384, 192], n_ens 32, 4 epochs, lr 0.025, first-layer lr factor 0.125, PLR sigma 1
  (lr factor 0.16), wd 0.014, p_drop 0.03 (0.04 in `*_drop04`), label smoothing 0.005; n models per fold with offsets
  100, 200, ... averaged.
- `realmlp_fold_pipeline_a_arith_mish`: fold-local extended Pipeline A, no target encoding (41 columns); original rows
  appended; mish with parametric activations, [768, 384, 192], n_ens 32, 6 epochs, lr 0.0175, wd 0.017, p_drop 0.05,
  PLR sigma 0.94 (lr factor 0.08), label smoothing 0.005; one model per fold, offset 100.
- `realmlp_shape_state_te_no_original`: fold-local extended Pipeline A + string keys (Race x Compound x Stint,
  Driver x Race, Driver x Compound, Compound x Stint, Race x Stint, Race x Driver x Compound), LapTime / LapTime_Delta /
  Cumulative_Degradation / RaceProgress rounded to 0-2 decimals, lap-delta shape features (`q` = piecewise quadratic of
  LapTime_Delta, `(q + 4.4)^2`, its bins of width 2, `(LapNumber - 50)^2`, `(RaceProgress - 0.6)^2`, LapTime rounded to
  0.1, Cumulative_Degradation bins of width 2, first letter of the driver code) and race-state features (stint start
  lap, laps left, tyre / degradation / delta ratios, race quarter, tyre-age and laps-left bins, Compound crosses);
  `TargetEncoder` of 30 keys (113 columns); no original rows; silu [512, 512, 256], n_ens 24, lr 0.0105, wd 0.014, PLR
  24 / 12 with sigma 2.2, one model per fold, offset 102200.

**`train_builder.py`** (2 pairs, `lib/fold_builder.py`): a builder fitted per fold on the fold's train rows + original
rows + test rows: the 14 columns (low-cardinality numerics also as strings), 11 ratios and pit-window counters, tyre /
lap / progress / position bins, values rounded to 1-3 decimals, hundredths and 50 / 100 quantile bins of the 4
continuous columns, 10 combination keys, relative frequencies of all categories, mean / std / count / residual of 7
numerics per 9 keys, smoothed (m 40) original target rates and log counts of 16 keys (from the fold's original rows),
a source flag; nested target encodings of the first candidate keys (5 inner folds; training rows out-of-fold, validation
and test rows the full map). Original rows appended. RealMLP: n_ens 16, batch 512, early stopping on the validation
1-AUC, no l2-normalize, seed 42 + fold.
- `realmlp_sequence_anchor_priors`: + race-stint sequence features (rows ordered by Year, Race, Driver, Stint, lap:
  position and size of the group, previous / next values and differences of 6 columns, 2 slopes); 14 keys x smoothing
  10 / 50 (460 columns); mish [768, 384, 192], lr 0.024, up to 72 epochs.
- `realmlp_pair_grid_te`: + all 36 pairs of 9 categorical keys; 24 keys x smoothing 5 / 20 / 100 (547 columns); silu
  [1024, 512, 256], lr 0.021, up to 80 epochs.

**`train_three_configs.py`** (`realmlp_3config_rank_freq_drift`): features computed once on train + test + original
(278 columns): race-state numerics, frequency and percentile rank of 20 numerics over all rows, log / sqrt / inverse /
absolute / rounded / mod-100 transforms, 64 / 256 quantile bins, 7 pair and 3 triple keys, and per category (31) log
count, rare flag, smoothed (m 25) original target rate and count, and train / original frequency drift. All original
rows appended with an is-original category. Three tuned RealMLP configurations per fold (n_ens 8, up to 32 epochs,
batch 1024; seeds 42 + 100 fold + 1, 2, 3), probabilities averaged.

**`train_original_rank_te.py`** (`realmlp_original_rank_zscore_te`): no original rows in training; percentile ranks and
z-scores against the original values, first two decimals as categories, 7 relationship features, quantile bins, 10 pair
and 3 triple keys, original target rates / log counts of 27 keys and original frequencies of 10 columns (123 columns),
plus per-fold nested target encodings (smoothing 20, 5 inner folds) of 23 keys (146 columns). RealMLP: n_ens 8,
4 x 352 mish, lr 0.075, batch 256, up to 20 epochs with early stopping on the validation classification error,
seed 42 + fold.

```
python train.py realmlp_fold_pipeline_a_te_nens20
python train.py realmlp_fold_pipeline_a_te_nens24
python train.py realmlp_fold_pipeline_a_te_light_reg
python train.py realmlp_fold_pipeline_a_te_light_reg_wide
python train.py realmlp_pipeline_a_mish_2seed
python train.py realmlp_pipeline_a_mish_2seed_ls0075
python train.py realmlp_pipeline_a_mish_3seed
python train.py realmlp_pipeline_a_mish_3seed_drop04
python train.py realmlp_pipeline_a_mish_4seed
python train.py realmlp_pipeline_a_mish_4seed_drop04
python train.py realmlp_pipeline_a_mish_5seed
python train.py realmlp_pipeline_a_mish_5seed_drop04
python train.py realmlp_fold_pipeline_a_arith_mish
python train.py realmlp_shape_state_te_no_original
python train_builder.py realmlp_sequence_anchor_priors
python train_builder.py realmlp_pair_grid_te
python train_three_configs.py
python train_original_rank_te.py
```
Each run writes `preds/<pair>_{oof,test}.npy` (probabilities). Only `data/` is read; no other repo folder has to run
first. One GPU per run.

| Pair | Model | OOF AUC |
|---|---|---:|
| realmlp_fold_pipeline_a_te_nens20 | fold-local Pipeline A + 2 TEs; silu [512, 256, 128], n_ens 20, p_drop 0.04, wd 0.016 | 0.954037 |
| realmlp_fold_pipeline_a_te_nens24 | same with n_ens 24 | 0.954083 |
| realmlp_fold_pipeline_a_te_light_reg | n_ens 24, p_drop 0.03, wd 0.014 | 0.954095 |
| realmlp_fold_pipeline_a_te_light_reg_wide | n_ens 24, p_drop 0.03, wd 0.014, [768, 384, 192] | 0.954111 |
| realmlp_pipeline_a_mish_2seed | global Pipeline A, mish, 2 models per fold | 0.954391 |
| realmlp_pipeline_a_mish_2seed_ls0075 | label smoothing 0.0075, 2 models per fold | 0.954387 |
| realmlp_pipeline_a_mish_3seed | 3 models per fold | 0.954409 |
| realmlp_pipeline_a_mish_3seed_drop04 | p_drop 0.04, 3 models per fold | 0.954410 |
| realmlp_pipeline_a_mish_4seed | 4 models per fold | 0.954419 |
| realmlp_pipeline_a_mish_4seed_drop04 | p_drop 0.04, 4 models per fold | 0.954416 |
| realmlp_pipeline_a_mish_5seed | 5 models per fold | 0.954423 |
| realmlp_pipeline_a_mish_5seed_drop04 | p_drop 0.04, 5 models per fold | 0.954426 |
| realmlp_fold_pipeline_a_arith_mish | fold-local extended Pipeline A, mish, 6 epochs, PLR sigma 0.94 | 0.954366 |
| realmlp_shape_state_te_no_original | extended Pipeline A + combos, rounded values, lap-delta shape, race state, 30 TEs; no original rows | 0.950537 |
| realmlp_sequence_anchor_priors | per-fold builder + race-stint sequence features, 28 nested TEs | 0.949876 |
| realmlp_pair_grid_te | per-fold builder + pair grid, 72 nested TEs | 0.949185 |
| realmlp_3config_rank_freq_drift | rank / frequency / original-rate / drift features, all original rows, 3 configurations per fold | 0.948729 |
| realmlp_original_rank_zscore_te | original ranks / z-scores / priors + 23 nested TEs, no original rows | 0.948437 |

Requirements: pytabkit, torch (CUDA), scikit-learn, pandas, numpy.
