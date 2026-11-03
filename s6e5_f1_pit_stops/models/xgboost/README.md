# XGBoost: fold-safe target-encoding harness and six standalone recipes

Twelve GPU XGBoost models: six recipes on one harness (`train.py` + `lib/harness.py`) and six self-contained scripts.
All of them use the competition folds of `common/data.py` and write `preds/<pair>_{oof,test}.npy` (probabilities).

## Harness recipes (`train.py`, `lib/harness.py`)

**Base matrix** (label-free, built once on the train rows + the original rows + the test rows):
- the 14 raw columns (Driver / Compound / Race as string keys) and 10 combination keys (Year x Race, Year x Driver,
  Race x Driver, Race x Compound, Race x Stint, Driver x Compound, Stint x Compound, Year x Race x Driver,
  Race x Driver x Compound, Race x Stint x Compound);
- 24 domain numerics: logs and squares, RaceProgress logit / cube / remaining, estimated total laps and laps remaining
  (LapNumber / RaceProgress), tyre-life ratios and products, position inverse / log / products, position-change flags;
- group aggregates: mean and std of 9 numerics within Race, Driver, Compound, Year x Race, Race x Driver,
  Race x Compound, Driver x Compound and Stint x Compound, and each row's difference / ratio to the group mean;
- previous-lap value, difference and 3 / 5-lap rolling means of 7 numerics within (Year, Race, Driver) ordered by
  LapNumber (over all rows);
- the count and frequency of every string key over all rows. Columns constant on the train + original rows are dropped.

**Optional blocks** (`Config` flags, added in this order; the column order matters because XGBoost samples columns):
- `use_public_formula_features` (`PUB_*`, 28 columns): ratio / interaction formulas of public notebooks, race-phase and
  tyre-age flags, compound-life ratios;
- `use_sequence_next_features`: next-lap value and difference of the 7 lag columns;
- `use_hazard_features` (`HZ__*`): tyre-life / race-progress / lap bins and 6 keys with race, compound, stint, pit stop;
- `use_relative_rank_features` (`REL_*`): percentile rank, difference and ratio to the median, robust z of 6 numerics
  within 4 race / stint / compound groupings (96 columns);
- `use_pipeline_a_features` (`PIPA_*`, "Pipeline A"): LapNumber / RaceProgress and TyreLife / LapNumber, floor categories
  of the 11 numerics and these 2 ratios, counts of Driver / Compound / Race / Year / PitStop, 200 RaceProgress quantile
  bins, Race x Compound and Race x Year keys;
- `use_catboost_style_features` (`X2_*`): 18 pit-window ratios / stress terms, phase / tyre / position bins, integer
  categories, rounded / floor / digit categories of 7 timing columns, 12 category combinations;
- `use_extended_group_residuals` (`X2_GRP_*`): log size of 13 groupings and, per numeric (7, or 11 with the CatBoost-style
  block), the difference to the group mean, the z-score and the percentile rank within the group;
- `use_pit_window_key_features` (`X2_PIT_*`): integer / sign / bin categories of lap, tyre life, position, progress and 12
  keys with race, compound, stint, driver;
- `use_pit_cadence_features` (`X2_CAD_*`): last pit lap, average stint length so far, expected next pit lap and the
  distance to it, pit rates, stress terms, their bins and 7 keys;
- `use_artifact_hash_keys` (`X2_HASH_*`): rounded timing values and integer race state, alone and in 8 keys;
- `use_source_indicator`: 1 on the original rows.

**Per fold:** training side = the competition rows of the 4 other folds + all 101,371 original rows (sample weight
`external_sample_weight`: 0.35, 0.5 or 0.8; competition rows 1.0). Every string key with 2..`max_te_cardinality` levels
gets a smoothed target encoding `(sum + s * prior) / (count + s)` for each s in `te_smooths`, out-of-fold over an inner
`StratifiedKFold(5, shuffle=True, random_state=42)` of the training side (original rows included) and fitted on the whole
training side for the validation / test rows, plus its count and frequency on the training side; then all string keys are
ordinal-encoded.

**Model:** XGBoost hist on GPU, lossguide, 96 leaves, lr 0.025, min_child_weight 15.8, subsample 0.967, colsample_bytree
0.965, colsample_bynode 0.739, gamma 0.32, alpha 1.5e-4, lambda 4.19, max_bin 384, up to 16,000 rounds with early stopping
(AUC, 400 rounds) on the validation fold, random_state 43..47 for folds 0..4 (recipe overrides in `train.py`). Test = mean
of the 5 fold models.

```
python train.py xgb_relrank_next_lap_original_w05
python train.py xgb_relrank_pipea_formulas_smooth_te_original_w05
python train.py xgb_relrank_pipea_formulas_next_lap_original_w05
python train.py xgb_relrank_original_w035
python train.py xgb_group_residuals_original_w08
python train.py xgb_hazard_pit_window_cadence_original_w08
```
A recipe takes 15-20 minutes on one GPU (the hazard recipe about 100 minutes). `--folds 0,1` runs a subset (each process
rebuilds the base matrix); finished folds are kept in `parts/` and the pair is written when the last fold is done.

## Standalone scripts

- `train_nearest_original.py`: 10 race-state ratios; a cKDTree over the standardized original rows (20 numerics) gives,
  for k = 1, 3, 5, 10, the mean / std of the neighbours' original targets and the mean distance; original target means and
  counts of every categorical-like column and 6 crosses; frequency, original percentile rank, rounding and decimals of
  the 4 high-cardinality numerics. Depth 6, lr 0.03, 500 rounds. About 25 minutes (mostly the KDTree queries).
- `train_union_prebinned.py`: three label-free pipelines (A: transforms, state interactions, bins, crosses, counts,
  original priors; B: quantile / round / digit / fraction keys of the timing columns, train / original drift, original
  target means, key pairs; LGBM-port keys: integer, quantile, equal-width and log bins of 6 numerics, digit pairs), then
  per fold nested target mean / std / min / max encodings of about 250 keys; every column except the 3 raw timing columns
  is pre-binned to <= 256 quantile bins of the fold's training rows, XGBoost max_bin 16000. Competition rows only.
  About 1 hour per fold (mostly the encodings).
- `train_self_supervised.py`: 70 auxiliary XGBoost models trained on the original rows predict each of the 14 columns
  from the 13 others (regression or class max-probability / entropy); the final model (depth 3, lr 0.1, native
  categoricals) uses the 14 columns + 25 predictions. The auxiliary stage takes about 7.5 hours; the folds seconds.
- `train_tyrelife_anchor.py`: raw TyreLife as an anchor: inner-fold target statistics and quantiles per anchor value,
  target means of category x anchor keys, anchor digit pairs and category pairs; native categoricals, lr 0.01.
  Competition rows only, about 7 minutes.
- `train_pairwise_combos.py`: smoothed (10) target means and counts of all 91 pairs of the 14 columns, with the original
  rows appended to the training side of every fold; depth 8, lr 0.01. About 20 minutes.
- `train_laptime_anchor_union.py`: pipelines A and B (pairs of up to 20,000 levels), LapTime (s) anchor statistics and
  category x anchor / digit-pair / category-pair target means, fold-safe target and count encodings (1541 columns);
  depth 3, lr 0.02. Competition rows only, about 3 hours.

```
python train_nearest_original.py
python train_union_prebinned.py
python train_self_supervised.py
python train_tyrelife_anchor.py
python train_pairwise_combos.py
python train_laptime_anchor_union.py
```
Each script also takes `--folds` / `--force` (the feature stage before the folds runs in every process).

No other repo folder has to run first; every script reads only `data/` (train.csv, test.csv, f1_strategy_dataset_v4.csv).

| Pair | Model | OOF AUC |
|---|---|---:|
| xgb_relrank_next_lap_original_w05 | harness: base + relative ranks + next-lap values (499 columns, 13 keys, TE s 5 / 20 / 100); original weight 0.5 | 0.951730 |
| xgb_relrank_pipea_formulas_smooth_te_original_w05 | harness: base + relative ranks + Pipeline A + formulas (568 columns, 29 keys), TE s 20 / 100 / 500; original weight 0.5 | 0.952116 |
| xgb_relrank_pipea_formulas_next_lap_original_w05 | harness: base + relative ranks + Pipeline A + formulas + next-lap values (582 columns, 29 keys); original weight 0.5 | 0.952012 |
| xgb_relrank_original_w035 | harness: base + relative ranks (485 columns, 13 keys); original weight 0.35 | 0.951866 |
| xgb_group_residuals_original_w08 | harness: relative ranks, Pipeline A, formulas, group residuals, source flag (855 columns; TE s 20 on the 27 keys with <= 25,000 levels); colsample 0.82 / 0.62, lambda 7.5; original weight 0.8 | 0.952977 |
| xgb_hazard_pit_window_cadence_original_w08 | harness: + hazard keys, CatBoost-style block, pit-window keys, pit cadence, hash keys (1478 columns; TE s 20 on the 116 keys with <= 750 levels); lr 0.006, min_child_weight 18, subsample 0.88, colsample 0.70 / 0.54, lambda 15, gamma 0.7, patience 1000; original weight 0.8 | 0.953553 |
| xgb_nearest_original_knn | nearest-original-row target means / distances (k 1, 3, 5, 10), original priors, state ratios; 500 rounds | 0.946744 |
| xgb_union_te_prebinned | pipelines A / B / LGBM-port union, nested target statistics, pre-binned to 256 bins except the timing columns, max_bin 16000 | 0.950423 |
| xgb_original_self_supervised | raw columns + 25 predictions of each column by models trained on the original rows | 0.940117 |
| xgb_tyrelife_anchor_te | TyreLife-anchor target statistics / quantiles, category x anchor, digit-pair and category-pair target means | 0.946144 |
| xgb_pairwise_combo_te_original | target means + counts of the 91 column pairs; original rows appended to every training fold | 0.951525 |
| xgb_union_te_laptime_anchor | pipelines A / B, LapTime anchor statistics, fold-safe target / count encodings | 0.949137 |

Requirements: xgboost (CUDA), pandas, numpy, scikit-learn, scipy.
