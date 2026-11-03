# TabM: fold-local artifact / group-stat features and original-data reference features

Five GPU TabM models (pytabkit `TabM_D_Classifier`). Three share one fold-local feature recipe (`lib/tabm_harness.py`, run
by `train.py`) and train on the competition rows plus all original rows; `train_original_priors_ngram.py` uses the original
rows only as a feature reference; `train_domain_group_source.py` is a configuration of the shared harness in
`../neural_nets/lib/shared_harness.py`.

**Shared recipe** (`train.py`, `lib/tabm_harness.py`; 112 columns). Per fold, fitted on the fold's competition training rows
and applied to the validation, test and original rows:
- the 14 columns, LapNumber / RaceProgress and TyreLife / LapNumber;
- every numeric column and the two ratios floored and factorized as a category; row counts of the Driver, Compound, Race,
  Year and PitStop values; RaceProgress in 200 quantile bins; Race x Compound and Race x Year;
- **artifact digits** (categories) of LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress: integer digits 1-3,
  decimal digits 1-2 and the 4 lowest digits of round(100 |x|);
- **source flag** `SOURCE_is_original`;
- **group stats** over the fold's training + validation + test + original rows: group size, value minus group mean and
  z-score of LapTime (s), LapTime_Delta, Cumulative_Degradation, TyreLife within Race, Compound, Driver, Race x Compound;
- scikit-learn `TargetEncoder` (smooth auto, 5-fold cross fitting) of Race x Compound and Race x Year.

Training rows of fold k: the competition rows of the other 4 folds followed by all 101,371 original rows. TabM (`tabm`
architecture, k 32, 4 blocks, piecewise-linear numeric embeddings, dropout 0.1, weight decay 3e-5, batch 512, AMP, gradient
clipping 1), string columns as categoricals, up to 55 epochs with patience 9 on the validation fold's AUC, `n_cv` 2 (two
models on the same split with different seeds, averaged); seed 42 + (k + 1) + offset. Test = mean of the 5 fold models.

**`train_original_priors_ngram.py`** (138 columns): per fold, with the fold-matched original subset (the original rows outside
fold k of the same StratifiedKFold on the original target) as reference - value frequencies, degradation / lap-time
relations, race-context flags, the original target mean of every value, the position of Cumulative_Degradation within the
original positive / negative / per-Compound / per-Race distributions (percentile ranks, z-scores, quartile distances),
LapTime / degradation residuals to the original Compound / Race means, 15 digit features of each of LapTime (s),
LapTime_Delta, Cumulative_Degradation, and nested target-mean encodings (inner 5 folds on the training rows) of the 14 base
columns and of the 19 bigrams / trigrams of Driver, Race, Compound, Year, PitStop, Stint. Numerics standardized,
categoricals ordinal-encoded. TabM `tabm-mini-normal`, k 32, 3 blocks x 256, d_embedding 24, dropout 0.2, lr 1e-3,
weight decay 1e-3, batch 512, up to 50 epochs, patience 10; seed 42 + (k + 1).

**`train_domain_group_source.py`**: the shared feature / training harness of `../neural_nets` with global features (fitted on
all competition rows): the harness base block, domain features (estimated total laps, laps remaining, tyre-age /
pit-window / degradation ratios, race-phase / tyre-life / position bins), the source flag and the group stats over
train + test + original rows (112 columns, no target encodings); training rows = the competition rows of the other 4 folds +
the fold-matched original subset. TabM `tabm-mini`, k 16, 2 blocks x 256, d_embedding 8, dropout 0.18, lr 2.5e-4, batch 512,
8 epochs (best epoch on the validation fold); seed 42 + (k + 1).

```
python train.py tabm_group_stats_digits_d448
python train.py tabm_group_stats_digits_d512
python train.py tabm_group_stats_digits_d512_reseed
python train_original_priors_ngram.py
python train_domain_group_source.py
```
`train.py` takes `--folds 0,1` to run a subset of folds (e.g. one process per GPU; finished folds are kept in `parts/` and the
pair is written when all 5 are done); one recipe takes about an hour for 5 folds on one GPU. `train_original_priors_ngram.py`
takes about 50 minutes and `train_domain_group_source.py` about 11 minutes. Each run writes `preds/<pair>_{oof,test}.npy`
(probabilities). `train_domain_group_source.py` imports `../neural_nets/lib/shared_harness.py` (keep the folders side by side);
nothing has to run first. Reads only `data/`.

| Pair | Model | OOF AUC |
|---|---|---:|
| tabm_group_stats_digits_d448 | shared recipe; d_block 448, d_embedding 24, 64 embedding bins, lr 3.5e-4; seed offset 6700 | 0.953361 |
| tabm_group_stats_digits_d512 | shared recipe; d_block 512, d_embedding 32, 96 embedding bins, lr 3.2e-4; seed offset 7500 | 0.953360 |
| tabm_group_stats_digits_d512_reseed | the d512 model with seed offset 8900 | 0.953371 |
| tabm_original_priors_ngram_te | TabM-mini-normal on fold-local original priors / distribution / digit features and nested TEs of base and n-gram keys | 0.949046 |
| tabm_mini_domain_group_source | TabM-mini on global domain + group-stat features with a source flag, fold-matched original rows | 0.951176 |

Requirements: pytabkit, torch (CUDA), scikit-learn (>= 1.3 for `TargetEncoder`), pandas, numpy.
