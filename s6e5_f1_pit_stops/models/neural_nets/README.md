# Neural networks: embedding nets, factorization machines, transformers, TabNet and other custom networks

Fourteen models. Seven are configs of one shared fold harness, `lib/shared_harness.py`, run with `train.py <pair>`. The
harness is also used by `../tabm` and `../classic`, so keep the folders side by side. The other seven are standalone
scripts, `train_*.py`. Every model uses the repo folds (`common.data.folds`) and saves each finished fold in `parts/`.
Once all 5 folds are done, it writes `preds/<pair>_{oof,test}.npy` (probabilities; test = mean of the 5 fold models).
`--folds 0,1` runs a subset of the folds. Scripts that seed once before the fold loop reproduce their pair only when all
folds run in order in one process.

## Harness models (`train.py`, `lib/shared_harness.py`)

**Features** (flags of `ExpConfig`; the harness docstring has the details):
- **base block** (every model):
  - the 14 columns;
  - LP_RP = LapNumber / (RaceProgress + 1e-6) and TL_LP = TyreLife / max(LapNumber, 1);
  - a categorical code of floor(value) of every numeric column (unseen -> -1);
  - counts of the string columns and of Year / PitStop;
  - a 200-bin quantile bin of RaceProgress;
  - the combo codes Race x Compound and Race x Year.

  These are fitted on all train rows, except in `emb_resnet_periodic_group_stats_all_original`, which fits them on each
  fold's synthetic training rows.
- **domain** (added before the base block): 16 race-state ratios / products:
  - estimated total laps = LapNumber / RaceProgress, and laps remaining;
  - tyre-age ratio, LapNumber / (TyreLife + 1), LapNumber - TyreLife, TyreLife x progress, TyreLife / laps remaining;
  - degradation and lap-time delta per tyre lap (signed and absolute), |delta|, |cumulative degradation|;
  - |position change|, position x progress, stint x TyreLife;
  - race-phase, tyre-life and position bins.
- **group stats**: within Race, Compound, Driver and Race x Compound, the group size plus, for LapTime, LapTime_Delta,
  Cumulative_Degradation and TyreLife, the value minus the group mean and the z-score. They are computed over the
  train, test and original rows together.
- **extended group stats**: the same (plus RaceProgress) within Driver x Race, Driver x Compound, Race x Year,
  Race x Compound x Stint and Compound x Stint.
- **source flag**: 1 on original rows.
- **combo TE** (per fold): sklearn `TargetEncoder` (5 inner folds, smooth 'auto', seed 42 + fold) of the two combo
  codes, fitted on the fold's training rows.

**Original rows** are appended to every fold's training rows with weight 1. Either all of them are added, or only the
fold-matched 4/5: the original rows whose `common.data.orig_folds` id is not the held-out fold.

**Column typing** (per fold): string columns and numerics with < 100 distinct values become embeddings (vocabulary of
the training rows; embedding dim round(2 card^0.25), clipped to 4-48). The other numerics are QuantileTransformer
(normal)-scaled, except TabNet, which uses StandardScaler.

**Training** of the PyTorch nets:
- AdamW, OneCycle (pct_start 0.12, div_factor 12, final_div_factor lr / 2e-5);
- BCE with label smoothing, mixed precision, gradient clip 1, batch 4096;
- the held-out fold's AUC after every epoch drives early stopping, and the best epoch predicts the held-out and test
  rows.

Seeds are 42 + fold (folds 1-5) + offset. Probabilities are clipped to [1e-7, 1 - 1e-7].

**Models:**
- `emb_resnet_periodic_group_stats_all_original`:
  - Network: a residual MLP with an input projection to 640, then 5 pre-LayerNorm residual blocks (hidden 1280, SiLU,
    dropout 0.08).
  - Inputs: the embeddings (dropout 0.02), periodic numeric embeddings (8 sin / cos frequencies per numeric,
    LayerNorm) and the raw numerics. The low-cardinality numerics enter both as embeddings and as numerics.
  - Features: base block fitted per fold, combo TE, group + extended group stats, source flag, all original rows.
  - Training: lr 6e-4, weight decay 1.2e-4, 46 epochs, patience 7, label smoothing 0.01, seed offset 2200, 2
    persistent data-loader workers.
- `tabtransformer_domain_group_stats_fold_original`:
  - Network: TabTransformer. Every categorical is a 96-d token plus a field embedding, passed through 2 pre-norm
    Transformer layers (4 heads, FFN 256, GELU) and mean-pooled. The pooled vector, a projection of the numerics and
    the raw numerics go to a head (384, SiLU).
  - Features: domain, group stats, source flag; fold-matched original rows.
  - Training: lr 7e-4, dropout 0.07, label smoothing 0.008, 18 epochs, patience 5.
- `tabtransformer_medium_domain_group_stats_all_original`: the same network with d 128, 3 layers, FFN 384 and head 512.
  Dropout 0.09, lr 5.5e-4, 15 epochs; the same features, with all original rows.
- `tabnet_combo_te`:
  - Model: pytorch-tabnet `TabNetClassifier` with n_d = n_a = 32, 3 steps, gamma 1.4, sparsity 1e-5 and entmax masks.
  - Training: AdamW lr 2e-3, weight decay 1e-5, StepLR (10 epochs, x0.9), batch 8192, virtual batch 1024, up to 35
    epochs with patience 8 on the held-out AUC; seed 42 + fold.
  - Inputs: the base block + combo TE, with categorical codes as numbers and standardized numerics; fold-matched
    original rows.
- `ffm_domain_group_stats_all_original`:
  - Network: a field-gated FM network. 16-d embeddings are scaled by a learned per-field gate. The flattened
    embeddings, the FM pairwise-interaction vector of the first 26 fields (plus a projection of the numerics) and the
    raw numerics feed an MLP 512-384-192.
  - Training: dropout 0.12, weight decay 2e-4, lr 6e-4, label smoothing 0.012, 16 epochs.
  - Features: domain, group stats, source flag; all original rows.
- `graph_mlp_domain_extended_group_stats`:
  - Network: an attention graph MLP with 128-d field embeddings and one round of scaled dot-product attention messages
    between the fields. The mean embedding, the mean message, a projection of the numerics and the raw numerics feed
    an MLP 512-256-128.
  - Training: lr 7e-4, 18 epochs, patience 4.
  - Features: domain, group + extended group stats; fold-matched original rows.
- `fm_domain_group_stats_fold_original`:
  - Network: a factorization machine. Linear terms plus pairwise interactions of 48-d embeddings and of one projected
    numeric vector; the interaction vector and the raw numerics feed a head (384).
  - Training: dropout 0.1, weight decay 2e-4, lr 6e-4, label smoothing 0.012, 16 epochs.
  - Features: domain, group stats, source flag; fold-matched original rows.

## Standalone scripts

**residual_embedding_nn_original_priors_te** (`train_residual_prior_nn.py`). Features are built once on the train,
test and original rows; the original rows are not trained on.
- Columns: the 14 columns, with the string columns and numerics with < 100 distinct values as categoricals.
- 18 race-state relationships: tyre life / lap, degradation per tyre lap, progress products, pit-window flags, ...
- log1p / sqrt / inverse / rank / frequency transforms of up to 36 numerics.
- Round / floor / fraction / decimal-digit categoricals of 8 high-cardinality numerics, 12-quantile bins, and 2- and
  3-way crosses.
- Log counts and train / original frequency drift of the first 80 categoricals; original target means (smoothing 25)
  and log counts of the first 50.
- Distances to the two original class centroids.
- Per fold: target encodings of the first 60 categoricals (smoothing 20, 5 inner folds, seed 142 + fold).

Model and training:
- Network: batch-normed numerics + embeddings of the first 80 categoricals -> 384 -> 4 residual blocks (768, GELU,
  dropout 0.15, post-LayerNorm) -> 192 -> 1.
- BCE with label smoothing 0.02, AdamW lr 1e-3, weight decay 1e-4, batch 2048, gradient clip 1.
- ReduceLROnPlateau on the held-out AUC (factor 0.5, patience 3); up to 40 epochs with early stopping patience 8, and
  the best epoch predicts.
- Seed 42 + fold.

**tabular_resnet_original_manifold_priors** (`train_resnet_manifold.py`). The original rows are not trained on.
- Categoricals (factorized on train + test):
  - Driver, Compound, Race and the numerics with < 100 distinct values;
  - LapTime, LapTime_Delta, Cumulative_Degradation and RaceProgress rounded to 2 and 3 decimals, plus their
    32-quantile rank bins;
  - 6 combos.
- Numerics (standardized per fold):
  - the high-cardinality raw numerics;
  - sin / cos (periods 8, 12, 24) of LapNumber, TyreLife, LapTime_Delta and RaceProgress;
  - 7 F1 support features;
  - the original-data target mean and log count of every categorical and high-cardinality numeric;
  - 12 PCA and 12 Gaussian random-projection components fitted on the original rows (seed 42).

Model and training:
- Network: embeddings (dim 12) + numerics -> 256 (BatchNorm, ReLU) -> 4 residual blocks (dropout 0.25) -> 2 logits.
- KL-divergence on 0.02-smoothed targets, AdamW lr 2e-3, weight decay 1e-4, batch 1024.
- ReduceLROnPlateau (factor 0.5, patience 3, min lr 1e-5); up to 35 epochs with patience 8 on the held-out AUC, and the
  best epoch predicts.
- Seeded once (42) before the folds.

**dae_knn_discriminator_xgb** (`train_dae_xgb.py`). GPU XGBoost on features from a denoising autoencoder, nearest
original neighbours and a real-vs-synthetic discriminator.

Label-free features, built once on the train, test and original rows:
- Driver, Compound, Race and the low-cardinality numerics as string categoricals.
- 10 ratio / interaction features and 6 categorical pair keys.
- Exact `%.6f` value tokens of the numerics and 4 token pairs.
- Six numerics snapped to the nearest original value (the value, a `%.3f` token, a 2-decimal token).
- Two row fingerprints.

Per fold (seeds 42 + fold):
- Ordinal codes and standardized numerics fitted on the fold's training rows.
- Nested target encodings (5 inner folds, smoothing 10) and frequencies of every key; original-only target mean /
  count / seen flag per key; whether the row's fingerprints occur in the original data.
- A DAE (128 hidden, 16 latent; Gaussian noise 0.02 and 3% masking; SmoothL1, AdamW lr 1e-3, batch 2048, 25 epochs)
  trained on the original rows. It gives reconstruction errors and the 16 latent coordinates.
- The 5 nearest original rows in the base and the latent space: distances, density, the neighbours' original target
  and categorical agreement.
- An XGBoost real-vs-synthetic discriminator (original rows x4 against the fold's training rows). It gives p(real),
  its logit, |p - 0.5|, and their products with the first 4 latent coordinates.

Final model: XGBoost (depth 5, lr 0.025, up to 4000 trees, min_child_weight 10, subsample / colsample 0.88, lambda 3,
early stopping 200 on the held-out fold), trained on the fold's training rows only. Python / numpy / torch are seeded
once (2026) before the folds.

**ffm_original_bins_decimal_digits** (`train_ffm.py`). A field-aware factorization machine on 22 fields:
- 9 columns label-encoded on train + test + original: Driver, Compound, Race, Year, PitStop, LapNumber, Stint, Position,
  Position_Change.
- 5 bin fields: TyreLife, LapTime_Delta, Cumulative_Degradation, LapTime and RaceProgress, each cut into
  100 / 512 / 2048 / 512 / 512 equal-count chunks of its distinct original-data values.
- The first decimal digit of LapTime_Delta, Cumulative_Degradation and LapTime.
- The 5 continuous columns as numeric fields, standardized on each fold's training rows.

Model and training: k = 16 per field pair plus linear terms, BCE, AdamW lr 2e-3, weight decay 1e-6, batch 4096, up to
30 epochs with early stopping on the held-out AUC (patience 5). All original rows are appended once to every fold's
training rows. Seed 42 is set once (deterministic cuDNN).

**liquid_nn_f1_state_original_priors** (`train_liquid_nn.py`). Features are built once on train + test, with the
original rows as reference only (not trained on):
- The 14 columns.
- 6 race-state features: LapNumber - TyreLife, (|LapTime_Delta| + Cumulative_Degradation) / (TyreLife + 1), ...
- 8-quantile bins of 7 numerics, with edges from train + test + original.
- The 4 high-cardinality numerics rounded to 2 and to 3 decimals.
- 6 combo keys.
- The original-data target mean and log1p count of every categorical, key and high-cardinality numeric.

Per fold:
- Columns that are constant on the training rows are dropped.
- Categoricals are ordinal-encoded and fed as numbers.
- Continuous columns are min-max scaled; the scaler is fitted on the training values plus N(0, 1e-5) noise (seed
  42 + fold).

Model and training:
- Network: 3 liquid layers of 256 units, h <- h + (SiLU(W x + U h) - h) / tau with a learnable tau per unit. Each is
  followed by a residual connection when the shapes match, GroupNorm(4) and dropout 0.2; linear 2-class readout.
- Cross-entropy, AdamW lr 1e-3, weight decay 1e-4, OneCycle (20% warm-up), batch 256, gradient clip 0.1, bf16.
- Up to 30 epochs with early stopping on the held-out AUC (patience 10); the best epoch predicts.
- Seeded once (42) before the folds.

**ft_transformer_snapped_simple_features** (`train_ft_transformer.py`). A pytorch_tabular FT-Transformer; no original
rows. Inputs, built per fold:
- 9 categoricals: Driver, Compound, Race, Year, PitStop, LapNumber, Stint, Position, Position_Change.
- The 5 continuous columns, with missing values set to the training median.
- For each continuous column, a category of the value snapped to the nearest value seen at least 25 times in the
  fold's training rows, plus a rare flag.

Model and training: embed dim 64, 4 attention blocks, 8 heads, attention / FF dropout 0.1, FF multiplier 4; AdamW lr
3e-4, batch 4096, up to 200 epochs with early stopping on the held-out loss (patience 20, min delta 1e-4); the best
checkpoint is restored. Seed 42 for every fit. Probabilities are clipped to [1e-7, 1 - 1e-7].

**gandalf_artifact_combos_original_priors** (`train_gandalf.py`). pytorch_tabular GANDALF (library defaults: 6 GFLU
stages, initial feature sparsity 0.3). Inputs:
- Driver, Compound, Race and the numerics with < 100 distinct values, as categoricals.
- 6 race-state features.
- The 4 high-cardinality numerics rounded to 1 and to 2 decimals, plus their 32-bin rank keys.
- 6 combo keys.
- The original-data target mean and log1p count of every categorical, key and high-cardinality numeric, from all
  original rows (which are not trained on).

Training: AdamW lr 3e-4, batch 4096, up to 30 epochs with early stopping on the held-out loss (patience 5, min delta
1e-4); the best checkpoint is restored. Seed 42 for every fit.

Both pytorch_tabular scripts write their checkpoints and logs to `cache/`.

## Commands

```
python train.py emb_resnet_periodic_group_stats_all_original
python train.py tabtransformer_domain_group_stats_fold_original
python train.py tabtransformer_medium_domain_group_stats_all_original
python train.py tabnet_combo_te
python train.py ffm_domain_group_stats_all_original
python train.py graph_mlp_domain_extended_group_stats
python train.py fm_domain_group_stats_fold_original
python train_residual_prior_nn.py
python train_resnet_manifold.py
python train_dae_xgb.py
python train_ffm.py
python train_liquid_nn.py
python train_ft_transformer.py
python train_gandalf.py
```
Every command takes `--folds 0,1,2,3,4` (the default). No other repo folders are needed.

| Pair | Model | OOF AUC |
|---|---|---:|
| emb_resnet_periodic_group_stats_all_original | residual MLP (d 640, 5 blocks) on embeddings + periodic numeric embeddings; per-fold base block, combo TE, group + extended group stats, source flag, all original rows | 0.948923 |
| tabtransformer_domain_group_stats_fold_original | TabTransformer (d 96, 2 layers) on base + domain + group stats + source flag; fold-matched original rows | 0.947101 |
| tabtransformer_medium_domain_group_stats_all_original | TabTransformer (d 128, 3 layers) on base + domain + group stats + source flag; all original rows | 0.947160 |
| tabnet_combo_te | TabNet (n_d = n_a = 32, 3 steps) on the base block + Race x Compound / Race x Year target encodings; fold-matched original rows | 0.937210 |
| ffm_domain_group_stats_all_original | field-gated FM network (dim 16, MLP 512-384-192) on base + domain + group stats + source flag; all original rows | 0.949178 |
| graph_mlp_domain_extended_group_stats | attention graph MLP over field embeddings (d 128) on base + domain + group + extended group stats; fold-matched original rows | 0.947632 |
| fm_domain_group_stats_fold_original | factorization machine (dim 48, head 384) on base + domain + group stats + source flag; fold-matched original rows | 0.947381 |
| residual_embedding_nn_original_priors_te | residual embedding MLP on dense race-state features, original priors / centroid distances, inner-fold target encodings | 0.939197 |
| tabular_resnet_original_manifold_priors | tabular ResNet on cyclic / support features, artifact keys, original priors and an original-data PCA / random-projection manifold | 0.940993 |
| dae_knn_discriminator_xgb | GPU XGBoost on DAE (original rows) reconstruction / latent, nearest-original, nested TE / frequency / original-stat and real-vs-synthetic discriminator features | 0.946046 |
| ffm_original_bins_decimal_digits | field-aware FM (k 16) on label-encoded categoricals, original-value bins, first decimal digits and 5 standardized numerics; all original rows | 0.943717 |
| liquid_nn_f1_state_original_priors | liquid NN (3 x 256) on race-state features, quantile bins, rounded / combo keys, original priors | 0.893791 |
| ft_transformer_snapped_simple_features | pytorch_tabular FT-Transformer on the raw columns + fold-local snapped-value / rare-flag categories | 0.933224 |
| gandalf_artifact_combos_original_priors | pytorch_tabular GANDALF on race-state features, rounded / rank keys, combo keys, original priors | 0.937605 |

Packages: torch (CUDA), scikit-learn, pandas, numpy; pytorch-tabnet (`tabnet_combo_te`); xgboost (CUDA,
`dae_knn_discriminator_xgb`); pytorch_tabular 1.2 with pytorch_lightning (`ft_transformer_snapped_simple_features`,
`gandalf_artifact_combos_original_priors`).
