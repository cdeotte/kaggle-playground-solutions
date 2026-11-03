# Neural networks: factorization machines, DeepFM, NAM, GraphSAGE, liquid NN, TabNet, TabTransformer, FT-Transformer, GANDALF, DAE

Thirteen models: nine members of stack A and four of stack B (the two stacks of `stack/`). One script per model; the
field-aware FM and the FM are two presets of `train_fm.py`. Every script uses the repo folds (`common.data.folds`),
saves each finished fold in `parts/` and, once all 5 folds are done, writes `preds/<pair>_{oof,test}.npy`:
(630000, 3) / (270000, 3) softmax probabilities in class order Low, Medium, High; test = mean of the 5 fold models.
`--folds 0,1` runs a subset of the folds. GANDALF, both TabNets and the full-batch GraphSAGE seed every fold; the other
scripts seed once before the fold loop and reproduce their pair only when all folds run in order in one process.
Original rows (`irrigation_prediction.csv`) are training rows only in the two factorization machines and the
TabTransformer; elsewhere they only shape features or are not used.

A recurring feature in stack B is a rule score: 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] +
[Temperature_C > 30] + [Wind_Speed_kmh > 10] - 2 [stage Harvest] - 2 [stage Sowing] - [mulch Yes], with is_low
(score <= 0) and is_high (score > 3), plus "original priors": per class, the share of that class among the original
rows of each categorical level.

## Stack A

**Factorization machines** (`train_fm.py ffm`, `train_fm.py fm`). 35 fields: the 8 categoricals (label-encoded on
train + test + original), 11 bin fields (the sorted distinct original values of each numeric cut into 96-256
equal-count chunks; a value takes the chunk whose value range holds it, else chunk 0 below the first chunk and the last
chunk otherwise), the first decimal digit of 5 numerics, and the 11 numerics as numeric fields (standardized per fold).
All original rows are appended to every fold's training rows.
- `ffm`: field-aware FM, k = 16 per (feature, other field), per-class linear terms; the scalar interaction sum enters
  the 3 logits through a learned 1 -> 3 map.
- `fm`: FM with k = 32 and per-class linear terms; the scalar interaction sum is added to each of the 3 logits.

Both: class-weighted cross-entropy, AdamW lr 2e-3, weight decay 1e-6, batch 4096, up to 30 epochs, early stopping on
the held-out balanced accuracy (patience 5); seed 42.

**GANDALF** (`train_gandalf.py`). pytorch_tabular GANDALF with the library defaults on the 8 categoricals and the 11
numerics, nothing else. lr 3e-4, AdamW, batch 4096, up to 30 epochs, early stopping on the held-out loss (patience 5);
the best checkpoint predicts. Checkpoints and Lightning logs go to `cache/gandalf/`.

**DeepFM** (`train_deepfm.py`). 44 token fields: the categoricals; every numeric snapped to the nearest original value;
original-quantile bins (up to 100) of every numeric; floor(5 x original mean label code) of every categorical; 6 crosses
(Crop x Season, Water_Source x Irrigation_Type, snapped Rainfall x Season, snapped Previous_Irrigation x
Irrigation_Type, snapped Soil_Moisture x stage, Crop x Season x Irrigation_Type). One shared vocabulary, 16-d
embeddings; linear + FM-interaction (projected to 3 logits) + MLP 128-64-32 (BatchNorm, ReLU, dropout 0.3). Class-
weighted cross-entropy, AdamW lr 3e-3, weight decay 1e-4, batch 2048, 20 epochs; the best held-out epoch predicts.

**Neural additive model** (`train_nam.py`). One 3-logit embedding per categorical and one 1 -> 32 -> 32 -> 3 network per
continuous column, summed. Features: the raw columns; a flag for categorical profiles that never occur in the original
data; distances to the 1st and 20th nearest original rows and their ratio; 5 water / heat / salinity interactions;
10-, 50- and 100-bin original-quantile bins of every numeric (42 categoricals, 19 continuous). KL loss on 0.01-smoothed
targets, AdamW lr 3e-3, batch 1024, ReduceLROnPlateau, up to 35 epochs (patience 8 on the held-out balanced accuracy).

**Mini-batch GraphSAGE** (`train_graphsage_sampled.py`). 79 dense features (raw numerics, log1p / sqrt / inverse
transforms, water interactions, management flags, category frequencies) and 28 string columns (categoricals, fixed-edge
bins, 12 pairs, 2 triples). A cuML KNN graph (k = 20) over the standardized dense features of all train + test rows.
Batches of 8192 training rows with their 2-hop neighbourhood sampled with fanouts 6 / 4 from a neighbour window that
shifts every epoch. Embeddings + dense -> 256 -> 2 SAGEConv layers (LayerNorm, residual x 0.5) -> 64 -> 3. AdamW lr 1e-3,
fp16, up to 80 epochs; validation every 2 epochs, patience 8 checks.

**Liquid NN** (`train_liquid_nn.py`). Raw columns + water balance, temperature x humidity, irrigation load, a management
count and 8-quantile bins of Rainfall and Soil_Moisture. Categoricals ordinal-encoded and fed as numbers, continuous
columns min-max scaled per fold. 3 liquid layers of 256 units (learnable time constants, GroupNorm, dropout 0.2).
AdamW lr 1e-3, OneCycle, batch 256, bf16, up to 30 epochs (patience 10 on the held-out balanced accuracy).

**TabNet with crosses** (`train_tabnet_crosses.py`). 124 inputs: the raw columns, 8 water / heat / salinity features,
log1p / sqrt / rounded versions of 5 numerics, management flags and crosses, floor(x / 5) bins and bin crosses,
original count and mean label code of 7 categoricals, train + test count and rarity flag of every string, pair and bin
column. 31 columns are TabNet categorical embeddings. n_d = n_a = 32, 5 steps, entmax, Adam lr 2e-2, StepLR (20, x0.7),
batch 8192, up to 300 epochs (patience 30 on the held-out balanced accuracy).

**TabTransformer with pair target encodings** (`train_tabtransformer.py`). 51 token columns (categoricals, integer /
decimal digits of 6 numerics, quantile bins, 3 combos, every numeric rounded as a category) and 19 numerics (raw + 8
arithmetic). Per fold, with all original rows added to the training rows: per-class target encodings of all 1,275
token pairs (inner 5-fold out-of-fold for the training rows), expanded to logit, logit^2, logit^3 and standardized
(11,475 columns); a ridge regression on [numerics, encodings] whose 3 outputs join the numerics. Network: 16-d token
embeddings -> 2 Transformer blocks, an encoding branch (BatchNorm -> 256), MLP 128-64 -> 3. Adam lr 1e-3,
ReduceLROnPlateau, batch 4096, 10 epochs; the best held-out epoch predicts. The 11,475 float32 encoding columns of all
training, held-out and test rows take about 42 GB of RAM, more while the ridge inputs are stacked.

## Stack B

**DAE + KNN + discriminator -> XGBoost** (`train_dae_xgb.py`). Features: the numerics, the rule score, train + test
frequency and log1p of every numeric, original priors, 6 categorical bigrams; a denoising autoencoder (latent 8,
hidden 96) trained on the original rows gives latent codes and reconstruction errors; the 5 nearest original rows give
distances and class shares; per fold, nested per-class target encodings of the 14 categoricals / bigrams and a
real-vs-synthetic XGBoost discriminator (original rows x6 against an equal sample of the fold's training rows) give
p(original) and its logit. Final GPU XGBoost (multi:softprob, depth 5, lr 0.03, early stopping 300 on the held-out
fold) on 126 numeric + 14 native categorical columns.

**FT-Transformer** (`train_ft_transformer.py`). pytorch-frame FTTransformer (channels 128, 4 layers; embedding encoder
for categoricals, linear-periodic encoder for numericals) on the raw columns, the rule score with its 4 threshold flags,
8 categorical bigrams, floor-5 bins of the 4 rule numerics and original priors (42 numerical, 20 categorical columns).
Class-weighted cross-entropy with label smoothing 0.01, AdamW lr 5e-4, OneCycle, batch 1024, bf16, up to 18 epochs
(patience 4 on the held-out balanced accuracy).

**Full-batch GraphSAGE** (`train_graphsage_full.py`). The raw columns, the rule score and original priors (38 numeric
columns, standardized; 8 categorical embeddings). A cuML KNN graph (k = 20) over all 900,000 train + test rows; every
fold trains on the whole graph with the loss on its training rows. 3 SAGEConv(256) layers with LayerNorm and residual
connections, head 128. Class-weighted cross-entropy with label smoothing 0.01, AdamW lr 1e-3, up to 80 full-graph steps;
validation every 5 steps (patience 10 checks).

**TabNet** (`train_tabnet_formula.py`). 68 inputs: the 8 categoricals (TabNet embeddings, dim 3), the numerics with
log1p and sqrt, the rule score and original priors. n_d = n_a = 32, 5 steps, sparsemax, Adam lr 2e-2, StepLR (10, x0.9),
batch 8192, class-weighted sampling of the training rows, up to 300 epochs (patience 30 on the held-out balanced
accuracy).

Each script's docstring has the full details (features, hyper-parameters, seeds).

## Commands

```
python train_fm.py ffm
python train_fm.py fm
python train_gandalf.py
python train_deepfm.py
python train_nam.py
python train_graphsage_sampled.py
python train_liquid_nn.py
python train_tabnet_crosses.py
python train_tabtransformer.py
python train_dae_xgb.py
python train_ft_transformer.py
python train_graphsage_full.py
python train_tabnet_formula.py
```
Every command takes `--folds 0,1,2,3,4` (the default). No other repo folders are needed.

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| ffm_orig_bins_decimal_digits_a | A | field-aware FM (k 16) on label-encoded categoricals, original-value bins, first decimal digits and standardized numerics; original rows added | 0.968381 |
| fm_orig_bins_decimal_digits_a | A | FM (k 32) on the same 35 fields; original rows added | 0.969510 |
| gandalf_raw_features_a | A | pytorch_tabular GANDALF on the raw columns | 0.946313 |
| deepfm_snapped_quantile_tokens_a | A | DeepFM on 44 token fields: categoricals, original-snapped numerics, original-quantile bins, original label-mean deciles, crosses | 0.975683 |
| nam_orig_density_multires_bins_a | A | neural additive model with unseen-profile flag, original-neighbour distances, interactions, multi-resolution original-quantile bins | 0.945443 |
| graphsage_sampled_knn_graph_a | A | mini-batch GraphSAGE (2 layers, sampled neighbourhoods) on a train + test KNN graph, dense engineered features + categorical embeddings | 0.960709 |
| liquid_nn_water_balance_bins_a | A | liquid NN (3 x 256) on the raw columns, water-balance / load features and quantile bins | 0.959375 |
| tabnet_crosses_counts_orig_priors_a | A | TabNet (entmax) on numeric transforms, crosses, bins, train + test counts and original priors | 0.961107 |
| tabtransformer_pair_te_logit3_ridge_a | A | TabTransformer on 51 digit / bin / rounded tokens + 1,275 pair target encodings (logit^1-3) + ridge outputs; original rows added | 0.973951 |
| dae_knn_discriminator_xgb_b | B | GPU XGBoost on DAE latent / reconstruction, nearest-original, nested target-encoding and real-vs-synthetic discriminator features | 0.962718 |
| ft_transformer_formula_pairs_orig_priors_b | B | pytorch-frame FT-Transformer on the raw columns, rule-score flags, bigrams, floor-5 bins and original priors | 0.971540 |
| graphsage_transductive_knn_graph_b | B | full-batch 3-layer GraphSAGE on a train + test KNN graph with rule-score features and original priors | 0.961385 |
| tabnet_formula_orig_priors_b | B | TabNet (sparsemax, class-weighted sampling) on the raw columns, log1p / sqrt, rule score and original priors | 0.964900 |

Packages (versions of our environment): torch (CUDA) 2.x, numpy 2.2, pandas 2.3 (string columns are detected as object
dtype), scikit-learn 1.6; pytorch_tabular 1.2 with pytorch_lightning (`train_gandalf.py`); pytorch-tabnet 4.1 (both
TabNets); pytorch-frame 0.3 (`train_ft_transformer.py`); torch_geometric 2.7 and RAPIDS cuML 26.02 (both GraphSAGE
scripts); xgboost 3.2 with CUDA (`train_dae_xgb.py`).
