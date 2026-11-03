# Neural networks

## Embedding MLPs, residual MLPs, TabTransformers, FT-Transformers, a tokenized CNN, NAM, IFAN and a prototype network

Eighteen models in fifteen scripts; `train_emb_mlp_snapped.py` (2 presets) and `train_engineered_emb_nets.py`
(3 presets) hold several models each. Every script reads `data/` through `common/data.py`, cross-validates on the shared
5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)` on the target, identical to `common.data.folds`), runs all
of its folds in one process and writes `preds/<pair>_{oof,test}.npy` (churn probabilities). Unless stated otherwise a
network early-stops on the AUC of the held-out fold and its best epoch predicts. Full feature lists and
hyper-parameters are in each docstring.

The 7,043 original IBM rows ("original data") are training rows only in `train_tabtransformer_pair_te.py` (appended
to every fold's training rows); elsewhere they only feed features (label statistics, nearest values, fitted
projections) or are not used. "Snapped value" (`train_emb_mlp_snapped.py`, `train_engineered_emb_nets.py`,
`train_pytorch_tabular_ft_transformer.py`): for each of tenure, MonthlyCharges and TotalCharges, the value moved to the
nearest value that occurs at least 25 times in the fold's training rows, plus a rare flag.

**Embedding MLPs on snapped values** (`train_emb_mlp_snapped.py <pair>`). 22 categorical embeddings (the 16
categoricals, the 3 snapped values and 3 rare flags; vocabularies of the fold's training rows) + the 3 standardized
numerics -> MLP 512-256 (ReLU) -> 1 logit; BCE on 0.02-smoothed labels, AdamW, batch 8192, up to 80 epochs (patience
10).
- `emb_mlp_snapped_numerics`: embedding dim sqrt(cardinality) (4-128), dropout 0.25, constant lr 2e-3.
- `emb_mlp_snapped_numerics_layernorm_cosine`: embedding dim 1.8 cardinality^0.25 (4-64), LayerNorm, dropout 0.3, lr
  1e-3 with cosine annealing, weight decay 3e-4, gradient clipping 1.

**Networks on engineered features** (`train_engineered_emb_nets.py <pair>`). Features built once on train + test +
original: frequency, rank, log1p / sqrt / inverse transforms of the numerics; 7 charge relationships (TotalCharges -
tenure x MonthlyCharges, ratios, products); service counts and yes / no / other flags; tenure and quantile bins, 6 pair
crosses and a triple (as embeddings); counts and rare flags of the 26 categorical columns; their original-data churn
means (plus those of the 3 numerics' values) and train-vs-original frequency drift. 32 embeddings (26 + the snapped
values and rare flags of the fold) and 174 standardized numerics. BCE on 0.02-smoothed labels, AdamW, batch 256, up to
80 epochs, cosine annealing, gradient clipping 1.
- `emb_mlp_engineered_original_priors_drift`: MLP 512-256 (LayerNorm, ReLU, dropout 0.3); lr 2.5e-5.
- `emb_resmlp_engineered_original_priors_drift`: two residual blocks 512-512 and 256-256 (BatchNorm, SiLU, projected
  shortcuts); lr 2.5e-5.
- `tabtransformer_engineered_original_priors_drift`: a 128-d token per categorical plus one token for all numerics, 5
  Transformer encoder layers (8 heads), flattened -> 512-256 head; lr 1e-3.

**Column-pair models**. All 171 pairs of the 19 raw columns:
- `train_pair_te_mlp.py`: per fold, cuML `TargetEncoder` of every pair (smooth 0; out-of-fold over 5 inner random folds
  for the training rows), standardized -> MLP 512-256-128 (LayerNorm, ReLU, dropout 0.3); AdamW lr 2e-3, batch 2048,
  cosine annealing.
- `train_pair_embedding_mlp.py`: per fold, values seen fewer than 25 times in the training rows become a rare token;
  each pair 'a||b' is a token with its own embedding table (dim 1.6 cardinality^0.25, 4-16) -> MLP 512-256-128; AdamW lr
  2e-3, batch 1024, up to 60 epochs (patience 8).

**TabTransformer on pair target encodings** (`train_tabtransformer_pair_te.py`). 34 categoricals: the 15 string
columns, integer / decimal digits of the three numerics, 200 / 4,000-bin quantile bins of the charges and value codes
of the 4 numeric columns; 7 numerics (with 3 charge ratios). Per fold the training rows plus all original rows feed cuML
`TargetEncoder` on all 561 categorical pairs, each encoding expanded to logit, logit^2 and logit^3 (1,683 standardized
columns). 16-d categorical tokens -> 2 Transformer blocks; encodings -> BatchNorm -> 256; concatenated with the numerics
-> MLP 128-64. Adam lr 1e-3, ReduceLROnPlateau, label smoothing 0.01, batch 4096, 10 epochs.

**FT-Transformers**
- `train_pytorch_tabular_ft_transformer.py`: pytorch_tabular FTTransformer (embed dim 64, 4 blocks, 8 heads) on the 16
  categoricals, the 3 snapped values and rare flags (as categoricals) and the 3 numerics; lr 3e-4, batch 4096, up to
  200 epochs with early stopping on the held-out loss (patience 20). Checkpoints and logs go to `cache/`.
- `train_torch_frame_ft_transformer.py`: PyTorch Frame FTTransformer (64 channels, 5 layers, LinearPeriodicEncoder for
  numericals) on the 15 string columns, value codes of the 4 numeric columns, 8 interaction codes, the 4 numerics
  (min-max scaled) and the original-data churn mean of their values. AdamW lr 1e-3, OneCycle, batch 512, 6 epochs;
  the last epoch predicts.
- `train_torch_frame_ft_transformer_digits.py`: the same model family (128 channels, 4 layers) on 54 categoricals and
  158 numericals: decimal digits of the charges as written in the csv, service flags and counts, 20 charge-arithmetic
  columns, fixed bins, integer / step codes of the numerics, 16 pair crosses, and the original-data churn mean and row
  count of 35 columns and 12 pairs. Label smoothing 0.01, AdamW lr 5e-4, OneCycle over 18 epochs, bf16, batch 1024,
  patience 4.
- `train_ft_transformer_cls.py`: a PyTorch FT-Transformer (CLS token, d 128, 3 layers, 8 heads) on 20 categoricals (the
  16 columns + 4 crosses) and 48 numerics: the 3 numerics with their frequency, rank, log1p and sqrt, 3 charge
  relationships, service count, has-internet flag, length / first / last digit / fraction of the charges, and per-fold
  nested scikit-learn `TargetEncoder` values of the 20 categoricals. AdamW lr 5e-4, cosine annealing, mixed precision, batch 2048, up to 40 epochs (patience 6).

**Networks with original-data geometry features**. Features on train + test; the projections, clusters, neighbours,
bins and class statistics are fitted on the original rows. Categoricals (factorized on train + test) enter as
embeddings; batch 1024, AdamW, label smoothing 0.01-0.02, up to 35-40 epochs (patience 8-10).
- `train_tabular_resnet_manifold.py`: 12 PCA and 12 Gaussian random-projection components (original one-hot +
  numerics), sin / cos of tenure (periods 12, 24); tabular ResNet (256, 4 pre-activation residual blocks).
- `train_tokenized_cnn.py`: 16 Gaussian-mixture archetype probabilities, cosine similarity to the original churn /
  no-churn centroids, 3 charge features; each feature becomes a 16-d token and two Conv1d layers run over the token
  sequence.
- `train_nam.py`: flag for categorical profiles absent from the original data, distances to the 1st / 20th nearest
  original rows and their ratio, 10 / 50 / 100-bin original quantile bins; neural additive model (one scalar embedding
  per categorical, one 1-32-32-1 MLP per numeric).
- `train_ifan.py`: distance of each charge to the nearest original value, TotalCharges / MonthlyCharges vs tenure, the
  original churn rate of the row's KMeans cluster (50 clusters), 5 cosine kernel-PCA components; instance-wise feature
  attention network (a softmax weight per feature token, then MLP 256-128).
- `train_prototype_net.py`: Mahalanobis distances to the original churn / no-churn rows, 3 crosses; prototypical
  network (MLP encoder to a 32-d latent space, 16 learned prototypes per class, logit = log-sum-exp similarity
  difference).

**Residual MLP on dense priors** (`train_residual_mlp_dense_priors.py`, features in `lib/dense_priors_features.py`).
Needs `../snap_features` first. On train + test: the snap / decimal-string columns of `combo.csv`, frequency / rank /
transforms of the numerics and snap values, charge relationships, service counts and flags, bins, 20 crosses, charge
digits, 32 n-gram crosses, counts and rare flags of every categorical-like column, group means, original-data churn
priors / counts and class-mean distances, train / test drift ratios, and out-of-fold smoothed target means of the
categoricals, bins and a few crosses. 64 ordinal-encoded embeddings + 491 standardized numerics -> Linear(384) -> 4
residual blocks (GELU, LayerNorm) -> 192 -> 1. AdamW lr 1e-3, ReduceLROnPlateau, batch 1024, up to 40 epochs (patience
8).

Credits: `train_torch_frame_ft_transformer.py` and `train_torch_frame_ft_transformer_digits.py` adapt yekenot's public
notebook [PS S5E11 FTTransformer (PyTorch Frame)](https://www.kaggle.com/code/yekenot/ps-s5-e11-fttransformer-pytorch-frame);
the residual blocks of `emb_resmlp_engineered_original_priors_drift` follow masayakawamata's
[S5E5 ResMLP](https://www.kaggle.com/code/masayakawamata/s5e5-resmlp-cv0-05990); the TabTransformer of
`tabtransformer_engineered_original_priors_drift` follows omidbaghchehsaraei's
[TabTransformer](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-cv-0-35327-lb-0-36542);
`train_tabtransformer_pair_te.py` is a PyTorch version of include4eto's
[TabTransformer notebook](https://www.kaggle.com/code/include4eto/tabtransfomer-chatgpt-vibe-coding) for this
competition, with more digit features.

### Commands

```
python train_emb_mlp_snapped.py emb_mlp_snapped_numerics
python train_emb_mlp_snapped.py emb_mlp_snapped_numerics_layernorm_cosine
python train_engineered_emb_nets.py emb_mlp_engineered_original_priors_drift
python train_engineered_emb_nets.py emb_resmlp_engineered_original_priors_drift
python train_engineered_emb_nets.py tabtransformer_engineered_original_priors_drift
python train_pair_te_mlp.py
python train_pair_embedding_mlp.py
python train_tabtransformer_pair_te.py
python train_pytorch_tabular_ft_transformer.py
python train_torch_frame_ft_transformer.py
python train_torch_frame_ft_transformer_digits.py
python train_ft_transformer_cls.py
python train_tabular_resnet_manifold.py
python train_tokenized_cnn.py
python train_nam.py
python train_ifan.py
python train_prototype_net.py
python train_residual_mlp_dense_priors.py   # after ../snap_features/build.py
```
`train_residual_mlp_dense_priors.py` needs `../snap_features` (`python build.py` there); the other scripts need no
other folder.

| Pair | Model | OOF AUC |
|---|---|---:|
| emb_mlp_snapped_numerics | embedding MLP 512-256 on the categoricals, fold-local snapped values / rare flags and the numerics | 0.915297 |
| emb_mlp_snapped_numerics_layernorm_cosine | the same with LayerNorm, smaller embeddings, cosine lr schedule and gradient clipping | 0.916081 |
| emb_mlp_engineered_original_priors_drift | embedding MLP on engineered transforms, bins / crosses, counts, original churn means and drift ratios | 0.917578 |
| emb_resmlp_engineered_original_priors_drift | residual MLP (BatchNorm, SiLU) on the same features | 0.917505 |
| tabtransformer_engineered_original_priors_drift | TabTransformer (128-d tokens, 5 layers) on the same features | 0.909929 |
| mlp_pair_target_encoding | MLP on per-fold cuML target encodings of all 171 column pairs | 0.915907 |
| pair_embedding_mlp | MLP on one learned embedding per column pair (171 pairs, rare values bucketed) | 0.914909 |
| tabtransformer_pair_te_logit3_digits_original_rows | TabTransformer on digit / bin categoricals + logit, logit^2, logit^3 of 561 pair target encodings; original rows added | 0.917488 |
| pytorch_tabular_ft_transformer_snapped_numerics | pytorch_tabular FT-Transformer on the categoricals, fold-local snapped values / rare flags and the numerics | 0.915700 |
| torch_frame_ft_transformer_original_te_combos | PyTorch Frame FT-Transformer on value codes, 8 interaction codes and original churn means of the numerics | 0.913323 |
| torch_frame_ft_transformer_decimal_digits_original_maps | PyTorch Frame FT-Transformer on decimal digits, charge arithmetic, bins, crosses and original churn mean / count maps | 0.915617 |
| ft_transformer_cls_engineered_nested_te | CLS FT-Transformer on frequency / rank / digit features, 4 crosses and nested target encodings | 0.914055 |
| tabular_resnet_original_pca_projection | tabular ResNet on embeddings, cyclic tenure and original-data PCA / random projections | 0.912869 |
| tokenized_cnn_original_gmm_archetypes | 1D-CNN over feature tokens with original-data GMM archetype probabilities and centroid similarities | 0.913764 |
| nam_original_density_bins | neural additive model on original profile flags, nearest-original distances and original quantile bins | 0.914941 |
| ifan_original_noise_kmeans_kpca | instance-wise feature attention network on nearest-original-value noise, KMeans cluster churn rates and kernel PCA | 0.913336 |
| prototype_net_original_mahalanobis | prototypical network with Mahalanobis distances to the original classes | 0.913463 |
| residual_mlp_dense_priors | residual MLP on dense engineered features, snap / digit columns, original priors and target encodings | 0.916244 |

Packages (versions of our environment): torch (CUDA) 2.x, numpy 2.2, pandas 2.3 (string columns load as `object`;
`train_tabtransformer_pair_te.py` and `train_nam.py` find them by that dtype), scikit-learn 1.6; RAPIDS cudf / cupy /
cuML 26.02 (`train_pair_te_mlp.py`, `train_tabtransformer_pair_te.py`); pytorch_tabular 1.2 with pytorch_lightning
(`train_pytorch_tabular_ft_transformer.py`); pytorch-frame 0.3 (`torch_frame`, the two PyTorch Frame scripts).

## Factorization machines, DeepFM, kernel and self-normalizing nets, GraphSAGE, liquid NNs, TabNet, GANDALF, DAE + XGBoost

Thirteen models in eleven scripts; `train_factorization_machines.py` (presets `ffm`, `fm`) and
`train_graphsage_full_graph.py` (presets `engineered`, `simple`) hold two models each. Every script reads `data/`
through `common/data.py`, uses the shared 5 folds (`common.data.folds`), saves each finished fold in `parts/` and, once
all 5 folds are done, writes `preds/<pair>_{oof,test}.npy` (churn probabilities; test = mean of the 5 fold models).
`--folds 0,1` runs a subset of the folds. Most scripts seed once before the fold loop, so they reproduce their pair only
when all 5 folds run in order in one process. Full feature lists and hyper-parameters are in each docstring.

The 7,043 original IBM rows ("original data") are training rows only in the two factorization machines (appended once
to every fold's training rows). The DeepFM, the RFF network, the SNN, the DAE model, the embedding liquid NN and TabNet
use them only to build features; GANDALF, the GraphSAGE models and the ordinal liquid NN do not use them.

**Factorization machines** (`train_factorization_machines.py ffm|fm`). 24 fields: the 16 categoricals (label-encoded
on train + test + original), tenure / MonthlyCharges / TotalCharges cut into at most 100 / 512 / 2,048 equal-count
chunks of their distinct original values, the first decimal digit of the two charges, and the 3 numerics as numeric fields
(standardized per fold). `ffm`: field-aware FM with k = 16 per (feature, other field), batch 4096, up to 30 epochs.
`fm`: FM with k = 32, batch 8192, up to 50 epochs. Both: BCE, AdamW lr 2e-3, early stopping on the held-out AUC
(patience 5); the best epoch predicts.

**DeepFM** (`train_deepfm_snapped_tokens.py`). 41 token fields in one vocabulary: the 16 categoricals, the 3 numerics
snapped to the nearest original value, 100-bin original-quantile bins of the numerics, deciles of the original churn
rate of every categorical, and 3 crosses (Contract x InternetService, PaymentMethod x PaperlessBilling, snapped
MonthlyCharges x Contract). 16-d embeddings; linear + FM + MLP 128-64-32 (BatchNorm, ReLU, dropout 0.3). AdamW lr
3e-3, cosine schedule, batch 1024, up to 25 epochs (patience 6 on the held-out AUC).

**Random Fourier feature network** (`train_rff_kernel_net.py`). 20 categoricals (the 16 columns + 4 crosses; 8-d
embeddings) and 38 numerics: the raw numerics, the distances to the 32 centroids of a KMeans of the standardized
original numerics, and a RankGauss of the numerics fitted on the original rows (all standardized). The numerics pass a
frozen random Fourier map (1,024 features, sigma 0.5); [embeddings, RFF] -> dropout 0.3 -> 128 (BatchNorm, ReLU) -> 1.
Label smoothing 0.01, AdamW lr 2e-3, cosine schedule, batch 1024, up to 35 epochs (patience 8).

**Self-normalizing network** (`train_snn_charge_ngrams.py`). TF-IDF of 128 character 2-4-grams of the text
"MC<MonthlyCharges> TC<TotalCharges>" (vocabulary from the original + train rows), Benford probabilities of the
charges' leading digits, the mean churn label and mean distance of the 3 nearest original rows, one-hot categoricals
and the numerics, all standardized. SELU layers 256-128-64-32 (LeCun-normal init, alpha dropout 0.05). Label
smoothing 0.015, AdamW lr 1e-3, ReduceLROnPlateau, batch 1024, up to 40 epochs (patience 8).

**DAE + nearest original rows + discriminator -> XGBoost** (`train_dae_knn_discriminator_xgb.py`). Label-free columns
on the train, test and original rows: 5 charge relations and missing flags, 6 category pairs, exact `%.6f` value
tokens of the 3 numerics and their 3 pairs, and 2 row fingerprints (tokens + key categoricals). Per fold:
- ordinal codes of the 29 string columns and the 13 standardized numerics;
- for 20 keys (6 categoricals, the pairs, tokens, token pairs and fingerprints): nested target encodings (inner 5
  folds, smoothing 10), counts and frequencies in the training rows, and the original-data churn mean, count,
  frequency and seen flag; whether the fingerprints occur in the original data;
- a denoising autoencoder (42 -> 128 -> 128 -> 16, Gaussian noise + masking, SmoothL1) trained on the original rows:
  reconstruction errors and the 16 latent coordinates;
- the 1 / 3 / 5 nearest original rows in the autoencoder input space and in the latent space (exact GPU search):
  distances, density, the nearest row's churn label and its categorical agreement;
- an XGBoost discriminator (original rows x6 against the fold's training rows): p(real), its logit and margin; 7
  products of these features.

Final model: GPU XGBoost on 227 columns (depth 5, lr 0.025, early stopping 200 on the held-out AUC).

**GraphSAGE** (`train_graphsage_full_graph.py engineered|simple`, `train_graphsage_sampled.py`). One cuML KNN graph
over all 848,849 train + test rows (transductive); each fold's loss uses only its training rows.
- `engineered`: 24 categoricals (the columns, tenure / charge bins, 4 category pairs and a triple) and 28 numerics
  (log / sqrt / inverse transforms, missing flags, 7 charge relations, service counts); every row linked to its 20
  nearest rows in the standardized numeric space. Full-graph training, 3 SAGEConv(256) layers, up to 80 epochs
  (patience 10).
- `simple`: 22 categoricals (the columns plus, per numeric, the value snapped to a value seen at least 25 times in
  train and a rare flag) and the 3 standardized numerics; 20 nearest rows in [one-hot categoricals, 3 x standardized
  numerics]. Full-graph training, 3 SAGEConv(192) layers, up to 200 epochs (patience 20).
- `train_graphsage_sampled.py`: the `simple` features and graph space with 8 neighbours; mini-batches of 8,192 seed
  rows with sampled 2-hop subgraphs (fanouts 6 and 4), 2 SAGEConv(128) layers, fp16, up to 80 epochs (validation every
  2 epochs, patience 8 checks).

All three: categorical embeddings + numerics -> Linear -> SAGEConv layers with LayerNorm and residuals -> head -> 1
logit; BCE with label smoothing 0.01, AdamW (weight decay 3e-4), gradient clip 1; the best epoch predicts.

**Liquid NNs** (`train_liquid_nn_ordinal.py`, `train_liquid_nn_embeddings.py`). Liquid cells
h <- h + (SiLU(W x + U h) - h) / tau with a learnable time constant per unit.
- `liquid_nn_ordinal_service_bins`: the columns + a service count, 2 charge relations, tenure bins and MonthlyCharges
  octiles; categoricals ordinal-coded and fed as numbers, continuous columns min-max scaled. 3 liquid layers of 256
  (GroupNorm, dropout 0.2); AdamW lr 1e-3, OneCycle, batch 256, bf16, up to 30 epochs (patience 10).
- `liquid_nn_embeddings_digits_orig_priors`: the columns + the decimal digits of the charges as written in the csv,
  service counts, 19 charge relations / transforms, fixed and floor bins, 12 pairs, and the original-data churn mean
  and count of 29 columns and 8 pairs. Categorical embeddings + projected continuous inputs -> 4 gated liquid blocks
  (384, LayerNorm residual) -> head; label smoothing 0.01, AdamW lr 3e-4, warm-up + cosine, batch 1024, bf16, up to 40
  epochs (patience 6).

**TabNet** (`train_tabnet_snap_counts.py`). About 140 inputs: the columns; the snapped charges and snap differences
from `snap_features/cache/combo.csv` with their gaps and bins; log / sqrt / round / floor-bin transforms; 8 charge
relations; service counts; 10 crosses; the original-data count and churn rate of 7 columns; train + test counts and
rare flags of every string, cross and bin column; missing flags. The string columns and the 8 named crosses (with
their count and rare-flag columns) are TabNet embeddings. n_d = n_a = 32, 5 steps, entmax, Adam lr 2e-2 with ReduceLROnPlateau, batch 8192, up to
300 epochs (patience 30 on the held-out AUC).

**GANDALF** (`train_gandalf_raw_columns.py`). pytorch_tabular GANDALF (library defaults) on the 19 raw columns (the
15 string columns as categoricals; SeniorCitizen and the numerics continuous). lr 3e-4, AdamW, batch 4096, up to 30
epochs with early stopping on the held-out loss (patience 5); the model at the end of training predicts. Lightning
logs go to `cache/gandalf_raw_columns/`.

The two liquid NNs adapt yekenot's [PS S5E5 Liquid NN (PyTorch)](https://www.kaggle.com/code/yekenot/ps-s5-e5-liquid-nn-pytorch)
notebook (Playground S5E5, regression) to this binary task.

### Commands

```
python train_factorization_machines.py ffm
python train_factorization_machines.py fm
python train_deepfm_snapped_tokens.py
python train_rff_kernel_net.py
python train_snn_charge_ngrams.py
python train_dae_knn_discriminator_xgb.py
python train_graphsage_full_graph.py engineered
python train_graphsage_full_graph.py simple
python train_graphsage_sampled.py
python train_liquid_nn_ordinal.py
python train_liquid_nn_embeddings.py
python train_tabnet_snap_counts.py
python train_gandalf_raw_columns.py
```
Every command takes `--folds 0,1,2,3,4` (the default). `train_tabnet_snap_counts.py` needs `../snap_features` first
(`python build.py` there); the other scripts need no other folder.

| Pair | Model | OOF AUC |
|---|---|---:|
| ffm_orig_bins_decimal_digits | field-aware FM (k 16) on label-encoded categoricals, original-value bins, first decimal digits and standardized numerics; original rows added | 0.915888 |
| fm_orig_bins_decimal_digits | FM (k 32) on the same 24 fields; original rows added | 0.915541 |
| deepfm_snapped_quantile_prior_tokens | DeepFM on 41 token fields: categoricals, original-snapped numerics, original-quantile bins, original churn-rate deciles, crosses | 0.915703 |
| rff_kernel_net_orig_kmeans_rankgauss | random Fourier feature kernel network on categorical embeddings, original-data KMeans distances and RankGauss numerics | 0.913635 |
| snn_charge_char_ngrams_benford_orig_knn | self-normalizing (SELU) network on charge-string character n-gram TF-IDF, Benford probabilities, nearest-original churn votes, one-hot categoricals | 0.912932 |
| dae_knn_discriminator_token_te_xgb | GPU XGBoost on value tokens / fingerprints with nested TE and original statistics, DAE (original rows) reconstruction / latent, nearest-original and real-vs-synthetic discriminator features | 0.917661 |
| graphsage_engineered_numeric_knn_graph | full-graph 3-layer GraphSAGE on a train + test KNN graph over 28 engineered numerics, with 24 categorical embeddings | 0.913761 |
| graphsage_snapped_rare_flags_ohe_knn_graph | full-graph 3-layer GraphSAGE on a train + test KNN graph over one-hot categoricals + numerics; snapped values and rare flags as categoricals | 0.914215 |
| graphsage_sampled_snapped_rare_flags_ohe_knn_graph | mini-batch 2-layer GraphSAGE on sampled subgraphs of the same kind of graph (8 neighbours), same features | 0.915789 |
| liquid_nn_ordinal_service_bins | liquid NN (3 x 256) on the columns, a service count, charge relations and bins (ordinal-coded categoricals) | 0.912839 |
| liquid_nn_embeddings_digits_orig_priors | gated liquid NN (4 x 384) with categorical embeddings on decimal digits, bins, pairs and original churn rates | 0.915116 |
| tabnet_snap_counts_orig_priors | TabNet (entmax) on snapped charges, transforms, crosses, train + test counts and original churn rates | 0.913415 |
| gandalf_raw_columns | pytorch_tabular GANDALF on the raw columns | 0.911962 |

Packages (versions of our environment): torch (CUDA) 2.x, numpy 2.2, pandas 2.3 (`train_dae_knn_discriminator_xgb.py`
finds the string columns by their `object` dtype), scikit-learn 1.6; xgboost 3.2 with CUDA
(`train_dae_knn_discriminator_xgb.py`); torch_geometric 2.7 and RAPIDS cuML 26.02 (the GraphSAGE scripts);
pytorch-tabnet 4.1 (`train_tabnet_snap_counts.py`); pytorch_tabular 1.2 with pytorch_lightning
(`train_gandalf_raw_columns.py`).
