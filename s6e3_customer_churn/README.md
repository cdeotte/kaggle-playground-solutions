# Playground S6E3: Predict Customer Churn (1st place)

Writeup: [1st Place - GPT5.4, Gemini3.1, ClaudeOpus4.6 - KGMON Playbook!](https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm)

The competition data (594,194 train and 254,655 test rows; 19 features; binary target `Churn`; metric ROC AUC) was
generated from the 7,043-row IBM Telco customer churn dataset. Our final submission stacked 154 models: 137 models plus
17 second-level models trained on their out-of-fold predictions, combined by one logistic regression. It finished 1st.
This folder holds 100 of the 137 first-level models: every model class and the best model of each class, combined by the
same logistic regression. On their own they score private LB 0.91853, against 0.91856 for the full final submission.

All 100 models are trained on one shared 5-fold split. The features that mattered most exploit how the synthetic data
relates to the original data:
- **snap features**: each synthetic value mapped to its nearest value in the original data, plus the difference;
- **digit and decimal features** of the charges;
- **nested (inner-fold) target encodings** of categoricals, pairs / triples of columns, binned numerics and anchor keys,
  with statistics beyond the mean, and **churn priors** from the original data;
- arithmetic interactions (e.g. `TotalCharges - tenure * MonthlyCharges`), multi-scale bins, frequency encodings,
  nearest-original-row lookups, radix keys (one integer for a numeric and a categorical value), synthetic-artifact and
  projection features; several models append the original rows to the training side of each fold.

The models: XGBoost (including a pairwise ranker), CatBoost, LightGBM (gbdt / GOSS / DART / extra trees), YDF, a cuML
random forest, logistic regressions on pair target encodings, KNN feature models, RealMLP, TabM, TabICL, and neural networks
of many kinds (embedding and residual MLPs, TabTransformer, FT-Transformer, a tokenized CNN, NAM, IFAN, a prototype
network, factorization machines, DeepFM, a random-Fourier-feature network, a self-normalizing network, GraphSAGE, liquid
NNs, TabNet, GANDALF, a denoising autoencoder), plus 9 public notebooks run on our folds (`models/public_models`).

**The stacker** (`stack/stack.py`) is an NVIDIA cuML L2 logistic regression (C = 0.01) on the logits of the 100 models.
Before the logit, each model's OOF probabilities are calibrated fold by fold (each validation fold is ranked onto a shared
rank grid and every probability is replaced by the mean probability of its rank), so the 5 fold models of a model share
one scale. CV comes from the 5 folds; the test prediction is the mean of 10 fits on all train rows.

The OOF / test predictions of all 154 models of our final submission are in the Kaggle dataset
[s6e3-oof-and-test-pred-v2](https://www.kaggle.com/datasets/cdeotte/s6e3-oof-and-test-pred-v2), and the final
logistic-regression notebook is [1st place NVIDIA cuML logistic regression](https://www.kaggle.com/code/cdeotte/1st-place-nvidia-cuml-logistic-regression).

## Results

| Stack | CV | Public LB | Private LB |
|---|---:|---:|---:|
| **The 100 models in this repo** | 0.91983 | 0.91765 | **0.91853** |
| Our final submission (154 models, two levels of stacking) | 0.91986 | 0.91773 | 0.91856 |

## Models

| Folder | Pair | Model | OOF AUC |
|---|---|---|---:|
| [xgboost](models/xgboost) | xgb_original_knn5_class_distances | raw columns + 5-nearest-original churn mean + distances to the nearest original churner / non-churner | 0.915798 |
|  | xgb_pairwise_ranker_platt | XGBRanker (rank:pairwise) on the raw columns, Platt-scaled | 0.914916 |
|  | xgb_snap_radix_original_priors | stumps on snaps, decimal strings, original churn rates of the snaps, 48 radix keys | 0.915747 |
|  | xgb_union_snap_priors_drift_te | label-free union (snaps, transforms, bins, counts, original priors, drift) + nested TE of 18 columns | 0.918657 |
|  | xgb_bigram_trigram_te_stats | original-distribution and digit features, nested TE statistics, bigram / trigram TE | 0.919015 |
|  | xgb_ngram_te_binned_keys_drift | the n-gram TE recipe + binned / rounded / digit-pair keys, counts, drift, original churn rates | 0.919175 |
|  | xgb_te_stats_original_proba | original churn rates, frequencies, nested TE std / min / max / mean of categoricals and numerics | 0.918067 |
|  | xgb_monthly_anchor_te | MonthlyCharges-anchor target statistics / quantiles, category x anchor, digit-pair and category-pair TE | 0.915428 |
|  | xgb_total_round0_anchor_te | the same with a TotalCharges anchor keyed by round(TotalCharges) | 0.917728 |
|  | xgb_native_cat_tabpfn_original | native categoricals + the prediction of TabPFN models fitted on the original rows | 0.916507 |
|  | xgb_total_round1_anchor_te | TotalCharges-anchor statistics keyed by round(TotalCharges, 1) | 0.917678 |
|  | xgb_tenure_anchor_te | the same with a tenure anchor | 0.915757 |
|  | xgb_original_self_supervised | raw columns + predictions of each column by models trained on the original rows | 0.915474 |
|  | xgb_union_ab_te | two feature pipelines, nested target std / min / max and target-encoder means of their keys and pairs | 0.918667 |
|  | xgb_pairwise_combo_te_original | target means + counts of the 171 column pairs; original rows appended | 0.917745 |
|  | xgb_union_ab_total_anchor_te | two feature pipelines + TotalCharges-anchor target statistics | 0.918490 |
|  | xgb_union_ab_lgbm_port_te | two feature pipelines + target-encoded bin / digit keys | 0.918798 |
|  | xgb_union_ab_lgbm_port_prebinned | the same, pre-binned to 256 quantile bins, max_bin 16,000 | 0.918804 |
|  | xgb_snap_lgbm_port_radix_nested_te | snap / decimal-string / bin / radix / pair-key union with nested target-encoder means | 0.918645 |
|  | xgb_numcat_pair_cuml_te_digits | native categoricals + digit positions + cuML TE of 48 numeric x categorical pairs | 0.917838 |
| [catboost](models/catboost) | catboost_numcat_digits_orig_priors_drift | numeric transforms, digit / count artifacts, 65 native categoricals, original priors and drift | 0.918957 |
|  | catboost_numcat_digits_orig_rows | the same frame without original lookups, original rows appended | 0.919017 |
|  | catboost_fold_density_knn_orig_priors | string artifacts, fold-local frequency / PMI / KD-tree density / group statistics, original priors and neighbours | 0.918762 |
|  | catboost_fold_density_knn_orig_rows | the same without original lookups, original rows appended | 0.918723 |
|  | catboost_fold_artifacts_orig_priors | string-precision artifacts, fold frequencies of decimal strings, original profile match and neighbours | 0.918544 |
|  | catboost_fold_artifacts_orig_rows | the same without original lookups, original rows appended | 0.918565 |
|  | catboost_triple_cross_ranked_orig_rows | compact frame with the exact numeric triple category, original rows, depth 6 | 0.918874 |
|  | catboost_orig_knn_class_distances | raw columns + nearest-original churn rate and churner / non-churner distances | 0.915216 |
|  | catboost_snap_radix_union_nested_te | 1,770-column snap / radix / bin / count / original-prior union + nested target encodings | 0.918133 |
|  | catboost_native_cats_balanced | raw columns, native categoricals, balanced class weights | 0.915667 |
|  | catboost_native_cats_fold_stats_balanced | + fold frequency / target encodings and quantile-bin categoricals | 0.915854 |
|  | catboost_orig_dist_te_stats_ngrams_balanced | original probabilities / distributions / digits + inner target statistics, depth 5 | 0.918707 |
|  | catboost_all_categorical_orig_bins_d1 | label-encoded columns, original-value bins and first decimal digits as categoricals | 0.917963 |
|  | catboost_snap_combo_native_cats | snap combo frame with keys, n-grams, counts and original priors, max_ctr_complexity 4 | 0.918268 |
| [lightgbm](models/lightgbm) | lgbm_highcard_bins_digits_te | high-cardinality keys, counts, original priors, bins, digits and digit pairs; target encoder; 7,500 rounds | 0.918611 |
|  | lgbm_union_te | + union blocks (391 columns: transforms, flags, crosses, original priors, counts, codes, drift, group means) | 0.919025 |
|  | lgbm_union_dist_ngram_te | + original-distribution, digit and n-gram key blocks (466 columns); 10,000 rounds | 0.919138 |
|  | lgbm_union_dist_ngram_te_slow | the same features; 30,000 rounds, lr 0.0025 | 0.919164 |
|  | lgbm_union_dist_ngram_te_20k | the same features; 20,000 rounds | 0.918791 |
|  | lgbm_union_dist_ngram_te_goss | the same features; GOSS, 63 leaves, early stopping | 0.918806 |
|  | lgbm_union_dist_ngram_te_dart | the same features; DART, 31 leaves, depth 4 | 0.918742 |
|  | lgbm_extra_trees_pairs_digits_two_stage_te | category / digit pairs, roundings, bins, original priors; two-stage target encoding; extra trees | 0.918361 |
|  | lgbm_union_snap_pairs_nested_te | union with snap features, decimal strings, pair keys, original priors; nested target means | 0.918782 |
|  | lgbm_decimal_ngram_svd_prototypes | hashed charge-text n-grams in a TF-IDF / SVD embedding, original-row distances, k-means prototypes | 0.914912 |
|  | lgbm_te_stats_orig_priors | original churn rates, frequencies, nested target std / min / max and target-encoder means of 19 keys | 0.917891 |
|  | lgbm_synthetic_fingerprint | value / pair / signature counts, nearest-original distances, k-means prototypes, target encodings | 0.915831 |
|  | lgbm_nearest_original_churn | raw columns + churn label of the nearest original row | 0.915976 |
| [realmlp](models/realmlp) | realmlp_3config_digits_bins_rare_original | 3 RealMLP configurations on digit / bin / rounded-category / rare features, original rows appended | 0.917064 |
|  | realmlp_3config_crosses_priors_drift_original | 3 RealMLP configurations on rank / frequency / crosses / original priors / drift + digits | 0.917810 |
|  | realmlp_risk_score_crosses_priors_drift_original | RealMLP on rank / frequency / crosses / original priors / drift + risk score | 0.917546 |
|  | realmlp_sincos_digit_combo_te_stats | RealMLP on sin / cos, digits and mean / std / skew TEs of 14 pair codes | 0.916634 |
|  | realmlp_nested_te_stats_ngram | RealMLP with nested label-statistic TEs of categories / numeric values and n-gram means | 0.913508 |
|  | realmlp_pair_te_logit3_kbins_priors | RealMLP on KBins bins, value codes, original priors and 105 pair TEs as logit powers | 0.918292 |
| [tabm](models/tabm) | tabm_mini_snapped_proxies_rare_flags | TabM-mini on the categoricals plus snapped proxies and rare flags of the 3 numerics | 0.916734 |
| [neural_nets](models/neural_nets) | emb_mlp_snapped_numerics | embedding MLP 512-256 on the categoricals, fold-local snapped values / rare flags and the numerics | 0.915297 |
|  | emb_mlp_snapped_numerics_layernorm_cosine | the same with LayerNorm, smaller embeddings, cosine schedule and gradient clipping | 0.916081 |
|  | emb_mlp_engineered_original_priors_drift | embedding MLP on engineered transforms, bins / crosses, counts, original churn means and drift | 0.917578 |
|  | emb_resmlp_engineered_original_priors_drift | residual MLP on the same features | 0.917505 |
|  | tabtransformer_engineered_original_priors_drift | TabTransformer on the same features | 0.909929 |
|  | mlp_pair_target_encoding | MLP on per-fold cuML target encodings of all 171 column pairs | 0.915907 |
|  | pair_embedding_mlp | MLP on one learned embedding per column pair (171 pairs) | 0.914909 |
|  | tabtransformer_pair_te_logit3_digits_original_rows | TabTransformer on digit / bin categoricals + logit powers of 561 pair TEs; original rows added | 0.917488 |
|  | pytorch_tabular_ft_transformer_snapped_numerics | pytorch_tabular FT-Transformer on the categoricals, snapped values / rare flags and the numerics | 0.915700 |
|  | torch_frame_ft_transformer_original_te_combos | PyTorch Frame FT-Transformer on value codes, interaction codes and original churn means | 0.913323 |
|  | torch_frame_ft_transformer_decimal_digits_original_maps | PyTorch Frame FT-Transformer on decimal digits, charge arithmetic, bins, crosses and original maps | 0.915617 |
|  | ft_transformer_cls_engineered_nested_te | CLS FT-Transformer on frequency / rank / digit features, crosses and nested target encodings | 0.914055 |
|  | tabular_resnet_original_pca_projection | tabular ResNet on embeddings, cyclic tenure and original-data PCA / random projections | 0.912869 |
|  | tokenized_cnn_original_gmm_archetypes | 1D-CNN over feature tokens with original-data GMM archetype probabilities and similarities | 0.913764 |
|  | nam_original_density_bins | neural additive model on original profile flags, nearest-original distances and quantile bins | 0.914941 |
|  | ifan_original_noise_kmeans_kpca | instance-wise feature attention network on nearest-original noise, KMeans churn rates and kernel PCA | 0.913336 |
|  | prototype_net_original_mahalanobis | prototypical network with Mahalanobis distances to the original classes | 0.913463 |
|  | residual_mlp_dense_priors | residual MLP on dense engineered features, snap / digit columns, original priors and target encodings | 0.916244 |
|  | ffm_orig_bins_decimal_digits | field-aware FM on categoricals, original-value bins, first decimal digits and numerics; original rows added | 0.915888 |
|  | fm_orig_bins_decimal_digits | FM on the same fields; original rows added | 0.915541 |
|  | deepfm_snapped_quantile_prior_tokens | DeepFM on 41 token fields: categoricals, snapped numerics, quantile bins, churn-rate deciles, crosses | 0.915703 |
|  | rff_kernel_net_orig_kmeans_rankgauss | random Fourier feature kernel network on embeddings, original-data KMeans distances and RankGauss numerics | 0.913635 |
|  | snn_charge_char_ngrams_benford_orig_knn | self-normalizing network on charge-string character n-grams, Benford probabilities, nearest-original votes | 0.912932 |
|  | dae_knn_discriminator_token_te_xgb | XGBoost on value tokens, nested TE, DAE reconstructions / latents, nearest-original and discriminator features | 0.917661 |
|  | graphsage_engineered_numeric_knn_graph | full-graph GraphSAGE on a train + test KNN graph over engineered numerics with categorical embeddings | 0.913761 |
|  | graphsage_snapped_rare_flags_ohe_knn_graph | full-graph GraphSAGE on a train + test KNN graph over one-hot categoricals + numerics | 0.914215 |
|  | graphsage_sampled_snapped_rare_flags_ohe_knn_graph | mini-batch GraphSAGE on sampled subgraphs of the same kind of graph | 0.915789 |
|  | liquid_nn_ordinal_service_bins | liquid neural network on the columns, a service count, charge relations and bins | 0.912839 |
|  | liquid_nn_embeddings_digits_orig_priors | gated liquid neural network with embeddings on decimal digits, bins, pairs and original churn rates | 0.915116 |
|  | tabnet_snap_counts_orig_priors | TabNet on snapped charges, transforms, crosses, train + test counts and original churn rates | 0.913415 |
|  | gandalf_raw_columns | pytorch_tabular GANDALF on the raw columns | 0.911962 |
| [classic](models/classic) | lr_pair_te_logit3 | cuML logistic regression on 171 pair target encodings -> [z, z^2, z^3] | 0.916014 |
|  | lr_pair_te_logit2 | cuML logistic regression on 171 pair target encodings -> [z, z^2] | 0.914806 |
|  | logreg_ohe_crosses_hinges_group_dense | cuML logistic regression on 4,362 one-hot bins / digits / crosses + 148 dense columns | 0.916847 |
|  | rf_snap_digits_te | cuML random forest on snap / decimal-digit features and per-fold target encodings | 0.914143 |
|  | xgb_nested_knn_k1_k80_top_schemes | XGBoost on original-data KNN (k 1), nested train-data KNN (k 80), LR meta features | 0.915721 |
|  | xgb_nested_knn_k10_k40_random_schemes | XGBoost on original-data KNN (k 10), nested train-data KNN (k 40), LR meta features | 0.915561 |
| [public_models/nalgirayergn_xgb](models/public_models/nalgirayergn_xgb) | nalgirayergn_xgb | nalgirayergn's XGBoost notebook | 0.918210 |
| [public_models/siukeitin_ydf](models/public_models/siukeitin_ydf) | siukeitin_ydf | siukeitin's YDF (discussion post) | 0.917976 |
| [public_models/johnnyhyland_survival_xgb](models/public_models/johnnyhyland_survival_xgb) | johnnyhyland_survival_xgb | survival-curve features after johnnyhyland's Bayesian survival notebook, into XGBoost | 0.916741 |
| [public_models/blamerx_tabm](models/public_models/blamerx_tabm) | blamerx_tabm | blamerx's TabM notebook | 0.918788 |
| [public_models/yekenot_realmlp_pytabkit](models/public_models/yekenot_realmlp_pytabkit) | yekenot_realmlp_pytabkit_mish | yekenot's pytabkit RealMLP notebook, earlier version | 0.918679 |
|  | yekenot_realmlp_pytabkit_silu | yekenot's pytabkit RealMLP notebook, later version | 0.919118 |
| [public_models/siukeitin_tabicl](models/public_models/siukeitin_tabicl) | siukeitin_tabicl | TabICL from siukeitin's comment on yunsuxiaozi's TabICL discussion | 0.915195 |
| [public_models/blamerx_catboost](models/public_models/blamerx_catboost) | blamerx_catboost | blamerx's CatBoost notebook | 0.918818 |
| [public_models/yunsuxiaozi_realmlp_pytorch](models/public_models/yunsuxiaozi_realmlp_pytorch) | yunsuxiaozi_realmlp_pytorch | yunsuxiaozi's from-scratch PyTorch RealMLP notebook | 0.918943 |
| [stack](stack) | | **L2 logistic regression on the 100 pairs above** | **0.91983** |

Rows of one folder are listed in recipe order; "the same" refers to the row above. OOF AUCs are from our competition
runs (5 folds, `StratifiedKFold(5, shuffle=True, random_state=42)`).

## How to run
1. Put the data in `data/` (see [data/README.md](data/README.md)).
2. Build the shared inputs first:
   - `python models/snap_features/build.py`: the snap / decimal-string feature files that several XGBoost, CatBoost,
     LightGBM and neural-network scripts read;
   - `models/xgboost/build_tabpfn_original.py` (for `xgb_native_cat_tabpfn_original`) and
     `models/classic/select_knn_schemes.py` (for `xgb_nested_knn_k1_k80_top_schemes`).
3. Train the models (each folder's README has its commands). Every script writes `preds/<pair>_oof.npy` and
   `preds/<pair>_test.npy` (probabilities).
4. `python stack/stack.py` finds every pair under `models/*/preds/` and `models/public_models/**/preds/`, prints the stack
   CV and writes `stack/submission.csv`.

The scripts use the default CUDA device; pick a GPU with `CUDA_VISIBLE_DEVICES`.

## Requirements
Base (`requirements.txt`): numpy, pandas 2.x, scipy, scikit-learn, xgboost, lightgbm, catboost, ydf, torch, pytabkit,
pytorch_tabular, pytorch-tabnet, pytorch-frame, torch_geometric, tabicl, tabpfn, pymc. cuML / cuDF / CuPy (RAPIDS) run the
stacker and the cuML models. Model-specific notes are in each folder's README.
