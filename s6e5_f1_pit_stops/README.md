# Playground S6E5: Predicting F1 Pit Stops (2nd place)

Writeup: [2nd Place - Autonomous Codex Yolo!](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/2nd-place-autonomous-codex-yolo)

The competition data (439,140 train and 188,165 test rows; 14 features; binary target `PitNextLap`; metric ROC AUC) was
generated from the 101,371-row F1 strategy dataset. Our final submission was an L2 logistic regression stack of 219
models from 34 model classes, which finished 2nd. This folder holds 100 of them: every model class and the best model of
each class. On their own they score private LB 0.95503, the score of 1st place. All 100 are trained on one shared 5-fold
split, most of them with the original rows appended to the training side of each fold:
- **Gradient-boosted trees** (XGBoost, CatBoost, LightGBM): nested (inner-fold) target encodings, group statistics,
  ranks and residuals within race / stint / compound groups, previous / next lap features, count and quantile features,
  and priors from the original data.
- **RealMLP and TabM**: the strongest single models, several of them averaging 2-5 seeds per fold.
- **TabICL v2**, zero-shot and fine-tuned, with original rows appended.
- **Other neural networks**: TabTransformer, TabNet, field-aware and plain factorization machines, a graph network, a DAE,
  a liquid neural network, FT-Transformer, GANDALF, residual MLPs, and 11 models from the TALENT library (ModernNCA, TabR,
  NODE, ExcelFormer, Trompt, AMFormer, DCNv2, MLP-PLR, AutoInt, a SAINT-style and a GrowNet-style network).
- **Classic models**: random forest, logistic regression, YDF, and XGBoost on kNN, hazard and snap / radix features.
- **8 public notebooks** run on our folds (`models/public_models`).

## Results

| Stack | CV | Public LB | Private LB |
|---|---:|---:|---:|
| **The 100 models in this repo** | 0.95528 | 0.95487 | **0.95503** |
| Our final submission (219 models) | 0.95520 | 0.95487 | 0.95502 |
| 1st place on the private LB | | | 0.95503 |

## Models

| Folder | Pair | Model | OOF AUC |
|---|---|---|---:|
| [realmlp](models/realmlp) | realmlp_fold_pipeline_a_te_nens20 | fold-local Pipeline A features + 2 combo TEs; SiLU [512, 256, 128], n_ens 20, dropout 0.04 | 0.954037 |
|  | realmlp_fold_pipeline_a_te_nens24 | same with n_ens 24 | 0.954083 |
|  | realmlp_fold_pipeline_a_te_light_reg | n_ens 24, dropout 0.03, weight decay 0.014 | 0.954095 |
|  | realmlp_fold_pipeline_a_te_light_reg_wide | + wider hidden layers [768, 384, 192] | 0.954111 |
|  | realmlp_fold_pipeline_a_arith_mish | fold-local extended Pipeline A (arithmetic features), Mish, 6 epochs, PLR sigma 0.94 | 0.954366 |
|  | realmlp_pipeline_a_mish_2seed | global Pipeline A, Mish, [768, 384, 192], 2 models per fold | 0.954391 |
|  | realmlp_pipeline_a_mish_2seed_ls0075 | label smoothing 0.0075, 2 models per fold | 0.954387 |
|  | realmlp_pipeline_a_mish_3seed | 3 models per fold | 0.954409 |
|  | realmlp_pipeline_a_mish_3seed_drop04 | dropout 0.04, 3 models per fold | 0.954410 |
|  | realmlp_pipeline_a_mish_4seed | 4 models per fold | 0.954419 |
|  | realmlp_pipeline_a_mish_4seed_drop04 | dropout 0.04, 4 models per fold | 0.954416 |
|  | realmlp_pipeline_a_mish_5seed | 5 models per fold | 0.954423 |
|  | realmlp_pipeline_a_mish_5seed_drop04 | dropout 0.04, 5 models per fold; best single model | 0.954426 |
|  | realmlp_shape_state_te_no_original | extended Pipeline A + combos, rounded values, lap-delta shape, race state, 30 TEs; no original rows | 0.950537 |
|  | realmlp_sequence_anchor_priors | per-fold feature builder + race / stint sequence features, 28 nested TEs | 0.949876 |
|  | realmlp_pair_grid_te | per-fold feature builder + pair grid, 72 nested TEs | 0.949185 |
|  | realmlp_3config_rank_freq_drift | rank / frequency / original-rate / drift features, all original rows, 3 configurations per fold | 0.948729 |
|  | realmlp_original_rank_zscore_te | original-data ranks / z-scores / priors + 23 nested TEs, no original rows | 0.948437 |
| [xgboost](models/xgboost) | xgb_relrank_original_w035 | base + group aggregates, lags, relative ranks, frequencies (485 cols), nested TEs; original rows weight 0.35 | 0.951866 |
|  | xgb_relrank_next_lap_original_w05 | + next-lap values (499 cols); original weight 0.5 | 0.951730 |
|  | xgb_relrank_pipea_formulas_smooth_te_original_w05 | + Pipeline A + formulas (568 cols), smoother TEs; original weight 0.5 | 0.952116 |
|  | xgb_relrank_pipea_formulas_next_lap_original_w05 | + Pipeline A + formulas + next-lap values (582 cols); original weight 0.5 | 0.952012 |
|  | xgb_group_residuals_original_w08 | + 13-grouping residuals + source flag (855 cols), TE on keys <= 25k levels; original weight 0.8 | 0.952977 |
|  | xgb_hazard_pit_window_cadence_original_w08 | + hazard, pit-window, cadence and hash keys (1478 cols), TE on keys <= 750 levels, lr 0.006; best single XGBoost | 0.953553 |
|  | xgb_nearest_original_knn | nearest-original-row target means / distances (k 1, 3, 5, 10), original priors, state ratios | 0.946744 |
|  | xgb_union_te_prebinned | union of three feature pipelines, nested target statistics, pre-binned features, max_bin 16000 | 0.950423 |
|  | xgb_original_self_supervised | raw columns + 25 predictions of each column by models trained on the original rows | 0.940117 |
|  | xgb_tyrelife_anchor_te | TyreLife-anchor target statistics, category x anchor, digit-pair and category-pair target means | 0.946144 |
|  | xgb_pairwise_combo_te_original | target means + counts of the 91 column pairs; original rows appended | 0.951525 |
|  | xgb_union_te_laptime_anchor | two feature pipelines + LapTime-anchor statistics, fold-safe target / count encodings | 0.949137 |
| [catboost](models/catboost) | catboost_artifact_cats_priors_depth9 | artifact categories + 6 original-data priors, all original rows, depth 9, balanced classes | 0.953235 |
|  | catboost_artifact_cats_next_lap_sequence | + previous / next row sequence features, depth 8 | 0.952805 |
|  | catboost_pit_state_cats_original_w115 | + formulas, ranks, hazard / pit-state / strategy crosses, original weight 1.15, depth 9; best single CatBoost | 0.953404 |
|  | catboost_native_combos_sequence_rich_priors | all feature blocks, 37 original priors, original weight 0.9, max_ctr_complexity 4 | 0.952457 |
|  | catboost_pair_grid_original_subset_w07 | per-fold feature builder with a 36-pair grid, fold-matched original rows (weight 0.7), depth 8 | 0.952226 |
|  | catboost_anchor_priors_original_subset_w085 | per-fold feature builder without the pair grid, fold-matched original rows (weight 0.85), depth 9 | 0.952044 |
|  | catboost_numeric_artifacts_numcat | numeric transforms / digit artifacts + native value categories and n-grams, all original rows | 0.950532 |
|  | catboost_fold_density_knn | fold-local frequency / PMI / kNN-density features + native categories | 0.951296 |
|  | catboost_top150_select_depth6 | fold-local top-150 feature selection, depth 6, Bernoulli bootstrap | 0.950348 |
|  | catboost_dense_priors_crosses | dense original priors, drift, class centroids, 16 crosses, max_ctr_complexity 4 | 0.950737 |
|  | catboost_nearest_original_targets | nearest-original-row label features + original priors, depth 6 | 0.945827 |
|  | catboost_original_value_bins | 4 numerics + label-encoded categories and original-value bins, depth 6 | 0.949583 |
| [lightgbm](models/lightgbm) | lgbm_domain_counts_te_original_w035 | domain features + interactions + counts + per-fold TEs (71 features); original rows weight 0.35 | 0.951988 |
|  | lgbm_domain_counts_group_stats_original_w085 | domain + interactions + counts + group stats (87 features); original weight 0.85 | 0.952319 |
|  | lgbm_lean_original_w105 | lean 55 features; original weight 1.05; feature_fraction 0.10 | 0.952938 |
|  | lgbm_lean_original_w125 | lean 55 features; original weight 1.25 | 0.952929 |
|  | lgbm_lean_original_w125_slow | lean 55 features; original weight 1.25; lr 0.012; best single LightGBM | 0.953023 |
|  | lgbm_snap_priors_pairs_nested_te | nearest-original snap / original priors / pair keys (638 columns) with nested TEs | 0.948752 |
|  | lgbm_original_knn_target | nearest-original-row target features + original priors | 0.946744 |
|  | lgbm_fingerprint_knn_kmeans | value / pair / signature counts, distances to original rows, k-means prototypes, fold TEs | 0.950035 |
| [tabm](models/tabm) | tabm_group_stats_digits_d448 | fold-local base + artifact digits + source flag + group stats + combo TEs, all original rows; k 32, 4 x 448 | 0.953361 |
|  | tabm_group_stats_digits_d512 | same recipe, 4 x 512, embedding dim 32 | 0.953360 |
|  | tabm_group_stats_digits_d512_reseed | the 4 x 512 model with another seed; best single TabM | 0.953371 |
|  | tabm_mini_domain_group_source | TabM-mini (k 16) on domain + group-stat features + source flag; fold-matched original rows | 0.951176 |
|  | tabm_original_priors_ngram_te | TabM-mini on fold-local original priors / distribution / digit features + nested TEs of base and n-gram keys | 0.949046 |
| [tabicl](models/tabicl) | tabicl_v2_zero_shot_8seed | zero-shot TabICL (v2 checkpoint), 8 seeds x 2 estimators, raw columns | 0.949538 |
|  | tabicl_v2_finetune_8seed | fine-tuned TabICL v2, 8 members, raw columns | 0.950652 |
|  | tabicl_v2_finetune_lowcard_cats | fine-tuned TabICL v2, 7 low-cardinality numerics as categoricals | 0.948527 |
|  | tabicl_v2_finetune_original_append | fine-tuned TabICL v2, all original rows appended to every fold's training rows | 0.950827 |
|  | tabicl_v2_finetune_bigger_context | fine-tuned TabICL v2, larger context (14,000 rows), 10 epochs | 0.950681 |
| [neural_nets](models/neural_nets) | emb_resnet_periodic_group_stats_all_original | residual MLP on embeddings + periodic numeric embeddings, combo TE, group stats, source flag, all original rows | 0.948923 |
|  | tabtransformer_domain_group_stats_fold_original | TabTransformer (d 96, 2 layers), domain + group stats + source flag, fold-matched original rows | 0.947101 |
|  | tabtransformer_medium_domain_group_stats_all_original | TabTransformer (d 128, 3 layers), domain + group stats + source flag, all original rows | 0.947160 |
|  | tabnet_combo_te | TabNet (n_d = n_a = 32, 3 steps), base block + two combo TEs | 0.937210 |
|  | ffm_domain_group_stats_all_original | field-gated FM network, domain + group stats + source flag, all original rows | 0.949178 |
|  | graph_mlp_domain_extended_group_stats | attention graph network over field embeddings, domain + extended group stats | 0.947632 |
|  | fm_domain_group_stats_fold_original | factorization machine (dim 48), domain + group stats + source flag, fold-matched original rows | 0.947381 |
|  | residual_embedding_nn_original_priors_te | residual embedding MLP on race-state features, original priors / centroid distances, inner-fold TEs | 0.939197 |
|  | tabular_resnet_original_manifold_priors | tabular ResNet on cyclic / support features, artifact keys, original priors, original-data PCA manifold | 0.940993 |
|  | dae_knn_discriminator_xgb | XGBoost on DAE reconstructions / latents, nearest-original, TE and real-vs-synthetic discriminator features | 0.946046 |
|  | ffm_original_bins_decimal_digits | field-aware FM on categoricals, original-value bins and decimal digits, all original rows | 0.943717 |
|  | liquid_nn_f1_state_original_priors | liquid neural network on race-state features, quantile bins, rounded / combo keys, original priors | 0.893791 |
|  | ft_transformer_snapped_simple_features | FT-Transformer (pytorch_tabular) on raw columns + snapped-value / rare-flag categories | 0.933224 |
|  | gandalf_artifact_combos_original_priors | GANDALF (pytorch_tabular) on race-state features, rounded / rank keys, combo keys, original priors | 0.937605 |
| [talent](models/talent) | modernnca_lowcard_ohe_cap160k | ModernNCA (TALENT), low-cardinality keys one-hot, original priors + frequencies | 0.941058 |
|  | tabr_lowcard_ohe_cap220k | TabR (TALENT), low-cardinality keys one-hot, PLR numerics | 0.943445 |
|  | node_rich_binary_orig_rows | NODE (TALENT), binary-encoded keys, fold-matched original rows | 0.936121 |
|  | excelformer_rich_binary_orig_rows | ExcelFormer (TALENT), binary-encoded keys, fold-matched original rows | 0.946912 |
|  | trompt_lowcard_indices_cap240k | Trompt (TALENT), low-cardinality key indices | 0.943453 |
|  | amformer_lowcard_indices_orig_rows | AMFormer (TALENT), low-cardinality key indices, fold-matched original rows | 0.944542 |
|  | dcn2_rich_indices_orig_rows | DCNv2 (TALENT), key indices, fold-matched original rows | 0.939530 |
|  | mlp_plr_lowcard_ohe_cap300k | MLP-PLR (TALENT), low-cardinality keys one-hot, PLR numerics | 0.943629 |
|  | autoint_lowcard_indices_cap260k | AutoInt (TALENT), low-cardinality key indices | 0.942763 |
|  | saint_style_lowcard_orig_rows_cap300k | SAINT-style transformer, key tokens + numeric tokens, fold-matched original rows | 0.941141 |
|  | grownet_style_lowcard_orig_rows_cap300k | GrowNet-style additive network, key embeddings, fold-matched original rows | 0.942110 |
| [classic](models/classic) | rf_domain_group_stats_combo_te | cuML random forest (950 trees, depth 20), domain features + group stats + combo TEs | 0.948845 |
|  | xgb_hazard_frailty_pair_te_original_w075 | XGBoost on tyre-life hazard / frailty features, original priors, 210 pair TEs | 0.946209 |
|  | xgb_knn_stack_lr_meta | XGBoost on original-kNN priors, inner-OOF kNN target features and LR meta features | 0.947231 |
|  | ydf_gbt_depth2_raw | YDF gradient-boosted trees, depth 2, raw columns | 0.943788 |
|  | logreg_ohe_crosses_splines_group_dense | cuML logistic regression on 8,143 one-hot bins / digits / crosses + 167 dense columns | 0.933935 |
|  | xgb_snap_radix_depth2 | XGBoost depth 2 on nearest-original snap / digit / radix features | 0.932076 |
|  | xgb_snap_radix_lapnumber_depth1 | XGBoost stumps on snap / digit / radix features + LapNumber radix block | 0.921717 |
| [public_models/yekenot_realmlp_pytabkit](models/public_models/yekenot_realmlp_pytabkit) | yekenot_realmlp_pytabkit_v11 | yekenot's pytabkit RealMLP notebook, version 11 | 0.953258 |
|  | yekenot_realmlp_pytabkit_v16 | yekenot's pytabkit RealMLP notebook, version 16 | 0.954088 |
| [public_models/yekenot_realmlp_pytorch](models/public_models/yekenot_realmlp_pytorch) | yekenot_realmlp_pytorch | yekenot's PyTorch RealMLP notebook | 0.953484 |
| [public_models/amanatar_realmlp](models/public_models/amanatar_realmlp) | amanatar_realmlp | amanatar's RealMLP (Pipeline A of the 3-model ensemble notebook) | 0.953442 |
| [public_models/rohit8527kmr7518_catboost](models/public_models/rohit8527kmr7518_catboost) | rohit8527kmr7518_catboost | rohit8527kmr7518's CatBoost notebook | 0.952703 |
| [public_models/berkayzkan_xgb](models/public_models/berkayzkan_xgb) | berkayzkan_xgb | berkayzkan's XGBoost baseline notebook | 0.949626 |
| [public_models/tamerlanomralinov_hgb](models/public_models/tamerlanomralinov_hgb) | tamerlanomralinov_hgb | tamerlanomralinov's HistGradientBoosting baseline notebook | 0.947546 |
| [public_models/shivamgravity_lgbm](models/public_models/shivamgravity_lgbm) | shivamgravity_lgbm | shivamgravity's LightGBM (Optuna) notebook | 0.945580 |
| [stack](stack) | | **L2 logistic regression on the 100 pairs above** | **0.955280** |

Rows of one folder are listed in recipe order; "+" means "the previous row plus". OOF AUCs are from our competition runs
(5 folds, `StratifiedKFold(5, shuffle=True, random_state=42)`).

## How to run
1. Put the data in `data/` (see [data/README.md](data/README.md)).
2. Train the models. The folders are independent (each needs only `common/`); `models/tabm` and `models/classic` reuse the
   shared harness in `models/neural_nets/lib/`. Each folder's README has its commands; most scripts run the 5 folds
   (`--folds` runs a subset; finished folds are kept) and write their pair(s) to `preds/`.
3. `python stack/stack.py` finds every pair under `models/*/preds/` and `models/public_models/**/preds/`, prints the stack
   CV and writes `stack/submission.csv`.

The scripts use the default CUDA device; pick a GPU with `CUDA_VISIBLE_DEVICES`.

## Requirements
Base (`requirements.txt`): numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, catboost, torch, pytabkit, ydf,
pytorch_tabular, pytorch-tabnet, tabicl. cuML / CuPy (RAPIDS) run the cuML models and make the stacker fast. The TALENT
models need the TALENT library (see `models/talent/README.md`). Model-specific notes are in each folder's README.
