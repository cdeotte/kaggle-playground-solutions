# Playground S6E4: Predicting Irrigation Need (2nd place)

Writeup: [2nd Place - Claude Code and Codex - GPU LogReg](https://www.kaggle.com/competitions/playground-series-s6e4/writeups/2nd-place-claude-code-and-codex-gpu-logreg)

The competition data (630,000 train and 270,000 test rows; 19 features; 3-class target `Irrigation_Need` = Low / Medium /
High; metric balanced accuracy) was generated from a 10,000-row original dataset. Our final submission blended two
stacks with 229 models in total and finished 2nd. This folder holds 100 of them, in the same two stacks: every model class and
the best model of each class. On their own they score private LB 0.98161, above our final submission's 0.98151.

All 100 models are trained on one shared 5-fold split and write 3-class probabilities. Most of them use features built
from the original data (snapping values to the original grid, original-data class priors, the original data's rule
score) and many append the original rows to the training side of each fold:
- **Gradient-boosted trees** (XGBoost, LightGBM, CatBoost, YDF): nested (inner-fold) per-class target encodings of value
  keys, pairs and n-grams, snapped / radix / digit features, unions of several feature pipelines, formula base margins.
- **RealMLP, TabM, TabICL** and other neural networks: DeepFM, a neural additive model, field-aware and plain
  factorization machines, GraphSAGE graph networks, a liquid neural network, TabNet, TabTransformer, FT-Transformer,
  GANDALF, and a denoising autoencoder.
- **Classic models**: logistic regressions on pair target encodings, random forests, KNN feature models.
- **9 public notebooks** run on our folds (`models/public_models`).

**The stacker** (`stack/stack.py`) is a multinomial logistic regression written in PyTorch so it runs on the GPU and uses
class weights: the metric is balanced accuracy, so every row is weighted by the balanced weight of its class. It takes the
per-class logits of every model as inputs, applies L2 regularization to the weights (Adam weight decay 1 / (C n),
C = 10), and runs 5-fold cross-validation with 5 seeds. Stack A (53 models, pair names ending in `_a`) and stack B (47
models: names ending in `_b` and the public notebooks) are fitted separately, and the submission is
argmax(P_A + 0.25 * P_B).

## Results

| Stack | CV | Public LB | Private LB |
|---|---:|---:|---:|
| **The 100 models in this repo (stack A + 0.25 x stack B)** | 0.98172 | 0.98161 | **0.98161** |
| Our final submission (229 models) | 0.98170 | 0.98195 | 0.98151 |

## Models

| Folder | Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---|---:|
| [xgboost](models/xgboost) | xgb_union_snap_te_a | A | 479-column union (snaps, bins, drift, priors, digits, string keys, radix, pairs) + nested per-class TE of 59 keys | 0.976857 |
|  | xgb_union_snap_te_cs080_a | A | the same union, lighter regularization, colsample 0.80 | 0.976639 |
|  | xgb_union_snap_te_cs060_a | A | the same, colsample 0.60 | 0.976888 |
|  | xgb_union_snap_te_cs040_a | A | the same, colsample 0.40 | 0.976827 |
|  | xgb_snap_radix_depth1_a | A | depth-1 stumps on rainfall / irrigation snaps, decimal digits, radix keys snap x category | 0.962792 |
|  | xgb_snap_radix_field_area_depth1_a | A | the same + Field_Area_hectare radix keys | 0.962861 |
|  | xgb_snap_radix_sunlight_depth1_a | A | the same + Sunlight_Hours radix keys | 0.962780 |
|  | xgb_numeric_pair_te_a | A | per-class TE of 88 numeric x categorical pairs + numeric values, counts, digits, rule score | 0.976064 |
|  | xgb_numeric_pair_te_maxbin16000_a | A | the same, max_bin 16,000 | 0.976075 |
|  | xgb_snap_priors_drift_te_a | A | snaps, quantile / round / digit keys, counts, original priors, drift + nested per-class TE of 25 keys | 0.974354 |
|  | xgb_ngram_te_stats_a | A | original distribution features + inner-fold label statistics and per-class TE of 19 keys, n-gram class rates | 0.971595 |
|  | xgb_original_self_supervised_a | A | raw columns + predictions of each column by models trained on the original rows | 0.968615 |
|  | xgb_pairwise_combo_te_original_a | A | class rates + counts of the 171 column pairs, original rows appended | 0.970693 |
|  | xgb_minimal_fe_depth2_a | A | depth 2; raw columns + 7 water-balance interactions, native categoricals | 0.967529 |
|  | xgb_group_priors_interaction_te_a | A | engineered numerics, counts / ranks, original group priors, in-fold class rates of 6 crosses | 0.969715 |
|  | xgb_digit_round_categories_a | A | depth 3; digit positions, rounded numerics and category triples as native categoricals | 0.971352 |
|  | xgb_rainfall_anchor_union_te_a | A | Rainfall_mm anchor keys / statistics, two feature pipelines, original priors + nested class rates of 108 keys | 0.970961 |
|  | xgb_numeric_keys_ngram_te_a | A | 18 keys per numeric, n-grams, original statistics + nested class rates of 211 keys | 0.973352 |
|  | xgb_union_te_prebinned_a | A | two feature pipelines + string keys + multiclass TE of 84 keys, pre-binned, max_bin 16,000 | 0.972771 |
|  | xgb_rank_ordinal_ovr_platt_b | B | ordinal + 3 one-vs-rest pairwise rankers, balanced multinomial LR on their scores + 91 features | 0.972911 |
|  | xgb_rule_formula_margin_b | B | base margin from the original data's rule class, raw columns | 0.969886 |
|  | xgb_logit_formula_margin_b | B | base margin = linear logits of the rule flags, raw columns | 0.969102 |
|  | xgb_snap_radix_5num_depth2_b | B | depth 2; snaps, digits, priors, radix keys of 5 numerics x 8 categoricals | 0.971374 |
|  | xgb_snap_radix_11num_depth2_b | B | depth 2; snaps, digits, priors, radix keys of all 11 numerics x 8 categoricals | 0.970539 |
|  | xgb_snap_original_te_nested_b | B | nested 5x5; formula numerics snapped to the original grid, original TE of the snaps | 0.973015 |
|  | xgb_native_cat_digit_positions_b | B | native categoricals, integer / decimal digit positions, rule flags | 0.966304 |
|  | xgb_num_cat_pair_te_digits_b | B | nested one-vs-rest TE of 88 numeric x categorical pairs + numeric values, digits | 0.976527 |
|  | xgb_nums_as_cats_te_stats_b | B | numerics as categories: inner-fold label statistics + per-class TE of 19 keys | 0.976824 |
|  | xgb_cat_ngram_nested_te_b | B | per-class nested TE of bigrams / trigrams / categoricals | 0.968950 |
|  | xgb_rain_moisture_snap_anchor_te_b | B | Rainfall / Soil_Moisture snap anchors: label statistics, quantiles, class shares of anchor keys | 0.973955 |
|  | xgb_prev_irrigation_anchor_te_b | B | rounded Previous_Irrigation_mm anchor: label statistics, quantiles, class shares | 0.966615 |
|  | xgb_union_nested_te_original_b | B | transforms / bins / digits / counts / priors / drift + nested per-class TE of 26 keys, original rows appended | 0.972985 |
|  | xgb_union_multiclass_te_original_b | B | three feature pipelines (441 columns) + label std and multiclass TE of 218 keys, original rows appended | 0.976833 |
|  | xgb_union_multiclass_te_original_prebinned_b | B | the same, pre-binned to 256 quantile bins, max_bin 16,000 | 0.976740 |
|  | xgb_mega_union_snap_radix_nested_te_b | B | 882-column union (snaps, bins, drift, priors, digits, string keys, radix) + nested per-class TE | 0.976777 |
| [lightgbm](models/lightgbm) | lgbm_highcard_te_counts_digits_a | A | value keys, counts, original priors, numeric tokens, digits; nested class-frequency TE | 0.973248 |
|  | lgbm_extra_trees_round_digit_two_stage_te_a | A | roundings, floor bins, digits, 28 category pairs; two-stage nested TE; extra trees | 0.970390 |
|  | lgbm_highcard_te_artifact_tokens_a | A | value keys, counts, original priors, quantile / digit tokens; nested class-frequency TE | 0.973201 |
|  | lgbm_union_dist_ngrams_class_freq_te_a | A | union of keys, digits, drift, original priors, class distributions, n-grams; class-frequency TE | 0.973689 |
|  | lgbm_goss_union_nested_te_a | A | union of keys, digits, ranks, drift, distributions, n-grams; nested class-frequency TE; GOSS | 0.973062 |
|  | lgbm_dart_priors_ngrams_sklearn_te_a | A | frequencies, original class distributions and priors, digits; TargetEncoder of 36 keys; DART | 0.970240 |
|  | lgbm_class_pctrank_zscore_te_b | B | percentile ranks / z-scores within the original classes, counts, drift; TargetEncoder | 0.977026 |
|  | lgbm_three_block_union_nested_te_b | B | three feature blocks; nested smoothed per-class TE; balanced weights | 0.978524 |
|  | lgbm_goss_crosses_binary_te_b | B | transforms, crosses, bins, digits, original priors; nested per-class binary TE; GOSS | 0.972563 |
|  | lgbm_extra_trees_two_stage_binary_te_b | B | roundings, digits, bins, pairs; two-stage nested per-class binary TE; extra trees | 0.976727 |
|  | lgbm_nested5x5_cat_pairs_te_b | B | nested 5x5; categoricals, 28 pairs, counts, original priors, bins, digits; TargetEncoder | 0.972146 |
| [catboost](models/catboost) | catboost_numcat_crosses_original_rows_a | A | numeric transforms + native value categories and pairs, original rows appended | 0.966253 |
|  | catboost_snap_bins_keys_nested_te_a | A | snapped / binned values, pair / triple / snap x category keys, original priors, nested class-frequency TE | 0.969425 |
|  | catboost_freq_priors_te_stats_balanced_a | A | frequencies, original expected-class priors, inner-fold TE statistics, balanced classes | 0.975863 |
|  | catboost_class_te_stats_ngrams_b | B | per-class TE mean / std / min / max + n-gram TE, balanced classes | 0.975334 |
|  | catboost_radix_bins_nested_te_b | B | 88 numeric x category radix keys, quantile bins, nested class TE, balanced classes | 0.971697 |
|  | catboost_nested5x5_freq_priors_ngrams_b | B | nested 5x5; frequencies, per-class original priors, soil-moisture class distances, n-grams | 0.971122 |
| [realmlp](models/realmlp) | realmlp_3config_digits_bins_rare_original_a | A | 3 RealMLP configurations on digit / bin / rounded-category / rare features, original rows appended | 0.968247 |
|  | realmlp_digit_cats_crosses_rank_priors_a | A | digit categories, crosses, rank / frequency and original priors, original rows appended | 0.971830 |
|  | realmlp_pair_te_logit_priors_drift_a | A | 28 pairwise multiclass TEs (logit powers), original priors and drift, original rows appended | 0.972210 |
|  | realmlp_nested_te_stats_ngram_a | A | nested label-statistic / class-probability TEs of categories, numeric values and n-grams | 0.971327 |
|  | realmlp_kbins_value_codes_combo_te_a | A | KBins bins, numeric value codes and a crop x season x irrigation TE | 0.971071 |
|  | realmlp_snapped_proxies_rare_flags_a | A | compact RealMLP on raw columns plus snapped proxies and rare flags | 0.966372 |
|  | realmlp_3config_digits_bins_formula_original_b | B | 3 RealMLP configurations on formula / digit / bin / rounded-category features, original rows | 0.965882 |
|  | realmlp_digits_pairs_priors_drift_b | B | digits, roundings, 6 pairs, formula score, original priors, counts and drift | 0.967651 |
|  | realmlp_nested_te_stats_ngram_formula_b | B | nested label-statistic TEs and n-gram means, formula score, original priors and ranks | 0.971956 |
|  | realmlp_formula_original_priors_b | B | raw columns, formula score and per-class original priors | 0.960693 |
|  | realmlp_torch_nested_combo_te_b | B | PyTorch RealMLP (8 members), nested 5x5, inner-fold TEs of the high-cardinality codes | 0.978822 |
| [tabm](models/tabm) | tabm_mini_snapped_proxies_rare_flags_a | A | TabM-mini on the categoricals plus snapped proxies and rare flags | 0.965773 |
|  | tabm_original_priors_te_ngram_a | A | TabM-mini on original priors / distribution / digit features and TEs of values, categoricals and n-grams | 0.972872 |
|  | tabm_mini_snapped_formula_priors_b | B | TabM-mini on snapped proxies and rare flags, formula score and original priors | 0.964803 |
|  | tabm_nested_binary_te_ngram_b | B | TabM-mini, nested 5x5, per-class target means of categoricals, numeric values and n-grams | 0.972351 |
| [tabicl](models/tabicl) | tabicl_zero_shot_8seed_a | A | zero-shot TabICL, 8 seeds x 2 estimators, raw columns | 0.961396 |
| [neural_nets](models/neural_nets) | ffm_orig_bins_decimal_digits_a | A | field-aware FM on categoricals, original-value bins, decimal digits and numerics; original rows added | 0.968381 |
|  | fm_orig_bins_decimal_digits_a | A | FM on the same 35 fields; original rows added | 0.969510 |
|  | gandalf_raw_features_a | A | GANDALF (pytorch_tabular) on the raw columns | 0.946313 |
|  | deepfm_snapped_quantile_tokens_a | A | DeepFM on 44 token fields: categoricals, snapped numerics, quantile bins, label-mean deciles, crosses | 0.975683 |
|  | nam_orig_density_multires_bins_a | A | neural additive model with original-neighbour distances, interactions, multi-resolution bins | 0.945443 |
|  | graphsage_sampled_knn_graph_a | A | mini-batch GraphSAGE on a train + test KNN graph | 0.960709 |
|  | liquid_nn_water_balance_bins_a | A | liquid neural network on raw columns, water-balance features and quantile bins | 0.959375 |
|  | tabnet_crosses_counts_orig_priors_a | A | TabNet on numeric transforms, crosses, bins, counts and original priors | 0.961107 |
|  | tabtransformer_pair_te_logit3_ridge_a | A | TabTransformer on 51 tokens + 1,275 pair TEs + ridge outputs; original rows added | 0.973951 |
|  | dae_knn_discriminator_xgb_b | B | XGBoost on DAE latents / reconstructions, nearest-original, TE and real-vs-synthetic features | 0.962718 |
|  | ft_transformer_formula_pairs_orig_priors_b | B | FT-Transformer (pytorch-frame) on raw columns, rule flags, bigrams, bins and original priors | 0.971540 |
|  | graphsage_transductive_knn_graph_b | B | full-batch GraphSAGE on a train + test KNN graph with rule features and original priors | 0.961385 |
|  | tabnet_formula_orig_priors_b | B | TabNet on raw columns, transforms, rule score and original priors | 0.964900 |
| [classic](models/classic) | rf_water_snap_digit_te_a | A | cuML random forest, snaps, digits, water balance, per-class TEs | 0.959458 |
|  | xgb_original_knn_k10_lr_meta_a | A | XGBoost on original-data KNN (k 10) + LR meta features | 0.961572 |
|  | xgb_original_knn_k1_lr_meta_a | A | XGBoost on original-data KNN (k 1) + LR meta features | 0.961310 |
|  | logreg_ohe_crosses_bins_group_dense_a | A | logistic regression on 4,518 one-hot + 174 dense columns | 0.968458 |
|  | lr_pair_te_logit3_a | A | 171 pair TEs -> [z, z^2, z^3] -> logistic regression | 0.957787 |
|  | lr_pair_te_logit2_formula_a | A | 171 pair TEs -> [z, z^2] + rule score -> logistic regression | 0.975519 |
|  | ydf_gbt_depth2_raw_a | A | YDF gradient-boosted trees, depth 2, raw columns | 0.969607 |
|  | xgb_nested_knn_lr_meta_b | B | XGBoost on original KNN + nested train KNN (k 40) + LR meta features | 0.970090 |
|  | rf_snap_digit_formula_te_b | B | cuML random forest, snaps + digits of all numerics, rule score, per-class TEs | 0.960169 |
|  | lr_pair_te_logit3_formula_b | B | 171 pair TEs -> [z, z^2, z^3] + rule score -> logistic regression | 0.975785 |
| [public_models/blamerx_xgb](models/public_models/blamerx_xgb) | blamerx_xgb | B | blamerx's XGBoost notebook (pseudo-labelling and threshold tuning removed) | 0.978298 |
| [public_models/blamerx_catboost](models/public_models/blamerx_catboost) | blamerx_catboost | B | blamerx's CatBoost notebook (threshold tuning removed) | 0.977986 |
| [public_models/blamerx_lgbm](models/public_models/blamerx_lgbm) | blamerx_lgbm | B | blamerx's LightGBM notebook (pseudo-labelling and threshold tuning removed) | 0.975711 |
| [public_models/aliafzal9323_catboost](models/public_models/aliafzal9323_catboost) | aliafzal9323_catboost | B | aliafzal9323's CatBoost notebook, nested 5x5 | 0.977471 |
| [public_models/utaazu_catboost](models/public_models/utaazu_catboost) | utaazu_catboost | B | utaazu's CatBoost notebook (bias tuning removed) | 0.977055 |
| [public_models/utaazu_xgb](models/public_models/utaazu_xgb) | utaazu_xgb | B | utaazu's XGBoost notebook (bias tuning removed) | 0.974203 |
| [public_models/mahoganybuttstrings_logreg](models/public_models/mahoganybuttstrings_logreg) | mahoganybuttstrings_logreg | B | mahoganybuttstrings' logistic regression notebook | 0.976519 |
| [public_models/yekenot_realmlp_pytabkit](models/public_models/yekenot_realmlp_pytabkit) | yekenot_realmlp_pytabkit | B | yekenot's pytabkit RealMLP notebook | 0.973410 |
| [public_models/emanuellcs_catboost](models/public_models/emanuellcs_catboost) | emanuellcs_catboost | B | emanuellcs' CatBoost notebook, nested 5x5 | 0.970478 |
| [stack](stack) | | | **stack A + 0.25 x stack B (multinomial logistic regressions)** | **0.98172** |

Rows of one folder are listed by stack; "the same" refers to the row above. OOF scores are from our competition runs
(5 folds, `StratifiedKFold(5, shuffle=True, random_state=42)`).

## How to run
1. Put the data in `data/` (see [data/README.md](data/README.md)).
2. Train the models. The folders are independent (each needs only `common/`); each folder's README has its commands.
   Every script writes `preds/<pair>_oof.npy` (n_train x 3; `5 x n_train x 3` for the nested 5x5 models) and
   `preds/<pair>_test.npy` (n_test x 3), class order Low, Medium, High.
3. `python stack/stack.py` finds every pair under `models/*/preds/` and `models/public_models/**/preds/`, fits stack A
   and stack B, prints their CV and the blend's, and writes `stack/submission.csv`. GPU training is not bit-for-bit
   deterministic, so a rerun can differ in a few rows.

The scripts use the default CUDA device; pick a GPU with `CUDA_VISIBLE_DEVICES`.

## Requirements
Base (`requirements.txt`): numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, catboost, torch, pytabkit, ydf,
pytorch_tabular, pytorch-tabnet, pytorch-frame, torch_geometric, tabicl. cuML / cuDF / CuPy (RAPIDS) run the cuML
models. Model-specific notes (package versions, memory) are in each folder's README.
