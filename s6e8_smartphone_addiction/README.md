# Playground S6E8: Predicting Smartphone Addiction (1st place)

Writeup: [1st Place - Distributed Intelligence - NVIDIA Inference Hub](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe)

The competition data (691,369 train and 296,302 test rows; 9 numeric and 3 categorical features, many of them missing;
binary target `addicted_label`; metric ROC AUC) was generated from a 7,500-row original dataset. Our final submission
was an L2 logistic regression stack of 468 models. This folder holds the 102 of them that carry the stack. On their own
they score private LB 0.97171, so they would also have won (2nd place: 0.97123). All 102 are trained on one shared
5-fold split:
- **Forensic pipeline**: fold-local target encodings, imputed screen-time fields, original-data likelihoods, pair PMI and
  lift, autoencoder / XGBoost reconstructions of every field, a global density mixture fitted on the train + test
  predictors and "repair-posterior" channels (how a row's screen-time budget sits under that mixture). Learners: RealMLP,
  XGBoost, CatBoost, LightGBM, TabM, pair-lift models, source-row attention / graph / generative models and cuML models.
  Its last RealMLP, `realmlp_forensic_repair20`, is our best single model: OOF AUC 0.970706 and private LB 0.97145, which
  alone also beats 2nd place.
- **Density pipeline**: masked-language-model (MLM) density, completion and fill channels, plus cross-fitted lift banks.
  Learners: XGBoost and CatBoost with WOE / completion base margins, RealMLP, TabM, lookup transformers, DCNv2, YDF and
  cuML models.
- **More model families** with their own features: LightGBM / XGBoost / CatBoost variants, NODE, an FT-Transformer, an
  elastic-net wide model, a gcForest cascade, RuleFit, an EBM, a spline GAM, factorization machines, naive-Bayes and
  likelihood-ratio classifiers, and TabPFN v3.
- **25 public notebooks** run on our folds (`models/public_models`).

## Results

| Stack | CV | Public LB | Private LB |
|---|---:|---:|---:|
| **The 102 models in this repo** | 0.97101 | 0.97203 | **0.97171** |
| Our final submission (468 models) | 0.97099 | 0.97207 | 0.97176 |
| `realmlp_forensic_repair20` alone (single model) | 0.97071 | 0.97174 | 0.97145 |
| 2nd place on the private LB | | | 0.97123 |

## Models

| Folder | Pair | Model | OOF AUC |
|---|---|---|---:|
| [forensic_features](models/forensic_features) | - | fixed original-data channels and screen-response blocks: the first feature step of the forensic pipeline | - |
| [lgbm_margin](models/lgbm_margin) | lgbm_forensic_margin_frequency | LightGBM (63 leaves, depth 6, 4,400 rounds), numerics + category log-frequencies, forensic margin as init_score | 0.969471 |
|  | lgbm_forensic_margin_wide | LightGBM (127 leaves, 1,800 rounds), numerics only, forensic margin as init_score | 0.969398 |
| [xgb_forensic](models/xgb_forensic) | xgb_forensic_woe_margin | XGBoost on the 102-column forensic matrix (nested TEs, imputed screens, pair PMI / copula / response evidence), TE-WOE + unsupported-pair base margin | 0.969415 |
|  | xgb_forensic_ae_imputation | the same + 25 masked-autoencoder fill features (127 columns) | 0.969421 |
|  | xgb_forensic_original_residual_refit | XGBoost on the 290-column frontier matrix, completion margin, round selector + 10/9 refit, 6 parallel trees | 0.969973 |
|  | xgb_forensic_periodic_rbf_regions | frontier XGBoost + region lifts, periodic and RBF coordinates (479 features), interaction constraints | 0.970128 |
| [catboost_forensic](models/catboost_forensic) | catboost_forensic_fragments | CatBoost on the forensic matrix + source channels + fragment / semantic categories, forensic baseline | 0.969769 |
|  | catboost_forensic_reconstruction | the same + reconstruction / density-completion evidence, 120 columns pruned, max_ctr_complexity 3 | 0.969941 |
| [tabm_forensic](models/tabm_forensic) | tabm_forensic_strong | TabM-D k32 on row-local views + 104 forensic columns + pair contrasts, 5 internal members | 0.969763 |
|  | tabm_forensic_imputed | TabM-D k32 + neurally completed view, reconstruction / density-completion channels, 10 internal members | 0.969964 |
| [realmlp_repair](models/realmlp_repair) | realmlp_forensic_base | RealMLP on raw + exact-value tokens + screen geometry + 95 forensic fold columns (nested TEs, responses, PMI, original LLRs / priors) | 0.970055 |
|  | realmlp_forensic_density_completion | + 10 density-completion aggregates from a per-fold masked density transformer | 0.970106 |
|  | realmlp_forensic_reconstruction | + 71 cross-fitted XGBoost / autoencoder reconstruction channels | 0.970214 |
|  | realmlp_forensic_pairlift | + 10 exact pair-lift residuals | 0.970299 |
|  | realmlp_forensic_global_density | global train+test 4-state density-mixture channels and completion, coarse / derived lift banks, generator temperature | 0.970662 |
|  | realmlp_forensic_repair20 | + 20 repair-posterior channels (screen-budget band mass / shifts under the global mixture); best single model | 0.970706 |
| [pairlift_models](models/pairlift_models) | pairlift_xgb | XGBoost on the forensic matrix + autoencoder fill view + pair lift, base margin | 0.969458 |
|  | pairlift_lgbm | LightGBM on the forensic core + fixed-original channels + category counts + pair lift, init-score margin | 0.969516 |
|  | pairlift_catboost | CatBoost reconstruction parent + pair lift (feature weight 0.1), baseline margin | 0.969954 |
|  | pairlift_tabm | TabM (10 inner members) on completed-row blocks + forensic core + reconstruction + pair lift | 0.970011 |
| [structural_nets](models/structural_nets) | source_attention_net | mask-routed source-attention network over 6x16 original-row donors, forensic + reconstruction context, auxiliary loss | 0.967932 |
|  | source_graph_lr | source-genealogy graph label messages + structural donor features, L2 logistic regression | 0.959733 |
|  | source_transition_likelihood | class-conditional histogram + top-4 donor transition likelihood (generative) | 0.911932 |
| [cuml_models](models/cuml_models) | cuml_original_knn_lr | original-neighbour geometry + response bank -> cuML logistic regression | 0.961365 |
|  | cuml_residual_pca_svm | whitened PCA-64 + response bank -> cuML linear SVM | 0.917456 |
|  | cuml_landmark_rbf_lr | PCA-40 + 256 KMeans landmarks x 3 RBF maps + response bank -> cuML logistic regression | 0.952970 |
|  | cuml_umap_lr | UMAP-16 + response bank -> cuML logistic regression | 0.955080 |
|  | cuml_artifact_rf | 240 label-free source / generator-artifact columns -> 900-tree cuML random forest | 0.947677 |
|  | cuml_cluster_posterior_lr | PCA-32 + KMeans 64/256/1024 leave-one-out cluster posteriors + response bank -> cuML logistic regression | 0.953891 |
| [density_features](models/density_features) | - | base channels and masked-language-model (MLM) density, completion and fill channels: the first feature step of the density pipeline | - |
| [lift_banks](models/lift_banks) | - | cross-fitted pair-lift, coarse-lattice and response-surface lift banks | - |
| [xgb_density_margin](models/xgb_density_margin) | xgb_te_woe_margin | XGBoost on base + counts / original / PMI / imputers / lattice + nested TEs with region and geometry keys; WOE base margin; 172 cols | 0.969484 |
|  | xgb_te_mlm3_margin | + TE s1 x3 seeds, MLM completion / density / fills, completion margin; 263 cols | 0.969753 |
|  | xgb_te_mlm3_sf_pruned | + smooth fields, artifact residuals, token-embedding PCA, dsx; gain-pruned 337 -> 217 | 0.969987 |
|  | xgb_te_mlm3_plift | + pair lifts; 347 -> 227 | 0.970011 |
|  | xgb_te_mlm3_plift_cbank | + coarse lift bank; 356 -> 236 | 0.970055 |
|  | xgb_te_mlm4_plift_cbank | v4-MLM blocks replace the d384 ones; 341 -> 221 | 0.970137 |
|  | xgb_te_mlm4_qfill_wbank | + den_v5, qfill, wave bank; 386 -> 266 | 0.970186 |
|  | xgb_te_mlm4_qfill_wbank12_sbank | + wbank12, sbank; 404 -> 284 | 0.970203 |
| [catboost_density](models/catboost_density) | catboost_tokens_fills | CatBoost, 27 ordered-CTR token categoricals + 36 numerics + fills | 0.969002 |
|  | catboost_tokens_mlm3 | + mlmfill, mlmfill2, rrd, cmp3, bp2 | 0.969088 |
|  | catboost_tokens_mlm4 | + mlmfill4, cmp4, den_v4, rrd, bp2 | 0.969351 |
|  | catboost_mlm4_banks_baseline | 348-column numeric channel matrix + all lift banks, WOE + completion baseline, lr 0.005 | 0.969802 |
| [realmlp_density](models/realmlp_density) | realmlp_density_knn_mc | RealMLP on base channels (nested TEs, counts, original LLR, PMI, imputers, Monte-Carlo posteriors, kNN in MLM space, d384 MLM density) + exact-token categoricals | 0.969246 |
|  | realmlp_density_battery_sf_emb | RealMLP battery (5 CV + 5 refit + 16 ensemble members), base channels + smooth fields + 48 MLM-embedding PCs | 0.969521 |
|  | realmlp_density_lean_cmp3 | RealMLP short refit, lean matrix + smooth fields + MLM completion (cmp3) + d384 density | 0.970110 |
|  | realmlp_density_lean_mfill_rrd | + MLM expected-value fills (mfill, mfill2) + reconstruction residuals | 0.970255 |
|  | realmlp_density_lean_plift | + pair-lift bank | 0.970329 |
|  | realmlp_density_lean_cbank | + coarse-lattice lift bank | 0.970390 |
|  | realmlp_density_lean_mlm4_wbank | v4-MLM channels (cmp4, den_v4, mfill4) + pair / coarse / wave lift banks | 0.970490 |
|  | realmlp_density_lean_mlm4_qfill | + MLM posterior width / skew (qfill) + v5-MLM density channels | 0.970567 |
| [tabm_density](models/tabm_density) | tabm_density_mfill_bp2 | TabM k32 (PWL embeddings), lean matrix + cmp3 / den384v3 / fills / rrd + budget-mixture posteriors | 0.969672 |
|  | tabm_density_mlm4_banks_std | TabM on v4-MLM channels + lift banks + bp2, per-fold logit standardization | 0.970078 |
| [lookup_transformer](models/lookup_transformer) | lookup_transformer_evidence7 | lookup transformer, 7-dim per-column evidence, 6 MLM-initialized replicas, d256 / 8 layers | 0.968608 |
|  | lookup_transformer_evidence11_fills | lookup transformer, 11-dim evidence (+ completion, residual, budget fill / sd) + fill-geometry token | 0.968977 |
| [dcn](models/dcn) | dcn_te_mlm3_plift | DCNv2 (K=16, field-space gated crossing, PLR) on 253 channels + 12 tokens | 0.969464 |
|  | dcn_te_mlm4_banks | DCNv2 on 292 channels (v4 MLM + lift banks) + 12 tokens | 0.969582 |
| [ydf](models/ydf) | ydf_hessian_te_mlm3_cbank | YDF gradient-boosted trees with hessian gain on the 253-column matrix + coarse bank (262 cols) | 0.969298 |
| [cuml_density_models](models/cuml_density_models) | cuml_svc_rbf_density | cuML SVC (RBF) on the 213-column density / TE matrix; per-fold ranks + cross-fitted isotonic calibration | 0.964832 |
|  | cuml_svc_rbf_density_rich | the same SVC on the 346-column matrix (+ MLM / lift-bank channels) | 0.965210 |
|  | cuml_rff_logreg_density | cuML logistic regression on 4096 random Fourier features of the 213-column matrix | 0.965029 |
|  | cuml_rf_density_rich | cuML random forest (300 trees, depth 20) on the 346-column matrix | 0.967049 |
|  | cuml_built_features_xgb | XGBoost on 99 cuML-built features (TargetEncoder, KMeans, kNN, LR margin, embedding PCA) | 0.968557 |
|  | cuml_rf_nested_margins | 3 cuML random forests on the per-fold top 40 of cuML features + MLM channels + lift banks + nested LR / SVC / kNN margins | 0.968667 |
| [gbdt_variants](models/gbdt_variants) | lgbm_exact_te36 | LightGBM on raw + 23 derived + 18 original-data rates + 36 nested exact-value TEs | 0.967857 |
|  | xgb_dual_smooth_te | XGBoost on 49 base columns + 24 exact-value TEs at smoothing 20 and 2 | 0.968152 |
|  | catboost_quantile_bins | CatBoost on quantile bins, conditional-median imputations and native categoricals | 0.960246 |
| [neural_variants](models/neural_variants) | node_oblivious_te | NODE (2 x 128 oblivious trees, depth 6) on raw + derived + original rates + nested TEs | 0.956847 |
|  | wide_elasticnet_te | elastic-net logistic regression on nested TEs, their products, raw + derived, original rates | 0.959015 |
|  | ftt_imputed_tokens_te | FT-Transformer on imputation features, value-token embeddings and nested TEs | 0.966821 |
| [deep_forest](models/deep_forest) | gcforest_rf_et_te | gcForest cascade (RandomForest + ExtraTrees) on raw + derived + original rates + nested TEs | 0.963966 |
| [glassbox](models/glassbox) | rulefit_rf_et_te | RuleFit: random-forest / ExtraTrees rules (depth 3) + elastic-net logistic regression | 0.960875 |
|  | spline_gam_nb_poly2 | exact-token naive Bayes + spline GAM, combined by a degree-2 polynomial logistic regression with original-data evidence | 0.950482 |
|  | ebm_ga2m_te | EBM with 25 pairwise interactions on raw + derived columns and 4 nested TEs | 0.958392 |
| [factorization_machines](models/factorization_machines) | fm_exact_tokens | factorization machine (k 16) on the 12 exact value tokens | 0.963130 |
|  | afn_tokens_density_heads | AFN-style pair-product network (k 32, 8 heads) on the tokens + imputation and density-model side input | 0.967969 |
| [generative_classifiers](models/generative_classifiers) | naive_bayes_exact_tokens | naive Bayes over the 12 exact value tokens, logistic calibration | 0.944478 |
|  | bayes_llr_channels_catboost | CatBoost on 105 generative channels (token LLRs, pair lifts, budget and original-data LLRs) | 0.966292 |
| [tabpfn](models/tabpfn) | tabpfn_v3_context_ensemble | TabPFN v3, 8 stratified 50k-row contexts per fold; raw + nested exact-value TEs + 4 derived columns | 0.966141 |
| [public_models/nawfeelrahman1124444_realmlp](models/public_models/nawfeelrahman1124444_realmlp) | nawfeelrahman1124444_realmlp | nawfeelrahman1124444's notebook: custom 16-member RealMLP, digit / bin / combo categories, fold-local TE + frequencies | 0.968255 |
| [public_models/omidbaghchehsaraei_realmlp](models/public_models/omidbaghchehsaraei_realmlp) | omidbaghchehsaraei_realmlp | omidbaghchehsaraei's notebook: pytabkit RealMLP, ratio features + fold-local TE / frequencies | 0.968680 |
| [public_models/omidbaghchehsaraei_xgb_v2](models/public_models/omidbaghchehsaraei_xgb_v2) | omidbaghchehsaraei_xgb_v2 | omidbaghchehsaraei's notebook: XGBoost v2: ratio / cyclic / decile-bin features + inner-OOF TE / frequencies | 0.968659 |
| [public_models/mikhailnaumov_xgb](models/public_models/mikhailnaumov_xgb) | mikhailnaumov_xgb | mikhailnaumov's notebook: XGBoost: original-data target means, rounding frequencies, constrained imputation, TE | 0.968702 |
| [public_models/najiama_lgbm](models/public_models/najiama_lgbm) | najiama_lgbm | najiama's notebook: LightGBM: constrained imputation, ratio / flag features, frequencies + inner-OOF numeric TE | 0.968488 |
| [public_models/redamountassir_lgbm](models/public_models/redamountassir_lgbm) | redamountassir_lgbm | redamountassir's notebook: LightGBM: constrained imputation + frequencies + fold TargetEncoder | 0.968259 |
| [public_models/ern711_multilevel_spline_transformer](models/public_models/ern711_multilevel_spline_transformer) | ern711_multilevel_spline_transformer | ern711's notebook: multi-level deep univariate spline transformer (+ stage-2 hypernetwork) | 0.966793 |
| [public_models/omidbaghchehsaraei_tabm](models/public_models/omidbaghchehsaraei_tabm) | omidbaghchehsaraei_tabm | omidbaghchehsaraei's notebook: pytabkit TabM, ratio features + fold-local TE / frequencies | 0.967801 |
| [public_models/omidbaghchehsaraei_cnn](models/public_models/omidbaghchehsaraei_cnn) | omidbaghchehsaraei_cnn | omidbaghchehsaraei's notebook: tabular CNN on PLR + periodic embeddings, fold TE / frequencies | 0.967646 |
| [public_models/redamountassir_hgb](models/public_models/redamountassir_hgb) | redamountassir_hgb | redamountassir's notebook: HistGradientBoosting: constrained imputation + frequencies + fold TargetEncoder | 0.968018 |
| [public_models/omidbaghchehsaraei_tabnet](models/public_models/omidbaghchehsaraei_tabnet) | omidbaghchehsaraei_tabnet | omidbaghchehsaraei's notebook: TabNet, ratio features + fold-local TE / frequencies | 0.967774 |
| [public_models/zhenruiweng_realmlp](models/public_models/zhenruiweng_realmlp) | zhenruiweng_realmlp | zhenruiweng's notebook: custom 8-member RealMLP, missing flags / value categories + fold TargetEncoder | 0.968180 |
| [public_models/ern711_contextual_spline_transformer](models/public_models/ern711_contextual_spline_transformer) | ern711_contextual_spline_transformer | ern711's notebook: contextualized deep univariate spline transformer | 0.966833 |
| [public_models/omidbaghchehsaraei_ft_transformer](models/public_models/omidbaghchehsaraei_ft_transformer) | omidbaghchehsaraei_ft_transformer | omidbaghchehsaraei's notebook: pytabkit FT-Transformer, ratio features + fold-local TE / frequencies | 0.967269 |
| [public_models/szymonkapiski_lattice_te_xgb](models/public_models/szymonkapiski_lattice_te_xgb) | szymonkapiski_lattice_te_xgb | szymonkapiski's notebook: XGBoost on full-lattice (single + pair cell) TE / count features | 0.967663 |
| [public_models/omidbaghchehsaraei_flaml_xgb](models/public_models/omidbaghchehsaraei_flaml_xgb) | omidbaghchehsaraei_flaml_xgb | omidbaghchehsaraei's notebook: FLAML-tuned XGBoost, fold-local TE / frequencies | 0.966447 |
| [public_models/omidbaghchehsaraei_tabtransformer](models/public_models/omidbaghchehsaraei_tabtransformer) | omidbaghchehsaraei_tabtransformer | omidbaghchehsaraei's notebook: TabTransformer (PLR tokens, mixup, EMA), fold TE / frequencies | 0.967496 |
| [public_models/omidbaghchehsaraei_resnet](models/public_models/omidbaghchehsaraei_resnet) | omidbaghchehsaraei_resnet | omidbaghchehsaraei's notebook: pytabkit ResNet, ratio features + fold-local TE / frequencies | 0.966670 |
| [public_models/abdullahsafwan333_lgbm](models/public_models/abdullahsafwan333_lgbm) | abdullahsafwan333_lgbm | abdullahsafwan333's notebook: LightGBM on domain + digit features | 0.966918 |
| [public_models/omidbaghchehsaraei_catboost](models/public_models/omidbaghchehsaraei_catboost) | omidbaghchehsaraei_catboost | omidbaghchehsaraei's notebook: pytabkit CatBoost, ratio features + fold-local TE / frequencies | 0.967334 |
| [public_models/omidbaghchehsaraei_xgb](models/public_models/omidbaghchehsaraei_xgb) | omidbaghchehsaraei_xgb | omidbaghchehsaraei's notebook: XGBoost, native categoricals + ratio features + fold-local TE / frequencies | 0.967041 |
| [public_models/abdullahsafwan333_xgb](models/public_models/abdullahsafwan333_xgb) | abdullahsafwan333_xgb | abdullahsafwan333's notebook: XGBoost on domain + digit features | 0.966692 |
| [public_models/donmarch14_catboost](models/public_models/donmarch14_catboost) | donmarch14_catboost | donmarch14's notebook: CatBoost: digit / floor categoricals + inner-OOF combo / quantile-bin TE | 0.966667 |
| [public_models/vh10935cse20_lgbm](models/public_models/vh10935cse20_lgbm) | vh10935cse20_lgbm | vh10935cse20's notebook: LightGBM: usage features, group z-scores, combo TE | 0.966486 |
| [public_models/gauravduttaiiitb_flaml](models/public_models/gauravduttaiiitb_flaml) | gauravduttaiiitb_flaml | gauravduttaiiitb's notebook: FLAML AutoML single estimator on one-hot raw features | 0.958878 |
| [stack](stack) | | **L2 logistic regression on the 102 pairs above** | **0.971014** |

Rows of one folder are listed in recipe order; "+" means "the previous row plus". OOF AUCs are from our competition runs
(5 folds, `StratifiedKFold(5, shuffle=True, random_state=42)`).

## How to run
1. Put the data in `data/` (see [data/README.md](data/README.md)).
2. Forensic pipeline, in this order (each folder's README has its commands):
   - `models/forensic_features`;
   - `models/lgbm_margin`: its fold-cache builder writes the forensic fold matrices that most forensic folders read;
   - `models/xgb_forensic`: its cache builders (observed joint, source kNN);
   - `models/realmlp_repair`: the chain of density, completion, reconstruction, lift and repair caches, which the other
     forensic folders also read;
   - `models/pairlift_models`;
   - then the models of `xgb_forensic`, `catboost_forensic`, `tabm_forensic`, `lgbm_margin`, `realmlp_repair`,
     `structural_nets` and `cuml_models`.
3. Density pipeline: `models/density_features`, then `models/lift_banks`, then the models of `xgb_density_margin`,
   `catboost_density`, `realmlp_density`, `tabm_density`, `lookup_transformer`, `dcn` (its cache builder also feeds `ydf`),
   `ydf` and `cuml_density_models`.
4. The other folders (`gbdt_variants`, `neural_variants`, `deep_forest`, `glassbox`, `factorization_machines`,
   `generative_classifiers`, `tabpfn`) and every `models/public_models/*` folder are self-contained.
5. `python stack/stack.py` finds every pair under `models/*/preds/` and `models/public_models/**/preds/`, prints the stack
   CV and writes `stack/submission.csv`.

The scripts use the default CUDA device; pick a GPU with `CUDA_VISIBLE_DEVICES`. Most models need one large GPU.

## Requirements
Base (`requirements.txt`): numpy, pandas, pyarrow, scipy, scikit-learn, xgboost, lightgbm, catboost, torch, pytabkit, ydf,
interpret, tabpfn, huggingface_hub, flaml, pytorch-tabnet. cuML / cuDF / CuPy (RAPIDS) run the cuML models and make the
stacker fast. Model-specific notes are in each folder's README; the TabPFN v3 weights are downloaded by the model
package, and their licence must be accepted on Hugging Face first.
