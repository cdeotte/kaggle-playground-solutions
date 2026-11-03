# public_models

Models from other Kagglers' public notebooks (thank you!). These are the top public single-model notebooks of the S6E8
public leaderboard (notebooks that blend or stack several models, pseudo-label or post-process predictions were skipped;
a 5-fold average of one recipe counts as a single model). Each subfolder holds our run of one notebook's model on our
5 folds (`StratifiedKFold(5, shuffle=True, random_state=42)` on the target), with a README that credits the author, links
the notebook and says what we changed to run it on our folds. The changes are small: every notebook uses our 5 folds
(most already used the same split; others used another seed, 7 or 10 folds, or one full-data fit), target encodings
that a notebook computed on all training rows (or fitted on the training rows it then encoded) are recomputed inside
each fold with inner out-of-fold encodings for the fold's training rows, and rank post-processing is removed.

| Folder | Notebook | Author | OOF AUC |
|---|---|---|---:|
| [nawfeelrahman1124444_realmlp](nawfeelrahman1124444_realmlp) | [RealMLP --> 0.97014](https://www.kaggle.com/code/nawfeelrahman1124444/realmlp-0-97014) | nawfeelrahman1124444 | 0.968255 |
| [omidbaghchehsaraei_realmlp](omidbaghchehsaraei_realmlp) | [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/realmlp-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.968680 |
| [omidbaghchehsaraei_xgb_v2](omidbaghchehsaraei_xgb_v2) | [XGBoost v2 for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/xgboost-v2-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.968659 |
| [mikhailnaumov_xgb](mikhailnaumov_xgb) | [Smartphone Addiction \| XGB](https://www.kaggle.com/code/mikhailnaumov/smartphone-addiction-xgb) | Mikhail Naumov (@mikhailnaumov) | 0.968702 |
| [najiama_lgbm](najiama_lgbm) | [Single LGBM Model LB 0.96988 CV 0.96863](https://www.kaggle.com/code/najiama/single-lgbm-model-lb-0-96988-cv-0-96863) | Naji (@najiama) | 0.968488 |
| [redamountassir_lgbm](redamountassir_lgbm) | [S6E8: LGBM \| LB 0.96965](https://www.kaggle.com/code/redamountassir/s6e8-lgbm-lb-0-96965) | Reda Mountassir (@redamountassir) | 0.968259 |
| [ern711_multilevel_spline_transformer](ern711_multilevel_spline_transformer) | [Multi-Level Deep Univariate Spline Transformer](https://www.kaggle.com/code/ern711/multi-level-deep-univariate-spline-transformer) | Ern711 (@ern711) | 0.966793 |
| [omidbaghchehsaraei_tabm](omidbaghchehsaraei_tabm) | [TabM for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.967801 |
| [omidbaghchehsaraei_cnn](omidbaghchehsaraei_cnn) | [CNN for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.967646 |
| [redamountassir_hgb](redamountassir_hgb) | [S6E8: HistGradientBoosting \| LB 0.96945](https://www.kaggle.com/code/redamountassir/s6e8-histgradientboosting-lb-0-96945) | Reda Mountassir (@redamountassir) | 0.968018 |
| [omidbaghchehsaraei_tabnet](omidbaghchehsaraei_tabnet) | [TabNet for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabnet-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.967774 |
| [zhenruiweng_realmlp](zhenruiweng_realmlp) | [RealMLP for Predicting Smartphone Addiction](https://www.kaggle.com/code/zhenruiweng/realmlp-for-predicting-smartphone-addiction) | Zhenrui.Weng (@zhenruiweng) | 0.968180 |
| [ern711_contextual_spline_transformer](ern711_contextual_spline_transformer) | [Contextualized Deep Univariate Spline Transformer](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer) | Ern711 (@ern711) | 0.966833 |
| [omidbaghchehsaraei_ft_transformer](omidbaghchehsaraei_ft_transformer) | [FT-Transformer for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/ft-transformer-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.967269 |
| [szymonkapiski_lattice_te_xgb](szymonkapiski_lattice_te_xgb) | [S6E8 full lattice target encoding with XGBoost](https://www.kaggle.com/code/szymonkapiski/s6e8-full-lattice-target-encoding-with-xgboost) | Szymon Kłapiński (@szymonkapiski) | 0.967663 |
| [omidbaghchehsaraei_flaml_xgb](omidbaghchehsaraei_flaml_xgb) | [FLAML[XGBoost] for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/flaml-xgboost-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.966447 |
| [omidbaghchehsaraei_tabtransformer](omidbaghchehsaraei_tabtransformer) | [TabTransformer : Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/tabtransformer-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.967496 |
| [omidbaghchehsaraei_resnet](omidbaghchehsaraei_resnet) | [ResNet for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/resnet-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.966670 |
| [abdullahsafwan333_lgbm](abdullahsafwan333_lgbm) | [S6E8: LightGBM \| SAP](https://www.kaggle.com/code/abdullahsafwan333/s6e8-lightgbm-sap) | AbdullahSafwan333 (@abdullahsafwan333) | 0.966918 |
| [omidbaghchehsaraei_catboost](omidbaghchehsaraei_catboost) | [CatBoost for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/catboost-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.967334 |
| [omidbaghchehsaraei_xgb](omidbaghchehsaraei_xgb) | [XGBoost for Predicting Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/xgboost-for-predicting-smartphone-addiction) | OMID BAGHCHEH SARAEI (@omidbaghchehsaraei) | 0.967041 |
| [abdullahsafwan333_xgb](abdullahsafwan333_xgb) | [S6E8: XGBoost \| SAP](https://www.kaggle.com/code/abdullahsafwan333/s6e8-xgboost-sap) | AbdullahSafwan333 (@abdullahsafwan333) | 0.966692 |
| [donmarch14_catboost](donmarch14_catboost) | [S6E8: CatBoost](https://www.kaggle.com/code/donmarch14/s6e8-catboost) | Don Mani (@donmarch14) | 0.966667 |
| [vh10935cse20_lgbm](vh10935cse20_lgbm) | [Mobile Addiction \|\| LGBM](https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm) | SARANG (@vh10935cse20) | 0.966486 |
| [gauravduttaiiitb_flaml](gauravduttaiiitb_flaml) | [Ex51 : S6E8 : FLAML : roc_auc](https://www.kaggle.com/code/gauravduttaiiitb/ex51-s6e8-flaml-roc-auc) | Gaurav Dutta (@gauravduttaiiitb) | 0.958878 |

Run any of them with `python train.py` inside its folder; it writes `preds/<folder>_oof.npy` and `preds/<folder>_test.npy`
(probabilities, train.csv / test.csv row order) and needs only `data/`. Most scripts take `--folds 0,1,...`: finished folds
are kept in `<folder>/parts/` and skipped, and the pair is written once all 5 folds are done. Shared code in `lib/`:
`runner.py` (the fold loop), `ratio_te.py` (the ratio features and per-fold target / frequency encodings of the
omidbaghchehsaraei notebooks), `constrained_te.py` (the two redamountassir notebooks) and `sap_features.py` (the two
abdullahsafwan333 notebooks).

The stacker searches this folder and all its subfolders, so each subfolder's `preds/` pairs are stacked like any other model.
Pairs that someone shared publicly (without code) can be dropped into `preds/` here or into any subfolder's `preds/`:
- `<name>_oof.npy` (one value per train row, train.csv order) and `<name>_test.npy` (one value per test row, test.csv order), or
- `<name>_oof.csv` and `<name>_test.csv`, each with an `id` column and one prediction column.

Values can be probabilities or logits. Each pair must use the same folds as everything else
(`StratifiedKFold(5, shuffle=True, random_state=42)`); otherwise the stack CV is leaky.
