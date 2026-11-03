"""Shared code of the forensic models (this folder, ../catboost_forensic, ../lgbm_margin, ../tabm_forensic).

parent.py              29 label-free parent columns, nested target encodings + WOE sums, partial-budget residuals
external.py            fold-owned external blocks: l2 screen imputer, screen-response slack logits
pair_evidence.py       observed-joint PMIs, exact original copula, unsupported-pair response, imputed screen response
strong.py              build_strong_fold(): the 102-column forensic matrix of one outer fold, base margin, XGBoost parameters
ae_features.py         the 25 masked-autoencoder fill features (reconstruction cache of ../realmlp_repair)
source_knn.py          mask-aware nearest-neighbour geometry / imputations against the original rows (read by ../cuml_models)
frontier_features.py   the 290-column frontier matrix, its completion base margin and XGBoost parameters
residual_ares.py       source-tempered exact-token responses ('ares' contrasts)
residual_surfaces.py   daily x social residual surfaces, splitmix64 owner groups
refit_xgb.py           selector + full refit XGBoost
region_lift.py         strict region x exact-value lifts
realmlp_embeddings.py  periodic / RBF coordinates of the raw numerics and semantic interaction constraints
"""
