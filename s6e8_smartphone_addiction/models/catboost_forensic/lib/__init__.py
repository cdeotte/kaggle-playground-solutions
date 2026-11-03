"""Shared code of the forensic CatBoost models (strong_cat.py is also used by ../lgbm_margin).

strong_cat.py      the forensic matrix of ../xgb_forensic as float32 'strong__' columns plus its base margin, the cached
                   source-fragment categories, the fixed-original source channels and the semantic interaction categories
recon_features.py  the parent CatBoost frames and the reconstruction / density-completion inputs (caches of
                   ../realmlp_repair) of train_reconstruction_catboost.py
"""
