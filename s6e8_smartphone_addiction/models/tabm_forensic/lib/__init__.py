"""Shared code of the forensic TabM models.

row_features.py    label-free row-local TabM blocks (raw, exact grid, smooth latent geometry, missing-gated views,
                   fixed bins)
fold_features.py   fold-fitted pair contrasts, median fill and categorical casting
forensic_cache.py  reader of the ../lgbm_margin fold cache
inner_splits.py    repeated stratified inner holdouts (TabM's internal members)
completion.py      reconstruction / density-completion inputs of ../realmlp_repair and the dual completed view
runner.py          one outer fold: frames, TabM-D fit, validation and test predictions
"""
