"""Shared code of the forensic LightGBM models (the fold cache is also read by ../tabm_forensic and the RealMLP models).

target_encoder.py  CPUExactTargetEncoder: exact-key nested target encoder with cuML TargetEncoder's interface
fold_cache.py      layout, builders and loaders of the per-fold feature cache (build_fold_features.py)
"""
