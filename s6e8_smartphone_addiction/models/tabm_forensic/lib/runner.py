"""One outer fold of the forensic TabM (pytabkit TabM_D_Classifier).

Frames: the row-local blocks of lib/row_features.py built on all train rows / all test rows, the selected columns of the
../lgbm_margin fold cache (default: the 104 'strong__' columns), optional extra blocks, then the fold-fitted
preprocessing of lib/fold_features.py (pair contrasts, outer-training median fill, string categoricals).
Model: TabM-D, k = 32, 2 blocks of 512, d_embedding 16, piecewise-linear numeric embeddings (48 bins) after
quantile_tabr, dropout 0.1, lr 0.002, batch 1024, AMP, gradient clipping 1.0, no weight decay, at most 40 epochs with
patience 8 on 1 - AUC; n_cv internal members, each early-stopped on its own stratified holdout of val_fraction of the
outer-training rows (StratifiedShuffleSplit, seed 20260824 + fold); random_state 142 + fold."""
import os
import random

import numpy as np

from common.data import TARGET, labels, read_test, read_train

from .fold_features import prepare_fold_frames
from .forensic_cache import append_cache_features, load_fold, numeric_matrix, resolve_cache_columns
from .inner_splits import repeated_stratified_holdouts
from .row_features import build_rowwise_features

BASE_BLOCKS = ["raw", "exact_grid_onehots", "smooth_latent_geometry", "missing_gated_views", "fixed_bins",
               "sparse_pair_contrast_12"]
MODEL_DEFAULTS = {
    "allow_amp": True, "arch_type": "tabm", "batch_size": 1024, "compile_model": False, "d_block": 512,
    "d_embedding": 16, "dropout": 0.1, "gradient_clipping_norm": 1.0, "lr": 0.002, "n_blocks": 2, "n_cv": 1,
    "n_epochs": 40, "n_refit": 0, "n_threads": 4, "num_emb_n_bins": 48, "num_emb_type": "pwl", "patience": 8,
    "share_training_batches": False, "tabm_k": 32, "tfms": ["quantile_tabr"], "val_fraction": 0.05,
    "val_metric_name": "1-auc_ovr", "verbosity": 0, "weight_decay": 0.0,
}
INNER_SEED_BASE = 20260824


def configure_environment():
    """Thread counts and the cuBLAS workspace setting of the original runs (set before torch is imported)."""
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def default_rowwise(frame, blocks, split):
    return build_rowwise_features(frame, blocks)


def run_fold(fold, model_params, include_regex=("^strong__",), blocks=BASE_BLOCKS, rowwise_builder=default_rowwise,
             append_extra=None):
    """rowwise_builder(frame, blocks, split) builds the row-local frame of all rows of train.csv (split 'train') or
    test.csv ('test'); append_extra(frame, role) adds columns to the train / valid / test frames after the cache
    columns. Returns (validation prediction, test prediction, validation labels)."""
    import torch
    from pytabkit import TabM_D_Classifier

    manifest, train_idx, valid_idx = load_fold(fold)
    selected_idx, selected_names = resolve_cache_columns(manifest["base_numeric_names"], list(include_regex))
    train = read_train()
    target = labels(train[TARGET])
    base_full, categorical_names = rowwise_builder(train, blocks, "train")
    frames = {}
    for role, rows in (("train", train_idx), ("valid", valid_idx)):
        frame = base_full.iloc[rows].reset_index(drop=True).copy()
        frame = append_cache_features(frame, numeric_matrix(fold, role), selected_idx, selected_names)
        frames[role] = append_extra(frame, role) if append_extra is not None else frame
    base_test, _ = rowwise_builder(read_test(), blocks, "test")
    test_frame = append_cache_features(base_test, numeric_matrix(fold, "test"), selected_idx, selected_names)
    if append_extra is not None:
        test_frame = append_extra(test_frame, "test")
    train_frame, valid_frame, test_frame, fold_categorical = prepare_fold_frames(
        frames["train"], frames["valid"], test_frame, categorical_names, blocks)
    print(f"fold {fold}: {train_frame.shape[1]} features, {len(fold_categorical)} categorical", flush=True)

    n_cv = int(model_params["n_cv"])
    val_idxs = None
    if n_cv > 1:
        val_idxs = repeated_stratified_holdouts(train_idx, target, n_splits=n_cv,
                                                validation_fraction=float(model_params["val_fraction"]),
                                                random_state=INNER_SEED_BASE + fold)
    fold_seed = 142 + fold
    params = dict(model_params, device="cuda:0", random_state=fold_seed)
    seed_everything(fold_seed, torch)
    model = TabM_D_Classifier(**params)
    model.fit(train_frame, target[train_idx], cat_col_names=fold_categorical, val_idxs=val_idxs)
    valid_pred = model.predict_proba(valid_frame)[:, 1].astype(np.float64)
    test_pred = model.predict_proba(test_frame)[:, 1].astype(np.float64)
    return valid_pred, test_pred, target[valid_idx]
