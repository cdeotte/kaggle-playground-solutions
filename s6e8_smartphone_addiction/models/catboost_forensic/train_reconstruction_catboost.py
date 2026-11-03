#!/usr/bin/env python
"""CatBoost on the forensic CatBoost matrix plus predictor-reconstruction and density-completion evidence.

Features: the matrix of train_forensic_catboost.py (forensic float32 columns, 114 fixed-original source channels,
native fragment categories, semantic categories), then
  71 numeric reconstruction columns (masked autoencoder / XGBoost completions, uncertainties, residuals; the first 71
     columns of the reconstruction cache of ../realmlp_repair)
  10 density-completion aggregates ('__aggregate__' columns of the density-completion cache of ../realmlp_repair)
  30 reconstruction categories ('core' view of lib/recon_features.py)
minus the 120 pruned numeric columns of lib/recon_features.DROP_FEATURE_NAMES and the two margin columns: 280 features,
103 categorical. The forensic base margin (logit(prior) + 0.40 raw_te_woe_sum + 0.10 unsupported-pair response) is
CatBoost's baseline. CatBoost (GPU): 24,000 iterations, depth 6, lr 0.01, l2_leaf_reg 30, random_strength 0.3,
Bernoulli subsample 0.9, border_count 254, one_hot_max_size 16, max_ctr_complexity 3, score_function NewtonL2,
model_size_reg 0, seed 42 + fold; no early stopping. Test = mean of the 5 fold models.
Usage: python train_reconstruction_catboost.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.catboost_forensic.lib.recon_features import (DROP_FEATURE_NAMES, load_density,  # noqa: E402
                                                         load_reconstruction, parent_frames,
                                                         reconstruction_categories, reconstruction_view_indices)

PAIR = "catboost_forensic_reconstruction"
PARAMS = {"iterations": 24000, "depth": 6, "learning_rate": 0.01, "l2_leaf_reg": 30.0, "random_strength": 0.3,
          "bootstrap_type": "Bernoulli", "subsample": 0.9, "border_count": 254, "one_hot_max_size": 16,
          "max_ctr_complexity": 3, "score_function": "NewtonL2", "model_size_reg": 0.0}
MARGIN_COLUMNS = ["strong__selected_base_margin", "strong__selected_base_probability"]


def run_fold(fold):
    from catboost import CatBoostClassifier, Pool
    frames, cat_names, y, train_idx, valid_idx, margins = parent_frames(fold)

    recon_names, *recon = load_reconstruction(fold)
    density_names, *density = load_density(fold, "compact10")
    selected = reconstruction_view_indices(recon_names, "compact71")
    numeric_names = ([f"reconstruction__{recon_names[i]}" for i in selected]
                     + [f"density_response__{name}" for name in density_names])
    frames = [pd.concat([frame.reset_index(drop=True), pd.DataFrame(
        np.column_stack([matrix[:, selected], dens]).astype(np.float32), columns=numeric_names)], axis=1)
        for frame, matrix, dens in zip(frames, recon, density)]

    raw_train_all = read_train()
    raw_rows = [raw_train_all.iloc[train_idx].reset_index(drop=True),
                raw_train_all.iloc[valid_idx].reset_index(drop=True), read_test().reset_index(drop=True)]
    recon_cat_train, recon_cat_apply, recon_cat_names = reconstruction_categories(
        raw_rows[0], raw_rows[1:], recon[0], recon[1:], recon_names)
    frames = [pd.concat([frame, cats], axis=1) for frame, cats in zip(frames, [recon_cat_train, *recon_cat_apply])]
    cat_names = list(cat_names) + recon_cat_names
    frames = [frame.drop(columns=MARGIN_COLUMNS + list(DROP_FEATURE_NAMES)) for frame in frames]
    names = frames[0].columns.tolist()
    print(f"fold {fold}: {len(names)} features, {len(cat_names)} categorical", flush=True)

    pools = [Pool(frames[0], label=y[train_idx], cat_features=cat_names, feature_names=names, baseline=margins[0]),
             Pool(frames[1], cat_features=cat_names, feature_names=names, baseline=margins[1]),
             Pool(frames[2], cat_features=cat_names, feature_names=names, baseline=margins[2])]
    del frames
    gc.collect()
    model = CatBoostClassifier(**PARAMS, loss_function="Logloss", eval_metric="AUC", task_type="GPU", devices="0",
                               random_seed=42 + fold, thread_count=8, allow_writing_files=False, verbose=500)
    model.fit(pools[0])
    valid_pred = np.asarray(model.predict_proba(pools[1])[:, 1], np.float64)
    test_pred = np.asarray(model.predict_proba(pools[2])[:, 1], np.float64)
    save_part(HERE, PAIR, fold, valid_pred, test_pred)
    from sklearn.metrics import roc_auc_score
    print(f"{PAIR} fold {fold}: AUC {roc_auc_score(y[valid_idx], valid_pred):.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    for fold in fold_list(args.folds):
        if not have_part(HERE, PAIR, fold):
            run_fold(fold)
    y = target()
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
