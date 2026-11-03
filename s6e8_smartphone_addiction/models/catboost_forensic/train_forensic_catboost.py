#!/usr/bin/env python
"""CatBoost on the forensic matrix + source-fragment categories, started from the forensic base margin.

Features: the 102 forensic columns of ../xgb_forensic as float32 (lib/strong_cat.py), the fixed-original univariate
LLR and log-count channels (114 columns), native source-fragment categories (raw_cat, mask, exact_value,
semantic_state) and the semantic interaction categories (source_qbins, source_q16_pairs, imputed_screen, source_rule).
The forensic base margin (logit(prior) + 0.40 raw_te_woe_sum + 0.10 unsupported-pair response) is CatBoost's
baseline and is not a feature. CatBoost (GPU): 24,000 iterations, depth 6, lr 0.01, l2_leaf_reg 30,
random_strength 0.3, Bernoulli subsample 0.9, border_count 254, one_hot_max_size 16, max_ctr_complexity 2,
score_function NewtonL2, model_size_reg 0, seed 42 + fold; no early stopping. Test = mean of the 5 fold models.
Usage: python train_forensic_catboost.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.catboost_forensic.lib.strong_cat import (build_strong_application, cat_semantic_categories,  # noqa: E402
                                                     fixed_source_features, native_categories, source_test_features)

PAIR = "catboost_forensic_fragments"
CAT_GROUPS = ["raw_cat", "mask", "exact_value", "semantic_state"]
SOURCE_NUMERIC_GROUPS = ["univariate_llr_logcount"]
SEMANTIC_GROUPS = ["source_qbins", "source_q16_pairs", "imputed_screen", "source_rule"]
PARAMS = {"iterations": 24000, "depth": 6, "learning_rate": 0.01, "l2_leaf_reg": 30.0, "random_strength": 0.3,
          "bootstrap_type": "Bernoulli", "subsample": 0.9, "border_count": 254, "one_hot_max_size": 16,
          "max_ctr_complexity": 2, "score_function": "NewtonL2", "model_size_reg": 0.0}
MARGIN_COLUMNS = ["strong__selected_base_margin", "strong__selected_base_probability"]


def run_fold(fold):
    from catboost import CatBoostClassifier, Pool
    (strong_train, strong_valid, strong_test, y, train_idx, valid_idx,
     margin_train, margin_valid, margin_test) = build_strong_application(fold)
    n_train = len(y)
    test_rows = n_train + np.arange(len(strong_test), dtype=np.int64)
    native = [native_categories(rows, CAT_GROUPS) for rows in (train_idx, valid_idx, test_rows)]
    cat_names = list(native[0][1])
    semantic_train, semantic_apply, semantic_names = cat_semantic_categories(
        strong_train, pd.concat([strong_valid, strong_test], ignore_index=True), SEMANTIC_GROUPS)
    semantic = [semantic_train, semantic_apply.iloc[:len(valid_idx)].reset_index(drop=True),
                semantic_apply.iloc[len(valid_idx):].reset_index(drop=True)]
    cat_names += semantic_names
    source_train, source_cat_train, source_cat_names = fixed_source_features(train_idx, SOURCE_NUMERIC_GROUPS)
    source_valid, source_cat_valid, _ = fixed_source_features(valid_idx, SOURCE_NUMERIC_GROUPS)
    source_test, source_cat_test = source_test_features(source_train.columns.tolist(), source_cat_names)
    cat_names += source_cat_names
    frames = []
    for strong, source, nat, source_cat, sem in zip((strong_train, strong_valid, strong_test),
                                                    (source_train, source_valid, source_test), native,
                                                    (source_cat_train, source_cat_valid, source_cat_test), semantic):
        frames.append(pd.concat([strong, source, nat[0], source_cat, sem], axis=1).drop(columns=MARGIN_COLUMNS))
    names = frames[0].columns.tolist()
    pools = [Pool(frames[0], label=y[train_idx], cat_features=cat_names, feature_names=names, baseline=margin_train),
             Pool(frames[1], cat_features=cat_names, feature_names=names, baseline=margin_valid),
             Pool(frames[2], cat_features=cat_names, feature_names=names, baseline=margin_test)]
    del frames
    gc.collect()
    print(f"fold {fold}: {len(names)} features, {len(cat_names)} categorical", flush=True)
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
