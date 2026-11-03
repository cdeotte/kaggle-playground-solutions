#!/usr/bin/env python
"""LightGBM on the per-fold forensic cache, boosted from the forensic base margin (LightGBM init_score).

Recipes (pair name):
  lgbm_forensic_margin_frequency  numeric strong + fixed_source columns, log-frequencies of the native_base and semantic
                                  categories, the 6 category summaries; 63 leaves, max_depth 6, min_data_in_leaf 500,
                                  lambda_l2 24; 4,400 rounds
  lgbm_forensic_margin_wide       numeric strong + fixed_source columns only; 127 leaves, unlimited depth,
                                  min_data_in_leaf 350, lambda_l2 32; 1,800 rounds
Both: init_score = strong__selected_base_margin (logit(prior) + 0.40 raw_te_woe_sum + 0.10 unsupported-pair response;
the margin and probability columns are dropped from the features), boost_from_average off, lr 0.02, max_bin 511,
feature_fraction 0.85, bagging 0.9 / 1, lambda_l1 0.05, CPU deterministic, 12 threads, seeds 42 / 143 / 253 / 349 / 443
+ fold; no early stopping. Prediction = sigmoid(clip(tree raw score + init_score, -50, 50)). Test = mean of the 5 fold
models.
Usage: python train_lgbm.py <pair> [--folds 0,1,2,3,4] [--threads 12]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.lgbm_margin.lib.fold_cache import fold_root, load, load_manifest  # noqa: E402

BASE_PARAMS = {"learning_rate": 0.02, "max_bin": 511, "feature_fraction": 0.85, "bagging_fraction": 0.9,
               "bagging_freq": 1, "lambda_l1": 0.05}
RECIPES = {
    "lgbm_forensic_margin_frequency": dict(
        numeric_groups=["strong", "fixed_source"], frequency_groups=["native_base", "semantic"],
        include_category_summary=True, num_boost_round=4400,
        params=dict(BASE_PARAMS, num_leaves=63, max_depth=6, min_data_in_leaf=500, lambda_l2=24.0)),
    "lgbm_forensic_margin_wide": dict(
        numeric_groups=["strong", "fixed_source"], frequency_groups=[], include_category_summary=False,
        num_boost_round=1800, params=dict(BASE_PARAMS, num_leaves=127, min_data_in_leaf=350, lambda_l2=32.0)),
}
MARGIN, PROBABILITY = "strong__selected_base_margin", "strong__selected_base_probability"


def ordered_union(groups, mapping):
    selected = []
    for group in groups:
        selected += [index for index in mapping[group] if index not in selected]
    return selected


def matrix_for_role(fold, manifest, recipe, role):
    """Selected numeric columns, then category log-frequencies, then category summaries; returns (values, names,
    init_score) with the margin columns removed and the margin returned as init_score."""
    numeric_idx = ordered_union(recipe["numeric_groups"], manifest["numeric_groups"])
    frequency_idx = ordered_union(recipe["frequency_groups"], manifest["category_groups"])
    parts = [np.asarray(load(fold, role, "numeric")[:, numeric_idx], dtype=np.float32)]
    names = [manifest["base_numeric_names"][i] for i in numeric_idx]
    if frequency_idx:
        parts.append(np.asarray(load(fold, role, "category_frequency")[:, frequency_idx], dtype=np.float32))
        names += [manifest["category_frequency_names"][i] for i in frequency_idx]
    if recipe["include_category_summary"]:
        parts.append(np.asarray(load(fold, role, "category_summary"), dtype=np.float32))
        names += manifest["category_summary_names"]
    values = np.ascontiguousarray(np.column_stack(parts).astype(np.float32, copy=False))
    init_score = np.asarray(values[:, names.index(MARGIN)], np.float64) * 1.0
    keep = [i for i, name in enumerate(names) if name not in (MARGIN, PROBABILITY)]
    return np.ascontiguousarray(values[:, keep], dtype=np.float32), [names[i] for i in keep], init_score


def predict_probability(model, values, init_score, num_iteration=None):
    tree_score = np.asarray(model.predict(values, num_iteration=num_iteration, raw_score=True), np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(tree_score + init_score, -50.0, 50.0)))


def run_fold(pair, fold, threads):
    import lightgbm as lgb
    recipe = RECIPES[pair]
    manifest = load_manifest(fold)
    y = target()
    train_idx = np.load(fold_root(fold) / "train_indices.npy")
    valid_idx = np.load(fold_root(fold) / "valid_indices.npy")
    train, names, train_init = matrix_for_role(fold, manifest, recipe, "train")
    train_set = lgb.Dataset(train, label=y[train_idx], init_score=train_init, feature_name=names,
                            categorical_feature=[], free_raw_data=True)
    params = {"objective": "binary", "metric": "auc", "device_type": "cpu", "deterministic": True,
              "force_col_wise": True, "num_threads": threads, "seed": 42 + fold, "bagging_seed": 143 + fold,
              "feature_fraction_seed": 253 + fold, "data_random_seed": 349 + fold, "extra_seed": 443 + fold,
              "verbosity": -1}
    params.update(recipe["params"])
    params["boost_from_average"] = False
    print(f"{pair} fold {fold}: {len(names)} features", flush=True)
    model = lgb.train(params, train_set, num_boost_round=recipe["num_boost_round"],
                      callbacks=[lgb.log_evaluation(period=200)])
    del train, train_set
    valid, _, valid_init = matrix_for_role(fold, manifest, recipe, "valid")
    valid_pred = predict_probability(model, valid, valid_init, num_iteration=recipe["num_boost_round"])
    test, _, test_init = matrix_for_role(fold, manifest, recipe, "test")
    test_pred = predict_probability(model, test, test_init)
    save_part(HERE, pair, fold, valid_pred, test_pred)
    from sklearn.metrics import roc_auc_score
    print(f"{pair} fold {fold}: AUC {roc_auc_score(y[valid_idx], valid_pred):.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(RECIPES))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    for fold in fold_list(args.folds):
        if not have_part(HERE, args.pair, fold):
            run_fold(args.pair, fold, args.threads)
    y = target()
    assemble(HERE, args.pair, folds(y), y)


if __name__ == "__main__":
    main()
