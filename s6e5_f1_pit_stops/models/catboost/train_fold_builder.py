#!/usr/bin/env python
"""GPU CatBoost on the per-fold features of lib/fold_builder.py with the fold-matched original subset.

Per outer fold k: the original rows outside fold k of the same 5-fold split of the original data (common.data.orig_folds)
are the original subset. The builder is fit on the fold's training rows + the original subset + the test rows, the
original subset (source flag 1) is appended to the training side with sample weight `original_weight`, and CatBoost
early-stops on the validation fold (use_best_model=True). Seed 42 + fold number (1-5).
Recipes (pair name):
  catboost_pair_grid_original_subset_w07       with all 36 pairs of 9 keys as native categoricals (475 columns, 84
                                               categorical); depth 8, lr 0.024, l2 10, original weight 0.7
  catboost_anchor_priors_original_subset_w085  without the pair grid (403 columns, 48 categorical); depth 9, lr 0.028,
                                               l2 12, original weight 0.85
Test = mean of the 5 fold models (probabilities clipped to [1e-7, 1 - 1e-7]).

Usage: python train_fold_builder.py <pair> [--folds 0,1,2,3,4] [--force]"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, orig_folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_part  # noqa: E402
from models.catboost.lib.fold_builder import FoldFeatureBuilder, align_columns, prepare_cat  # noqa: E402
from models.catboost.lib.runner import FOLDER, finish, fold_parser, todo_folds  # noqa: E402

SEED = 42
CARDINALITY_CAT_THRESHOLD = 100
COMMON_PARAMS = dict(allow_writing_files=False, bootstrap_type="Bayesian", border_count=254, devices="0",
                     eval_metric="AUC", loss_function="Logloss", od_type="Iter", task_type="GPU", use_best_model=True,
                     verbose=250)
RECIPES = {
    "catboost_pair_grid_original_subset_w07": dict(
        use_pair_grid=True, original_weight=0.7,
        params=dict(COMMON_PARAMS, bagging_temperature=0.85, depth=8, iterations=15000, l2_leaf_reg=10.0,
                    learning_rate=0.024, od_wait=750, random_strength=0.75)),
    "catboost_anchor_priors_original_subset_w085": dict(
        use_pair_grid=False, original_weight=0.85,
        params=dict(COMMON_PARAMS, bagging_temperature=0.75, depth=9, iterations=13000, l2_leaf_reg=12.0,
                    learning_rate=0.028, od_wait=650, random_strength=0.8)),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    from catboost import CatBoostClassifier, Pool

    args = fold_parser(RECIPES).parse_args()
    recipe = RECIPES[args.pair]
    todo = todo_folds(args.pair, args.folds, args.force)
    if todo:
        seed_everything(SEED)
        train, test, orig = read_train(), read_test(), read_original()
        feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
        y = train[TARGET].astype("int8").reset_index(drop=True)
        y_orig = orig[TARGET].astype("int8").reset_index(drop=True)

        object_cats = [c for c in feature_cols if train[c].dtype == "object" or str(train[c].dtype).startswith("string")]
        numeric_like_cols = [c for c in feature_cols if c not in object_cats]
        low_card_numeric_cols = [c for c in numeric_like_cols if train[c].nunique(dropna=False) < CARDINALITY_CAT_THRESHOLD]
        fold_id = folds(y.to_numpy())
        orig_fold_id = orig_folds(y_orig.to_numpy())

        for k in todo:
            seed_everything(SEED + k + 1)
            tr_idx, val_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            orig_tr_idx = np.where(orig_fold_id != k)[0]

            X_tr_raw = train.iloc[tr_idx][feature_cols].reset_index(drop=True)
            y_tr_syn = y.iloc[tr_idx].reset_index(drop=True)
            X_val_raw = train.iloc[val_idx][feature_cols].reset_index(drop=True)
            y_val = y.iloc[val_idx].reset_index(drop=True)
            X_test_raw = test[feature_cols].reset_index(drop=True)
            X_orig_raw = orig.iloc[orig_tr_idx][feature_cols].reset_index(drop=True)
            y_orig_tr = y_orig.iloc[orig_tr_idx].reset_index(drop=True)

            builder = FoldFeatureBuilder(feature_cols, object_cats, low_card_numeric_cols, numeric_like_cols,
                                         recipe["use_pair_grid"]).fit(X_tr_raw, X_orig_raw, y_orig_tr, X_test_raw)
            X_tr = builder.transform(X_tr_raw, source_flag=0)
            X_val = builder.transform(X_val_raw, source_flag=0)
            X_test_fold = builder.transform(X_test_raw, source_flag=0)
            X_orig = builder.transform(X_orig_raw, source_flag=1)

            X_train = pd.concat([X_tr, X_orig], axis=0, ignore_index=True)
            y_train = pd.concat([y_tr_syn, y_orig_tr], axis=0, ignore_index=True).astype("int8")
            sample_weight = np.concatenate([
                np.ones(len(X_tr), dtype="float32"),
                np.full(len(X_orig), recipe["original_weight"], dtype="float32"),
            ])
            X_train, X_val, X_test_fold = align_columns(X_train, X_val, X_test_fold)
            cat_cols = [c for c in builder.cat_cols_ if c in X_train.columns]

            X_fit, X_eval, X_tst, cat_cols = prepare_cat(X_train, X_val, X_test_fold, cat_cols)
            cat_idx = [X_fit.columns.get_loc(c) for c in cat_cols]
            params = dict(recipe["params"], random_seed=SEED + k + 1)
            print(f"{args.pair} fold {k}: {X_fit.shape[1]} columns ({len(cat_cols)} categorical), "
                  f"{len(X_fit)} training rows", flush=True)
            train_pool = Pool(X_fit, y_train, cat_features=cat_idx, weight=sample_weight)
            val_pool = Pool(X_eval, y_val, cat_features=cat_idx)
            test_pool = Pool(X_tst, cat_features=cat_idx)
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool)
            val_pred = model.predict_proba(val_pool)[:, 1].astype("float32")
            fold_test = model.predict_proba(test_pool)[:, 1].astype("float32")
            save_part(FOLDER, args.pair, k, val_pred, fold_test)
            print(f"{args.pair} fold {k}: AUC {roc_auc_score(y_val, val_pred):.6f}", flush=True)
    finish(args.pair, clip=1e-7)


if __name__ == "__main__":
    main()
