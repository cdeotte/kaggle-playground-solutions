#!/usr/bin/env python
"""pairlift_lgbm: the forensic CPU LightGBM (fixed 4,400 rounds, selected base margin as init score) + the 10 pair-lift
columns.

Features per outer fold (shared forensic fold cache + lib/pairlift.py), in this order:
  strong__* (104) and fixed_source__* (114) numerics, minus strong__selected_base_margin / _probability
  log1p outer-training counts of the native_base (20) and semantic (53) categories, 6 category-count summaries
  10 pair-lift columns                                                                   -> 305 features
The selected base margin (strong__selected_base_margin) is LightGBM's init_score (boost_from_average off); predictions
are sigmoid(tree raw score + margin). No early stopping; one model per fold, test = mean of the 5 fold models.
Usage: python train_lgbm.py [--folds 0,1,2,3,4] [--threads 12]"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import pairlift  # noqa: E402
from lib.paths import fold_indices  # noqa: E402
from lib.upstream import MARGIN, PROBABILITY, FoldCache  # noqa: E402

PAIR = 'pairlift_lgbm'
NUM_BOOST_ROUND = 4400
PARAMS = {'learning_rate': 0.02, 'num_leaves': 63, 'max_depth': 6, 'min_data_in_leaf': 500, 'max_bin': 511,
          'feature_fraction': 0.85, 'bagging_fraction': 0.9, 'bagging_freq': 1, 'lambda_l1': 0.05, 'lambda_l2': 24.0}


def fold_matrix(cache, fold, role):
    """(values without the margin columns, feature names, init score)."""
    numeric_idx = cache.numeric_indices(['strong', 'fixed_source'])
    frequency_idx = cache.category_indices(['native_base', 'semantic'])
    parts = [np.asarray(cache.array(role, 'numeric')[:, numeric_idx], np.float32),
             np.asarray(cache.array(role, 'category_frequency')[:, frequency_idx], np.float32),
             np.asarray(cache.array(role, 'category_summary'), np.float32),
             pairlift.load(fold, role)]
    names = ([cache.numeric_names[i] for i in numeric_idx] + [cache.frequency_names[i] for i in frequency_idx]
             + cache.summary_names + list(pairlift.FEATURE_NAMES))
    values = np.column_stack(parts).astype(np.float32)
    init_score = values[:, names.index(MARGIN)].astype(np.float64)
    keep = [i for i, n in enumerate(names) if n not in (MARGIN, PROBABILITY)]
    return np.ascontiguousarray(values[:, keep]), [names[i] for i in keep], init_score


def predict(model, values, init_score):
    raw = np.asarray(model.predict(values, raw_score=True), np.float64) + init_score
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0)))


def run_fold(fold, y, fold_ids, threads):
    train_idx, valid_idx = fold_indices(fold, fold_ids)
    cache = FoldCache(fold)
    train, names, train_init = fold_matrix(cache, fold, 'train')
    valid, _, valid_init = fold_matrix(cache, fold, 'valid')
    test, _, test_init = fold_matrix(cache, fold, 'test')
    params = {'objective': 'binary', 'metric': 'auc', 'device_type': 'cpu', 'deterministic': True,
              'force_col_wise': True, 'num_threads': threads, 'seed': 42 + fold, 'bagging_seed': 143 + fold,
              'feature_fraction_seed': 253 + fold, 'data_random_seed': 349 + fold, 'extra_seed': 443 + fold,
              'verbosity': -1, **PARAMS, 'boost_from_average': False}
    train_set = lgb.Dataset(train, label=y[train_idx], init_score=train_init, feature_name=names,
                            categorical_feature=[], free_raw_data=True)
    model = lgb.train(params, train_set, num_boost_round=NUM_BOOST_ROUND, callbacks=[lgb.log_evaluation(period=200)])
    valid_pred = predict(model, valid, valid_init)
    test_pred = predict(model, test, test_init)
    print(f'fold {fold}: {len(names)} features, AUC {roc_auc_score(y[valid_idx], valid_pred):.6f}', flush=True)
    return valid_pred, test_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--threads', type=int, default=12)
    args = ap.parse_args()
    y = target()
    fold_ids = folds(y)
    for fold in fold_list(args.folds):
        if have_part(HERE, PAIR, fold):
            continue
        valid_pred, test_pred = run_fold(fold, y, fold_ids, args.threads)
        save_part(HERE, PAIR, fold, valid_pred, test_pred)
    assemble(HERE, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
