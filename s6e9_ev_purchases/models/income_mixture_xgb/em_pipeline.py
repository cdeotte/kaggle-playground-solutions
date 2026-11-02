#!/usr/bin/env python
"""Features of the information-aware income EM model, per outer fold (finished steps are skipped):
  1. base: static blocks of ../response_features + log counts of the raw fields + nested target encodings of the raw
     fields, income / commute bins, the leading income BPE piece and the parent (income // 1000, leading piece);
     the nested response margin of ../response_features  -> cache/em_info/base/fold_{k}/
  2. income context teachers: XGB on the raw fields with income coarsened to its 1000-bin center + nested encodings
     without exact income; pool 0 = all outer-train rows, pools 1-5 = inner splits; predicts every row
     -> cache/em_info/teachers_gpu/income/fold_{k}/pool_{j}/{p.npy, fit.npy}
  3. exact-income EM features (mixtures.py)            -> cache/em_info/features_gpu/fold_{k}/
  4. information features (information.py)             -> cache/em_info/information_features/fold_{k}/
  5. Bayes posterior margins (bayes_margin.py)         -> cache/em_info/bayes_margins/fold_{k}/
Also writes cache/em_info/{folds, train_ids, test_ids}.npy.
Usage: python em_pipeline.py [--folds 0,1,2,3,4]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.pairs import fold_list  # noqa: E402
import bayes_margin  # noqa: E402
import information  # noqa: E402
import mixtures  # noqa: E402
from lib import BASE_DIR, EM_INFO, FEATURES, INFORMATION, TEACHERS, Data, dump, splits  # noqa: E402

TEACHER = {'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cuda', 'max_depth': 4, 'eta': .05,
           'num_parallel_tree': 2, 'min_child_weight': 30, 'lambda': 10, 'subsample': .85, 'colsample_bytree': .9,
           'max_bin': 512, 'seed': 42}


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def baseline(d, fold):
    folder = BASE_DIR/f'fold_{fold}'
    if (folder/'manifest.json').exists():
        return
    a, b, X, Q, names = d.base(fold)
    folder.mkdir(parents=True, exist_ok=True)
    for name, v in [('train', X), ('query', Q), ('train_indices', a), ('valid_indices', b)]:
        np.save(folder/f'{name}.npy', v)
    M = d.margin(fold)
    np.save(folder/'margin_train.npy', M[0])
    np.save(folder/'margin_query.npy', np.r_[M[1], M[2]])
    dump(folder/'manifest.json', {'names': names, 'fold': fold})
    log(f'base fold={fold} shape={X.shape}/{Q.shape}')


def teacher(d, fold, field='income'):
    import xgboost as xgb
    a = np.flatnonzero(d.folds != fold)
    b = np.flatnonzero(d.folds == fold)
    plan = [(a, np.r_[b, d.test])]+splits(a, d.y)
    for j, (fit, _) in enumerate(plan):
        folder = TEACHERS/field/f'fold_{fold}'/f'pool_{j}'
        if (folder/'p.npy').exists():
            continue
        query = np.setdiff1d(np.arange(d.N), fit)
        X, Q = d.context(fit, query, field)
        dt = xgb.DMatrix(X, label=d.y[fit])
        dq = xgb.DMatrix(Q)
        model = xgb.train(TEACHER, dt, num_boost_round=600)
        p = np.empty(d.N, np.float32)
        p[fit] = model.predict(dt)
        p[query] = model.predict(dq)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder/'fit.npy', fit)
        np.save(folder/'p.npy', p)
        dump(folder/'manifest.json', {'params': TEACHER, 'rounds': 600, 'fold': fold, 'pool': j, 'field': field,
                                      'fit_count': len(fit), 'query_count': len(query)})
        log(f'teacher field={field} fold={fold} pool={j} done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    d = Data()
    EM_INFO.mkdir(parents=True, exist_ok=True)
    np.save(EM_INFO/'folds.npy', d.folds)
    np.save(EM_INFO/'train_ids.npy', d.ids)
    np.save(EM_INFO/'test_ids.npy', d.test_ids)
    for fold in fold_list(a.folds):
        baseline(d, fold)
        teacher(d, fold, 'income')
        if not (FEATURES/f'fold_{fold}'/'manifest.json').exists():
            mixtures.build(d, fold, log)
        if not (INFORMATION/f'fold_{fold}'/'manifest.json').exists():
            information.build(d, fold, log)
        bayes_margin.build(d, fold, log)
        log(f'fold {fold} complete')


if __name__ == '__main__':
    main()
