#!/usr/bin/env python
"""C of copula_gam_contrast and copula_gpt2_contrast: an XGB on the information-aware EM base features + density-ratio
experts, started from the nested response-logit margin R.

Per outer fold k, columns of the outer-train / query rows:
  - the em_info base pack of ../income_mixture_xgb (cache/em_info/base/fold_<k>), keeping compact.select's columns
    without the source_pair_core block, + compact.build's source-rule geometry and inner-held conditional intercepts;
    Gaussian noise (sd 0.04, lib.train_noise) on the target-encoding columns of the training rows
  - the 4 inc_global EM columns of cache/em_info/features_gpu/fold_<k> (feature weight 0.5)
  - copula_density.py's experts after 350 and 700 rounds: class-0 margin, class-1 margin and their difference
    (feature weight 0.5; inner-held for the training rows)
base_margin = R (response_margin.py --link logit). XGB: depth 7, eta 0.03, 4 parallel trees, min_child_weight 10,
lambda 5, alpha 0.1, subsample 0.85, colsample_bytree 0.85, colsample_bynode 0.8, max_bin 512, 400 rounds, seeds 42-49.
Writes cache/contrast/copula/models/production_raw/fold_<k>/seed_<s>/{valid,test}.npy (probabilities).
Needs lib.py / compact.py and the em_info caches of ../income_mixture_xgb.
Usage: python copula_xgb.py [--folds 0,1,2,3,4] [--seeds 42-49]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.append(str(HERE.parent / 'income_mixture_xgb'))
from contrast_data import CONTRAST, EM_INFO, ints  # noqa: E402
from copula_density import CHANNELS  # noqa: E402
from compact import build, select  # noqa: E402
from lib import Data, splits, train_noise  # noqa: E402

CHECKPOINTS = [350, 700]
DENSITY_WEIGHT = .5
ROUNDS = 400
PARAMS = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=7, eta=.03, num_parallel_tree=4,
              min_child_weight=10, reg_lambda=5, reg_alpha=.1, subsample=.85, colsample_bytree=.85, colsample_bynode=.8,
              max_bin=512)


def pack(folder):
    """train.npy / query.npy of a fold pack and the column names in its manifest.json."""
    return (np.load(folder / 'train.npy'), np.load(folder / 'query.npy'),
            json.loads((folder / 'manifest.json').read_text())['names'])


def density_features(d, k):
    a, b = np.flatnonzero(d.folds != k), np.flatnonzero(d.folds == k)
    lookup = np.full(d.n, -1, np.int64)
    lookup[a] = np.arange(len(a))
    T = Q = None
    for j, (fit, apply) in enumerate([(a, np.r_[b, d.test])] + splits(a, d.y)):
        P = np.load(CONTRAST / 'copula' / 'densities' / 'raw' / f'fold_{k}' / f'pool_{j}' / 'features.npy')
        columns = []
        for rounds in CHECKPOINTS:
            c0, c1 = P[:, CHANNELS.index(f'raw__class0__r{rounds}')], P[:, CHANNELS.index(f'raw__class1__r{rounds}')]
            columns += [c0, c1, c1 - c0]
        block = np.column_stack(columns).astype(np.float32)
        if j == 0:
            Q, T = block, np.full((len(a), block.shape[1]), np.nan, np.float32)
        else:
            T[lookup[apply]] = block
    assert np.isfinite(T).all() and np.isfinite(Q).all()
    names = [f'copula__raw__r{r}__{s}' for r in CHECKPOINTS for s in ('negative', 'positive', 'difference')]
    return T, Q, names


def run_fold(d, k, seeds):
    import xgboost as xgb
    root = CONTRAST / 'copula' / 'models' / 'production_raw' / f'fold_{k}'
    seeds = [s for s in seeds if not (root / f'seed_{s}' / 'test.npy').exists()]
    if not seeds:
        return
    a, b = np.flatnonzero(d.folds != k), np.flatnonzero(d.folds == k)
    T, Q, names = pack(EM_INFO / 'base' / f'fold_{k}')
    ix = [j for j in select(names) if not names[j].startswith('source_pair_core__')]
    T, Q, names = T[:, ix], Q[:, ix], [names[j] for j in ix]
    A, B, new = build(d, k, True)
    T, Q, names = np.column_stack([T, A]), np.column_stack([Q, B]), names + new
    T = train_noise(T, names, k, sd=.04)
    weights = np.ones(len(names), np.float32)
    A, B, nm = pack(EM_INFO / 'features_gpu' / f'fold_{k}')
    ix = [j for j, n in enumerate(nm) if n.startswith('inc_global__')]
    assert len(ix) == 4
    T, Q = np.column_stack([T, A[:, ix]]), np.column_stack([Q, B[:, ix]])
    names += [nm[j] for j in ix]
    weights = np.r_[weights, np.full(4, .5, np.float32)]
    A, B, nm = density_features(d, k)
    T, Q, names = np.column_stack([T, A]), np.column_stack([Q, B]), names + nm
    weights = np.r_[weights, np.full(len(nm), DENSITY_WEIGHT, np.float32)]
    folder = CONTRAST / 'margins' / 'response_logit' / f'fold_{k}'
    mt, mq = np.load(folder / 'train.npy'), np.load(folder / 'query.npy')
    T, Q = T.astype(np.float32), Q.astype(np.float32)
    assert np.isfinite(T).all() and np.isfinite(Q).all()
    dt = xgb.DMatrix(T, label=d.y[a], base_margin=mt, feature_weights=weights)
    dq = xgb.DMatrix(Q, base_margin=mq)
    for seed in seeds:
        start = time.time()
        model = xgb.train({**PARAMS, 'seed': seed}, dt, num_boost_round=ROUNDS)
        p = model.predict(dq).astype(np.float64)
        out = root / f'seed_{seed}'
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'valid.npy', p[:len(b)])
        np.save(out / 'test.npy', p[len(b):])
        print(f'copula XGB fold {k} seed {seed}: {T.shape} ({time.time() - start:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--seeds', default='42-49')
    args = ap.parse_args()
    d = Data()
    for k in ints(args.folds):
        run_fold(d, k, ints(args.seeds))


if __name__ == '__main__':
    main()
