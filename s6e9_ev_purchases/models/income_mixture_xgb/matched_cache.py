#!/usr/bin/env python
"""Fold matrices of the XGB models: for outer fold k, the stage-1 features with source-conditional intercepts
(score 'intercept'; encodings cross-fitted on the outer-train rows, one noise stream per column, sd 0.04) and the
nested LR margins of nested_margin.py (computed here if missing).
Writes cache/stage1/matched_f{k}/{train, query, fit_index, query_index, train_margin, query_margin}.npy + manifest.json
(feature_names, feature_types, n_valid).
Usage: python matched_cache.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.pairs import fold_list  # noqa: E402
from features import Builder  # noqa: E402
from layout import STAGE1 as CACHE  # noqa: E402
from nested_margin import margins  # noqa: E402


def prepare_fold(b, k):
    p = CACHE/f'matched_f{k}'
    if (p/'manifest.json').exists():
        return
    p.mkdir(parents=True, exist_ok=True)
    f = np.load(CACHE/'fold_id.npy')
    fit = np.flatnonzero(f != k)
    valid = np.flatnonzero(f == k)
    query = np.r_[valid, np.arange(b.n, len(b.base))]
    X, V, names, types = b.build(fit, query, score='intercept', noise_mode='column')
    mf, mq, qi = margins(b, k)
    assert np.array_equal(query, qi)
    for name, value in [('train', X), ('query', V), ('fit_index', fit), ('query_index', query), ('train_margin', mf),
                        ('query_margin', mq)]:
        np.save(p/(name+'.npy'), value)
    (p/'manifest.json').write_text(json.dumps(dict(fold=k, feature_names=names, feature_types=types, n_valid=len(valid)),
                                              indent=2))
    print('fold matrices', k, X.shape, V.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    b = Builder()
    for k in fold_list(a.folds):
        prepare_fold(b, k)


if __name__ == '__main__':
    main()
