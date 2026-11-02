#!/usr/bin/env python
"""Causilo, zero-shot, full context, on the 21 raw features.

Context = all outer-train rows of the fold (559,708 rows) with their labels; queries = the held-out fold + all test rows.
8 estimators; Causilo's own precision policy (fp16 autocast for classification).
Streaming: Causilo's built-in use_kv_cache=True builds the caches of all 8 estimators at fit time and keeps them on the GPU
(out of memory at full context). stream_proba() uses Causilo's own cache functions one estimator at a time
(build_cache -> cached_predictions -> free), in the same member order, and averages the logits + softmax as the Engine does.
The 4 categoricals are passed as pandas category columns; NaN kept.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'causilo'
N_ESTIMATORS = 8


def frame(X, meta):
    df = pd.DataFrame(X, columns=meta['features'])
    for j in meta['cat_idx']:
        c = meta['features'][j]
        df[c] = pd.Categorical(df[c].round().astype(int).map(dict(enumerate(meta['cat_codes'][c]))))
    return df


def stream_proba(clf, Xq):
    """== Engine.predict for a use_kv_cache fit, but each member's cache is built, used and freed in turn."""
    import torch
    from causilo.execution.cached import cached_predictions
    from causilo.execution.runner import ModelRunner
    eng = clf._engine
    state = eng.require_state()
    assert state.codec is None, 'binary classification expected (no ECOC codec)'
    fitted = state.dataset
    transformed = fitted.query_tables(Xq)
    runner = ModelRunner(eng.model)
    logits = []
    with torch.inference_mode():
        for i, member in enumerate(fitted.members_by_normalization()):
            features = fitted.training_table(member.normalization)[:, member.feature_order]
            cache = runner.build_cache(eng._tensor(features), eng._tensor(fitted.targets_for(member)))
            for m, result in cached_predictions(eng.model, fitted, transformed, ((member, cache),), eng.device, lambda v: v):
                logits.append(result[:, m.class_order].numpy())
            del cache
            torch.cuda.empty_cache()
            print(f'  member {i + 1} done', flush=True)
    combined = np.mean(logits, axis=0)
    prob = np.exp(combined - combined.max(axis=-1, keepdims=True))
    return prob / prob.sum(axis=-1, keepdims=True)


def predict(Xc, yc, Xq, meta, fold):
    from causilo import CausiloClassifier
    clf = CausiloClassifier(n_estimators=N_ESTIMATORS, random_state=42 + fold, device='cuda', use_kv_cache=False)
    clf.fit(frame(Xc, meta), yc)
    return stream_proba(clf, frame(Xq, meta))[:, 1].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    X, Xt, meta = raw_arrays()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        tr, va = f != k, f == k
        p = predict(X[tr], y[tr], np.concatenate([X[va], Xt]), meta, k)
        nv = int(va.sum())
        save_part(HERE, NAME, k, p[:nv], p[nv:])
        print(f'fold {k}: AUC {roc_auc_score(y[va], p[:nv]):.6f}', flush=True)
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
