#!/usr/bin/env python
"""EXAONE-Tabular, zero-shot, full context, on the 21 raw features.

Context = all outer-train rows of the fold (559,708 rows) with their labels; queries = the held-out fold + all test rows.
8 ensemble members; EXAONE's default compute dtype (float16).
Two changes to the stock classifier:
  * support_row_limit is raised to the context size (the stock default of 100,000 silently subsamples the context in fit());
  * the query chunk size is set to 8,192. At full context EXAONE's worst-case memory estimate picks one query row per forward
    (about 16 h per fold), although the fused attention needs far less memory. Queries only attend to the cached support, so
    the chunk size changes speed and memory, not the predictions.
Input = the raw float array (EXAONE takes NumPy only and auto-detects columns with < 10 levels as categorical); NaN kept.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'exaone'
N_ESTIMATORS = 8
QUERY_STEP = 8192


def patch_query_step(k):
    """Keep EXAONE's own execution plan but override its query chunk size (GPU-resident support cache built once per member)."""
    import dataclasses

    import exaonetabular._execution as ex
    orig = ex._InferenceExecutor._plan_with_budget

    def plan(self, model, support_batch, query_batch, *args, **kw):
        p = orig(self, model, support_batch, query_batch, *args, **kw)
        if p.query_step < k:
            p = dataclasses.replace(p, ensemble_step=1, query_step=min(k, query_batch.shape[1]), support_only_cache=False,
                                    query_cache=True, cache_to_cpu=False, reload_cache=False, compact_cache=True)
        return p
    ex._InferenceExecutor._plan_with_budget = plan


def predict(Xc, yc, Xq, fold):
    from dataclasses import replace

    from exaonetabular import EXAONETabularClassifier
    from exaonetabular.presets import released_checkpoint
    base = released_checkpoint('classification').manifest
    base = replace(base, runtime=replace(base.runtime, support_row_limit=max(len(Xc), base.runtime.support_row_limit)))
    clf = EXAONETabularClassifier.from_pretrained(device='cuda:0', manifest=base, ensemble_count=N_ESTIMATORS, seed=42 + fold)
    clf.fit(Xc.astype(np.float32), yc)
    return clf.predict_proba(Xq.astype(np.float32))[:, 1].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    patch_query_step(QUERY_STEP)
    X, Xt, _ = raw_arrays()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        tr, va = f != k, f == k
        p = predict(X[tr], y[tr], np.concatenate([X[va], Xt]), k)
        nv = int(va.sum())
        save_part(HERE, NAME, k, p[:nv], p[nv:])
        print(f'fold {k}: AUC {roc_auc_score(y[va], p[:nv]):.6f}', flush=True)
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
