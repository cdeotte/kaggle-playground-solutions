#!/usr/bin/env python
"""Exact-income class mixtures by EM from labeled counts and unlabeled context.

For a label pool `fit`: an XGB context teacher is trained on the pool rows on the stage-1 features without any
income-derived column, plus the 1000-bin income center, the source-law score at that center, the leading income BPE
piece and the parent key = (income // 1000, leading BPE piece), and predicts every row outside the pool. Its logits are shifted per
parent so the mean predicted rate matches the pool's parent rate (20 pseudo-rows); the result, minus the parent
log-odds, is the context log-likelihood ratio of each unlabeled row. The rate of each exact income is then estimated by
EM: labeled counts + alpha x parent prior + weight x the unlabeled rows' posteriors (alpha 1 and 20; weight 0, 0.25 and
1, plus a control with the ratios permuted within parent).
Per outer fold k the pool is each inner split of the outer-train rows (features for the inner-held rows) and then all
outer-train rows (features for fold k + test). Writes cache/stage1/semi_f{k}.npz (+ .json with the column names).
Usage: python semisupervised.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.pairs import fold_list  # noqa: E402
from features import Builder  # noqa: E402
from layout import STAGE1 as CACHE  # noqa: E402

TEACHER = dict(objective='binary:logistic', tree_method='hist', device='cuda', eta=.05, max_depth=4, min_child_weight=30,
               reg_lambda=10, subsample=.85, colsample_bytree=.9, max_bin=512, num_parallel_tree=2, seed=42)


def infer(builder, fit, query):
    import xgboost as xgb
    alln = len(builder.base)
    unlab = np.setdiff1d(np.arange(alln), fit)
    X, V, names, types = builder.build(fit, unlab, noise=.04)
    keep = []
    for j, n in enumerate(names):
        low = n.lower()
        if 'income' in low or 'inc' in low or low.startswith('source_gap') or low in ['source_exact', 'source_score',
                                                                                     'source_probability', 'source_logit']:
            continue
        keep.append(j)
    kk = builder.keys
    parent = pd.factorize(pd.MultiIndex.from_arrays([kk.inc1000.to_numpy(), kk.inc_bpe0.to_numpy()]), sort=True)[0]
    sz = int(parent.max())+1
    pc = np.bincount(parent[fit], minlength=sz)
    ps = np.bincount(parent[fit], weights=builder.y[fit], minlength=sz)
    pi = ((ps+20*builder.y[fit].mean())/(pc+20)).clip(1e-4, 1-1e-4)
    original = builder.base
    inc = original.Annual_Income_USD.to_numpy()
    coarse = np.floor(inc/1000)*1000+500
    rule = original.source_score.to_numpy()+1.2*(coarse-inc)/1e5
    X = np.column_stack([X[:, keep], coarse[fit], rule[fit], kk.inc_bpe0.to_numpy()[fit], parent[fit]])
    V = np.column_stack([V[:, keep], coarse[unlab], rule[unlab], kk.inc_bpe0.to_numpy()[unlab], parent[unlab]])
    names = [names[j] for j in keep]+['context_coarse_income', 'context_coarse_rule', 'context_bpe', 'context_parent']
    types = [types[j] for j in keep]+['q', 'q', 'c', 'c']
    dt = xgb.DMatrix(X, label=builder.y[fit], feature_names=names, feature_types=types, enable_categorical=True)
    dq = xgb.DMatrix(V, feature_names=names, feature_types=types, enable_categorical=True)
    model = xgb.train(TEACHER, dt, num_boost_round=600, verbose_eval=False)
    q = model.predict(dq).clip(1e-5, 1-1e-5).astype('float64')
    z = logit(q)
    pu = parent[unlab]
    nu = np.bincount(pu, minlength=sz)
    # match the implied parent rate to the pool's parent rate
    adjust = np.zeros(sz)
    for _ in range(12):
        q = expit(z+adjust[pu])
        s = np.bincount(pu, weights=q, minlength=sz)
        h = np.bincount(pu, weights=q*(1-q), minlength=sz)
        step = (pi*nu-s)/np.maximum(h, 1e-3)
        adjust += np.clip(step, -1, 1)
    llr = z+adjust[pu]-logit(pi[pu])
    # control: ratios permuted within parent
    null = llr.copy()
    rng = np.random.default_rng(421337)
    for ix in pd.Series(np.arange(len(unlab))).groupby(pu).indices.values():
        null[ix] = rng.permutation(llr[ix])
    code = kk.Annual_Income_USD.to_numpy()
    g, u, v = code[fit], code[unlab], code[query]
    ng = builder.meta['key_sizes']['Annual_Income_USD']
    cnt = np.bincount(g, minlength=ng)
    pos = np.bincount(g, weights=builder.y[fit], minlength=ng)
    uc = np.bincount(u, minlength=ng)
    parent_for_value = np.zeros(ng, dtype='int64')
    parent_for_value[code] = parent
    prior = pi[parent_for_value]
    values, fn = [], []
    for alpha in [1., 20.]:
        start = (pos+alpha*prior)/(cnt+alpha)
        for w, label, lr in [(0., 'w0', llr), (.25, 'w25', llr), (1., 'w1', llr), (1., 'null', null)]:
            p = start.copy()
            for it in range(100 if w else 0):
                r = expit(logit(p.clip(1e-5, 1-1e-5))[u]+lr)
                pnew = (pos+alpha*prior+w*np.bincount(u, weights=r, minlength=ng))/(cnt+alpha+w*uc)
                change = float(np.max(abs(pnew-p)))
                p = pnew
                if change < 1e-7:
                    break
            values.append(logit(p.clip(1e-5, 1-1e-5))[v])
            fn.append(f'semi_{label}_a{alpha:g}')
    return np.column_stack(values).astype('float32'), fn


def build(b, k):
    path = CACHE/f'semi_f{k}.npz'
    if path.exists():
        return
    folds = np.load(CACHE/'fold_id.npy')
    fit = np.flatnonzero(folds != k)
    valid = np.flatnonzero(folds == k)
    query = np.r_[valid, np.arange(b.n, len(b.base))]
    F = np.empty((len(fit), 8), dtype='float32')
    for j, (a, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(fit, b.y[fit])):
        F[v], names = infer(b, fit[a], fit[v])
        print('EM inner', k, j, flush=True)
    Q, names = infer(b, fit, query)
    np.savez(path, fit=fit, query=query, train_features=F, query_features=Q)
    (CACHE/f'semi_f{k}.json').write_text(json.dumps(dict(feature_names=names, fold=k), indent=2))
    print('EM fold', k, 'done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    b = Builder()
    for k in fold_list(a.folds):
        build(b, k)


if __name__ == '__main__':
    main()
