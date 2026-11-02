"""Nested Bayes posterior margins (em_info): context log-likelihood ratio of the income teacher + exact-income mixture
log-odds (mean of alpha 1 and 20), with EM ('em_posterior'), labeled-only and permuted-ratio EM.
Writes cache/em_info/bayes_margins/fold_{k}/{train,query}.npy + manifest.json (names)."""
import json

import numpy as np

from lib import BAYES, FEATURES, TEACHERS, dump
from mixtures import calibrated_llr


def build(d, fold, log=print):
    root = BAYES/f'fold_{fold}'
    if (root/'manifest.json').exists():
        return np.load(root/'train.npy'), np.load(root/'query.npy')
    a = np.flatnonzero(d.folds != fold)
    b = np.flatnonzero(d.folds == fold)
    q = np.r_[b, d.test]
    local = np.full(d.n, -1, np.int64)
    local[a] = np.arange(len(a))
    fr = FEATURES/f'fold_{fold}'
    names = json.loads((fr/'manifest.json').read_text())['names']
    F = np.load(fr/'train.npy')
    G = np.load(fr/'query.npy')
    T = np.empty((len(a), 3), np.float32)
    Q = None
    for pool in range(6):
        tr = TEACHERS/'income'/f'fold_{fold}'/f'pool_{pool}'
        fit = np.load(tr/'fit.npy')
        p = np.load(tr/'p.npy')
        unlabeled = np.setdiff1d(np.arange(d.N), fit)
        llr, prior = calibrated_llr(p, d.keys['inc_parent'], fit, unlabeled, d.y)
        apply = q if pool == 0 else np.setdiff1d(a, fit)
        pos = np.searchsorted(unlabeled, apply)
        assert np.array_equal(unlabeled[pos], apply)
        features = G if pool == 0 else F[local[apply]]
        columns = []
        for tag, use_em in [('inc_global', True), ('inc_global', False), ('shuffle', True)]:
            estimates = []
            for alpha in [1, 20]:
                z = features[:, names.index(f'{tag}__labeled_a{alpha}')].astype(np.float64)
                if use_em:
                    z = z+features[:, names.index(f'{tag}__delta_a{alpha}')]
                estimates.append(z)
            columns.append(llr[pos]+np.mean(estimates, axis=0))
        V = np.column_stack(columns).astype(np.float32)
        if pool == 0:
            Q = V
        else:
            T[local[apply]] = V
    root.mkdir(parents=True, exist_ok=True)
    assert np.isfinite(T).all() and np.isfinite(Q).all()
    np.save(root/'train.npy', T)
    np.save(root/'query.npy', Q)
    dump(root/'manifest.json', dict(fold=fold, names=['em_posterior', 'labeled_posterior', 'shuffled_em_posterior']))
    log(f'Bayes margins fold={fold} {T.shape} {Q.shape}')
    return T, Q
