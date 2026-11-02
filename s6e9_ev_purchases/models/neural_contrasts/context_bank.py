#!/usr/bin/env python
"""Label-free core income-context factors (bank core_income of fine_context_contrast).

Contexts of each row: subsidy x anxiety, subsidy x concern, anxiety x concern, all three, exact age and exact commute.
For every exact income value: its context counts minus count x the parent's profile (parent = income // 1000 + leading BPE
piece; parent profile smoothed with 0.5 pseudo-counts), divided by (count + 20) and a per-cell scale
sqrt(profile + 0.05 / n_cells) * sqrt(n_cells). Basis: 32 right singular vectors of these residuals (rows weighted by
sqrt(min(count, 500)); randomized SVD, 5 iterations, seed 42). Each row gets its value's projected residual with its own
contexts removed (real_0..31), plus per-context log posterior / parent ratios of its own context cell (real_at_<context>);
the null_* columns repeat this with the contexts shuffled within parents.
Writes cache/stage5/core_income_all.npy (+ .json names, _state.npz).
Usage: python context_bank.py"""
import time

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from contrast_data import STAGE, Stage1, within_parent_permutation, write_names

ALPHA = 20


def compute():
    start = time.time()
    b = Stage1()
    d, keys, meta = b.base, b.keys, b.meta

    def code(a):
        return pd.factorize(a, sort=True)[0].astype('int32')

    def joint(*arrays):
        return code(pd.MultiIndex.from_arrays(arrays))
    key = keys['Annual_Income_USD'].to_numpy()
    ng = meta['key_sizes']['Annual_Income_USD']
    parent = joint(keys['inc1000'], keys['inc_bpe0'])
    npg = parent.max() + 1
    pv = np.zeros(ng, 'int32')
    pv[key] = parent
    sub = d.Subsidy_Available.to_numpy('int32')
    anx = d.Range_Anxiety_Level.to_numpy('int32')
    conc = d.Environmental_Concern_Level.to_numpy('int32')
    ctx = dict(sa=joint(sub, anx), sc=joint(sub, conc), ac=joint(anx, conc), sac=joint(sub, anx, conc), age=keys.Age.to_numpy())
    ctx['other_exact'] = keys['Daily_Commute_km'].to_numpy()
    perm = within_parent_permutation(parent, len(d))
    n = np.bincount(key, minlength=ng)
    pc = np.bincount(parent, minlength=npg)
    counts, shuffled, prior, offsets, offset = [], [], [], [], 0
    for cname, c in ctx.items():
        ns = int(c.max()) + 1
        offsets.append((cname, offset, ns))
        offset += ns
        counts.append(np.bincount(key * ns + c, minlength=ng * ns).reshape(ng, ns))
        shuffled.append(np.bincount(key * ns + c[perm], minlength=ng * ns).reshape(ng, ns))
        a = np.bincount(parent * ns + c, minlength=npg * ns).reshape(npg, ns)
        prior.append((a + .5 / ns) / (pc[:, None] + .5))
    C = np.column_stack(counts).astype('float64')
    S = np.column_stack(shuffled).astype('float64')
    P = np.column_stack(prior)
    del counts, shuffled, prior
    scale = np.column_stack([np.sqrt(P[:, st:st + ns] + .05 / ns) * np.sqrt(ns) for _, st, ns in offsets])
    R = (C - n[:, None] * P[pv]) / (n[:, None] + ALPHA) / scale[pv]
    weighted = R * np.sqrt(np.minimum(n, 500))[:, None]
    _, singular, V = randomized_svd(weighted, n_components=min(32, min(weighted.shape) - 1), n_iter=5, random_state=42)
    basis = V.T
    del R, weighted
    output, names = [], []
    for mode, A, contexts in [('real', C, ctx), ('null', S, {c: v[perm] for c, v in ctx.items()})]:
        full = ((A - n[:, None] * P[pv]) / scale[pv]) @ basis
        parproj = (P[:, :, None] * basis[None, :, :] / scale[:, :, None]).sum(axis=1)
        values = full[key] + parproj[parent]
        for cname, st, ns in offsets:
            ix = contexts[cname] + st
            values -= basis[ix] / scale[parent, ix][:, None]
        values /= (n[key, None] - 1 + ALPHA)
        output.append(values.astype('float32'))
        names += [f'{mode}_{j}' for j in range(values.shape[1])]
        scores = []
        for cname, st, ns in offsets:
            ix = contexts[cname] + st
            pr = P[parent, ix]
            obs = A[key, ix] - 1
            post = (obs + ALPHA * pr) / (n[key] - 1 + ALPHA)
            scores.append(np.log((post + 1e-9) / (pr + 1e-9)))
        output.append(np.column_stack(scores).astype('float32'))
        names += [f'{mode}_at_{c}' for c, _, _ in offsets]
    Z = np.column_stack(output)
    assert np.isfinite(Z).all()
    STAGE[5].mkdir(parents=True, exist_ok=True)
    np.save(STAGE[5] / 'core_income_all.npy', Z)
    np.savez(STAGE[5] / 'core_income_state.npz', basis=basis, singular=singular, key=key, parent=parent, scale=scale,
             parent_profile=P)
    write_names(STAGE[5] / 'core_income_all.npy', names, contexts=[c for c, _, _ in offsets])
    print(f'core_income bank {Z.shape} ({time.time() - start:.0f}s)', flush=True)


if __name__ == '__main__':
    if not (STAGE[5] / 'core_income_all.npy').exists():
        compute()
