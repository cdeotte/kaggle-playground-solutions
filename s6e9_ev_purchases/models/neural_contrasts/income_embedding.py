#!/usr/bin/env python
"""Label-free income-context embedding (predictors of all train + test rows, no labels).

For every exact income value: its context histograms over 13 context families (subsidy x anxiety, subsidy x concern,
anxiety x concern, all three, age // 5, commute // 5, gender, city, car, home charging, cars owned, stations near home /
work) minus the parent's smoothed profile, scaled per cell (sqrt(parent profile + 0.01), equal energy per family), shrunk
by 20 pseudo-rows. A 16-dim basis = top right singular vectors of these residual profiles (rows weighted by
sqrt(min(count, 500))). Each row gets its income value's projected residual with the row's own contexts removed
(local self-removal). 'real' uses the observed contexts, 'null' the contexts shuffled within income parents.
Writes cache/stage3/embedding_income_all.npy (+ .json names: embed_{real,null}_income_{0..15}) and the per-fold banks
embedding_income_f<k>.npz (fit / query rows) in cache/stage3 (folds 0, 4) / cache/stage4 (folds 1-3).
Usage: python income_embedding.py"""
import time

import numpy as np

from contrast_data import (STAGE, IncomeParents, Stage1, fold_bank_dir, split_indices, within_parent_permutation,
                           write_names)


def compute():
    t = time.time()
    b = Stage1()
    p = IncomeParents(b)
    d = b.base
    key, parent, ng = p.key, p.parent, p.ng
    npg = int(parent.max()) + 1
    pv = np.zeros(ng, 'int32')
    pv[key] = parent
    contexts = dict(p.ctx)
    contexts['age'] = np.minimum(d.Age.to_numpy('int64') // 5, 20)
    contexts['commute'] = np.minimum((d.Daily_Commute_km.to_numpy('float64') // 5).astype('int64'), 30)
    for col in ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Number_of_Cars_Owned',
                'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work']:
        contexts[col] = d[col].to_numpy('int64')
    perm = within_parent_permutation(parent, len(d))
    nv = np.bincount(key, minlength=ng)
    pc = np.bincount(parent, minlength=npg)
    cols, null, pref, offset, total = [], [], [], [], 0
    for name, ctx in contexts.items():
        ns = int(ctx.max()) + 1
        offset.append((name, total, ns))
        total += ns
        cols.append(np.bincount(key * ns + ctx, minlength=ng * ns).reshape(ng, ns))
        null.append(np.bincount(key * ns + ctx[perm], minlength=ng * ns).reshape(ng, ns))
        par = np.bincount(parent * ns + ctx, minlength=npg * ns).reshape(npg, ns)
        pref.append((par + .5) / (pc[:, None] + .5 * ns))
    C, N, P = np.column_stack(cols), np.column_stack(null), np.column_stack(pref)
    scale = np.sqrt(P + .01)
    # each context family gets equal total energy before the factorization
    family = np.concatenate([np.full(ns, np.sqrt(ns)) for _, _, ns in offset])
    scale *= family[None, :]
    R = (C - nv[:, None] * P[pv]) / (nv[:, None] + 20) / scale[pv]
    weighted = R * np.sqrt(np.minimum(nv, 500))[:, None]
    _, singular, V = np.linalg.svd(weighted, full_matrices=False)
    basis = V[:16].T
    output, names = [], []
    parproj = (P[:, :, None] * basis[None, :, :] / scale[:, :, None]).sum(axis=1)
    for mode, counts, ccs in [('real', C, contexts), ('null', N, {n: c[perm] for n, c in contexts.items()})]:
        emb = ((counts - nv[:, None] * P[pv]) / scale[pv]) @ basis
        selfproj = np.zeros((len(d), 16))
        for name, start, ns in offset:
            j = ccs[name] + start
            selfproj += basis[j] / scale[parent, j][:, None]
        values = (emb[key] - selfproj + parproj[parent]) / (nv[key, None] - 1 + 20)
        output.append(values.astype('float32'))
        names += [f'embed_{mode}_income_{j}' for j in range(16)]
    A = np.column_stack(output)
    assert np.isfinite(A).all()
    STAGE[3].mkdir(parents=True, exist_ok=True)
    np.save(STAGE[3] / 'embedding_income_all.npy', A)
    np.savez(STAGE[3] / 'embedding_income_state.npz', basis=basis, scale=scale, parent_profile=P, singular=singular, key=key,
             parent=parent)
    write_names(STAGE[3] / 'embedding_income_all.npy', names, contexts=[x[0] for x in offset])
    print(f'income embedding {A.shape} ({time.time() - t:.0f}s)', flush=True)
    return A, names


def main():
    path = STAGE[3] / 'embedding_income_all.npy'
    if path.exists():
        import json
        A, names = np.load(path), json.loads(path.with_suffix('.json').read_text())['names']
    else:
        A, names = compute()
    for k in range(5):
        fit, valid, query = split_indices(k)
        out = fold_bank_dir(k) / f'embedding_income_f{k}.npz'
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, fit=fit, query=query, train_features=A[fit], query_features=A[query])
        write_names(out, names)


if __name__ == '__main__':
    main()
