#!/usr/bin/env python
"""Stage-2 artifacts: fold-local value mixtures, context-shape lack-of-fit and context-conditional mixtures.

From the stage-2 teacher of each label pool (cache/stage2/teachers): the exact-value class mixture by EM (as in
semisupervised.py, alpha 1 and 20, real and within-parent-permuted ratios); then for three context pairs
(subsidy x anxiety, subsidy x concern, anxiety x concern) the observed context frequencies of each value among the rows
outside the pool are compared with the frequencies implied by its class mixture (class-conditional parent profiles
with 100 global pseudo-counts, 20 mixture-expected pseudo-counts, own record removed); the residual, its part
orthogonal to the class-mix direction and its norm are standardized and clipped to +-4 (shape_real_* = the 9 columns
the XGB models use). Also context-stratified mixtures (stratum_*).
Per outer fold k: inner-split pools for the outer-train rows, the outer pool for fold k + test.
Writes cache/stage2/artifacts_{field}_f{k}.npz (fit, query, train_features, query_features) + .json (feature_names).
Usage: python artifact_features.py [--fields income] [--folds 0,1,2,3,4]"""
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
from features import Builder, split_indices  # noqa: E402
from layout import STAGE2 as CACHE  # noqa: E402
from teachers import field_state  # noqa: E402


def parent_prior(y, fit, parent, strength=20.):
    size = int(parent.max())+1
    count = np.bincount(parent[fit], minlength=size)
    pos = np.bincount(parent[fit], weights=y[fit], minlength=size)
    return ((pos+strength*y[fit].mean())/(count+strength)).clip(1e-4, 1-1e-4)


def calibrated_llr(raw, parent_u, prior):
    count = np.bincount(parent_u, minlength=len(prior))
    adjust = np.zeros(len(prior))
    for _ in range(12):
        q = expit(raw+adjust[parent_u])
        s = np.bincount(parent_u, weights=q, minlength=len(prior))
        h = np.bincount(parent_u, weights=q*(1-q), minlength=len(prior))
        adjust += np.clip((prior*count-s)/np.maximum(h, 1e-3), -1, 1)
    return raw+adjust[parent_u]-logit(prior[parent_u])


def conditional_permutation(parent_u):
    result = np.arange(len(parent_u))
    rng = np.random.default_rng(421337)
    for ix in pd.Series(result).groupby(parent_u).indices.values():
        result[ix] = rng.permutation(ix)
    return result


def estimate(count, pos, prior, ukey, llr, alpha, weight=1.):
    start = (pos+alpha*prior)/(count+alpha)
    uc = np.bincount(ukey, minlength=len(count))
    p = start.copy()
    for _ in range(100 if weight else 0):
        r = expit(logit(p.clip(1e-5, 1-1e-5))[ukey]+llr)
        nxt = (pos+alpha*prior+weight*np.bincount(ukey, weights=r, minlength=len(count)))/(count+alpha+weight*uc)
        change = np.max(abs(nxt-p))
        p = nxt
        if change < 1e-7:
            break
    return start, p


def context_groups(b):
    d = b.base
    sub = d.Subsidy_Available.to_numpy(dtype='int64')
    anx = d.Range_Anxiety_Level.to_numpy(dtype='int64')
    raw_concern = d.Environmental_Concern_Level.to_numpy(dtype='int64')
    concern = pd.factorize(raw_concern, sort=True)[0]
    singles = dict(subsidy=sub, anxiety=anx, concern=concern)
    pairs = dict(subsidy_anxiety=sub*3+anx,
                 subsidy_concern=sub*int(concern.max()+1)+concern,
                 anxiety_concern=anx*int(concern.max()+1)+concern)
    return singles, pairs


def shape_features(b, fit, query, unlab, key, parent, parent_for_key, mixture, permutation, context_set='core'):
    """Context frequency residuals and their part orthogonal to the class-mix direction.

    expected(v, s) = pi_v f1(parent, s) + (1 - pi_v) f0(parent, s); the residual's weighted projection onto f1 - f0 is
    removed for the orthogonal part. Counts exclude each query row's own (real or permuted) context record."""
    yp = b.y[fit].astype('int64')
    ng, npg = len(parent_for_key), int(parent.max())+1
    g, u, v = key[fit], key[unlab], key[query]
    uq = np.searchsorted(unlab, query)
    assert np.array_equal(unlab[uq], query)
    nu = np.bincount(u, minlength=ng)
    singles, pairs = context_groups(b)
    if context_set == 'noncore':
        d = b.base
        gender = d.Gender.to_numpy(dtype='int64')
        city = d.City_Type.to_numpy(dtype='int64')
        car = d.Current_Car_Type.to_numpy(dtype='int64')
        home = d.Home_Charging_Possible.to_numpy(dtype='int64')
        anxiety = d.Range_Anxiety_Level.to_numpy(dtype='int64')
        nc = int(city.max())+1
        na = int(anxiety.max())+1
        pairs = dict(gender_city=gender*nc+city, car_city=car*nc+city, home_anxiety=home*na+anxiety)
    else:
        assert context_set == 'core'
    out, names = [], []
    for ctx_name, ctx in pairs.items():
        ns = int(ctx.max())+1
        global_counts = np.bincount(yp*ns+ctx[fit], minlength=2*ns).reshape(2, ns)+.5
        global_dist = global_counts/global_counts.sum(axis=1, keepdims=True)
        local = np.bincount((parent[fit]*2+yp)*ns+ctx[fit], minlength=npg*2*ns).reshape(npg, 2, ns)
        f = (local+100*global_dist[None, :, :])/(local.sum(axis=2, keepdims=True)+100)
        f0, f1 = f[parent_for_key, 0], f[parent_for_key, 1]
        expected = (1-mixture[:, None])*f0+mixture[:, None]*f1
        direction = f1-f0
        metric = 1/(expected+.01)
        denom = np.maximum((direction*direction*metric).sum(axis=1), 1e-6)
        for mode, order in [('real', np.arange(len(unlab))), ('null', permutation)]:
            cu = ctx[unlab][order]
            counts = np.bincount(u*ns+cu, minlength=ng*ns).reshape(ng, ns)
            residual = counts[v].astype('float64')
            residual[np.arange(len(query)), cu[uq]] -= 1
            residual -= (nu[v, None]-1)*expected[v]
            residual /= (nu[v, None]-1+20)
            coefficient = (residual*direction[v]*metric[v]).sum(axis=1)/denom[v]
            orth = residual-coefficient[:, None]*direction[v]
            scale = np.sqrt(expected[v]*(1-expected[v])+.01)
            rawz = np.clip(residual/scale, -4, 4)
            orthz = np.clip(orth/scale, -4, 4)
            rowctx = ctx[query]
            for suffix, values in [('raw_at', rawz[np.arange(len(query)), rowctx]),
                                   ('orth_at', orthz[np.arange(len(query)), rowctx]),
                                   ('orth_norm', np.sqrt((orthz*orthz).mean(axis=1)))]:
                names.append(f'shape_{mode}_{ctx_name}_{suffix}')
                out.append(values)
            for j in range(ns):
                names.append(f'shapevec_{mode}_{ctx_name}_{j}')
                out.append(orthz[:, j])
    return np.column_stack(out).astype('float32'), names


def stratum_features(b, fit, query, unlab, raw, key, parent, parent_for_key, mixture, permutation, pi_parent):
    """Context-specific value mixtures anchored on the same global mixture; labeled-only subgroup corrections kept
    as separate columns; the control permutes (context, ratio) jointly within parent."""
    ng, npg = len(parent_for_key), int(parent.max())+1
    y = b.y[fit].astype('float64')
    singles, _ = context_groups(b)
    out, names = [], []
    for context, ctx in singles.items():
        ns = int(ctx.max())+1
        parent_ctx = parent*ns+ctx
        cp = np.bincount(parent_ctx[fit], minlength=npg*ns)
        sp = np.bincount(parent_ctx[fit], weights=y, minlength=npg*ns)
        cat_count = np.bincount(ctx[fit], minlength=ns)
        cat_pos = np.bincount(ctx[fit], weights=y, minlength=ns)
        cat_prior = (cat_pos+20*y.mean())/(cat_count+20)
        pi_pc = ((sp+20*np.tile(cat_prior, npg))/(cp+20)).clip(1e-4, 1-1e-4)
        lr = calibrated_llr(raw, parent_ctx[unlab], pi_pc)
        pc_for_vc = (parent_for_key[:, None]*ns+np.arange(ns)).ravel()
        vp = np.repeat(mixture, ns)
        base = expit(logit(vp.clip(1e-5, 1-1e-5))+logit(pi_pc[pc_for_vc])-np.repeat(logit(pi_parent[parent_for_key]), ns))
        vc_fit, vc_query = key[fit]*ns+ctx[fit], key[query]*ns+ctx[query]
        count = np.bincount(vc_fit, minlength=ng*ns)
        pos = np.bincount(vc_fit, weights=y, minlength=ng*ns)
        for alpha in (20., 100.):
            zbase = logit(base.clip(1e-5, 1-1e-5))[vc_query].astype('float32')
            for mode, order in [('real', np.arange(len(unlab))), ('null', permutation)]:
                uk = key[unlab]*ns+ctx[unlab][order]
                start, p = estimate(count, pos, base, uk, lr[order], alpha)
                zs = logit(start.clip(1e-5, 1-1e-5))[vc_query].astype('float32')
                z = logit(p.clip(1e-5, 1-1e-5))[vc_query].astype('float32')
                if mode == 'real':
                    names.append(f'stratum_{context}_label_a{alpha:g}')
                    out.append(zs-zbase)
                names.append(f'stratum_{context}_{mode}_a{alpha:g}')
                out.append(z-zs)
    return np.column_stack(out).astype('float32'), names


def map_features(b, fit, query, field, teacher, contexts=None, context_set='core', include_strata=True):
    assert not np.intersect1d(fit, query).size
    assert np.array_equal(fit, teacher['fit'])
    unlab, raw = teacher['unlab'], teacher['raw_logit']
    key, parent, pv, _ = field_state(b, field)
    ng = len(pv)
    pi = parent_prior(b.y, fit, parent)
    lr = calibrated_llr(raw, parent[unlab], pi)
    perm = conditional_permutation(parent[unlab])
    count = np.bincount(key[fit], minlength=ng)
    pos = np.bincount(key[fit], weights=b.y[fit], minlength=ng)
    out, names = [], []
    mixture20 = None
    for alpha in (1., 20.):
        start, real = estimate(count, pos, pi[pv], key[unlab], lr, alpha)
        _, null = estimate(count, pos, pi[pv], key[unlab], lr[perm], alpha)
        zs = logit(start.clip(1e-5, 1-1e-5))[key[query]].astype('float32')
        zr = logit(real.clip(1e-5, 1-1e-5))[key[query]].astype('float32')
        zn = logit(null.clip(1e-5, 1-1e-5))[key[query]].astype('float32')
        out.extend([zs, zr-zs, zn-zs])
        names.extend([f'mix_prior_a{alpha:g}', f'mix_real_delta_a{alpha:g}', f'mix_null_delta_a{alpha:g}'])
        if alpha == 20:
            mixture20 = real
    if contexts is None:
        contexts = field == 'income'
    if contexts:
        a, fn = shape_features(b, fit, query, unlab, key, parent, pv, mixture20, perm, context_set)
        out.extend([a[:, j] for j in range(a.shape[1])])
        names.extend(fn)
        if include_strata:
            a, fn = stratum_features(b, fit, query, unlab, raw, key, parent, pv, mixture20, perm, pi)
            out.extend([a[:, j] for j in range(a.shape[1])])
            names.extend(fn)
    result = np.column_stack(out).astype('float32')
    assert np.isfinite(result).all() and len(names) == result.shape[1]
    return result, names


def build_one(b, field, k):
    p = CACHE/f'artifacts_{field}_f{k}.npz'
    if p.exists():
        return
    fit, valid, query = split_indices(k)
    train = None
    for j, (a, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(fit, b.y[fit])):
        teacher = np.load(CACHE/'teachers'/f'{field}_f{k}_inner{j}.npz')
        z, names = map_features(b, fit[a], fit[v], field, teacher)
        if train is None:
            train = np.empty((len(fit), len(names)), dtype='float32')
        train[v] = z
        print('artifacts inner', field, k, j, z.shape, flush=True)
    teacher = np.load(CACHE/'teachers'/f'{field}_f{k}_outer.npz')
    q, names = map_features(b, fit, query, field, teacher)
    np.savez(p, fit=fit, query=query, train_features=train, query_features=q)
    p.with_suffix('.json').write_text(json.dumps(dict(field=field, fold=k, feature_names=names), indent=2))
    print('artifacts', field, k, 'done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fields', default='income')
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    b = Builder()
    for field in a.fields.split(','):
        for k in fold_list(a.folds):
            build_one(b, field, k)


if __name__ == '__main__':
    main()
