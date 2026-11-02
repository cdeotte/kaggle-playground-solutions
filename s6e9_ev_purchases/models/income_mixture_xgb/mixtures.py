"""Nested context mixtures (em_info).

Each call receives a label pool: teacher, counts and parent priors use the pool; the rows outside the pool are the
unlabeled evidence of the EM. The 'loo' variant is a first-order influence approximation of a leave-one-out EM refit."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from lib import FEATURES, TEACHERS, dump, logit


def calibrated_llr(p, parents, fit, unlabeled, y):
    card = int(parents.max())+1
    n = np.bincount(parents[fit], minlength=card)
    s = np.bincount(parents[fit], weights=y[fit], minlength=card)
    prior = (s+20*y[fit].mean())/(n+20)
    u = parents[unlabeled]
    count = np.bincount(u, minlength=card)
    z = logit(p[unlabeled])
    shift = np.zeros(card)
    # bounded bisection of the per-parent shift
    lo = np.full(card, -15.)
    hi = np.full(card, 15.)
    for _ in range(32):
        shift = (lo+hi)*.5
        mean = np.bincount(u, weights=expit(z+shift[u]), minlength=card)/np.maximum(count, 1)
        smaller = mean < prior
        lo = np.where(smaller, shift, lo)
        hi = np.where(smaller, hi, shift)
    llr = z+shift[u]-logit(prior[u])
    return llr, prior


def permute_within(llr, parents, seed):
    rng = np.random.default_rng(seed)
    ordered = np.argsort(parents, kind='stable')
    permuted = np.lexsort((rng.random(len(parents)), parents))
    out = np.empty_like(llr)
    out[ordered] = llr[permuted]
    return out


def estimate(d, fit, apply, p, key, parent_key, alpha, shuffle=False, influence=False, cache=None):
    c = d.keys[key]
    parent = d.keys[parent_key]
    card = d.card[key]
    cache = {} if cache is None else cache
    ck = (parent_key, shuffle)
    if ck not in cache:
        mask = np.ones(d.N, bool)
        mask[fit] = False
        unlabeled = np.flatnonzero(mask)
        llr, pp = calibrated_llr(p, parent, fit, unlabeled, d.y)
        if shuffle:
            llr = permute_within(llr, parent[unlabeled], 73491+len(fit))
        cache[ck] = (unlabeled, llr, pp)
    unlabeled, llr, pp = cache[ck]
    group_parent = np.zeros(card, np.int32)
    group_parent[c] = parent
    prior = pp[group_parent]
    n = np.bincount(c[fit], minlength=card)
    s = np.bincount(c[fit], weights=d.y[fit], minlength=card)
    u = c[unlabeled]
    nu = np.bincount(u, minlength=card)
    pi0 = (s+alpha*prior)/(n+alpha)
    pi = pi0.copy()
    den = n+alpha+nu
    for iteration in range(100):
        r = expit(logit(pi[u])+llr)
        updated = (s+alpha*prior+np.bincount(u, weights=r, minlength=card))/den
        change = float(np.max(np.abs(updated-pi)))
        pi = updated
        if change < 1e-7:
            break
    est = pi[c[apply]]
    if influence:
        r = expit(logit(pi[u])+llr)
        derivative = r*(1-r)/np.maximum(pi[u]*(1-pi[u]), 1e-9)
        h = den-np.bincount(u, weights=derivative, minlength=card)
        ix = np.searchsorted(unlabeled, apply)
        assert np.array_equal(unlabeled[ix], apply)
        step = (pi[u[ix]]-r[ix])/np.maximum(h[u[ix]]-1+derivative[ix], alpha*.05)
        est = np.clip(est+np.clip(step, -.2, .2), 1e-6, 1-1e-6)
    return np.column_stack([logit(pi0[c[apply]]), logit(est)-logit(pi0[c[apply]])]).astype(np.float32)


def offsets(d, fit, apply, p, key, penalty):
    c = d.keys[key]
    card = d.card[key]
    g = c[fit]
    z = logit(p[fit])
    delta = np.zeros(card)
    for _ in range(12):
        prob = expit(z+delta[g])
        score = np.bincount(g, weights=d.y[fit]-prob, minlength=card)-penalty*delta
        info = np.bincount(g, weights=prob*(1-prob), minlength=card)+penalty
        delta += np.clip(score/info, -1, 1)
    return delta[c[apply]].astype(np.float32)


def pool_features(d, fit, apply, p_income, p_commute):
    columns, names = [], []
    cache = {}
    specs = [('inc_global', 'Annual_Income_USD', 'inc_parent', p_income, False, False),
             ('inc_sub', 'inc_sub', 'inc_parent_sub', p_income, False, False),
             ('inc_anx', 'inc_anx', 'inc_parent_anx', p_income, False, False),
             ('inc_env', 'inc_env', 'inc_parent_env', p_income, False, False),
             ('shuffle', 'Annual_Income_USD', 'inc_parent', p_income, True, False),
             ('shuffle_sub', 'inc_sub', 'inc_parent_sub', p_income, True, False),
             ('shuffle_anx', 'inc_anx', 'inc_parent_anx', p_income, True, False),
             ('shuffle_env', 'inc_env', 'inc_parent_env', p_income, True, False),
             ('loo', 'Annual_Income_USD', 'inc_parent', p_income, False, True)]
    if p_commute is not None:
        specs.append(('commute', 'Daily_Commute_km', 'km_parent', p_commute, False, False))
    for tag, key, parent, p, shuf, infl in specs:
        for alpha in [1., 20.]:
            columns.append(estimate(d, fit, apply, p, key, parent, alpha, shuf, infl, cache))
            names.extend([f'{tag}__labeled_a{alpha:g}', f'{tag}__delta_a{alpha:g}'])
    for field, key, p in [('income', 'Annual_Income_USD', p_income), ('commute', 'Daily_Commute_km', p_commute)]:
        if p is None:
            continue
        for penalty in [.5, 2., 10.]:
            columns.append(offsets(d, fit, apply, p, key, penalty)[:, None])
            names.append(f'offset__{field}_r{penalty:g}')
        columns.append(logit(p[apply])[:, None])
        names.append(f'teacher__{field}')
    return np.column_stack(columns).astype(np.float32), names


def build(d, fold, log=print):
    """Pool 0 (all outer-train rows) -> features of fold k + test; pools 1-5 (inner splits) -> inner-held rows.
    Writes cache/em_info/features_gpu/fold_{k}/{train,query}.npy + manifest.json (names)."""
    a = np.flatnonzero(d.folds != fold)
    b = np.flatnonzero(d.folds == fold)
    q = np.r_[b, d.test]
    train = query = None
    local = np.full(d.n, -1, np.int64)
    local[a] = np.arange(len(a))
    for pool in range(6):
        root = TEACHERS/'income'/f'fold_{fold}'/f'pool_{pool}'
        fit = np.load(root/'fit.npy')
        pi = np.load(root/'p.npy')
        other = TEACHERS/'commute'/f'fold_{fold}'/f'pool_{pool}'
        pc = None
        if (other/'p.npy').exists():
            pc = np.load(other/'p.npy')
            assert np.array_equal(np.load(other/'fit.npy'), fit)
        apply = q if pool == 0 else np.setdiff1d(a, fit)
        F, names = pool_features(d, fit, apply, pi, pc)
        if pool == 0:
            query = F
            train = np.empty((len(a), F.shape[1]), np.float32)
        else:
            train[local[apply]] = F
        log(f'EM features fold={fold} pool={pool} shape={F.shape}')
    folder = FEATURES/f'fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    assert np.isfinite(train).all() and np.isfinite(query).all()
    np.save(folder/'train.npy', train)
    np.save(folder/'query.npy', query)
    dump(folder/'manifest.json', {'names': names, 'fold': fold})
