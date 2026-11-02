"""Observed information and fractional unlabeled evidence for the exact-income mixtures (em_info).

The mixture likelihood is concave in the income rate; a safeguarded Newton solve of the same fixed-ratio mixture model
as the EM gives the rate for unlabeled-evidence weights 0.25, 1 and 4 (alpha 1 and 20), plus the effective number of
unlabeled observations, its fraction and the weight sensitivity.
Writes cache/em_info/information_features/fold_{k}/{train,query}.npy + manifest.json (names)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from lib import INFORMATION, TEACHERS, dump, logit
from mixtures import calibrated_llr, permute_within


def solve(c, fit, u, y, llr, parent_prior, alpha, weight, card):
    n = np.bincount(c[fit], minlength=card).astype(np.float64)
    s = np.bincount(c[fit], weights=y[fit], minlength=card)
    positive = s+alpha*parent_prior
    negative = n-s+alpha*(1-parent_prior)
    pi = positive/(positive+negative)
    lo = np.full(card, 1e-9)
    hi = np.full(card, 1-1e-9)
    g = c[u]
    for iteration in range(40):
        prob = np.clip(pi[g], 1e-9, 1-1e-9)
        r = expit(np.log(prob)-np.log1p(-prob)+llr)
        t = (r-prob)/(prob*(1-prob))
        score = positive/pi-negative/(1-pi)+weight*np.bincount(g, weights=t, minlength=card)
        info = positive/pi**2+negative/(1-pi)**2+weight*np.bincount(g, weights=t*t, minlength=card)
        lo = np.where(score > 0, pi, lo)
        hi = np.where(score < 0, pi, hi)
        proposed = pi+score/info
        proposed = np.where((proposed >= lo) & (proposed <= hi), proposed, (lo+hi)*.5)
        change = float(np.max(np.abs(proposed-pi)))
        pi = np.clip(proposed, 1e-9, 1-1e-9)
        if change < 1e-8:
            break
    prob = pi[g]
    r = expit(np.log(prob)-np.log1p(-prob)+llr)
    t = (r-prob)/(prob*(1-prob))
    uinfo = np.bincount(g, weights=t*t, minlength=card)
    n_eff = pi*(1-pi)*uinfo
    return pi, n_eff


def pool_features(d, fit, apply, p, shuffle=False):
    u = np.setdiff1d(np.arange(d.N), fit)
    parent = d.keys['inc_parent']
    c = d.keys['Annual_Income_USD']
    card = d.card['Annual_Income_USD']
    llr, pp = calibrated_llr(p, parent, fit, u, d.y)
    if shuffle:
        llr = permute_within(llr, parent[u], 73491+len(fit))
    parent_by_value = np.zeros(card, np.int32)
    parent_by_value[c] = parent
    prior = pp[parent_by_value]
    n = np.bincount(c[fit], minlength=card)
    nu = np.bincount(c[u], minlength=card)
    s = np.bincount(c[fit], weights=d.y[fit], minlength=card)
    cols, names = [], []
    for alpha in [1., 20.]:
        p0 = (s+alpha*prior)/(n+alpha)
        solutions = {}
        for weight in [.25, 1., 4.]:
            pi, n_eff = solve(c, fit, u, d.y, llr, prior, alpha, weight, card)
            solutions[weight] = (pi, n_eff)
            cols.append((logit(pi)-logit(p0))[c[apply]])
            names.append(f'info_delta_a{alpha:g}_w{weight:g}')
        pi, n_eff = solutions[1.]
        cols.extend([np.log1p(n_eff)[c[apply]], (n_eff/(n+alpha+n_eff))[c[apply]],
                     (n_eff/np.maximum(nu, 1))[c[apply]],
                     (logit(solutions[4.][0])-logit(solutions[.25][0]))[c[apply]]])
        names += [f'info_logeffective_a{alpha:g}', f'info_fraction_a{alpha:g}', f'info_per_unlabeled_a{alpha:g}',
                  f'info_weight_sensitivity_a{alpha:g}']
    return np.column_stack(cols).astype(np.float32), names


def build(d, fold, log=print):
    a = np.flatnonzero(d.folds != fold)
    b = np.flatnonzero(d.folds == fold)
    q = np.r_[b, d.test]
    local = np.full(d.n, -1, np.int64)
    local[a] = np.arange(len(a))
    T = Q = None
    for pool in range(6):
        root = TEACHERS/'income'/f'fold_{fold}'/f'pool_{pool}'
        fit = np.load(root/'fit.npy')
        p = np.load(root/'p.npy')
        apply = q if pool == 0 else np.setdiff1d(a, fit)
        V, names = pool_features(d, fit, apply, p)
        if pool == 0:
            Q = V
            T = np.empty((len(a), V.shape[1]), np.float32)
        else:
            T[local[apply]] = V
        log(f'information fold={fold} pool={pool} shape={V.shape}')
    folder = INFORMATION/f'fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    assert np.isfinite(T).all() and np.isfinite(Q).all()
    np.save(folder/'train.npy', T)
    np.save(folder/'query.npy', Q)
    dump(folder/'manifest.json', dict(fold=fold, names=names))
