"""Compact view of the em_info base features: source-rule geometry and inner-held source-conditional intercepts."""
import numpy as np
from scipy.special import expit, ndtr

from lib import RAW, logit, splits


def source_rule(d):
    t = d.df
    score = (1.2*t.Annual_Income_USD.to_numpy(np.float64)/100000
             + .6*t.Environmental_Concern_Level.to_numpy(np.float64)
             + 2*(t.Subsidy_Available.to_numpy() == 'Yes')
             - (t.Range_Anxiety_Level.to_numpy() == 'Medium')
             - 3*(t.Range_Anxiety_Level.to_numpy() == 'High')-5.5)
    p = ndtr(score)
    return np.column_stack([score, p, logit(p)]).astype(np.float32)


def select(names):
    keys = set(RAW+['inc100', 'inc1000', 'bpe'])
    chosen = []
    for j, name in enumerate(names):
        if name.startswith('source__'):
            continue
        if name.startswith('te__'):
            key = name.split('__')[1]
            if key not in keys or (key == 'bpe' and name.endswith('__diff')):
                continue
        chosen.append(j)
    return chosen


def offsets(d, fit, apply):
    fixed = source_rule(d)[:, 2].astype(np.float64)
    columns, names = [], []
    for key in ['Annual_Income_USD', 'Daily_Commute_km', 'Age', 'inc100', 'inc1000', 'bpe']:
        c = d.keys[key]
        g = c[fit]
        card = d.card[key]
        for penalty in [1., 10.]:
            delta = np.zeros(card)
            for _ in range(7):
                p = expit(fixed[fit]+delta[g])
                score = np.bincount(g, weights=d.y[fit]-p, minlength=card)-penalty*delta
                info = np.bincount(g, weights=p*(1-p), minlength=card)+penalty
                delta += np.clip(score/info, -1., 1.)
            columns.append(delta[c[apply]])
            names.append(f'te__source_offset_{key}_r{penalty:g}')
    return np.column_stack(columns).astype(np.float32), names


def build(d, fold, include_offsets=True):
    a = np.flatnonzero(d.folds != fold)
    b = np.flatnonzero(d.folds == fold)
    q = np.r_[b, d.test]
    fixed = source_rule(d)
    names = ['rule_score', 'rule_probability', 'rule_logit']
    if not include_offsets:
        return fixed[a], fixed[q], names
    Q, extra = offsets(d, a, q)
    T = np.empty((len(a), len(extra)), np.float32)
    local = np.full(d.n, -1, np.int64)
    local[a] = np.arange(len(a))
    for fit, hold in splits(a, d.y):
        T[local[hold]] = offsets(d, fit, hold)[0]
    return np.column_stack([fixed[a], T]), np.column_stack([fixed[q], Q]), names+extra
