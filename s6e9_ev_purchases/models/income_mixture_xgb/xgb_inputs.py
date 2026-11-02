"""Fold inputs of the XGB models (also used by the ../distillation students).

base(k): the 141 columns of cache/stage1/matched_f{k} + 4 exact-income class-mixture columns from cache/stage1/semi_f{k}
(EM prior and EM delta at alpha 1 and 20; feature weight 0.5); base margin = the nested LR margin.
Blocks (feature weight 0.25) add banks of this folder (shape, digits) and of ../neural_contrasts (stage 3-6);
transform income_diffs adds 4 contrasts of income target encodings at different resolutions (weight 0.5).
Every fold matrix: fit = outer-train rows; query = fold-k rows, then all test rows."""
import json
from functools import lru_cache

import numpy as np

from features import split_indices
from layout import NEURAL, STAGE1, STAGE2, STAGE8

MODELS = {
    'mixture': dict(blocks=[]),
    'shape': dict(blocks=[('shape', .25)]),
    'digits_diffs': dict(blocks=[('shape', .25), ('embedding', .25), ('generative', .25), ('coarse', .25), ('lift', .25),
                                 ('profile', .25), ('masked', .25), ('digits', .25)],
                         transforms=['income_diffs']),
}
STAGE6_BANKS = {
    'lift': ('dist_p100bpe', [f'loo_lift_{c}' for c in [
        'Age', 'Daily_Commute_km', 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
        'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Subsidy_Available',
        'Range_Anxiety_Level', 'Environmental_Concern_Level', 'sac']]),
    'profile': ('dist_p5000bpe', [f'profile_{j}' for j in range(16)]),
    'masked': ('masked_density', [f'conditional_lift_{c}' for c in [
        'Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
        'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Subsidy_Available',
        'Range_Anxiety_Level', 'Environmental_Concern_Level']]
        + [f'source_expectation{s}_{c}' for c in ['Annual_Income_USD', 'Subsidy_Available', 'Range_Anxiety_Level',
                                                  'Environmental_Concern_Level'] for s in ('', '_shift')]),
}


@lru_cache(maxsize=1)
def base(k):
    fit, v, q = split_indices(k)
    p = STAGE1/f'matched_f{k}'
    m = json.loads((p/'manifest.json').read_text())
    assert np.array_equal(np.load(p/'fit_index.npy'), fit) and np.array_equal(np.load(p/'query_index.npy'), q)
    d = np.load(STAGE1/f'semi_f{k}.npz')
    assert np.array_equal(d['fit'], fit) and np.array_equal(d['query'], q)

    def mix(a):
        b = a[:, [0, 4, 2, 6]].copy()
        b[:, 2:] -= b[:, :2]
        return b
    X = np.column_stack([np.load(p/'train.npy'), mix(d['train_features'])])
    Q = np.column_stack([np.load(p/'query.npy'), mix(d['query_features'])])
    names = m['feature_names']+['mixture_prior_1', 'mixture_prior_20', 'mixture_delta_1', 'mixture_delta_20']
    types = m['feature_types']+['q']*4
    weights = np.r_[np.ones(len(names)-4), np.full(4, .5)].astype('float32')
    mf, mq = np.load(p/'train_margin.npy'), np.load(p/'query_margin.npy')
    assert mf.shape == (len(fit),) and mq.shape == (len(q),)
    return X, Q, names, types, weights, fit, q, len(v), mf, mq


def block(k, name, fit, query):
    if name in ('shape', 'embedding', 'generative'):
        if name == 'shape':
            p = STAGE2/f'artifacts_income_f{k}.npz'
            names = json.loads(p.with_suffix('.json').read_text())['feature_names']
            ids = [j for j, n in enumerate(names) if n.startswith('shape_real_')]
        else:
            bank = 'embedding_income' if name == 'embedding' else 'generative'
            p = NEURAL/'stage3'/f'{bank}_f{k}.npz'
            if not p.exists():
                p = NEURAL/'stage4'/f'{bank}_f{k}.npz'
            m = json.loads(p.with_suffix('.json').read_text())
            names = m.get('names', m.get('feature_names'))
            want = ([f'embed_real_income_{j}' for j in range(8)] if name == 'embedding'
                    else [f'generator_context_delta_{v}' for v in ('llr', 'h0', 'h1', 'hdiff')])
            ids = [names.index(n) for n in want]
        d = np.load(p)
        assert np.array_equal(d['fit'], fit) and np.array_equal(d['query'], query)
        return d['train_features'][:, ids], d['query_features'][:, ids], [name+'__'+names[j] for j in ids]
    if name == 'digits':
        p = STAGE8/'digits_all.npy'
        names = json.loads(p.with_suffix('.json').read_text())['names']
        ids = list(range(len(names)))
    elif name == 'coarse':
        p = NEURAL/'stage5'/'coarse_entropy_all.npy'
        names = json.loads(p.with_suffix('.json').read_text())['names']
        ids = list(range(len(names)))
    elif name in STAGE6_BANKS:
        bank, columns = STAGE6_BANKS[name]
        p = NEURAL/'stage6'/f'{bank}_all.npy'
        names = json.loads(p.with_suffix('.json').read_text())['names']
        ids = [names.index(c) for c in columns]
    else:
        raise ValueError(name)
    a = np.load(p, mmap_mode='r')
    return a[fit][:, ids], a[query][:, ids], [name+'__'+names[j] for j in ids]


def inputs(k, model):
    cfg = MODELS[model]
    X, Q, names, types, fw, fit, query, nv, mf, mq = base(k)
    names, types = list(names), list(types)
    for name, weight in cfg['blocks']:
        a, b, nm = block(k, name, fit, query)
        X, Q = np.column_stack([X, a]), np.column_stack([Q, b])
        names += nm
        types += ['q']*len(nm)
        fw = np.r_[fw, np.full(len(nm), weight)].astype('float32')
    for transform in cfg.get('transforms', []):
        assert transform == 'income_diffs'
        pairs = [('te_Annual_Income_USD_1', 'te_inc10_20'), ('te_Annual_Income_USD_1', 'te_inc100_20'),
                 ('te_Annual_Income_USD_1', 'te_inc1000_20'), ('te_inc100_20', 'te_inc1000_20')]

        def derived(A):
            return np.column_stack([A[:, names.index(a)]-A[:, names.index(b)] for a, b in pairs])
        a, b = derived(X), derived(Q)
        nm = [transform+'__'+str(j) for j in range(a.shape[1])]
        X, Q = np.column_stack([X, a]), np.column_stack([Q, b])
        names += nm
        types += ['q']*len(nm)
        fw = np.r_[fw, np.full(len(nm), .5)].astype('float32')
    assert len(set(names)) == len(names) == X.shape[1]
    assert np.isfinite(X).all() and np.isfinite(Q).all()
    return X, Q, names, types, fw, fit, query, nv, mf, mq
