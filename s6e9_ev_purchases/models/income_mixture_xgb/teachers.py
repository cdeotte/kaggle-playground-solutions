#!/usr/bin/env python
"""Stage-2 field-excluding context teachers (income; commute and age are also supported).

For a label pool `fit`, an XGB teacher is trained on the stage-1 features without the exact field (income teacher:
the same columns as the stage-1 EM teacher: 1000-bin income center, source-law score at that center, leading BPE piece,
parent key) and predicts every row outside the pool. Per outer fold k: one teacher per inner split of the outer-train
rows and one on all outer-train rows. Writes cache/stage2/teachers/{field}_f{k}_inner{j}.npz and {field}_f{k}_outer.npz
(fit, unlab, raw_logit).
Usage: python teachers.py [--fields income] [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.pairs import fold_list  # noqa: E402
from features import Builder, split_indices  # noqa: E402
from layout import STAGE1, STAGE2 as CACHE  # noqa: E402

FIELDS = {'income': 'Annual_Income_USD', 'commute': 'Daily_Commute_km', 'age': 'Age'}
PARAMS = dict(objective='binary:logistic', tree_method='hist', device='cuda', eta=.05, max_depth=4, min_child_weight=30,
              reg_lambda=10, subsample=.85, colsample_bytree=.9, max_bin=512, num_parallel_tree=2, seed=42)


def field_state(b, field):
    d = b.base
    key = b.keys[FIELDS[field]].to_numpy()
    if field == 'income':
        coarse = np.floor(d.Annual_Income_USD.to_numpy(dtype='float64')/1000)*1000+500
        parts = [b.keys.inc1000.to_numpy(), b.keys.inc_bpe0.to_numpy()]
    elif field == 'commute':
        # parent = (commute // 10, leading BPE piece); the exact tenth is excluded
        raw = pd.read_parquet(STAGE1/'predictors.parquet', columns=[FIELDS[field]])
        coarse = np.floor(raw[FIELDS[field]].to_numpy()/10)*10+5
        parts = [np.floor(raw[FIELDS[field]].to_numpy()/10), b.keys.com_bpe0.to_numpy()]
    else:
        coarse = np.floor(d.Age.to_numpy(dtype='float64')/10)*10+5
        parts = [np.floor(d.Age.to_numpy(dtype='float64')/10)]
    parent = pd.factorize(pd.MultiIndex.from_arrays(parts), sort=True)[0].astype('int32')
    sz = b.meta['key_sizes'][FIELDS[field]]
    parent_for_key = np.zeros(sz, dtype='int32')
    parent_for_key[key] = parent
    assert np.array_equal(parent_for_key[key], parent)
    return key, parent, parent_for_key, coarse


def keep_feature(name, field):
    low = name.lower()
    if field == 'income':
        return not ('income' in low or 'inc' in low or low.startswith('source_gap')
                    or low in ['source_exact', 'source_score', 'source_probability', 'source_logit'])
    if field == 'commute':
        return not ('commute' in low or low.startswith('com_') or low.startswith('te_com') or low == 'anxiety_score')
    return 'age' not in low


def train_teacher(b, fit, field, tag, save=True):
    import xgboost as xgb
    out = CACHE/'teachers'/tag
    if save and out.with_suffix('.npz').exists():
        d = np.load(out.with_suffix('.npz'))
        assert np.array_equal(d['fit'], fit)
        return d, json.loads(out.with_suffix('.json').read_text())
    unlab = np.setdiff1d(np.arange(len(b.base)), fit)
    X, V, names, types = b.build(fit, unlab, noise=.04)
    kept = [j for j, n in enumerate(names) if keep_feature(n, field)]
    _, parent, _, coarse = field_state(b, field)
    extra, enames, etypes = [coarse, parent], [f'context_coarse_{field}', 'context_parent'], ['q', 'c']
    if field == 'income':
        inc = b.base.Annual_Income_USD.to_numpy()
        # same columns and float32 arithmetic as the stage-1 EM teacher
        coarse_old = np.floor(inc/1000)*1000+500
        rule_old = b.base.source_score.to_numpy()+1.2*(coarse_old-inc)/1e5
        extra = [coarse_old, rule_old, b.keys.inc_bpe0.to_numpy(), parent]
        enames = ['context_coarse_income', 'context_coarse_rule', 'context_bpe', 'context_parent']
        etypes = ['q', 'q', 'c', 'c']
    elif field == 'commute':
        extra.append(b.keys.com_bpe0.to_numpy())
        enames.append('context_commute_bpe')
        etypes.append('c')
    X = np.column_stack([X[:, kept]]+[a[fit] for a in extra])
    V = np.column_stack([V[:, kept]]+[a[unlab] for a in extra])
    fn = [names[j] for j in kept]+enames
    ft = [types[j] for j in kept]+etypes
    dt = xgb.DMatrix(X, label=b.y[fit], feature_names=fn, feature_types=ft, enable_categorical=True)
    du = xgb.DMatrix(V, feature_names=fn, feature_types=ft, enable_categorical=True)
    print('teacher', tag, X.shape, flush=True)
    model = xgb.train(PARAMS, dt, num_boost_round=600, verbose_eval=False)
    prob = model.predict(du).clip(1e-5, 1-1e-5).astype('float64')
    data = dict(fit=fit, unlab=unlab, raw_logit=logit(prob))
    rec = dict(tag=tag, field=field, feature_names=fn, feature_types=ft, rounds=600)
    if save:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix('.npz'), **data)
        out.with_suffix('.json').write_text(json.dumps(rec, indent=2))
    return data, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fields', default='income')
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    b = Builder()
    for field in a.fields.split(','):
        for k in fold_list(a.folds):
            fit, valid, query = split_indices(k)
            for j, (tr, _) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(fit, b.y[fit])):
                train_teacher(b, fit[tr], field, f'{field}_f{k}_inner{j}')
            train_teacher(b, fit, field, f'{field}_f{k}_outer')


if __name__ == '__main__':
    main()
