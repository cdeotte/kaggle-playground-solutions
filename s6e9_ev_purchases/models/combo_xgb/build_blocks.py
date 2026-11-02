#!/usr/bin/env python
"""Extra feature blocks for train.py (and ../te_bank_lr).

  digits       label-free income digit / modulus / rounding features and commute integer / decimal parts over train + test
               -> features/digits.npy (train rows then test rows) + features/digits_names.json
  digit_te     digit-bin response TEs: centered-logit TE of the exact income (alpha 1) and of income // 10, // 100, // 1000
               (alpha 20), plus 4 contrasts. Outer-train rows: 5-fold inner cross-fit (StratifiedKFold seed 4200 + k);
               valid / test rows: all outer-train rows.                       -> features/fold_blocks/digit_te/fold_k/
  income_em    the 4 exact-income EM columns (labelled log-odds and EM delta, alpha 1 / 20) of the income-mixture caches
  income_info  the 14 information-aware EM columns of the income-mixture caches
               (both read from ../income_mixture_xgb/cache/em_info: fold-local, train = outer-train rows, query = valid
               rows then test rows)                                            -> features/fold_blocks/<name>/fold_k/
Each fold block: {train,valid,test}.npy (outer-train rows, fold-k rows, test rows) + meta.json (column names).
Usage: python build_blocks.py digits | digit_te | income_em | income_info | all"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, read_test, read_train, target  # noqa: E402

FEATURES = HERE / 'features'
EM_INFO = HERE.parent / 'income_mixture_xgb' / 'cache' / 'em_info'
EM_BLOCKS = {'income_em': ('features_gpu', lambda n: n.startswith('inc_global__')),
             'income_info': ('information_features', lambda n: True)}


def save_block(name, k, Xtr, Xva, Xte, names):
    d = FEATURES / 'fold_blocks' / name / f'fold_{k}'
    d.mkdir(parents=True, exist_ok=True)
    for r, X in (('train', Xtr), ('valid', Xva), ('test', Xte)):
        X = np.asarray(X, np.float32)
        assert np.isfinite(X).all(), (name, r)
        np.save(d / f'{r}.npy', X)
    (d / 'meta.json').write_text(json.dumps(dict(fold=k, names=names)))
    print(name, k, np.shape(Xtr), flush=True)


def te(codes, y, fit, apply, alpha):
    prior = y[fit].mean()
    size = int(codes.max()) + 1
    cnt = np.bincount(codes[fit], minlength=size).astype(np.float64)
    pos = np.bincount(codes[fit], weights=y[fit], minlength=size).astype(np.float64)
    rate = (pos[codes[apply]] + alpha * prior) / (cnt[codes[apply]] + alpha)
    return logit(np.clip(rate, 1e-6, 1 - 1e-6)) - logit(prior)


def build_digits():
    cols = ['Annual_Income_USD', 'Daily_Commute_km']
    df = pd.concat([read_train()[cols], read_test()[cols]], ignore_index=True)
    inc = np.rint(df['Annual_Income_USD'].to_numpy()).astype(np.int64)
    com = df['Daily_Commute_km'].to_numpy()
    tz = np.zeros(len(inc))
    for p in (10, 100, 1000, 10000):
        tz += (inc % p == 0)
    feats = {
        'inc_mod10': inc % 10, 'inc_mod100': inc % 100, 'inc_mod1000': inc % 1000, 'inc_mod10000': inc % 10000,
        'inc_d_units': inc % 10, 'inc_d_tens': (inc // 10) % 10, 'inc_d_hundreds': (inc // 100) % 10, 'inc_d_thousands': (inc // 1000) % 10,
        'inc_trailing_zeros': tz,
        'inc_dist1000': np.minimum(inc % 1000, 1000 - inc % 1000), 'inc_dist5000': np.minimum(inc % 5000, 5000 - inc % 5000),
        'inc_dist10000': np.minimum(inc % 10000, 10000 - inc % 10000),
        'com_int': np.floor(com), 'com_dec': np.rint((com - np.floor(com)) * 10),
    }
    X = np.column_stack([np.asarray(v, np.float32) for v in feats.values()])
    FEATURES.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES / 'digits.npy', X)
    (FEATURES / 'digits_names.json').write_text(json.dumps(list(feats)))
    print('digits', X.shape)


def build_digit_te(y, f):
    inc = np.concatenate([read_train()['Annual_Income_USD'].to_numpy(), read_test()['Annual_Income_USD'].to_numpy()])
    inc = np.rint(inc).astype(np.int64)
    keys = {'exact': (inc, 1.0), 'd10': (inc // 10, 20.0), 'd100': (inc // 100, 20.0), 'd1000': (inc // 1000, 20.0)}
    codes = {n: np.unique(v, return_inverse=True)[1] for n, (v, _) in keys.items()}
    yy = np.concatenate([y, np.zeros(len(inc) - len(y))])  # test rows are never fit rows
    te_idx = np.arange(len(y), len(inc))
    names = ['te_exact_a1', 'te_d10_a20', 'te_d100_a20', 'te_d1000_a20', 'c_exact_d10', 'c_exact_d100', 'c_exact_d1000', 'c_d100_d1000']
    for k in range(5):
        tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        cols_tr, cols_va, cols_te = [], [], []
        for n, (_, alpha) in keys.items():
            c = codes[n]
            v = np.empty(len(tr))
            for fi, ai in StratifiedKFold(5, shuffle=True, random_state=4200 + k).split(np.zeros(len(tr)), y[tr]):
                v[ai] = te(c, yy, tr[fi], tr[ai], alpha)
            cols_tr.append(v)
            cols_va.append(te(c, yy, tr, va, alpha))
            cols_te.append(te(c, yy, tr, te_idx, alpha))

        def contrasts(X):
            return np.column_stack([X, X[:, 0] - X[:, 1], X[:, 0] - X[:, 2], X[:, 0] - X[:, 3], X[:, 2] - X[:, 3]])
        Xtr, Xva, Xte = (contrasts(np.column_stack(c)) for c in (cols_tr, cols_va, cols_te))
        save_block('digit_te', k, Xtr, Xva, Xte, names)


def build_em(name, f):
    sub, keep = EM_BLOCKS[name]
    n_test = len(read_test())
    for k in range(5):
        src = EM_INFO / sub / f'fold_{k}'
        all_names = json.loads((src / 'manifest.json').read_text())['names']
        cols = [i for i, n in enumerate(all_names) if keep(n)]
        n_tr, n_va = int((f != k).sum()), int((f == k).sum())
        T, Q = np.load(src / 'train.npy'), np.load(src / 'query.npy')
        assert T.shape[0] == n_tr and Q.shape[0] == n_va + n_test, (T.shape, Q.shape)
        save_block(name, k, T[:, cols], Q[:n_va, cols], Q[n_va:, cols], [all_names[i] for i in cols])


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else 'all'
    y = target().astype(np.float64)
    f = folds()
    if kind in ('digits', 'all'):
        build_digits()
    if kind in ('digit_te', 'all'):
        build_digit_te(y, f)
    for name in EM_BLOCKS:
        if kind in (name, 'all'):
            build_em(name, f)


if __name__ == '__main__':
    main()
