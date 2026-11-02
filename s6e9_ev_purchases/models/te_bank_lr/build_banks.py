#!/usr/bin/env python
"""Fold-local target-encoding banks for the logistic regressions and the RealMLP (GPU, CuPy).

Keys: the 13 raw fields; income // 10, 100, 1000, 5000, 10000; the GPT-2 BPE first piece and first + second piece of the
income; BPE piece count; income mod 10 / 100 / 1000; commute integer / 5 km / decimal; age // 5; 64 quantile bins of the
generator-formula score (rule = 1.2 income / 1e5 + 0.6 concern + 2 subsidy - [anxiety medium] - 3 [anxiety high] - 5.5).
Banks:
  S   29 singles: TE alpha 20 + TE alpha 1 + log1p(train + test count)
  P   105 pairs of 15 core keys (TE alpha 20)            P5 / P1: the same pairs at alpha 5 / 1
  I   36 income-granularity x context pairs (TE alpha 20 + log count)    I5 / I1: alpha 5 / 1
  T   41 triples (TE alpha 20)                           T5 / T1: alpha 5 / 1
  SR, PR, IR, TR   the same keys, smoothed mean of the formula residual y - Phi(rule) (alpha 20)
Encoding per outer fold k: outer-train rows by a 5-fold inner cross-fit (StratifiedKFold seed 4200 + k), valid / test rows by
the state fitted on all outer-train rows; priors from the fit rows only. TE value = centered logit
logit((pos + a * prior) / (cnt + a)) - logit(prior).
Writes cache/fold_k/BANK_{train,valid,test}.npy (float32), cache/BANK_names.json, the label-free numeric pack cache/num.npy
(+ num_names.json: rule score / logit, BPE piece count, digits) and cache/income_bpe_pieces.csv (label-free).
Usage: python build_banks.py [--banks S,P,I,T,P5,I5,T5,P1,I1,T1,SR,PR,IR,TR]"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.bpe import income_bpe_table  # noqa: E402
from common.data import folds, read_test, read_train, target  # noqa: E402

CACHE = HERE / 'cache'
ALL_BANKS = 'S,P,I,T,P5,I5,T5,P1,I1,T1,SR,PR,IR,TR'
RESIDUAL = ('SR', 'PR', 'IR', 'TR')


def bpe_pieces(incomes):
    """cache/income_bpe_pieces.csv: income, first piece, second piece, last piece, piece count (GPT-2 BPE, label-free)."""
    path = CACHE / 'income_bpe_pieces.csv'
    if not path.exists():
        t = income_bpe_table(incomes)
        df = pd.DataFrame({'income': t.index, 'p1': t['p1'].to_numpy(), 'p2': [p[1] for p in t['pieces']],
                           'plast': [p[-1] for p in t['pieces']], 'npieces': t['n_pieces'].to_numpy()})
        df.to_csv(path, index=False)
    return pd.read_csv(path, dtype={'p1': str, 'p2': str, 'plast': str}, keep_default_na=False).set_index('income')


def base_codes():
    tr, te = read_train(), read_test()
    df = pd.concat([tr, te], ignore_index=True)
    inc = np.rint(df['Annual_Income_USD'].to_numpy()).astype(np.int64)
    com = df['Daily_Commute_km'].to_numpy()
    bpe = bpe_pieces(df['Annual_Income_USD'].to_numpy())
    incf = df['Annual_Income_USD'].astype(float)
    env = df['Environmental_Concern_Level'].astype(float).to_numpy()
    sub = (df['Subsidy_Available'] == 'Yes').to_numpy(float)
    med = (df['Range_Anxiety_Level'] == 'Medium').to_numpy(float)
    high = (df['Range_Anxiety_Level'] == 'High').to_numpy(float)
    rule = 1.2 * inc / 1e5 + 0.6 * env + 2 * sub - med - 3 * high - 5.5
    raw = {'age': df['Age'], 'inc': inc, 'com': com, 'ncars': df['Number_of_Cars_Owned'], 'sth': df['Charging_Stations_Near_Home'],
           'stw': df['Charging_Stations_Near_Work'], 'env': df['Environmental_Concern_Level'], 'gender': df['Gender'], 'city': df['City_Type'],
           'car': df['Current_Car_Type'], 'home': df['Home_Charging_Possible'], 'sub': df['Subsidy_Available'], 'anx': df['Range_Anxiety_Level'],
           'inc10': inc // 10, 'inc100': inc // 100, 'inc1000': inc // 1000, 'inc5000': inc // 5000, 'inc10000': inc // 10000,
           'bp1': incf.map(bpe['p1']).to_numpy(), 'bp12': (incf.map(bpe['p1']) + '|' + incf.map(bpe['p2'])).to_numpy(),
           'npieces': incf.map(bpe['npieces']).to_numpy(), 'mod10': inc % 10, 'mod100': inc % 100, 'mod1000': inc % 1000,
           'com_int': np.floor(com), 'com5': np.floor(com / 5), 'com_dec': np.rint((com - np.floor(com)) * 10), 'age5': df['Age'] // 5,
           'rulebin': pd.qcut(rule, 64, labels=False, duplicates='drop')}
    codes = {k: pd.factorize(pd.Series(np.asarray(v)).astype(str))[0].astype(np.int64) for k, v in raw.items()}
    num = np.column_stack([rule, np.log(np.clip(norm.cdf(rule), 1e-12, 1) / np.clip(1 - norm.cdf(rule), 1e-12, 1)),
                           raw['npieces'].astype(float), inc % 10, (inc % 100 == 0).astype(float), (inc % 1000 == 0).astype(float)]).astype(np.float32)
    return codes, num, ['rule_score', 'rule_logit', 'bpe_npieces', 'inc_mod10', 'inc_round100', 'inc_round1000']


def banks():
    S = [(c,) for c in ['age', 'inc', 'com', 'ncars', 'sth', 'stw', 'env', 'gender', 'city', 'car', 'home', 'sub', 'anx', 'inc10', 'inc100',
                        'inc1000', 'inc5000', 'inc10000', 'bp1', 'bp12', 'npieces', 'mod10', 'mod100', 'mod1000', 'com_int', 'com5', 'com_dec',
                        'age5', 'rulebin']]
    core = ['env', 'sub', 'anx', 'home', 'city', 'car', 'gender', 'ncars', 'sth', 'stw', 'age5', 'com_int', 'inc1000', 'bp1', 'rulebin']
    P = list(itertools.combinations(core, 2))
    ctx = ['env', 'sub', 'anx', 'home', 'city', 'car', 'gender', 'ncars', 'sth', 'stw', 'age5', 'com_int']
    I = [(g, c) for g in ('inc', 'inc100', 'bp12') for c in ctx]  # noqa: E741
    T = list(itertools.combinations(['env', 'sub', 'anx', 'home', 'inc1000', 'bp1', 'rulebin'], 3)) + \
        [('city', 'home', 'anx'), ('car', 'env', 'sub'), ('gender', 'env', 'anx'), ('ncars', 'sub', 'anx'), ('sth', 'home', 'anx'), ('age5', 'env', 'sub')]
    return {'S': S, 'P': P, 'I': I, 'T': T, 'P5': P, 'I5': I, 'T5': T, 'P1': P, 'I1': I, 'T1': T, 'SR': S, 'PR': P, 'IR': I, 'TR': T}


def key_codes(codes, key):
    if len(key) == 1:
        return codes[key[0]]
    c = codes[key[0]].copy()
    for f in key[1:]:
        c = c * (int(codes[f].max()) + 1) + codes[f]
    return np.unique(c, return_inverse=True)[1].astype(np.int64)


def te_res_gpu(c, rg, fit, apply, alpha):
    """Smoothed mean of the formula residual per key, minus its prior."""
    prior = float(rg[fit].mean())
    size = int(c.max()) + 1
    cnt = cp.bincount(c[fit], minlength=size).astype(cp.float64)
    tot = cp.bincount(c[fit], weights=rg[fit], minlength=size)
    return (tot[c[apply]] + alpha * prior) / (cnt[c[apply]] + alpha) - prior


def te_gpu(c, yg, fit, apply, alpha):
    prior = float(yg[fit].mean())
    size = int(c.max()) + 1
    cnt = cp.bincount(c[fit], minlength=size).astype(cp.float64)
    pos = cp.bincount(c[fit], weights=yg[fit], minlength=size)
    p = (pos[c[apply]] + alpha * prior) / (cnt[c[apply]] + alpha)
    p = cp.clip(p, 1e-6, 1 - 1e-6)
    return cp.log(p / (1 - p)) - np.log(prior / (1 - prior))


def bank_names(bn, keys):
    names = []
    for key in keys:
        nm = '*'.join(key)
        if bn in RESIDUAL:
            names += [f'res20:{nm}']
            continue
        names += [f'te{bn[1:]}:{nm}' if (len(bn) == 2 and bn[1] in '15') else f'te20:{nm}'] + \
                 ([f'te1:{nm}', f'logn:{nm}'] if bn == 'S' else []) + ([f'logn:{nm}'] if bn == 'I' else [])
    return names


def build(bank_list):
    CACHE.mkdir(parents=True, exist_ok=True)
    codes, num, num_names = base_codes()
    np.save(CACHE / 'num.npy', num)
    (CACHE / 'num_names.json').write_text(json.dumps(num_names))
    y = target().astype(np.float64)
    f = folds()
    n_tr, n_all = len(y), len(codes['inc'])
    yall = cp.asarray(np.concatenate([y, np.zeros(n_all - n_tr)]))
    prule = norm.cdf(num[:, 0].astype(np.float64))  # label-free generator-formula probability
    rall = yall - cp.asarray(prule)  # residuals; test rows are never fit rows
    B = banks()
    for bn in bank_list:
        keys = B[bn]
        names = bank_names(bn, keys)
        (CACHE / f'{bn}_names.json').write_text(json.dumps(names))
        kc = {key: cp.asarray(key_codes(codes, key)) for key in keys}
        for k in range(5):
            t0 = time.time()
            ti, vi = np.flatnonzero(f != k), np.flatnonzero(f == k)
            te_idx = np.arange(n_tr, n_all)
            inner = np.empty(len(ti), np.int32)
            for j, (_, a) in enumerate(StratifiedKFold(5, shuffle=True, random_state=4200 + k).split(np.zeros(len(ti)), y[ti])):
                inner[a] = j
            tig, vig, teg = cp.asarray(ti), cp.asarray(vi), cp.asarray(te_idx)
            parts = [(cp.asarray(ti[inner != j]), cp.asarray(np.flatnonzero(inner == j))) for j in range(5)]
            cols_tr, cols_va, cols_te = [], [], []
            for key in keys:
                c = kc[key]
                if bn in RESIDUAL:
                    v = cp.empty(len(ti))
                    for fit, rel in parts:
                        v[rel] = te_res_gpu(c, rall, fit, tig[rel], 20.0)
                    cols_tr.append(v)
                    cols_va.append(te_res_gpu(c, rall, tig, vig, 20.0))
                    cols_te.append(te_res_gpu(c, rall, tig, teg, 20.0))
                    continue
                alphas = (20.0, 1.0) if bn == 'S' else ((float(bn[1:]),) if len(bn) == 2 and bn[1] in '15' else (20.0,))
                for a in alphas:
                    v = cp.empty(len(ti))
                    for fit, rel in parts:
                        v[rel] = te_gpu(c, yall, fit, tig[rel], a)
                    cols_tr.append(v)
                    cols_va.append(te_gpu(c, yall, tig, vig, a))
                    cols_te.append(te_gpu(c, yall, tig, teg, a))
                if bn in ('S', 'I'):  # label-free frequency over train + test
                    cnt = cp.bincount(c, minlength=int(c.max()) + 1).astype(cp.float64)
                    lc = cp.log1p(cnt[c])
                    cols_tr.append(lc[tig])
                    cols_va.append(lc[vig])
                    cols_te.append(lc[teg])
            out = CACHE / f'fold_{k}'
            out.mkdir(parents=True, exist_ok=True)
            for r, cols in (('train', cols_tr), ('valid', cols_va), ('test', cols_te)):
                X = cp.asnumpy(cp.stack(cols, 1)).astype(np.float32)
                assert np.isfinite(X).all() and X.shape[1] == len(names)
                np.save(out / f'{bn}_{r}.npy', X)
            print(f'bank {bn} fold {k}: {len(names)} columns [{time.time() - t0:.1f}s]', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--banks', default=ALL_BANKS)
    build(ap.parse_args().banks.split(','))
