"""Data and nested estimators of the information-aware income EM model (cache/em_info).

Reads, besides the competition data, the caches of ../response_features: static feature blocks + their nested
response margins (cache/static), the original-only source distribution channels (cache/base) and the income -> leading
GPT-2 BPE piece map (cache/bpe; else common/bpe.py)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import read_test, read_train  # noqa: E402
from layout import EM_INFO, RESPONSE  # noqa: E402

TARGET = 'Will_Buy_EV'
RAW = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
       'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
       'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
       'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
STATIC_BLOCKS = RESPONSE/'static'                    # {block}_{train,test}.npy + manifest.json
NESTED_MARGIN = RESPONSE/'static'                    # fold_{k}/nested_response_margin_{role}.npy
SOURCE_CHANNELS = RESPONSE/'base'/'fold_0'           # source_full_global_{role}.npy + manifest.json
BPE_CSV = RESPONSE/'bpe'/'income_to_first_piece.csv'
BASE_DIR = EM_INFO/'base'
TEACHERS = EM_INFO/'teachers_gpu'
FEATURES = EM_INFO/'features_gpu'
INFORMATION = EM_INFO/'information_features'
BAYES = EM_INFO/'bayes_margins'


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1-1e-7)
    return np.log(p)-np.log1p(-p)


def splits(idx, y):
    return [(idx[a], idx[b]) for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(idx)), y[idx])]


def codes(values):
    return pd.factorize(values, sort=True)[0].astype(np.int32)


def income_first_piece(income):
    """Leading GPT-2 BPE piece of each income, as an integer."""
    if BPE_CSV.exists():
        return income.map(pd.read_csv(BPE_CSV).set_index('income').first_piece)
    from common.bpe import income_bpe_table
    table = income_bpe_table(income.to_numpy())
    return income.map(table['p1'].astype(np.int64))


class Data:
    def __init__(self):
        tr, te = read_train(), read_test()
        self.y = (tr[TARGET] == 'Yes').to_numpy(np.int8)
        self.n = len(tr)
        self.nt = len(te)
        self.df = pd.concat([tr[RAW], te[RAW]], ignore_index=True)
        self.N = len(self.df)
        self.ids = tr.id.to_numpy()
        self.test_ids = te.id.to_numpy()
        self.folds = np.full(self.n, -1, np.int8)
        for f, (_, b) in enumerate(splits(np.arange(self.n), self.y)):
            self.folds[b] = f
        self.keys = {k: codes(self.df[k]) for k in RAW}
        income = self.df.Annual_Income_USD.to_numpy(np.int64)
        commute = np.rint(self.df.Daily_Commute_km.to_numpy(np.float64)*10).astype(np.int64)
        for w in [50, 100, 500, 1000, 2500, 5000, 10000]:
            self.keys[f'inc{w}'] = codes(income//w)
        for w in [10, 50, 100]:
            self.keys[f'km{w}'] = codes(commute//w)
        bp = income_first_piece(self.df.Annual_Income_USD)
        assert bp.notna().all()
        self.keys['bpe'] = codes(bp)
        self.keys['inc_parent'] = codes(pd.MultiIndex.from_arrays([income//1000, bp]))
        self.keys['km_parent'] = codes(commute//50)
        for short, key in [('sub', 'Subsidy_Available'), ('anx', 'Range_Anxiety_Level'), ('env', 'Environmental_Concern_Level')]:
            self.keys[f'inc_{short}'] = codes(pd.MultiIndex.from_arrays([income, self.df[key]]))
            self.keys[f'inc_parent_{short}'] = codes(pd.MultiIndex.from_arrays([income//1000, bp, self.df[key]]))
        self.card = {k: int(c.max())+1 for k, c in self.keys.items()}
        self.raw = pd.get_dummies(self.df, columns=[k for k in RAW if self.df[k].dtype == 'object'], dtype=np.float32)
        self.raw_names = list(self.raw.columns)
        self.raw = self.raw.to_numpy(np.float32)
        self.test = np.arange(self.n, self.N)
        self._static = None

    def static(self):
        if self._static is not None:
            return self._static
        m = json.loads((STATIC_BLOCKS/'manifest.json').read_text())
        arr, names = [], []
        for block in ['raw', 'smooth_focus', 'source_pair_core', 'lineage']:
            arr.append(np.concatenate([np.load(STATIC_BLOCKS/f'{block}_{role}.npy') for role in ['train', 'test']]))
            names += [f'{block}__{x}' for x in m['block_names'][block]]
        for key in RAW:
            count = np.bincount(self.keys[key], minlength=self.card[key])
            arr.append(np.log1p(count[self.keys[key]])[:, None].astype(np.float32))
            names.append('freq__'+key)
        # original-only source distribution channels, restored into full row order
        m = json.loads((SOURCE_CHANNELS/'manifest.json').read_text())
        allnames = m['source_full_global_names']
        selected = m['source_focus_names']
        take = [allnames.index(x) for x in selected]
        whole = np.empty((self.N, len(take)), np.float32)
        a = np.flatnonzero(self.folds != 0)
        b = np.flatnonzero(self.folds == 0)
        for role, idx in [('train', a), ('valid', b), ('test', self.test)]:
            whole[idx] = np.load(SOURCE_CHANNELS/f'source_full_global_{role}.npy')[:, take]
        arr.append(whole)
        names += ['source__'+s for s in selected]
        X = np.column_stack(arr).astype(np.float32)
        assert np.isfinite(X).all()
        self._static = (X, names)
        return self._static

    def margin(self, fold):
        """Nested response margin of ../response_features for the outer-train, fold and test rows."""
        folder = NESTED_MARGIN/f'fold_{fold}'
        return [np.load(folder/f'nested_response_margin_{role}.npy').ravel() for role in ['train', 'valid', 'test']]

    def response(self, fit, query, keys, alphas=(1., 20.), y=None):
        y = self.y if y is None else y

        def apply(a, b):
            pi = float(y[a].mean())
            cols = []
            for key in keys:
                c = self.keys[key]
                n = np.bincount(c[a], minlength=self.card[key])
                s = np.bincount(c[a], weights=y[a], minlength=self.card[key])
                z = []
                for alpha in alphas:
                    z.append(logit((s[c[b]]+alpha*pi)/(n[c[b]]+alpha))-logit(pi))
                cols.extend(z)
                if len(z) == 2:
                    cols.append(z[0]-z[1])
            return np.column_stack(cols).astype(np.float32)
        width = len(keys)*(len(alphas)+(len(alphas) == 2))
        F = np.empty((len(fit), width), np.float32)
        lookup = np.full(self.n, -1, np.int64)
        lookup[fit] = np.arange(len(fit))
        for a, b in splits(fit, y):
            F[lookup[b]] = apply(a, b)
        Q = apply(fit, query)
        names = []
        for key in keys:
            names += [f'te__{key}__a{a:g}' for a in alphas]
            if len(alphas) == 2:
                names.append(f'te__{key}__diff')
        return F, Q, names

    def base(self, fold):
        a = np.flatnonzero(self.folds != fold)
        b = np.flatnonzero(self.folds == fold)
        q = np.concatenate([b, self.test])
        X, names = self.static()
        keys = RAW+['inc50', 'inc100', 'inc500', 'inc1000', 'inc2500', 'inc5000', 'inc10000', 'km10', 'km50', 'bpe', 'inc_parent']
        F, Q, tnames = self.response(a, q, keys)
        return a, b, np.column_stack([X[a], F]), np.column_stack([X[q], Q]), names+tnames

    def context(self, fit, query, field):
        if field == 'income':
            excluded = 'Annual_Income_USD'
            extra = ['inc1000', 'bpe', 'inc_parent']
            raw = self.raw.copy()
            raw[:, self.raw_names.index(excluded)] = np.floor(raw[:, self.raw_names.index(excluded)]/1000)*1000+500
        else:
            excluded = 'Daily_Commute_km'
            extra = ['km50', 'inc1000', 'bpe']
            raw = self.raw.copy()
            raw[:, self.raw_names.index(excluded)] = np.floor(raw[:, self.raw_names.index(excluded)]/5)*5+2.5
        keys = [k for k in RAW if k != excluded]+extra
        F, Q, names = self.response(fit, query, keys)
        return np.column_stack([raw[fit], F]), np.column_stack([raw[query], Q])


def train_noise(X, names, fold, sd=.04):
    """Gaussian noise on the train-side target-encoding columns; one stream per column name and fold."""
    X = X.copy()
    for j, name in enumerate(names):
        if name.startswith('te__'):
            seed = int.from_bytes(hashlib.sha256(f'{name}/{fold}/42'.encode()).digest()[:8], 'little')
            X[:, j] += np.random.default_rng(seed).normal(0, sd, len(X)).astype(np.float32)
    return X
