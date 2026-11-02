#!/usr/bin/env python
"""Stage-1 tables shared by every model of this folder.

Keys (integer codes over train + test + original rows) of the 13 raw fields, income bins (10, 100, 1000, 5000), commute
bins (1, 5, 10 km) and the GPT-2 BPE pieces of the integer income / commute strings (leading piece, first two pieces,
second piece, last piece, last two pieces, digit-length pattern). Label-free features: raw values (categoricals coded
by their sorted levels), BPE chunk counts, log counts of every key in train + test and in the original data, the
original-data label log-odds of every key (smoothed with 5 pseudo-rows), the gap to the nearest original income, the
known source-law score (1.2 income / 1e5 + 0.6 concern + 2 subsidy - anxiety - 5.5), its probit probability and logit.
Writes cache/stage1/{base.parquet, keys.parquet, source_keys.parquet, predictors.parquet, source_base.npy, source_y.npy,
y.npy, train_ids.npy, test_ids.npy, fold_id.npy, splits.npz, manifest.json}.
Usage: python prepare.py"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit, ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.bpe import tokenizer  # noqa: E402
from common.data import TARGET, folds, read_original, read_test, read_train  # noqa: E402
from layout import STAGE1 as CACHE  # noqa: E402

INC = 'Annual_Income_USD'
COM = 'Daily_Commute_km'


class NumericBPE:
    """GPT-2 BPE pieces of a number string in byte-level form, e.g. ' 92887' -> ('Ġ9', '28', '87')."""
    def __init__(self):
        self.tok = tokenizer()

    @lru_cache(maxsize=300000)
    def word(self, text):
        return tuple(self.tok.tokenize(text))


def source_score(d):
    return (1.2*d[INC].fillna(85000).to_numpy()/1e5
            + .6*d.Environmental_Concern_Level.fillna(3).to_numpy()
            + 2*d.Subsidy_Available.eq('Yes').to_numpy()
            - d.Range_Anxiety_Level.map({'Low': 0, 'Medium': 1, 'High': 3}).to_numpy() - 5.5)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    tr, te, orig = read_train(), read_test(), read_original()
    y = tr[TARGET].eq('Yes').to_numpy('int8')
    oy = orig[TARGET].eq('Yes').to_numpy('int8')
    both = pd.concat([tr.drop(columns=['id', TARGET]), te.drop(columns='id')], ignore_index=True)
    assert both.notna().all().all()
    n = len(both)
    numeric = list(both.select_dtypes('number'))
    allframe = pd.concat([both, orig[both.columns]], ignore_index=True)
    keys, knames, raw, ftypes, categorical = {}, [], {}, [], []
    for c in both:
        if c in numeric:
            scale = 10 if c == COM else 1
            val = (allframe[c]*scale).round().fillna(-1).to_numpy('int64')
            raw[c] = both[c].to_numpy('float32')
            ftypes.append('q')
        else:
            val = allframe[c].fillna('__missing__').to_numpy()
            raw[c] = pd.Categorical(both[c], categories=np.sort(allframe[c].dropna().unique())).codes.astype('float32')
            ftypes.append('c')
            categorical.append(c)
        keys[c] = pd.factorize(val, sort=True)[0].astype('int32')
        knames.append(c)
    for width in (10, 100, 1000, 5000):
        keys[f'inc{width}'] = pd.factorize((allframe[INC].fillna(-1)//width).to_numpy('int64'), sort=True)[0].astype('int32')
    for width in (1, 5, 10):
        keys[f'com{width}'] = pd.factorize((allframe[COM].fillna(-1)//width).to_numpy('int64'), sort=True)[0].astype('int32')
    tok = NumericBPE()
    for prefix, col in [('inc', INC), ('com', COM)]:
        integers = allframe[col].fillna(-1).astype('int64').to_numpy()
        unique, inv = np.unique(integers, return_inverse=True)
        seq = [tok.word(' '+str(int(v))) if v >= 0 else ('None',) for v in unique]
        values = {f'{prefix}_bpe0': [s[0] for s in seq],
                  f'{prefix}_bpe01': ['|'.join(s[:2]) for s in seq],
                  f'{prefix}_bpe1': [s[1] if len(s) > 1 else '_END' for s in seq],
                  f'{prefix}_bpelast': [s[-1] for s in seq],
                  f'{prefix}_bpepair': ['|'.join(s[-2:]) for s in seq],
                  f'{prefix}_pattern': ['-'.join(str(len(t.replace('Ġ', ''))) for t in s) for s in seq]}
        for name, v in values.items():
            keys[name] = pd.factorize(np.asarray(v)[inv], sort=True)[0].astype('int32')
        raw[f'{prefix}_bpe_chunks'] = np.asarray([len(s) for s in seq])[inv[:n]].astype('float32')
        ftypes.append('q')
        raw[f'{prefix}_bpe_first_length'] = np.asarray([len(s[0].replace('Ġ', '')) for s in seq])[inv[:n]].astype('float32')
        ftypes.append('q')
    keyframe = pd.DataFrame(keys)
    # train + test support and original-data label evidence of every key
    for c in knames+[f'inc{w}' for w in (10, 100, 1000, 5000)]+['inc_bpe0', 'inc_bpe01']:
        a = keys[c][:n]
        b = keys[c][n:]
        size = int(keys[c].max())+1
        ns = np.bincount(a, minlength=size)
        no = np.bincount(b, minlength=size)
        yo = np.bincount(b, weights=oy, minlength=size)
        raw['logcount_'+c] = np.log1p(ns[a]).astype('float32')
        ftypes.append('q')
        raw['source_count_'+c] = np.log1p(no[a]).astype('float32')
        ftypes.append('q')
        raw['source_llr_'+c] = (logit(((yo+5*oy.mean())/(no+5)).clip(1e-5, 1-1e-5))-logit(oy.mean()))[a].astype('float32')
        ftypes.append('q')
    inc = both[INC].to_numpy()
    src = np.sort(orig[INC].dropna().unique())
    j = np.searchsorted(src, inc).clip(0, len(src)-1)
    left = src[np.maximum(j-1, 0)]
    right = src[j]
    gap = np.where(abs(inc-left) <= abs(inc-right), inc-left, inc-right)
    raw['source_gap_signed_log'] = np.sign(gap)*np.log1p(abs(gap))
    ftypes.append('q')
    raw['source_exact'] = (gap == 0).astype('float32')
    ftypes.append('q')
    s = source_score(both)
    p = ndtr(s)
    base = logit(p.clip(1e-5, 1-1e-5))
    for name, v in [('source_score', s), ('source_probability', p), ('source_logit', base),
                    ('anxiety_score', both[COM].to_numpy()-5*both.Charging_Stations_Near_Home.to_numpy()
                     - 5*both.Charging_Stations_Near_Work.to_numpy()-150*both.Home_Charging_Possible.eq('Yes').to_numpy()),
                    ('log_income', np.log1p(inc)), ('log_commute', np.log1p(both[COM].to_numpy()))]:
        raw[name] = v.astype('float32')
        ftypes.append('q')
    pd.DataFrame(raw).astype('float32').to_parquet(CACHE/'base.parquet', index=False)
    keyframe.iloc[:n].to_parquet(CACHE/'keys.parquet', index=False)
    keyframe.iloc[n:].to_parquet(CACHE/'source_keys.parquet', index=False)
    both.to_parquet(CACHE/'predictors.parquet', index=False)
    np.save(CACHE/'source_base.npy', base.astype('float32'))
    np.save(CACHE/'source_y.npy', oy)
    np.save(CACHE/'y.npy', y)
    np.save(CACHE/'train_ids.npy', tr.id.to_numpy())
    np.save(CACHE/'test_ids.npy', te.id.to_numpy())
    fold = folds(y)
    parts = {}
    for k in range(5):
        parts[f'fit{k}'] = np.flatnonzero(fold != k)
        parts[f'valid{k}'] = np.flatnonzero(fold == k)
    np.save(CACHE/'fold_id.npy', fold)
    np.savez(CACHE/'splits.npz', **parts)
    assert len(ftypes) == len(raw)
    meta = dict(n_train=len(tr), n_test=len(te), raw_columns=list(raw), feature_types=ftypes, numeric=numeric,
                categorical=categorical, key_columns=list(keys), key_sizes={c: int(keyframe[c].max())+1 for c in keys})
    (CACHE/'manifest.json').write_text(json.dumps(meta, indent=2))
    print(f'stage 1 tables: {len(raw)} base features, {len(keys)} keys', flush=True)


if __name__ == '__main__':
    main()
