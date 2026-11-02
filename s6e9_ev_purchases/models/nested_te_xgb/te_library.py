"""Nested target-encoding library, shared by the XGBoost pairs (train.py) and the RealMLP / TabM pairs
(../nested_te_realmlp_tabm).

Tokens (tok()): the exact values of every raw column; floor bins of income / commute / age of many widths (IncQ<w>, ComQ<w>,
Inc<w>, Com<w>, Age<w>); log-income bins (logInc<n>); the GPT-2 BPE leading piece and the first two pieces of the income
string (IncBpePre, IncBpePre2, from common/bpe.py).
A target encoding of a token with smoothing s maps each token value to (positives + s * prior) / (count + s) over the fit
rows (unseen value -> prior); channels store it as a logit.

Channel names (spec()):
  base_<col>, atom_inc30k, atom_com5      raw columns (categoricals coded by their sorted train levels) and two atom flags
  te1_<col>, te20_<col>, teav_<col>        exact values, s = 1 / s = 20 (teav: s = 1, averaged over 3 inner splits)
  teq1_, teq20_, teqav_, teqbv_ (IncQ / ComQ), w4a_, w4b_ (IncQ / ComQ), w6a_, w6b_, w7a_, w7b_ (Inc / Com / Age),
  w7c_logInc<n>_s<s>                       floor-bin encodings (teq1 / teqav / w*a: s = 1; teq20 / teqbv / w*b: s = 20)
  ares_<col>                               te1 - te20
  tex_<token>_s<s>                         any token, any smoothing
  hte_<parent>_s<s>_p<s0>                  exact-income rate shrunk (s) toward the rate of its parent token value, the parent
                                           rate itself shrunk (s0) toward the prior

build_library.py stores the 203 library channels per outer fold.  encode_val_like() encodes rows from the full map of a set
of fit rows; encode_inner() gives library-style (nested) encodings of training rows for channels outside the library."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, read_test, read_train, target  # noqa: E402

FEATURES = HERE / 'features'
INNER_SEED = 777
SCAT = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
# channel families: Q8 = all 203 library columns; MID = 170 of them (without ares, w4a, w4b, w7c)
Q8 = 'base,atom,te20,te1,teq20,teq1,ares,teav,teqav,teqbv,w4a,w4b,w6a,w6b,w7a,w7b,w7c'
MID = 'base,atom,te20,te1,teq20,teq1,teav,teqav,teqbv,w6a,w6b,w7a,w7b'

TR, TE = read_train(), read_test()
NTR, NTE = len(TR), len(TE)
P = [c for c in TE.columns if c != 'id']                    # the 13 predictors in csv order
_y = target()
Y = _y.astype(np.float64)
FOLDS = folds(_y)
INC_TR, INC_TE = TR.Annual_Income_USD.values, TE.Annual_Income_USD.values
COM_TR, COM_TE = TR.Daily_Commute_km.values, TE.Daily_Commute_km.values


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def codes_of(a, b):
    """codes of the train rows followed by the test rows, number of distinct values"""
    u, inv = np.unique(np.r_[a, b], return_inverse=True)
    return inv.astype(np.int64), len(u)


def rate(codes, y, K, s, pri):
    """smoothed positive rate of every token value (unseen -> prior)"""
    c = np.bincount(codes, minlength=K)
    pz = np.bincount(codes, weights=y, minlength=K)
    m = (pz + s * pri) / (c + s)
    m[c == 0] = pri
    return m


def _floor(col, w, rnd=None):
    a, b = (np.floor(d[col].values.astype(float) / w) * w for d in (TR, TE))
    if rnd is not None:
        a, b = a.round(rnd), b.round(rnd)
    return a, b


_TOK = {}


def tok(name):
    """(codes of the train rows then the test rows, number of codes) of a token"""
    if name not in _TOK:
        _TOK[name] = _make_tok(name)
    return _TOK[name]


def _make_tok(name):
    if name in P:
        return codes_of(TR[name].astype(str).values, TE[name].astype(str).values)
    m = re.fullmatch(r'IncQ(\d+)', name)
    if m:
        a, b = _floor('Annual_Income_USD', int(m.group(1)))
        return codes_of(a.astype(str), b.astype(str))
    m = re.fullmatch(r'ComQ([\d.]+)', name)
    if m:
        w = float(m.group(1))
        a, b = _floor('Daily_Commute_km', w, 2 if w in (0.5, 1.0, 2.0, 5.0) else 3)
        return codes_of(a.astype(str), b.astype(str))
    m = re.fullmatch(r'(Inc|Com|Age)([\d.]+)', name)
    if m:
        col = {'Inc': 'Annual_Income_USD', 'Com': 'Daily_Commute_km', 'Age': 'Age'}[m.group(1)]
        w = float(m.group(2)) if m.group(1) == 'Com' else int(m.group(2))
        return codes_of(*_floor(col, w, 3 if m.group(1) == 'Com' else None))
    m = re.fullmatch(r'logInc(\d+)', name)
    if m:
        inc_t, inc_e = INC_TR.astype(float), INC_TE.astype(float)
        edges = np.linspace(np.log(inc_t.min()), np.log(inc_t.max() + 1), int(m.group(1)) + 1)
        return codes_of(np.digitize(np.log(inc_t), edges).astype(str), np.digitize(np.log(inc_e), edges).astype(str))
    if name in ('IncBpePre', 'IncBpePre2'):
        piece = _bpe_table()['p1' if name == 'IncBpePre' else 'p1p2']
        return codes_of(pd.Series(INC_TR).map(piece).values.astype(str), pd.Series(INC_TE).map(piece).values.astype(str))
    raise KeyError(name)


_BPE = []


def _bpe_table():
    if not _BPE:
        from common.bpe import income_bpe_table
        _BPE.append(income_bpe_table(np.r_[INC_TR, INC_TE]))
    return _BPE[0]


def _static():
    out = {}
    for c in P:
        if c in SCAT:
            cats = sorted(TR[c].dropna().unique())
            out['base_' + c] = np.r_[pd.Categorical(TR[c], categories=cats).codes,
                                     pd.Categorical(TE[c], categories=cats).codes].astype(np.float32)
        else:
            out['base_' + c] = np.r_[TR[c].values, TE[c].values].astype(np.float32)
    out['atom_inc30k'] = np.r_[INC_TR == 30000, INC_TE == 30000].astype(np.float32)
    out['atom_com5'] = np.r_[COM_TR == 5.0, COM_TE == 5.0].astype(np.float32)
    return out


STATIC = _static()                                           # train rows then test rows


def spec(name):
    """-> ('static', values) | ('te', token, s) | ('ares', column) | ('hte', parent, s, s0); KeyError for other names"""
    if name in STATIC:
        return 'static', STATIC[name]
    fam, rest = name.split('_', 1)
    if fam in ('te1', 'teq1', 'teav', 'w6a', 'w7a'):
        return 'te', rest, 1.0
    if fam in ('te20', 'teq20', 'w6b', 'w7b'):
        return 'te', rest, 20.0
    if fam in ('teqav', 'w4a'):
        return 'te', rest.rsplit('_s', 1)[0], 1.0
    if fam in ('teqbv', 'w4b'):
        return 'te', rest.rsplit('_s', 1)[0], 20.0
    if fam in ('w7c', 'tex'):
        t, s = rest.rsplit('_s', 1)
        return 'te', t, float(s)
    if fam == 'ares':
        return 'ares', rest
    if fam == 'hte':
        parent, r2 = rest.rsplit('_s', 1)
        s, s0 = r2.split('_p')
        return 'hte', parent, float(s), float(s0)
    raise KeyError(name)


def hte_values(fit_idx, app, parent, s, s0):
    """hierarchical exact-income encoding of the rows `app` (indices into train rows then test rows), as logits"""
    ke, Ke = tok('Annual_Income_USD')
    kp, Kp = tok(parent)
    pri = Y[fit_idx].mean()
    ce = np.bincount(ke[fit_idx], minlength=Ke).astype(np.float64)
    se = np.bincount(ke[fit_idx], weights=Y[fit_idx], minlength=Ke)
    cp = np.bincount(kp[fit_idx], minlength=Kp).astype(np.float64)
    sp = np.bincount(kp[fit_idx], weights=Y[fit_idx], minlength=Kp)
    prior_row = ((sp + s0 * pri) / (cp + s0))[kp[app]]
    return lg((se[ke[app]] + s * prior_row) / (ce[ke[app]] + s))


def encode_val_like(fit_idx, app_idx, names, fallback=None, test=False):
    """full-map encoding of the rows app_idx (train rows, or test rows when test=True) from the labels of the train rows
    fit_idx, one column per name.  Names outside spec() are copied from `fallback` (same columns as names)."""
    pri = Y[fit_idx].mean()
    app = np.asarray(app_idx) + (NTR if test else 0)
    maps = {}

    def te(t, s):
        if (t, s) not in maps:
            kt, K = tok(t)
            maps[(t, s)] = lg(rate(kt[fit_idx], Y[fit_idx], K, s, pri)[kt[app]])
        return maps[(t, s)]
    out = np.zeros((len(app_idx), len(names)), np.float32)
    for j, n in enumerate(names):
        try:
            sp = spec(n)
        except KeyError:
            if fallback is None:
                raise
            out[:, j] = fallback[:, j]
            continue
        if sp[0] == 'static':
            out[:, j] = sp[1][app]
        elif sp[0] == 'te':
            out[:, j] = te(sp[1], sp[2])
        elif sp[0] == 'hte':
            out[:, j] = hte_values(fit_idx, app, sp[1], sp[2], sp[3])
        else:
            out[:, j] = te(sp[1], 1.0) - te(sp[1], 20.0)
    return np.nan_to_num(out, nan=0., posinf=0., neginf=0.)


def encode_inner(rows, names, seed=INNER_SEED, K=5):
    """library-style encodings of training rows: inner StratifiedKFold(K) within `rows`, each part encoded by the map of the
    other parts (prior = mean label of all `rows`)"""
    rows = np.asarray(rows)
    pri = Y[rows].mean()
    out = np.zeros((len(rows), len(names)), np.float32)
    splits = list(StratifiedKFold(K, shuffle=True, random_state=seed).split(rows, Y[rows]))
    maps = {}

    def te(t, s):
        if (t, s) not in maps:
            kt, Kt = tok(t)
            v = np.zeros(len(rows))
            for a, b in splits:
                f = rows[a]
                v[b] = rate(kt[f], Y[f], Kt, s, pri)[kt[rows[b]]]
            maps[(t, s)] = lg(v)
        return maps[(t, s)]
    for j, n in enumerate(names):
        sp = spec(n)
        if sp[0] == 'static':
            out[:, j] = sp[1][rows]
        elif sp[0] == 'te':
            out[:, j] = te(sp[1], sp[2])
        elif sp[0] == 'hte':
            v = np.zeros(len(rows))
            for a, b in splits:
                v[b] = hte_values(rows[a], rows[b], sp[1], sp[2], sp[3])
            out[:, j] = v
        else:
            out[:, j] = te(sp[1], 1.0) - te(sp[1], 20.0)
    return np.nan_to_num(out, nan=0., posinf=0., neginf=0.)
