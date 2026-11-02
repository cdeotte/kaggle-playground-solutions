#!/usr/bin/env python
"""XGBoost on the nested target-encoding library (te_library.py): one runner for the 8 pairs of this folder.

Every pair is a logit mean of streams (PAIRS, STREAMS below).  A stream is run per outer fold k with one XGBoost recipe:
depth 4, eta 0.02, min_child_weight 5, lambda 4, subsample 0.8, colsample_bytree 0.8, colsample_bynode 0.6,
num_parallel_tree 6, max_bin 256, and a base margin from a logistic regression (C=1, standardised inputs) on the te1 / te20 /
teq1 / teq20 channels.  Rounds: early stopping (200) on a 10% holdout of the training rows, then a refit on all of them for
1.11x the best round.  Test = mean of the 5 fold models of the stream.

Two schemes for the label channels of the training rows:
  library (noise line): the nested encodings of the library; esval re-encodes the early-stopping rows from the fit rows'
      labels only (validation-style); tenoise adds Gaussian noise (sd 0.1, logit units) to every label channel of the
      outer-train rows; tn_reps noise draws are averaged in logit space.  The LR margin of the outer-train rows is inner
      cross-fitted (StratifiedKFold(5, seed 777)), validation / test rows get the LR fitted on all outer-train rows.
  pool (half): outer-train is split 50/50 (half_reps times); a model trains on one half with every label channel encoded
      from the other half, its validation and test rows are encoded from that same other half; both directions of every
      split are averaged in logit space (6 models per fold).
Options: static (label-free bases of static_bases.py), ic col (interaction constraints: one clique per source column, the
raw / atom / static columns in every clique), prune (drop the lowest-gain columns of the early-stopping model before the
refit), testall (test rows encoded from all train labels), deltas (differences between the income channels appended as
columns with column-sampling weight fw_delta), bpedelta (the BPE channels also get deltas), extra (channels outside the
library, encoded like the library: nested within outer-train, full map for validation / test).

Usage: python train.py all | <pair> [--folds 0,1,2,3,4]   runs the missing folds of every stream, then writes preds/<pair>
       python train.py --stream <stream> [--folds 2]       one stream (finished folds are kept in runs/parts/)"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import assemble, fold_list, have_part, save_pair, save_part  # noqa: E402
import te_library as L  # noqa: E402

RUN_DIR = HERE / 'runs'
Y, F, NTR, NTE = L.Y, L.FOLDS, L.NTR, L.NTE

XGB = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', max_depth=6, eta=0.05,
           subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=4, max_bin=256, num_parallel_tree=1)
BASE = dict(mrg=1.0, num_parallel_tree=6, max_depth=4, eta=0.02, min_child_weight=5, colsample_bynode=0.6, rounds=30000)
NOISE = dict(BASE, esval=1, tenoise=0.1)
POOL = dict(BASE, half=1, half_reps=3)
PER = dict(prune=150, static='per', ic='col')
DLT = dict(deltas=1, fw_delta=0.5)
BPE2 = 'tex_IncBpePre_s1,tex_IncBpePre_s20'
BPE4 = BPE2 + ',tex_IncBpePre2_s1,tex_IncBpePre2_s20'
HIER = BPE4 + ',hte_IncBpePre_s10_p20,hte_IncBpePre2_s5_p20'
BEST = ('tex_IncBpePre_s1,tex_IncBpePre_s5,tex_IncBpePre_s20,tex_IncBpePre2_s1,tex_IncBpePre2_s20,'
        'hte_IncBpePre_s3_p20,hte_IncBpePre_s10_p20')

STREAMS = {   # name: (library channel families, settings, seed)
    'q8_pool': (L.Q8, POOL, 42),
    'q8_noise': (L.Q8, dict(NOISE, tn_reps=2), 42),
    'per_noise': (L.Q8, dict(NOISE, **PER, tn_reps=2, testall=1), 42),
    'per_pool': (L.Q8, dict(POOL, **PER, testall=1), 42),
    'dlt_noise_fw02_s42': (L.Q8, dict(NOISE, **PER, deltas=1, fw_delta=0.2), 42),
    'dlt_noise_fw02_s7': (L.Q8, dict(NOISE, **PER, deltas=1, fw_delta=0.2), 7),
    'dlt_noise_s11': (L.Q8, dict(NOISE, **PER, **DLT), 11),
    'dlt_noise_s13': (L.Q8, dict(NOISE, **PER, **DLT), 13),
    'dlt_noise_s42': (L.Q8, dict(NOISE, **PER, **DLT), 42),
    'dlt_noise_s7': (L.Q8, dict(NOISE, **PER, **DLT), 7),
    'dlt_pool_s42': (L.Q8, dict(POOL, **PER, **DLT), 42),
    'dlt_pool_s7': (L.Q8, dict(POOL, **PER, **DLT), 7),
    'bpe_s42': (L.Q8, dict(NOISE, **PER, **DLT, extra=BPE4), 42),
    'bpe_s7': (L.Q8, dict(NOISE, **PER, **DLT, extra=BPE4), 7),
    'bpe_hier_s42': (L.Q8, dict(NOISE, **PER, **DLT, extra=HIER), 42),
    'bpe_hier_s7': (L.Q8, dict(NOISE, **PER, **DLT, extra=HIER), 7),
    'bpe_best_s42': (L.Q8, dict(NOISE, **PER, **DLT, bpedelta=1, extra=BEST), 42),
    'bpe_best_s7': (L.Q8, dict(NOISE, **PER, **DLT, bpedelta=1, extra=BEST), 7),
    'per_bpe_noise': (L.Q8, dict(NOISE, **PER, tn_reps=2, testall=1, extra=BPE2), 42),
    'per_bpe_pool': (L.Q8, dict(POOL, **PER, testall=1, extra=BPE2), 42),
    'q8_bpe_pool': (L.Q8, dict(POOL, extra=BPE2), 42),
    'q8_bpe_noise': (L.Q8, dict(NOISE, tn_reps=2, extra=BPE2), 42),
}
PAIRS = {     # pair: lines; a line is the logit mean of its streams, the pair is the logit mean of its lines
    'xgb_te_dual': [['q8_pool'], ['q8_noise']],
    'xgb_te_periodic': [['per_noise'], ['per_pool']],
    'xgb_te_deltas': [['dlt_noise_fw02_s42', 'dlt_noise_fw02_s7', 'dlt_noise_s11', 'dlt_noise_s13', 'dlt_noise_s42',
                       'dlt_noise_s7'], ['dlt_pool_s42', 'dlt_pool_s7']],
    'xgb_te_bpe': [['bpe_s42', 'bpe_s7']],
    'xgb_te_bpe_hier': [['bpe_hier_s42', 'bpe_hier_s7']],
    'xgb_te_bpe_hier_deltas': [['bpe_best_s42', 'bpe_best_s7']],
    'xgb_te_periodic_bpe': [['per_bpe_noise', 'per_bpe_pool']],
    'xgb_te_dual_bpe': [['q8_bpe_pool', 'q8_bpe_noise']],
}
RAW = ('base_', 'atom_')
STAT = ('rbf_', 'per_', 'per2_', 'rk_', 'cnt_')
MRG_COLS = ('te1_', 'te20_', 'teq1_', 'teq20_')


def lg7(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sg(z):
    return 1 / (1 + np.exp(-z))


def frame(A, names):
    return pd.DataFrame(A, columns=names)


def is_fine(n):
    """fine channels (exact income / commute values, narrow bins); their noise block is drawn first"""
    if n.startswith(RAW + STAT):
        return False
    if 'Annual_Income_USD' in n or 'Daily_Commute_km' in n:
        return True
    m = re.search(r'Inc(?:Q)?(\d+)', n)
    if m:
        return int(m.group(1)) <= 500
    m = re.search(r'Com(?:Q)?([\d.]+)', n)
    if m:
        return float(m.group(1)) <= 1.0
    m = re.search(r'logInc(\d+)', n)
    if m:
        return int(m.group(1)) >= 240
    return False


def ic_groups(names):
    """interaction constraints: one clique per source column; raw / atom / static columns join every clique"""
    def colkey(n):
        return ('Inc' if ('Annual_Income' in n or 'Inc' in n) else 'Com' if ('Commute' in n or 'Com' in n)
                else 'Age' if 'Age' in n else 'Sta' if 'Stations' in n else 'other')
    shared = [i for i, n in enumerate(names) if n.startswith(RAW + STAT)]
    groups = {}
    for i, n in enumerate(names):
        if i not in shared:
            groups.setdefault(colkey(n), []).append(i)
    return '[' + ','.join('[' + ','.join(str(j) for j in sorted(g + shared)) + ']' for g in groups.values()) + ']'


def delta_index(names, bpedelta):
    """income response geometry: (i) the exact income encoding minus every income bin channel of the same smoothing,
    (ii) differences between adjacent widths of each (family, smoothing) ladder, (iii) s = 1 minus s = 20 of the same token.
    Returns (I, J, delta names): the delta block is A[:, I] - A[:, J]."""
    idx = {n: i for i, n in enumerate(names)}
    exact = 'Annual_Income_USD'
    bpew = {'IncBpePre': 10000.5, 'IncBpePre2': 100.5}

    def kind(n):
        fam = n.split('_')[0]
        if fam == 'tex' and 'IncBpe' in n and bpedelta:
            t, s = n[4:].rsplit('_s', 1)
            return fam, 'a' if float(s) <= 5 else 'b', bpew.get(t, 1.0), 'bpe' + t
        if n.endswith('_' + exact) and fam in ('te1', 'te20', 'teav'):
            return fam, 'a' if fam in ('te1', 'teav') else 'b', 0.0, 'exact'
        m = re.search(r'_(IncQ|logInc|Inc)(\d+(?:\.\d+)?)(?:_|$)', n)
        if not m:
            return None
        typ = ('log' if m.group(1) == 'logInc' else 'lin') + 'inc'
        if fam == 'w7c':
            sm = 'a' if n.endswith('_s1') else 'b' if n.endswith('_s20') else None
        else:
            sm = ('a' if fam in ('teq1', 'teqav', 'w4a', 'w6a', 'w7a') else
                  'b' if fam in ('teq20', 'teqbv', 'w4b', 'w6b', 'w7b') else None)
        return fam, sm, float(m.group(2)), typ
    info = {n: kind(n) for n in names}
    info = {n: v for n, v in info.items() if v is not None and v[1] is not None}
    I, J, dn = [], [], []

    def add(a, b, tag):
        if a in idx and b in idx and a != b:
            I.append(idx[a])
            J.append(idx[b])
            dn.append(f'dlt_{tag}_{a}__{b}')
    for n, (fam, sm, w, typ) in info.items():
        if typ != 'exact':
            add(('te1_' if sm == 'a' else 'te20_') + exact, n, 'ex')
    groups = {}
    for n, (fam, sm, w, typ) in info.items():
        if typ != 'exact':
            groups.setdefault((fam, sm, typ), []).append((w, n))
    for g in groups.values():
        g = sorted(g)
        for (_, n1), (_, n2) in zip(g[:-1], g[1:]):
            add(n1, n2, 'adj')
    ab = [('te1', 'te20'), ('teq1', 'teq20'), ('teqav', 'teqbv'), ('w4a', 'w4b'), ('w6a', 'w6b'), ('w7a', 'w7b')]
    for n, (fam, sm, w, typ) in info.items():
        for fa, fb in ab:
            if fam == fa:
                add(n, fb + n[len(fa):], 'ab')
        if fam == 'w7c' and n.endswith('_s1'):
            add(n, n[:-3] + '_s20', 'ab')
        if fam == 'tex' and n.endswith('_s5'):
            add(n, n[:-3] + '_s20', 'ab')
    return np.array(I, dtype=int), np.array(J, dtype=int), dn


class Stream:
    def __init__(self, name):
        fams, par, self.seed = STREAMS[name]
        par = dict(par)
        self.es, self.rf, self.rounds = par.pop('es', 200), par.pop('rf', 1.11), par.pop('rounds', 8000)
        self.es_frac = par.pop('es_frac', 0.10)
        self.mrg, self.mrg_c = par.pop('mrg', 0.0), par.pop('mrg_C', 1.0)
        self.esval, self.tenoise, self.tn_reps = par.pop('esval', 0), par.pop('tenoise', 0.0), par.pop('tn_reps', 1)
        self.half, self.half_reps = par.pop('half', 0), par.pop('half_reps', 1)
        self.prune, self.ic, self.testall = par.pop('prune', 0), par.pop('ic', ''), par.pop('testall', 0)
        self.deltas, self.fw_delta, self.bpedelta = par.pop('deltas', 0), par.pop('fw_delta', 0.05), par.pop('bpedelta', 0)
        static = [s for s in par.pop('static', '').split(',') if s]
        self.extra = [e for e in par.pop('extra', '').split(',') if e]
        self.p = dict(XGB, **par, seed=self.seed)
        cols = json.load(open(L.FEATURES / 'cols.json'))
        self.sel = [i for i, c in enumerate(cols) if any(c.startswith(f + '_') for f in fams.split(','))]
        names = [cols[i] for i in self.sel]
        self.xs_tr = self.xs_te = None
        if static:
            cs = json.load(open(L.FEATURES / 'static_cols.json'))
            ssel = [i for i, c in enumerate(cs) if c.split('_')[0] in static]
            self.xs_tr = np.load(L.FEATURES / 'static_tr.npy')[:, ssel]
            self.xs_te = np.load(L.FEATURES / 'static_te.npy')[:, ssel]
            names += [cs[i] for i in ssel]
        self.names = names + self.extra
        self.mrg_cols = [i for i, n in enumerate(self.names) if n.startswith(MRG_COLS)]
        if self.deltas:
            di, dj, dn = delta_index(self.names, self.bpedelta)
            self.nmX = self.names + dn
            self.aug = lambda A: np.concatenate([A, A[:, di] - A[:, dj]], axis=1)
            self.fw = np.array([self.fw_delta if n.startswith('dlt_') else 1.0 for n in self.nmX], np.float32)
        else:
            self.nmX, self.aug, self.fw = self.names, (lambda A: A), None

    def load(self, k):
        a = np.load(L.FEATURES / f'X_f{k}_tr.npy')[:, self.sel]
        b = np.load(L.FEATURES / f'X_f{k}_te.npy')[:, self.sel]
        if self.xs_tr is not None:
            a, b = np.concatenate([a, self.xs_tr], 1), np.concatenate([b, self.xs_te], 1)
        if self.extra:                                   # placeholders, encoded per fold
            a = np.concatenate([a, np.zeros((len(a), len(self.extra)), np.float32)], 1)
            b = np.concatenate([b, np.zeros((len(b), len(self.extra)), np.float32)], 1)
        return a, b

    def margin(self, Z_fit, y_fit, Z_app):
        """LR decision function on standardised inputs (scaler fitted on Z_fit)"""
        sc = StandardScaler().fit(Z_fit)
        return self.lr().fit(sc.transform(Z_fit), y_fit).decision_function(sc.transform(Z_app))

    def lr(self):
        return LogisticRegression(max_iter=1000, C=self.mrg_c)

    def prune_keep(self, b):
        g = b.get_score(importance_type='gain')
        rank = sorted(self.nmX, key=lambda n: -g.get(n, 0.0))
        return sorted([self.nmX.index(n) for n in rank[:max(len(self.nmX) - self.prune, 8)]])

    def run(self, k):
        """(validation predictions of fold k, test predictions) of this stream's fold-k model(s)"""
        if self.half:
            return self.fold_pool(k)
        if self.tn_reps == 1:
            return self.fold_library(k, 0)
        zv = zt = 0.0
        for r in range(self.tn_reps):
            pv, pt = self.fold_library(k, r)
            zv, zt = zv + lg7(pv) / self.tn_reps, zt + lg7(pt) / self.tn_reps
        return sg(zv), sg(zt)

    def fold_library(self, k, rep):
        X, XT = self.load(k)
        names, nmX, aug = self.names, self.nmX, self.aug
        itr, iva = np.where(F != k)[0], np.where(F == k)[0]
        if self.extra:
            ne = len(self.extra)
            X[np.ix_(itr, range(X.shape[1] - ne, X.shape[1]))] = L.encode_inner(itr, self.extra)
            X[np.ix_(iva, range(X.shape[1] - ne, X.shape[1]))] = L.encode_val_like(itr, iva, self.extra)
            XT[:, -ne:] = L.encode_val_like(itr, np.arange(NTE), self.extra, test=True)
        Xtrn = X
        if self.tenoise:
            Xtrn = X.copy()
            rng = np.random.default_rng(self.seed * 7919 + k + 100003 * rep)
            fine = [i for i, n in enumerate(names) if is_fine(n)]
            coarse = [i for i, n in enumerate(names) if not n.startswith(RAW + STAT) and not is_fine(n)]
            for blk in (fine, coarse):
                if blk:
                    Xtrn[np.ix_(itr, blk)] += rng.normal(0, self.tenoise, (len(itr), len(blk))).astype(np.float32)
        rs = np.random.RandomState(1000 + k)
        sub = rs.permutation(len(itr))
        nc = int((1 - self.es_frac) * len(itr))
        i_fit, i_es = itr[sub[:nc]], itr[sub[nc:]]

        bm_tr = bm_va = bm_te = m_fit = m_es = None
        if self.mrg:                                     # LR margin: inner cross-fitted on outer-train
            Z = X[:, self.mrg_cols]
            sc = StandardScaler().fit(Z[itr])
            m_tr = np.zeros(len(itr))
            for ifit, iapp in StratifiedKFold(5, shuffle=True, random_state=777).split(itr, Y[itr]):
                lr = self.lr().fit(sc.transform(Z[itr[ifit]]), Y[itr[ifit]])
                m_tr[iapp] = lr.decision_function(sc.transform(Z[itr[iapp]]))
            lrf = self.lr().fit(sc.transform(Z[itr]), Y[itr])
            bm_tr = self.mrg * m_tr
            bm_va = self.mrg * lrf.decision_function(sc.transform(Z[iva]))
            bm_te = self.mrg * lrf.decision_function(sc.transform(XT[:, self.mrg_cols]))
            m_fit, m_es = bm_tr[sub[:nc]], bm_tr[sub[nc:]]

        X_es = Xtrn[i_es]
        if self.esval:                                   # early-stopping rows encoded from the fit rows only
            X_es = L.encode_val_like(i_fit, i_es, names, fallback=X[i_es])
            if self.mrg:
                m_es = self.mrg * self.margin(X[i_fit][:, self.mrg_cols], Y[i_fit], X_es[:, self.mrg_cols])
        pf = dict(self.p)
        if self.ic:
            pf['interaction_constraints'] = ic_groups(nmX)
        d1 = xgb.DMatrix(frame(aug(Xtrn[i_fit]), nmX), Y[i_fit])
        d2 = xgb.DMatrix(frame(aug(X_es), nmX), Y[i_es])
        for d, m in ((d1, m_fit), (d2, m_es)):
            if m is not None:
                d.set_base_margin(m)
            if self.fw is not None:
                d.set_info(feature_weights=self.fw)
        b = xgb.train(pf, d1, self.rounds, evals=[(d2, 'es')], early_stopping_rounds=self.es, verbose_eval=False)
        nb = max(int(b.best_iteration * self.rf), 30)

        keep = self.prune_keep(b) if self.prune else None
        cut = (lambda A: A[:, keep]) if keep is not None else (lambda A: A)
        nf = [nmX[i] for i in keep] if keep is not None else nmX
        pr = dict(self.p)
        if self.ic:
            pr['interaction_constraints'] = ic_groups(nf)
        dF = xgb.DMatrix(frame(cut(aug(Xtrn[itr])), nf), Y[itr])
        if bm_tr is not None:
            dF.set_base_margin(bm_tr)
        if self.fw is not None:
            dF.set_info(feature_weights=(self.fw[keep] if keep is not None else self.fw))
        bf = xgb.train(pr, dF, nb)

        dV = xgb.DMatrix(frame(cut(aug(X[iva])), nf))
        if bm_va is not None:
            dV.set_base_margin(bm_va)
        pv = bf.predict(dV)
        if self.testall:                                 # test rows encoded from all train labels
            XT = L.encode_val_like(np.arange(NTR), np.arange(NTE), names, fallback=XT, test=True)
            if self.mrg:
                bm_te = self.mrg * self.margin(X[itr][:, self.mrg_cols], Y[itr], XT[:, self.mrg_cols])
        dT = xgb.DMatrix(frame(cut(aug(XT)), nf))
        if bm_te is not None:
            dT.set_base_margin(bm_te)
        return pv, bf.predict(dT)

    def fold_pool(self, k):
        X, XT = self.load(k)                             # only the raw / atom / static columns are reused
        names, nmX, aug = self.names, self.nmX, self.aug
        itr, iva = np.where(F != k)[0], np.where(F == k)[0]
        rs = np.random.RandomState(5000 + 31 * k + self.seed)
        splits = []
        for _ in range(self.half_reps):
            perm = rs.permutation(len(itr))
            H1, H2 = itr[perm[:len(itr) // 2]], itr[perm[len(itr) // 2:]]
            splits += [(H1, H2), (H2, H1)]               # (model rows, encoding pool)
        zv, zt = np.zeros(len(iva)), np.zeros(NTE)
        for mrows, pool in splits:
            Xm = L.encode_val_like(pool, mrows, names, fallback=X[mrows])
            Xv = L.encode_val_like(pool, iva, names, fallback=X[iva])
            if self.mrg:                                 # LR margin, inner cross-fitted on the model rows
                Zm = Xm[:, self.mrg_cols]
                sc = StandardScaler().fit(Zm)
                m_m = np.zeros(len(mrows))
                for ifit, iapp in StratifiedKFold(5, shuffle=True, random_state=777).split(mrows, Y[mrows]):
                    lr = self.lr().fit(sc.transform(Zm[ifit]), Y[mrows[ifit]])
                    m_m[iapp] = lr.decision_function(sc.transform(Zm[iapp]))
                lrf = self.lr().fit(sc.transform(Zm), Y[mrows])
                bm_m, bm_v = self.mrg * m_m, self.mrg * lrf.decision_function(sc.transform(Xv[:, self.mrg_cols]))
            sub = rs.permutation(len(mrows))
            nc = int((1 - self.es_frac) * len(mrows))
            a_fit, a_es = sub[:nc], sub[nc:]
            XmA = aug(Xm)
            ph = dict(self.p)
            if self.ic:
                ph['interaction_constraints'] = ic_groups(nmX)
            d1 = xgb.DMatrix(XmA[a_fit], Y[mrows[a_fit]], feature_names=nmX)
            d2 = xgb.DMatrix(XmA[a_es], Y[mrows[a_es]], feature_names=nmX)
            if self.mrg:
                d1.set_base_margin(bm_m[a_fit])
                d2.set_base_margin(bm_m[a_es])
            if self.fw is not None:
                d1.set_info(feature_weights=self.fw)
            b = xgb.train(ph, d1, self.rounds, evals=[(d2, 'es')], early_stopping_rounds=self.es, verbose_eval=False)
            nb = max(int(b.best_iteration * self.rf), 30)
            keep, nf = None, nmX
            if self.prune:
                keep = self.prune_keep(b)
                nf = [nmX[i] for i in keep]
                ph = dict(self.p)
                if self.ic:
                    ph['interaction_constraints'] = ic_groups(nf)
            cut = (lambda A: A[:, keep]) if keep is not None else (lambda A: A)
            dF = xgb.DMatrix(cut(XmA), Y[mrows], feature_names=nf)
            if self.mrg:
                dF.set_base_margin(bm_m)
            if self.fw is not None:
                dF.set_info(feature_weights=(self.fw[keep] if keep is not None else self.fw))
            bf = xgb.train(ph, dF, nb)
            dV = xgb.DMatrix(cut(aug(Xv)), feature_names=nf)
            if self.mrg:
                dV.set_base_margin(bm_v)
            zv += lg7(bf.predict(dV)) / len(splits)
            Xt = L.encode_val_like(np.arange(NTR) if self.testall else pool, np.arange(NTE), names, fallback=XT, test=True)
            dT = xgb.DMatrix(cut(aug(Xt)), feature_names=nf)
            if self.mrg:
                dT.set_base_margin(self.mrg * lrf.decision_function(sc.transform(Xt[:, self.mrg_cols])))
            zt += lg7(bf.predict(dT)) / len(splits)
        return sg(zv), sg(zt)


def write_pair(pair):
    lines = PAIRS[pair]
    if not all(have_part(RUN_DIR, s, k) for line in lines for s in line for k in range(5)):
        print(f'{pair}: not all streams are done yet', flush=True)
        return
    zo = zt = 0.0
    for line in lines:
        lo = lt = 0.0
        for s in line:
            assemble(RUN_DIR, s, F, Y)                   # stream pair: runs/preds/<stream>_{oof,test}.npy
            o, t = (np.load(RUN_DIR / 'preds' / f'{s}_{x}.npy') for x in ('oof', 'test'))
            lo, lt = lo + lg7(o) / len(line), lt + lg7(t) / len(line)
        zo, zt = zo + lo / len(lines), zt + lt / len(lines)
    save_pair(HERE, pair, sg(zo), sg(zt))
    print(f'{pair}: OOF AUC {roc_auc_score(Y, zo):.6f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', nargs='?', default='all', choices=['all'] + list(PAIRS))
    ap.add_argument('--stream', choices=list(STREAMS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    pairs = list(PAIRS) if a.pair == 'all' else [a.pair]
    streams = [a.stream] if a.stream else list(dict.fromkeys(s for p in pairs for line in PAIRS[p] for s in line))
    for name in streams:
        st = None
        for k in fold_list(a.folds):
            if have_part(RUN_DIR, name, k):
                continue
            st = st or Stream(name)
            pv, pt = st.run(k)
            save_part(RUN_DIR, name, k, pv, pt)
            print(f'{name} fold {k}: AUC {roc_auc_score(Y[F == k], pv):.6f}', flush=True)
    if not a.stream:
        for p in pairs:
            write_pair(p)


if __name__ == '__main__':
    main()
