#!/usr/bin/env python
"""RealMLP and TabM (pytabkit) on the nested target-encoding library of ../nested_te_xgb: one runner for the 3 pairs of
this folder.

Per outer fold k and net: inputs = the fold's library channels (MID: 170 columns, Q8: all 203) and, for the BPE streams,
the target encodings of the income's GPT-2 leading piece (s = 1, 20), encoded like the library (nested within outer-train,
full map for validation / test rows).  The 6 categorical base_ columns are passed as categoricals.  Gaussian noise
(sd 0.1, logit units) is added to every label channel of the outer-train rows.  A random 15% of the outer-train rows is the
epoch-selection holdout, re-encoded from the labels of the other 85% (validation-style); the nets train on the 85%.
RealMLP: pytabkit's refit after the epoch selection (n_refit=1); TabM: the selected checkpoint (n_refit=0).
reps nets per fold (seed + 100 r, new noise draw) are averaged in logit space; test = mean of the 5 fold models.
TabM streams are standardised per fold before the blend: z = (logit - mean) / sd with the mean / sd of the fold's
validation logits, the same map on that fold model's test logits.

Usage: python train.py all | <pair> [--folds 0,1,2,3,4]   runs the missing folds of every stream, then writes preds/<pair>
       python train.py --stream <stream> [--folds 2]       one stream (finished folds are kept in runs/parts/)"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / 'nested_te_xgb'))
from common.pairs import assemble, fold_list, have_part, save_pair, save_part  # noqa: E402
import te_library as L  # noqa: E402  (../nested_te_xgb/te_library.py)

RUN_DIR = HERE / 'runs'
Y, F, NTE = L.Y.astype(np.int64), L.FOLDS, L.NTE
ES_FRAC, NOISE_SD = 0.15, 0.1
CAT_BASE = ['base_Gender', 'base_City_Type', 'base_Current_Car_Type', 'base_Home_Charging_Possible',
            'base_Subsidy_Available', 'base_Range_Anxiety_Level']
REALMLP = dict(device='cuda', n_cv=1, n_refit=1, n_ens=8, n_epochs=32, batch_size=256, lr=0.01, val_metric_name='1-auc_ovr',
               val_fraction=ES_FRAC, num_emb_type='pblrd', plr_sigma=8.0, hidden_sizes=[768, 384, 192], act='mish', wd=0.017,
               ls_eps=0.005, add_front_scale=True, verbosity=0)
TABM = dict(device='cuda', n_cv=1, n_refit=0, tabm_k=32, num_emb_type='pwl', num_emb_n_bins=48, n_epochs=64, lr=1e-3,
            batch_size=512, val_metric_name='1-auc_ovr', val_fraction=ES_FRAC, verbosity=0)
SHORT = dict(plr_sigma=1.0, n_ens=8, n_epochs=6, lr=0.025, batch_size=512, act='gelu')
BPE2 = 'tex_IncBpePre_s1,tex_IncBpePre_s20'

STREAMS = {   # name: (net, library channel families, extra channels, net parameters, nets per fold, seed)
    'realmlp_mid_ens16_s42': ('realmlp', L.MID, '', dict(plr_sigma=1.0, n_ens=16), 2, 42),
    'realmlp_q8_s42': ('realmlp', L.Q8, '', dict(plr_sigma=1.0), 2, 42),
    'realmlp_mid_ens16_s7': ('realmlp', L.MID, '', dict(plr_sigma=1.0, n_ens=16), 2, 7),
    'realmlp_short_s42': ('realmlp', L.MID, '', SHORT, 4, 42),
    'realmlp_short_s7': ('realmlp', L.MID, '', SHORT, 4, 7),
    'tabm_s42': ('tabm', L.MID, '', dict(n_refit=0), 1, 42),
    'tabm_s7': ('tabm', L.MID, '', dict(n_refit=0), 1, 7),
    'realmlp_short_bpe_s42': ('realmlp', L.MID, BPE2, SHORT, 4, 42),
    'tabm_bpe_s42': ('tabm', L.MID, BPE2, dict(n_refit=0), 1, 42),
}
PAIRS = {     # pair: [(stream, weight, per-fold standardisation)], weighted logit blend
    'realmlp_te': [('realmlp_mid_ens16_s42', 1 / 3, False), ('realmlp_q8_s42', 1 / 3, False),
                   ('realmlp_mid_ens16_s7', 1 / 3, False)],
    'realmlp_tabm_te': [('realmlp_short_s42', 0.25, False), ('realmlp_short_s7', 0.25, False), ('tabm_s42', 0.25, True),
                        ('tabm_s7', 0.25, True)],
    'realmlp_tabm_te_bpe': [('realmlp_short_bpe_s42', 0.5, False), ('tabm_bpe_s42', 0.5, True)],
}


def lg7(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sg(z):
    return 1 / (1 + np.exp(-z))


def run_fold(stream, k):
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
    net, fams, extra, params, reps, seed = STREAMS[stream]
    cls, mp = (TabM_D_Classifier, dict(TABM, **params)) if net == 'tabm' else (RealMLP_TD_Classifier, dict(REALMLP, **params))
    cols = json.load(open(L.FEATURES / 'cols.json'))
    sel = [i for i, c in enumerate(cols) if any(c.startswith(f + '_') for f in fams.split(','))]
    extra = [e for e in extra.split(',') if e]
    names = [cols[i] for i in sel] + extra
    cat = [n for n in names if n in CAT_BASE]
    labcols = [i for i, n in enumerate(names) if not n.startswith(('base_', 'atom_'))]

    def frame(A):
        df = pd.DataFrame(A, columns=names)
        for c in cat:
            df[c] = df[c].astype(int).astype('category')
        return df
    Xtr = np.load(L.FEATURES / f'X_f{k}_tr.npy')[:, sel]
    Xte = np.load(L.FEATURES / f'X_f{k}_te.npy')[:, sel]
    ne = len(extra)
    if ne:                                               # placeholders, encoded below
        Xtr = np.concatenate([Xtr, np.zeros((len(Xtr), ne), np.float32)], 1)
        Xte = np.concatenate([Xte, np.zeros((len(Xte), ne), np.float32)], 1)
    itr, iva = np.where(F != k)[0], np.where(F == k)[0]
    rs = np.random.RandomState(1000 + k + 7 * seed)
    zv, zt = np.zeros(len(iva)), np.zeros(NTE)
    for rep in range(reps):
        Xt = Xtr.copy()
        if ne:
            Xt[itr, -ne:] = L.encode_inner(itr, extra)
            Xt[iva, -ne:] = L.encode_val_like(itr, iva, extra)
            Xte[:, -ne:] = L.encode_val_like(itr, np.arange(NTE), extra, test=True)
        rng = np.random.default_rng(seed * 7919 + k + 100003 * rep)
        Xt[np.ix_(itr, labcols)] += rng.normal(0, NOISE_SD, (len(itr), len(labcols))).astype(np.float32)
        sub = rs.permutation(len(itr))
        nc = int((1 - ES_FRAC) * len(itr))
        i_fit, i_es = itr[sub[:nc]], itr[sub[nc:]]
        X_es = L.encode_val_like(i_fit, i_es, names, fallback=Xtr[i_es])
        m = cls(random_state=seed + 100 * rep, **mp)
        m.fit(frame(Xt[i_fit]), Y[i_fit], X_val=frame(X_es), y_val=Y[i_es], cat_col_names=cat)
        zv += lg7(m.predict_proba(frame(Xt[iva]))[:, 1])
        zt += lg7(m.predict_proba(frame(Xte))[:, 1])
    return sg(zv / reps), sg(zt / reps)


def stream_preds(stream, stdz):
    """OOF and test probabilities of a stream (test = mean of the 5 fold models)"""
    if not stdz:
        assemble(RUN_DIR, stream, F, Y)
        return [np.load(RUN_DIR / 'preds' / f'{stream}_{x}.npy') for x in ('oof', 'test')]
    oof, zt = np.zeros(len(Y)), 0.0
    for k in range(5):
        pv, pt = (np.load(RUN_DIR / 'parts' / f'{stream}_f{k}_{x}.npy') for x in ('valid', 'test'))
        zv = lg7(pv)
        mu, sd = zv.mean(), zv.std()
        oof[F == k] = sg((zv - mu) / sd)
        zt = zt + (lg7(pt) - mu) / sd / 5
    return oof, sg(zt)


def write_pair(pair):
    parts = PAIRS[pair]
    if not all(have_part(RUN_DIR, s, k) for s, _, _ in parts for k in range(5)):
        print(f'{pair}: not all streams are done yet', flush=True)
        return
    zo = zt = 0.0
    for s, w, stdz in parts:
        o, t = stream_preds(s, stdz)
        zo, zt = zo + w * lg7(o), zt + w * lg7(t)
    save_pair(HERE, pair, sg(zo), sg(zt))
    print(f'{pair}: OOF AUC {roc_auc_score(Y, zo):.6f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', nargs='?', default='all', choices=['all'] + list(PAIRS))
    ap.add_argument('--stream', choices=list(STREAMS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    pairs = list(PAIRS) if a.pair == 'all' else [a.pair]
    streams = [a.stream] if a.stream else list(dict.fromkeys(s for p in pairs for s, _, _ in PAIRS[p]))
    for name in streams:
        for k in fold_list(a.folds):
            if have_part(RUN_DIR, name, k):
                continue
            pv, pt = run_fold(name, k)
            save_part(RUN_DIR, name, k, pv, pt)
            print(f'{name} fold {k}: AUC {roc_auc_score(Y[F == k], pv):.6f}', flush=True)
    if not a.stream:
        for p in pairs:
            write_pair(p)


if __name__ == '__main__':
    main()
