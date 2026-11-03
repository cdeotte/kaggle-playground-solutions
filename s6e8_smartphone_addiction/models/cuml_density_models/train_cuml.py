#!/usr/bin/env python
"""cuML SVC, random-Fourier-feature logistic regression and random forest on the per-fold density matrix (lib/assemble.py).

Pairs:
  cuml_svc_rbf_density        cuml.svm.SVC(rbf, C 0.5, gamma 0.002) on all outer-train rows, 213 columns
  cuml_svc_rbf_density_rich   the same SVC on the 346-column matrix (213 + the per-fold channels)
  cuml_rff_logreg_density     cuML LogisticRegression(C 3, max_iter 1000) on 4096 CuPy random Fourier features
                              (gamma 0.002, seed 42), 213 columns
  cuml_rf_density_rich        cuml RandomForestClassifier(300 trees, max_depth 20, max_features 0.3, min_samples_leaf 20,
                              n_bins 128, bootstrap, random_state 42), 346 columns
Matrix: NaN -> outer-train median; standardised with the outer-train mean / sd (not for the forest); inf / NaN -> 0.
Each fold model is fitted on the outer-train rows and scores its validation rows and the test rows (SVC: decision
function, the others: probability); one seed. When all 5 folds are done, lib/calibrate.py turns the fold outputs into
probabilities (per-fold ranks -> cross-fitted isotonic) and writes preds/<pair>_{oof,test}.npy.
Usage: python train_cuml.py <pair> [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import fold_list, have_part, save_part  # noqa: E402
from lib.assemble import assemble  # noqa: E402
from lib.calibrate import calibrate  # noqa: E402

SVC_PAR = dict(nsub=600000, C=0.5, gamma=0.002)   # nsub >= outer-train size: all outer-train rows
RECIPES = {
    'cuml_svc_rbf_density': dict(model='svc', rich=False, par=SVC_PAR),
    'cuml_svc_rbf_density_rich': dict(model='svc', rich=True, par=SVC_PAR),
    'cuml_rff_logreg_density': dict(model='rff', rich=False, par=dict(D=4096, gamma=0.002, C=3.0)),
    'cuml_rf_density_rich': dict(model='rf', rich=True, par=dict(n=300, depth=20, mf=0.3, msl=20)),
}
t0 = time.time()


def log(m):
    print(f'[{time.time() - t0:7.1f}s] {m}', flush=True)


def run_fold(pair, fold):
    import cupy as cp
    r = RECIPES[pair]
    model, PAR = r['model'], r['par']
    D = assemble(fold, rich=r['rich'], log=log)
    A, B, y, trn_idx, val_idx = D['A'], D['B'], D['y'], D['trn_idx'], D['val_idx']

    # NaN impute (outer-train median), then standardise (outer-train statistics; trees are not scaled)
    med = A.iloc[trn_idx].median()
    A = A.fillna(med).astype(np.float32)
    B = B.fillna(med).astype(np.float32)
    if model != 'rf':
        mu = A.iloc[trn_idx].mean()
        sd = A.iloc[trn_idx].std().replace(0, 1.0)
        A = ((A - mu) / sd).astype(np.float32)
        B = ((B - mu) / sd).astype(np.float32)
    A = A.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    B = B.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    Xtr = cp.asarray(A.values[trn_idx])
    Xva = cp.asarray(A.values[val_idx])
    Xte = cp.asarray(B.values)
    ytr = cp.asarray(y[trn_idx].astype(np.float32))
    log(f'matrices Xtr{Xtr.shape} Xva{Xva.shape} Xte{Xte.shape} model={model} par={PAR}')

    if model == 'rf':
        from cuml.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=int(PAR['n']), max_depth=int(PAR['depth']), max_features=PAR['mf'],
                                   min_samples_leaf=int(PAR['msl']), n_bins=128, bootstrap=True, random_state=42,
                                   n_streams=1)
        m.fit(Xtr, ytr.astype(cp.int32))
        pv = m.predict_proba(Xva)[:, 1]
        pt = m.predict_proba(Xte)[:, 1]

    elif model == 'rff':
        # linear-time kernel machine: random Fourier features of an RBF kernel + logistic regression
        from cuml.linear_model import LogisticRegression
        Dm, gamma = int(PAR['D']), float(PAR['gamma'])
        rng = cp.random.RandomState(42)
        W = rng.normal(0, np.sqrt(2 * gamma), size=(Xtr.shape[1], Dm)).astype(cp.float32)
        b = rng.uniform(0, 2 * np.pi, size=Dm).astype(cp.float32)
        sc = np.float32(np.sqrt(2.0 / Dm))

        def rff(X, chunk=100000):
            out = cp.empty((X.shape[0], Dm), cp.float32)
            for i in range(0, X.shape[0], chunk):
                out[i:i + chunk] = sc * cp.cos(X[i:i + chunk] @ W + b)
            return out
        Ztr = rff(Xtr)
        m = LogisticRegression(C=float(PAR['C']), max_iter=1000)
        m.fit(Ztr, ytr)
        del Ztr
        cp.get_default_memory_pool().free_all_blocks()
        pv = m.predict_proba(rff(Xva))[:, 1]
        pt = m.predict_proba(rff(Xte))[:, 1]

    elif model == 'svc':
        from cuml.svm import SVC
        rng = np.random.default_rng(42)
        sub = rng.choice(len(trn_idx), min(int(PAR['nsub']), len(trn_idx)), replace=False)
        Xs, ys = Xtr[cp.asarray(sub)], ytr[cp.asarray(sub)]
        m = SVC(kernel='rbf', C=float(PAR['C']), gamma=PAR['gamma'], cache_size=4000, tol=1e-3)
        m.fit(Xs, ys)
        log(f'  SVC fitted on {len(sub)} rows, n_support={getattr(m, "n_support_", "?")}')
        pv = cp.asnumpy(m.decision_function(Xva)).ravel()
        pt = cp.asnumpy(m.decision_function(Xte)).ravel()
    else:
        raise SystemExit(f'unknown model {model}')

    pv = np.asarray(cp.asnumpy(pv) if isinstance(pv, cp.ndarray) else pv, np.float64).ravel()
    pt = np.asarray(cp.asnumpy(pt) if isinstance(pt, cp.ndarray) else pt, np.float64).ravel()
    log(f'{pair} fold {fold}: VAL_AUC={roc_auc_score(y[val_idx], pv):.6f}')
    del m, Xtr, Xva, Xte, ytr
    cp.get_default_memory_pool().free_all_blocks()
    return pv, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    for k in fold_list(a.folds):
        if have_part(HERE, a.pair, k):
            continue
        pv, pt = run_fold(a.pair, k)
        save_part(HERE, a.pair, k, pv, pt)
    if all(have_part(HERE, a.pair, k) for k in range(5)):
        calibrate(HERE, a.pair)
    else:
        print(f'{a.pair}: not all folds are done yet', flush=True)


if __name__ == '__main__':
    main()
