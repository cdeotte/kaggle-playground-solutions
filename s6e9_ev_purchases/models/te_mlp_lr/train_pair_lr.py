#!/usr/bin/env python
"""Logistic regression on single + pair target encodings (pair_te.py) -> pair lr_pair_te.

Per outer fold, design = [z, z^2, z^3] with z = logit of the 215 single / pair TEs (keys with <= 50,000 levels), + the 6 inputs of
the purchase formula (income / 1000, env, subsidy, anxiety Medium, anxiety High, home charging), + 11 residual TEs (y - p_rule) of
the champion keys; standardized with the fit rows' mean / std (662 columns). cuML LogisticRegression (L2, C=0.1, max_iter 1000,
tol 1e-6) on the GPU. OOF = held-out folds; test = mean of the 5 fold models.
Usage: python train_pair_lr.py   (needs cuML / cupy)"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "lr_margin_xgb"))
sys.path.insert(0, str(HERE))
from common.pairs import save_pair  # noqa: E402
from lib.features import CHAMPION_TE, ResidTE, logit, rule_prob, setup  # noqa: E402
from pair_te import load_or_build, selected  # noqa: E402

NAME = "lr_pair_te"
C, POLY, MAX_ITER, TOL = 0.1, 3, 1000, 1e-6


def formula_frame(df):
    return np.column_stack([df.Annual_Income_USD.values / 1000.0, df.Environmental_Concern_Level.values,
                            (df.Subsidy_Available == "Yes").astype(float).values, (df.Range_Anxiety_Level == "Medium").astype(float).values,
                            (df.Range_Anxiety_Level == "High").astype(float).values,
                            (df.Home_Charging_Possible == "Yes").astype(float).values]).astype(np.float32)


def main():
    import cupy as cp
    from cuml.linear_model import LogisticRegression as CuLR
    ctx = setup()
    B, y, folds, ntr = ctx.B, ctx.y, ctx.folds, ctx.B.n_tr
    cache = load_or_build(ctx)
    te_folds = cache["folds"]
    sel = selected(cache)
    print(f"{len(sel)} of {len(cache['specs'])} keys used", flush=True)
    FF = np.vstack([formula_frame(ctx.tr), formula_frame(ctx.te)])
    RTE = ResidTE(B, y, y - rule_prob(ctx.both)[:ntr])

    def design(D, rows):
        z = logit(D.values[:, sel].astype(np.float64))
        parts = [z] + [z ** p for p in range(2, POLY + 1)]
        parts.append(FF[rows])
        return np.hstack(parts).astype(np.float32)

    t0 = time.time()
    oof, tpred = np.zeros(len(y)), np.zeros(B.n_te)
    test_rows = np.arange(ntr, ntr + B.n_te)
    for f, (a, b) in enumerate(folds):
        F, V, T = te_folds[f]
        Xa, Xb, Xt = design(F, a), design(V, b), design(T, test_rows)
        Rf, Rv, Rt = RTE.fold(a, b, CHAMPION_TE)
        Xa, Xb, Xt = np.hstack([Xa, Rf.values]), np.hstack([Xb, Rv.values]), np.hstack([Xt, Rt.values])
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        Xa, Xb, Xt = (Xa - mu) / sd, (Xb - mu) / sd, (Xt - mu) / sd
        lr = CuLR(C=C, max_iter=MAX_ITER, tol=TOL)
        lr.fit(cp.asarray(Xa), cp.asarray(y[a], dtype=cp.float32))
        oof[b] = cp.asnumpy(lr.predict_proba(cp.asarray(Xb))[:, 1])
        tpred += cp.asnumpy(lr.predict_proba(cp.asarray(Xt))[:, 1]) / 5
        print(f"  fold {f}: AUC {roc_auc_score(y[b], oof[b]):.6f}, {Xa.shape[1]} columns ({time.time() - t0:.0f}s)", flush=True)
    print(f"{NAME}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, NAME, oof, tpred)


if __name__ == "__main__":
    main()
