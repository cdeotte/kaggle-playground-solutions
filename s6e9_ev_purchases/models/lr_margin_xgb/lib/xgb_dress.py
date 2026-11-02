"""XGBoost parameters and the two-pass fixed-round protocol with a base margin.

Pass 1: each fold is fitted with the first seed and early stopping; rounds = int(1.1 * median best iteration over the folds).
Pass 2: every seed fits every fold with that fixed number of rounds. OOF = the fold models on their held-out rows; test = mean of
the 5 fold models; both averaged over the seeds."""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

DEFAULT = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda", learning_rate=0.03, max_depth=7,
               min_child_weight=5, subsample=0.7, colsample_bytree=0.5, max_bin=1024, n_estimators=8000, early_stopping_rounds=200,
               random_state=42, nthread=6)
FAST = {**DEFAULT, "colsample_bytree": 0.3}
SLOW = {**DEFAULT, "learning_rate": 0.015, "colsample_bytree": 0.3, "reg_lambda": 5.0, "max_depth": 6, "early_stopping_rounds": 300}


def fold_matrices(ctx, base, f, extra_tr=None, extra_te=None):
    """(xa, xb, xt) for outer fold f: champion matrix + TE frame (+ diffs) [+ residual TEs] [+ extra label-free columns]."""
    a, b = ctx.folds[f]
    Fd, Vd, Td = base["te_folds"][f]
    Xtr, Xte = base["Xtr"], base["Xte"]
    xa = pd.concat([Xtr.iloc[a].reset_index(drop=True), Fd], axis=1)
    xb = pd.concat([Xtr.iloc[b].reset_index(drop=True), Vd], axis=1)
    xt = pd.concat([Xte, Td], axis=1)
    if base.get("r_folds") is not None:
        Rf, Rv, Rt = base["r_folds"][f]
        xa, xb, xt = pd.concat([xa, Rf], axis=1), pd.concat([xb, Rv], axis=1), pd.concat([xt, Rt], axis=1)
    if extra_tr is not None:
        xa = pd.concat([xa, extra_tr.iloc[a].reset_index(drop=True)], axis=1)
        xb = pd.concat([xb, extra_tr.iloc[b].reset_index(drop=True)], axis=1)
        xt = pd.concat([xt, extra_te.reset_index(drop=True)], axis=1)
    return xa, xb, xt


def dress(ctx, name, base, seeds, params=FAST, extra_tr=None, extra_te=None):
    """base: dict(Xtr, Xte, te_folds, margins[, r_folds]). Returns the seed-averaged (oof, test) probabilities."""
    y, folds, n_te = ctx.y, ctx.folds, ctx.B.n_te
    t0 = time.time()
    its = []
    for f, (a, b) in enumerate(folds):
        xa, xb, _ = fold_matrices(ctx, base, f, extra_tr, extra_te)
        Ma, Mb, Mt = base["margins"][f]
        m = xgb.XGBClassifier(**{**params, "random_state": seeds[0]})
        m.fit(xa, y[a], eval_set=[(xb, y[b])], base_margin=Ma, base_margin_eval_set=[Mb], verbose=False)
        its.append(int(m.best_iteration))
    n_rounds = int(np.median(its) * 1.1)
    print(f"{name}: pass 1 best iterations {its} -> {n_rounds} rounds ({time.time() - t0:.0f}s)", flush=True)
    p2 = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    p2["n_estimators"] = n_rounds
    oof, tpred = np.zeros(len(y)), np.zeros(n_te)
    for s in seeds:
        oof_s, t_s = np.zeros(len(y)), np.zeros(n_te)
        for f, (a, b) in enumerate(folds):
            xa, xb, xt = fold_matrices(ctx, base, f, extra_tr, extra_te)
            Ma, Mb, Mt = base["margins"][f]
            m = xgb.XGBClassifier(**{**p2, "random_state": s})
            m.fit(xa, y[a], base_margin=Ma, verbose=False)
            oof_s[b] = m.predict_proba(xb, base_margin=Mb)[:, 1]
            t_s += m.predict_proba(xt, base_margin=Mt)[:, 1] / 5
        print(f"  seed {s}: OOF AUC {roc_auc_score(y, oof_s):.6f} ({time.time() - t0:.0f}s)", flush=True)
        oof += oof_s / len(seeds)
        tpred += t_s / len(seeds)
    print(f"{name}: {len(seeds)}-seed OOF AUC {roc_auc_score(y, oof):.6f}, {xa.shape[1]} features, {n_rounds} rounds", flush=True)
    return oof, tpred
