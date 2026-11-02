#!/usr/bin/env python
"""XGBoost on the champion matrix without a base margin: raw + log counts + flags + 11 nested TEs + 4 income TE differences
(45 features); slow parameters (lr 0.015, depth 6, colsample 0.3, lambda 5).
Two batches of 5 seeds (42-46 and 47-51). In each batch, pass 1 (first seed, early stopping) sets
rounds = int(1.1 * median(best_iteration + 1)) and pass 2 trains every seed with that fixed number of rounds.
OOF / test of a batch = mean over its seeds (test = mean of the 5 fold models); the pair is the mean of the two batches.
Usage: python train_te_diffs.py"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import save_pair  # noqa: E402
from lib.features import BLOCKS, CHAMPION_TE, add_diffs, assemble, setup  # noqa: E402
from lib.xgb_dress import SLOW  # noqa: E402

NAME = "xgb_te_diffs"
BATCHES = ([42, 43, 44, 45, 46], [47, 48, 49, 50, 51])
ROUND_MULT = 1.1


def run_batch(ctx, Xtr, Xte, te_folds, params, seeds):
    y, folds, n_te = ctx.y, ctx.folds, ctx.B.n_te

    def fold_data(f):
        a, b = folds[f]
        F, V, T = te_folds[f]
        return (pd.concat([Xtr.iloc[a].reset_index(drop=True), F], axis=1), pd.concat([Xtr.iloc[b].reset_index(drop=True), V], axis=1),
                pd.concat([Xte, T], axis=1))
    t0 = time.time()
    its = []
    for f, (a, b) in enumerate(folds):
        xa, xb, _ = fold_data(f)
        m = xgb.XGBClassifier(**{**params, "random_state": seeds[0]})
        m.fit(xa, y[a], eval_set=[(xb, y[b])], verbose=False)
        its.append(int(m.best_iteration) + 1)
    n_rounds = int(np.median(its) * ROUND_MULT)
    print(f"pass 1 best iterations {its} -> {n_rounds} rounds ({time.time() - t0:.0f}s)", flush=True)
    p2 = dict(params)
    p2.pop("early_stopping_rounds", None)
    p2["n_estimators"] = n_rounds
    oof, tpred = np.zeros(len(y)), np.zeros(n_te)
    for s in seeds:
        oof_s, t_s = np.zeros(len(y)), np.zeros(n_te)
        for f, (a, b) in enumerate(folds):
            xa, xb, xt = fold_data(f)
            m = xgb.XGBClassifier(**{**p2, "random_state": s})
            m.fit(xa, y[a], verbose=False)
            oof_s[b] = m.predict_proba(xb)[:, 1]
            t_s += m.predict_proba(xt)[:, 1] / 5
        print(f"  seed {s}: OOF AUC {roc_auc_score(y, oof_s):.6f} ({time.time() - t0:.0f}s)", flush=True)
        oof += oof_s / len(seeds)
        tpred += t_s / len(seeds)
    return oof, tpred


def main():
    ctx = setup()
    B = ctx.B
    X = assemble(B, BLOCKS)
    Xtr, Xte = B.train_part(X), B.test_part(X)
    te_folds = [tuple(add_diffs(D) for D in ctx.TE.fold(a, b, CHAMPION_TE)) for a, b in ctx.folds]
    oofs, tests = [], []
    for seeds in BATCHES:
        oof, test = run_batch(ctx, Xtr, Xte, te_folds, SLOW, seeds)
        print(f"batch {seeds}: OOF AUC {roc_auc_score(ctx.y, oof):.6f}", flush=True)
        oofs.append(oof)
        tests.append(test)
    oof, test = np.mean(oofs, axis=0), np.mean(tests, axis=0)
    print(f"{NAME}: OOF AUC {roc_auc_score(ctx.y, oof):.6f}", flush=True)
    save_pair(HERE, NAME, oof, test)


if __name__ == "__main__":
    main()
