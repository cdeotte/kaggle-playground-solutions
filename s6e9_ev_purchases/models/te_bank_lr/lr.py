#!/usr/bin/env python
"""Logistic regression on the fold-local TE banks (cuML, GPU).

Per outer fold k: X = the recipe's design (design.py), standardized with outer-train mean / sd; GAM-style hinges
max(0, x - q) at the outer-train 20/40/60/80% quantiles of the columns matching spline_re (re-standardized on outer-train);
cuML LogisticRegression (qn solver, elastic net) fit on the outer-train rows; predict fold k and test.
OOF = the held-out predictions; test = mean of the 5 fold probabilities.
Recipes:
  te_bank_lr            banks S,P,I,T,P5,I5,T5,P1,I1,T1 + packs num, digits, footprint, replica + 168 acceptance columns
                        + income EM blocks; hinges on the singles' TEs, the rule score / logit and the EM blocks; C 0.02
  te_bank_lr_residual   formula-residual banks SR,PR,IR,TR + num pack; hinges on the residual singles and the rule; C 0.05
Usage: python lr.py te_bank_lr | te_bank_lr_residual"""
import re
import sys
from pathlib import Path

import cupy as cp
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from design import fold_design  # noqa: E402

RECIPES = {
    'te_bank_lr': dict(banks=['S', 'P', 'I', 'T', 'P5', 'I5', 'T5', 'P1', 'I1', 'T1'], packs=['num', 'digits', 'footprint', 'replica'],
                       acc=True, blocks=['income_em', 'income_info'], spline_re=r'^S:te(20|1):|^pack:num:rule|^block:income_', C=0.02),
    'te_bank_lr_residual': dict(banks=['SR', 'PR', 'IR', 'TR'], packs=['num'], acc=False, blocks=[],
                                spline_re=r'^SR:res20:|^pack:num:rule', C=0.05),
}
KNOTS, L1_RATIO, MAX_ITER, TOL = 4, 0.5, 3000, 1e-6


def to_gpu(A):
    """Chunked host -> device copy (cuML's own conversion of > 2 GB inputs fails)."""
    G = cp.empty(A.shape, dtype=cp.float32)
    step = max(1, int(2.5e8 // (A.shape[0] * 4)))
    for j in range(0, A.shape[1], step):
        G[:, j:j + step] = cp.asarray(np.ascontiguousarray(A[:, j:j + step]))
    return G


def main():
    name = sys.argv[1]
    rc = RECIPES[name]
    from cuml.linear_model import LogisticRegression
    y = target()
    f = folds(y)
    oof, test = np.full(len(y), np.nan), 0.0
    for k in range(5):
        ti, vi = np.flatnonzero(f != k), np.flatnonzero(f == k)
        X, names = fold_design(k, f, rc['banks'], rc['packs'], rc['acc'], rc['blocks'])
        mu, sd = X['train'].mean(0), X['train'].std(0) + 1e-6
        for r in X:
            X[r] = (X[r] - mu) / sd
        sp = [i for i, n in enumerate(names) if re.search(rc['spline_re'], n)]
        qq = [(j + 1) / (KNOTS + 1) for j in range(KNOTS)]
        qs = np.quantile(X['train'][:, sp], qq, axis=0)
        hin = {r: np.column_stack([np.maximum(0, X[r][:, sp] - qs[j]) for j in range(KNOTS)]) for r in X}
        hm, hs = hin['train'].mean(0), hin['train'].std(0) + 1e-6
        for r in X:
            X[r] = np.ascontiguousarray(np.column_stack([X[r], (hin[r] - hm) / hs]), dtype=np.float32)
        del hin
        m = LogisticRegression(penalty='elasticnet', C=rc['C'], l1_ratio=L1_RATIO, fit_intercept=True, max_iter=MAX_ITER, tol=TOL,
                               solver='qn', output_type='numpy', verbose=False)
        m.fit(to_gpu(X['train']), cp.asarray(y[ti].astype(np.float32)))
        oof[vi] = cp.asnumpy(cp.asarray(m.predict_proba(to_gpu(X['valid']))))[:, 1]
        test = test + cp.asnumpy(cp.asarray(m.predict_proba(to_gpu(X['test']))))[:, 1].astype(np.float64) / 5
        print(f'{name} fold {k}: {X["train"].shape[1]} columns, AUC {roc_auc_score(y[vi], oof[vi]):.6f}', flush=True)
        del X, m
    save_pair(HERE, name, oof, test)
    print(f'{name}: OOF AUC {roc_auc_score(y, oof):.6f}')


if __name__ == '__main__':
    main()
