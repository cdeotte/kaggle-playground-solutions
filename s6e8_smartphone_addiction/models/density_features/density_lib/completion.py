"""Fold-wise exact-token response tables (label rates per token id), used by the completion channels.

For outer fold f the outer-train rows are split into 5 inner groups (StratifiedKFold(5, shuffle, seed 777)). Table g
(0-4) of fold f is fit on the outer-train rows outside inner group g, table 5 on all outer-train rows. Every row reads
the table that did not see its label: an outer-train row the table of its own group, validation and test rows table 5.
Rates are smoothed toward the table's prior with strengths 20 and 3."""
import numpy as np
from sklearn.model_selection import StratifiedKFold

SMS = (20.0, 3.0)


def lgt(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inner_groups(y, folds):
    """grp[i, f]: inner group of train row i in outer fold f (-1 if row i is a validation row of f);
    tab[i, f]: the table that row i reads in fold f (its group, or 5)."""
    grp = np.full((len(y), 5), -1, np.int8)
    for f in range(5):
        tid = np.where(folds != f)[0]
        inner = StratifiedKFold(5, shuffle=True, random_state=777)
        for g, (_, i_app) in enumerate(inner.split(np.zeros(len(tid)), y[tid])):
            grp[tid[i_app], f] = g
    tab = np.where(grp >= 0, grp, 5).astype(np.int64)
    return grp, tab


def response_table(idcol, V, y, folds, grp):
    """R[v, f, t, s]: label rate of token id v (train rows of one column, id 0 = [NA] is not counted) in table t of fold
    f at smoothing SMS[s]. Shape (V, 5, 6, 2), float32."""
    R = np.zeros((V, 5, 6, 2), np.float32)
    for f in range(5):
        base = (folds != f) & (idcol != 0)
        for t in range(6):
            m = base if t == 5 else base & (grp[:, f] != t)
            pri = y[folds != f].mean() if t == 5 else y[(folds != f) & (grp[:, f] != t)].mean()
            pos = np.bincount(idcol[m], weights=y[m], minlength=V)
            cnt = np.bincount(idcol[m], minlength=V)
            for si, sm in enumerate(SMS):
                R[:, f, t, si] = (pos + sm * pri) / (cnt + sm)
    return R


def row_tables(r, tab, n_tr):
    """(len(r), 5) table id per row of r (global row ids, test rows >= n_tr) and fold."""
    sel = np.full((len(r), 5), 5, np.int64)
    is_tr = r < n_tr
    sel[is_tr] = tab[r[is_tr]]
    return sel
