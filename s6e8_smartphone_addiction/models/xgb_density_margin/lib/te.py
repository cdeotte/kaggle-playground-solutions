"""Nested target encoding of string keys for one outer fold.

The outer-train rows are split into n_inner stratified inner folds (seed 777 unless given); each outer-train row gets the
smoothed mean (pos + s * prior) / (cnt + s) fit on the other inner folds. The outer-validation and test rows get the map
fit on all outer-train rows. Unseen keys get the outer-train prior."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TE_SMOOTH = 20.0


def _te_map(vals, yy, prior, smooth=TE_SMOOTH):
    g = pd.DataFrame({"v": vals, "y": yy}).groupby("v")["y"].agg(["mean", "size"])
    return (g["mean"] * g["size"] + prior * smooth) / (g["size"] + smooth)


def build_te_fold(y, trn_idx, keys, inner_seed=777, n_inner=5, smooth=TE_SMOOTH):
    """keys: list of (name, token Series over all train rows).
    Returns (te of the outer-train rows from the inner cross-fit, maps fit on all outer-train rows)."""
    prior = y[trn_idx].mean()
    te_trn = {name: np.full(len(trn_idx), np.nan, np.float32) for name, _ in keys}
    inner = StratifiedKFold(n_inner, shuffle=True, random_state=inner_seed)
    for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y[trn_idx]):
        fit_rows = trn_idx[i_fit]
        for name, tk in keys:
            mp = _te_map(tk.iloc[fit_rows].values, y[fit_rows], prior, smooth)
            te_trn[name][i_app] = pd.Series(tk.iloc[trn_idx[i_app]].values).map(mp).fillna(prior).values
    full_maps = {}
    for name, tk in keys:
        full_maps[name] = (_te_map(tk.iloc[trn_idx].values, y[trn_idx], prior, smooth), prior)
    return te_trn, full_maps


def apply_te(full_maps, keys):
    out = {}
    for name, tk in keys:
        mp, prior = full_maps[name]
        out[name] = tk.map(mp).fillna(prior).values.astype(np.float32)
    return out
