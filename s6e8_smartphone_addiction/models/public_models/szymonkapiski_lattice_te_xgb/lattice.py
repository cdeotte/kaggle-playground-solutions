# Code of the public notebook "S6E8 full lattice target encoding with XGBoost" (version 1), cells 3-9:
# https://www.kaggle.com/code/szymonkapiski/s6e8-full-lattice-target-encoding-with-xgboost
# Constants, constrained imputation, the feature / lattice-key builder (make_frames) and the fold-safe lattice
# target encoder (te_block). The data-path search was removed (train.py reads the data from common.data).
# notebook cell 3
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

N_SPLITS, SEED = 5, 42
TARGET, ID = "addicted_label", "id"
NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
       "work_study_hours", "sleep_hours", "notifications_per_day",
       "app_opens_per_day", "weekend_screen_time"]
CAT = ["gender", "stress_level", "academic_work_impact"]
COMP = ["social_media_hours", "gaming_hours", "work_study_hours"]
DAILY = "daily_screen_time_hours"

# Encode all 36 numeric pair cells, not the 4 hand-picked ones. This is the change that
# took XGBoost from 0.96749 to 0.96769.
WIDE_PAIRS = True
# Also emit pair cells at 0.1 resolution alongside the integer-floor ones. Finer cells are
# thinner and noisier, which is exactly what the CT_ count columns are there to handle.
# Worth another +0.00011 on XGBoost.
PAIR_RES = "r1"
# 3-way screen-block cells. Helps LightGBM a little, roughly neutral on XGBoost, left off
# here so this notebook matches the 0.96780 arrays exactly.
TRIPLES = False

# notebook cell 5
def constrained_impute(df):
    """Reconstruct missing screen-block values using the generator inequality.

    Returns a frame of imputed copies + bound-width columns. Target-free and row-wise, so
    it is leakage-safe and can be computed once on train+test together.
    """
    o = pd.DataFrame(index=df.index)
    d = df[DAILY]
    C = df[COMP]
    comp_sum_obs = C.sum(axis=1, skipna=True)
    n_comp_known = C.notna().sum(axis=1)

    # --- statistical backbone: iterative imputer over the correlated screen block ---
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    block = [DAILY, "weekend_screen_time", "social_media_hours", "gaming_hours",
             "work_study_hours", "sleep_hours"]
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=12, random_state=SEED)
    est = pd.DataFrame(imp.fit_transform(df[block]), columns=block, index=df.index)

    # --- CASE 1: daily missing -> hard LOWER bound from the observed components ---
    lo_daily = comp_sum_obs.where(n_comp_known > 0, 0.0)
    daily_imp = d.copy()
    need = d.isna()
    # clip the statistical estimate up to the feasible region
    daily_imp[need] = np.maximum(est[DAILY][need], lo_daily[need])
    o["daily_imp"] = daily_imp
    o["daily_lower_bound"] = lo_daily
    # how much the constraint actually moved the estimate (0 => constraint was slack)
    o["daily_bound_bind"] = np.where(need, np.maximum(0.0, lo_daily - est[DAILY]), 0.0)
    o["daily_was_missing"] = need.astype("int8")

    # --- CASE 2: one component missing, daily known -> TWO-SIDED bound [0, slack] ---
    for c in COMP:
        others = [x for x in COMP if x != c]
        others_sum = df[others].sum(axis=1, skipna=True)
        others_known = df[others].notna().all(axis=1)
        # slack is only a valid upper bound when daily and the OTHER components are known
        slack = (daily_imp - others_sum).where(others_known, np.nan)
        slack = slack.clip(lower=0.0)
        v = df[c].copy()
        m = v.isna()
        cand = est[c].clip(lower=0.0)
        # clip the estimate into [0, slack] where slack is available
        upper = slack.where(slack.notna(), np.inf)
        v[m] = np.minimum(np.maximum(cand[m], 0.0), upper[m])
        o[f"{c}_imp"] = v
        o[f"{c}_upper_bound"] = slack
        # width of the feasible interval = how certain this imputation is
        o[f"{c}_bound_width"] = np.where(m & slack.notna(), slack, np.nan)
        o[f"{c}_was_missing"] = m.astype("int8")

    # remaining block members: plain estimates
    for c in ["weekend_screen_time", "sleep_hours"]:
        o[f"{c}_imp"] = df[c].fillna(est[c])
        o[f"{c}_was_missing"] = df[c].isna().astype("int8")

    # --- consistency features computed on the RECONSTRUCTED values ---
    ci = o[[f"{c}_imp" for c in COMP]].sum(axis=1)
    o["other_screen_imp"] = o["daily_imp"] - ci
    o["comp_share_imp"] = ci / (o["daily_imp"] + 1e-6)
    o["identity_violation"] = np.clip(ci - o["daily_imp"], 0, None)  # should be ~0
    o["n_screen_missing"] = df[[DAILY] + COMP].isna().sum(axis=1).astype("int8")
    o["n_missing_all"] = df[NUM + CAT].isna().sum(axis=1).astype("int8")
    return o

# notebook cell 7
def make_frames(tr, te):
    """Numeric design matrix + the string lattice keys that get target-encoded."""
    ntr = len(tr)
    full = pd.concat([tr.drop(columns=[TARGET]), te], axis=0, ignore_index=True)

    imp = constrained_impute(full)

    X = pd.concat([full[NUM].astype("float32"), imp.astype("float32")], axis=1)
    for c in CAT:
        X[c] = pd.Categorical(full[c].astype(str)).codes.astype("float32")

    # domain relations that measured positive earlier
    X["leisure_hours"] = full["social_media_hours"] + full["gaming_hours"]
    X["weekend_minus_daily"] = full["weekend_screen_time"] - full[DAILY]
    X["notif_per_open"] = full["notifications_per_day"] / (full["app_opens_per_day"] + 1.0)

    # --- the lattice keys: every predictor at FULL resolution, plus coarsenings ---
    keys = pd.DataFrame(index=full.index)
    for c in NUM + CAT:
        keys[c] = full[c].astype(str)
    # coarsenings give denser cells where the full-resolution one is too thin to trust
    for c in NUM:
        v = pd.to_numeric(full[c], errors="coerce")
        keys[f"{c}_r1"] = v.round(1).astype(str)
        keys[f"{c}_fl"] = np.floor(v).astype("Int32").astype(str)
    # pairs: 2-D lattice cells, where the generator's joint rounding shows up
    PAIRS = [(DAILY, "weekend_screen_time"), ("social_media_hours", "gaming_hours"),
             (DAILY, "social_media_hours"), ("stress_level", "academic_work_impact")]
    if WIDE_PAIRS:
        # NEW MECHANISM: encode EVERY numeric pair, not the 4 hand-picked
        # ones. The full-lattice TE that paid +0.001 was applied to single columns only;
        # the pair lattice was never swept, so a joint rounding cell the generator
        # actually uses may simply be absent from the current key set. This is the same
        # mechanism that already paid, applied at the interaction order it was never
        # tried at -- distinct from tuning smoothing (measured 0.00000) or adding another
        # GBDT on the same keys (also 0.00000).
        allp = [(a, b) for i, a in enumerate(NUM) for b in NUM[i + 1:]]
        PAIRS = sorted(set(PAIRS) | set(allp))
    def part(c, res="floor"):
        if c not in NUM:
            return full[c].astype(str)
        v = pd.to_numeric(full[c], errors="coerce")
        if res == "r1":
            return v.round(1).astype(str)
        return np.floor(v).astype("Int32").astype(str)

    for l, r in PAIRS:
        keys[f"P_{l}__{r}"] = part(l) + "__" + part(r)

    if PAIR_RES == "r1":
        # FINER pair cells (0.1 resolution instead of integer floor). The single-column
        # lattice pays BECAUSE it is full resolution; the pair lattice was only ever built
        # at floor resolution, so the joint cell may be too coarse to see the generator's
        # actual joint rounding. Thinner cells are noisier, which is why the CT_ count
        # column exists -- the model can discount them.
        for l, r in PAIRS:
            keys[f"R_{l}__{r}"] = part(l, "r1") + "__" + part(r, "r1")

    if TRIPLES:
        # 3-way cells over the screen block only. The generator identity binds daily to
        # its three components jointly, so a 3-way cell is the lowest order at which that
        # relation appears as a single lattice key rather than being reconstructed from
        # marginals. Restricted to the screen block: all-numeric triples would be 84 keys
        # of mostly-empty cells.
        trip = [(DAILY, "social_media_hours", "gaming_hours"),
                (DAILY, "social_media_hours", "work_study_hours"),
                (DAILY, "gaming_hours", "work_study_hours"),
                ("social_media_hours", "gaming_hours", "work_study_hours"),
                (DAILY, "weekend_screen_time", "sleep_hours")]
        for a3, b3, c3 in trip:
            keys[f"T_{a3}__{b3}__{c3}"] = part(a3) + "__" + part(b3) + "__" + part(c3)

    X = X.replace([np.inf, -np.inf], np.nan)
    return (X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True),
            keys.iloc[:ntr].reset_index(drop=True), keys.iloc[ntr:].reset_index(drop=True))

# notebook cell 9
def te_block(Ktr, y_tr, Kva, Kte, smooth=20.0):
    """Fold-safe smoothed TE + frequency for every key column.

    Train part uses an inner 4-fold OOF so a row never encodes itself; valid/test use the
    map fit on the whole outer training part. Emits the cell COUNT alongside the mean so
    the model can discount thin, noisy lattice cells instead of trusting them equally.
    """
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 101)
    gm = float(y_tr.mean())
    otr, ova, ote = {}, {}, {}
    splits = list(inner.split(Ktr, y_tr))
    for c in Ktr.columns:
        col = Ktr[c].to_numpy()
        oof_te = np.full(len(Ktr), gm, dtype="float32")
        oof_ct = np.zeros(len(Ktr), dtype="float32")
        for fi, hi in splits:
            g = pd.DataFrame({"k": col[fi], "y": y_tr.to_numpy()[fi]}) \
                .groupby("k").y.agg(["sum", "count"])
            mp = (g["sum"] + smooth * gm) / (g["count"] + smooth)
            s = pd.Series(col[hi])
            oof_te[hi] = s.map(mp).fillna(gm).to_numpy()
            oof_ct[hi] = s.map(g["count"]).fillna(0.0).to_numpy()
        g = pd.DataFrame({"k": col, "y": y_tr.to_numpy()}).groupby("k").y.agg(["sum", "count"])
        mp = (g["sum"] + smooth * gm) / (g["count"] + smooth)
        otr[f"TE_{c}"] = oof_te
        otr[f"CT_{c}"] = oof_ct
        ova[f"TE_{c}"] = Kva[c].map(mp).fillna(gm).to_numpy("float32")
        ova[f"CT_{c}"] = Kva[c].map(g["count"]).fillna(0.0).to_numpy("float32")
        ote[f"TE_{c}"] = Kte[c].map(mp).fillna(gm).to_numpy("float32")
        ote[f"CT_{c}"] = Kte[c].map(g["count"]).fillna(0.0).to_numpy("float32")
    return (pd.DataFrame(otr, index=Ktr.index), pd.DataFrame(ova, index=Kva.index),
            pd.DataFrame(ote, index=Kte.index))
