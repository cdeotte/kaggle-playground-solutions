#!/usr/bin/env python
"""Predictor-only screen fills and region posterior for CatBoost (fold-independent, no labels). Writes
cache/fill_{tr,te}.parquet.

fl_<col>: for each of the 5 screen columns, an XGBoost regressor (700 trees, lr 0.07, depth 9, subsample / colsample 0.8,
min_child_weight 15, seed 1) fit on the observed rows of train + test from the other 11 columns; observed values are kept,
missing ones filled. Geometry on the fills: fl_slack, fl_d_m8, fl_s_m4, fl_wkd. rp_lo / rp_mid / rp_hi: a 3-class
XGBoost (800 trees) for the (daily, social) region, fit on the rows where it is determined, overwritten by the true
region there."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import FEATURES, NUMERICS, cache_dir, read_test, read_train  # noqa: E402

SCREEN = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "weekend_screen_time"]


def main():
    t0 = time.time()
    out = cache_dir("catboost_density")
    train, test = read_train(), read_test()
    both = pd.concat([train[list(FEATURES)], test[list(FEATURES)]], ignore_index=True)
    n_tr = len(train)
    cat_maps = {"gender": {"Male": 0, "Female": 1, "Other": 2},
                "stress_level": {"Low": 0, "Medium": 1, "High": 2},
                "academic_work_impact": {"No": 0, "Yes": 1}}
    XB = pd.DataFrame({c: both[c] for c in NUMERICS})
    for c, mp in cat_maps.items():
        XB[c] = both[c].map(mp)

    F = pd.DataFrame(index=both.index)
    for c in SCREEN:
        obs = both[c].notna().values
        feats = [f for f in XB.columns if f != c]
        m = xgb.XGBRegressor(n_estimators=700, learning_rate=0.07, max_depth=9, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=15, tree_method="hist", device="cuda", random_state=1, n_jobs=16,
                             verbosity=0)
        m.fit(XB.loc[obs, feats], both.loc[obs, c])
        pred = m.predict(XB[feats]).astype(np.float32)
        F[f"fl_{c}"] = np.where(obs, both[c].astype(np.float32), pred)
        print(f"fill {c} ({time.time() - t0:.0f}s)", flush=True)

    fd, fs = F["fl_daily_screen_time_hours"], F["fl_social_media_hours"]
    fg, fw = F["fl_gaming_hours"], F["fl_work_study_hours"]
    fwk = F["fl_weekend_screen_time"]
    F["fl_slack"] = (fd - (fs + fg + fw)).astype(np.float32)
    F["fl_d_m8"] = (fd - 8).astype(np.float32)
    F["fl_s_m4"] = (fs - 4).astype(np.float32)
    F["fl_wkd"] = (fwk - fd).astype(np.float32)

    d, s = both["daily_screen_time_hours"], both["social_media_hours"]
    hi = (d > 8) | (s > 4)
    lo = (d <= 6) & (s <= 4) & ~hi
    mid = d.notna() & s.notna() & ~hi & ~lo
    known = (hi | lo | mid).values
    lab = np.where(hi, 2, np.where(mid, 1, 0))
    mr = xgb.XGBClassifier(n_estimators=800, learning_rate=0.07, max_depth=9, subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=15, objective="multi:softprob", num_class=3, tree_method="hist",
                           device="cuda", random_state=1, n_jobs=16, verbosity=0)
    mr.fit(XB.loc[known], lab[known])
    P = mr.predict_proba(XB).astype(np.float32)
    P[known] = np.eye(3, dtype=np.float32)[lab[known]]
    F["rp_lo"], F["rp_mid"], F["rp_hi"] = P[:, 0], P[:, 1], P[:, 2]
    print(f"region posterior ({time.time() - t0:.0f}s)", flush=True)

    F.iloc[:n_tr].reset_index(drop=True).to_parquet(out / "fill_tr.parquet")
    F.iloc[n_tr:].reset_index(drop=True).to_parquet(out / "fill_te.parquet")
    print(f"saved ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
