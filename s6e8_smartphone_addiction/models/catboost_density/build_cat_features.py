#!/usr/bin/env python
"""Target-free CatBoost inputs (fold-independent): categorical string tokens for CatBoost's ordered CTRs and a numeric
block. Writes cache/cat_{tr,te}.parquet and cache/num_{tr,te}.parquet.

Categoricals (27): tk_<col> exact %.2f tokens of the 12 columns (NaN -> "NA"); t1_<col> 0.1-rounded tokens of the 6 hour
columns; reg = (daily, social) region (0: daily <= 6 and social <= 4; 2: daily > 8 or social > 4; 1: otherwise; m:
undetermined); pair tokens p_ds, p_dw (daily x social, daily x weekend at 0.1), p_na (notifications x app opens), p_gw
(gaming x work at 0.1); region crossings r_notif, r_app, r_age, r_gen.
Numerics: raw 9 + 3 categorical codes, 12 missing flags + count, budget geometry; and original-data channels (kernel-
smoothed original LLR at bandwidths 0.1 / 0.3, class-conditional original CDF gap, synthetic count and synthetic /
original frequency log ratio) that the runner leaves out."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, FEATURES, NUMERICS, TARGET, cache_dir, read_original, read_test, read_train  # noqa: E402

NUMS, FEATS = list(NUMERICS), list(FEATURES)


def tokx(s, r=2):
    if s.dtype == object:
        return s.astype(object).where(s.notna(), "NA").astype(str)
    fmt = f"{{:.{r}f}}"
    return s.round(r).map(lambda x: fmt.format(x) if pd.notna(x) else "NA")


def main():
    t0 = time.time()
    out = cache_dir("catboost_density")
    train, test = read_train(), read_test()
    orig = read_original()
    for c in NUMS + [TARGET]:
        orig[c] = pd.to_numeric(orig[c], errors="coerce")
    n_tr = len(train)
    both = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)

    C = pd.DataFrame(index=both.index)   # categorical frame (strings)
    N = pd.DataFrame(index=both.index)   # numeric frame (float32)

    for c in FEATS:
        C[f"tk_{c}"] = tokx(both[c])
    for c in ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time", "gaming_hours", "work_study_hours",
              "sleep_hours"]:
        C[f"t1_{c}"] = tokx(both[c], 1)
    d = both["daily_screen_time_hours"]
    s = both["social_media_hours"]
    hi = (d > 8) | (s > 4)
    lo = (d <= 6) & (s <= 4) & ~hi
    mid = d.notna() & s.notna() & ~hi & ~lo
    reg = pd.Series("m", index=both.index)
    reg[hi] = "2"
    reg[lo] = "0"
    reg[mid] = "1"
    C["reg"] = reg
    # explicit crossings (oblivious trees cannot chain them)
    C["p_ds"] = C["t1_daily_screen_time_hours"] + "|" + C["t1_social_media_hours"]
    C["p_dw"] = C["t1_daily_screen_time_hours"] + "|" + C["t1_weekend_screen_time"]
    C["p_na"] = C["tk_notifications_per_day"] + "|" + C["tk_app_opens_per_day"]
    C["p_gw"] = C["t1_gaming_hours"] + "|" + C["t1_work_study_hours"]
    C["r_notif"] = reg + "|" + C["tk_notifications_per_day"]
    C["r_app"] = reg + "|" + C["tk_app_opens_per_day"]
    C["r_age"] = reg + "|" + C["tk_age"]
    C["r_gen"] = reg + "|" + C["tk_gender"]
    print(f"categoricals: {C.shape[1]} columns ({time.time() - t0:.0f}s)", flush=True)

    for c in NUMS:
        N[c] = both[c].astype(np.float32)
    cat_maps = {"gender": {"Male": 0, "Female": 1, "Other": 2},
                "stress_level": {"Low": 0, "Medium": 1, "High": 2},
                "academic_work_impact": {"No": 0, "Yes": 1}}
    for c in CATEGORICALS:
        N[f"n_{c}"] = both[c].map(cat_maps[c]).astype(np.float32)
    M = both[FEATS].isna()
    for c in FEATS:
        N[f"na_{c}"] = M[c].astype(np.float32)
    N["n_missing"] = M.sum(1).astype(np.float32)
    sgw = both[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(1, min_count=3)
    N["sgw"] = sgw.astype(np.float32)
    N["slack"] = (d - sgw).astype(np.float32)
    N["slack_frac"] = ((d - sgw) / d.clip(lower=.1)).astype(np.float32)
    for c in ["social_media_hours", "gaming_hours", "work_study_hours"]:
        N[f"share_{c}"] = (both[c] / d.clip(lower=.1)).astype(np.float32)
    wk = both["weekend_screen_time"]
    N["wk_minus_d"] = (wk - d).astype(np.float32)
    N["wk_minus_sgw"] = (wk - sgw).astype(np.float32)
    N["d_m6"] = (d - 6).astype(np.float32)
    N["d_m8"] = (d - 8).astype(np.float32)
    N["s_m4"] = (s - 4).astype(np.float32)

    # original-data channels
    prior = orig[TARGET].mean()
    for c, bws in [("daily_screen_time_hours", (0.1, 0.3)), ("weekend_screen_time", (0.1, 0.3)),
                   ("social_media_hours", (0.1, 0.3)), ("gaming_hours", (0.3,)), ("work_study_hours", (0.3,)),
                   ("sleep_hours", (0.3,))]:
        ov = orig[c].dropna().values
        oy = orig.loc[orig[c].notna(), TARGET].values
        v = both[c].values.astype(np.float64)
        grid = np.unique(np.round(v[~np.isnan(v)], 2))
        for bw in bws:   # Gaussian-kernel smoothed original LLR on the value grid
            d2 = (grid[:, None] - ov[None, :]) / bw
            w = np.exp(-0.5 * d2 ** 2)
            s1 = (w * oy[None, :]).sum(1)
            s0 = (w * (1 - oy[None, :])).sum(1)
            k = 3.0
            p = (s1 + prior * k) / (s1 + s0 + k)
            mp = dict(zip(grid, np.log(p / (1 - p))))
            N[f"kllr{int(bw * 10)}_{c}"] = pd.Series(np.round(v, 2)).map(mp).astype(np.float32).values
    for c in NUMS:   # class-conditional original CDF gap
        o1 = np.sort(orig.loc[orig[TARGET] == 1, c].dropna().values)
        o0 = np.sort(orig.loc[orig[TARGET] == 0, c].dropna().values)
        v = both[c].values.astype(np.float64)
        g = np.searchsorted(o1, v, side="right") / len(o1) - np.searchsorted(o0, v, side="right") / len(o0)
        g[np.isnan(v)] = np.nan
        N[f"cdfgap_{c}"] = g.astype(np.float32)
    for c in NUMS:   # synthetic count and synthetic / original frequency log ratio
        sv, ovv = tokx(both[c]), tokx(orig[c])
        scnt = sv[sv != "NA"].value_counts()
        ocnt = ovv[ovv != "NA"].value_counts(normalize=True)
        N[f"cnt_{c}"] = sv.map(scnt).astype(np.float32)
        N.loc[sv == "NA", f"cnt_{c}"] = np.nan
        ratio = np.log((sv.map(scnt) / scnt.sum()) / sv.map(ocnt))
        N[f"amp_{c}"] = ratio.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    kcols = [c for c in N.columns if c.startswith("kllr")]
    N["kllr_sum"] = N[kcols].sum(1, min_count=1).astype(np.float32)
    print(f"numerics: {N.shape[1]} columns ({time.time() - t0:.0f}s)", flush=True)

    C.iloc[:n_tr].reset_index(drop=True).to_parquet(out / "cat_tr.parquet")
    C.iloc[n_tr:].reset_index(drop=True).to_parquet(out / "cat_te.parquet")
    N.iloc[:n_tr].reset_index(drop=True).to_parquet(out / "num_tr.parquet")
    N.iloc[n_tr:].reset_index(drop=True).to_parquet(out / "num_te.parquet")
    print(f"saved ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
