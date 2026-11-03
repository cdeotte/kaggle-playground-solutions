"""Base channel matrix of one outer fold: the static target-free block plus the label-derived nested target encodings,
WOE sums and the WOE base margin.

Static inputs (built by ../density_features): static_{tr,te}.parquet (raw 12 columns, missing flags, budget geometry,
exact-value counts, original-data LLR / amplification, 18 pair PMIs, predictor-only screen imputers and the 3-class
region posterior) and tok_{tr,te}.parquet (exact %.2f value tokens, NaN -> "__NA__").
Label-derived parts are cross-fitted inside the outer-train rows (5 inner folds, seed 777); the outer-validation and test
rows get the maps fit on all outer-train rows.

cfg flags (the base block = raw 12 + missing flags + budget geometry is always in):
  cnt    exact-value counts                        orig   original-data LLR / amplification channels
  pmi    18 pair PMIs                              imp    imputer fills, filled geometry, region posterior
  lat    fractional part and first decimal of the 6 hour columns
  te     nested TE of the 12 exact tokens and of 3 pair keys (daily|social and daily|weekend at 0.1, notif|app exact),
         plus the rte / geo keys below
  rte    region x token keys (notifications, app opens, age, sleep, gender)
  geo    8 quantile-bin keys of geometry / imputer channels
  teavg  the te block averaged over 3 inner seeds (777, 1778, 2779)
  te2s   a second TE of the 12 tokens with its own smoothing (tes_ columns)
  woe    WOE sums of the te columns (9 numerics; daily/social/weekend)
  mrg    base margin = logit(prior) + mrg_scale * sum of the 12 single-token WOEs
  slim   drop the per-column missing flags, the rounded original LLRs and the categorical original channels
par: te_smooth (default 20), te2s_smooth (4), n_te_seeds (3), n_inner (5), mrg_scale (0.4)"""
import numpy as np
import pandas as pd

from common.data import CATEGORICALS, NUMERICS, cache_dir, folds, read_test, read_train, target

from .te import apply_te, build_te_fold

NUMS = list(NUMERICS)
CATS = list(CATEGORICALS)
FEATS = NUMS + CATS
BUDGET = ["sgw", "slack", "slack_frac", "share_social_media_hours", "share_gaming_hours", "share_work_study_hours",
          "wk_minus_d", "wk_minus_sgw", "wk_ratio", "d_m6", "d_m8", "s_m4", "region_obs"]
LAT_COLS = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
            "weekend_screen_time"]
RTE_COLS = ["notifications_per_day", "app_opens_per_day", "age", "sleep_hours", "gender"]
GEO_KEYS = [("slack", 64), ("share_social_media_hours", 16), ("wk_minus_d", 64), ("fill_slack", 64), ("fill_d_m8", 64),
            ("p_reg_hi", 64), ("imp_daily_screen_time_hours", 64), ("fill_s_m4", 64)]
SLIM_DROP = ("amp_gender", "amp_stress_level", "amp_academic_work_impact", "ollr_gender", "ollr_stress_level",
             "ollr_academic_work_impact")


def woe_sum(frame, names, lp):
    """sum over the TE columns `names` of logit(p) - logit(prior); NaN cells count 0"""
    w = np.zeros(len(frame))
    for nm in names:
        col = frame[nm].values
        p = np.clip(col, 1e-4, 1 - 1e-4)
        w += np.where(np.isnan(col), 0, np.log(p / (1 - p)) - lp)
    return w


def _r1key(sa, sb):
    va = sa.round(1).map(lambda x: f"{x:.1f}" if pd.notna(x) else "__NA__")
    vb = sb.round(1).map(lambda x: f"{x:.1f}" if pd.notna(x) else "__NA__")
    return va + "|" + vb


def _place(A, B, keys_tr, keys_te, te_trn, full_maps, trn_idx, val_idx):
    te_val = apply_te(full_maps, [(n, s.iloc[val_idx].reset_index(drop=True)) for n, s in keys_tr])
    te_test = apply_te(full_maps, keys_te)
    for name, _ in keys_tr:
        col = np.full(len(A), np.nan, np.float32)
        col[trn_idx] = te_trn[name]
        col[val_idx] = te_val[name]
        A[name] = col
        B[name] = te_test[name]


def assemble(fold, cfg, par=None):
    """Returns dict(A, B, y, trn_idx, val_idx, margin_tr, margin_te, train, test); A has one row per train row (the
    outer-train rows carry inner cross-fitted TEs, the outer-validation rows the full outer-train maps), B per test row."""
    par = par or {}
    train, test = read_train(), read_test()
    y = target()
    fold_id = folds(y)
    trn_idx = np.where(fold_id != fold)[0]
    val_idx = np.where(fold_id == fold)[0]

    src = cache_dir("density_features")
    Xtr = pd.read_parquet(src / "static_tr.parquet")
    Xte = pd.read_parquet(src / "static_te.parquet")
    tok_tr = pd.read_parquet(src / "tok_tr.parquet")
    tok_te = pd.read_parquet(src / "tok_te.parquet")

    missing = [c for c in Xtr.columns if c.startswith("na_")] + ["n_missing", "na_screen", "na_beh"]
    counts = [c for c in Xtr.columns if c.startswith("cnt_")]
    orig = [c for c in Xtr.columns if c.startswith(("ollr_", "ollr1_", "amp_"))] + ["ollr_sum"]
    pmi = [c for c in Xtr.columns if c.startswith("pmi_")]
    imp = [c for c in Xtr.columns if c.startswith(("imp_", "fill_"))] + ["p_reg_lo", "p_reg_mid", "p_reg_hi", "reg_exp"]

    cols = FEATS + missing + BUDGET
    if "cnt" in cfg:
        cols += counts
    if "orig" in cfg:
        cols += orig
    if "pmi" in cfg:
        cols += pmi
    if "imp" in cfg:
        cols += imp
    cols = list(dict.fromkeys(cols))
    A = Xtr[cols].copy()
    B = Xte[cols].copy()

    if "lat" in cfg:
        for c in LAT_COLS:
            for D in (A, B):
                v = D[c].values
                D[f"frac_{c}"] = (v - np.floor(v)).astype(np.float32)
                D[f"d1_{c}"] = (np.floor(v * 10) % 10).astype(np.float32)

    if "te" in cfg:
        keys_tr = [(f"te_{c}", tok_tr[c]) for c in FEATS]
        keys_te = [(f"te_{c}", tok_te[c]) for c in FEATS]
        keys_tr += [("te2_ds", _r1key(train["daily_screen_time_hours"], train["social_media_hours"])),
                    ("te2_dw", _r1key(train["daily_screen_time_hours"], train["weekend_screen_time"])),
                    ("te2_na", tok_tr["notifications_per_day"] + "|" + tok_tr["app_opens_per_day"])]
        keys_te += [("te2_ds", _r1key(test["daily_screen_time_hours"], test["social_media_hours"])),
                    ("te2_dw", _r1key(test["daily_screen_time_hours"], test["weekend_screen_time"])),
                    ("te2_na", tok_te["notifications_per_day"] + "|" + tok_te["app_opens_per_day"])]
        if "rte" in cfg:
            reg_tr = Xtr["region_obs"].map(lambda v: "m" if pd.isna(v) else str(int(v)))
            reg_te = Xte["region_obs"].map(lambda v: "m" if pd.isna(v) else str(int(v)))
            for c in RTE_COLS:
                keys_tr.append((f"ter_{c}", reg_tr + "|" + tok_tr[c]))
                keys_te.append((f"ter_{c}", reg_te + "|" + tok_te[c]))
        if "geo" in cfg:
            for gcol, q in GEO_KEYS:
                if gcol not in Xtr.columns:
                    continue
                allv = pd.concat([Xtr[gcol], Xte[gcol]])
                edges = np.unique(np.nanquantile(allv, np.linspace(0, 1, q + 1)))
                btr = pd.cut(Xtr[gcol], edges, labels=False, include_lowest=True)
                bte = pd.cut(Xte[gcol], edges, labels=False, include_lowest=True)
                keys_tr.append((f"teg_{gcol}", btr.astype(str)))
                keys_te.append((f"teg_{gcol}", bte.astype(str)))

        smooth = float(par.get("te_smooth", 20.0))
        n_inner = int(par.get("n_inner", 5))
        if "teavg" in cfg:   # average the inner cross-fit noise over 3 inner seeds
            n_seeds = int(par.get("n_te_seeds", 3))
            te_trn = None
            for seed in [777 + 1001 * k for k in range(n_seeds)]:
                t_, full_maps = build_te_fold(y, trn_idx, keys_tr, inner_seed=seed, n_inner=n_inner, smooth=smooth)
                if te_trn is None:
                    te_trn = {k: v.astype(np.float64) for k, v in t_.items()}
                else:
                    for k in te_trn:
                        te_trn[k] += t_[k]
            te_trn = {k: (v / n_seeds).astype(np.float32) for k, v in te_trn.items()}
        else:
            te_trn, full_maps = build_te_fold(y, trn_idx, keys_tr, n_inner=n_inner, smooth=smooth)
        _place(A, B, keys_tr, keys_te, te_trn, full_maps, trn_idx, val_idx)

        if "te2s" in cfg:
            keys1 = [(f"tes_{c}", tok_tr[c]) for c in FEATS]
            keys1t = [(f"tes_{c}", tok_te[c]) for c in FEATS]
            te_trn2, full_maps2 = build_te_fold(y, trn_idx, keys1, smooth=float(par.get("te2s_smooth", 4.0)))
            _place(A, B, keys1, keys1t, te_trn2, full_maps2, trn_idx, val_idx)

        if "woe" in cfg:
            lp_w = np.log(y[trn_idx].mean() / (1 - y[trn_idx].mean()))
            s9 = [f"te_{c}" for c in NUMS]
            s3 = [f"te_{c}" for c in ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time"]]
            for D in (A, B):
                D["woe_sum9"] = woe_sum(D, s9, lp_w).astype(np.float32)
                D["woe_sum_dsw"] = woe_sum(D, s3, lp_w).astype(np.float32)

    margin_tr = margin_te = None
    if "mrg" in cfg:
        prior = y[trn_idx].mean()
        lp = np.log(prior / (1 - prior))
        scale = float(par.get("mrg_scale", 0.4))
        singles = [f"te_{c}" for c in FEATS if f"te_{c}" in A.columns] or [f"te_{c}" for c in FEATS]
        margin_tr = (lp + scale * woe_sum(A, singles, lp)).astype(np.float32)
        margin_te = (lp + scale * woe_sum(B, singles, lp)).astype(np.float32)

    if "slim" in cfg:
        drop = [c for c in A.columns
                if (c.startswith("na_") and c not in ("na_screen", "na_beh")) or c.startswith("ollr1_") or c in SLIM_DROP]
        A = A.drop(columns=drop)
        B = B.drop(columns=drop)

    assert not A.columns.duplicated().any()
    return dict(A=A, B=B, y=y, trn_idx=trn_idx, val_idx=val_idx, margin_tr=margin_tr, margin_te=margin_te,
                train=train, test=test)
