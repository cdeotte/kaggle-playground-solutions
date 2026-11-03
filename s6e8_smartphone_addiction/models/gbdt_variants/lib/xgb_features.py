"""Features of the XGBoost model (xgb_dual_smooth_te): 49 base columns + 24 nested exact-value target encodings.

Base columns (target-free; the conditional medians use train + test predictors):
  A  9 raw numerics (NaN kept -> XGBoost's native missing branches)
  B  8 derived geometry: comp_sum, slack, utilization, slack_frac, sleep_daily, sleep_daily_gap (20 - sleep - daily),
     weekend_over_daily, social_over_daily
  C  5 conditional-median imputations: daily | (social, gaming), social | (daily, gaming), weekend | (daily, social),
     notifications | app opens, app opens | notifications (fallback: first key only, then the global median)
  D  2 block observation counts (daily / social / weekend; notifications / app opens)
  E  11 one-hot categoricals (NA = own column)
  F  14 original-dataset per-token label rates (smoothing 3 toward the original prior): 9 numerics, 3 categoricals,
     notifications x app opens, daily x weekend
Target encodings: the exact-value tokens ('%.2f' / '__NA__') of 9 numerics and 4 pairs (notifications x app opens,
daily x weekend, sleep x daily, daily x social), each with smoothing 20 (te_) and 2 (teS2_); the two age columns are
pruned -> 24 columns. Per outer fold f: outer-train rows are encoded by 5 inner folds (StratifiedKFold(5, shuffle=True,
random_state=2042 + f)), validation and test rows by the map of all outer-train rows."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.data import TARGET, folds as fold_ids, read_original, read_test, read_train

SEED_INNER = 2042          # inner-fold seed base (inner seed = SEED_INNER + outer fold)
N_INNER = 5
SMOOTH = 20.0              # TE shrinkage toward the outer-train prior (smooth component)
SMOOTH_SHARP = 2.0         # low-shrinkage TE: sharp per-token residuals
ORIG_SMOOTH = 3.0          # original-evidence rate shrinkage (small: 7,500 rows)

NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
       "work_study_hours", "sleep_hours", "notifications_per_day",
       "app_opens_per_day", "weekend_screen_time"]
CAT = ["gender", "stress_level", "academic_work_impact"]
TE_PAIRS = [("notifications_per_day", "app_opens_per_day"),
            ("daily_screen_time_hours", "weekend_screen_time"),
            ("sleep_hours", "daily_screen_time_hours"),
            ("daily_screen_time_hours", "social_media_hours")]
TE_KEYS = NUM + [f"{a}x{b}" for a, b in TE_PAIRS]
PRUNE_TE = ["te_age", "teS2_age"]


def tok_num(s: pd.Series) -> np.ndarray:
    """Exact-value token, 2-decimal formatted. NA -> '__NA__'."""
    v = s.to_numpy(dtype="float64", na_value=np.nan)
    return np.where(np.isnan(v), "__NA__", np.char.mod("%.2f", v))


def tok_cat(s: pd.Series) -> np.ndarray:
    v = s.to_numpy()
    return np.where(pd.isna(v), "__NA__", v.astype(str))


def te_map(tokens, labels, prior, smooth=SMOOTH):
    g = pd.DataFrame({"t": tokens, "y": labels}).groupby("t")["y"].agg(["sum", "count"])
    return (g["sum"] + smooth * prior) / (g["count"] + smooth)


def apply_map(tokens, mp, prior):
    return pd.Series(tokens).map(mp).fillna(prior).to_numpy(dtype=np.float64)


def safe_div(a, b):
    return a / b.where(b > 0)


def derived(df):
    d, so, g, w = (df[c] for c in ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours"])
    sl, we = df["sleep_hours"], df["weekend_screen_time"]
    comp = so + g + w
    out = pd.DataFrame(index=df.index)
    out["comp_sum"] = comp
    out["slack"] = d - comp
    out["utilization"] = safe_div(comp, d)
    out["slack_frac"] = safe_div(out["slack"], d)
    out["sleep_daily"] = sl + d
    out["sleep_daily_gap"] = 20.0 - (sl + d)
    out["weekend_over_daily"] = safe_div(we, d)
    out["social_over_daily"] = safe_div(so, d)
    return out


def build():
    """Returns Xb_tr (n_train, 49), Xb_te (n_test, 49), te_oof (5, n_train, 24), te_te (5, n_test, 24), y, folds, names."""
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].values.astype(np.int8)
    folds = fold_ids(y)
    n_tr, n_te = len(train), len(test)

    # A. raw numeric
    Xb_tr = train[NUM].to_numpy(dtype=np.float32)
    Xb_te = test[NUM].to_numpy(dtype=np.float32)
    base_cols = list(NUM)

    # B. derived geometry
    Xb_tr = np.hstack([Xb_tr, derived(train).to_numpy(dtype=np.float32)])
    Xb_te = np.hstack([Xb_te, derived(test).to_numpy(dtype=np.float32)])
    base_cols += ["comp_sum", "slack", "utilization", "slack_frac", "sleep_daily", "sleep_daily_gap",
                  "weekend_over_daily", "social_over_daily"]

    # C. conditional-median imputations (train + test predictors, no labels)
    both = pd.concat([train, test], ignore_index=True)
    tcache = {c: tok_num(both[c]) for c in NUM}
    tcache.update({c: tok_cat(both[c]) for c in CAT})

    def cond_medians(target, key_cols):
        d = pd.DataFrame({"t": both[target].to_numpy(dtype="float64"),
                          "k0": tcache[key_cols[0]],
                          "k1": tcache[key_cols[1]] if len(key_cols) > 1 else None})
        keys = [k for k in ("k0", "k1") if d[k].notna().any()]
        return d.groupby(keys)["t"].median()

    def fill_cond(col, key_cols):
        med2 = cond_medians(col, key_cols)
        med1 = pd.DataFrame({"t": both[col].to_numpy(dtype="float64"),
                             "k0": tcache[key_cols[0]]}).groupby("k0")["t"].median()
        med0 = both[col].median()
        out = both[col].to_numpy(dtype="float32").copy()
        m = np.isnan(out)
        if m.any():
            k0 = tcache[key_cols[0]][m]
            k1 = tcache[key_cols[1]][m] if len(key_cols) > 1 else None
            if k1 is not None:
                v = med2.reindex(pd.MultiIndex.from_arrays([k0, k1])).to_numpy(dtype=np.float32)
            else:
                v = med1.reindex(pd.Index(k0)).to_numpy(dtype=np.float32)
            still = np.isnan(v)
            v[still] = med1.reindex(pd.Index(k0[still])).to_numpy(dtype=np.float32)
            still = np.isnan(v)
            v[still] = np.float32(med0)
            out[m] = v
        return out

    imp = {
        "daily_cmi": fill_cond("daily_screen_time_hours", ["social_media_hours", "gaming_hours"]),
        "social_cmi": fill_cond("social_media_hours", ["daily_screen_time_hours", "gaming_hours"]),
        "weekend_cmi": fill_cond("weekend_screen_time", ["daily_screen_time_hours", "social_media_hours"]),
        "notif_cmi": fill_cond("notifications_per_day", ["app_opens_per_day"]),
        "app_cmi": fill_cond("app_opens_per_day", ["notifications_per_day"]),
    }
    imp_block = np.stack([imp[k] for k in ["daily_cmi", "social_cmi", "weekend_cmi", "notif_cmi", "app_cmi"]], axis=1)
    Xb_tr = np.hstack([Xb_tr, imp_block[:n_tr]])
    Xb_te = np.hstack([Xb_te, imp_block[n_tr:]])
    base_cols += list(imp)

    # D. block observation counts
    scr = ["daily_screen_time_hours", "social_media_hours", "weekend_screen_time"]
    napp = ["notifications_per_day", "app_opens_per_day"]
    Xb_tr = np.hstack([Xb_tr, train[scr].notna().sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1),
                       train[napp].notna().sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1)])
    Xb_te = np.hstack([Xb_te, test[scr].notna().sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1),
                       test[napp].notna().sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1)])
    base_cols += ["n_screen_obs", "n_napp_obs"]

    # E. one-hot categoricals (aligned across train / test, NA = own column)
    ohe_tr, ohe_te = [], []
    for c in CAT:
        vals = sorted(set(train[c].dropna().astype(str)) | set(test[c].dropna().astype(str)))
        tr_str = train[c].astype(object).where(train[c].notna(), "__NA__").astype(str)
        te_str = test[c].astype(object).where(test[c].notna(), "__NA__").astype(str)
        for v in vals + ["__NA__"]:
            base_cols.append(f"{c}={v}")
            ohe_tr.append((tr_str == v).to_numpy(dtype=np.float32).reshape(-1, 1))
            ohe_te.append((te_str == v).to_numpy(dtype=np.float32).reshape(-1, 1))
    Xb_tr = np.hstack([Xb_tr, np.hstack(ohe_tr)])
    Xb_te = np.hstack([Xb_te, np.hstack(ohe_te)])

    # F. original-dataset per-token label rates (fixed constants; original rows are not training rows)
    orig_prior = orig[TARGET].mean()
    yorig = orig[TARGET].to_numpy(dtype=np.float64)

    def orig_rate(tokens):
        g = pd.DataFrame({"t": tokens, "y": yorig}).groupby("t")["y"].agg(["sum", "count"])
        return (g["sum"] + ORIG_SMOOTH * orig_prior) / (g["count"] + ORIG_SMOOTH)

    of_tr, of_te = [], []
    for c in NUM:
        mp = orig_rate(tok_num(orig[c]))
        of_tr.append(apply_map(tok_num(train[c]), mp, orig_prior).reshape(-1, 1))
        of_te.append(apply_map(tok_num(test[c]), mp, orig_prior).reshape(-1, 1))
        base_cols.append(f"orig_rate_{c}")
    for c in CAT:
        mp = orig_rate(tok_cat(orig[c]))
        of_tr.append(apply_map(tok_cat(train[c]), mp, orig_prior).reshape(-1, 1))
        of_te.append(apply_map(tok_cat(test[c]), mp, orig_prior).reshape(-1, 1))
        base_cols.append(f"orig_rate_{c}")
    for a, b in [("notifications_per_day", "app_opens_per_day"), ("daily_screen_time_hours", "weekend_screen_time")]:
        mp = orig_rate(tok_num(orig[a]) + "|" + tok_num(orig[b]))
        of_tr.append(apply_map(tok_num(train[a]) + "|" + tok_num(train[b]), mp, orig_prior).reshape(-1, 1))
        of_te.append(apply_map(tok_num(test[a]) + "|" + tok_num(test[b]), mp, orig_prior).reshape(-1, 1))
        base_cols.append(f"orig_rate_{a}x{b}")
    Xb_tr = np.hstack([Xb_tr, np.hstack(of_tr)])
    Xb_te = np.hstack([Xb_te, np.hstack(of_te)])
    print(f"base features: {Xb_tr.shape[1]}", flush=True)

    # G. nested target encodings (smooth S=20 and sharp S=2 copy of every key, minus the pruned columns)
    train_toks, test_toks = {}, {}
    for c in NUM:
        train_toks[c], test_toks[c] = tok_num(train[c]), tok_num(test[c])
    for a, b in TE_PAIRS:
        train_toks[f"{a}x{b}"] = tok_num(train[a]) + "|" + tok_num(train[b])
        test_toks[f"{a}x{b}"] = tok_num(test[a]) + "|" + tok_num(test[b])
    te_cols = [nm for k in TE_KEYS for nm in (f"te_{k}", f"teS2_{k}") if nm not in PRUNE_TE]

    te_oof = np.zeros((5, n_tr, len(te_cols)), dtype=np.float32)
    te_te = np.zeros((5, n_te, len(te_cols)), dtype=np.float32)
    for f in range(5):
        trn_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        y_trn = y[trn_idx].astype(np.float64)
        prior = y_trn.mean()
        inner = StratifiedKFold(N_INNER, shuffle=True, random_state=SEED_INNER + f)
        for j, name in enumerate(te_cols):
            key = name[5:] if name.startswith("teS2") else name[3:]
            smooth = SMOOTH_SHARP if name.startswith("teS2") else SMOOTH
            t_tr = train_toks[key][trn_idx]
            # inner cross-fitting: outer-train rows never use their own label
            for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y_trn):
                mp = te_map(t_tr[i_fit], y_trn[i_fit], prior, smooth)
                te_oof[f, trn_idx[i_app], j] = apply_map(t_tr[i_app], mp, prior)
            # map of all outer-train rows: validation and test rows
            mp_full = te_map(t_tr, y_trn, prior, smooth)
            te_oof[f, val_idx, j] = apply_map(train_toks[key][val_idx], mp_full, prior)
            te_te[f, :, j] = apply_map(test_toks[key], mp_full, prior)
        print(f"fold {f}: {len(te_cols)} target encodings built", flush=True)
    return Xb_tr, Xb_te, te_oof, te_te, y, folds, base_cols + te_cols
