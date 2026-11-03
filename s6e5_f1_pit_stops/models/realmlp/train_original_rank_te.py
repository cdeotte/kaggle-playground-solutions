#!/usr/bin/env python
"""RealMLP on original-data rank / z-score features, relationship features, bin / pair / triple keys, original priors and
nested target encodings. No original rows are added to the training rows; the original data only feeds the features.

Features (train, test and original rows; the 3 string columns and the numerics with < 100 distinct values are first cast
to strings, numerics then back to float32):
- the 14 columns;
- for LapTime, LapTime_Delta, Cumulative_Degradation, RaceProgress, TyreLife: percentile rank and z-score against the
  original values, and the first two decimals (of the value printed with 4 decimals) as categories;
- Pit_state_total = PitStop + RaceProgress, LapNumber - TyreLife, |LapTime_Delta|, Cumulative_Degradation - |delta|,
  LapTime / (RaceProgress + 0.01), (Cumulative_Degradation + |delta|) / (TyreLife + 1), Position / (RaceProgress + 0.02);
- quantile bins (fitted on train + test + original) of LapTime (10), LapTime_Delta (10), Cumulative_Degradation (10),
  RaceProgress (8), TyreLife (8); 10 pair keys and 3 triple keys of raw columns and bins (categories);
- original target rate and log count of 27 keys (raw columns, bins, pairs, triples); original relative frequency of the
  10 category-like raw columns (Driver, Compound, Race and the 7 numerics with < 100 values) (123 columns);
- per fold: nested target encodings (smoothing 20; 5 inner folds, seed 42 + fold) of the 5 bins, 6 decimal digits and
  12 pair / triple keys, the validation and test rows get the map of all training rows (146 columns).
Model: pytabkit RealMLP_TD_Classifier, n_ens 8, 4 x 352 mish, lr 0.075, wd 0.0236, batch 256, up to 20 epochs with early
stopping on the validation classification error (additive patience 20), seed 42 + fold (1-based).
Test = mean of the 5 fold models.
Usage: python train_original_rank_te.py"""
import gc
import os
import random
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, N_FOLDS, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

PAIR = "realmlp_original_rank_zscore_te"
SEED = 42
TE_INNER_SPLITS = 5
TE_SMOOTH = 20.0

REALMLP_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_state": SEED,
    "verbosity": 2,
    "n_epochs": 20,
    "batch_size": 256,
    "n_ens": 8,
    "val_metric_name": "class_error",
    "use_early_stopping": True,
    "early_stopping_additive_patience": 20,
    "early_stopping_multiplicative_patience": 1,
    "act": "mish",
    "embedding_size": 6,
    "first_layer_lr_factor": 0.25,
    "hidden_sizes": "rectangular",
    "hidden_width": 352,
    "lr": 0.075,
    "ls_eps": 0.01,
    "ls_eps_sched": "coslog4",
    "max_one_hot_cat_size": 18,
    "n_hidden_layers": 4,
    "p_drop": 0.05,
    "p_drop_sched": "flat_cos",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "scale_lr_factor": 2.24,
    "sq_mom": 0.988,
    "wd": 0.0236,
}

TE_CANDIDATES = [
    "LapTime_bin", "LapDelta_bin", "Degradation_bin", "RaceProgress_bin", "TyreLife_bin",
    "LapTime_s0", "LapTime_s1", "LapTime_Delta0", "LapTime_Delta1", "Cumulative_Degradation0", "Cumulative_Degradation1",
    "Driver__Compound", "Race__Year", "Driver__Race", "Compound__TyreLife_bin", "Race__LapTime_bin",
    "Race__RaceProgress_bin", "Compound__Degradation_bin", "Driver__LapDelta_bin", "Stint__TyreLife_bin",
    "Driver__Race__Compound", "Race__Compound__TyreLife_bin", "Driver__Compound__PitStop",
]


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe_name(x):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(x)).strip("_")


def safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)


def safe_str(s):
    return s.astype("string").fillna("missing").str.strip()


def pctrank_against(values, reference):
    ref_sorted = np.sort(np.asarray(reference, dtype=np.float32))
    if len(ref_sorted) == 0:
        return np.zeros(len(values), dtype=np.float32)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype(np.float32)


def zscore_against(values, reference):
    ref = np.asarray(reference, dtype=np.float32)
    if len(ref) == 0:
        return np.zeros(len(values), dtype=np.float32)
    mu = np.mean(ref)
    sigma = np.std(ref)
    if sigma == 0:
        return np.zeros(len(values), dtype=np.float32)
    return ((values - mu) / sigma).astype(np.float32)


def smoothed_binary_encoding(train_keys, train_y, apply_keys, smooth=20.0):
    train_y = np.asarray(train_y, dtype=np.float32)
    prior = float(np.mean(train_y))
    stat_df = pd.DataFrame({"key": train_keys.astype(str), "y": train_y})
    stats = stat_df.groupby("key", dropna=False)["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + smooth * prior) / (stats["count"] + smooth)
    return apply_keys.astype(str).map(enc).fillna(prior).astype(np.float32).values


def build_combo_binary_te(xtr, ytr, xva, xte, te_cols, inner_splits=5, seed=42, smooth=20.0):
    ytr = np.asarray(ytr, dtype=np.int64)
    tr_te = pd.DataFrame(index=xtr.index)
    va_te = pd.DataFrame(index=xva.index)
    te_te = pd.DataFrame(index=xte.index)
    inner_kf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    for col in te_cols:
        tr_keys = xtr[col].astype(str)
        va_keys = xva[col].astype(str)
        te_keys = xte[col].astype(str)
        tr_oof = np.zeros(len(xtr), dtype=np.float32)
        for it, iv in inner_kf.split(xtr, ytr):
            tr_oof[iv] = smoothed_binary_encoding(tr_keys.iloc[it], ytr[it], tr_keys.iloc[iv], smooth=smooth)
        out_col = f"_{safe_name(col)}_te"
        tr_te[out_col] = tr_oof
        va_te[out_col] = smoothed_binary_encoding(tr_keys, ytr, va_keys, smooth=smooth)
        te_te[out_col] = smoothed_binary_encoding(tr_keys, ytr, te_keys, smooth=smooth)
    return tr_te, va_te, te_te


def load_data():
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].astype("int8").to_numpy()
    y_orig = orig[TARGET].astype("int8").to_numpy()
    train = train.drop(columns=[ID, TARGET], errors="ignore")
    test = test.drop(columns=[ID], errors="ignore")
    orig = orig.drop(columns=[ID, TARGET], errors="ignore")
    return train, test, orig, y, y_orig


def infer_groups(train, test, orig):
    feature_cols = [c for c in train.columns if c in test.columns and c in orig.columns]
    cats, nums, numeric_like = [], [], []
    both = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
    for c in feature_cols:
        if train[c].dtype == object or test[c].dtype == object or orig[c].dtype == object:
            cats.append(c)
        else:
            numeric_like.append(c)
            if both[c].nunique(dropna=False) < 100:
                cats.append(c)
            else:
                nums.append(c)
    for df in (train, test, orig):
        for c in cats:
            df[c] = safe_str(df[c])
        for c in numeric_like:
            df[c] = safe_num(df[c])
    return feature_cols, cats, nums, numeric_like


def fit_bins(reference, col, bins):
    vals = safe_num(reference[col]).values.reshape(-1, 1)
    n_bins = int(max(2, min(bins, pd.Series(vals.ravel()).nunique())))
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    est.fit(vals)
    return est


def transform_bin(est, df, col):
    return est.transform(safe_num(df[col]).values.reshape(-1, 1)).ravel().astype(np.int32).astype(str)


def make_pair_key(df, f1, f2):
    return df[f1].astype(str) + "__" + df[f2].astype(str)


def make_triple_key(df, f1, f2, f3):
    return df[f1].astype(str) + "__" + df[f2].astype(str) + "__" + df[f3].astype(str)


def add_key_feature_columns(train, test, orig):
    bin_specs = {
        "LapTime_bin": ("LapTime (s)", 10),
        "LapDelta_bin": ("LapTime_Delta", 10),
        "Degradation_bin": ("Cumulative_Degradation", 10),
        "RaceProgress_bin": ("RaceProgress", 8),
        "TyreLife_bin": ("TyreLife", 8),
    }
    ref = pd.concat([train, test, orig], axis=0, ignore_index=True)
    for out_col, (src, bins) in bin_specs.items():
        if src not in train.columns:
            continue
        est = fit_bins(ref, src, bins)
        for df in (train, test, orig):
            df[out_col] = transform_bin(est, df, src)

    cross_pairs = [
        ("Driver", "Compound"),
        ("Race", "Year"),
        ("Driver", "Race"),
        ("Compound", "TyreLife_bin"),
        ("Race", "LapTime_bin"),
        ("Race", "RaceProgress_bin"),
        ("Compound", "Degradation_bin"),
        ("Driver", "LapDelta_bin"),
        ("Stint", "TyreLife_bin"),
        ("Position", "Position_Change"),
    ]
    triple_cats = [
        ("Driver", "Race", "Compound"),
        ("Race", "Compound", "TyreLife_bin"),
        ("Driver", "Compound", "PitStop"),
    ]
    for a, b in cross_pairs:
        if a in train.columns and b in train.columns:
            n = f"{safe_name(a)}__{safe_name(b)}"
            for df in (train, test, orig):
                df[n] = make_pair_key(df, a, b)
    for a, b, c in triple_cats:
        if a in train.columns and b in train.columns and c in train.columns:
            n = f"{safe_name(a)}__{safe_name(b)}__{safe_name(c)}"
            for df in (train, test, orig):
                df[n] = make_triple_key(df, a, b, c)
    return train, test, orig


def full_feature_engineering(train, test, orig, y_orig, cats):
    artifact_nums = [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "TyreLife"] if c in train.columns]
    for df in (train, test, orig):
        for col in artifact_nums:
            vals = safe_num(df[col]).values
            df[f"{safe_name(col)}_rank"] = pctrank_against(vals, safe_num(orig[col]).values)
            df[f"{safe_name(col)}_z"] = zscore_against(vals, safe_num(orig[col]).values)
            s = safe_num(df[col]).round(4).map(lambda x: f"{x:.4f}")
            dec = s.str.split(".").str[1].fillna("")
            prefix = safe_name(col)
            df[f"{prefix}0"] = dec.str[0].fillna("-1").replace("", "-1")
            df[f"{prefix}1"] = dec.str[1].fillna("-1").replace("", "-1")
        lap = safe_num(df["LapNumber"])
        tyre = safe_num(df["TyreLife"])
        lt = safe_num(df["LapTime (s)"])
        ld = safe_num(df["LapTime_Delta"])
        deg = safe_num(df["Cumulative_Degradation"])
        prog = safe_num(df["RaceProgress"])
        pos = safe_num(df["Position"])
        pit = safe_num(df["PitStop"])
        df["Pit_state_total"] = (pit + prog).astype(np.float32)
        df["Lap_tyre_diff"] = (lap - tyre).astype(np.float32)
        df["Lap_delta_abs"] = np.abs(ld).astype(np.float32)
        df["Degradation_gap"] = (deg - np.abs(ld)).astype(np.float32)
        df["Time_to_progress_ratio"] = (lt / (prog + 0.01)).astype(np.float32)
        df["Stress_to_tyre_ratio"] = ((deg + np.abs(ld)) / (tyre + 1.0)).astype(np.float32)
        df["Position_pressure"] = (pos / (prog + 0.02)).astype(np.float32)

    train, test, orig = add_key_feature_columns(train, test, orig)

    orig_labeled = orig.copy()
    orig_labeled["_y_"] = y_orig
    prior = float(np.mean(y_orig))
    orig_map_cols = [
        "Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "Position_Change",
        "LapTime_bin", "LapDelta_bin", "Degradation_bin", "RaceProgress_bin", "TyreLife_bin",
        "Driver__Compound", "Race__Year", "Driver__Race", "Compound__TyreLife_bin", "Race__LapTime_bin",
        "Race__RaceProgress_bin", "Compound__Degradation_bin", "Driver__LapDelta_bin", "Stint__TyreLife_bin",
        "Driver__Race__Compound", "Race__Compound__TyreLife_bin", "Driver__Compound__PitStop",
    ]
    for col in orig_map_cols:
        if col not in train.columns or col not in orig_labeled.columns:
            continue
        grp = orig_labeled.groupby(col, dropna=False)["_y_"].agg(["mean", "count"])
        for df in (train, test, orig):
            df[f"ORIG_proba_{safe_name(col)}"] = df[col].map(grp["mean"]).fillna(prior).astype(np.float32)
            df[f"ORIG_count_{safe_name(col)}"] = df[col].map(np.log1p(grp["count"].astype(np.float32))).fillna(0).astype(np.float32)

    for col in cats:
        vc = orig[col].value_counts(normalize=True, dropna=False)
        for df in (train, test, orig):
            df[f"{safe_name(col)}_orig_freq"] = df[col].map(vc).fillna(0).astype(np.float32)

    return train, test, orig


def predict_positive(model, X):
    p = np.asarray(model.predict_proba(X), dtype=np.float32)
    if p.ndim == 1:
        return p
    if p.shape[1] == 1:
        return p[:, 0]
    return p[:, 1]


def main():
    seed_everything(SEED)
    train, test, orig, y, y_orig = load_data()
    _, cats, _, _ = infer_groups(train, test, orig)
    train, test, orig = full_feature_engineering(train, test, orig, y_orig, cats)

    feature_cols = [c for c in train.columns if c in test.columns and c in orig.columns]
    te_cols = [c for c in TE_CANDIDATES if c in train.columns]
    cat_cols = [c for c in feature_cols if str(train[c].dtype) in ("object", "string") or str(train[c].dtype).startswith("category")]
    print(f"features={len(feature_cols)} cat_cols={len(cat_cols)} te_cols={len(te_cols)}", flush=True)

    fold_id = folds(y)
    oof = np.zeros(len(train), dtype=np.float32)
    pred_test = np.zeros(len(test), dtype=np.float32)
    for k in range(N_FOLDS):
        fold = k + 1
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        x_tr = train.iloc[tr_idx][feature_cols].reset_index(drop=True).copy()
        y_tr = y[tr_idx].copy()
        x_va = train.iloc[va_idx][feature_cols].reset_index(drop=True).copy()
        y_va = y[va_idx].copy()
        x_te = test[feature_cols].reset_index(drop=True).copy()

        tr_te, va_te, te_te = build_combo_binary_te(
            x_tr, y_tr, x_va, x_te, te_cols=te_cols, inner_splits=TE_INNER_SPLITS, seed=SEED + fold, smooth=TE_SMOOTH
        )
        x_tr = pd.concat([x_tr, tr_te], axis=1)
        x_va = pd.concat([x_va, va_te], axis=1)
        x_te = pd.concat([x_te, te_te], axis=1)

        for c in cat_cols:
            if c in x_tr.columns:
                x_tr[c] = x_tr[c].astype("category")
                x_va[c] = x_va[c].astype("category")
                x_te[c] = x_te[c].astype("category")
        for c in x_tr.columns:
            if c not in cat_cols:
                x_tr[c] = safe_num(x_tr[c])
                x_va[c] = safe_num(x_va[c])
                x_te[c] = safe_num(x_te[c])

        params = dict(REALMLP_PARAMS)
        params["random_state"] = SEED + fold
        model = RealMLP_TD_Classifier(**params)
        model.fit(x_tr, y_tr, X_val=x_va, y_val=y_va)
        va_pred = predict_positive(model, x_va)
        te_pred = predict_positive(model, x_te)
        oof[va_idx] = va_pred
        pred_test += te_pred / N_FOLDS
        print(f"fold {fold}: AUC {roc_auc_score(y_va, va_pred):.6f}", flush=True)
        del x_tr, x_va, x_te, model, tr_te, va_te, te_te, va_pred, te_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, PAIR, oof, pred_test)


if __name__ == "__main__":
    main()
