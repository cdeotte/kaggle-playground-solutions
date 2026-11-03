#!/usr/bin/env python
"""TabM-mini on fold-safe snapped numeric proxies and rare flags, a formula score and per-class original priors
(pair tabm_mini_snapped_formula_priors_b).

Features (66 columns, competition rows only):
- the 8 categoricals; for each of the 11 numerics a snapper fitted on the fold's training rows maps every value seen
  fewer than 25 times to the nearest frequent value (ties -> the smaller one): the snapped value (`__cat`) and the rare
  flag (`__is_rare`) as categories;
- numerics, standardized on the fold's training rows: the 11 raw numerics, the formula score (Soil_Moisture < 25,
  Rainfall_mm < 300, Temperature_C > 30, Wind_Speed_kmh > 10, growth stage Harvest / Sowing, mulching) and the
  per-class original target rate of the 8 categoricals (24 columns).
Model: pytabkit TabM_D_Classifier, tabm-mini, k 16, 2 blocks x 256, piecewise-linear numeric embeddings (d 8),
dropout 0.2, lr 3e-4, batch 1024, 40 epochs (best epoch on the validation fold's cross-entropy), random_state 100.
Test = mean of the 5 fold models.
Usage: python train_mini_snapped_formula_priors_b.py"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit import TabM_D_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()

FOLDER = "tabm"
PAIR = "tabm_mini_snapped_formula_priors_b"
N_SPLITS = 5
RARE_MIN_COUNT = 25

NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

TABM_PARAMS = dict(
    device="cuda",
    val_metric_name="cross_entropy",
    random_state=100,
    verbosity=2,
    arch_type="tabm-mini",
    tabm_k=16,
    num_emb_type="pwl",
    d_embedding=8,
    batch_size=1024,
    lr=3e-4,
    n_epochs=40,
    dropout=0.20,
    d_block=256,
    n_blocks=2,
)

STATIC_NUMS = ["_formula_score"] + [f"ORIG_p{k}_{c}" for k in range(3) for c in CATS]
ALL_NUMS = NUMS + STATIC_NUMS


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def add_static_features(df, orig_df, orig_y_arr):
    """Formula score and per-class original priors."""
    df = df.copy()
    sm = df["Soil_Moisture"].values
    rf = df["Rainfall_mm"].values
    tc = df["Temperature_C"].values
    ws = df["Wind_Speed_kmh"].values
    df["_formula_score"] = (
        (sm < 25).astype(int) * 2 + (rf < 300).astype(int) * 2 +
        (tc > 30).astype(int) + (ws > 10).astype(int) -
        (df["Crop_Growth_Stage"].values == "Harvest").astype(int) * 2 -
        (df["Crop_Growth_Stage"].values == "Sowing").astype(int) * 2 -
        (df["Mulching_Used"].values == "Yes").astype(int)
    ).astype("float32")
    for col in CATS:
        for k in range(3):
            tmp = orig_df.copy()
            tmp["_yk"] = (orig_y_arr == k).astype(float)
            prior = tmp.groupby(col)["_yk"].mean()
            overall = (orig_y_arr == k).mean()
            df[f"ORIG_p{k}_{col}"] = df[col].map(prior).fillna(overall).astype("float32")
    return df


def build_numeric_snapper(train_series, rare_min_count=25):
    """Rare values (< rare_min_count in train_series) snap to the nearest frequent value (ties -> the smaller one)."""
    s = pd.to_numeric(train_series, errors="coerce").astype("float32")
    vc = s.value_counts()
    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype="float32")
    if frequent_vals.size == 0:
        frequent_vals = np.sort(s.dropna().unique().astype("float32"))
    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def snap(series):
        x = pd.to_numeric(series, errors="coerce").values.astype("float32")
        is_nan = np.isnan(x)
        is_rare = np.ones(len(x), dtype="int32")
        x_snapped = x.copy()
        for i, v in enumerate(x):
            if not is_nan[i]:
                is_rare[i] = 0 if float(v) in frequent_set else 1
        idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
        if idx_snap.size > 0 and frequent_vals.size > 0:
            v = x[idx_snap]
            pos = np.searchsorted(frequent_vals, v)
            pos = np.clip(pos, 0, len(frequent_vals) - 1)
            left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
            lvals = frequent_vals[left]
            rvals = frequent_vals[pos]
            nearest = np.where(np.abs(v - lvals) <= np.abs(v - rvals), lvals, rvals)
            x_snapped[idx_snap] = nearest.astype("float32")
        return x_snapped, is_rare
    return snap


def add_fold_safe_proxy_features(tr_df, va_df, te_df):
    """{col}__cat and {col}__is_rare of every numeric, snapper fitted on the fold's training rows."""
    tr_df = tr_df.copy(); va_df = va_df.copy(); te_df = te_df.copy()
    for col in NUMS:
        snapper = build_numeric_snapper(tr_df[col], rare_min_count=RARE_MIN_COUNT)
        tr_snap, tr_rare = snapper(tr_df[col])
        va_snap, va_rare = snapper(va_df[col])
        te_snap, te_rare = snapper(te_df[col])
        for df, snap, rare in [(tr_df, tr_snap, tr_rare),
                               (va_df, va_snap, va_rare),
                               (te_df, te_snap, te_rare)]:
            df[f"{col}__cat"] = pd.Series(snap).astype(str).values
            df[f"{col}__is_rare"] = pd.Series(rare).astype(str).values
    return tr_df, va_df, te_df


def build_tabm_matrices(tr_df, va_df, te_df):
    """Standardize the numerics (fitted on the fold's training rows) and cast the categoricals to category."""
    tr_df = tr_df.copy(); va_df = va_df.copy(); te_df = te_df.copy()

    cat_all = (CATS +
               [f"{c}__cat" for c in NUMS] +
               [f"{c}__is_rare" for c in NUMS])
    num_all = ALL_NUMS
    features = cat_all + num_all

    scaler = StandardScaler()
    tr_df[num_all] = scaler.fit_transform(tr_df[num_all].astype("float32")).astype("float32")
    va_df[num_all] = scaler.transform(va_df[num_all].astype("float32")).astype("float32")
    te_df[num_all] = scaler.transform(te_df[num_all].astype("float32")).astype("float32")

    for c in cat_all:
        for df in [tr_df, va_df, te_df]:
            df[c] = df[c].astype("category")

    return (tr_df[features].copy(), va_df[features].copy(), te_df[features].copy(),
            cat_all, num_all, features)


def main():
    log("Loading data...")
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    orig = read_original()

    y = train[TARGET].map(TARGET_MAP).astype(int).values
    y_orig = orig[TARGET].map(TARGET_MAP).astype(int).values
    train = train.drop(columns=[TARGET])
    orig = orig.drop(columns=[TARGET])

    for df in [train, test, orig]:
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        for c in CATS:
            df[c] = df[c].fillna("Missing").astype(str)

    N, M = len(train), len(test)
    log(f"Train: {N:,} | Test: {M:,}")

    log("Adding static features...")
    train = add_static_features(train, orig, y_orig)
    test = add_static_features(test, orig, y_orig)

    fold_id = folds(y)
    oof_preds = np.zeros((N, 3), dtype="float32")
    test_preds = np.zeros((M, 3), dtype="float32")
    fold_accs = []

    for fold in range(1, N_SPLITS + 1):
        tr_idx, va_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        fold_t0 = time.time()
        log("=" * 80)
        log(f"Fold {fold}/{N_SPLITS}")

        tr_df = train.iloc[tr_idx].reset_index(drop=True).copy()
        va_df = train.iloc[va_idx].reset_index(drop=True).copy()
        te_df = test.reset_index(drop=True).copy()

        y_tr = y[tr_idx]
        y_va = y[va_idx]

        tr_df, va_df, te_df = add_fold_safe_proxy_features(tr_df, va_df, te_df)

        X_tr, X_va, X_te, cat_all, num_all, features = build_tabm_matrices(tr_df, va_df, te_df)

        log(f"X_tr={X_tr.shape} | n_cat={len(cat_all)} | n_num={len(num_all)}")

        model = TabM_D_Classifier(**TABM_PARAMS)
        model.fit(X_tr, y_tr, X_va, y_va, cat_col_names=cat_all)

        p_va = model.predict_proba(X_va).astype("float32")
        p_te = model.predict_proba(X_te).astype("float32")

        oof_preds[va_idx] = p_va
        test_preds += p_te / N_SPLITS

        fold_acc = balanced_accuracy_score(y_va, np.argmax(p_va, axis=1))
        fold_accs.append(fold_acc)
        log(f"Fold {fold} Balanced ACC: {fold_acc:.5f} | time={time.time() - fold_t0:.1f}s")

        del model, tr_df, va_df, te_df, X_tr, X_va, X_te
        gc.collect()

    overall = balanced_accuracy_score(y, np.argmax(oof_preds, axis=1))
    log(f"Mean fold ACC : {np.mean(fold_accs):.5f} +/- {np.std(fold_accs):.5f}")
    log(f"Overall OOF Balanced ACC: {overall:.5f}")
    save_pair(model_dir(FOLDER), PAIR, oof_preds, test_preds)


if __name__ == "__main__":
    main()
