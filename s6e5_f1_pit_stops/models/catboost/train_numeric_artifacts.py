#!/usr/bin/env python
"""GPU CatBoost on numeric transforms, digit artifacts and CatBoost-native categorical expansions of every column.

Column groups: object columns and numerics with < 100 distinct train + test values are categorical (CATS), the other
numerics are continuous (NUMS); every numeric column is numeric-like. Features, built once on train + test + original:
- per numeric-like column: value, log1p |x|, signed log1p, sqrt |x|, x^2, 1 / (x + 1), percentile rank; lap-time /
  delta / degradation / tyre-life / stint / progress ratios; differences, absolute differences, ratios and products of
  every pair of NUMS;
- counts of every numeric-like and categorical value (train + test + original);
- digit artifacts: first digit, last integer digit, mod 12 / 100, digit count, fractional part, round flag, 2 / 3
  fractional digits, 4 integer and 4 decimal digits;
- race-state copies, zero flags and mod 10 of 7 low-cardinality columns, top-3 / top-10 position and pit-stop flags;
- CatBoost-native categoricals: the raw CATS, per numeric-like column the exact value, rounding to 0 / 1 decimals,
  floor, 50-quantile bin and floor to even numbers, pairs of the 6 highest-cardinality CATS and a triple of the top 3,
  crosses of the 0.1-rounded lap time / delta with the top 3 CATS, and two-digit integer / decimal digit pairs.
All original rows are appended to the training side of every fold (weight 1). CatBoost: depth 8, lr 0.03, l2 3,
random_strength 0.5, Bayesian bagging_temperature 0.7, border_count 254, one_hot_max_size 4, max_ctr_complexity 2,
5000 iterations, early stopping 300 on the validation fold (use_best_model=True), seed 42.
Test = mean of the 5 fold models (probabilities clipped to [1e-7, 1 - 1e-7]).

Usage: python train_numeric_artifacts.py [--folds 0,1,2,3,4] [--force]"""
import gc
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_part  # noqa: E402
from models.catboost.lib.runner import FOLDER, finish, fold_parser, todo_folds  # noqa: E402

PAIR = "catboost_numeric_artifacts_numcat"
RANDOM_SEED = 42
CARDINALITY_CAT_THRESHOLD = 100
CB_PARAMS = dict(
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    random_strength=0.5,
    bagging_temperature=0.7,
    border_count=254,
    grow_policy="SymmetricTree",
    bootstrap_type="Bayesian",
    eval_metric="AUC",
    loss_function="Logloss",
    task_type="GPU",
    devices="0",
    random_seed=RANDOM_SEED,
    verbose=200,
    early_stopping_rounds=300,
    use_best_model=True,
    one_hot_max_size=4,
    max_ctr_complexity=2,
    allow_writing_files=False,
)


def safe_name(name) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    s = re.sub(r"_+", "_", s)
    if not s:
        s = "feature"
    if s[0].isdigit():
        s = "f_" + s
    return s


def make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        base = safe_name(c)
        k = seen.get(base, 0)
        out.append(base if k == 0 else f"{base}_dup{k}")
        seen[base] = k + 1
    return out


def sanitize_series(s: pd.Series, fill: str = "__NA__") -> pd.Series:
    return s.astype("string").fillna(fill).str.strip()


def signed_log1p(s) -> pd.Series:
    x = pd.to_numeric(pd.Series(s), errors="coerce").fillna(0).astype("float32")
    return np.sign(x) * np.log1p(np.abs(x))


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)


def safe_inverse(vals: np.ndarray) -> np.ndarray:
    denom = vals + 1.0
    out = np.zeros(len(vals), dtype="float32")
    mask = np.abs(denom) > 1e-6
    out[mask] = (1.0 / denom[mask]).astype("float32")
    return out


def safe_qbin_str(s: pd.Series, q: int = 50) -> pd.Series:
    x = safe_numeric(s)
    try:
        q_eff = int(min(q, max(2, x.nunique(dropna=True))))
        return pd.qcut(x, q=q_eff, duplicates="drop").astype(str)
    except Exception:
        try:
            return pd.cut(x, bins=min(q, max(2, x.nunique(dropna=True))), duplicates="drop").astype(str)
        except Exception:
            return pd.Series(["NA"] * len(s), index=s.index, dtype="object")


def ensure_cat_strings(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("__NA__").astype(str)
    return df


def infer_feature_groups(train, test):
    feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
    cats, nums, numeric_like = [], [], []
    for col in feature_cols:
        is_object = train[col].dtype == "object" or str(train[col].dtype).startswith("string")
        if is_object:
            cats.append(col)
        else:
            numeric_like.append(col)
            nunique = pd.concat([train[col], test[col]], axis=0, ignore_index=True).nunique(dropna=False)
            if nunique < CARDINALITY_CAT_THRESHOLD:
                cats.append(col)
            else:
                nums.append(col)
    cat_card = {c: pd.concat([train[c], test[c]], axis=0, ignore_index=True).nunique(dropna=False) for c in cats}
    top_cats = sorted(cats, key=lambda c: cat_card[c], reverse=True)[:6]
    print(f"CATS: {cats}\nNUMS: {nums}\nTOP_CATS_FOR_NGRAM: {top_cats}", flush=True)
    return cats, nums, numeric_like, top_cats


def build_features(train, test, orig, cats, nums, numeric_like_cols, top_cats):
    feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
    n_train = len(train)
    n_test = len(test)

    train_base = train[feature_cols + [TARGET]].copy()
    test_base = test[feature_cols].copy()
    orig_base = orig[feature_cols + [TARGET]].copy()

    y_train = train_base[TARGET].values.astype("int8")
    y_orig = orig_base[TARGET].values.astype("int8")

    combined = pd.concat(
        [
            train_base.drop(columns=[TARGET]).reset_index(drop=True),
            test_base.reset_index(drop=True),
            orig_base.drop(columns=[TARGET]).reset_index(drop=True),
        ],
        axis=0,
        ignore_index=True,
    )
    for c in cats:
        if c in combined.columns:
            combined[c] = sanitize_series(combined[c])
    for c in numeric_like_cols:
        if c in combined.columns:
            combined[c] = safe_numeric(combined[c])

    feat = pd.DataFrame(index=combined.index)

    # numeric transforms, interactions, ranks
    for c in numeric_like_cols:
        if c not in combined.columns:
            continue
        p = safe_name(c)
        vals = safe_numeric(combined[c]).values.astype("float64")
        feat[p] = vals.astype("float32")
        feat[f"{p}_log1p_abs"] = np.log1p(np.abs(vals)).astype("float32")
        feat[f"{p}_signed_log1p"] = signed_log1p(vals).astype("float32").values
        feat[f"{p}_sqrt_abs"] = np.sqrt(np.abs(vals)).astype("float32")
        feat[f"{p}_sq"] = np.square(vals).astype("float32")
        feat[f"{p}_inv"] = safe_inverse(vals)
        feat[f"{p}_rank"] = pd.Series(vals).rank(pct=True).astype("float32").values

    # lap-time / degradation state ratios
    if {"LapTime (s)", "LapTime_Delta"}.issubset(combined.columns):
        lap = safe_numeric(combined["LapTime (s)"]).values.astype("float32")
        delta = safe_numeric(combined["LapTime_Delta"]).values.astype("float32")
        feat["lap_time_plus_delta"] = (lap + delta).astype("float32")
        feat["lap_delta_ratio"] = (delta / (np.abs(lap) + 1.0)).astype("float32")
        feat["lap_delta_abs"] = np.abs(delta).astype("float32")
    if {"Cumulative_Degradation", "TyreLife"}.issubset(combined.columns):
        deg = safe_numeric(combined["Cumulative_Degradation"]).values.astype("float32")
        life = safe_numeric(combined["TyreLife"]).values.astype("float32")
        feat["degradation_per_tyre_life"] = (deg / (life + 1.0)).astype("float32")
        if "LapTime_Delta" in combined.columns:
            delta = safe_numeric(combined["LapTime_Delta"]).values.astype("float32")
            feat["degradation_minus_life_x_delta"] = (deg - life * delta).astype("float32")
    if {"LapNumber", "Stint"}.issubset(combined.columns):
        lapn = safe_numeric(combined["LapNumber"]).values.astype("float32")
        stint = safe_numeric(combined["Stint"]).values.astype("float32")
        feat["lap_per_stint"] = (lapn / (stint + 1.0)).astype("float32")
    if {"LapTime (s)", "RaceProgress"}.issubset(combined.columns):
        lap = safe_numeric(combined["LapTime (s)"]).values.astype("float32")
        prog = safe_numeric(combined["RaceProgress"]).values.astype("float32")
        feat["lap_time_per_race_progress"] = (lap / (prog + 0.01)).astype("float32")

    for a, b in itertools.combinations(nums, 2):
        if a in combined.columns and b in combined.columns:
            va = safe_numeric(combined[a]).values.astype("float32")
            vb = safe_numeric(combined[b]).values.astype("float32")
            pa, pb = safe_name(a), safe_name(b)
            feat[f"diff_{pa}_minus_{pb}"] = (va - vb).astype("float32")
            feat[f"absdiff_{pa}_{pb}"] = np.abs(va - vb).astype("float32")
            feat[f"ratio_{pa}_over_{pb}"] = (va / (np.abs(vb) + 1.0)).astype("float32")
            feat[f"prod_{pa}_x_{pb}"] = (va * vb).astype("float32")

    # value counts over train + test + original
    for c in numeric_like_cols:
        if c in combined.columns:
            p = safe_name(c)
            vc = combined[c].value_counts(dropna=False)
            feat[f"{p}_freq"] = combined[c].map(vc).fillna(0).astype("float32").values
    for c in cats:
        if c in combined.columns:
            p = safe_name(c)
            vc = combined[c].value_counts(dropna=False)
            feat[f"{p}_cat_freq"] = combined[c].map(vc).fillna(0).astype("float32").values

    # digit-level artifacts
    for c in numeric_like_cols:
        if c not in combined.columns:
            continue
        p = safe_name(c)
        vals = safe_numeric(combined[c]).values.astype("float64")
        flat = pd.Series(vals).map(lambda v: f"{v:.6f}".replace("-", "").replace(".", ""))
        int_part = np.floor(np.abs(vals)).astype(np.int64)
        frac = np.abs(vals) - np.floor(np.abs(vals))

        feat[f"{p}_first_digit"] = flat.str[0].fillna("0").astype("int8").values
        feat[f"{p}_last_digit"] = (int_part % 10).astype("int8")
        feat[f"{p}_mod12"] = (int_part % 12).astype("int8")
        feat[f"{p}_mod100"] = (int_part % 100).astype("int16")
        feat[f"{p}_ndigits"] = np.floor(np.log10(np.abs(vals) + 1) + 1).astype("int8")
        feat[f"{p}_frac"] = frac.astype("float32")
        feat[f"{p}_is_round"] = (frac < 0.005).astype("int8")
        feat[f"{p}_frac_x100"] = (frac * 100).astype("int16")
        feat[f"{p}_frac_x1000"] = (frac * 1000).astype("int16")
        for d in range(4):
            feat[f"{p}_intdigit_{d}"] = ((int_part // (10 ** d)) % 10).astype("int8")
        dec_shifted = (frac * 10000).astype(np.int64)
        for d in range(4):
            feat[f"{p}_decdigit_{d}"] = ((dec_shifted // (10 ** (3 - d))) % 10).astype("int8")

    # race-state copies and flags of the low-cardinality columns
    for c in ["PitStop", "Stint", "Position", "LapNumber", "TyreLife", "Position_Change", "Year"]:
        if c in combined.columns:
            p = safe_name(c)
            vals = safe_numeric(combined[c]).values.astype("float32")
            feat[f"state_{p}_as_num"] = vals
            feat[f"state_{p}_is_zero"] = (vals == 0).astype("int8")
            feat[f"state_{p}_mod10"] = (vals % 10).astype("int8")
    if "Position" in combined.columns:
        pos = safe_numeric(combined["Position"]).values.astype("float32")
        feat["position_top3"] = (pos <= 3).astype("int8")
        feat["position_top10"] = (pos <= 10).astype("int8")
    if "PitStop" in combined.columns:
        pit = safe_numeric(combined["PitStop"]).values.astype("float32")
        feat["is_pitstop_row"] = (pit > 0).astype("int8")

    # CatBoost-native categorical strings
    cat_native = pd.DataFrame(index=combined.index)
    for c in cats:
        if c in combined.columns:
            cat_native[f"RAW_{safe_name(c)}"] = sanitize_series(combined[c])

    for c in numeric_like_cols:
        if c not in combined.columns:
            continue
        p = safe_name(c)
        s = safe_numeric(combined[c])
        cat_native[f"NUMCAT_{p}_exact"] = s.astype(str)
        cat_native[f"NUMCAT_{p}_round0"] = np.round(s, 0).astype("int64").astype(str)
        cat_native[f"NUMCAT_{p}_round1"] = np.round(s, 1).astype(str)
        cat_native[f"NUMCAT_{p}_int"] = np.floor(s.values).astype(int).astype(str)
        cat_native[f"NUMCAT_{p}_qbin50"] = safe_qbin_str(s, 50).astype(str)
        cat_native[f"NUMCAT_{p}_div2"] = (np.floor(s.values / 2) * 2).astype(int).astype(str)

    for i, c1 in enumerate(top_cats):
        for c2 in top_cats[i + 1:]:
            if c1 in combined.columns and c2 in combined.columns:
                cat_native[f"BI_{safe_name(c1)}_{safe_name(c2)}"] = sanitize_series(combined[c1]) + "_" + sanitize_series(combined[c2])
    if len(top_cats) >= 3:
        c0, c1, c2 = top_cats[:3]
        cat_native[f"TRI_{safe_name(c0)}_{safe_name(c1)}_{safe_name(c2)}"] = (
            sanitize_series(combined[c0]) + "_" + sanitize_series(combined[c1]) + "_" + sanitize_series(combined[c2])
        )

    cross_nums = [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"] if c in numeric_like_cols]
    if len(cross_nums) < 2:
        cross_nums = numeric_like_cols[:2]
    for nc in cross_nums[:2]:
        s_num = np.round(safe_numeric(combined[nc]), 1).astype(str)
        for cc in top_cats[:3]:
            if cc in combined.columns:
                cat_native[f"CROSS_{safe_name(nc)}_{safe_name(cc)}"] = s_num + "_" + sanitize_series(combined[cc])

    for c in numeric_like_cols:
        if c not in combined.columns:
            continue
        p = safe_name(c)
        vals = safe_numeric(combined[c]).values.astype("float64")
        int_part = np.floor(np.abs(vals)).astype(np.int64)
        d0 = (int_part % 10).astype(str)
        d1 = ((int_part // 10) % 10).astype(str)
        cat_native[f"DIGITPAIR_{p}_01"] = pd.Series(d1 + d0, index=combined.index).astype(str)
        frac = np.abs(vals) - np.floor(np.abs(vals))
        dec_shifted = (frac * 100).astype(np.int64)
        cat_native[f"DIGITPAIR_{p}_dec01"] = pd.Series(dec_shifted, index=combined.index).astype(str)

    # assembly and split
    feat.columns = make_unique_columns(feat.columns)
    cat_native.columns = make_unique_columns(cat_native.columns)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in feat.columns:
        if not pd.api.types.is_numeric_dtype(feat[c]):
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0)
        if str(feat[c].dtype) == "float64":
            feat[c] = feat[c].astype("float32")
    cat_native = ensure_cat_strings(cat_native, cat_native.columns.tolist())

    train_feat = feat.iloc[:n_train].reset_index(drop=True)
    test_feat = feat.iloc[n_train:n_train + n_test].reset_index(drop=True)
    orig_feat = feat.iloc[n_train + n_test:].reset_index(drop=True)
    train_cat = cat_native.iloc[:n_train].reset_index(drop=True)
    test_cat = cat_native.iloc[n_train:n_train + n_test].reset_index(drop=True)
    orig_cat = cat_native.iloc[n_train + n_test:].reset_index(drop=True)

    train_all = pd.concat([train_feat, train_cat], axis=1)
    test_all = pd.concat([test_feat, test_cat], axis=1)
    orig_all = pd.concat([orig_feat, orig_cat], axis=1)

    numeric_features = train_feat.columns.tolist()
    catboost_cat_features = train_cat.columns.tolist()
    all_features = numeric_features + catboost_cat_features
    print(f"features: {len(all_features)} ({len(numeric_features)} numeric, {len(catboost_cat_features)} categorical)",
          flush=True)
    return train_all, test_all, orig_all, all_features, catboost_cat_features, y_train, y_orig


def main():
    from catboost import CatBoostClassifier, Pool

    args = fold_parser().parse_args()
    todo = todo_folds(PAIR, args.folds, args.force)
    if todo:
        train, test, orig = read_train(), read_test(), read_original()
        cats, nums, numeric_like_cols, top_cats = infer_feature_groups(train, test)
        train_all, test_all, orig_all, all_features, cat_features, y_train, y_orig = build_features(
            train, test, orig, cats, nums, numeric_like_cols, top_cats)
        np.random.seed(RANDOM_SEED)
        fold_id = folds(y_train)
        cat_indices = [all_features.index(c) for c in cat_features]
        for k in todo:
            train_idx, val_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            X_tr_base = train_all.loc[train_idx, all_features]
            y_tr_base = y_train[train_idx]
            X_val = train_all.loc[val_idx, all_features].reset_index(drop=True)
            y_val = y_train[val_idx]
            X_te = test_all[all_features].reset_index(drop=True)

            # every original row is appended to the training side
            X_tr = pd.concat([X_tr_base, orig_all[all_features]], axis=0).reset_index(drop=True)
            y_tr = np.concatenate([y_tr_base, y_orig])

            train_pool = Pool(data=X_tr, label=y_tr, cat_features=cat_indices, feature_names=all_features)
            val_pool = Pool(data=X_val, label=y_val, cat_features=cat_indices, feature_names=all_features)
            test_pool = Pool(data=X_te, cat_features=cat_indices, feature_names=all_features)
            model = CatBoostClassifier(**CB_PARAMS)
            model.fit(train_pool, eval_set=val_pool)

            val_pred = model.predict_proba(val_pool)[:, 1].astype("float32")
            te_pred = model.predict_proba(test_pool)[:, 1].astype("float32")
            save_part(FOLDER, PAIR, k, val_pred, te_pred)
            print(f"{PAIR} fold {k}: AUC {roc_auc_score(y_val, val_pred):.6f} best_iter {model.get_best_iteration()}",
                  flush=True)
            del X_tr_base, X_val, X_te, X_tr, train_pool, val_pool, test_pool, model
            gc.collect()
    finish(PAIR, clip=1e-7)


if __name__ == "__main__":
    main()
