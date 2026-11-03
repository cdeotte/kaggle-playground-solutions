#!/usr/bin/env python
"""GPU XGBoost on a union of label-free blocks with fold-safe encodings, every column except Rainfall_mm and
Previous_Irrigation_mm pre-binned to <= 256 quantile bins of the fold's training rows, max_bin 16,000
(pair xgb_union_te_prebinned_a).

Label-free features (train / test / original rows):
- pipeline A: rain + / - previous irrigation, moisture / humidity, temperature x sunlight, field area x previous
  irrigation, EC / pH, carbon x moisture; a crop x season x irrigation-type key;
- pipeline B: 5-quantile bins (train edges) of rain, previous irrigation, moisture, humidity and temperature, their
  rounding (-1 / 0 decimals) and digits ((x * 10^k) mod 10, k = -2 .. 1 or 2), coded over train + test + original;
  4 category-pair keys; original mean class id and count of the pairs, rounds, digits and 5 categoricals;
- keys in the style of a public LightGBM notebook for the 11 numerics (> 7 levels): the value as a key, its count,
  original mean class id and size, count / size ratio; round / int / 5,000-quantile / equal-width (5,000 / 8,000 / 200)
  / log-width bins of rain, previous irrigation, moisture and humidity; integer / decimal digit keys of rain and
  previous irrigation.
Per fold: class rates of the crop x season x irrigation key (inner StratifiedKFold(5, shuffle, 42) for the training
rows) and sklearn multiclass TargetEncoder (cv 5, shuffle, smooth 'auto', random_state 42) encodings of that key, of
the 32 pipeline-B keys (plus their counts on the training rows and the row mean / std / min / max of the encodings) and
of the 51 LightGBM-style keys (plus mean / std / min / max); the keys are dropped, the remaining strings are coded over
the training + validation + test frames, and every column with > 256 distinct values except Rainfall_mm and
Previous_Irrigation_mm is replaced by its 256-quantile bin on the training rows (444 columns).
Model: XGBoost hist on GPU, depth 6, lr 0.03, subsample 0.8, colsample_bytree 0.4, min_child_weight 5, alpha 0.1,
lambda 1, gamma 0.05, max_bin 16,000, up to 1,000,000 rounds, early stopping 500 (mlogloss) on the validation fold,
random_state 42. Test = mean of the 5 fold models. Competition rows only (the original rows only feed the statistics).
About 3.5 minutes per fold on one GPU.
Usage: python train_union_prebinned_a.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)

PAIR = "xgb_union_te_prebinned_a"
ID_COL = ID
CLASS_IDS = [0, 1, 2]

N_FOLDS = 5
INNER_FOLDS = 5
SEED = 42
MAKE_PAIR_FEATURES = True
PAIR_MAX_UNIQUE = 20000
BIN_SMALL = 256
BIN_EXCLUDE_COLS = {"Rainfall_mm", "Previous_Irrigation_mm"}
BIN_VERBOSE_EVERY = 150
HIGH_CARD_THRESHOLD = 7
QCUT_BINS = 5000
CUT_BINS_RAIN = 5000
CUT_BINS_PREV = 8000

NUMS_CAND = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"]
Q_LIST = [5]
DIGIT_SPECS = {
    "Rainfall_mm": list(range(-2, 2)),
    "Previous_Irrigation_mm": list(range(-2, 3)),
    "Soil_Moisture": list(range(-2, 2)),
    "Humidity": list(range(-2, 2)),
    "Temperature_C": list(range(-2, 2)),
}
ROUND_SPECS = {
    "Rainfall_mm": [-1, 0],
    "Previous_Irrigation_mm": [-1, 0],
    "Soil_Moisture": [0],
    "Humidity": [0],
    "Temperature_C": [0],
}
A_STATS = ["mean", "std", "min", "max"]

CAT_COLS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]

XGB_PARAMS = {
    "n_estimators": 1_000_000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.05,
    "early_stopping_rounds": 500,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "enable_categorical": False,
    "device": "cuda",
    "tree_method": "hist",
    "max_bin": 16000,
    "verbosity": 0,
    "random_state": SEED,
}


def _reset(df):
    return df.reset_index(drop=True)


def sanitize(df):
    out = df.copy()
    for c in CAT_COLS:
        out[c] = out[c].astype(str).fillna("missing").str.strip()
    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
        out[c] = out[c].fillna(out[c].median())
    return out


def create_count_features(df, cols):
    out = {}
    for c in cols:
        vc = df[c].astype(str).value_counts(dropna=False)
        out[f"CE_{c}"] = df[c].astype(str).map(vc).fillna(0).astype("float32")
    return pd.DataFrame(out)


def create_groupby_features_original(comb_df, agg_inputs, orig_df, target):
    outs = {}
    orig_tmp = orig_df.copy()
    for cols, how in agg_inputs:
        key = cols[0]
        if how == "mean":
            gp = orig_tmp.groupby(key, observed=False)[target].mean().astype("float32")
            outs[f"GMEAN_{key}"] = comb_df[key].map(gp).fillna(gp.mean()).astype("float32")
        elif how == "size":
            gp = orig_tmp.groupby(key, observed=False).size().astype("float32")
            outs[f"GSIZE_{key}"] = comb_df[key].map(gp).fillna(0).astype("float32")
    return pd.DataFrame(outs)


def add_qcut_bins(train, test, orig, num_cols, q_list):
    for c in num_cols:
        for q in q_list:
            name = f"B_qcut_{c}_q{q}"
            try:
                b_train, bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates="drop")
                train[name] = b_train.astype("float32")
                test[name] = pd.cut(test[c], bins=bins, labels=False, include_lowest=True).astype("float32")
                orig[name] = pd.cut(orig[c], bins=bins, labels=False, include_lowest=True).astype("float32")
            except Exception:
                train[name] = 0.0; test[name] = 0.0; orig[name] = 0.0
    return train, test, orig


def add_round_features(train, test, orig, round_specs):
    round_cols = []
    for c, rr in round_specs.items():
        for r in rr:
            n = f"B_round_{c}_r{r}"
            train[n] = train[c].round(r)
            test[n] = test[c].round(r)
            orig[n] = orig[c].round(r)
            round_cols.append(n)
    return train, test, orig, round_cols


def add_digit_features(train, test, orig, digit_specs):
    digits = []
    for c, ks in digit_specs.items():
        for k in ks:
            n = f"B_digit_{c}_d{k}"
            for df in (train, test, orig):
                df[n] = (((df[c] * (10 ** k)) % 10).fillna(-1)).astype("int8")
            digits.append(n)
    return train, test, orig, digits


def factorize_across_all(train, test, orig, cols):
    for c in cols:
        if c not in train.columns or c not in test.columns or c not in orig.columns:
            continue
        combined = pd.concat([train[c], test[c], orig[c]], ignore_index=True)
        codes, _ = combined.factorize(sort=False)
        train[c] = codes[:len(train)]
        test[c] = codes[len(train):len(train) + len(test)]
        orig[c] = codes[len(train) + len(test):]
    return train, test, orig


def make_pair_feature(train, test, orig, cols, pair_max_unique=20000):
    name = "B_pair_" + "__".join(cols)
    for df in (train, test, orig):
        s = df[cols[0]].astype(str)
        for col in cols[1:]:
            s = s + "_" + df[col].astype(str)
        df[name] = s
    combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
    codes, _ = combined.factorize(sort=False)
    nuniq = pd.Series(codes).nunique()
    if nuniq > pair_max_unique:
        train.drop(columns=[name], inplace=True)
        test.drop(columns=[name], inplace=True)
        orig.drop(columns=[name], inplace=True)
        return None
    train[name] = codes[:len(train)]
    test[name] = codes[len(train):len(train) + len(test)]
    orig[name] = codes[len(train) + len(test):]
    return name


def add_orig_te_ce(train, test, orig, cols, y_orig):
    te_cols, ce_cols = [], []
    orig_tmp = orig.copy()
    orig_tmp["_y_orig_"] = y_orig.astype("float32")
    for c in cols:
        if c not in orig_tmp.columns:
            continue
        te_name = f"B_TEORIG_{c}"
        m = orig_tmp.groupby(c, observed=False)["_y_orig_"].mean().astype("float32")
        m.name = te_name
        train = train.merge(m, on=c, how="left")
        test = test.merge(m, on=c, how="left")
        fillv = float(orig_tmp["_y_orig_"].mean())
        train[te_name] = train[te_name].fillna(fillv)
        test[te_name] = test[te_name].fillna(fillv)
        te_cols.append(te_name)

        ce_name = f"B_CEORIG_{c}"
        vc = orig_tmp[c].value_counts(dropna=False).astype("float32")
        train[ce_name] = train[c].map(vc).fillna(0).astype("float32")
        test[ce_name] = test[c].map(vc).fillna(0).astype("float32")
        ce_cols.append(ce_name)
    return train, test, te_cols, ce_cols


def add_lgbm_port_features(train, test, orig, target, id_col, y_orig):
    base_cols = [c for c in train.columns if c not in [target, id_col] and not c.startswith("A_") and not c.startswith("B_")]
    base_cols = [c for c in base_cols if c in test.columns]
    comb = pd.concat([_reset(train[base_cols]), _reset(test[base_cols])], axis=0, ignore_index=True)
    n_all = len(comb)

    for c in base_cols:
        if comb[c].dtype == "object" or str(comb[c].dtype).startswith("string"):
            comb[c] = comb[c].astype(str).fillna("missing").str.strip()

    high_card = [c for c in base_cols if comb[c].nunique(dropna=False) > HIGH_CARD_THRESHOLD]
    blocks = {}
    if len(high_card) > 0:
        blocks["te1"] = _reset(comb[high_card].astype(str).add_prefix("L_").add_suffix("_te"))
        blocks["ce1"] = _reset(create_count_features(comb, high_card))

        orig_tmp = orig.copy()
        for c in high_card:
            if c in orig_tmp.columns and (orig_tmp[c].dtype == "object" or str(orig_tmp[c].dtype).startswith("string")):
                orig_tmp[c] = orig_tmp[c].astype(str).fillna("missing").str.strip()
        mean_inputs = [([c], "mean") for c in high_card if c in orig_tmp.columns]
        size_inputs = [([c], "size") for c in high_card if c in orig_tmp.columns]
        if len(mean_inputs) > 0:
            blocks["gmean"] = _reset(create_groupby_features_original(comb[high_card], mean_inputs, orig.assign(**{target: y_orig}), target))
        if len(size_inputs) > 0:
            blocks["gsize"] = _reset(create_groupby_features_original(comb[high_card], size_inputs, orig.assign(**{target: y_orig}), target))
        if "ce1" in blocks and "gsize" in blocks:
            ratios = {}
            ce_df = blocks["ce1"]; sz_df = blocks["gsize"]
            for j, ccol in enumerate(ce_df.columns):
                numer = ce_df.iloc[:, j].astype("float32")
                denom = sz_df.iloc[:, j].astype("float32")
                ratios[f"{ccol}_ratio"] = (numer / denom.replace(0, np.nan)).fillna(0.0)
            blocks["ratios"] = _reset(pd.DataFrame(ratios))

    num_bins = {}
    for col in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity"]:
        if col not in comb.columns:
            continue
        s = pd.to_numeric(comb[col], errors="coerce")
        num_bins[f"L_{col}_round_div2_te"] = (s.round().fillna(-1).astype("int64") // 2).astype(str)
        num_bins[f"L_{col}_int_te"] = s.fillna(-1).astype("int64").astype(str)
        try:
            q = int(min(QCUT_BINS, max(2, s.nunique(dropna=True))))
            num_bins[f"L_{col}_quantiled_te"] = pd.qcut(s, q=q, duplicates="drop").astype(str)
        except Exception:
            num_bins[f"L_{col}_quantiled_te"] = pd.Series(["NA"] * n_all)
        bins = CUT_BINS_RAIN if col == "Rainfall_mm" else (CUT_BINS_PREV if col == "Previous_Irrigation_mm" else 200)
        try:
            num_bins[f"L_{col}_binned_te"] = pd.cut(s, bins=bins, duplicates="drop").astype(str)
        except Exception:
            num_bins[f"L_{col}_binned_te"] = pd.Series(["NA"] * n_all)
        try:
            num_bins[f"L_{col}_log_binned_te"] = pd.cut(np.log1p(s.clip(lower=0)), bins=max(200, bins // 2), duplicates="drop").astype(str)
        except Exception:
            num_bins[f"L_{col}_log_binned_te"] = pd.Series(["NA"] * n_all)
    if len(num_bins) > 0:
        blocks["num_bins"] = _reset(pd.DataFrame(num_bins))

    digits = {}
    for col in ["Rainfall_mm", "Previous_Irrigation_mm"]:
        if col not in comb.columns:
            continue
        s = pd.to_numeric(comb[col], errors="coerce").fillna(0).astype("float64")
        ss = s.map(lambda x: f"{x:.6f}")
        sp = ss.str.split(".", n=1, expand=True)
        left = sp[0].str.replace("-", "", regex=False).str.zfill(4)
        right = sp[1].fillna("").str.ljust(6, "0")
        for i in range(4):
            digits[f"L_{col}_intd{i}_te"] = left.str[i]
        for i in range(4):
            digits[f"L_{col}_decd{i}_te"] = right.str[i]
        digits[f"L_{col}_d01_te"] = left.str[0] + left.str[1]
        digits[f"L_{col}_d23_te"] = left.str[2] + left.str[3]
    if len(digits) > 0:
        blocks["digits"] = _reset(pd.DataFrame(digits))

    out = pd.concat([comb] + [v for v in blocks.values()], axis=1)
    n_tr = len(train); n_te = len(test)
    train2 = out.iloc[:n_tr].copy().reset_index(drop=True)
    test2 = out.iloc[n_tr:n_tr+n_te].copy().reset_index(drop=True)
    te_cols = [c for c in train2.columns if c.endswith("_te")]
    return train2, test2, te_cols


def fold_count_encode(X_tr, X_va, X_te, cols):
    for c in cols:
        vc = X_tr[c].astype(str).value_counts(dropna=False)
        X_tr[f"CE_{c}"] = X_tr[c].astype(str).map(vc).fillna(0).astype("float32")
        X_va[f"CE_{c}"] = X_va[c].astype(str).map(vc).fillna(0).astype("float32")
        X_te[f"CE_{c}"] = X_te[c].astype(str).map(vc).fillna(0).astype("float32")
    return X_tr, X_va, X_te


def prebin_except_selected(X_tr, X_va, X_te, exclude_cols):
    X_tr = X_tr.copy(); X_va = X_va.copy(); X_te = X_te.copy()
    cols = list(X_tr.columns)
    for i, c in enumerate(cols, start=1):
        if c in exclude_cols:
            continue
        vals = pd.to_numeric(X_tr[c], errors="coerce")
        if vals.nunique(dropna=True) <= BIN_SMALL:
            continue
        try:
            _, edges = pd.qcut(vals, q=BIN_SMALL, retbins=True, duplicates='drop')
            X_tr[c] = pd.cut(pd.to_numeric(X_tr[c], errors='coerce'), bins=edges, labels=False, include_lowest=True).fillna(-1).astype('float32')
            X_va[c] = pd.cut(pd.to_numeric(X_va[c], errors='coerce'), bins=edges, labels=False, include_lowest=True).fillna(-1).astype('float32')
            X_te[c] = pd.cut(pd.to_numeric(X_te[c], errors='coerce'), bins=edges, labels=False, include_lowest=True).fillna(-1).astype('float32')
        except Exception:
            pass
        if i % BIN_VERBOSE_EVERY == 0:
            print(f"  pre-binned {i}/{len(cols)} columns")
    return X_tr, X_va, X_te


def build_features():
    print('Loading data...')
    train = sanitize(read_train())
    test = sanitize(read_test())
    orig = sanitize(read_original())
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(int)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(int)
    y = train[TARGET].values
    y_orig = orig[TARGET].values

    # pipeline A / B base
    train_base = train.drop(columns=[TARGET]).copy()
    test_base = test.copy()
    orig_base = orig.drop(columns=[TARGET]).copy()
    for df in (train_base, test_base, orig_base):
        df["A_water_total"] = (df["Rainfall_mm"] + df["Previous_Irrigation_mm"]).astype("float32")
        df["A_water_gap"] = (df["Rainfall_mm"] - df["Previous_Irrigation_mm"]).astype("float32")
        df["A_ratio_moist_hum"] = (df["Soil_Moisture"] / (df["Humidity"] + 1e-3)).astype("float32")
        df["A_temp_sun_x"] = (df["Temperature_C"] * df["Sunlight_Hours"]).astype("float32")
        df["A_field_prev_x"] = (df["Field_Area_hectare"] * df["Previous_Irrigation_mm"]).astype("float32")
        df["A_cond_ph_ratio"] = (df["Electrical_Conductivity"] / (df["Soil_pH"] + 1e-3)).astype("float32")
        df["A_org_moist_x"] = (df["Organic_Carbon"] * df["Soil_Moisture"]).astype("float32")
        df["A_crop_season_irr_te"] = df[["Crop_Type","Season","Irrigation_Type"]].astype(str).agg("_".join, axis=1)

    train_base, test_base, orig_base = add_qcut_bins(train_base, test_base, orig_base, NUMS_CAND, Q_LIST)
    train_base, test_base, orig_base, round_cols = add_round_features(train_base, test_base, orig_base, ROUND_SPECS)
    train_base, test_base, orig_base, digit_cols = add_digit_features(train_base, test_base, orig_base, DIGIT_SPECS)
    train_base, test_base, orig_base = factorize_across_all(train_base, test_base, orig_base, round_cols + digit_cols + [c for c in train_base.columns if c.startswith('B_qcut_')])

    pair_cols = []
    if MAKE_PAIR_FEATURES:
        for cols in [("Crop_Type", "Season"), ("Crop_Type", "Irrigation_Type"), ("Water_Source", "Region"), ("Soil_Type", "Crop_Type")]:
            p = make_pair_feature(train_base, test_base, orig_base, cols, pair_max_unique=PAIR_MAX_UNIQUE)
            if p is not None:
                pair_cols.append(p)

    train_base, test_base, teorig_cols, ceorig_cols = add_orig_te_ce(
        train_base, test_base, orig_base,
        [c for c in pair_cols + round_cols + digit_cols + ["Crop_Type", "Season", "Irrigation_Type", "Water_Source", "Region"] if c in train_base.columns],
        y_orig,
    )
    train_lgb, test_lgb, LGBM_TE_COLUMNS = add_lgbm_port_features(train_base.copy(), test_base.copy(), orig_base.copy(), TARGET, ID_COL, y_orig)

    # merge lgb port back onto union base
    extra_cols = [c for c in train_lgb.columns if c not in train_base.columns]
    train = pd.concat([_reset(train_base), _reset(train_lgb[extra_cols])], axis=1)
    test = pd.concat([_reset(test_base), _reset(test_lgb[extra_cols])], axis=1)

    A_TE_COLUMNS = [c for c in ["A_crop_season_irr_te"] if c in train.columns]
    B_TE_COLUMNS = [c for c in pair_cols + round_cols + digit_cols if c in train.columns]
    B_CE_COLUMNS = [c for c in pair_cols + round_cols + digit_cols if c in train.columns]
    BASE_UNION_FEATURES = [c for c in train.columns if c != ID_COL]
    DROP_KEYS_AFTER_ENCODING = list(dict.fromkeys(A_TE_COLUMNS + B_TE_COLUMNS + LGBM_TE_COLUMNS))

    print(f'Base union feature count = {len(BASE_UNION_FEATURES)}')
    print(f'A_TE = {len(A_TE_COLUMNS)}, B_TE = {len(B_TE_COLUMNS)}, LGBM_TE = {len(LGBM_TE_COLUMNS)}')
    return (train, test, y, BASE_UNION_FEATURES, A_TE_COLUMNS, B_TE_COLUMNS, B_CE_COLUMNS, LGBM_TE_COLUMNS,
            DROP_KEYS_AFTER_ENCODING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        (train, test, y, BASE_UNION_FEATURES, A_TE_COLUMNS, B_TE_COLUMNS, B_CE_COLUMNS, LGBM_TE_COLUMNS,
         DROP_KEYS_AFTER_ENCODING) = build_features()
        fold_id = folds(y)
        for k in todo:
            t_fold = time.time()
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr_base = train.iloc[tr_idx][BASE_UNION_FEATURES].copy()
            y_tr = y[tr_idx].copy()
            X_va_base = train.iloc[va_idx][BASE_UNION_FEATURES].copy()
            y_va = y[va_idx].copy()
            X_te_base = test[BASE_UNION_FEATURES].copy()

            # A) TE1 inner-fold stats
            X_trA = X_tr_base.copy()
            X_trA[TARGET] = y_tr
            skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
            for _, (in_tr, in_va) in enumerate(skf_inner.split(X_trA, y_tr)):
                X_in_tr = X_trA.iloc[in_tr][A_TE_COLUMNS + [TARGET]].copy()
                X_in_va = X_trA.iloc[in_va][A_TE_COLUMNS].copy()
                for col in A_TE_COLUMNS:
                    tmp = pd.get_dummies(X_in_tr[TARGET], prefix='cls')
                    tmp[col] = X_in_tr[col].values
                    grp = tmp.groupby(col, observed=False).mean()
                    for cls in CLASS_IDS:
                        cname = f"A_TE1_{col}_c{cls}"
                        vals = X_in_va[col].map(grp.get(f'cls_{cls}', pd.Series(dtype='float32')))
                        X_trA.loc[X_trA.index[in_va], cname] = vals.fillna((y_tr == cls).mean()).values.astype('float32')
            for col in A_TE_COLUMNS:
                tmp = pd.get_dummies(pd.Series(y_tr), prefix='cls')
                tmp[col] = X_tr_base[col].values
                grp = tmp.groupby(col, observed=False).mean()
                for cls in CLASS_IDS:
                    cname = f"A_TE1_{col}_c{cls}"
                    vals_va = X_va_base[col].map(grp.get(f'cls_{cls}', pd.Series(dtype='float32')))
                    vals_te = X_te_base[col].map(grp.get(f'cls_{cls}', pd.Series(dtype='float32')))
                    X_va_base[cname] = vals_va.fillna((y_tr == cls).mean()).astype('float32')
                    X_te_base[cname] = vals_te.fillna((y_tr == cls).mean()).astype('float32')
                    X_tr_base[cname] = pd.to_numeric(X_trA.get(cname, 0), errors='coerce').fillna(0).astype('float32').values

            # A) TargetEncoder on A_TE_COLUMNS
            if len(A_TE_COLUMNS) > 0:
                A_TE_MEAN_COLS = [f"A_TE_{c}" for c in A_TE_COLUMNS for _ in CLASS_IDS]
                teA = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth="auto", target_type="multiclass", random_state=SEED)
                X_tr_base[[f"A_TE_{c}_{j}" for c in A_TE_COLUMNS for j in CLASS_IDS]] = teA.fit_transform(X_tr_base[A_TE_COLUMNS], y_tr)
                X_va_base[[f"A_TE_{c}_{j}" for c in A_TE_COLUMNS for j in CLASS_IDS]] = teA.transform(X_va_base[A_TE_COLUMNS])
                X_te_base[[f"A_TE_{c}_{j}" for c in A_TE_COLUMNS for j in CLASS_IDS]] = teA.transform(X_te_base[A_TE_COLUMNS])

            # B) Count encode selected cols
            if len(B_CE_COLUMNS) > 0:
                X_tr_base, X_va_base, X_te_base = fold_count_encode(X_tr_base, X_va_base, X_te_base, B_CE_COLUMNS)

            # B) TargetEncoder on B_TE_COLUMNS
            if len(B_TE_COLUMNS) > 0:
                teB = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth="auto", target_type="multiclass", random_state=SEED)
                bcols = [f"B_TE_{c}_{j}" for c in B_TE_COLUMNS for j in CLASS_IDS]
                X_tr_base[bcols] = teB.fit_transform(X_tr_base[B_TE_COLUMNS], y_tr)
                X_va_base[bcols] = teB.transform(X_va_base[B_TE_COLUMNS])
                X_te_base[bcols] = teB.transform(X_te_base[B_TE_COLUMNS])
                for agg in ["mean", "std", "min", "max"]:
                    X_tr_base[f"B_TEagg_{agg}"] = X_tr_base[bcols].agg(agg, axis=1).astype("float32")
                    X_va_base[f"B_TEagg_{agg}"] = X_va_base[bcols].agg(agg, axis=1).astype("float32")
                    X_te_base[f"B_TEagg_{agg}"] = X_te_base[bcols].agg(agg, axis=1).astype("float32")

            # LGBM port TE columns
            if len(LGBM_TE_COLUMNS) > 0:
                teL = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth="auto", target_type="multiclass", random_state=SEED)
                lcols = [f"L_TE_{c}_{j}" for c in LGBM_TE_COLUMNS for j in CLASS_IDS]
                X_tr_base[lcols] = teL.fit_transform(X_tr_base[LGBM_TE_COLUMNS], y_tr)
                X_va_base[lcols] = teL.transform(X_va_base[LGBM_TE_COLUMNS])
                X_te_base[lcols] = teL.transform(X_te_base[LGBM_TE_COLUMNS])
                for agg in ["mean", "std", "min", "max"]:
                    X_tr_base[f"L_TEagg_{agg}"] = X_tr_base[lcols].agg(agg, axis=1).astype("float32")
                    X_va_base[f"L_TEagg_{agg}"] = X_va_base[lcols].agg(agg, axis=1).astype("float32")
                    X_te_base[f"L_TEagg_{agg}"] = X_te_base[lcols].agg(agg, axis=1).astype("float32")

            drop_now = [c for c in DROP_KEYS_AFTER_ENCODING if c in X_tr_base.columns]
            X_tr = X_tr_base.drop(columns=drop_now, errors="ignore")
            X_va = X_va_base.drop(columns=drop_now, errors="ignore")
            X_te = X_te_base.drop(columns=drop_now, errors="ignore")

            obj_cols = sorted(
                set(X_tr.select_dtypes(include=["object", "string"]).columns)
                | set(X_va.select_dtypes(include=["object", "string"]).columns)
                | set(X_te.select_dtypes(include=["object", "string"]).columns)
            )
            for c in obj_cols:
                cats = pd.concat([X_tr[c], X_va[c], X_te[c]], axis=0).astype(str)
                codes, _ = pd.factorize(cats, sort=False)
                ntr, nva = len(X_tr), len(X_va)
                X_tr[c] = codes[:ntr]
                X_va[c] = codes[ntr:ntr+nva]
                X_te[c] = codes[ntr+nva:]
            for df_ in (X_tr, X_va, X_te):
                for c in df_.columns:
                    df_[c] = pd.to_numeric(df_[c], errors='coerce').fillna(0).astype('float32')

            X_tr, X_va, X_te = prebin_except_selected(X_tr, X_va, X_te, BIN_EXCLUDE_COLS)

            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
            p_va = model.predict_proba(X_va).astype(np.float32)
            p_te = model.predict_proba(X_te).astype(np.float32)
            save_part(HERE, PAIR, k, p_va, p_te)
            fold_ba = balanced_accuracy_score(y_va, p_va.argmax(axis=1))
            print(f"{PAIR} fold {k}: BAC = {fold_ba:.8f} | time={(time.time()-t_fold)/60:.1f} min", flush=True)
            del X_tr_base, X_va_base, X_te_base, X_trA, X_tr, X_va, X_te, model, p_va, p_te
            gc.collect()
    y = read_train()[TARGET].map(TARGET_MAP).astype(int).values
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
