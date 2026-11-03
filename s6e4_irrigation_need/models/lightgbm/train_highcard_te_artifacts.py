#!/usr/bin/env python
"""LightGBM on value tokens, counts, original priors and digit / quantile tokens with nested target encodings
(pair lgbm_highcard_te_artifact_tokens_a).

Label-free features (train + test rows; the original rows give the priors):
  - the 11 numerics and the 8 categoricals (pandas categories, native in LightGBM)
  - every column with > 7 distinct values (the 11 numerics): its value as a string key, its count over train + test,
    its original row count and per-class original frequencies, and count / original count
  - rainfall, previous irrigation, soil moisture, temperature, humidity as string keys: round(x) // 2, int(x),
    <= 5000 quantile bins, <= 5000 equal-width bins
  - 6 category-pair keys and their counts
  - rainfall, previous irrigation, soil moisture, temperature: 5-quantile bin (edges from the train rows) and the
    digits at 100 / 10 / 1 / 0.1, as string keys
  - soil moisture / (rainfall + 1), previous irrigation / (rainfall + 1), temperature x wind speed, rainfall - previous
    irrigation, soil moisture + previous irrigation
Per fold: each of the 57 string keys is replaced by its 3 class frequencies, out-of-fold over an inner
StratifiedKFold(5, seed 42) of the training rows and from all training rows for the validation / test rows.
Model: LightGBM multiclass, lr 0.005, max_depth 4, subsample 0.5, colsample_bytree 0.2, reg_lambda 15, reg_alpha 10,
seed 42, 32 threads, up to 20,000 rounds with early stopping (500) on the validation fold.
Test = mean of the 5 fold models.
Usage: python train_highcard_te_artifacts.py"""
import gc
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

PAIR = "lgbm_highcard_te_artifact_tokens_a"
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5

CAT_COLS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
HIGH_CARD_THRESHOLD = 7
Q_LIST = [5]
DIGIT_COLS = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Temperature_C"]
ROUND_SPECS = {
    "Rainfall_mm": [-1, 0],
    "Previous_Irrigation_mm": [-1, 0],
    "Soil_Moisture": [0, 1],
    "Temperature_C": [0, 1],
}

LGB_PARAMS = dict(
    objective="multiclass",
    metric="multi_logloss",
    num_class=3,
    n_estimators=20000,
    learning_rate=0.005,
    max_depth=4,
    subsample=0.5,
    colsample_bytree=0.2,
    reg_lambda=15.0,
    reg_alpha=10.0,
    random_state=SEED,
    verbosity=-1,
    n_jobs=32,
)
EARLY_STOPPING_ROUNDS = 500
LOG_EVERY = 500


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reset(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)


def label_encode_across_all(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols):
    for c in cols:
        all_vals = pd.concat(
            [train[c].astype("string"), test[c].astype("string"), orig[c].astype("string")],
            ignore_index=True,
        ).fillna("__MISSING__")
        train[c] = pd.Categorical(all_vals[: len(train)])
        test[c] = pd.Categorical(all_vals[len(train): len(train) + len(test)])
        orig[c] = pd.Categorical(all_vals[len(train) + len(test):])
    return train, test, orig


def add_count_features(df: pd.DataFrame, inputs):
    out = {}
    for cols in inputs:
        cols = [cols] if isinstance(cols, str) else list(cols)
        name = f"count({'__'.join(cols)})"
        out[name] = df.groupby(cols, dropna=False).transform("size")
    return pd.DataFrame(out, index=df.index)


def add_groupby_original_features(df: pd.DataFrame, orig: pd.DataFrame, groups):
    out = {}
    y_orig = orig[TARGET].map(TARGET_MAP).astype(np.int32)
    orig_local = orig.copy()
    orig_local["_target_int"] = y_orig
    for cols in groups:
        cols = [cols] if isinstance(cols, str) else list(cols)
        key = "__".join(cols)
        size_map = orig_local.groupby(cols, dropna=False)["_target_int"].size()
        out[f"orig_size__{key}"] = df[cols].merge(size_map.rename("v"), on=cols, how="left")["v"].fillna(0).astype(np.float32)
        for k in range(3):
            mean_map = orig_local.groupby(cols, dropna=False)["_target_int"].apply(lambda s, kk=k: (s == kk).mean())
            out[f"orig_p{k}__{key}"] = (
                df[cols].merge(mean_map.rename("v"), on=cols, how="left")["v"].fillna((y_orig == k).mean()).astype(np.float32)
            )
    return pd.DataFrame(out, index=df.index)


def add_qcut_bins_3way(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols, q_list):
    new_cols = []
    for c in cols:
        if c not in train.columns:
            continue
        for q in q_list:
            name = f"B_qcut_{c}_q{q}"
            try:
                tr_vals = pd.to_numeric(train[c], errors="coerce")
                b_tr, bins = pd.qcut(tr_vals, q=q, labels=False, retbins=True, duplicates="drop")
                train[name] = pd.Categorical(b_tr.astype("float32").fillna(-1).astype("int32"))
                test[name] = pd.Categorical(pd.cut(pd.to_numeric(test[c], errors="coerce"), bins=bins, labels=False, include_lowest=True).astype("float32").fillna(-1).astype("int32"))
                orig[name] = pd.Categorical(pd.cut(pd.to_numeric(orig[c], errors="coerce"), bins=bins, labels=False, include_lowest=True).astype("float32").fillna(-1).astype("int32"))
            except Exception:
                train[name] = pd.Categorical(np.zeros(len(train), dtype=np.int32))
                test[name] = pd.Categorical(np.zeros(len(test), dtype=np.int32))
                orig[name] = pd.Categorical(np.zeros(len(orig), dtype=np.int32))
            new_cols.append(name)
    return train, test, orig, new_cols


def add_round_features_3way(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, round_specs):
    out = []
    for c, rr in round_specs.items():
        for r in rr:
            name = f"B_round_{c}_r{r}"
            train[name] = pd.to_numeric(train[c], errors="coerce").round(r).astype(np.float32)
            test[name] = pd.to_numeric(test[c], errors="coerce").round(r).astype(np.float32)
            orig[name] = pd.to_numeric(orig[c], errors="coerce").round(r).astype(np.float32)
            out.append(name)
    return train, test, orig, out


def add_digit_features_3way(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols):
    out = []
    for c in cols:
        for k in range(-2, 2):
            name = f"B_digit_{c}_d{k}"
            for df in (train, test, orig):
                v = pd.to_numeric(df[c], errors="coerce")
                df[name] = pd.Categorical((((v * (10 ** k)) % 10).fillna(-1)).astype("int8"))
            out.append(name)
    return train, test, orig, out


def add_formula_features(df: pd.DataFrame):
    out = pd.DataFrame(index=df.index)
    sm = pd.to_numeric(df["Soil_Moisture"], errors="coerce")
    rm = pd.to_numeric(df["Rainfall_mm"], errors="coerce")
    pi = pd.to_numeric(df["Previous_Irrigation_mm"], errors="coerce")
    tc = pd.to_numeric(df["Temperature_C"], errors="coerce")
    ws = pd.to_numeric(df["Wind_Speed_kmh"], errors="coerce")
    out["A_moisture_rain_ratio"] = (sm / (rm + 1.0)).astype(np.float32)
    out["A_prev_rain_ratio"] = (pi / (rm + 1.0)).astype(np.float32)
    out["A_temp_wind"] = (tc * ws).astype(np.float32)
    out["A_rain_minus_prev"] = (rm - pi).astype(np.float32)
    out["A_moisture_plus_prev"] = (sm + pi).astype(np.float32)
    return out


def make_num_token_blocks(combined: pd.DataFrame):
    out = pd.DataFrame(index=combined.index)
    selected = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Temperature_C", "Humidity"]
    for c in selected:
        s = pd.to_numeric(combined[c], errors="coerce")
        out[f"{c}_round_div2_te"] = (s.round().fillna(-1).astype(np.int64) // 2).astype(str)
        out[f"{c}_int_te"] = s.fillna(-1).astype(np.int64).astype(str)
        try:
            q = min(5000, max(2, int(s.nunique(dropna=True))))
            out[f"{c}_quantiled_te"] = pd.qcut(s, q=q, duplicates="drop").astype(str)
        except Exception:
            out[f"{c}_quantiled_te"] = "NA"
        try:
            bins = min(5000, max(2, int(s.nunique(dropna=True))))
            out[f"{c}_cut_te"] = pd.cut(s, bins=bins, duplicates="drop").astype(str)
        except Exception:
            out[f"{c}_cut_te"] = "NA"
    return out


def nested_multiclass_te(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, X_te: pd.DataFrame, te_cols):
    global_priors = np.bincount(y_tr, minlength=3) / len(y_tr)
    X_tr_num = X_tr.drop(columns=te_cols).copy()
    X_va_num = X_va.drop(columns=te_cols).copy()
    X_te_num = X_te.drop(columns=te_cols).copy()

    for col in te_cols:
        tr_s = X_tr[col].astype(str).fillna("NA")
        va_s = X_va[col].astype(str).fillna("NA")
        te_s = X_te[col].astype(str).fillna("NA")

        oof_block = np.zeros((len(X_tr), 3), dtype=np.float32)
        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        for itr, iva in inner.split(np.zeros(len(X_tr)), y_tr):
            fit_s = tr_s.iloc[itr]
            fit_y = y_tr[itr]
            apply_s = tr_s.iloc[iva]
            for k in range(3):
                means = pd.DataFrame({"key": fit_s.values, "y": (fit_y == k).astype(np.float32)}).groupby("key")["y"].mean()
                oof_block[iva, k] = apply_s.map(means).fillna(global_priors[k]).to_numpy(np.float32)

        for k in range(3):
            means = pd.DataFrame({"key": tr_s.values, "y": (y_tr == k).astype(np.float32)}).groupby("key")["y"].mean()
            X_tr_num[f"{col}_p{k}"] = oof_block[:, k]
            X_va_num[f"{col}_p{k}"] = va_s.map(means).fillna(global_priors[k]).to_numpy(np.float32)
            X_te_num[f"{col}_p{k}"] = te_s.map(means).fillna(global_priors[k]).to_numpy(np.float32)

    return X_tr_num, X_va_num, X_te_num


def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()

    y = train[TARGET].map(TARGET_MAP).astype(np.int32).values
    X = train.drop(columns=[TARGET, ID])
    T = test.drop(columns=[ID])
    orig_X = orig.drop(columns=[TARGET])
    X, T, orig_X = label_encode_across_all(X, T, orig_X, CAT_COLS)

    combined = pd.concat([reset(X), reset(T)], axis=0, ignore_index=True)
    orig_local = orig_X.copy()
    orig_local[TARGET] = orig[TARGET].values

    log("Features: value tokens, counts, original priors")
    high_cardinality = [c for c in combined.columns if combined[c].nunique(dropna=False) > HIGH_CARD_THRESHOLD]
    te1 = reset(combined[high_cardinality].astype(str).add_suffix("_te"))
    ce1 = reset(add_count_features(combined, high_cardinality))
    te1_original = reset(add_groupby_original_features(combined, orig_local, high_cardinality))

    count_ratios = {}
    for c in ce1.columns:
        base = c.replace("count(", "").replace(")", "")
        size_col = f"orig_size__{base.replace(',', '__')}"
        if size_col in te1_original.columns:
            denom = te1_original[size_col].replace(0, np.nan).astype(np.float32)
            count_ratios[f"{c}_ratio"] = (ce1[c].astype(np.float32) / denom).fillna(0.0)
    count_ratios = reset(pd.DataFrame(count_ratios))

    num_bins = reset(make_num_token_blocks(combined))

    combo_pairs = [
        ("Crop_Type", "Season"),
        ("Irrigation_Type", "Water_Source"),
        ("Crop_Type", "Irrigation_Type"),
        ("Season", "Crop_Growth_Stage"),
        ("Soil_Type", "Crop_Type"),
        ("Region", "Crop_Type"),
    ]
    combo_df = pd.DataFrame(index=combined.index)
    for a, b in combo_pairs:
        combo_df[f"{a}__{b}_te"] = combined[[a, b]].astype(str).agg("|".join, axis=1)
    combo_counts = reset(add_count_features(pd.concat([combined, combo_df], axis=1), list(combo_df.columns)))

    log("Features: quantile bins and digits")
    Xb = X.copy()
    Tb = T.copy()
    Ob = orig_X.copy()
    Xb, Tb, Ob, b_qcuts = add_qcut_bins_3way(Xb, Tb, Ob, ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Temperature_C"], Q_LIST)
    Xb, Tb, Ob, b_round = add_round_features_3way(Xb, Tb, Ob, ROUND_SPECS)
    Xb, Tb, Ob, b_digits = add_digit_features_3way(Xb, Tb, Ob, DIGIT_COLS)

    combined_b = pd.concat([reset(Xb), reset(Tb)], axis=0, ignore_index=True)
    b_tokens = combined_b[b_qcuts].astype(str).add_suffix("_te")
    for c in b_digits:
        b_tokens[f"{c}_te"] = combined_b[c].astype(str)

    log("Features: ratios")
    formula = pd.concat([add_formula_features(X), add_formula_features(T)], axis=0, ignore_index=True)

    X_fe = pd.concat([combined, te1, ce1, te1_original, count_ratios, num_bins, combo_df, combo_counts, b_tokens, formula], axis=1)
    T_fe = X_fe.iloc[len(X):].reset_index(drop=True)
    X_fe = X_fe.iloc[: len(X)].reset_index(drop=True)
    te_cols = [c for c in X_fe.columns if c.endswith("_te")]
    return X_fe, T_fe, y, te_cols


def main():
    X_fe, T_fe, y, te_cols = build_features()
    log(f"Final features before fold TE: {X_fe.shape[1]} | TE cols: {len(te_cols)}")

    fold_ids = folds(y)
    oof = np.zeros((len(X_fe), 3), dtype=np.float32)
    pred_test = np.zeros((len(T_fe), 3), dtype=np.float32)
    for k in range(N_FOLDS):
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        t_fold = time.time()
        log(f"---- Fold {k} ----")

        X_tr = X_fe.iloc[tr_idx].copy()
        y_tr = y[tr_idx].copy()
        X_va = X_fe.iloc[va_idx].copy()
        y_va = y[va_idx].copy()
        X_te = T_fe.copy()

        X_tr_num, X_va_num, X_te_num = nested_multiclass_te(X_tr, y_tr, X_va, X_te, te_cols)
        cat_cols_fold = [c for c in X_tr_num.columns if str(X_tr_num[c].dtype) == "category"]
        cat_idx = [X_tr_num.columns.get_loc(c) for c in cat_cols_fold]

        model = lgbm.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr_num,
            y_tr,
            eval_set=[(X_va_num, y_va)],
            eval_metric="multi_logloss",
            callbacks=[
                lgbm.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False),
                lgbm.log_evaluation(LOG_EVERY),
            ],
            categorical_feature=cat_idx,
        )

        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None:
            best_iter = LGB_PARAMS["n_estimators"]
        p_va = model.predict_proba(X_va_num, num_iteration=best_iter)
        p_te = model.predict_proba(X_te_num, num_iteration=best_iter)
        oof[va_idx] = p_va.astype(np.float32)
        pred_test += p_te.astype(np.float32) / N_FOLDS

        bac = balanced_accuracy_score(y_va, p_va.argmax(axis=1))
        log(f"Fold {k} BAC: {bac:.6f} | best_iter={best_iter} | elapsed {time.time() - t_fold:.1f}s | X_tr shape {X_tr_num.shape}")

        del X_tr, X_va, X_te, X_tr_num, X_va_num, X_te_num, model
        gc.collect()

    log(f"{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, oof.argmax(axis=1)):.6f}")
    save_pair(HERE, PAIR, oof, pred_test)


if __name__ == "__main__":
    main()
