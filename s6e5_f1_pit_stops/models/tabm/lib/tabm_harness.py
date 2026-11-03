"""TabM (pytabkit TabM_D_Classifier) on fold-local features of the competition rows plus all original rows.

fold_features(): per outer fold, the base features are fitted on the fold's competition training rows and applied to the
validation, test and original rows:
  - the 14 columns, _LP_RP = LapNumber / (RaceProgress + 1e-6) and _TL_LP = TyreLife / max(LapNumber, 1)
  - every numeric column and the two ratios floored and factorized as a category (<col>_cat_; unseen values -1)
  - row counts of Driver, Compound, Race, Year and PitStop values in the fold's training rows
  - RaceProgress in 200 quantile bins (category), Race x Compound and Race x Year (categories)
then, for every frame:
  - artifact digits (categories) of LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress: integer digits 1-3,
    decimal digits 1-2 and the 4 lowest digits of round(100 |x|)
  - SOURCE_is_original (1 for the original rows)
  - group stats over the fold's training + validation + test + original rows: LapTime (s), LapTime_Delta,
    Cumulative_Degradation, TyreLife minus their group mean and their z-score, and the group size, within Race, Compound,
    Driver and Race x Compound
run_fold(): the training rows are the fold's competition rows followed by all original rows; Race x Compound and Race x Year
get a scikit-learn TargetEncoder (smooth 'auto', 5-fold cross fitting on the training rows). The model treats the string
columns as categoricals and stops early on the validation fold. Seeds use the fold number f = k + 1: global seeds 42 + f,
TE 42 + f, model 42 + f + model_seed_offset."""
import os
import random
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

from common.data import ID, TARGET, read_original, read_test, read_train

EPS = 1e-6

BASE_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_state": 42,
    "verbosity": 2,
    "val_metric_name": "1-auc_ovr",
    "n_epochs": 40,
    "patience": 6,
    "n_refit": 0,
    "batch_size": 512,
    "allow_amp": True,
    "gradient_clipping_norm": 1.0,
    "share_training_batches": False,
    "arch_type": "tabm-mini",
    "tabm_k": 32,
    "n_blocks": 3,
    "d_block": 256,
    "dropout": 0.10,
    "num_emb_type": "pwl",
    "d_embedding": 16,
    "num_emb_n_bins": 32,
    "lr": 7e-4,
    "weight_decay": 1e-5,
}


@dataclass
class TabmConfig:
    seed: int = 42
    model_seed_offset: int = 0
    params_override: Dict = field(default_factory=dict)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clean_name(col):
    return str(col).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def tabm_params(cfg, fold_no):
    params = dict(BASE_PARAMS)
    params.update(cfg.params_override)
    params["random_state"] = int(cfg.seed + fold_no + cfg.model_seed_offset)
    return params


def base_features(df, fit=False, state=None):
    """Ratios, floored-value categories, value counts, RaceProgress quantile bins and the two combos. fit=True learns the
    factorizations / counts / bins (state) from df; fit=False applies a fitted state."""
    df = df.copy()
    if state is None:
        state = {}

    cat_cols_base = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols_base = df.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()
    num_cols_base = [c for c in num_cols_base if c not in [ID, TARGET]]

    df["_LP_RP"] = (pd.to_numeric(df["LapNumber"], errors="coerce") / (pd.to_numeric(df["RaceProgress"], errors="coerce") + EPS)).astype("float32")
    df["_TL_LP"] = (pd.to_numeric(df["TyreLife"], errors="coerce") / pd.to_numeric(df["LapNumber"], errors="coerce").clip(lower=1)).astype("float32")

    for col in num_cols_base + ["_LP_RP", "_TL_LP"]:
        cat_name = f"{col}_cat_" if col in num_cols_base else f"{col[1:]}_cat_"
        x = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if fit:
            _, uniques = np.floor(x).factorize()
            state[col] = uniques
        uniques = state[col]
        code_map = {cat: i for i, cat in enumerate(uniques)}
        codes = np.floor(x).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes.astype(str)

    for col in cat_cols_base + ["Year_cat_", "PitStop_cat_"]:
        cnt_name = f"_{col}_count" if col in cat_cols_base else f"_{col[:-1]}_count"
        if fit:
            state[cnt_name] = df[col].astype(str).value_counts(dropna=False)
        cm = state[cnt_name]
        df[cnt_name] = df[col].astype(str).map(cm).fillna(0).astype("int32")

    for col, nb in [("RaceProgress", 200)]:
        bn = f"{col}_{nb}_qbin_"
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
        if fit:
            kb = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy="quantile", subsample=None)
            kb.fit(x)
            state[bn] = kb
        df[bn] = state[bn].transform(x).ravel().astype("int32").astype(str)

    combos = [("Race", "Compound"), ("Race", "Year")]
    combo_names = []
    for cols in combos:
        cn = "_".join(cols) + "_"
        combo_names.append(cn)
        s = df[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "_" + df[c].astype(str)
        if fit:
            _, uniques = pd.factorize(s, sort=False)
            state[cn] = uniques
        uniques = state[cn]
        code_map = {cat: i for i, cat in enumerate(uniques)}
        codes = s.map(code_map).fillna(-1).astype("int32")
        df[cn] = codes.astype(str)

    return df, state, combo_names


def add_artifact_digit_features(df):
    out = df.copy()
    cols = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]
    for col in [c for c in cols if c in out.columns]:
        x = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(float).abs()
        safe = clean_name(col)
        for i in range(3):
            out[f"DIG_{safe}_id{i+1}_"] = ((x // (10 ** i)) % 10).astype("int8").astype(str)
        if pd.api.types.is_float_dtype(out[col]):
            for i in range(1, 3):
                out[f"DIG_{safe}_dd{i}_"] = ((x * (10 ** i)).round().astype(int) % 10).astype("int8").astype(str)
        sc = (x * 100).round().astype(int).abs()
        for i in range(4):
            out[f"SIG_{safe}_s{i+1}_"] = ((sc // (10 ** i)) % 10).astype("int8").astype(str)
    return out


def add_group_stat_features_parts(parts):
    combined = pd.concat(parts, axis=0, ignore_index=True)
    group_cols = [c for c in ["Race", "Compound", "Driver", "Race_Compound_"] if c in combined.columns]
    value_cols = [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "TyreLife"] if c in combined.columns]
    new_cols = []
    for g in group_cols:
        key = combined[g].astype(str)
        for v in value_cols:
            vals = pd.to_numeric(combined[v], errors="coerce").fillna(0).astype("float32")
            safe_v = clean_name(v)
            safe_g = clean_name(g)
            mean = vals.groupby(key).transform("mean")
            std = vals.groupby(key).transform("std").fillna(0)
            cnt = key.map(key.value_counts(dropna=False)).astype("float32")
            z_name = f"GST_{safe_v}_minus_{safe_g}_mean"
            s_name = f"GST_{safe_v}_z_{safe_g}"
            c_name = f"GST_{safe_g}_count"
            combined[z_name] = (vals - mean).astype("float32")
            combined[s_name] = ((vals - mean) / (std + EPS)).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
            combined[c_name] = cnt
            new_cols.extend([z_name, s_name, c_name])
    new_cols = list(dict.fromkeys(new_cols))
    starts = np.cumsum([0] + [len(p) for p in parts])
    out = []
    for i, p in enumerate(parts):
        block = combined.iloc[starts[i]:starts[i + 1]][new_cols].reset_index(drop=True)
        base = p.reset_index(drop=True).copy()
        out.append(pd.concat([base, block], axis=1))
    return out


def fold_features(X_tr_base, X_val_base, X_test_base, X_orig_base):
    X_tr, state, combo_names = base_features(X_tr_base, fit=True)
    X_val, _, _ = base_features(X_val_base, fit=False, state=state)
    X_test, _, _ = base_features(X_test_base, fit=False, state=state)
    X_orig, _, _ = base_features(X_orig_base, fit=False, state=state)

    X_tr = add_artifact_digit_features(X_tr)
    X_val = add_artifact_digit_features(X_val)
    X_test = add_artifact_digit_features(X_test)
    X_orig = add_artifact_digit_features(X_orig)

    X_tr["SOURCE_is_original"] = 0
    X_val["SOURCE_is_original"] = 0
    X_test["SOURCE_is_original"] = 0
    X_orig["SOURCE_is_original"] = 1

    X_tr, X_val, X_test, X_orig = add_group_stat_features_parts([X_tr, X_val, X_test, X_orig])
    te_cols = [c for c in dict.fromkeys(combo_names) if c in X_tr.columns and c in X_val.columns and c in X_test.columns and c in X_orig.columns]
    return X_tr, X_val, X_test, X_orig, te_cols


def add_target_encoding(X_tr, y_tr, X_val, X_test, te_cols, seed):
    if not te_cols:
        return X_tr, X_val, X_test, []
    te = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=seed)
    te_names = [f"TE_{c}" for c in te_cols]
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_tr[te_names] = te.fit_transform(X_tr[te_cols], y_tr)
    X_val[te_names] = te.transform(X_val[te_cols])
    X_test[te_names] = te.transform(X_test[te_cols])
    return X_tr, X_val, X_test, te_names


def load_data():
    return read_train(), read_test(), read_original()


def run_fold(cfg, data, fold_ids, k):
    """Outer fold k (0-4). Returns the probabilities of the fold's rows and of the test rows and the number of features."""
    fold_no = k + 1
    train_raw, test_raw, orig_raw = data
    seed_everything(cfg.seed + fold_no)
    X_all = train_raw.drop(columns=[ID, TARGET]).reset_index(drop=True)
    y_all = train_raw[TARGET].astype(int).reset_index(drop=True)
    X_test_all = test_raw.drop(columns=[ID]).reset_index(drop=True)
    X_orig_all = orig_raw.drop(columns=[TARGET]).reset_index(drop=True)
    y_orig = orig_raw[TARGET].astype(int).reset_index(drop=True)

    tr_idx, val_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
    X_tr_synth, X_val, X_test, X_orig_tr, te_cols = fold_features(
        X_all.iloc[tr_idx].copy(), X_all.iloc[val_idx].copy(), X_test_all, X_orig_all.copy())
    y_tr_synth = y_all.iloc[tr_idx].reset_index(drop=True)
    y_val = y_all.iloc[val_idx].reset_index(drop=True)
    X_tr = pd.concat([X_tr_synth, X_orig_tr], axis=0).reset_index(drop=True)
    y_tr = pd.concat([y_tr_synth, y_orig.reset_index(drop=True)], axis=0).reset_index(drop=True)

    X_tr, X_val, X_test, _ = add_target_encoding(X_tr, y_tr, X_val, X_test, te_cols, cfg.seed + fold_no)
    print(f"fold {k}: train {X_tr.shape}, valid {X_val.shape}, test {X_test.shape}", flush=True)

    model = TabM_D_Classifier(**tabm_params(cfg, fold_no))
    model.fit(X_tr, y_tr, X_val, y_val)
    val_pred = model.predict_proba(X_val)[:, 1].astype(np.float32)
    test_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)
    n_features = int(X_tr.shape[1])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_pred, test_pred, n_features
