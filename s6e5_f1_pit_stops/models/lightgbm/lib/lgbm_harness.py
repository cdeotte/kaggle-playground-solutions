"""LightGBM on the competition rows plus all original rows (sample weight `original_weight`); one feature recipe with switches.

build_data(cfg) makes the label-free features once, on the train + original + test rows together:
  base          the 14 columns and lap_tyre = LapNumber * TyreLife
  domain        LapNumber / RaceProgress (clipped to [1, 120]), TyreLife / max(LapNumber, 1), LapNumber / (TyreLife + 1),
                Cumulative_Degradation / max(TyreLife, 1) and its absolute value, LapTime_Delta / max(TyreLife, 1) and its
                absolute value, |LapTime_Delta|, |Position_Change|, Position * RaceProgress, Stint * TyreLife
  interactions  Race x Compound, Race x Year, Race x Compound x Stint, Driver x Race, Driver x Compound, Compound x Stint
                (categories)
  counts        Year, PitStop, LapNumber, Stint, TyreLife, Position, Position_Change as categories (<col>_cat) and, for every
                category column, the number of train + original + test rows with the row's value (<col>_count)
  group_stats   LapTime (s), LapTime_Delta, Cumulative_Degradation, TyreLife minus their group mean and their z-score within
                Race, Compound, Driver and Race x Compound
Every category column is label-encoded with the sorted string values of the train + original + test rows.

run_fold(cfg, data, fold_ids, k): the training rows are the competition rows outside fold k followed by all original rows.
Optional target encodings (TE_<col>) of every category column with 2-30,000 levels in the training rows: scikit-learn
TargetEncoder (smooth te_smooth); the training rows are encoded out-of-fold with StratifiedKFold(5) (encoder fitted on the
other 4/5), the validation and test rows with the encoder fitted on all training rows. LightGBM (CPU, 32 threads) with early
stopping (300 rounds) on the validation fold's AUC. Seeds use the fold number f = k + 1: TE seed 42 + 111 f, LightGBM
random_state 2026 + f - 1."""
import os
import random
from dataclasses import dataclass, field
from typing import Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from common.data import ID, TARGET, read_original, read_test, read_train

EPS = 1e-6

BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.80,
    "bagging_freq": 5,
    "lambda_l1": 0.10,
    "lambda_l2": 0.10,
    "n_estimators": 4000,
    "device": "cpu",
    "n_jobs": 32,
    "num_threads": 32,
    "verbosity": -1,
}


@dataclass
class LgbmConfig:
    original_weight: float = 0.50
    add_domain_features: bool = False
    add_interaction_cats: bool = False
    add_count_features: bool = False
    add_group_stats: bool = False
    add_target_encoding: bool = False
    te_inner_splits: int = 5
    te_smooth: float = 20.0
    seed: int = 42
    model_seed_base: int = 2026
    lgbm_params_override: Dict = field(default_factory=dict)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def lgbm_params(cfg, fold_no):
    params = dict(BASE_PARAMS)
    params.update(cfg.lgbm_params_override)
    params["random_state"] = int(cfg.model_seed_base + fold_no - 1)
    return params


def clean_name(col):
    return str(col).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def add_features(df, cfg):
    """The feature blocks of cfg; returns the frame and the sorted category columns."""
    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        out[c] = out[c].astype(str)

    lap = pd.to_numeric(out["LapNumber"], errors="coerce").fillna(0)
    tyre = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(0)
    progress = pd.to_numeric(out["RaceProgress"], errors="coerce").fillna(0)
    deg = pd.to_numeric(out["Cumulative_Degradation"], errors="coerce").fillna(0)
    delta = pd.to_numeric(out["LapTime_Delta"], errors="coerce").fillna(0)
    pos = pd.to_numeric(out["Position"], errors="coerce").fillna(0)
    pos_chg = pd.to_numeric(out["Position_Change"], errors="coerce").fillna(0)
    stint = pd.to_numeric(out["Stint"], errors="coerce").fillna(0)

    out["lap_tyre"] = (lap * tyre).astype("float32")
    if cfg.add_domain_features:
        out["lp_rp"] = (lap / progress.clip(lower=EPS)).clip(1, 120).astype("float32")
        out["tyre_lap_ratio"] = (tyre / lap.clip(lower=1)).astype("float32")
        out["lap_per_tyre"] = (lap / (tyre + 1)).astype("float32")
        out["deg_per_tyre"] = (deg / tyre.clip(lower=1)).astype("float32")
        out["abs_deg_per_tyre"] = (deg.abs() / tyre.clip(lower=1)).astype("float32")
        out["delta_per_tyre"] = (delta / tyre.clip(lower=1)).astype("float32")
        out["abs_delta_per_tyre"] = (delta.abs() / tyre.clip(lower=1)).astype("float32")
        out["abs_delta"] = delta.abs().astype("float32")
        out["abs_pos_change"] = pos_chg.abs().astype("float32")
        out["position_progress"] = (pos * progress).astype("float32")
        out["stint_tyre"] = (stint * tyre).astype("float32")

    if cfg.add_interaction_cats:
        combos = [
            ("Race", "Compound"),
            ("Race", "Year"),
            ("Race", "Compound", "Stint"),
            ("Driver", "Race"),
            ("Driver", "Compound"),
            ("Compound", "Stint"),
        ]
        for cols in combos:
            name = "_".join(cols) + "_cat"
            s = out[cols[0]].astype(str)
            for c in cols[1:]:
                s = s + "_" + out[c].astype(str)
            out[name] = s.astype(str)
            cat_cols.append(name)

    if cfg.add_count_features:
        count_cols = list(cat_cols)
        for c in ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "Position_Change"]:
            if c in out.columns:
                name = f"{c}_cat"
                out[name] = out[c].astype(str)
                cat_cols.append(name)
                count_cols.append(name)
        for c in count_cols:
            out[f"{c}_count"] = out[c].astype(str).map(out[c].astype(str).value_counts(dropna=False)).fillna(0).astype("float32")

    if cfg.add_group_stats:
        for g in ["Race", "Compound", "Driver", "Race_Compound_cat"]:
            if g not in out.columns:
                continue
            key = out[g].astype(str)
            for v in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "TyreLife"]:
                vals = pd.to_numeric(out[v], errors="coerce").fillna(0)
                mean = vals.groupby(key).transform("mean")
                std = vals.groupby(key).transform("std").fillna(0)
                out[f"{clean_name(v)}_minus_{g}_mean"] = (vals - mean).astype("float32")
                out[f"{clean_name(v)}_z_{g}"] = ((vals - mean) / (std + EPS)).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")

    cat_cols = sorted(set([c for c in cat_cols if c in out.columns]))
    return out, cat_cols


def build_data(cfg):
    """Features of the train + original + test rows. Returns a dict: X_all (train rows, then original rows), X_test, y_all,
    weight (1 for train rows, original_weight for original rows), cat_cols, cat_maps (global label maps)."""
    seed_everything(cfg.seed)
    train_comp, test, original = read_train(), read_test(), read_original()
    features = [c for c in test.columns if c != ID]
    train_base = train_comp[features + [TARGET]].copy()
    all_train = pd.concat([train_base, original], ignore_index=True)
    weight = np.r_[np.ones(len(train_base), dtype=np.float32), np.full(len(original), cfg.original_weight, dtype=np.float32)]

    combined = pd.concat([all_train.drop(columns=[TARGET]), test[features].copy()], ignore_index=True)
    combined_fe, cat_cols = add_features(combined, cfg)
    X_all = combined_fe.iloc[:len(all_train)].reset_index(drop=True)
    X_test = combined_fe.iloc[len(all_train):].reset_index(drop=True)
    cat_maps = {}
    for c in cat_cols:
        cats = pd.Index(sorted(pd.concat([X_all[c], X_test[c]], axis=0).astype(str).unique()))
        cat_maps[c] = {v: i for i, v in enumerate(cats)}
    y_all = all_train[TARGET].astype(int).reset_index(drop=True)
    return dict(X_all=X_all, X_test=X_test, y_all=y_all, weight=weight, cat_cols=cat_cols, cat_maps=cat_maps,
                n_comp=len(train_comp))


def add_fold_target_encoding(X_tr, y_tr, X_va, X_te, cat_cols, cfg, seed):
    if not cfg.add_target_encoding or not cat_cols:
        return X_tr, X_va, X_te, []
    te_cols = []
    for c in cat_cols:
        nunique = X_tr[c].astype(str).nunique(dropna=False)
        if 2 <= nunique <= 30000:
            te_cols.append(c)
    if not te_cols:
        return X_tr, X_va, X_te, []
    global_mean = float(np.mean(y_tr))
    X_tr_te = np.zeros((len(X_tr), len(te_cols)), dtype=np.float32)
    inner = StratifiedKFold(n_splits=cfg.te_inner_splits, shuffle=True, random_state=seed)
    X_te_source = X_tr[te_cols].astype(str).reset_index(drop=True)
    y_source = pd.Series(y_tr).reset_index(drop=True)
    for inner_tr, inner_va in inner.split(X_te_source, y_source):
        enc = TargetEncoder(smooth=cfg.te_smooth, target_type="binary", cv=5, shuffle=True, random_state=seed)
        enc.fit(X_te_source.iloc[inner_tr], y_source.iloc[inner_tr])
        X_tr_te[inner_va] = np.asarray(enc.transform(X_te_source.iloc[inner_va]), dtype=np.float32)
    enc = TargetEncoder(smooth=cfg.te_smooth, target_type="binary", cv=5, shuffle=True, random_state=seed)
    enc.fit(X_tr[te_cols].astype(str), y_tr)
    va_te = np.asarray(enc.transform(X_va[te_cols].astype(str)), dtype=np.float32)
    te_te = np.asarray(enc.transform(X_te[te_cols].astype(str)), dtype=np.float32)
    new_cols = [f"TE_{c}" for c in te_cols]
    for j, c in enumerate(new_cols):
        X_tr[c] = np.nan_to_num(X_tr_te[:, j], nan=global_mean).astype("float32")
        X_va[c] = np.nan_to_num(va_te[:, j], nan=global_mean).astype("float32")
        X_te[c] = np.nan_to_num(te_te[:, j], nan=global_mean).astype("float32")
    return X_tr, X_va, X_te, new_cols


def encode_categories(frames, cat_cols, cat_maps):
    for df in frames:
        for c in cat_cols:
            df[c] = df[c].astype(str).map(cat_maps[c]).fillna(-1).astype("int32")
    return frames


def finalize(df):
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-1).astype("float32")
    return out


def run_fold(cfg, data, fold_ids, k):
    """Outer fold k (0-4). Returns the probabilities of the fold's rows and of the test rows, the best iteration and the
    number of features."""
    fold_no = k + 1
    X_all, X_test, y_all, weight = data["X_all"], data["X_test"], data["y_all"], data["weight"]
    tr_idx_comp, va_idx_comp = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
    train_idx = np.concatenate([tr_idx_comp, np.arange(data["n_comp"], len(X_all))])
    X_tr = X_all.iloc[train_idx].copy().reset_index(drop=True)
    y_tr = y_all.iloc[train_idx].astype(int).reset_index(drop=True)
    sw_tr = weight[train_idx]
    X_va = X_all.iloc[va_idx_comp].copy().reset_index(drop=True)
    y_va = y_all.iloc[va_idx_comp].astype(int).reset_index(drop=True)
    X_te = X_test.copy().reset_index(drop=True)

    X_tr, X_va, X_te, te_added = add_fold_target_encoding(X_tr, y_tr, X_va, X_te, data["cat_cols"], cfg, cfg.seed + 111 * fold_no)
    X_tr, X_va, X_te = encode_categories([X_tr, X_va, X_te], data["cat_cols"], data["cat_maps"])
    X_tr, X_va, X_te = finalize(X_tr), finalize(X_va), finalize(X_te)
    print(f"fold {k}: train {X_tr.shape}, valid {X_va.shape}, TE columns {len(te_added)}", flush=True)

    model = lgb.LGBMClassifier(**lgbm_params(cfg, fold_no))
    model.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
    va_pred = model.predict_proba(X_va)[:, 1].astype(np.float32)
    te_pred = model.predict_proba(X_te)[:, 1].astype(np.float32)
    return va_pred, te_pred, model.best_iteration_, X_tr.shape[1]
