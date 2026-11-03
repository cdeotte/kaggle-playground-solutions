"""Shared fold harness: feature blocks, column typing, embedding networks (MLP, ResNet, FM, FFM, TabTransformer,
attention graph MLP) and wrappers for TabM (pytabkit), random forest (cuML) and TabNet (pytorch-tabnet).

Used by `train.py` in this folder and by `../tabm` and `../classic`. One `ExpConfig` = one OOF / test pair:

    from models.neural_nets.lib.shared_harness import ExpConfig, run_experiment
    cfg = ExpConfig(pair='<pair>', folds_to_run=[1, 2, 3, 4, 5], ...)
    run_experiment(cfg, model_dir('<folder>'))

Folds are the repo folds (`common.data.folds`, StratifiedKFold(5, shuffle=True, random_state=42)); `folds_to_run` numbers
them 1-5 (fold k = repo fold id k - 1) and the fold number enters every seed (`seed + fold + model_seed_offset`).

Features (ExpConfig flags). The base block is always built: the 14 raw columns, LP_RP = LapNumber / (RaceProgress + 1e-6),
TL_LP = TyreLife / max(LapNumber, 1), a categorical code of floor(value) of every numeric column and of both ratios
(`<col>_cat_`, unseen -> -1), counts of every string column (Driver, Compound, Race and the domain bins) and of Year and
PitStop, a 200-bin quantile bin of RaceProgress and the combo codes Race_Compound_ and Race_Year_. Codes, counts and bins
are fitted on all train rows (`use_global_fe=True`) or on the fold's synthetic training rows (`use_global_fe=False`).
- add_domain_features (added before the base block, so its numerics also get floor codes): 16 race-state ratios /
  products (estimated total laps, laps remaining, tyre age ratio, degradation and lap-time delta per tyre lap, pit-window
  and position pressure, ...) and 3 binned categoricals (race phase, tyre-life bin, position bin).
- add_group_stats: per group (Race, Compound, Driver, Race_Compound_) the group size and, for LapTime, LapTime_Delta,
  Cumulative_Degradation and TyreLife, the difference to the group mean and the z-score, computed over the train (or the
  fold's training + held-out rows), test and original rows together.
- add_extended_group_stats: the same for Driver x Race, Driver x Compound, Race x Year, Race x Compound x Stint and
  Compound x Stint, with RaceProgress added to the values.
- add_source_indicator: SOURCE_is_original (0 for competition rows, 1 for original rows).
- add_extra_combos, add_artifact_digits, add_round_cats, add_original_priors: string combos, decimal digits, rounded
  values and original-data target priors / counts.
- Target encoding: sklearn TargetEncoder(cv=n_splits, smooth='auto', shuffle=True, random_state=seed + fold) fitted on the
  fold's training rows (synthetic + original) for the combo columns (`target_encode_all_combos`), plus the extra columns
  (`target_encode_extra_combos`) or every categorical / low-cardinality column (`target_encode_all_cats`).
Original rows: `original_mode` 'fold' appends the fold-matched original rows (original fold id != k,
`common.data.orig_folds`) to the fold's training rows, 'all' all original rows, 'none' none; per-row loss weights
`synth_weight` / `original_weight`.
Column typing: object columns and numeric columns with < 100 distinct values (train + valid + test) are categorical
(vocabulary of the training rows, unseen -> 0); the others are numeric, scaled by a QuantileTransformer (normal output),
StandardScaler or RobustScaler fitted on the training rows. `duplicate_low_card_num_as_num` also keeps the
low-cardinality numerics as numeric inputs.
Training (PyTorch models): AdamW, OneCycle (pct_start 0.12, div_factor 12) or cosine schedule, label-smoothed weighted
BCE, mixed precision, gradient clipping; validation AUC on the held-out fold after every epoch, early stopping with
`patience`, the best epoch's weights predict. Each fold's held-out and test probabilities (clipped to [1e-7, 1 - 1e-7])
are saved as parts; the pair (test = mean of the 5 fold models) is written once all 5 folds are done."""
import gc
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, QuantileTransformer, RobustScaler, StandardScaler, TargetEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.data import ID, N_FOLDS, TARGET, folds, orig_folds, read_original, read_test, read_train
from common.pairs import assemble, save_part

try:
    from pytabkit import TabM_D_Classifier
except Exception:
    TabM_D_Classifier = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:
    TabNetClassifier = None

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

ID_COL = ID
EPS = 1e-6


@dataclass
class ExpConfig:
    pair: str
    folds_to_run: List[int] = field(default_factory=lambda: [1])
    seed: int = 42
    n_splits: int = 5
    use_global_fe: bool = False
    original_mode: str = "fold"  # fold, all, none
    add_source_indicator: bool = False
    add_extra_combos: bool = False
    add_domain_features: bool = False
    add_group_stats: bool = False
    add_extended_group_stats: bool = False
    add_original_priors: bool = False
    add_artifact_digits: bool = False
    add_round_cats: bool = False
    target_encode_extra_combos: bool = False
    target_encode_all_combos: bool = True
    target_encode_all_cats: bool = False
    model_seed_offset: int = 0
    params_override: Dict = field(default_factory=dict)


BASE_PARAMS = {
    "model_type": "emb_mlp",  # emb_mlp, emb_resnet, fm, ffm, tabtransformer, graph_mlp, tabm, rf, tabnet
    "epochs": 24,
    "batch_size": 4096,
    "eval_batch_size": 16384,
    "lr": 1.0e-3,
    "min_lr": 2.0e-5,
    "weight_decay": 1.0e-4,
    "patience": 7,
    "hidden": [768, 384, 192],
    "d_model": 512,
    "n_blocks": 4,
    "block_hidden_mult": 2.0,
    "dropout": 0.08,
    "emb_dropout": 0.03,
    "num_dropout": 0.00,
    "num_emb_dim": 0,
    "include_raw_num": True,
    "activation": "silu",
    "label_smoothing": 0.01,
    "scheduler": "onecycle",  # onecycle, cosine
    "grad_clip": 1.0,
    "use_amp": True,
    "num_workers": 0,
    "scaler": "quantile",  # quantile, robust, standard
    "duplicate_low_card_num_as_num": False,
    "emb_dim_scale": 2.0,
    "emb_dim_max": 48,
    "original_weight": 1.0,
    "synth_weight": 1.0,
    "tabm_params": {},
    "rf_params": {},
    "tabnet_params": {},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def clean_name(col: str) -> str:
    return str(col).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def params_for(cfg: ExpConfig, fold: int) -> Dict:
    params = dict(BASE_PARAMS)
    params.update(cfg.params_override)
    params["random_state"] = int(cfg.seed + fold + cfg.model_seed_offset)
    return params


def load_raw_data():
    """train.csv, test.csv and the original dataset (Normalized_TyreLife dropped)."""
    return read_train(), read_test(), read_original()


# ----------------------------------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------------------------------
def pipeline_a_fe(df: pd.DataFrame, fit: bool = False, state: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if state is None:
        state = {}

    cat_cols_base = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols_base = df.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()
    num_cols_base = [c for c in num_cols_base if c not in [ID_COL, TARGET]]

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
        if col not in df.columns:
            continue
        cnt_name = f"_{col}_count" if col in cat_cols_base else f"_{col[:-1]}_count"
        if fit:
            state[cnt_name] = df[col].astype(str).value_counts(dropna=False)
        cm = state[cnt_name]
        df[cnt_name] = df[col].astype(str).map(cm).fillna(0).astype("int32")

    for col, nb in [("RaceProgress", 200)]:
        if col not in df.columns:
            continue
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
        if not all(c in df.columns for c in cols):
            continue
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


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lap = pd.to_numeric(out["LapNumber"], errors="coerce").fillna(0)
    progress = pd.to_numeric(out["RaceProgress"], errors="coerce").fillna(0)
    tyre = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(0)
    deg = pd.to_numeric(out["Cumulative_Degradation"], errors="coerce").fillna(0)
    delta = pd.to_numeric(out["LapTime_Delta"], errors="coerce").fillna(0)
    pos = pd.to_numeric(out["Position"], errors="coerce").fillna(0)
    pos_chg = pd.to_numeric(out["Position_Change"], errors="coerce").fillna(0)
    stint = pd.to_numeric(out["Stint"], errors="coerce").fillna(0)

    out["DOM_EstTotalLaps"] = (lap / progress.clip(lower=EPS)).clip(1, 120).astype("float32")
    out["DOM_LapsRemaining"] = (out["DOM_EstTotalLaps"] - lap).clip(lower=0).astype("float32")
    out["DOM_TyreAgeRatio"] = (tyre / lap.clip(lower=1)).astype("float32")
    out["DOM_LapPerTyreLife"] = (lap / (tyre + 1)).astype("float32")
    out["DOM_LapMinusTyre"] = (lap - tyre).astype("float32")
    out["DOM_PitWindowPressure"] = (tyre * progress).astype("float32")
    out["DOM_TyreToLapsRem"] = (tyre / (out["DOM_LapsRemaining"] + 1)).astype("float32")
    out["DOM_DegPerTyreLap"] = (deg / tyre.clip(lower=1)).astype("float32")
    out["DOM_AbsDegPerTyreLap"] = (deg.abs() / tyre.clip(lower=1)).astype("float32")
    out["DOM_DeltaPerTyreLap"] = (delta / tyre.clip(lower=1)).astype("float32")
    out["DOM_AbsDeltaPerTyreLap"] = (delta.abs() / tyre.clip(lower=1)).astype("float32")
    out["DOM_DeltaAbs"] = delta.abs().astype("float32")
    out["DOM_AbsCumDeg"] = deg.abs().astype("float32")
    out["DOM_AbsPosChange"] = pos_chg.abs().astype("float32")
    out["DOM_PosPressure"] = (pos * progress).astype("float32")
    out["DOM_StintPressure"] = (stint * tyre).astype("float32")
    out["DOM_RacePhase_"] = pd.cut(progress, bins=[-np.inf, .2, .4, .6, .8, np.inf], labels=["P1", "P2", "P3", "P4", "P5"]).astype(str)
    out["DOM_TyreLifeBin_"] = pd.cut(tyre, bins=[-np.inf, 3, 7, 12, 20, 30, np.inf], labels=["T0", "T1", "T2", "T3", "T4", "T5"]).astype(str)
    out["DOM_PosBin_"] = pd.cut(pos, bins=[-np.inf, 3, 8, 14, np.inf], labels=["front", "upper", "lower", "back"]).astype(str)
    return out


def add_extra_combo_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    combos = [
        ("Race", "Compound", "Stint"),
        ("Driver", "Race"),
        ("Driver", "Compound"),
        ("Compound", "Stint"),
        ("Race", "Stint"),
        ("Race", "Driver", "Compound"),
        ("Compound", "DOM_RacePhase_"),
        ("Compound", "DOM_TyreLifeBin_"),
        ("DOM_RacePhase_", "DOM_TyreLifeBin_"),
    ]
    names = []
    for cols in combos:
        if not all(c in out.columns for c in cols):
            continue
        name = "_".join(cols) + "_"
        s = out[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "_" + out[c].astype(str)
        out[name] = s.astype(str)
        names.append(name)
    return out, names


def add_artifact_digit_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    cats = []
    cols = [
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
        "DOM_TyreAgeRatio", "DOM_DegPerTyreLap", "DOM_DeltaAbs", "DOM_PitWindowPressure",
        "DOM_EstTotalLaps", "DOM_LapsRemaining",
    ]
    for col in [c for c in cols if c in out.columns]:
        x = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(float).abs()
        safe = clean_name(col)
        for i in range(3):
            name = f"DIG_{safe}_id{i+1}_"
            out[name] = ((x // (10 ** i)) % 10).astype("int8").astype(str)
            cats.append(name)
        if pd.api.types.is_float_dtype(out[col]):
            for i in range(1, 3):
                name = f"DIG_{safe}_dd{i}_"
                out[name] = ((x * (10 ** i)).round().astype(int) % 10).astype("int8").astype(str)
                cats.append(name)
        sc = (x * 100).round().astype(int).abs()
        for i in range(4):
            name = f"SIG_{safe}_s{i+1}_"
            out[name] = ((sc // (10 ** i)) % 10).astype("int8").astype(str)
            cats.append(name)
    return out, cats


def add_round_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    cats = []
    cols = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]
    for col in [c for c in cols if c in out.columns]:
        x = pd.to_numeric(out[col], errors="coerce").fillna(0)
        safe = clean_name(col)
        for nd in [0, 1, 2]:
            name = f"ROUND_{safe}_{nd}_"
            out[name] = x.round(nd).astype(str)
            cats.append(name)
    return out, cats


def add_group_stat_features_parts(parts: List[pd.DataFrame]) -> List[pd.DataFrame]:
    combined = pd.concat(parts, axis=0, ignore_index=True)
    group_cols = [c for c in ["Race", "Compound", "Driver", "Race_Compound_"] if c in combined.columns]
    value_cols = [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "TyreLife"] if c in combined.columns]
    new_cols = []
    for g in group_cols:
        key = combined[g].astype(str)
        cnt = key.map(key.value_counts(dropna=False)).astype("float32")
        c_name = f"GST_{clean_name(g)}_count"
        combined[c_name] = cnt
        new_cols.append(c_name)
        for v in value_cols:
            vals = pd.to_numeric(combined[v], errors="coerce").fillna(0).astype("float32")
            safe_v = clean_name(v)
            safe_g = clean_name(g)
            mean = vals.groupby(key).transform("mean")
            std = vals.groupby(key).transform("std").fillna(0)
            z_name = f"GST_{safe_v}_minus_{safe_g}_mean"
            s_name = f"GST_{safe_v}_z_{safe_g}"
            combined[z_name] = (vals - mean).astype("float32")
            combined[s_name] = ((vals - mean) / (std + EPS)).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
            new_cols.extend([z_name, s_name])
    new_cols = list(dict.fromkeys(new_cols))
    starts = np.cumsum([0] + [len(p) for p in parts])
    out = []
    for i, p in enumerate(parts):
        block = combined.iloc[starts[i]:starts[i + 1]][new_cols].reset_index(drop=True)
        out.append(pd.concat([p.reset_index(drop=True).copy(), block], axis=1))
    return out


def add_extended_group_stat_features_parts(parts: List[pd.DataFrame]) -> List[pd.DataFrame]:
    combined = pd.concat(parts, axis=0, ignore_index=True)
    group_specs = [
        ("Driver_Race", ["Driver", "Race"]),
        ("Driver_Compound", ["Driver", "Compound"]),
        ("Race_Year", ["Race", "Year"]),
        ("Race_Compound_Stint", ["Race", "Compound", "Stint"]),
        ("Compound_Stint", ["Compound", "Stint"]),
    ]
    value_cols = [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "TyreLife", "RaceProgress"] if c in combined.columns]
    new_cols = []
    for name, cols in group_specs:
        if not all(c in combined.columns for c in cols):
            continue
        key = combined[cols[0]].astype(str)
        for c in cols[1:]:
            key = key + "_" + combined[c].astype(str)
        cnt = key.map(key.value_counts(dropna=False)).astype("float32")
        c_name = f"GST_EXT_{name}_count"
        combined[c_name] = cnt
        new_cols.append(c_name)
        for v in value_cols:
            vals = pd.to_numeric(combined[v], errors="coerce").fillna(0).astype("float32")
            safe_v = clean_name(v)
            mean = vals.groupby(key).transform("mean")
            std = vals.groupby(key).transform("std").fillna(0)
            z_name = f"GST_EXT_{safe_v}_minus_{name}_mean"
            s_name = f"GST_EXT_{safe_v}_z_{name}"
            combined[z_name] = (vals - mean).astype("float32")
            combined[s_name] = ((vals - mean) / (std + EPS)).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
            new_cols.extend([z_name, s_name])
    new_cols = list(dict.fromkeys(new_cols))
    starts = np.cumsum([0] + [len(p) for p in parts])
    out = []
    for i, p in enumerate(parts):
        block = combined.iloc[starts[i]:starts[i + 1]][new_cols].reset_index(drop=True)
        out.append(pd.concat([p.reset_index(drop=True).copy(), block], axis=1))
    return out


def add_original_prior_features_parts(parts: List[pd.DataFrame], orig_ref: pd.DataFrame, y_orig_ref: pd.Series, train_ref: pd.DataFrame) -> List[pd.DataFrame]:
    keys = [
        "Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "Position_Change",
        "Race_Compound_", "Race_Year_",
        "LapTime (s)_cat_", "LapTime_Delta_cat_", "Cumulative_Degradation_cat_", "RaceProgress_cat_",
        "LP_RP_cat_", "TL_LP_cat_",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    ]
    keys = [c for c in keys if c in orig_ref.columns and c in train_ref.columns]
    orig_tmp = orig_ref.reset_index(drop=True).copy()
    train_tmp = train_ref.reset_index(drop=True).copy()
    orig_tmp["_y_orig_"] = pd.Series(y_orig_ref).reset_index(drop=True).astype("float32")
    new_blocks = [pd.DataFrame(index=np.arange(len(p))) for p in parts]
    global_mean = float(orig_tmp["_y_orig_"].mean())
    for key in keys:
        safe = clean_name(key)
        o_key = orig_tmp[key].astype(str)
        t_key = train_tmp[key].astype(str)
        prior = orig_tmp.groupby(o_key, observed=False)["_y_orig_"].mean()
        orig_count = o_key.value_counts(dropna=False)
        train_count = t_key.value_counts(dropna=False)
        for block, df in zip(new_blocks, parts):
            k = df[key].astype(str)
            oc = k.map(orig_count).fillna(0).astype("float32")
            tc = k.map(train_count).fillna(0).astype("float32")
            block[f"OPRIOR_{safe}"] = k.map(prior).fillna(global_mean).astype("float32")
            block[f"OCOUNT_{safe}"] = np.log1p(oc).astype("float32")
            block[f"TCOUNT_{safe}"] = np.log1p(tc).astype("float32")
            block[f"DRIFT_TO_ORIG_{safe}"] = np.log1p((tc + 1.0) / (oc + 1.0)).astype("float32")
    return [pd.concat([p.reset_index(drop=True).copy(), b.reset_index(drop=True)], axis=1) for p, b in zip(parts, new_blocks)]


def build_features_global(cfg: ExpConfig, train_raw: pd.DataFrame, test_raw: pd.DataFrame, orig_raw: pd.DataFrame):
    X = train_raw.drop(columns=[ID_COL, TARGET]).copy()
    y = train_raw[TARGET].astype(int).reset_index(drop=True)
    Xt = test_raw.drop(columns=[ID_COL]).copy()
    orig = orig_raw.drop(columns=[TARGET]).copy()
    y_orig = orig_raw[TARGET].astype(int).reset_index(drop=True)

    if cfg.add_domain_features:
        X = add_domain_features(X)
        Xt = add_domain_features(Xt)
        orig = add_domain_features(orig)
    X, state, combo_names = pipeline_a_fe(X, fit=True)
    Xt, _, _ = pipeline_a_fe(Xt, fit=False, state=state)
    orig, _, _ = pipeline_a_fe(orig, fit=False, state=state)
    extra_te_cols = []
    if cfg.add_extra_combos:
        X, names = add_extra_combo_features(X)
        Xt, _ = add_extra_combo_features(Xt)
        orig, _ = add_extra_combo_features(orig)
        extra_te_cols.extend(names)
    if cfg.add_artifact_digits:
        X, names = add_artifact_digit_features(X)
        Xt, _ = add_artifact_digit_features(Xt)
        orig, _ = add_artifact_digit_features(orig)
        if cfg.target_encode_extra_combos:
            extra_te_cols.extend(names)
    if cfg.add_round_cats:
        X, names = add_round_categorical_features(X)
        Xt, _ = add_round_categorical_features(Xt)
        orig, _ = add_round_categorical_features(orig)
        if cfg.target_encode_extra_combos:
            extra_te_cols.extend(names)
    if cfg.add_source_indicator:
        X["SOURCE_is_original"] = 0
        Xt["SOURCE_is_original"] = 0
        orig["SOURCE_is_original"] = 1
    if cfg.add_group_stats:
        X, Xt, orig = add_group_stat_features_parts([X, Xt, orig])
    if cfg.add_extended_group_stats:
        X, Xt, orig = add_extended_group_stat_features_parts([X, Xt, orig])
    if cfg.add_original_priors:
        X, Xt, orig = add_original_prior_features_parts([X, Xt, orig], orig, y_orig, X)

    te_cols = list(combo_names)
    if cfg.target_encode_extra_combos:
        te_cols += extra_te_cols
    te_cols = [c for c in dict.fromkeys(te_cols) if c in X.columns and c in Xt.columns and c in orig.columns]
    return X.reset_index(drop=True), y, Xt.reset_index(drop=True), orig.reset_index(drop=True), y_orig, te_cols


def build_features_fold_local(cfg: ExpConfig, X_tr_base, X_val_base, X_test_base, X_orig_base):
    if cfg.add_domain_features:
        X_tr_base = add_domain_features(X_tr_base)
        X_val_base = add_domain_features(X_val_base)
        X_test_base = add_domain_features(X_test_base)
        X_orig_base = add_domain_features(X_orig_base)
    X_tr, state, combo_names = pipeline_a_fe(X_tr_base, fit=True)
    X_val, _, _ = pipeline_a_fe(X_val_base, fit=False, state=state)
    X_test, _, _ = pipeline_a_fe(X_test_base, fit=False, state=state)
    X_orig, _, _ = pipeline_a_fe(X_orig_base, fit=False, state=state)
    extra_te_cols = []
    if cfg.add_extra_combos:
        X_tr, names = add_extra_combo_features(X_tr)
        X_val, _ = add_extra_combo_features(X_val)
        X_test, _ = add_extra_combo_features(X_test)
        X_orig, _ = add_extra_combo_features(X_orig)
        extra_te_cols.extend(names)
    if cfg.add_artifact_digits:
        X_tr, names = add_artifact_digit_features(X_tr)
        X_val, _ = add_artifact_digit_features(X_val)
        X_test, _ = add_artifact_digit_features(X_test)
        X_orig, _ = add_artifact_digit_features(X_orig)
        if cfg.target_encode_extra_combos:
            extra_te_cols.extend(names)
    if cfg.add_round_cats:
        X_tr, names = add_round_categorical_features(X_tr)
        X_val, _ = add_round_categorical_features(X_val)
        X_test, _ = add_round_categorical_features(X_test)
        X_orig, _ = add_round_categorical_features(X_orig)
        if cfg.target_encode_extra_combos:
            extra_te_cols.extend(names)
    if cfg.add_source_indicator:
        X_tr["SOURCE_is_original"] = 0
        X_val["SOURCE_is_original"] = 0
        X_test["SOURCE_is_original"] = 0
        X_orig["SOURCE_is_original"] = 1
    if cfg.add_group_stats:
        X_tr, X_val, X_test, X_orig = add_group_stat_features_parts([X_tr, X_val, X_test, X_orig])
    if cfg.add_extended_group_stats:
        X_tr, X_val, X_test, X_orig = add_extended_group_stat_features_parts([X_tr, X_val, X_test, X_orig])

    te_cols = list(combo_names)
    if cfg.target_encode_extra_combos:
        te_cols += extra_te_cols
    te_cols = [c for c in dict.fromkeys(te_cols) if c in X_tr.columns and c in X_val.columns and c in X_test.columns and c in X_orig.columns]
    return X_tr, X_val, X_test, X_orig, te_cols


def add_target_encoding(X_tr, y_tr, X_val, X_test, te_cols, cfg: ExpConfig, fold: int):
    if not te_cols:
        return X_tr, X_val, X_test, []
    te = TargetEncoder(cv=cfg.n_splits, smooth="auto", shuffle=True, random_state=cfg.seed + fold)
    te_names = [f"TE_{c}" for c in te_cols]
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_tr[te_names] = te.fit_transform(X_tr[te_cols], y_tr)
    X_val[te_names] = te.transform(X_val[te_cols])
    X_test[te_names] = te.transform(X_test[te_cols])
    return X_tr, X_val, X_test, te_names


def infer_target_encoder_cols(X_tr: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, max_card: int = 50000):
    obj_cols = X_tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X_tr.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()
    low_card_num = []
    probe = pd.concat([X_tr[num_cols], X_val[num_cols], X_test[num_cols]], axis=0, ignore_index=True)
    for c in num_cols:
        if c.startswith("TE_"):
            continue
        nunique = probe[c].nunique(dropna=False)
        if 2 <= nunique <= max_card:
            low_card_num.append(c)
    return list(dict.fromkeys(obj_cols + low_card_num))


# ----------------------------------------------------------------------------------------------------------------------
# column typing and arrays
# ----------------------------------------------------------------------------------------------------------------------
def infer_cat_num_cols(X_tr: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, params: Dict):
    obj_cols = X_tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X_tr.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()
    low_card_num = []
    probe = pd.concat([X_tr[num_cols], X_val[num_cols], X_test[num_cols]], axis=0, ignore_index=True)
    for c in num_cols:
        if c.startswith("TE_"):
            continue
        nunique = probe[c].nunique(dropna=False)
        if nunique < 100:
            low_card_num.append(c)
    cat_cols = list(dict.fromkeys(obj_cols + low_card_num))
    if bool(params.get("duplicate_low_card_num_as_num", False)):
        num_cols = [c for c in X_tr.columns if c not in obj_cols]
    else:
        num_cols = [c for c in X_tr.columns if c not in cat_cols]
    return cat_cols, num_cols


def make_vocab_maps(train_df: pd.DataFrame, cols: List[str]):
    maps, sizes = {}, {}
    for c in cols:
        vals = train_df[c].astype("string").fillna("__nan__")
        uniq = vals.unique().tolist()
        v2i = {v: i + 1 for i, v in enumerate(uniq)}
        maps[c] = v2i
        sizes[c] = len(v2i) + 1
    return maps, sizes


def encode_cats(df: pd.DataFrame, cols: List[str], maps: Dict[str, Dict[str, int]]) -> np.ndarray:
    X = np.zeros((len(df), len(cols)), dtype=np.int64)
    for j, c in enumerate(cols):
        X[:, j] = df[c].astype("string").fillna("__nan__").map(maps[c]).fillna(0).astype(np.int64).values
    return X


def fit_num_scaler(params: Dict, X: np.ndarray):
    kind = params.get("scaler", "quantile")
    if kind == "robust":
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
    elif kind == "standard":
        scaler = StandardScaler()
    elif kind == "quantile":
        scaler = QuantileTransformer(n_quantiles=min(2048, max(10, X.shape[0] // 100)), output_distribution="normal", subsample=200000, random_state=params.get("random_state", 42))
    else:
        raise ValueError(f"unknown scaler: {kind}")
    return scaler.fit(X)


def prepare_arrays(X_tr: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, params: Dict):
    cat_cols, num_cols = infer_cat_num_cols(X_tr, X_val, X_test, params)
    for df in (X_tr, X_val, X_test):
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    maps, sizes = make_vocab_maps(X_tr, cat_cols)
    Xc_tr = encode_cats(X_tr, cat_cols, maps)
    Xc_val = encode_cats(X_val, cat_cols, maps)
    Xc_test = encode_cats(X_test, cat_cols, maps)
    Xn_tr_raw = X_tr[num_cols].astype("float32").values if num_cols else np.zeros((len(X_tr), 0), dtype=np.float32)
    Xn_val_raw = X_val[num_cols].astype("float32").values if num_cols else np.zeros((len(X_val), 0), dtype=np.float32)
    Xn_test_raw = X_test[num_cols].astype("float32").values if num_cols else np.zeros((len(X_test), 0), dtype=np.float32)
    if num_cols:
        scaler = fit_num_scaler(params, Xn_tr_raw)
        Xn_tr = scaler.transform(Xn_tr_raw).astype("float32")
        Xn_val = scaler.transform(Xn_val_raw).astype("float32")
        Xn_test = scaler.transform(Xn_test_raw).astype("float32")
        Xn_tr = np.nan_to_num(Xn_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xn_val = np.nan_to_num(Xn_val, nan=0.0, posinf=0.0, neginf=0.0)
        Xn_test = np.nan_to_num(Xn_test, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        Xn_tr, Xn_val, Xn_test = Xn_tr_raw, Xn_val_raw, Xn_test_raw
    cat_cardinals = [sizes[c] for c in cat_cols]
    return Xc_tr, Xn_tr, Xc_val, Xn_val, Xc_test, Xn_test, cat_cardinals, cat_cols, num_cols


class TabDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None, weight=None):
        self.Xc = torch.as_tensor(X_cat, dtype=torch.long)
        self.Xn = torch.as_tensor(X_num, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)
        self.w = None if weight is None else torch.as_tensor(weight, dtype=torch.float32)

    def __len__(self):
        return self.Xc.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.Xc[idx], self.Xn[idx]
        if self.w is None:
            return self.Xc[idx], self.Xn[idx], self.y[idx]
        return self.Xc[idx], self.Xn[idx], self.y[idx], self.w[idx]


# ----------------------------------------------------------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------------------------------------------------------
def activation_layer(name: str):
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "mish":
        return nn.Mish()
    return nn.SiLU(inplace=True)


def emb_dim(card: int, params: Dict) -> int:
    d = int(round(float(params.get("emb_dim_scale", 2.0)) * (card ** 0.25)))
    return int(np.clip(d, 4, int(params.get("emb_dim_max", 48))))


class EmbMLP(nn.Module):
    def __init__(self, cat_cardinals, n_num, params):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_out = 0
        for card in cat_cardinals:
            d = emb_dim(card, params)
            emb = nn.Embedding(card, d)
            nn.init.normal_(emb.weight, 0.0, 0.02)
            self.embs.append(emb)
            emb_out += d
        self.num_emb = NumericEmbedding(n_num, int(params.get("num_emb_dim", 0))) if n_num and int(params.get("num_emb_dim", 0)) > 0 else None
        self.include_raw_num = bool(params.get("include_raw_num", True))
        self.emb_drop = nn.Dropout(float(params.get("emb_dropout", 0.03)))
        self.num_drop = nn.Dropout(float(params.get("num_dropout", 0.0)))
        num_out = (n_num if self.include_raw_num else 0) + (self.num_emb.out_dim if self.num_emb is not None else 0)
        in_dim = emb_out + num_out
        layers = []
        for h in params.get("hidden", [768, 384, 192]):
            layers.extend([nn.Linear(in_dim, int(h)), nn.LayerNorm(int(h)), activation_layer(params.get("activation", "silu")), nn.Dropout(float(params.get("dropout", 0.08)))])
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        parts = []
        if self.embs:
            parts.append(self.emb_drop(torch.cat([emb(x_cat[:, j]) for j, emb in enumerate(self.embs)], dim=1)))
        if x_num.shape[1] and self.num_emb is not None:
            parts.append(self.num_emb(x_num))
        if x_num.shape[1] and self.include_raw_num:
            parts.append(self.num_drop(x_num))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.net(x).squeeze(1)


class ResBlock(nn.Module):
    def __init__(self, d_model, hidden, dropout, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            activation_layer(act),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class NumericEmbedding(nn.Module):
    """Periodic embedding of every numeric input: sin / cos of (x * w + b) with d_embedding frequencies each."""

    def __init__(self, n_num: int, d_embedding: int):
        super().__init__()
        self.n_num = int(n_num)
        self.d_embedding = int(d_embedding)
        self.out_dim = self.n_num * self.d_embedding * 2
        self.weight = nn.Parameter(torch.empty(self.n_num, self.d_embedding))
        self.bias = nn.Parameter(torch.empty(self.n_num, self.d_embedding))
        nn.init.normal_(self.weight, mean=0.0, std=0.08)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x_num):
        z = x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1).flatten(1)
        return self.norm(z)


class EmbResNet(nn.Module):
    def __init__(self, cat_cardinals, n_num, params):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_out = 0
        for card in cat_cardinals:
            d = emb_dim(card, params)
            emb = nn.Embedding(card, d)
            nn.init.normal_(emb.weight, 0.0, 0.02)
            self.embs.append(emb)
            emb_out += d
        self.num_emb = NumericEmbedding(n_num, int(params.get("num_emb_dim", 0))) if n_num and int(params.get("num_emb_dim", 0)) > 0 else None
        self.include_raw_num = bool(params.get("include_raw_num", True))
        self.emb_drop = nn.Dropout(float(params.get("emb_dropout", 0.03)))
        self.num_drop = nn.Dropout(float(params.get("num_dropout", 0.0)))
        d_model = int(params.get("d_model", 512))
        num_out = (n_num if self.include_raw_num else 0) + (self.num_emb.out_dim if self.num_emb is not None else 0)
        in_dim = emb_out + num_out
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), activation_layer(params.get("activation", "silu")))
        hidden = int(d_model * float(params.get("block_hidden_mult", 2.0)))
        self.blocks = nn.Sequential(*[ResBlock(d_model, hidden, float(params.get("dropout", 0.08)), params.get("activation", "silu")) for _ in range(int(params.get("n_blocks", 4)))])
        self.head = nn.Sequential(nn.LayerNorm(d_model), activation_layer(params.get("activation", "silu")), nn.Linear(d_model, 1))

    def forward(self, x_cat, x_num):
        parts = []
        if self.embs:
            parts.append(self.emb_drop(torch.cat([emb(x_cat[:, j]) for j, emb in enumerate(self.embs)], dim=1)))
        if x_num.shape[1] and self.num_emb is not None:
            parts.append(self.num_emb(x_num))
        if x_num.shape[1] and self.include_raw_num:
            parts.append(self.num_drop(x_num))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        x = self.proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)


class FMModel(nn.Module):
    """Factorization machine: linear terms + pairwise interactions of the categorical embeddings (and one projected
    numeric vector), the interaction vector and the raw numerics feed a small MLP head."""

    def __init__(self, cat_cardinals, n_num, params):
        super().__init__()
        k = int(params.get("fm_dim", 24))
        self.embs = nn.ModuleList([nn.Embedding(card, k) for card in cat_cardinals])
        self.lin_embs = nn.ModuleList([nn.Embedding(card, 1) for card in cat_cardinals])
        for emb in list(self.embs) + list(self.lin_embs):
            nn.init.normal_(emb.weight, 0.0, 0.02)
        self.num_proj = nn.Linear(n_num, k) if n_num else None
        self.num_linear = nn.Linear(n_num, 1) if n_num else None
        self.drop = nn.Dropout(float(params.get("dropout", 0.08)))
        self.include_raw_num = bool(params.get("include_raw_num", True))
        hidden = int(params.get("fm_hidden", 256))
        in_dim = k + (n_num if self.include_raw_num else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            activation_layer(params.get("activation", "silu")),
            nn.Dropout(float(params.get("dropout", 0.08))),
            nn.Linear(hidden, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_cat, x_num):
        cat_vecs = [emb(x_cat[:, j]) for j, emb in enumerate(self.embs)]
        if cat_vecs:
            V = torch.stack(cat_vecs, dim=1)
            lin = torch.cat([emb(x_cat[:, j]) for j, emb in enumerate(self.lin_embs)], dim=1).sum(1, keepdim=True)
        else:
            V = x_num.new_zeros((x_num.shape[0], 0, 1))
            lin = x_num.new_zeros((x_num.shape[0], 1))
        if self.num_proj is not None:
            V = torch.cat([V, self.num_proj(x_num).unsqueeze(1)], dim=1)
            lin = lin + self.num_linear(x_num)
        summed = V.sum(dim=1)
        interactions = 0.5 * (summed.pow(2) - V.pow(2).sum(dim=1))
        parts = [self.drop(interactions)]
        if x_num.shape[1] and self.include_raw_num:
            parts.append(x_num)
        h = torch.cat(parts, dim=1)
        return (self.head(h) + lin + self.bias.view(1, 1)).squeeze(1)


class FFMModel(nn.Module):
    """Field-gated embeddings: every categorical field's embedding is scaled by a learned per-field gate; the flattened
    embeddings, the pairwise-interaction vector of the first `ffm_max_pair_fields` fields (plus a numeric projection)
    and the raw numerics feed an MLP."""

    def __init__(self, cat_cardinals, n_num, params):
        super().__init__()
        k = int(params.get("ffm_dim", 12))
        self.n_fields = len(cat_cardinals)
        max_pair_fields = int(params.get("ffm_max_pair_fields", 20))
        self.pair_fields = min(self.n_fields, max_pair_fields)
        self.embs = nn.ModuleList([nn.Embedding(card, k) for card in cat_cardinals])
        self.field_gate = nn.Parameter(torch.ones(max(1, self.n_fields), k))
        for emb in self.embs:
            nn.init.normal_(emb.weight, 0.0, 0.02)
        self.num_proj = nn.Linear(n_num, k) if n_num else None
        emb_out = self.n_fields * k
        pair_out = k
        self.include_raw_num = bool(params.get("include_raw_num", True))
        num_out = n_num if self.include_raw_num else 0
        in_dim = emb_out + pair_out + num_out
        layers = []
        for h in params.get("hidden", [512, 256]):
            layers.extend([nn.Linear(in_dim, int(h)), nn.LayerNorm(int(h)), activation_layer(params.get("activation", "silu")), nn.Dropout(float(params.get("dropout", 0.08)))])
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        if self.embs:
            emb_list = [emb(x_cat[:, j]) * self.field_gate[j] for j, emb in enumerate(self.embs)]
            E = torch.stack(emb_list, dim=1)
            flat = E.flatten(1)
            P = E[:, :self.pair_fields, :]
            pair = 0.5 * (P.sum(1).pow(2) - P.pow(2).sum(1))
        else:
            flat = x_num.new_zeros((x_num.shape[0], 0))
            pair = x_num.new_zeros((x_num.shape[0], int(self.field_gate.shape[1])))
        if self.num_proj is not None:
            pair = pair + self.num_proj(x_num)
        parts = [flat, pair]
        if x_num.shape[1] and self.include_raw_num:
            parts.append(x_num)
        return self.net(torch.cat(parts, dim=1)).squeeze(1)


class TabTransformerModel(nn.Module):
    """Categorical tokens (embedding + field embedding) through a pre-norm Transformer encoder, mean-pooled; concatenated
    with a projection of the numerics and the raw numerics into an MLP head."""

    def __init__(self, cat_cardinals, n_num, params):
        super().__init__()
        d_model = int(params.get("d_model", 96))
        self.embs = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinals])
        for emb in self.embs:
            nn.init.normal_(emb.weight, 0.0, 0.02)
        self.field_emb = nn.Parameter(torch.zeros(max(1, len(cat_cardinals)), d_model))
        nn.init.normal_(self.field_emb, 0.0, 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(params.get("n_heads", 4)),
            dim_feedforward=int(params.get("transformer_ff", 256)),
            dropout=float(params.get("dropout", 0.08)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(params.get("n_blocks", 2)))
        self.num_proj = nn.Sequential(nn.Linear(n_num, d_model), nn.LayerNorm(d_model), activation_layer(params.get("activation", "silu"))) if n_num else None
        self.include_raw_num = bool(params.get("include_raw_num", True))
        head_in = d_model * 2 + (n_num if self.include_raw_num else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, int(params.get("head_hidden", 384))),
            activation_layer(params.get("activation", "silu")),
            nn.Dropout(float(params.get("dropout", 0.08))),
            nn.Linear(int(params.get("head_hidden", 384)), 1),
        )

    def forward(self, x_cat, x_num):
        if self.embs:
            toks = torch.stack([emb(x_cat[:, j]) for j, emb in enumerate(self.embs)], dim=1)
            toks = toks + self.field_emb[: toks.shape[1]].unsqueeze(0)
            z = self.encoder(toks).mean(dim=1)
        else:
            z = x_num.new_zeros((x_num.shape[0], int(self.field_emb.shape[1])))
        if self.num_proj is not None:
            zn = self.num_proj(x_num)
        else:
            zn = x_num.new_zeros_like(z)
        parts = [z, zn]
        if x_num.shape[1] and self.include_raw_num:
            parts.append(x_num)
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


class GraphMLP(nn.Module):
    """One round of attention message passing over the fully connected graph of categorical-field embeddings; the
    mean-pooled embeddings, the mean message, a projection of the numerics and the raw numerics feed an MLP."""

    def __init__(self, cat_cardinals, n_num, params):
        super().__init__()
        d = int(params.get("d_model", 128))
        self.embs = nn.ModuleList([nn.Embedding(card, d) for card in cat_cardinals])
        for emb in self.embs:
            nn.init.normal_(emb.weight, 0.0, 0.02)
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.num_proj = nn.Sequential(nn.Linear(n_num, d), nn.LayerNorm(d), activation_layer(params.get("activation", "silu"))) if n_num else None
        self.include_raw_num = bool(params.get("include_raw_num", True))
        in_dim = d * 3 + (n_num if self.include_raw_num else 0)
        layers = []
        for h in params.get("hidden", [512, 256, 128]):
            layers.extend([nn.Linear(in_dim, int(h)), nn.LayerNorm(int(h)), activation_layer(params.get("activation", "silu")), nn.Dropout(float(params.get("dropout", 0.08)))])
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        if self.embs:
            E = torch.stack([emb(x_cat[:, j]) for j, emb in enumerate(self.embs)], dim=1)
            q, k, v = self.query(E), self.key(E), self.value(E)
            attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(q.shape[-1]), dim=-1)
            msg = torch.bmm(attn, v).mean(1)
            pooled = E.mean(1)
        else:
            pooled = x_num.new_zeros((x_num.shape[0], int(self.key.out_features)))
            msg = pooled
        if self.num_proj is not None:
            num = self.num_proj(x_num)
        else:
            num = pooled.new_zeros(pooled.shape)
        parts = [pooled, msg, num]
        if x_num.shape[1] and self.include_raw_num:
            parts.append(x_num)
        return self.net(torch.cat(parts, dim=1)).squeeze(1)


def make_model(cat_cardinals, n_num, params):
    model_type = params.get("model_type", "emb_mlp")
    if model_type == "emb_resnet":
        return EmbResNet(cat_cardinals, n_num, params)
    if model_type == "fm":
        return FMModel(cat_cardinals, n_num, params)
    if model_type == "ffm":
        return FFMModel(cat_cardinals, n_num, params)
    if model_type == "tabtransformer":
        return TabTransformerModel(cat_cardinals, n_num, params)
    if model_type == "graph_mlp":
        return GraphMLP(cat_cardinals, n_num, params)
    return EmbMLP(cat_cardinals, n_num, params)


# ----------------------------------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------------------------------
def smooth_bce_with_logits(logits, targets, weights, eps):
    if eps and eps > 0:
        targets = targets * (1.0 - eps) + 0.5 * eps
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weights is not None:
        loss = loss * weights
    return loss.mean()


@torch.no_grad()
def predict_proba(model, loader, device, use_amp=True):
    model.eval()
    preds = []
    for batch in loader:
        if len(batch) == 2:
            xc, xn = batch
        else:
            xc, xn = batch[0], batch[1]
        xc = xc.to(device, non_blocking=True)
        xn = xn.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(use_amp) and device.type == "cuda"):
            logits = model(xc, xn)
        preds.append(torch.sigmoid(logits).float().detach().cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(Xc_tr, Xn_tr, y_tr, w_tr, Xc_val, Xn_val, y_val, cat_cardinals, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cat_cardinals, Xn_tr.shape[1], params).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    train_loader = DataLoader(
        TabDataset(Xc_tr, Xn_tr, y_tr, w_tr),
        batch_size=int(params["batch_size"]),
        shuffle=True,
        num_workers=int(params.get("num_workers", 2)),
        pin_memory=True,
        persistent_workers=int(params.get("num_workers", 2)) > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        TabDataset(Xc_val, Xn_val, y_val),
        batch_size=int(params.get("eval_batch_size", 16384)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    steps_per_epoch = max(1, len(train_loader))
    if params.get("scheduler") == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(params["lr"]), epochs=int(params["epochs"]), steps_per_epoch=steps_per_epoch, pct_start=0.12, div_factor=12.0, final_div_factor=max(1.0, float(params["lr"]) / float(params.get("min_lr", 2e-5))))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(params["epochs"]), eta_min=float(params.get("min_lr", 2e-5)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(params.get("use_amp", True)) and device.type == "cuda")
    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for xc, xn, yb, wb in train_loader:
            xc = xc.to(device, non_blocking=True)
            xn = xn.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(params.get("use_amp", True)) and device.type == "cuda"):
                logits = model(xc, xn)
                loss = smooth_bce_with_logits(logits, yb, wb, float(params.get("label_smoothing", 0.01)))
            scaler.scale(loss).backward()
            if float(params.get("grad_clip", 0)) > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), float(params["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            if params.get("scheduler") == "onecycle":
                sched.step()
            bs = int(xc.size(0))
            loss_sum += float(loss.detach().cpu()) * bs
            n_seen += bs
        if params.get("scheduler") != "onecycle":
            sched.step()
        p_val = predict_proba(model, val_loader, device, params.get("use_amp", True))
        auc = float(roc_auc_score(y_val, p_val))
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:03d} | lr {lr_now:.3e} | loss {loss_sum / max(1, n_seen):.5f} | val_auc {auc:.6f}", flush=True)
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(params["patience"]):
                print(f"early stop: best_auc={best_auc:.6f}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def to_numpy(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    try:
        return x.to_numpy()
    except Exception:
        return np.asarray(x)


def train_predict_pytabkit_tabm(X_tr, y_tr, X_val, y_val, X_test, cat_cols, params):
    if TabM_D_Classifier is None:
        raise ImportError("pytabkit TabM_D_Classifier is not available")
    tabm_params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_state": int(params.get("random_state", 42)),
        "n_epochs": int(params.get("tabm_epochs", params.get("epochs", 8))),
        "batch_size": int(params.get("tabm_batch_size", params.get("batch_size", 512))),
        "lr": float(params.get("tabm_lr", 3e-4)),
        "d_embedding": int(params.get("tabm_d_embedding", 8)),
        "tabm_k": int(params.get("tabm_k", 16)),
        "arch_type": params.get("tabm_arch_type", "tabm-mini"),
        "dropout": float(params.get("tabm_dropout", params.get("dropout", 0.2))),
        "d_block": int(params.get("tabm_d_block", 256)),
        "n_blocks": int(params.get("tabm_n_blocks", 2)),
        "compile_model": False,
        "verbosity": 2,
    }
    tabm_params.update(params.get("tabm_params", {}) or {})
    model = TabM_D_Classifier(**tabm_params)
    model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_cols)
    val_pred = model.predict_proba(X_val)[:, 1].astype(np.float32)
    test_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)
    best_auc = float(roc_auc_score(y_val, val_pred))
    return val_pred, test_pred, best_auc


def train_predict_rf(Xd_tr, y_tr, Xd_val, y_val, Xd_test, params):
    try:
        from cuml.ensemble import RandomForestClassifier as CuRF
    except Exception as err:
        raise ImportError("cuML RandomForestClassifier is required for model_type='rf'") from err
    rf_params = {
        "n_estimators": int(params.get("rf_n_estimators", 900)),
        "max_depth": int(params.get("rf_max_depth", 16)),
        "max_features": float(params.get("rf_max_features", 0.35)),
        "min_samples_leaf": int(params.get("rf_min_samples_leaf", 10)),
        "n_bins": int(params.get("rf_n_bins", 128)),
        "n_streams": 1,
        "random_state": int(params.get("random_state", 42)),
        "verbose": 0,
    }
    rf_params.update(params.get("rf_params", {}) or {})
    model = CuRF(**rf_params)
    model.fit(Xd_tr.astype(np.float32), y_tr.astype(np.int32))
    val_pred = to_numpy(model.predict_proba(Xd_val.astype(np.float32))[:, 1]).astype(np.float32)
    test_pred = to_numpy(model.predict_proba(Xd_test.astype(np.float32))[:, 1]).astype(np.float32)
    best_auc = float(roc_auc_score(y_val, val_pred))
    return val_pred, test_pred, best_auc


def train_predict_tabnet(Xd_tr, y_tr, Xd_val, y_val, Xd_test, params):
    if TabNetClassifier is None:
        raise ImportError("pytorch_tabnet TabNetClassifier is not available")
    tabnet_params = {
        "n_d": int(params.get("tabnet_n_d", 32)),
        "n_a": int(params.get("tabnet_n_a", 32)),
        "n_steps": int(params.get("tabnet_n_steps", 4)),
        "gamma": float(params.get("tabnet_gamma", 1.4)),
        "lambda_sparse": float(params.get("tabnet_lambda_sparse", 1e-5)),
        "optimizer_fn": torch.optim.AdamW,
        "optimizer_params": {
            "lr": float(params.get("tabnet_lr", params.get("lr", 1e-3))),
            "weight_decay": float(params.get("weight_decay", 1e-4)),
        },
        "scheduler_params": {"step_size": 10, "gamma": 0.90},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "mask_type": params.get("tabnet_mask_type", "entmax"),
        "seed": int(params.get("random_state", 42)),
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
        "verbose": 1,
    }
    tabnet_params.update(params.get("tabnet_params", {}) or {})
    model = TabNetClassifier(**tabnet_params)
    model.fit(
        X_train=Xd_tr.astype(np.float32),
        y_train=y_tr.astype(np.int64),
        eval_set=[(Xd_val.astype(np.float32), y_val.astype(np.int64))],
        eval_metric=["auc"],
        max_epochs=int(params.get("tabnet_max_epochs", params.get("epochs", 35))),
        patience=int(params.get("patience", 8)),
        batch_size=int(params.get("batch_size", 8192)),
        virtual_batch_size=int(params.get("tabnet_virtual_batch_size", 1024)),
        num_workers=0,
        drop_last=False,
    )
    val_pred = model.predict_proba(Xd_val.astype(np.float32))[:, 1].astype(np.float32)
    test_pred = model.predict_proba(Xd_test.astype(np.float32))[:, 1].astype(np.float32)
    best_auc = float(roc_auc_score(y_val, val_pred))
    return val_pred, test_pred, best_auc


def fit_predict_fold(X_tr, y_tr, w_tr, X_val, y_val, X_test, params):
    """Typed arrays + the model of params['model_type']; returns held-out and test probabilities."""
    Xc_tr, Xn_tr, Xc_val, Xn_val, Xc_test, Xn_test, cat_cardinals, cat_cols, num_cols = prepare_arrays(X_tr, X_val, X_test, params)
    print("Features:", len(cat_cols) + len(num_cols), "cat:", len(cat_cols), "num:", len(num_cols), flush=True)
    print("Arrays:", Xc_tr.shape, Xn_tr.shape, Xc_val.shape, Xn_val.shape, "train_pos", float(np.mean(y_tr)), flush=True)

    model_type = params.get("model_type", "emb_mlp")
    if model_type == "tabm":
        val_pred, fold_test, _ = train_predict_pytabkit_tabm(X_tr, y_tr, X_val, y_val, X_test, cat_cols, params)
    elif model_type in ("rf", "tabnet"):
        Xd_tr = np.concatenate([Xc_tr.astype(np.float32), Xn_tr.astype(np.float32)], axis=1)
        Xd_val = np.concatenate([Xc_val.astype(np.float32), Xn_val.astype(np.float32)], axis=1)
        Xd_test = np.concatenate([Xc_test.astype(np.float32), Xn_test.astype(np.float32)], axis=1)
        if model_type == "rf":
            val_pred, fold_test, _ = train_predict_rf(Xd_tr, y_tr.to_numpy(np.float32), Xd_val, y_val.to_numpy(np.int32), Xd_test, params)
        else:
            val_pred, fold_test, _ = train_predict_tabnet(Xd_tr, y_tr.to_numpy(np.int64), Xd_val, y_val.to_numpy(np.int64), Xd_test, params)
    else:
        model, _ = train_one_fold(
            Xc_tr, Xn_tr, y_tr.to_numpy(np.float32), w_tr,
            Xc_val, Xn_val, y_val.to_numpy(np.float32), cat_cardinals, params,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_loader = DataLoader(TabDataset(Xc_val, Xn_val), batch_size=int(params.get("eval_batch_size", 16384)), shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(TabDataset(Xc_test, Xn_test), batch_size=int(params.get("eval_batch_size", 16384)), shuffle=False, num_workers=0, pin_memory=True)
        val_pred = predict_proba(model, val_loader, device, params.get("use_amp", True)).astype(np.float32)
        fold_test = predict_proba(model, test_loader, device, params.get("use_amp", True)).astype(np.float32)
    return val_pred, fold_test


# ----------------------------------------------------------------------------------------------------------------------
# folds
# ----------------------------------------------------------------------------------------------------------------------
def run_experiment(cfg: ExpConfig, folder_path) -> Optional[np.ndarray]:
    """Train the folds in cfg.folds_to_run (numbered 1-5). Each fold's held-out and test predictions go to
    <folder_path>/parts/; <folder_path>/preds/<pair>_{oof,test}.npy is written once all 5 folds are done (returns the OOF
    then, else None)."""
    assert cfg.n_splits == N_FOLDS, "the repo folds are 5 folds"
    seed_everything(cfg.seed)
    print(f"{cfg.pair}: folds {cfg.folds_to_run}", flush=True)

    train_raw, test_raw, orig_raw = load_raw_data()
    y_comp = train_raw[TARGET].astype(int).reset_index(drop=True)
    y_orig_raw = orig_raw[TARGET].astype(int).reset_index(drop=True)
    # repo folds = StratifiedKFold(5, shuffle=True, random_state=42) of the train rows and, separately, of the original rows
    fold_ids = folds(y_comp.to_numpy())
    orig_fold_ids = orig_folds(y_orig_raw.to_numpy())
    selected_folds = [f for f in cfg.folds_to_run if 1 <= f <= cfg.n_splits]

    if cfg.use_global_fe:
        X_all, y_all, X_test_all, X_orig_all, y_orig, global_te_cols = build_features_global(cfg, train_raw, test_raw, orig_raw)
    else:
        X_all = train_raw.drop(columns=[ID_COL, TARGET]).reset_index(drop=True)
        y_all = y_comp
        X_test_all = test_raw.drop(columns=[ID_COL]).reset_index(drop=True)
        X_orig_all = orig_raw.drop(columns=[TARGET]).reset_index(drop=True)
        y_orig = y_orig_raw
        global_te_cols = []

    for fold in range(1, cfg.n_splits + 1):
        if fold not in selected_folds:
            continue
        tr_idx = np.where(fold_ids != fold - 1)[0]
        val_idx = np.where(fold_ids == fold - 1)[0]
        orig_tr_idx = np.where(orig_fold_ids != fold - 1)[0]
        seed_everything(cfg.seed + fold + cfg.model_seed_offset)
        params = params_for(cfg, fold)
        print("=" * 78, flush=True)
        print(f"{cfg.pair} fold {fold}/{cfg.n_splits}", flush=True)
        print("=" * 78, flush=True)

        if cfg.use_global_fe:
            X_tr_synth = X_all.iloc[tr_idx].copy()
            X_val = X_all.iloc[val_idx].copy()
            X_test = X_test_all.copy()
            X_orig_tr = X_orig_all.copy() if cfg.original_mode == "all" else X_orig_all.iloc[orig_tr_idx].copy()
            te_cols = list(global_te_cols)
        else:
            X_tr_base = X_all.iloc[tr_idx].copy()
            X_val_base = X_all.iloc[val_idx].copy()
            X_orig_base = X_orig_all.copy() if cfg.original_mode == "all" else X_orig_all.iloc[orig_tr_idx].copy()
            X_tr_synth, X_val, X_test, X_orig_tr, te_cols = build_features_fold_local(cfg, X_tr_base, X_val_base, X_test_all, X_orig_base)

        y_tr_synth = y_all.iloc[tr_idx].reset_index(drop=True)
        y_val = y_all.iloc[val_idx].reset_index(drop=True)
        if cfg.add_original_priors and not cfg.use_global_fe:
            y_orig_prior = y_orig if cfg.original_mode == "all" else y_orig.iloc[orig_tr_idx].reset_index(drop=True)
            X_tr_synth, X_val, X_test, X_orig_tr = add_original_prior_features_parts(
                [X_tr_synth, X_val, X_test, X_orig_tr], X_orig_tr, y_orig_prior, X_tr_synth
            )
        if cfg.original_mode == "none":
            X_tr = X_tr_synth.reset_index(drop=True)
            y_tr = y_tr_synth
            w_tr = np.full(len(y_tr), float(params.get("synth_weight", 1.0)), dtype=np.float32)
        else:
            y_orig_tr = y_orig if cfg.original_mode == "all" else y_orig.iloc[orig_tr_idx].reset_index(drop=True)
            X_tr = pd.concat([X_tr_synth, X_orig_tr], axis=0).reset_index(drop=True)
            y_tr = pd.concat([y_tr_synth, y_orig_tr.reset_index(drop=True)], axis=0).reset_index(drop=True)
            w_tr = np.concatenate([
                np.full(len(y_tr_synth), float(params.get("synth_weight", 1.0)), dtype=np.float32),
                np.full(len(y_orig_tr), float(params.get("original_weight", 1.0)), dtype=np.float32),
            ])
        del X_tr_synth, X_orig_tr

        if cfg.target_encode_all_cats:
            te_cols = infer_target_encoder_cols(X_tr, X_val, X_test)
        elif not cfg.target_encode_all_combos:
            te_cols = []
        print("TE columns:", len(te_cols), te_cols[:10], flush=True)
        X_tr, X_val, X_test, _ = add_target_encoding(X_tr, y_tr, X_val, X_test, te_cols, cfg, fold)

        val_pred, fold_test = fit_predict_fold(X_tr, y_tr, w_tr, X_val, y_val, X_test, params)
        print(f"{cfg.pair} fold {fold} AUC: {roc_auc_score(y_val, val_pred):.6f}", flush=True)
        save_part(folder_path, cfg.pair, fold - 1, np.clip(val_pred, 1e-7, 1 - 1e-7), np.clip(fold_test, 1e-7, 1 - 1e-7))

        del X_tr, X_val, X_test, y_tr, y_val, val_pred, fold_test
        torch.cuda.empty_cache()
        gc.collect()

    return assemble(folder_path, cfg.pair, fold_ids, y_comp.to_numpy())
