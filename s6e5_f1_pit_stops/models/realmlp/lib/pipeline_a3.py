"""Extended Pipeline A, optional lap-delta shape / race-state blocks and their RealMLP defaults. The fold loop is the
one of pipeline_a.py (run_experiment below passes this module's features and parameters); features are always built on
the training rows of each fold (fold-local), one model per fold.

Extended Pipeline A block (41 columns): Pipeline A of pipeline_a.py plus LapTime x Cumulative_Degradation,
LapTime x |Cumulative_Degradation|, LapTime / (|Cumulative_Degradation| + 1e-6) (after the two ratios) and 7 quantile bins
of LapTime (after the 200 RaceProgress bins).

Optional blocks, added to every frame in this order (strings are categorical inputs of RealMLP):
- add_extra_combos: string keys Race x Compound x Stint, Driver x Race, Driver x Compound, Compound x Stint, Race x Stint,
  Race x Driver x Compound;
- add_round_cats: LapTime, LapTime_Delta, Cumulative_Degradation, RaceProgress rounded to 0, 1 and 2 decimals;
- lap-delta shape: q = piecewise quadratic of LapTime_Delta (two branches split at 50), shape = (q + 4.4)^2;
  numerics shape, (LapNumber - 50)^2, (RaceProgress - 0.6)^2; categories LapTime rounded to 0.1, floor(shape / 2),
  floor(Cumulative_Degradation / 2), round((LapNumber - 50)^2); the first letter of the driver code;
- race state: laps left = max(clip(LapNumber / RaceProgress, 1, 120) - LapNumber, 0); 11 numerics (stint start lap,
  laps left, TyreLife / (laps left + 1), TyreLife x RaceProgress, degradation / tyre, delta / tyre, |delta| / tyre,
  LapTime - degradation, Position x RaceProgress, Stint x TyreLife, |Position_Change|); categories race quarter,
  tyre-age bin, laps-left bin, Compound x tyre-age bin, Compound x quarter.
With target_encode_extra_combos, the categorical columns of these blocks are target encoded together with the two
combination codes of Pipeline A (pipeline_a.add_target_encoding)."""
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from common.data import ID, TARGET

from . import pipeline_a
from .pipeline_a import EPS


@dataclass
class ExpConfig(pipeline_a.ExpConfig):
    use_global_fe: bool = False
    add_extra_combos: bool = False
    add_round_cats: bool = False
    add_shape_numeric: bool = False
    add_shape_cats: bool = False
    add_driver_initial: bool = False
    add_state_numeric: bool = False
    add_state_cats: bool = False
    target_encode_extra_combos: bool = False


BASE_PARAMS = {
    "random_state": 42,
    "verbosity": 2,
    "val_metric_name": "1-auc_ovr",
    "n_ens": 24,
    "n_epochs": 6,
    "batch_size": 256,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    "lr": 0.01,
    "wd": 0.016,
    "sq_mom": 0.99,
    "lr_sched": "lin_cos_log_15",
    "first_layer_lr_factor": 0.25,
    "embedding_size": 6,
    "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128],
    "act": "silu",
    "p_drop": 0.05,
    "p_drop_sched": "invsqrtp1e-3",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_act_name": "gelu",
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "ls_eps": 0.01,
    "ls_eps_sched": "sqrt_cos",
    "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
}


def clean_name(col: str) -> str:
    return str(col).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def pipeline_a_fe(df: pd.DataFrame, fit: bool = False, state: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if state is None:
        state = {}

    cat_cols_base = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols_base = df.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()
    num_cols_base = [c for c in num_cols_base if c not in [ID, TARGET]]

    lap = pd.to_numeric(df["LapNumber"], errors="coerce").fillna(0)
    progress = pd.to_numeric(df["RaceProgress"], errors="coerce").fillna(0)
    tyre = pd.to_numeric(df["TyreLife"], errors="coerce").fillna(0)
    laptime = pd.to_numeric(df["LapTime (s)"], errors="coerce").fillna(0)
    deg = pd.to_numeric(df["Cumulative_Degradation"], errors="coerce").fillna(0)

    df["_LP_RP"] = (lap / (progress + EPS)).astype("float32")
    df["_TL_LP"] = (tyre / lap.clip(lower=1)).astype("float32")
    df["_LT_CD"] = (laptime * deg).astype("float32")
    df["_LT_CD_ABS"] = (laptime * deg.abs()).astype("float32")
    df["_LT_DIV_CD_ABS"] = (laptime / (deg.abs() + EPS)).astype("float32")

    for col in num_cols_base + ["_LP_RP", "_TL_LP"]:
        cat_name = f"{col}_cat_" if col in num_cols_base else f"{col[1:]}_cat_"
        x = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if fit:
            _, uniques = np.floor(x).factorize()
            state[col] = uniques
        code_map = {cat: i for i, cat in enumerate(state[col])}
        codes = np.floor(x).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes.astype(str)

    for col in cat_cols_base + ["Year_cat_", "PitStop_cat_"]:
        cnt_name = f"_{col}_count" if col in cat_cols_base else f"_{col[:-1]}_count"
        if fit:
            state[cnt_name] = df[col].astype(str).value_counts(dropna=False)
        df[cnt_name] = df[col].astype(str).map(state[cnt_name]).fillna(0).astype("int32")

    for col, nb in [("RaceProgress", 200), ("LapTime (s)", 7)]:
        bn = f"{col}_{nb}_qbin_"
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
        if fit:
            kb = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy="quantile", subsample=None)
            kb.fit(x)
            state[bn] = kb
        df[bn] = state[bn].transform(x).ravel().astype("int32").astype(str)

    combo_names = []
    for cols in [("Race", "Compound"), ("Race", "Year")]:
        cn = "_".join(cols) + "_"
        combo_names.append(cn)
        s = df[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "_" + df[c].astype(str)
        if fit:
            _, uniques = pd.factorize(s, sort=False)
            state[cn] = uniques
        code_map = {cat: i for i, cat in enumerate(state[cn])}
        codes = s.map(code_map).fillna(-1).astype("int32")
        df[cn] = codes.astype(str)

    return df, state, combo_names


def add_extra_combo_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    names = []
    for cols in [("Race", "Compound", "Stint"), ("Driver", "Race"), ("Driver", "Compound"), ("Compound", "Stint"),
                 ("Race", "Stint"), ("Race", "Driver", "Compound")]:
        name = "_".join(cols) + "_"
        s = out[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "_" + out[c].astype(str)
        out[name] = s.astype(str)
        names.append(name)
    return out, names


def add_round_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    cats = []
    for col in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]:
        x = pd.to_numeric(out[col], errors="coerce").fillna(0)
        safe = clean_name(col)
        for nd in [0, 1, 2]:
            name = f"ROUND_{safe}_{nd}_"
            out[name] = x.round(nd).astype(str)
            cats.append(name)
    return out, cats


def add_delta_shape_features(df: pd.DataFrame, add_numeric: bool = False, add_cats: bool = False,
                             add_driver_initial: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    cats = []
    lap = pd.to_numeric(out["LapNumber"], errors="coerce").fillna(0).astype(float)
    progress = pd.to_numeric(out["RaceProgress"], errors="coerce").fillna(0).astype(float)
    delta = pd.to_numeric(out["LapTime_Delta"], errors="coerce").fillna(0).astype(float)
    laptime = pd.to_numeric(out["LapTime (s)"], errors="coerce").fillna(0).astype(float)
    deg = pd.to_numeric(out["Cumulative_Degradation"], errors="coerce").fillna(0).astype(float)

    delta_x2 = np.where(
        delta <= 50,
        -0.000001 * delta**2 - 0.001191 * delta + 0.195376,
        -0.000001 * delta**2 - 0.001288 * delta + 0.184553,
    )
    delta_x2 = (delta_x2 + 4.4) ** 2

    if add_numeric:
        out["SHAPE_LapDelta_x2"] = delta_x2.astype("float32")
        out["SHAPE_LapNumber_center50_sq"] = ((lap - 50) ** 2).astype("float32")
        out["SHAPE_RaceProgress_center06_sq"] = ((progress - 0.6) ** 2).astype("float32")
    if add_driver_initial and "Driver" in out.columns:
        out["SHAPE_driver0_"] = out["Driver"].astype(str).str[0].fillna("").astype(str)
        cats.append("SHAPE_driver0_")
    if add_cats:
        out["SHAPE_LapTime_round1_"] = laptime.round(1).astype(str)
        out["SHAPE_LapDelta_x2_bin2_"] = (np.floor(delta_x2 / 2).astype(int)).astype(str)
        out["SHAPE_CumDeg_bin2_"] = (np.floor(deg / 2).astype(int)).astype(str)
        out["SHAPE_LapNumber_sq_cat_"] = ((lap - 50) ** 2).round(0).astype(int).astype(str)
        cats.extend(["SHAPE_LapTime_round1_", "SHAPE_LapDelta_x2_bin2_", "SHAPE_CumDeg_bin2_", "SHAPE_LapNumber_sq_cat_"])
    return out, cats


def add_race_state_features(df: pd.DataFrame, add_numeric: bool = True, add_cats: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    cats = []
    lap = pd.to_numeric(out["LapNumber"], errors="coerce").fillna(0).astype(float)
    progress = pd.to_numeric(out["RaceProgress"], errors="coerce").fillna(0).astype(float)
    tyre = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(0).astype(float)
    stint = pd.to_numeric(out["Stint"], errors="coerce").fillna(0).astype(float)
    pos = pd.to_numeric(out["Position"], errors="coerce").fillna(0).astype(float)
    pos_chg = pd.to_numeric(out["Position_Change"], errors="coerce").fillna(0).astype(float)
    delta = pd.to_numeric(out["LapTime_Delta"], errors="coerce").fillna(0).astype(float)
    deg = pd.to_numeric(out["Cumulative_Degradation"], errors="coerce").fillna(0).astype(float)
    laptime = pd.to_numeric(out["LapTime (s)"], errors="coerce").fillna(0).astype(float)

    est_total = (lap / np.clip(progress, EPS, None)).clip(1, 120)
    laps_left = np.maximum(est_total - lap, 0)
    stint_start = lap - tyre

    if add_numeric:
        out["STATE_stint_start_lap"] = stint_start.astype("float32")
        out["STATE_laps_left"] = laps_left.astype("float32")
        out["STATE_tyre_to_laps_left"] = (tyre / (laps_left + 1)).astype("float32")
        out["STATE_tyre_progress"] = (tyre * progress).astype("float32")
        out["STATE_deg_per_tyre"] = (deg / np.clip(tyre, 1, None)).astype("float32")
        out["STATE_delta_per_tyre"] = (delta / np.clip(tyre, 1, None)).astype("float32")
        out["STATE_abs_delta_per_tyre"] = (np.abs(delta) / np.clip(tyre, 1, None)).astype("float32")
        out["STATE_laptime_minus_deg"] = (laptime - deg).astype("float32")
        out["STATE_pos_pressure"] = (pos * progress).astype("float32")
        out["STATE_stint_tyre"] = (stint * tyre).astype("float32")
        out["STATE_poschg_abs"] = np.abs(pos_chg).astype("float32")

    if add_cats:
        out["STATE_phase_"] = pd.cut(progress, bins=[-np.inf, 0.25, 0.50, 0.75, np.inf], labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
        out["STATE_tyre_bin_"] = pd.cut(tyre, bins=[-np.inf, 3, 7, 12, 18, 26, np.inf], labels=["T0", "T1", "T2", "T3", "T4", "T5"]).astype(str)
        out["STATE_laps_left_bin_"] = pd.cut(laps_left, bins=[-np.inf, 5, 12, 25, 40, np.inf], labels=["L0", "L1", "L2", "L3", "L4"]).astype(str)
        out["STATE_compound_tyre_"] = out["Compound"].astype(str) + "_" + out["STATE_tyre_bin_"].astype(str)
        out["STATE_compound_phase_"] = out["Compound"].astype(str) + "_" + out["STATE_phase_"].astype(str)
        cats.extend(["STATE_phase_", "STATE_tyre_bin_", "STATE_laps_left_bin_", "STATE_compound_tyre_", "STATE_compound_phase_"])
    return out, cats


def build_features_fold_local(cfg, X_tr_base, X_val_base, X_test_base, X_orig_base):
    """Extended Pipeline A fitted on the fold's training rows, then the enabled blocks on all four frames.
    Returns the four frames and the columns to target encode."""
    X_tr, state, combo_names = pipeline_a_fe(X_tr_base, fit=True)
    frames = [X_tr] + [pipeline_a_fe(X, state=state)[0] for X in (X_val_base, X_test_base, X_orig_base)]
    blocks = []
    if cfg.add_extra_combos:
        blocks.append(add_extra_combo_features)
    if cfg.add_round_cats:
        blocks.append(add_round_categorical_features)
    if cfg.add_shape_numeric or cfg.add_shape_cats or cfg.add_driver_initial:
        blocks.append(partial(add_delta_shape_features, add_numeric=cfg.add_shape_numeric, add_cats=cfg.add_shape_cats,
                              add_driver_initial=cfg.add_driver_initial))
    if cfg.add_state_numeric or cfg.add_state_cats:
        blocks.append(partial(add_race_state_features, add_numeric=cfg.add_state_numeric, add_cats=cfg.add_state_cats))
    extra_te_cols = []
    for block in blocks:
        results = [block(X) for X in frames]
        frames = [X for X, _ in results]
        extra_te_cols += results[0][1]
    te_cols = list(combo_names) + (extra_te_cols if cfg.target_encode_extra_combos else [])
    X_tr, X_val, X_test, X_orig = frames
    return X_tr, X_val, X_test, X_orig, list(dict.fromkeys(te_cols))


def run_experiment(cfg: ExpConfig):
    """Trains every fold (fold loop of pipeline_a.py) and returns (oof, test) probabilities."""
    return pipeline_a.run_experiment(cfg, base_params=BASE_PARAMS, fold_features=build_features_fold_local)
