"""Domain / artifact-category / original-prior feature frames and the per-fold CatBoost fit of ../train.py.

build_frames(cfg) builds one label-free feature frame each for the train, test and original rows (counts, group and
sequence statistics are taken over train + test + original; a few per-frame blocks use each frame alone). The optional
blocks are switched on by the cfg flags. fit_fold(cfg, frames, fold_id, k) then, for outer fold k, maps the
original-label priors OP_<key>_mean / OP_<key>_count of all original rows onto every frame, appends all original rows to
the training side (sample weight cfg['original_weight']) and fits GPU CatBoost with early stopping on the validation
fold (use_best_model=True). String columns are CatBoost-native categoricals."""
import os
import random
import re
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from common.data import ID, TARGET, read_original, read_test, read_train

warnings.filterwarnings("ignore", category=PerformanceWarning)

SEED = 42

BASE_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.033,
    "depth": 8,
    "l2_leaf_reg": 7.0,
    "random_strength": 0.8,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.55,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "task_type": "GPU",
    "devices": "0",
    "early_stopping_rounds": 200,
    "allow_writing_files": False,
    "verbose": 200,
}
# overrides of BASE_PARAMS; None removes the key
PARAM_SETS = {
    "deeper_reg": dict(iterations=5000, learning_rate=0.025, depth=9, l2_leaf_reg=10.0, random_strength=1.2,
                       bagging_temperature=0.7, early_stopping_rounds=250),
    "no_balanced": dict(auto_class_weights=None, iterations=5000, learning_rate=0.03, early_stopping_rounds=250),
    "deeper_reg_soft": dict(iterations=6500, learning_rate=0.020, depth=9, l2_leaf_reg=14.0, random_strength=1.35,
                            bagging_temperature=0.75, early_stopping_rounds=350),
    "deeper_reg_soft_ctr4": dict(iterations=7600, learning_rate=0.020, depth=9, l2_leaf_reg=14.0, random_strength=1.35,
                                 bagging_temperature=0.75, early_stopping_rounds=400, one_hot_max_size=12,
                                 max_ctr_complexity=4),
}


def catboost_params(variant, seed):
    params = dict(BASE_PARAMS, random_seed=seed)
    for key, value in PARAM_SETS[variant].items():
        if value is None:
            params.pop(key, None)
        else:
            params[key] = value
    return params


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def clean_cat(s):
    return s.astype("string").fillna("__MISS__").astype(str)


def safe_col(name):
    out = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    out = re.sub(r"_+", "_", out)
    return out or "feature"


def add_cross(df, name, cols):
    if not set(cols).issubset(df.columns):
        return
    v = df[cols[0]].astype(str)
    for col in cols[1:]:
        v = v + "_" + df[col].astype(str)
    df[name] = v.astype(str)


def add_domain_features(df):
    eps = 1e-6
    for col in ["Driver", "Compound", "Race"]:
        if col in df.columns:
            df[col] = clean_cat(df[col])

    if {"LapNumber", "RaceProgress"}.issubset(df.columns):
        df["EstTotalLaps"] = safe_div(df["LapNumber"], df["RaceProgress"].clip(lower=eps), eps).clip(1, 120)
        df["LapsRemaining"] = (df["EstTotalLaps"] - df["LapNumber"]).clip(lower=0)
    if {"TyreLife", "LapNumber"}.issubset(df.columns):
        df["TyreAgeRatio"] = safe_div(df["TyreLife"], df["LapNumber"].clip(lower=1), eps)
        df["LapPerTyreLife"] = safe_div(df["LapNumber"], df["TyreLife"] + 1, eps)
        df["LapMinusTyre"] = df["LapNumber"] - df["TyreLife"]
    if {"TyreLife", "RaceProgress"}.issubset(df.columns):
        df["PitWindowPressure"] = df["TyreLife"] * df["RaceProgress"]
    if {"TyreLife", "LapsRemaining"}.issubset(df.columns):
        df["TyreToLapsRem"] = safe_div(df["TyreLife"], df["LapsRemaining"] + 1, eps)
    if {"Cumulative_Degradation", "TyreLife"}.issubset(df.columns):
        df["DegPerTyreLap"] = safe_div(df["Cumulative_Degradation"], df["TyreLife"].clip(lower=1), eps)
        df["AbsDegPerTyreLap"] = safe_div(df["Cumulative_Degradation"].abs(), df["TyreLife"].clip(lower=1), eps)
    if {"LapTime_Delta", "TyreLife"}.issubset(df.columns):
        df["DeltaPerTyreLap"] = safe_div(df["LapTime_Delta"], df["TyreLife"].clip(lower=1), eps)
        df["AbsDeltaPerTyreLap"] = safe_div(df["LapTime_Delta"].abs(), df["TyreLife"].clip(lower=1), eps)
    if "LapTime_Delta" in df.columns:
        df["DeltaAbs"] = df["LapTime_Delta"].abs()
    if "Cumulative_Degradation" in df.columns:
        df["AbsCumDeg"] = df["Cumulative_Degradation"].abs()
    if "Position_Change" in df.columns:
        df["AbsPosChange"] = df["Position_Change"].abs()
    if {"Position", "RaceProgress"}.issubset(df.columns):
        df["PosPressure"] = df["Position"] * df["RaceProgress"]
    if {"Stint", "TyreLife"}.issubset(df.columns):
        df["StintPressure"] = df["Stint"] * df["TyreLife"]

    if "RaceProgress" in df.columns:
        df["RacePhase"] = pd.cut(
            df["RaceProgress"],
            bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
            labels=["P1", "P2", "P3", "P4", "P5"],
        ).astype(str)
    if "TyreLife" in df.columns:
        df["TyreLifeBin"] = pd.cut(
            df["TyreLife"],
            bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
            labels=["T0", "T1", "T2", "T3", "T4", "T5"],
        ).astype(str)
    if "Position" in df.columns:
        df["PosBin"] = pd.cut(
            df["Position"],
            bins=[-np.inf, 3, 8, 14, np.inf],
            labels=["front", "upper", "lower", "back"],
        ).astype(str)

    crosses = [
        ("Race_Year", ["Race", "Year"]),
        ("Compound_Stint", ["Compound", "Stint"]),
        ("Driver_Race", ["Driver", "Race"]),
        ("Driver_Compound", ["Driver", "Compound"]),
        ("Race_Compound", ["Race", "Compound"]),
        ("Race_Compound_Stint", ["Race", "Compound", "Stint"]),
        ("Compound_RacePhase", ["Compound", "RacePhase"]),
        ("Compound_TyreLifeBin", ["Compound", "TyreLifeBin"]),
        ("RacePhase_TyreLifeBin", ["RacePhase", "TyreLifeBin"]),
    ]
    for name, cols in crosses:
        add_cross(df, name, cols)

    if "TyreAgeRatio" in df.columns:
        df["TyreAgeRatio_str"] = df["TyreAgeRatio"].round(3).astype(str)
    if "EstTotalLaps" in df.columns:
        df["EstTotalLaps_str"] = df["EstTotalLaps"].round(1).astype(str)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype(np.float32)
    return df


def add_low_card_numeric_cats(df):
    for col in ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "Position_Change"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int64").astype(str)
            df[f"CAT_{col}"] = vals.replace("<NA>", "__MISS__")
    return df


def add_artifact_cats(df):
    """Rounded-value categories (the exact decimals of the synthetic numerics) and two integer crosses."""
    specs = [
        ("LapTime (s)", [0, 1, 2]),
        ("LapTime_Delta", [0, 1, 2, 3]),
        ("Cumulative_Degradation", [0, 1, 2, 3]),
        ("RaceProgress", [2, 3, 4]),
        ("TyreAgeRatio", [2, 3, 4]),
        ("DegPerTyreLap", [1, 2, 3]),
        ("DeltaAbs", [1, 2, 3]),
    ]
    for col, rounds in specs:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        for r in rounds:
            df[f"CAT_{col}_r{r}"] = x.round(r).astype(str)
        df[f"CAT_{col}_floor"] = np.floor(x.fillna(0)).astype("int32").astype(str)
    if {"LapNumber", "TyreLife"}.issubset(df.columns):
        add_cross(df, "CAT_LapNumber_TyreLife", ["LapNumber", "TyreLife"])
    if {"TyreLife", "Position_Change"}.issubset(df.columns):
        add_cross(df, "CAT_TyreLife_PositionChange", ["TyreLife", "Position_Change"])
    return df


def add_public_formula_features(df):
    """Ratio / interaction / phase / tyre-age formulas of the public notebooks (tyre quantiles taken per frame)."""
    eps = 1e-6
    lap_number = pd.to_numeric(df.get("LapNumber", pd.Series(0, index=df.index)), errors="coerce")
    tyre = pd.to_numeric(df.get("TyreLife", pd.Series(0, index=df.index)), errors="coerce")
    progress = pd.to_numeric(df.get("RaceProgress", pd.Series(0, index=df.index)), errors="coerce")
    lap_time = pd.to_numeric(df.get("LapTime (s)", pd.Series(0, index=df.index)), errors="coerce")
    delta = pd.to_numeric(df.get("LapTime_Delta", pd.Series(0, index=df.index)), errors="coerce")
    deg = pd.to_numeric(df.get("Cumulative_Degradation", pd.Series(0, index=df.index)), errors="coerce")
    position = pd.to_numeric(df.get("Position", pd.Series(0, index=df.index)), errors="coerce")
    pos_change = pd.to_numeric(df.get("Position_Change", pd.Series(0, index=df.index)), errors="coerce")
    remaining = 1.0 - progress

    df["PUB_Lap_TyreLife_Ratio"] = lap_number / (tyre + 1.0)
    df["PUB_TyreLife_per_Progress"] = tyre / (progress + 0.05)
    df["PUB_LapTime_per_TyreLife"] = lap_time / (tyre + 1.0)
    df["PUB_Degradation_per_TyreLife"] = deg / (tyre + 1.0)
    df["PUB_LapDelta_per_TyreLife"] = delta / (tyre + 1.0)
    df["PUB_Degradation_per_Lap"] = deg / (lap_number + 1.0)
    df["PUB_LapTime_x_RaceProgress"] = lap_time * progress
    df["PUB_LapDelta_x_TyreLife"] = delta * tyre
    df["PUB_Position_x_TyreLife"] = position * tyre
    df["PUB_RemainingRace_x_TyreLife"] = remaining * tyre
    df["PUB_Deg_Rate"] = deg / tyre.clip(lower=1)
    df["PUB_Cum_Stress"] = tyre * deg.abs().clip(upper=200)
    df["PUB_LTD_Abs"] = delta.abs()
    df["PUB_Pos_Mom"] = pos_change.clip(-5, 5)
    df["PUB_NoPositionChange"] = (pos_change.abs() < eps).astype(np.int8)
    df["PUB_IsEarlyRace"] = (progress <= 0.25).astype(np.int8)
    df["PUB_IsMidRace"] = ((progress > 0.25) & (progress < 0.75)).astype(np.int8)
    df["PUB_IsLateRace"] = (progress >= 0.75).astype(np.int8)
    tyre_q25, tyre_q50, tyre_q75 = tyre.quantile([0.25, 0.50, 0.75]).values
    df["PUB_IsFreshTyre"] = (tyre <= tyre_q25).astype(np.int8)
    df["PUB_IsOldTyre"] = (tyre >= tyre_q50).astype(np.int8)
    df["PUB_IsVeryOldTyre"] = (tyre >= tyre_q75).astype(np.int8)
    if "Compound" in df.columns:
        comp = clean_cat(df["Compound"]).str.upper()
        compound_life = comp.map({"SOFT": 18.0, "MEDIUM": 25.0, "HARD": 32.0, "INTERMEDIATE": 18.0, "WET": 15.0}).fillna(25.0)
        df["PUB_TL_Ratio_CompoundLife"] = tyre / compound_life
        for name in ["SOFT", "MEDIUM", "HARD"]:
            mask = comp.str.contains(name, regex=False).astype(np.int8)
            df[f"PUB_{name}_x_TyreLife"] = mask * tyre
            df[f"PUB_{name}_x_Progress"] = mask * progress
    return df


def add_pipeline_a_features(df):
    """Floor categories of the numerics, per-frame counts of a few keys, 200 per-frame progress quantile bins."""
    raw_num_cols = [
        "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
        "Position_Change",
    ]
    if {"LapNumber", "RaceProgress"}.issubset(df.columns):
        df["PIPA_LP_RP"] = pd.to_numeric(df["LapNumber"], errors="coerce") / (pd.to_numeric(df["RaceProgress"], errors="coerce") + 1e-6)
    if {"TyreLife", "LapNumber"}.issubset(df.columns):
        df["PIPA_TL_LP"] = pd.to_numeric(df["TyreLife"], errors="coerce") / pd.to_numeric(df["LapNumber"], errors="coerce").clip(lower=1)
    for col in raw_num_cols + ["PIPA_LP_RP", "PIPA_TL_LP"]:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        df[f"PIPA_{safe_col(col)}_floor_cat"] = np.floor(x).astype("Int64").astype("string").fillna("__MISSING__").astype(str)
    for col in ["Driver", "Compound", "Race", "PIPA_Year_floor_cat", "PIPA_PitStop_floor_cat"]:
        if col not in df.columns:
            continue
        key = clean_cat(df[col])
        vc = key.value_counts(dropna=False)
        df[f"PIPA_{safe_col(col)}_count"] = key.map(vc).fillna(0).astype(np.float32)
    if "RaceProgress" in df.columns:
        x = pd.to_numeric(df["RaceProgress"], errors="coerce")
        try:
            df["PIPA_RaceProgress_200_qbin"] = pd.qcut(x.rank(method="first"), q=200, labels=False, duplicates="drop").astype("Int64").astype("string").fillna("__MISSING__").astype(str)
        except ValueError:
            df["PIPA_RaceProgress_200_qbin"] = "0"
    for combo in [("Race", "Compound"), ("Race", "Year")]:
        if all(c in df.columns for c in combo):
            add_cross(df, "PIPA_" + "_".join(combo), list(combo))
    return df


def add_digit_features(df, src_cols, int_lim=3, dec_lim=2):
    for col in src_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).abs()
        for i in range(int_lim):
            df[f"{col}_id{i + 1}"] = ((s // (10**i)) % 10).astype(np.int8)
        if pd.api.types.is_float_dtype(df[col]):
            for i in range(1, dec_lim + 1):
                df[f"{col}_dd{i}"] = ((s * (10**i)).round().astype(int) % 10).astype(np.int8)
    return df


def add_sig_features(df):
    """The 4 lowest digits of 100 x value of 10 numerics, as categories."""
    selected = [
        "RaceProgress",
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "TyreAgeRatio",
        "DegPerTyreLap",
        "DeltaAbs",
        "PitWindowPressure",
        "EstTotalLaps",
        "LapsRemaining",
    ]
    for col in [c for c in selected if c in df.columns]:
        scaled = (pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float) * 100).round().astype(int).abs()
        for i in range(4):
            d = ((scaled // (10**i)) % 10).astype(np.int8)
            df[f"{col}_s{i + 1}"] = d.astype(str)
    return df


def rank_bin_str(x, n_bins=20):
    r = pd.to_numeric(x, errors="coerce").rank(pct=True, method="average").fillna(0.5).clip(0, 1)
    try:
        return pd.cut(
            r,
            bins=np.linspace(0, 1, n_bins + 1),
            include_lowest=True,
            duplicates="drop",
        ).astype("string").fillna("__MISSING__").astype(str)
    except ValueError:
        return pd.Series("0", index=x.index, dtype="object")


def rounded_cat(x, ndigits):
    y = pd.to_numeric(x, errors="coerce")
    if ndigits == 0:
        return y.round(0).astype("Int64").astype("string").fillna("__MISSING__").astype(str)
    return y.round(ndigits).astype("string").fillna("__MISSING__").astype(str)


def add_hazard_features(df):
    """Tyre age relative to a nominal compound life, risk products and their bins."""
    eps = 1e-6
    comp = clean_cat(df["Compound"]).str.upper() if "Compound" in df.columns else pd.Series("", index=df.index)
    lap = pd.to_numeric(df.get("LapNumber", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    tyre = pd.to_numeric(df.get("TyreLife", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    progress = pd.to_numeric(df.get("RaceProgress", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    delta = pd.to_numeric(df.get("LapTime_Delta", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    deg = pd.to_numeric(df.get("Cumulative_Degradation", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    pos = pd.to_numeric(df.get("Position", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    stint = pd.to_numeric(df.get("Stint", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    pitstop = pd.to_numeric(df.get("PitStop", pd.Series(0, index=df.index)), errors="coerce").fillna(0)

    compound_life = comp.map({
        "SOFT": 18.0,
        "MEDIUM": 25.0,
        "HARD": 32.0,
        "INTERMEDIATE": 18.0,
        "WET": 14.0,
    }).fillna(24.0).astype(np.float32)
    df["CAT2_CompoundLife"] = compound_life.astype("int16")
    df["CAT2_TyreFracCompoundLife"] = (tyre / (compound_life + eps)).astype(np.float32)
    df["CAT2_LapsPastCompoundLife"] = (tyre - compound_life).astype(np.float32)
    df["CAT2_LapsToCompoundLife"] = (compound_life - tyre).astype(np.float32)
    df["CAT2_RiskAgeProgress"] = (tyre / (compound_life + eps) * (0.35 + progress)).astype(np.float32)
    df["CAT2_RiskDegAge"] = (deg.abs() * tyre / (compound_life + eps)).astype(np.float32)
    df["CAT2_RiskDeltaAge"] = (delta.abs() * tyre / (compound_life + eps)).astype(np.float32)
    df["CAT2_TrackPositionPressure"] = ((21 - pos).clip(lower=0) * progress).astype(np.float32)
    df["CAT2_StopCountTyreAge"] = ((pitstop + 1) * tyre).astype(np.float32)
    df["CAT2_StintLapPressure"] = ((stint + 1) * lap * progress).astype(np.float32)

    df["CAT2_TyreFracBin"] = pd.cut(
        df["CAT2_TyreFracCompoundLife"],
        bins=[-np.inf, 0.25, 0.45, 0.65, 0.85, 1.05, 1.30, np.inf],
        labels=["tf0", "tf1", "tf2", "tf3", "tf4", "tf5", "tf6"],
    ).astype("string").fillna("__MISSING__").astype(str)
    df["CAT2_LapsToLifeBin"] = pd.cut(
        df["CAT2_LapsToCompoundLife"],
        bins=[-np.inf, -10, -5, 0, 3, 6, 10, 16, np.inf],
        labels=["past10", "past5", "due", "soon3", "soon6", "soon10", "soon16", "far"],
    ).astype("string").fillna("__MISSING__").astype(str)
    df["CAT2_RiskAgeBin"] = pd.cut(
        df["CAT2_RiskAgeProgress"],
        bins=[-np.inf, 0.15, 0.35, 0.55, 0.80, 1.10, np.inf],
        labels=["r0", "r1", "r2", "r3", "r4", "r5"],
    ).astype("string").fillna("__MISSING__").astype(str)
    return df


def add_native_combo_features(df):
    """Hazard features plus rounded / 20-quantile / sign categories of 12 numerics and 23 categorical crosses."""
    for col in ["Driver", "Compound", "Race"]:
        if col in df.columns:
            df[col] = clean_cat(df[col])
    add_hazard_features(df)

    if "RaceProgress" in df.columns and "CAT2_RaceProgress_q20" not in df.columns:
        df["CAT2_RaceProgress_q20"] = rank_bin_str(df["RaceProgress"], 20)
    if "LapNumber" in df.columns:
        df["CAT2_LapNumber_q20"] = rank_bin_str(df["LapNumber"], 20)
    if "TyreLife" in df.columns:
        df["CAT2_TyreLife_q20"] = rank_bin_str(df["TyreLife"], 20)
    if "Position" in df.columns:
        df["CAT2_Position_q10"] = rank_bin_str(df["Position"], 10)

    num_specs = {
        "LapTime (s)": [0, 1, 2],
        "LapTime_Delta": [1, 2, 3],
        "Cumulative_Degradation": [1, 2, 3],
        "RaceProgress": [2, 3, 4],
        "TyreAgeRatio": [2, 3],
        "DegPerTyreLap": [1, 2, 3],
        "DeltaPerTyreLap": [1, 2, 3],
        "CAT2_TyreFracCompoundLife": [1, 2],
        "CAT2_LapsToCompoundLife": [0, 1],
        "CAT2_RiskAgeProgress": [2, 3],
        "CAT2_RiskDegAge": [1, 2],
        "CAT2_RiskDeltaAge": [1, 2],
    }
    for col, rounds in num_specs.items():
        if col not in df.columns:
            continue
        safe = safe_col(col)
        x = pd.to_numeric(df[col], errors="coerce")
        for nd in rounds:
            df[f"CAT2_{safe}_r{nd}"] = rounded_cat(x, nd)
        df[f"CAT2_{safe}_q20"] = rank_bin_str(x, 20)
        df[f"CAT2_{safe}_sign"] = np.sign(x.fillna(0)).astype("int8").astype(str)

    crosses = [
        ("CAT2_Race_Year_Compound_Stint", ["Race", "Year", "Compound", "Stint"]),
        ("CAT2_Driver_Race_Year", ["Driver", "Race", "Year"]),
        ("CAT2_Driver_Race_Stint", ["Driver", "Race", "Stint"]),
        ("CAT2_Driver_Compound_Stint", ["Driver", "Compound", "Stint"]),
        ("CAT2_Race_Compound_TyreFracBin", ["Race", "Compound", "CAT2_TyreFracBin"]),
        ("CAT2_Compound_Stint_TyreFracBin", ["Compound", "Stint", "CAT2_TyreFracBin"]),
        ("CAT2_Compound_RacePhase_RiskAgeBin", ["Compound", "RacePhase", "CAT2_RiskAgeBin"]),
        ("CAT2_Race_Lap_q20_Tyre_q20", ["Race", "CAT2_LapNumber_q20", "CAT2_TyreLife_q20"]),
        ("CAT2_Compound_LapsToLifeBin", ["Compound", "CAT2_LapsToLifeBin"]),
        ("CAT2_Race_Compound_LapsToLifeBin", ["Race", "Compound", "CAT2_LapsToLifeBin"]),
        ("CAT2_Driver_Race_LapsToLifeBin", ["Driver", "Race", "CAT2_LapsToLifeBin"]),
        ("CAT2_Driver_Race_Compound_TyreFracBin", ["Driver", "Race", "Compound", "CAT2_TyreFracBin"]),
        ("CAT2_Race_Year_Compound_LapsToLifeBin", ["Race", "Year", "Compound", "CAT2_LapsToLifeBin"]),
        ("CAT2_Driver_Compound_RiskAgeBin", ["Driver", "Compound", "CAT2_RiskAgeBin"]),
        ("CAT2_Race_Compound_Position_q10", ["Race", "Compound", "CAT2_Position_q10"]),
        ("CAT2_Race_Progress_q20_Compound", ["Race", "CAT2_RaceProgress_q20", "Compound"]),
        ("CAT2_Driver_Race_Lap_q20", ["Driver", "Race", "CAT2_LapNumber_q20"]),
    ]
    for name, cols in crosses:
        add_cross(df, name, cols)

    high_crosses = [
        ("LapTime_Delta", "r2", ["Compound", "CAT2_TyreFracBin"]),
        ("Cumulative_Degradation", "r2", ["Compound", "CAT2_TyreFracBin"]),
        ("LapTime (s)", "r1", ["Race", "Compound"]),
        ("RaceProgress", "r3", ["Race", "Compound"]),
        ("CAT2_RiskAgeProgress", "r2", ["Compound", "Stint"]),
        ("CAT2_LapsToCompoundLife", "r0", ["Compound", "RacePhase"]),
    ]
    for col, r, other_cols in high_crosses:
        safe = safe_col(col)
        cat_col = f"CAT2_{safe}_{r}"
        if cat_col in df.columns and all(c in df.columns for c in other_cols):
            add_cross(df, f"CAT2_{safe}_{r}__" + "_".join(map(safe_col, other_cols)), [cat_col] + other_cols)
    return df


def add_compound_hazard_light_features(df):
    if "Compound" not in df.columns or "TyreLife" not in df.columns:
        return df
    comp = clean_cat(df["Compound"]).str.upper()
    tyre = pd.to_numeric(df["TyreLife"], errors="coerce").fillna(0).astype(float)
    progress = pd.to_numeric(df.get("RaceProgress", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).astype(float)
    stint = pd.to_numeric(df.get("Stint", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).astype(float)
    delta = pd.to_numeric(df.get("LapTime_Delta", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).astype(float)
    deg = pd.to_numeric(df.get("Cumulative_Degradation", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).astype(float)

    life = comp.map({"SOFT": 18.0, "MEDIUM": 26.0, "HARD": 36.0, "INTERMEDIATE": 16.0, "WET": 14.0}).fillna(26.0)
    frac = (tyre / life.clip(lower=1.0)).clip(0, 3)
    left = (life - tyre).clip(-25, 80)
    stress = (frac * (1.0 + progress) * (1.0 + 0.08 * stint) * (1.0 + np.log1p(delta.abs() + deg.abs()).clip(0, 6) / 6.0)).clip(0, 8)

    df["CB2_CompoundLife"] = life.astype(np.float32)
    df["CB2_TyreFracLife"] = frac.astype(np.float32)
    df["CB2_LapsToLife"] = left.astype(np.float32)
    df["CB2_CompoundStress"] = stress.astype(np.float32)
    df["CB2_TyreFracLifeBin"] = pd.cut(frac, [-0.01, 0.25, 0.45, 0.65, 0.85, 1.05, 1.30, np.inf], labels=False).astype("Int64").astype(str)
    df["CB2_LapsToLifeBin"] = pd.cut(left, [-np.inf, -8, -2, 2, 6, 12, 20, np.inf], labels=False).astype("Int64").astype(str)
    df["CB2_StressBin"] = pd.cut(stress, [-0.01, 0.4, 0.7, 1.0, 1.3, 1.8, 2.5, np.inf], labels=False).astype("Int64").astype(str)
    add_cross(df, "CB2_Compound_TyreFracLifeBin", ["Compound", "CB2_TyreFracLifeBin"])
    add_cross(df, "CB2_Compound_LapsToLifeBin", ["Compound", "CB2_LapsToLifeBin"])
    if "RacePhase" in df.columns:
        add_cross(df, "CB2_Compound_RacePhase_StressBin", ["Compound", "RacePhase", "CB2_StressBin"])
    if "Race" in df.columns:
        add_cross(df, "CB2_Race_Compound_TyreFracLifeBin", ["Race", "Compound", "CB2_TyreFracLifeBin"])
    return df


def add_pit_state_cats(df):
    lap = pd.to_numeric(df.get("LapNumber", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    tyre = pd.to_numeric(df.get("TyreLife", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    progress = pd.to_numeric(df.get("RaceProgress", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    position = pd.to_numeric(df.get("Position", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    pitstop = pd.to_numeric(df.get("PitStop", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    stint = pd.to_numeric(df.get("Stint", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    delta = pd.to_numeric(df.get("LapTime_Delta", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)

    df["CB2_LapWindowBin"] = pd.cut(lap, [-np.inf, 5, 10, 15, 22, 30, 40, 55, np.inf], labels=False).astype("Int64").astype(str)
    df["CB2_TyreWindowBin"] = pd.cut(tyre, [-np.inf, 2, 5, 8, 12, 16, 22, 30, np.inf], labels=False).astype("Int64").astype(str)
    df["CB2_ProgressWindowBin"] = pd.cut(progress, [-np.inf, .10, .20, .35, .50, .65, .80, .92, np.inf], labels=False).astype("Int64").astype(str)
    df["CB2_PositionBand"] = pd.cut(position, [-np.inf, 3, 6, 10, 15, np.inf], labels=False).astype("Int64").astype(str)
    df["CB2_DeltaSignBand"] = pd.cut(delta, [-np.inf, -2, -0.5, 0.5, 2, np.inf], labels=False).astype("Int64").astype(str)
    df["CB2_PitStopCat"] = pitstop.round().clip(0, 5).astype("int16").astype(str)
    df["CB2_StintCat"] = stint.round().clip(0, 8).astype("int16").astype(str)

    for name, cols in [
        ("CB2_Compound_Stint_TyreWindow", ["Compound", "CB2_StintCat", "CB2_TyreWindowBin"]),
        ("CB2_Race_Compound_ProgressWindow", ["Race", "Compound", "CB2_ProgressWindowBin"]),
        ("CB2_Race_Stint_LapWindow", ["Race", "CB2_StintCat", "CB2_LapWindowBin"]),
        ("CB2_Driver_Race_StintWindow", ["Driver", "Race", "CB2_StintCat"]),
        ("CB2_Compound_Position_TyreWindow", ["Compound", "CB2_PositionBand", "CB2_TyreWindowBin"]),
        ("CB2_PitStop_Compound_TyreWindow", ["CB2_PitStopCat", "Compound", "CB2_TyreWindowBin"]),
        ("CB2_Compound_DeltaSign_TyreWindow", ["Compound", "CB2_DeltaSignBand", "CB2_TyreWindowBin"]),
    ]:
        add_cross(df, name, cols)
    return df


def add_strategy_cats(df):
    if "CB2_TyreWindowBin" not in df.columns or "CB2_ProgressWindowBin" not in df.columns:
        add_pit_state_cats(df)
    if "CB2_TyreFracLifeBin" not in df.columns:
        add_compound_hazard_light_features(df)
    for name, cols in [
        ("CB3_Driver_Race_Stint_TyreWindow", ["Driver", "Race", "CB2_StintCat", "CB2_TyreWindowBin"]),
        ("CB3_Race_Year_Compound_Stint", ["Race", "Year", "Compound", "CB2_StintCat"]),
        ("CB3_Race_Year_PitStop_Progress", ["Race", "Year", "CB2_PitStopCat", "CB2_ProgressWindowBin"]),
        ("CB3_Driver_Compound_TyreWindow", ["Driver", "Compound", "CB2_TyreWindowBin"]),
        ("CB3_Compound_Stint_PositionBand", ["Compound", "CB2_StintCat", "CB2_PositionBand"]),
        ("CB3_Race_Position_Progress", ["Race", "CB2_PositionBand", "CB2_ProgressWindowBin"]),
        ("CB3_Compound_Stress_Position", ["Compound", "CB2_StressBin", "CB2_PositionBand"]),
        ("CB3_PitStop_Stint_TyreWindow", ["CB2_PitStopCat", "CB2_StintCat", "CB2_TyreWindowBin"]),
    ]:
        add_cross(df, name, cols)
    return df


def add_frequency_features(frames, cols):
    cols = [c for c in cols if all(c in f.columns for f in frames)]
    if not cols:
        return frames
    base = pd.concat([f[cols] for f in frames], ignore_index=True)
    total = len(base)
    for col in cols:
        counts = base[col].astype(str).value_counts(dropna=False)
        for f in frames:
            f[f"{col}_cnt"] = f[col].astype(str).map(counts).fillna(0).astype(np.float32)
            f[f"{col}_frq"] = (f[f"{col}_cnt"] / total).astype(np.float32)
    return frames


def add_source_frequency_features(frames, cols):
    """Counts of each key in train, test and original separately and their ratios."""
    cols = [c for c in cols if all(c in f.columns for f in frames)]
    if not cols:
        return frames

    names = ["train", "test", "orig"]
    for col in cols:
        counts = {
            name: frame[col].astype(str).value_counts(dropna=False).astype(np.float32)
            for name, frame in zip(names, frames)
        }
        for f in frames:
            mapped = {}
            for name in names:
                mapped[name] = f[col].astype(str).map(counts[name]).fillna(0).astype(np.float32)
                f[f"SRC_{safe_col(col)}_{name}_cnt"] = mapped[name]
            f[f"SRC_{safe_col(col)}_train_orig_ratio"] = ((mapped["train"] + 1.0) / (mapped["orig"] + 1.0)).astype(np.float32)
            f[f"SRC_{safe_col(col)}_test_train_ratio"] = ((mapped["test"] + 1.0) / (mapped["train"] + 1.0)).astype(np.float32)
            f[f"SRC_{safe_col(col)}_orig_share"] = (
                (mapped["orig"] + 1.0) / (mapped["train"] + mapped["test"] + mapped["orig"] + 3.0)
            ).astype(np.float32)
    return frames


def add_relative_rank_features(frames, add_categories=False):
    """Within-group percentile rank, minus / ratio to the median and robust z of 6 numerics (train + test + original)."""
    lengths = [len(f) for f in frames]
    all_df = pd.concat(frames, ignore_index=True)
    num_cols = [
        c for c in [
            "TyreLife", "LapNumber", "RaceProgress", "LapTime_Delta",
            "Cumulative_Degradation", "LapTime (s)",
        ]
        if c in all_df.columns
    ]
    group_sets = [
        ["Year", "Race", "Stint", "Compound"],
        ["Race", "Stint", "Compound"],
        ["Race", "Compound"],
        ["Compound", "Stint"],
    ]
    for group in group_sets:
        if not all(c in all_df.columns for c in group):
            continue
        gname = "__".join(group)
        gb = all_df.groupby(group, dropna=False, sort=False)
        for col in num_cols:
            x = pd.to_numeric(all_df[col], errors="coerce")
            med = gb[col].transform("median")
            q25 = gb[col].transform(lambda s: s.quantile(0.25))
            q75 = gb[col].transform(lambda s: s.quantile(0.75))
            safe = safe_col(col)
            all_df[f"REL_{gname}__{safe}__rank_pct"] = gb[col].rank(pct=True, method="average").astype(np.float32)
            all_df[f"REL_{gname}__{safe}__minus_median"] = (x - med).astype(np.float32)
            all_df[f"REL_{gname}__{safe}__ratio_median"] = (x / (med.abs() + 1e-6)).astype(np.float32)
            all_df[f"REL_{gname}__{safe}__robust_z"] = ((x - med) / ((q75 - q25).abs() + 1e-6)).astype(np.float32)

    if add_categories:
        rank_cols = [
            c for c in all_df.columns
            if c.startswith("REL_") and c.endswith("__rank_pct")
            and any(token in c for token in ["TyreLife", "LapTime_Delta", "Cumulative_Degradation"])
        ]
        for col in rank_cols:
            x = pd.to_numeric(all_df[col], errors="coerce").clip(0, 1)
            try:
                all_df[f"RELCAT_{safe_col(col)}_bin10"] = pd.cut(
                    x,
                    bins=np.linspace(0, 1, 11),
                    include_lowest=True,
                    duplicates="drop",
                ).astype("string").fillna("__MISSING__").astype(str)
            except ValueError:
                all_df[f"RELCAT_{safe_col(col)}_bin10"] = "__MISSING__"

    out = []
    start = 0
    for length in lengths:
        out.append(all_df.iloc[start:start + length].reset_index(drop=True))
        start += length
    return out


def add_sequence_features(frames, use_stint=False):
    """Previous / next row values, differences and rolling means of the previous 3 / 5 rows within
    Year x Race x Driver (x Stint), ordered by lap, over the combined train + test + original rows."""
    lengths = [len(f) for f in frames]
    all_df = pd.concat(frames, ignore_index=True)
    group_keys = [c for c in ["Year", "Race", "Driver"] if c in all_df.columns]
    if use_stint and "Stint" in all_df.columns:
        group_keys.append("Stint")
    if len(group_keys) < 2 or "LapNumber" not in all_df.columns:
        return frames

    all_df["_SEQ_ORDER"] = np.arange(len(all_df))
    all_df = all_df.sort_values(group_keys + ["LapNumber", "_SEQ_ORDER"]).reset_index(drop=True)
    lag_candidates = [
        "LapTime (s)", "LapTime_Delta", "Position", "TyreLife", "PitStop",
        "Position_Change", "Cumulative_Degradation", "RaceProgress",
    ]
    lag_cols = [c for c in lag_candidates if c in all_df.columns]
    for col in lag_cols:
        x = pd.to_numeric(all_df[col], errors="coerce")
        grp = x.groupby([all_df[g] for g in group_keys], sort=False)
        prev1 = grp.shift(1)
        all_df[f"SEQ_{safe_col(col)}_prev1"] = prev1.astype(np.float32)
        all_df[f"SEQ_{safe_col(col)}_diff_prev1"] = (x - prev1).astype(np.float32)
        all_df[f"SEQ_{safe_col(col)}_roll3_mean_prev"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).astype(np.float32)
        all_df[f"SEQ_{safe_col(col)}_roll5_mean_prev"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).astype(np.float32)
        next1 = grp.shift(-1)
        all_df[f"SEQ_{safe_col(col)}_next1"] = next1.astype(np.float32)
        all_df[f"SEQ_{safe_col(col)}_diff_next1"] = (next1 - x).astype(np.float32)

    all_df = all_df.sort_values("_SEQ_ORDER").drop(columns=["_SEQ_ORDER"]).reset_index(drop=True)
    out = []
    start = 0
    for length in lengths:
        out.append(all_df.iloc[start:start + length].reset_index(drop=True))
        start += length
    return out


def add_group_stats(frames, group_cols_list, val_cols):
    for gcols in group_cols_list:
        if not all(all(c in f.columns for c in gcols) for f in frames):
            continue
        gn = "_".join(gcols)
        for vc in val_cols:
            if not all(vc in f.columns for f in frames):
                continue
            base = pd.concat([f[gcols + [vc]] for f in frames], ignore_index=True)
            aggs = ["mean", "std"]
            st = base.groupby(gcols, dropna=False)[vc].agg(aggs).reset_index()
            st.columns = gcols + [f"{vc}_{a}_{gn}" for a in aggs]
            for i, f in enumerate(frames):
                frames[i] = f.merge(st, on=gcols, how="left")
                mean_col = f"{vc}_mean_{gn}"
                if mean_col in frames[i].columns:
                    frames[i][f"{vc}_diff_{gn}"] = (frames[i][vc] - frames[i][mean_col]).astype(np.float32)
            del base, st
    return frames


def add_stint_endpoint_features(frames, group_cols_list):
    """Group min / max / mean / std of tyre life, lap, progress and two pressures, the row's distance to them and the
    remaining tyre life / laps to the group maximum as categories."""
    val_cols = [
        c for c in ["TyreLife", "LapNumber", "RaceProgress", "StintPressure", "PitWindowPressure"]
        if all(c in f.columns for f in frames)
    ]
    if not val_cols:
        return frames

    for gcols in group_cols_list:
        if not all(all(c in f.columns for c in gcols) for f in frames):
            continue
        gn = "_".join(gcols)
        base = pd.concat([f[gcols + val_cols] for f in frames], ignore_index=True)
        agg_map = {}
        for vc in val_cols:
            agg_map[vc] = ["min", "max", "mean", "std"]
        st = base.groupby(gcols, dropna=False).agg(agg_map)
        st.columns = [f"END_{vc}_{agg}_{gn}" for vc, agg in st.columns]
        st = st.reset_index()

        for i, f in enumerate(frames):
            frames[i] = f.merge(st, on=gcols, how="left")
            for vc in val_cols:
                min_col = f"END_{vc}_min_{gn}"
                max_col = f"END_{vc}_max_{gn}"
                mean_col = f"END_{vc}_mean_{gn}"
                if max_col not in frames[i].columns:
                    continue
                spread = (frames[i][max_col] - frames[i][min_col]).replace(0, np.nan)
                frames[i][f"{vc}_to_group_max_{gn}"] = (frames[i][max_col] - frames[i][vc]).astype(np.float32)
                frames[i][f"{vc}_from_group_min_{gn}"] = (frames[i][vc] - frames[i][min_col]).astype(np.float32)
                frames[i][f"{vc}_group_frac_{gn}"] = ((frames[i][vc] - frames[i][min_col]) / spread).fillna(0).astype(np.float32)
                frames[i][f"{vc}_diff_group_mean_{gn}"] = (frames[i][vc] - frames[i][mean_col]).astype(np.float32)

            if "TyreLife" in val_cols:
                rem = frames[i][f"TyreLife_to_group_max_{gn}"].round().clip(-5, 80).fillna(-1).astype("int16")
                frames[i][f"CAT_TyreLife_to_group_max_{gn}"] = rem.astype(str)
            if "LapNumber" in val_cols:
                rem = frames[i][f"LapNumber_to_group_max_{gn}"].round().clip(-5, 120).fillna(-1).astype("int16")
                frames[i][f"CAT_LapNumber_to_group_max_{gn}"] = rem.astype(str)
        del base, st
    return frames


def add_original_priors(frames, orig_frame, y_orig, keys, smooth=20.0):
    """OP_<key>_mean: smoothed original-label mean of the key level; OP_<key>_count: its original row count."""
    prior = float(np.mean(y_orig))
    orig_tmp = orig_frame.copy()
    orig_tmp[TARGET] = y_orig
    for key in keys:
        if key not in orig_tmp.columns:
            continue
        g = orig_tmp.groupby(key, dropna=False)[TARGET].agg(["sum", "count"])
        enc = ((g["sum"] + prior * smooth) / (g["count"] + smooth)).astype(np.float32)
        cnt = g["count"].astype(np.float32)
        for f in frames:
            f[f"OP_{key}_mean"] = f[key].map(enc).fillna(prior).astype(np.float32)
            f[f"OP_{key}_count"] = f[key].map(cnt).fillna(0).astype(np.float32)
    return frames


def reduce_and_fill(train_df, test_df, orig_df):
    """Common sorted column set; string columns filled with their mode, numerics with their median (train + test +
    original)."""
    feature_cols = sorted(set(train_df.columns) | set(test_df.columns) | set(orig_df.columns))
    for f in [train_df, test_df, orig_df]:
        for col in feature_cols:
            if col not in f.columns:
                f[col] = np.nan
    train_df = train_df[feature_cols].copy()
    test_df = test_df[feature_cols].copy()
    orig_df = orig_df[feature_cols].copy()

    cat_cols = [
        c for c in feature_cols
        if train_df[c].dtype == "object"
        or str(train_df[c].dtype).startswith("string")
        or str(train_df[c].dtype).startswith("category")
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for col in cat_cols:
        vals = pd.concat([train_df[col], test_df[col], orig_df[col]], ignore_index=True).astype("string")
        mode = vals.mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "__MISS__"
        for f in [train_df, test_df, orig_df]:
            f[col] = f[col].astype("string").fillna(fill).astype(str)

    for col in num_cols:
        med = pd.concat([train_df[col], test_df[col], orig_df[col]], ignore_index=True)
        med = pd.to_numeric(med, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
        if pd.isna(med):
            med = 0.0
        for f in [train_df, test_df, orig_df]:
            f[col] = pd.to_numeric(f[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med)
            if pd.api.types.is_float_dtype(f[col]):
                f[col] = f[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(f[col]):
                f[col] = pd.to_numeric(f[col], downcast="integer")

    return train_df, test_df, orig_df, cat_cols


def build_frames(cfg):
    """Feature frames of the train, test and original rows: dict(X, X_test, X_orig, y, y_orig, cat_cols)."""
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].astype(int).reset_index(drop=True)
    y_orig = orig[TARGET].astype(int).reset_index(drop=True)
    feature_cols = [c for c in train.columns if c not in [ID, TARGET]]

    tr = train[feature_cols].copy()
    ts = test[feature_cols].copy()
    og = orig[feature_cols].copy()
    tr["IsOrig"] = 0
    ts["IsOrig"] = 0
    og["IsOrig"] = 1

    for df in [tr, ts, og]:
        add_domain_features(df)
        if cfg.get("public_formula_features"):
            add_public_formula_features(df)
        if cfg.get("pipeline_a_features"):
            add_pipeline_a_features(df)
        add_low_card_numeric_cats(df)
        add_artifact_cats(df)
        if cfg.get("compound_hazard_light"):
            add_compound_hazard_light_features(df)
        if cfg.get("pit_state_cats"):
            add_pit_state_cats(df)
        if cfg.get("strategy_cats"):
            add_strategy_cats(df)
        if cfg.get("native_combo_features"):
            add_native_combo_features(df)

    if cfg.get("relative_rank_features") or cfg.get("relative_category_features"):
        tr, ts, og = add_relative_rank_features(
            [tr, ts, og], add_categories=bool(cfg.get("relative_category_features")))
    if cfg.get("sequence_features"):
        tr, ts, og = add_sequence_features([tr, ts, og], use_stint=bool(cfg.get("sequence_stint_features")))

    digit_src = [
        "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
        "Position_Change", "EstTotalLaps", "LapsRemaining", "TyreAgeRatio",
        "DegPerTyreLap", "DeltaAbs", "PosPressure", "StintPressure", "PitWindowPressure",
    ]
    digit_src = [c for c in digit_src if c in tr.columns]
    for df in [tr, ts, og]:
        add_digit_features(df, digit_src)
        add_sig_features(df)

    freq_cols = [
        "Driver", "Race", "Compound", "Race_Year", "Compound_Stint",
        "Driver_Race", "Driver_Compound", "Race_Compound", "Race_Compound_Stint",
        "Compound_RacePhase", "Compound_TyreLifeBin", "RacePhase_TyreLifeBin",
    ]
    freq_cols += [c for c in tr.columns if c.startswith("CAT_LapTime") or c.startswith("CAT_LapTime_Delta") or c.startswith("CAT_Cumulative")]
    if cfg.get("native_combo_features"):
        freq_cols += [
            "CAT2_TyreFracBin",
            "CAT2_LapsToLifeBin",
            "CAT2_RiskAgeBin",
            "CAT2_RaceProgress_q20",
            "CAT2_LapNumber_q20",
            "CAT2_TyreLife_q20",
            "CAT2_LapTime_Delta_r2",
            "CAT2_Cumulative_Degradation_r2",
            "CAT2_RaceProgress_r3",
            "CAT2_CAT2_RiskAgeProgress_r2",
            "CAT2_Race_Year_Compound_Stint",
            "CAT2_Race_Compound_TyreFracBin",
            "CAT2_Compound_Stint_TyreFracBin",
            "CAT2_Compound_LapsToLifeBin",
            "CAT2_Race_Compound_LapsToLifeBin",
            "CAT2_Compound_RacePhase_RiskAgeBin",
        ]
    if cfg.get("compound_hazard_light"):
        freq_cols += [
            "CB2_TyreFracLifeBin",
            "CB2_LapsToLifeBin",
            "CB2_StressBin",
            "CB2_Compound_TyreFracLifeBin",
            "CB2_Compound_LapsToLifeBin",
            "CB2_Compound_RacePhase_StressBin",
            "CB2_Race_Compound_TyreFracLifeBin",
        ]
    if cfg.get("pit_state_cats"):
        freq_cols += [
            "CB2_LapWindowBin",
            "CB2_TyreWindowBin",
            "CB2_ProgressWindowBin",
            "CB2_PositionBand",
            "CB2_DeltaSignBand",
            "CB2_Compound_Stint_TyreWindow",
            "CB2_Race_Compound_ProgressWindow",
            "CB2_Race_Stint_LapWindow",
            "CB2_Compound_Position_TyreWindow",
            "CB2_PitStop_Compound_TyreWindow",
            "CB2_Compound_DeltaSign_TyreWindow",
        ]
    if cfg.get("strategy_cats"):
        freq_cols += [
            "CB3_Driver_Race_Stint_TyreWindow",
            "CB3_Race_Year_Compound_Stint",
            "CB3_Race_Year_PitStop_Progress",
            "CB3_Driver_Compound_TyreWindow",
            "CB3_Compound_Stint_PositionBand",
            "CB3_Race_Position_Progress",
            "CB3_Compound_Stress_Position",
            "CB3_PitStop_Stint_TyreWindow",
        ]
    freq_cols = list(dict.fromkeys([c for c in freq_cols if c in tr.columns]))
    tr, ts, og = add_frequency_features([tr, ts, og], freq_cols)

    if cfg.get("source_freq_features"):
        source_cols = [
            "Driver", "Race", "Compound", "Race_Year", "Race_Compound_Stint",
            "Driver_Race", "Driver_Compound", "Compound_TyreLifeBin",
            "RacePhase_TyreLifeBin", "TyreAgeRatio_str",
        ]
        source_cols += [
            c for c in tr.columns
            if c.startswith("CAT_LapTime")
            or c.startswith("CAT_LapTime_Delta")
            or c.startswith("CAT_Cumulative")
            or c.startswith("CAT_RaceProgress")
            or c.startswith("CAT_TyreAgeRatio")
            or c.startswith("CAT_DegPerTyreLap")
            or c.startswith("CAT_DeltaAbs")
        ]
        source_cols += [
            c for c in [
                "CAT2_TyreFracBin",
                "CAT2_LapsToLifeBin",
                "CAT2_RiskAgeBin",
                "CAT2_LapTime_Delta_r2",
                "CAT2_Cumulative_Degradation_r2",
                "CAT2_RaceProgress_r3",
                "CAT2_CAT2_RiskAgeProgress_r2",
                "CAT2_Race_Compound_TyreFracBin",
                "CAT2_Compound_LapsToLifeBin",
                "CAT2_Race_Compound_LapsToLifeBin",
                "CAT2_LapTime_Delta_r2__Compound_CAT2_TyreFracBin",
                "CAT2_Cumulative_Degradation_r2__Compound_CAT2_TyreFracBin",
            ]
            if c in tr.columns
        ]
        source_cols = list(dict.fromkeys([c for c in source_cols if c in tr.columns]))
        tr, ts, og = add_source_frequency_features([tr, ts, og], source_cols)

    group_cols = [["Race_Year"], ["Race_Compound_Stint"], ["Driver_Race"]]
    if cfg.get("wide_group_stats"):
        group_cols += [["Compound_Stint"], ["Race_Compound"], ["Driver_Compound"], ["Race_Year_Compound"]]
    val_cols = ["LapTime_Delta", "Position_Change", "RaceProgress", "TyreLife"]
    if cfg.get("wide_group_stats"):
        val_cols += ["LapTime (s)", "Cumulative_Degradation", "DeltaAbs", "TyreAgeRatio", "StintPressure"]
    tr, ts, og = add_group_stats(
        [tr, ts, og],
        [g for g in group_cols if all(c in tr.columns for c in g)],
        [c for c in val_cols if c in tr.columns],
    )

    if cfg.get("stint_endpoint_features"):
        endpoint_groups = [
            ["Race_Year"],
            ["Race_Compound_Stint"],
            ["Driver_Race"],
            ["Driver_Race_Stint"],
            ["Driver_Race_Compound"],
            ["Race_Year_Compound"],
            ["Race_Year_Stint"],
            ["Compound_Stint"],
        ]
        tr, ts, og = add_stint_endpoint_features(
            [tr, ts, og],
            [g for g in endpoint_groups if all(c in tr.columns for c in g)],
        )

    X, X_test, X_orig, cat_cols = reduce_and_fill(tr, ts, og)
    return dict(X=X, X_test=X_test, X_orig=X_orig, y=y, y_orig=y_orig, cat_cols=cat_cols)


def fit_fold(cfg, frames, fold_id, k, name=""):
    """Outer fold k (0-4). Returns the clipped probabilities of fold k's rows and of the test rows."""
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    X, Xt, Xo = frames["X"], frames["X_test"], frames["X_orig"]
    y, yo, cat_cols = frames["y"], frames["y_orig"], frames["cat_cols"]
    tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]

    X_tr = X.iloc[tr_idx].copy()
    y_tr = y.iloc[tr_idx].to_numpy()
    X_va = X.iloc[va_idx].copy()
    y_va = y.iloc[va_idx].to_numpy()
    X_te = Xt.copy()
    X_or = Xo.copy()  # every original row, in every fold
    y_or = yo.to_numpy()

    prior_keys = cfg.get("original_priors", [])
    if prior_keys:
        frames_k = add_original_priors([X_tr, X_va, X_te, X_or], X_or, y_or, prior_keys, smooth=float(cfg["prior_smooth"]))
        X_tr, X_va, X_te, X_or = frames_k

    X_fit = pd.concat([X_tr, X_or], axis=0, ignore_index=True)
    y_fit = np.concatenate([y_tr, y_or])
    sample_weight = None
    ow = float(cfg.get("original_weight", 1.0))
    if ow != 1.0:
        sample_weight = np.concatenate([np.ones(len(y_tr), dtype=np.float32), np.full(len(y_or), ow, dtype=np.float32)])

    fold_cat_cols = [c for c in X_fit.columns if c in cat_cols or str(X_fit[c].dtype) == "object"]
    for c in fold_cat_cols:
        X_fit[c] = X_fit[c].astype(str)
        X_va[c] = X_va[c].astype(str)
        X_te[c] = X_te[c].astype(str)

    # seed 42 + fold number (1-5) + the recipe's offset
    params = catboost_params(cfg["params"], SEED + k + 1 + int(cfg.get("seed_offset", 0)))
    print(f"{name} fold {k}: train={X_fit.shape} val={X_va.shape} cat={len(fold_cat_cols)}", flush=True)
    model = CatBoostClassifier(**params)
    model.fit(
        X_fit,
        y_fit,
        eval_set=(X_va, y_va),
        cat_features=fold_cat_cols,
        sample_weight=sample_weight,
        use_best_model=True,
    )
    val_pred = np.clip(model.predict_proba(X_va)[:, 1], 1e-7, 1 - 1e-7).astype(np.float32)
    te_pred = np.clip(model.predict_proba(X_te)[:, 1], 1e-7, 1 - 1e-7).astype(np.float32)
    print(f"{name} fold {k}: AUC {roc_auc_score(y_va, val_pred):.6f} best_iter {model.get_best_iteration()}", flush=True)
    return val_pred, te_pred
