"""Fold-safe target-encoding XGBoost harness used by ../train.py.

Base matrix (built once, label-free, on the competition train rows + the original rows + the test rows):
  the 14 raw columns (string keys for Driver / Compound / Race), 10 categorical combination keys, 24 domain numerics, group
  mean / std / difference / ratio aggregates over 8 groupings, previous-lap lags and rolling means within
  (Year, Race, Driver) sorted by LapNumber, the optional blocks switched on by `Config`, and the count / frequency of every
  string key over all rows. Columns constant on the train + original rows are dropped. The blocks are added in a fixed
  order; the column order is part of a recipe because XGBoost samples columns.

Per outer fold k (common.data.folds, the same StratifiedKFold(5, shuffle=True, random_state=42)):
  - training rows: the competition rows of the 4 other folds + all original rows (sample weight `external_sample_weight`,
    competition rows 1.0); validation rows: fold k;
  - each string key with 2..max_te_cardinality levels (counted on the train + original rows) gets a smoothed target
    encoding (sum + s * prior) / (count + s) for every s in `te_smooths`: out-of-fold over an inner
    StratifiedKFold(5, shuffle=True, random_state=42) of the training rows, and fitted on all training rows for the
    validation and test rows; plus its count and frequency among the training rows;
  - every string key is then ordinal-encoded (fitted on the training rows, unseen levels -> -1);
  - XGBoost (lossguide, 96 leaves, lr 0.025, max_bin 384, ...) with early stopping on the AUC of fold k,
    random_state 42 + k + 1. Test = mean of the 5 fold models; predictions clipped to [1e-7, 1 - 1e-7]."""
import gc
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from common.data import ID, TARGET, folds, read_original, read_test, read_train
from common.pairs import assemble, have_part, save_part

EPS = 1e-6
TE_INNER_SPLITS = 5
COMPETITION_SAMPLE_WEIGHT = 1.0


@dataclass
class Config:
    pair: str
    seed: int = 42
    external_sample_weight: float = 0.80
    te_smooths: List[int] = field(default_factory=lambda: [5, 20, 100])
    max_te_cardinality: int = 50000
    use_hazard_features: bool = False
    use_relative_rank_features: bool = False
    use_sequence_next_features: bool = False
    use_public_formula_features: bool = False
    use_pipeline_a_features: bool = False
    use_source_indicator: bool = False
    use_catboost_style_features: bool = False
    use_extended_group_residuals: bool = False
    use_pit_window_key_features: bool = False
    use_pit_cadence_features: bool = False
    use_artifact_hash_keys: bool = False
    xgb_params_override: Dict = field(default_factory=dict)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_colname(col: str) -> str:
    return str(col).strip().replace(" ", "_").replace("/", "_").replace("-", "_")


def try_numeric_cast(df: pd.DataFrame, exclude: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude or not (out[c].dtype == "object" or str(out[c].dtype).startswith("string")):
            continue
        parsed = pd.to_numeric(out[c], errors="coerce")
        non_missing = out[c].notna().sum()
        if non_missing > 0 and parsed.notna().sum() / non_missing > 0.95:
            out[c] = parsed
    return out


def string_key(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("__MISSING__")


def combo_key(df: pd.DataFrame, cols: Sequence[str], sep: str = "__") -> pd.Series:
    s = string_key(df[cols[0]])
    for c in cols[1:]:
        s = s + sep + string_key(df[c])
    return s


# ----------------------------------------------------------------------------------------------------------------------
# Base blocks (always on)
# ----------------------------------------------------------------------------------------------------------------------
def add_categorical_combinations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    candidate_pairs = [
        ("Year", "Race"), ("Year", "Driver"), ("Race", "Driver"),
        ("Race", "Compound"), ("Race", "Stint"), ("Driver", "Compound"),
        ("Driver", "Team"), ("Team", "Compound"), ("Circuit", "Compound"),
        ("Circuit", "Driver"), ("Stint", "Compound"),
    ]
    for a, b in candidate_pairs:
        if a in out.columns and b in out.columns:
            out[f"{a}__x__{b}"] = string_key(out[a]) + "__" + string_key(out[b])
    candidate_triples = [
        ("Year", "Race", "Driver"),
        ("Race", "Driver", "Compound"),
        ("Race", "Stint", "Compound"),
    ]
    for a, b, c in candidate_triples:
        if a in out.columns and b in out.columns and c in out.columns:
            out[f"{a}__x__{b}__x__{c}"] = string_key(out[a]) + "__" + string_key(out[b]) + "__" + string_key(out[c])
    return out


def add_domain_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if "LapNumber" in out.columns:
        out["LapNumber_log1p"] = np.log1p(out["LapNumber"].clip(lower=0))
        out["LapNumber_sq"] = out["LapNumber"] ** 2
    if "RaceProgress" in out.columns:
        out["RaceProgress_sq"] = out["RaceProgress"] ** 2
        out["RaceProgress_cu"] = out["RaceProgress"] ** 3
        out["RaceRemaining"] = 1.0 - out["RaceProgress"]
        clipped = out["RaceProgress"].clip(EPS, 1 - EPS)
        out["RaceProgress_logit"] = np.log(clipped / (1 - clipped))
    if "LapNumber" in out.columns and "RaceProgress" in out.columns:
        out["EstimatedTotalLaps"] = out["LapNumber"] / out["RaceProgress"].replace(0, np.nan)
        out["EstimatedLapsRemaining"] = out["EstimatedTotalLaps"] - out["LapNumber"]
        out["Lap_x_RaceProgress"] = out["LapNumber"] * out["RaceProgress"]
    if "TyreLife" in out.columns:
        out["TyreLife_log1p"] = np.log1p(out["TyreLife"].clip(lower=0))
        out["TyreLife_sq"] = out["TyreLife"] ** 2
        if "LapNumber" in out.columns:
            out["TyreLife_over_Lap"] = out["TyreLife"] / (out["LapNumber"].abs() + EPS)
            out["Lap_minus_TyreLife"] = out["LapNumber"] - out["TyreLife"]
        if "RaceProgress" in out.columns:
            out["TyreLife_x_RaceProgress"] = out["TyreLife"] * out["RaceProgress"]
            out["TyreLife_over_RaceRemaining"] = out["TyreLife"] / (out.get("RaceRemaining", 1).abs() + EPS)
    if "Position" in out.columns:
        out["Position_inv"] = 1.0 / (out["Position"].abs() + EPS)
        out["Position_log1p"] = np.log1p(out["Position"].clip(lower=0))
        if "RaceProgress" in out.columns:
            out["Position_x_RaceProgress"] = out["Position"] * out["RaceProgress"]
        if "LapNumber" in out.columns:
            out["Position_x_Lap"] = out["Position"] * out["LapNumber"]
    if "Position_Change" in out.columns:
        out["Position_Change_abs"] = out["Position_Change"].abs()
        out["Position_Change_is_gain"] = (out["Position_Change"] < 0).astype(int)
        out["Position_Change_is_loss"] = (out["Position_Change"] > 0).astype(int)
    if "Stint" in out.columns and "LapNumber" in out.columns:
        out["Stint_x_LapNumber"] = out["Stint"] * out["LapNumber"]
    if "Stint" in out.columns and "TyreLife" in out.columns:
        out["Stint_x_TyreLife"] = out["Stint"] * out["TyreLife"]
    if "LapTime" in out.columns and "LapTime_Delta" in out.columns:
        out["LapTime_plus_delta"] = out["LapTime"] + out["LapTime_Delta"]
        out["LapTime_delta_ratio"] = out["LapTime_Delta"] / (out["LapTime"].abs() + EPS)
    for c in num_cols:
        if out[c].isna().any():
            out[f"{c}_is_missing"] = out[c].isna().astype(int)
    return out


def add_group_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_candidates = [
        "LapNumber", "TyreLife", "LapTime", "LapTime_Delta", "Position",
        "RaceProgress", "Stint", "Position_Change", "EstimatedTotalLaps",
        "EstimatedLapsRemaining",
    ]
    agg_cols = [c for c in numeric_candidates if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    group_sets = [
        ["Race"], ["Driver"], ["Compound"], ["Team"], ["Circuit"],
        ["Year", "Race"], ["Race", "Driver"], ["Race", "Compound"],
        ["Driver", "Compound"], ["Team", "Compound"], ["Stint", "Compound"],
    ]
    group_sets = [g for g in group_sets if all(c in out.columns for c in g)]
    for group in group_sets:
        gname = "__".join(group)
        gb = out.groupby(group, dropna=False, sort=False)
        for c in agg_cols[:10]:
            mean_col = f"{gname}__{c}__mean"
            std_col = f"{gname}__{c}__std"
            out[mean_col] = gb[c].transform("mean")
            out[std_col] = gb[c].transform("std")
            out[f"{c}__minus__{mean_col}"] = out[c] - out[mean_col]
            out[f"{c}__ratio__{mean_col}"] = out[c] / (out[mean_col].abs() + EPS)
    return out


def add_sequence_features(df: pd.DataFrame, include_next: bool = False) -> pd.DataFrame:
    """Lags within (Year, Race, Driver) over all rows (train, original and test), ordered by LapNumber."""
    out = df.copy()
    group_keys = [c for c in ["Year", "Race", "Driver"] if c in out.columns]
    if len(group_keys) < 2 or "LapNumber" not in out.columns:
        return out
    out["_orig_order_for_seq"] = np.arange(len(out))
    out = out.sort_values(group_keys + ["LapNumber", "_orig_order_for_seq"]).reset_index(drop=True)
    lag_candidates = ["LapTime", "LapTime_Delta", "Position", "TyreLife", "PitStop", "Position_Change", "Cumulative_Degradation", "RaceProgress"]
    lag_cols = [c for c in lag_candidates if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    for c in lag_cols:
        grp = out.groupby(group_keys, sort=False)[c]
        prev1 = grp.shift(1)
        out[f"{c}_prev1"] = prev1
        out[f"{c}_diff_prev1"] = out[c] - prev1
        out[f"{c}_roll3_mean_prev"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        out[f"{c}_roll5_mean_prev"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        if include_next:
            next1 = grp.shift(-1)
            out[f"{c}_next1"] = next1
            out[f"{c}_diff_next1"] = next1 - out[c]
    out = out.sort_values("_orig_order_for_seq").drop(columns=["_orig_order_for_seq"]).reset_index(drop=True)
    return out


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for c in cat_cols:
        key = string_key(out[c])
        vc = key.value_counts(dropna=False)
        out[f"{c}_count_all"] = key.map(vc).astype(float)
        out[f"{c}_freq_all"] = out[f"{c}_count_all"] / len(out)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Optional blocks
# ----------------------------------------------------------------------------------------------------------------------
def add_public_formula_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio / interaction formulas of public notebooks (PUB_*), race-phase and tyre-age flags, compound life ratios."""
    out = df.copy()
    eps = EPS
    lap_number = pd.to_numeric(out.get("LapNumber", pd.Series(0, index=out.index)), errors="coerce")
    tyre = pd.to_numeric(out.get("TyreLife", pd.Series(0, index=out.index)), errors="coerce")
    progress = pd.to_numeric(out.get("RaceProgress", pd.Series(0, index=out.index)), errors="coerce")
    lap_time = pd.to_numeric(out.get("LapTime", out.get("LapTime_(s)", pd.Series(0, index=out.index))), errors="coerce")
    delta = pd.to_numeric(out.get("LapTime_Delta", pd.Series(0, index=out.index)), errors="coerce")
    deg = pd.to_numeric(out.get("Cumulative_Degradation", pd.Series(0, index=out.index)), errors="coerce")
    position = pd.to_numeric(out.get("Position", pd.Series(0, index=out.index)), errors="coerce")
    pos_change = pd.to_numeric(out.get("Position_Change", pd.Series(0, index=out.index)), errors="coerce")
    remaining = 1.0 - progress

    out["PUB_Lap_TyreLife_Ratio"] = lap_number / (tyre + 1.0)
    out["PUB_TyreLife_per_Progress"] = tyre / (progress + 0.05)
    out["PUB_LapTime_per_TyreLife"] = lap_time / (tyre + 1.0)
    out["PUB_Degradation_per_TyreLife"] = deg / (tyre + 1.0)
    out["PUB_LapDelta_per_TyreLife"] = delta / (tyre + 1.0)
    out["PUB_Degradation_per_Lap"] = deg / (lap_number + 1.0)
    out["PUB_LapTime_x_RaceProgress"] = lap_time * progress
    out["PUB_LapDelta_x_TyreLife"] = delta * tyre
    out["PUB_Position_x_TyreLife"] = position * tyre
    out["PUB_RemainingRace_x_TyreLife"] = remaining * tyre
    out["PUB_Deg_Rate"] = deg / tyre.clip(lower=1)
    out["PUB_Cum_Stress"] = tyre * deg.abs().clip(upper=200)
    out["PUB_LTD_Abs"] = delta.abs()
    out["PUB_Pos_Mom"] = pos_change.clip(-5, 5)
    out["PUB_NoPositionChange"] = (pos_change.abs() < eps).astype(int)
    out["PUB_IsEarlyRace"] = (progress <= 0.25).astype(int)
    out["PUB_IsMidRace"] = ((progress > 0.25) & (progress < 0.75)).astype(int)
    out["PUB_IsLateRace"] = (progress >= 0.75).astype(int)
    tyre_q25, tyre_q50, tyre_q75 = tyre.quantile([0.25, 0.50, 0.75]).values
    out["PUB_IsFreshTyre"] = (tyre <= tyre_q25).astype(int)
    out["PUB_IsOldTyre"] = (tyre >= tyre_q50).astype(int)
    out["PUB_IsVeryOldTyre"] = (tyre >= tyre_q75).astype(int)
    if "Compound" in out.columns:
        comp = string_key(out["Compound"]).str.upper()
        compound_life = comp.map({"SOFT": 18.0, "MEDIUM": 25.0, "HARD": 32.0, "INTERMEDIATE": 18.0, "WET": 15.0}).fillna(25.0)
        out["PUB_TL_Ratio_CompoundLife"] = tyre / compound_life
        for name in ["SOFT", "MEDIUM", "HARD"]:
            mask = comp.str.contains(name, regex=False).astype(int)
            out[f"PUB_{name}_x_TyreLife"] = mask * tyre
            out[f"PUB_{name}_x_Progress"] = mask * progress
    return out


def add_hazard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tyre-life / race-progress / lap bins and their keys with race, compound, stint and pit stop (HZ__*)."""
    out = df.copy()
    if "TyreLife" in out.columns:
        out["TyreLife_bin_hazard"] = pd.cut(pd.to_numeric(out["TyreLife"], errors="coerce"), bins=[-1, 0, 2, 4, 7, 10, 15, 20, 30, 1000], duplicates="drop").astype("string")
    if "RaceProgress" in out.columns:
        out["RaceProgress_bin_hazard"] = pd.cut(pd.to_numeric(out["RaceProgress"], errors="coerce"), bins=np.linspace(0, 1, 11), include_lowest=True, duplicates="drop").astype("string")
    if "LapNumber" in out.columns:
        out["LapNumber_bin_hazard"] = pd.cut(pd.to_numeric(out["LapNumber"], errors="coerce"), bins=[0, 5, 10, 15, 20, 25, 30, 35, 45, 55, 70, 200], include_lowest=True, duplicates="drop").astype("string")
    combos = [
        ("Year", "Race", "Compound", "Stint", "TyreLife_bin_hazard"),
        ("Race", "Compound", "Stint", "RaceProgress_bin_hazard"),
        ("Race", "Stint", "LapNumber_bin_hazard"),
        ("Compound", "TyreLife_bin_hazard", "PitStop"),
        ("Stint", "TyreLife_bin_hazard", "RaceProgress_bin_hazard"),
        ("Race", "Stint", "Compound", "TyreLife_bin_hazard"),
    ]
    for combo in combos:
        if all(c in out.columns for c in combo):
            out["HZ__" + "__x__".join(combo)] = combo_key(out, combo)
    return out


def add_relative_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rank, distance to the median, ratio to the median and robust z of 6 numerics within 4 race / stint / compound groups."""
    out = df.copy()
    num_cols = [c for c in ["TyreLife", "LapNumber", "RaceProgress", "LapTime_Delta", "Cumulative_Degradation", "LapTime_(s)"] if c in out.columns]
    group_sets = [
        ["Year", "Race", "Stint", "Compound"],
        ["Race", "Stint", "Compound"],
        ["Race", "Compound"],
        ["Compound", "Stint"],
    ]
    for group in group_sets:
        if not all(c in out.columns for c in group):
            continue
        gname = "__".join(group)
        gb = out.groupby(group, dropna=False, sort=False)
        for c in num_cols:
            x = pd.to_numeric(out[c], errors="coerce")
            med = gb[c].transform("median")
            q25 = gb[c].transform(lambda s: s.quantile(0.25))
            q75 = gb[c].transform(lambda s: s.quantile(0.75))
            out[f"REL_{gname}__{c}__rank_pct"] = gb[c].rank(pct=True, method="average")
            out[f"REL_{gname}__{c}__minus_median"] = x - med
            out[f"REL_{gname}__{c}__ratio_median"] = x / (med.abs() + EPS)
            out[f"REL_{gname}__{c}__robust_z"] = (x - med) / ((q75 - q25).abs() + EPS)
    return out


def add_pipeline_a_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline A block (PIPA_*): two ratios, floor categories of every numeric, counts of the categoricals, 200 quantile
    bins of RaceProgress, Race x Compound and Race x Year keys."""
    out = df.copy()
    raw_num_cols = [
        "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime_(s)", "LapTime", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
    ]
    if "LapNumber" in out.columns and "RaceProgress" in out.columns:
        out["PIPA_LP_RP"] = (pd.to_numeric(out["LapNumber"], errors="coerce") / (pd.to_numeric(out["RaceProgress"], errors="coerce") + EPS)).astype("float32")
    if "TyreLife" in out.columns and "LapNumber" in out.columns:
        out["PIPA_TL_LP"] = (pd.to_numeric(out["TyreLife"], errors="coerce") / pd.to_numeric(out["LapNumber"], errors="coerce").clip(lower=1)).astype("float32")
    for c in raw_num_cols + ["PIPA_LP_RP", "PIPA_TL_LP"]:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        out[f"PIPA_{normalize_colname(c)}_floor_cat"] = np.floor(x).astype("Int64").astype("string").fillna("__MISSING__")
    for c in ["Driver", "Compound", "Race", "PIPA_Year_floor_cat", "PIPA_PitStop_floor_cat"]:
        if c in out.columns:
            key = string_key(out[c])
            vc = key.value_counts(dropna=False)
            out[f"PIPA_{normalize_colname(c)}_count"] = key.map(vc).fillna(0).astype("float32")
    if "RaceProgress" in out.columns:
        x = pd.to_numeric(out["RaceProgress"], errors="coerce")
        try:
            out["PIPA_RaceProgress_200_qbin"] = pd.qcut(x.rank(method="first"), q=200, labels=False, duplicates="drop").astype("Int64").astype("string").fillna("__MISSING__")
        except ValueError:
            out["PIPA_RaceProgress_200_qbin"] = "0"
    for combo in [("Race", "Compound"), ("Race", "Year")]:
        if all(c in out.columns for c in combo):
            out["PIPA_" + "_".join(combo)] = combo_key(out, combo, sep="_")
    return out


def first_existing(df: pd.DataFrame, cols: Sequence[str]):
    for c in cols:
        if c in df.columns:
            return c
    return None


def add_catboost_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """CatBoost-style block (X2_*): pit-window ratios and stress terms, phase / tyre / position bins, integer categories,
    rounded / floor / digit categories of the timing columns and 12 category combinations."""
    out = df.copy()
    eps = EPS
    lap_col = first_existing(out, ["LapNumber"])
    tyre_col = first_existing(out, ["TyreLife"])
    progress_col = first_existing(out, ["RaceProgress"])
    lap_time_col = first_existing(out, ["LapTime_(s)", "LapTime", "LapTime (s)"])
    delta_col = first_existing(out, ["LapTime_Delta"])
    deg_col = first_existing(out, ["Cumulative_Degradation"])
    pos_col = first_existing(out, ["Position"])
    pos_chg_col = first_existing(out, ["Position_Change"])
    stint_col = first_existing(out, ["Stint"])

    lap = pd.to_numeric(out[lap_col], errors="coerce") if lap_col else pd.Series(0, index=out.index)
    tyre = pd.to_numeric(out[tyre_col], errors="coerce") if tyre_col else pd.Series(0, index=out.index)
    progress = pd.to_numeric(out[progress_col], errors="coerce") if progress_col else pd.Series(0, index=out.index)
    lap_time = pd.to_numeric(out[lap_time_col], errors="coerce") if lap_time_col else pd.Series(0, index=out.index)
    delta = pd.to_numeric(out[delta_col], errors="coerce") if delta_col else pd.Series(0, index=out.index)
    deg = pd.to_numeric(out[deg_col], errors="coerce") if deg_col else pd.Series(0, index=out.index)
    pos = pd.to_numeric(out[pos_col], errors="coerce") if pos_col else pd.Series(0, index=out.index)
    pos_chg = pd.to_numeric(out[pos_chg_col], errors="coerce") if pos_chg_col else pd.Series(0, index=out.index)
    stint = pd.to_numeric(out[stint_col], errors="coerce") if stint_col else pd.Series(0, index=out.index)

    out["X2_EstTotalLaps"] = (lap / progress.clip(lower=eps)).clip(1, 120).astype("float32")
    out["X2_LapsRemaining"] = (out["X2_EstTotalLaps"] - lap).clip(lower=0).astype("float32")
    out["X2_TyreAgeRatio"] = (tyre / lap.clip(lower=1)).astype("float32")
    out["X2_LapPerTyreLife"] = (lap / (tyre + 1)).astype("float32")
    out["X2_LapMinusTyre"] = (lap - tyre).astype("float32")
    out["X2_PitWindowPressure"] = (tyre * progress).astype("float32")
    out["X2_TyreToLapsRem"] = (tyre / (out["X2_LapsRemaining"] + 1)).astype("float32")
    out["X2_DegPerTyreLap"] = (deg / tyre.clip(lower=1)).astype("float32")
    out["X2_AbsDegPerTyreLap"] = (deg.abs() / tyre.clip(lower=1)).astype("float32")
    out["X2_DeltaPerTyreLap"] = (delta / tyre.clip(lower=1)).astype("float32")
    out["X2_AbsDeltaPerTyreLap"] = (delta.abs() / tyre.clip(lower=1)).astype("float32")
    out["X2_DeltaAbs"] = delta.abs().astype("float32")
    out["X2_AbsCumDeg"] = deg.abs().astype("float32")
    out["X2_AbsPosChange"] = pos_chg.abs().astype("float32")
    out["X2_PosPressure"] = (pos * progress).astype("float32")
    out["X2_StintPressure"] = (stint * tyre).astype("float32")
    out["X2_LapTimeResidualFromDelta"] = (lap_time - delta).astype("float32")
    out["X2_DegDeltaStress"] = (deg.abs() * delta.abs()).clip(0, 1000).astype("float32")

    out["X2_RacePhase"] = pd.cut(
        progress,
        bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=["P1", "P2", "P3", "P4", "P5"],
    ).astype("string").fillna("__MISSING__")
    out["X2_TyreLifeBin"] = pd.cut(
        tyre,
        bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
        labels=["T0", "T1", "T2", "T3", "T4", "T5"],
    ).astype("string").fillna("__MISSING__")
    out["X2_PosBin"] = pd.cut(
        pos,
        bins=[-np.inf, 3, 8, 14, np.inf],
        labels=["front", "upper", "lower", "back"],
    ).astype("string").fillna("__MISSING__")

    for c in ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "Position_Change"]:
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce").round(0)
            out[f"X2_CAT_{c}"] = x.astype("Int64").astype("string").fillna("__MISSING__")

    for col in [lap_time_col, delta_col, deg_col, progress_col, "X2_TyreAgeRatio", "X2_DegPerTyreLap", "X2_DeltaAbs"]:
        if col is None or col not in out.columns:
            continue
        x = pd.to_numeric(out[col], errors="coerce")
        safe = normalize_colname(col)
        rounds = [0, 1, 2, 3] if col in {delta_col, deg_col, "X2_TyreAgeRatio"} else [0, 1, 2]
        for nd in rounds:
            out[f"X2_CAT_{safe}_r{nd}"] = x.round(nd).astype("string").fillna("__MISSING__")
        out[f"X2_CAT_{safe}_floor"] = np.floor(x).astype("Int64").astype("string").fillna("__MISSING__")
        abs_x = x.abs().fillna(0)
        scaled = (abs_x * 100).round().astype("int64")
        for i in range(4):
            out[f"X2_SIG_{safe}_{i + 1}"] = ((scaled // (10 ** i)) % 10).astype("int8").astype("string")

    combo_specs = [
        ("X2_Race_Year", ["Race", "Year"]),
        ("X2_Compound_Stint", ["Compound", "Stint"]),
        ("X2_Driver_Race", ["Driver", "Race"]),
        ("X2_Driver_Compound", ["Driver", "Compound"]),
        ("X2_Race_Compound", ["Race", "Compound"]),
        ("X2_Race_Compound_Stint", ["Race", "Compound", "Stint"]),
        ("X2_Compound_RacePhase", ["Compound", "X2_RacePhase"]),
        ("X2_Compound_TyreLifeBin", ["Compound", "X2_TyreLifeBin"]),
        ("X2_RacePhase_TyreLifeBin", ["X2_RacePhase", "X2_TyreLifeBin"]),
        ("X2_Driver_Race_Compound", ["Driver", "Race", "Compound"]),
        ("X2_Race_Year_Stint", ["Race", "Year", "Stint"]),
        ("X2_Race_Year_TyreLifeBin", ["Race", "Year", "X2_TyreLifeBin"]),
    ]
    for name, cols in combo_specs:
        if all(c in out.columns for c in cols):
            out[name] = combo_key(out, cols)

    if {"LapNumber", "TyreLife"}.issubset(out.columns):
        out["X2_CAT_LapNumber_TyreLife"] = string_key(out["LapNumber"]) + "__" + string_key(out["TyreLife"])
    if {"TyreLife", "Position_Change"}.issubset(out.columns):
        out["X2_CAT_TyreLife_PositionChange"] = string_key(out["TyreLife"]) + "__" + string_key(out["Position_Change"])
    out["X2_TyreAgeRatio_str"] = out["X2_TyreAgeRatio"].round(3).astype("string").fillna("__MISSING__")
    out["X2_EstTotalLaps_str"] = out["X2_EstTotalLaps"].round(1).astype("string").fillna("__MISSING__")
    return out


def add_extended_group_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Group residual block (X2_GRP_*): for 13 groupings, the log group size and, for up to 11 numerics, the difference to
    the group mean, the z-score within the group and the percentile rank within the group."""
    out = df.copy()
    group_specs = [
        ("Race", ["Race"]),
        ("Compound", ["Compound"]),
        ("Driver", ["Driver"]),
        ("Race_Year", ["Race", "Year"]),
        ("Driver_Race", ["Driver", "Race"]),
        ("Driver_Compound", ["Driver", "Compound"]),
        ("Year", ["Year"]),
        ("Year_Compound", ["Year", "Compound"]),
        ("Year_Stint", ["Year", "Stint"]),
        ("Race_Compound", ["Race", "Compound"]),
        ("Race_Compound_Stint", ["Race", "Compound", "Stint"]),
        ("Compound_Stint", ["Compound", "Stint"]),
        ("Race_Year_Stint", ["Race", "Year", "Stint"]),
    ]
    value_cols = [
        c for c in [
            "LapTime_(s)", "LapTime_Delta", "Cumulative_Degradation", "TyreLife",
            "RaceProgress", "Position", "Position_Change", "X2_TyreAgeRatio",
            "X2_StintPressure", "X2_EstTotalLaps", "X2_DeltaAbs",
        ]
        if c in out.columns
    ]
    for gname, gcols in group_specs:
        if not all(c in out.columns for c in gcols):
            continue
        key = combo_key(out, gcols)
        cnt = key.map(key.value_counts(dropna=False)).astype("float32")
        out[f"X2_GRP_{gname}_logcnt"] = np.log1p(cnt).astype("float32")
        for v in value_cols:
            x = pd.to_numeric(out[v], errors="coerce")
            mean = x.groupby(key, sort=False).transform("mean")
            std = x.groupby(key, sort=False).transform("std").fillna(0)
            rank = x.groupby(key, sort=False).rank(pct=True, method="average")
            safe_v = normalize_colname(v)
            out[f"X2_GRP_{safe_v}_minus_{gname}_mean"] = (x - mean).astype("float32")
            out[f"X2_GRP_{safe_v}_z_{gname}"] = ((x - mean) / (std.abs() + EPS)).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
            out[f"X2_GRP_{safe_v}_rankpct_{gname}"] = rank.fillna(0.5).astype("float32")
    return out


def add_pit_window_key_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pit-window keys (X2_PIT_*): integer / sign / bin categories of lap, tyre life, position, progress and their
    combinations with race, compound, stint and driver."""
    out = df.copy()
    lap = pd.to_numeric(out.get("LapNumber", pd.Series(0, index=out.index)), errors="coerce")
    tyre = pd.to_numeric(out.get("TyreLife", pd.Series(0, index=out.index)), errors="coerce")
    progress = pd.to_numeric(out.get("RaceProgress", pd.Series(0, index=out.index)), errors="coerce")
    delta = pd.to_numeric(out.get("LapTime_Delta", pd.Series(0, index=out.index)), errors="coerce")
    deg = pd.to_numeric(out.get("Cumulative_Degradation", pd.Series(0, index=out.index)), errors="coerce")
    pos = pd.to_numeric(out.get("Position", pd.Series(0, index=out.index)), errors="coerce")
    pos_chg = pd.to_numeric(out.get("Position_Change", pd.Series(0, index=out.index)), errors="coerce")

    out["X2_PIT_LapInt"] = lap.round(0).astype("Int64").astype("string").fillna("__MISSING__")
    out["X2_PIT_TyreInt"] = tyre.round(0).astype("Int64").astype("string").fillna("__MISSING__")
    out["X2_PIT_PosInt"] = pos.round(0).astype("Int64").astype("string").fillna("__MISSING__")
    out["X2_PIT_PosChgInt"] = pos_chg.round(0).astype("Int64").astype("string").fillna("__MISSING__")
    out["X2_PIT_DeltaSign"] = np.sign(delta).astype("Int64").astype("string").fillna("__MISSING__")
    out["X2_PIT_DegSign"] = np.sign(deg).astype("Int64").astype("string").fillna("__MISSING__")
    out["X2_PIT_LapBin"] = pd.cut(
        lap,
        bins=[-np.inf, 3, 6, 9, 12, 16, 20, 25, 32, 42, 60, np.inf],
        labels=[f"L{i}" for i in range(11)],
    ).astype("string").fillna("__MISSING__")
    out["X2_PIT_TyreBin"] = pd.cut(
        tyre,
        bins=[-np.inf, 1, 3, 5, 8, 12, 16, 22, 30, np.inf],
        labels=[f"T{i}" for i in range(9)],
    ).astype("string").fillna("__MISSING__")
    out["X2_PIT_ProgBin"] = pd.cut(
        progress,
        bins=np.linspace(0, 1, 13),
        include_lowest=True,
        labels=[f"P{i}" for i in range(12)],
    ).astype("string").fillna("__MISSING__")
    tyre_ratio = (tyre / lap.clip(lower=1)).clip(0, 3)
    out["X2_PIT_TyreLapRatioBin"] = pd.cut(
        tyre_ratio,
        bins=[-np.inf, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.15, np.inf],
        labels=[f"R{i}" for i in range(8)],
    ).astype("string").fillna("__MISSING__")

    combo_specs = [
        ("X2_PIT_Race_Compound_Stint_Tyre", ["Race", "Compound", "Stint", "X2_PIT_TyreInt"]),
        ("X2_PIT_Race_Compound_Lap_Tyre", ["Race", "Compound", "X2_PIT_LapInt", "X2_PIT_TyreInt"]),
        ("X2_PIT_Year_Race_Compound_Tyre", ["Year", "Race", "Compound", "X2_PIT_TyreInt"]),
        ("X2_PIT_Race_Stint_Lap_Tyre", ["Race", "Stint", "X2_PIT_LapInt", "X2_PIT_TyreInt"]),
        ("X2_PIT_Driver_Race_Lap_Tyre", ["Driver", "Race", "X2_PIT_LapInt", "X2_PIT_TyreInt"]),
        ("X2_PIT_Compound_Stint_LapBin_TyreBin", ["Compound", "Stint", "X2_PIT_LapBin", "X2_PIT_TyreBin"]),
        ("X2_PIT_Race_Compound_Phase_TyreBin", ["Race", "Compound", "X2_PIT_ProgBin", "X2_PIT_TyreBin"]),
        ("X2_PIT_Race_Compound_TyreRatio", ["Race", "Compound", "X2_PIT_TyreLapRatioBin"]),
        ("X2_PIT_Year_Compound_TyreRatio", ["Year", "Compound", "X2_PIT_TyreLapRatioBin"]),
        ("X2_PIT_Race_Compound_Tyre_DeltaSign", ["Race", "Compound", "X2_PIT_TyreInt", "X2_PIT_DeltaSign"]),
        ("X2_PIT_Race_Compound_Tyre_PosChg", ["Race", "Compound", "X2_PIT_TyreInt", "X2_PIT_PosChgInt"]),
        ("X2_PIT_Race_Lap_Tyre_Pos", ["Race", "X2_PIT_LapInt", "X2_PIT_TyreInt", "X2_PIT_PosInt"]),
    ]
    for name, cols in combo_specs:
        if all(c in out.columns for c in cols):
            out[name] = combo_key(out, cols)
    return out


def add_pit_cadence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pit cadence (X2_CAD_*): last pit lap, average stint length so far, expected next pit lap and the distance to it,
    pit rates, stress terms, their bins and keys with race / compound / stint."""
    out = df.copy()
    lap = pd.to_numeric(out.get("LapNumber", pd.Series(0, index=out.index)), errors="coerce")
    tyre = pd.to_numeric(out.get("TyreLife", pd.Series(0, index=out.index)), errors="coerce")
    progress = pd.to_numeric(out.get("RaceProgress", pd.Series(0, index=out.index)), errors="coerce")
    pit = pd.to_numeric(out.get("PitStop", pd.Series(0, index=out.index)), errors="coerce")
    stint = pd.to_numeric(out.get("Stint", pd.Series(0, index=out.index)), errors="coerce")
    deg = pd.to_numeric(out.get("Cumulative_Degradation", pd.Series(0, index=out.index)), errors="coerce")
    delta = pd.to_numeric(out.get("LapTime_Delta", pd.Series(0, index=out.index)), errors="coerce")
    pos = pd.to_numeric(out.get("Position", pd.Series(0, index=out.index)), errors="coerce")

    total_laps = (lap / progress.clip(lower=EPS)).clip(1, 120)
    laps_remaining = (total_laps - lap).clip(lower=0)
    last_pit_lap = (lap - tyre).clip(lower=0)
    stint_count = (pit + 1).clip(lower=1)
    avg_stint = lap / stint_count
    expected_next = last_pit_lap + avg_stint

    out["X2_CAD_LastPitLap"] = last_pit_lap.astype("float32")
    out["X2_CAD_AvgStintSoFar"] = avg_stint.astype("float32")
    out["X2_CAD_TyreMinusAvgStint"] = (tyre - avg_stint).astype("float32")
    out["X2_CAD_TyreOverAvgStint"] = (tyre / (avg_stint.abs() + 1.0)).astype("float32")
    out["X2_CAD_PitRateByLap"] = (pit / (lap + 1.0)).astype("float32")
    out["X2_CAD_PitRateByProgress"] = (pit / (progress + 0.05)).astype("float32")
    out["X2_CAD_ExpectedNextPitLap"] = expected_next.astype("float32")
    out["X2_CAD_LapsPastExpectedNext"] = (lap - expected_next).astype("float32")
    out["X2_CAD_LapsToExpectedNext"] = (expected_next - lap).astype("float32")
    out["X2_CAD_TyreOverRemaining"] = (tyre / (laps_remaining + 1.0)).astype("float32")
    out["X2_CAD_StintPitMismatch"] = (stint - pit - 1).astype("float32")
    out["X2_CAD_DegCadenceStress"] = (deg.abs() * out["X2_CAD_TyreOverAvgStint"]).clip(0, 1000).astype("float32")
    out["X2_CAD_DeltaCadenceStress"] = (delta.abs() * out["X2_CAD_TyreOverAvgStint"]).clip(0, 1000).astype("float32")
    out["X2_CAD_PositionPitPressure"] = (pos * out["X2_CAD_PitRateByProgress"]).astype("float32")

    out["X2_CAD_LastPitLapBin"] = pd.cut(
        last_pit_lap,
        bins=[-np.inf, 0, 3, 6, 10, 15, 22, 32, np.inf],
        labels=[f"LP{i}" for i in range(8)],
    ).astype("string").fillna("__MISSING__")
    out["X2_CAD_AvgStintBin"] = pd.cut(
        avg_stint,
        bins=[-np.inf, 3, 6, 10, 14, 18, 24, 32, np.inf],
        labels=[f"AS{i}" for i in range(8)],
    ).astype("string").fillna("__MISSING__")
    out["X2_CAD_PastExpectedBin"] = pd.cut(
        lap - expected_next,
        bins=[-np.inf, -12, -6, -3, 0, 3, 6, 12, np.inf],
        labels=[f"PE{i}" for i in range(8)],
    ).astype("string").fillna("__MISSING__")
    out["X2_CAD_TyreOverAvgBin"] = pd.cut(
        tyre / (avg_stint.abs() + 1.0),
        bins=[-np.inf, 0.25, 0.5, 0.75, 1.0, 1.25, 1.6, 2.2, np.inf],
        labels=[f"TA{i}" for i in range(8)],
    ).astype("string").fillna("__MISSING__")

    combo_specs = [
        ("X2_CAD_Race_Compound_AvgStint", ["Race", "Compound", "X2_CAD_AvgStintBin"]),
        ("X2_CAD_Race_Compound_PastExpected", ["Race", "Compound", "X2_CAD_PastExpectedBin"]),
        ("X2_CAD_Year_Compound_PastExpected", ["Year", "Compound", "X2_CAD_PastExpectedBin"]),
        ("X2_CAD_Compound_Stint_TyreOverAvg", ["Compound", "Stint", "X2_CAD_TyreOverAvgBin"]),
        ("X2_CAD_Race_PitStop_LastPit", ["Race", "PitStop", "X2_CAD_LastPitLapBin"]),
        ("X2_CAD_Race_Stint_AvgStint", ["Race", "Stint", "X2_CAD_AvgStintBin"]),
        ("X2_CAD_Race_Compound_Stint_TyreOverAvg", ["Race", "Compound", "Stint", "X2_CAD_TyreOverAvgBin"]),
    ]
    for name, cols in combo_specs:
        if all(c in out.columns for c in cols):
            out[name] = combo_key(out, cols)
    return out


def add_artifact_hash_key_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hash keys (X2_HASH_*): rounded timing values and integer race state, alone and in 8 combinations."""
    out = df.copy()

    def rounded(col: str, nd: int) -> pd.Series:
        if col not in out.columns:
            return pd.Series("__MISSING__", index=out.index, dtype="string")
        return pd.to_numeric(out[col], errors="coerce").round(nd).astype("string").fillna("__MISSING__")

    pieces = {
        "LTD_r1": rounded("LapTime_Delta", 1),
        "LTD_r2": rounded("LapTime_Delta", 2),
        "DEG_r1": rounded("Cumulative_Degradation", 1),
        "DEG_r2": rounded("Cumulative_Degradation", 2),
        "RP_r2": rounded("RaceProgress", 2),
        "RP_r3": rounded("RaceProgress", 3),
        "LT_r1": rounded("LapTime_(s)", 1),
        "LT_r2": rounded("LapTime_(s)", 2),
        "TY": rounded("TyreLife", 0),
        "LAP": rounded("LapNumber", 0),
        "POS": rounded("Position", 0),
    }
    for name, series in pieces.items():
        out[f"X2_HASH_{name}"] = series

    combo_specs = [
        ("X2_HASH_Race_Compound_Tyre_LTD", ["Race", "Compound", "X2_HASH_TY", "X2_HASH_LTD_r2"]),
        ("X2_HASH_Race_Compound_Tyre_DEG", ["Race", "Compound", "X2_HASH_TY", "X2_HASH_DEG_r2"]),
        ("X2_HASH_Race_Stint_Progress_Tyre", ["Race", "Stint", "X2_HASH_RP_r2", "X2_HASH_TY"]),
        ("X2_HASH_Year_Race_Progress_Tyre", ["Year", "Race", "X2_HASH_RP_r3", "X2_HASH_TY"]),
        ("X2_HASH_Compound_Stint_LTD_DEG", ["Compound", "Stint", "X2_HASH_LTD_r1", "X2_HASH_DEG_r1"]),
        ("X2_HASH_Race_Compound_Lap_Tyre", ["Race", "Compound", "X2_HASH_LAP", "X2_HASH_TY"]),
        ("X2_HASH_Race_Compound_LT_LTD", ["Race", "Compound", "X2_HASH_LT_r1", "X2_HASH_LTD_r1"]),
        ("X2_HASH_Race_Tyre_Pos", ["Race", "X2_HASH_TY", "X2_HASH_POS"]),
    ]
    for name, cols in combo_specs:
        if all(c in out.columns for c in cols):
            out[name] = combo_key(out, cols)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Data, base matrix
# ----------------------------------------------------------------------------------------------------------------------
def make_augmented_train(cfg: Config, train_comp: pd.DataFrame, test: pd.DataFrame, ext: pd.DataFrame):
    """Competition rows, then the original rows (minus any row that repeats a competition row or an earlier original row,
    compared as strings of the 14 features + target), with per-row sample weights."""
    feature_cols = [c for c in test.columns if c != ID and c in train_comp.columns]
    required_cols = feature_cols + [TARGET, "_source"]
    comp = train_comp[required_cols].copy().reset_index(drop=True)
    comp_n = len(comp)
    ext_aligned = ext[feature_cols + [TARGET, "_source"]].copy()
    ext_aligned[TARGET] = pd.to_numeric(ext_aligned[TARGET], errors="coerce")
    ext_aligned = ext_aligned[ext_aligned[TARGET].notna()].copy()
    ext_aligned[TARGET] = ext_aligned[TARGET].astype(int)
    before = len(ext_aligned)
    comp_keys = set(map(tuple, comp[feature_cols + [TARGET]].astype(str).values.tolist()))
    ext_keys = ext_aligned[feature_cols + [TARGET]].astype(str).apply(tuple, axis=1)
    ext_aligned = ext_aligned.loc[~ext_keys.isin(comp_keys)].copy()
    ext_aligned = ext_aligned.drop_duplicates(subset=feature_cols + [TARGET], keep="first").reset_index(drop=True)
    print(f"Dropped {before - len(ext_aligned)} duplicate original rows.", flush=True)
    train_aug = pd.concat([comp, ext_aligned], axis=0, ignore_index=True)
    train_aug[TARGET] = pd.to_numeric(train_aug[TARGET], errors="coerce")
    train_aug = train_aug[train_aug[TARGET].notna()].copy()
    train_aug[TARGET] = train_aug[TARGET].astype(int)
    source_weight = train_aug["_source"].map({
        "competition": COMPETITION_SAMPLE_WEIGHT,
        "external": cfg.external_sample_weight,
    }).fillna(1.0).astype(float).values
    return train_aug.reset_index(drop=True), comp_n, source_weight


def build_base_features(cfg: Config, train_aug: pd.DataFrame, test: pd.DataFrame):
    feature_cols = [c for c in test.columns if c != ID and c in train_aug.columns]
    tr = train_aug[feature_cols].copy()
    te = test[feature_cols].copy()
    tr["_is_train"] = 1
    te["_is_train"] = 0
    all_df = pd.concat([tr, te], axis=0, ignore_index=True)
    all_df.columns = [normalize_colname(c) for c in all_df.columns]
    all_df = try_numeric_cast(all_df, exclude=["_is_train"])
    cat_cols_initial = all_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for c in cat_cols_initial:
        all_df[c] = string_key(all_df[c])
    all_df = add_categorical_combinations(all_df)
    all_df = add_domain_numeric_features(all_df)
    if cfg.use_public_formula_features:
        all_df = add_public_formula_features(all_df)
    all_df = add_group_aggregates(all_df)
    all_df = add_sequence_features(all_df, include_next=cfg.use_sequence_next_features)
    if cfg.use_hazard_features:
        all_df = add_hazard_features(all_df)
    if cfg.use_relative_rank_features:
        all_df = add_relative_rank_features(all_df)
    if cfg.use_pipeline_a_features:
        all_df = add_pipeline_a_features(all_df)
    if cfg.use_catboost_style_features:
        all_df = add_catboost_style_features(all_df)
    if cfg.use_extended_group_residuals:
        all_df = add_extended_group_residuals(all_df)
    if cfg.use_pit_window_key_features:
        all_df = add_pit_window_key_features(all_df)
    if cfg.use_pit_cadence_features:
        all_df = add_pit_cadence_features(all_df)
    if cfg.use_artifact_hash_keys:
        all_df = add_artifact_hash_key_features(all_df)
    all_df = add_frequency_features(all_df)
    all_df = all_df.replace([np.inf, -np.inf], np.nan)
    train_fe = all_df[all_df["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
    test_fe = all_df[all_df["_is_train"] == 0].drop(columns=["_is_train"]).reset_index(drop=True)
    constant_cols = [c for c in train_fe.columns if train_fe[c].nunique(dropna=False) <= 1]
    if constant_cols:
        train_fe = train_fe.drop(columns=constant_cols)
        test_fe = test_fe.drop(columns=constant_cols)
    test_fe = test_fe[train_fe.columns]
    cat_cols = train_fe.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return train_fe, test_fe, cat_cols


# ----------------------------------------------------------------------------------------------------------------------
# Per-fold encodings
# ----------------------------------------------------------------------------------------------------------------------
def get_target_encode_columns(X: pd.DataFrame, cat_cols: Sequence[str], max_cardinality: int) -> List[str]:
    out = []
    for c in cat_cols:
        n = X[c].nunique(dropna=False)
        if 1 < n <= max_cardinality:
            out.append(c)
    return out


def _fit_te_mapping(cat: pd.Series, y_values: np.ndarray, smooth: float):
    prior = float(np.mean(y_values))
    tmp = pd.DataFrame({"cat": string_key(cat), "target": y_values})
    stats = tmp.groupby("cat", dropna=False)["target"].agg(["count", "sum"])
    enc = (stats["sum"] + smooth * prior) / (stats["count"] + smooth)
    return enc, prior


def add_fold_safe_target_encoding(X_tr, y_tr, X_val, X_test, cols, smooth_values, inner_splits, seed):
    X_tr = X_tr.reset_index(drop=True).copy()
    X_val = X_val.reset_index(drop=True).copy()
    X_test = X_test.reset_index(drop=True).copy()
    y_tr = np.asarray(y_tr).astype(int)
    class_counts = pd.Series(y_tr).value_counts()
    max_inner = int(min(inner_splits, class_counts.min())) if len(class_counts) == 2 else 0
    use_inner_cv = max_inner >= 2
    for col in cols:
        tr_cat = string_key(X_tr[col])
        val_cat = string_key(X_val[col])
        test_cat = string_key(X_test[col])
        for smooth in smooth_values:
            new_col = f"{col}__te_s{smooth}"
            prior = float(np.mean(y_tr))
            oof = np.full(len(X_tr), prior, dtype=float)
            if use_inner_cv:
                inner = StratifiedKFold(n_splits=max_inner, shuffle=True, random_state=seed)
                for inner_tr_idx, inner_val_idx in inner.split(X_tr, y_tr):
                    enc, inner_prior = _fit_te_mapping(tr_cat.iloc[inner_tr_idx], y_tr[inner_tr_idx], smooth)
                    oof[inner_val_idx] = tr_cat.iloc[inner_val_idx].map(enc).fillna(inner_prior).astype(float).values
            else:
                enc, prior = _fit_te_mapping(tr_cat, y_tr, smooth)
                oof = tr_cat.map(enc).fillna(prior).astype(float).values
            full_enc, full_prior = _fit_te_mapping(tr_cat, y_tr, smooth)
            X_tr[new_col] = oof
            X_val[new_col] = val_cat.map(full_enc).fillna(full_prior).astype(float).values
            X_test[new_col] = test_cat.map(full_enc).fillna(full_prior).astype(float).values
    return X_tr, X_val, X_test


def add_fold_frequency_encoding(X_tr, X_val, X_test, cols):
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    n = max(len(X_tr), 1)
    for col in cols:
        vc = string_key(X_tr[col]).value_counts(dropna=False)
        for frame in (X_tr, X_val, X_test):
            key = string_key(frame[col])
            frame[f"{col}__fold_count"] = key.map(vc).fillna(0).astype(float).values
            frame[f"{col}__fold_freq"] = frame[f"{col}__fold_count"] / n
    return X_tr, X_val, X_test


def ordinal_encode_fold(X_tr, X_val, X_test, cat_cols):
    if not cat_cols:
        return X_tr, X_val, X_test
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    for frame in (X_tr, X_val, X_test):
        for c in cat_cols:
            frame[c] = string_key(frame[c]).astype(str)
    X_tr[cat_cols] = pd.DataFrame(enc.fit_transform(X_tr[cat_cols]), index=X_tr.index, columns=cat_cols)
    X_val[cat_cols] = pd.DataFrame(enc.transform(X_val[cat_cols]), index=X_val.index, columns=cat_cols)
    X_test[cat_cols] = pd.DataFrame(enc.transform(X_test[cat_cols]), index=X_test.index, columns=cat_cols)
    return X_tr, X_val, X_test


def finalize_matrix(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype(int)
    obj_cols = out.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if obj_cols:
        raise ValueError(f"Unencoded categorical/object columns remain: {obj_cols[:10]}")
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def make_fold_matrices(cfg, X_base, y_aug, X_test_base, train_idx, valid_idx, cat_cols, target_encode_cols, seed):
    X_tr = X_base.iloc[train_idx].copy()
    X_val = X_base.iloc[valid_idx].copy()
    X_te = X_test_base.copy()
    y_tr = y_aug.iloc[train_idx].astype(int).values
    y_val = y_aug.iloc[valid_idx].astype(int).values
    X_tr, X_val, X_te = add_fold_safe_target_encoding(
        X_tr, y_tr, X_val, X_te,
        cols=target_encode_cols,
        smooth_values=cfg.te_smooths,
        inner_splits=TE_INNER_SPLITS,
        seed=seed,
    )
    X_tr, X_val, X_te = add_fold_frequency_encoding(X_tr, X_val, X_te, target_encode_cols)
    X_tr, X_val, X_te = ordinal_encode_fold(X_tr, X_val, X_te, cat_cols)
    X_tr = finalize_matrix(X_tr)
    X_val = finalize_matrix(X_val)
    X_te = finalize_matrix(X_te)
    return X_tr, X_val, X_te, y_tr, y_val


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
BEST_XGB_HYPERPARAMS = {
    "grow_policy": "lossguide",
    "learning_rate": 0.024987538635254592,
    "min_child_weight": 15.75630717524033,
    "subsample": 0.9670845170126153,
    "colsample_bytree": 0.9650051590993955,
    "colsample_bynode": 0.7394791685463444,
    "gamma": 0.32044599184254663,
    "reg_alpha": 0.00014710455077413345,
    "reg_lambda": 4.18625488834861,
    "max_bin": 384,
    "max_leaves": 96,
}


def fixed_model_params(cfg: Config) -> Dict:
    params = {
        "n_estimators": 16000,
        "max_depth": 0,
        "early_stopping_rounds": 400,
        **BEST_XGB_HYPERPARAMS,
    }
    params.update(cfg.xgb_params_override)
    return params


def xgb_runtime_params(seed: int) -> Dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
        "device": "cuda",
    }


def run(cfg: Config, model_dir, fold_ids, force: bool = False) -> None:
    """Trains the requested outer folds (0-4) of recipe `cfg`, keeps each in parts/, writes the pair when all are done."""
    from xgboost import XGBClassifier

    train_comp = read_train()
    y_comp = train_comp[TARGET].astype(int).reset_index(drop=True)
    fold_of_row = folds(y_comp.values)
    todo = [k for k in fold_ids if force or not have_part(model_dir, cfg.pair, k)]
    for k in sorted(set(fold_ids) - set(todo)):
        print(f"{cfg.pair} fold {k}: done", flush=True)
    if todo:
        seed_everything(cfg.seed)
        test = read_test()
        ext = read_original()
        train_comp["_source"] = "competition"
        ext["_source"] = "external"
        train_aug, comp_n, source_weight = make_augmented_train(cfg, train_comp, test, ext)
        y_aug = train_aug[TARGET].astype(int).reset_index(drop=True)
        external_idx = np.arange(comp_n, len(train_aug))
        X_base, X_test_base, base_cat_cols = build_base_features(cfg, train_aug, test)
        if cfg.use_source_indicator:
            source_flag = np.zeros(len(X_base), dtype=np.float32)
            source_flag[external_idx] = 1.0
            X_base["SOURCE_is_external"] = source_flag
            X_test_base["SOURCE_is_external"] = np.zeros(len(X_test_base), dtype=np.float32)
        target_encode_cols = get_target_encode_columns(X_base, base_cat_cols, cfg.max_te_cardinality)
        print(f"Base feature shape: train={X_base.shape}, test={X_test_base.shape}", flush=True)
        print(f"Categorical columns: {len(base_cat_cols)}; target-encoded columns: {len(target_encode_cols)}", flush=True)
        base_params = fixed_model_params(cfg)

        for k in todo:
            tr_idx_comp = np.flatnonzero(fold_of_row != k)
            va_idx_comp = np.flatnonzero(fold_of_row == k)
            train_idx_aug = np.concatenate([tr_idx_comp, external_idx])
            X_tr, X_va, X_te, y_tr, y_va = make_fold_matrices(
                cfg, X_base, y_aug, X_test_base, train_idx_aug, va_idx_comp,
                base_cat_cols, target_encode_cols, cfg.seed,
            )
            sw_tr = source_weight[train_idx_aug]
            params = dict(base_params)
            params.update(xgb_runtime_params(cfg.seed + k + 1))   # random_state 43..47 for folds 0..4
            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)
            va_pred = model.predict_proba(X_va)[:, 1]
            te_pred = model.predict_proba(X_te)[:, 1]
            save_part(model_dir, cfg.pair, k, np.clip(va_pred, 1e-7, 1 - 1e-7), np.clip(te_pred, 1e-7, 1 - 1e-7))
            print(f"{cfg.pair} fold {k}: {X_tr.shape[1]} columns, AUC {roc_auc_score(y_va, va_pred):.6f}, "
                  f"best_iteration {getattr(model, 'best_iteration', None)}", flush=True)
            del X_tr, X_va, X_te, model, va_pred, te_pred
            gc.collect()
    assemble(model_dir, cfg.pair, fold_of_row, y_comp.values)
