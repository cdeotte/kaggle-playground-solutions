#!/usr/bin/env python
"""XGBoost on empirical tyre-life hazard / frailty features (pair xgb_hazard_frailty_pair_te_original_w075).

Per outer fold, the training rows are the fold's train rows + the fold-matched 4/5 of the original rows (sample weight
1.0 / 0.75); every encoder below is fitted on these rows only (original priors on the original rows only):
- base: the 11 numeric columns and 21 state / domain features (estimated total laps, laps remaining, tyre-age ratios,
  degradation and lap-time-delta rates, pit-window / position / stint pressures, ...); codes and log counts of the 10
  low-cardinality columns, of 9 crosses / bins (Race x Compound, Driver x Compound, Compound x Stint, Race x Year,
  Race x Driver, race-phase, tyre-life, lap and position bins) and of the 1- / 2-decimal roundings and floors of the 4
  continuous columns; the empirical CDF rank of the 11 numeric columns.
- hazard / frailty (a piecewise-constant hazard over 11 TyreLife bins, smoothing 45): the baseline hazard of the row's
  bin and the next bin, the implied survival curve (churn probability now / next, velocity), per-compound hazards and
  their ratio / difference to the baseline, and smoothed target means ("frailties") of Race x Compound, Driver x Compound,
  Compound x Stint, Race x Driver, tyre-life bin, race phase with their ratio / difference to the baseline hazard.
- original priors (smoothing 25): per key, the smoothed original target mean, log original / train counts and their
  drift log1p((train + 1) / (original + 1)).
- pairwise interactions of 15 categoricals and 12-quantile bins of 6 numerics (210 pairs), target-encoded with sklearn
  TargetEncoder (5 shuffled inner folds, seed 42 + fold), plus the row mean / std / min / max of those encodings.
Model: XGBoost hist on GPU, lr 0.005, depth 7, min_child_weight 6, subsample 0.9, colsample_bytree 0.35, lambda 5,
alpha 0.05, max_bin 1024, up to 50,000 rounds with early stopping 500 on the fold's validation rows (AUC), seed 42 + fold
(fold = 1..5). Probabilities clipped to [1e-7, 1 - 1e-7]. Test = mean of the 5 fold models.
Usage: python train_hazard_xgb.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, orig_folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = 'xgb_hazard_frailty_pair_te_original_w075'
FOLDER = 'classic'
SEED = 42
ORIGINAL_WEIGHT = 0.75
SYNTH_WEIGHT = 1.0

CAT_COLS = [
    "Driver", "Compound", "Race", "Year", "PitStop", "LapNumber", "Stint", "TyreLife",
    "Position", "Position_Change",
]
NUM_COLS = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]
STATE_NUM_COLS = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "Position_Change"] + NUM_COLS
HIGH_CARD_NUMS = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]

XGB_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "tree_method": "hist",
    "eval_metric": "auc",
    "n_estimators": 50000,
    "max_depth": 7,
    "learning_rate": 0.005,
    "n_jobs": 32,
    "reg_lambda": 5.0,
    "reg_alpha": 0.05,
    "colsample_bytree": 0.35,
    "subsample": 0.90,
    "min_child_weight": 6,
    "max_bin": 1024,
    "random_state": SEED,
    "verbosity": 0,
    "early_stopping_rounds": 500,
    "device": "cuda",
}


def clean_name(name: str) -> str:
    out = str(name)
    for ch in [" ", "(", ")", "/", "-", ":", ";", ",", "[", "]", "<", ">", "=", "{", "}"]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def make_unique(cols: Iterable[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for col in cols:
        base = clean_name(col)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = make_unique(out.columns)
    return out


def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)


def drop_model_cols(df: pd.DataFrame) -> pd.DataFrame:
    drops = [c for c in [ID, TARGET, "Normalized_TyreLife"] if c in df.columns]
    return df.drop(columns=drops).copy()


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CAT_COLS:
        if c in out.columns:
            out[c] = out[c].astype(str).fillna("missing")
    lap = to_numeric(out["LapNumber"])
    tyre = to_numeric(out["TyreLife"])
    pos = to_numeric(out["Position"])
    progress = to_numeric(out["RaceProgress"])
    out["Race_Compound_"] = out["Race"].astype(str) + "_" + out["Compound"].astype(str)
    out["Driver_Compound_"] = out["Driver"].astype(str) + "_" + out["Compound"].astype(str)
    out["Compound_Stint_"] = out["Compound"].astype(str) + "_" + out["Stint"].astype(str)
    out["Race_Year_"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)
    out["Race_Driver_"] = out["Race"].astype(str) + "_" + out["Driver"].astype(str)
    out["RacePhase_"] = pd.cut(progress, bins=[-np.inf, .2, .4, .6, .8, np.inf], labels=["P1", "P2", "P3", "P4", "P5"]).astype(str)
    out["TyreLifeBin_"] = pd.cut(tyre, bins=[-np.inf, 2, 4, 7, 11, 16, 24, np.inf], labels=["T0", "T1", "T2", "T3", "T4", "T5", "T6"]).astype(str)
    out["LapBin_"] = pd.cut(lap, bins=[-np.inf, 3, 8, 15, 25, 40, np.inf], labels=["L0", "L1", "L2", "L3", "L4", "L5"]).astype(str)
    out["PosBin_"] = pd.cut(pos, bins=[-np.inf, 3, 8, 14, np.inf], labels=["front", "upper", "lower", "back"]).astype(str)
    for col in HIGH_CARD_NUMS:
        safe = clean_name(col)
        x = to_numeric(out[col])
        out[f"ROUND_{safe}_1_"] = x.round(1).astype(str)
        out[f"ROUND_{safe}_2_"] = x.round(2).astype(str)
        out[f"FLOOR_{safe}_"] = np.floor(x).astype("int32").astype(str)
    return out


def add_domain_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    lap = to_numeric(df["LapNumber"]).astype(float)
    tyre = to_numeric(df["TyreLife"]).astype(float)
    pit = to_numeric(df["PitStop"]).astype(float)
    stint = to_numeric(df["Stint"]).astype(float)
    pos = to_numeric(df["Position"]).astype(float)
    pos_chg = to_numeric(df["Position_Change"]).astype(float)
    lap_time = to_numeric(df["LapTime (s)"]).astype(float)
    delta = to_numeric(df["LapTime_Delta"]).astype(float)
    deg = to_numeric(df["Cumulative_Degradation"]).astype(float)
    progress = to_numeric(df["RaceProgress"]).clip(lower=1e-6).astype(float)

    for c in STATE_NUM_COLS:
        out[f"raw_{clean_name(c)}"] = to_numeric(df[c]).astype("float32")

    est_total = (lap / progress).clip(1, 120)
    laps_remaining = (est_total - lap).clip(lower=0)
    out["est_total_laps"] = est_total.astype("float32")
    out["laps_remaining"] = laps_remaining.astype("float32")
    out["tyre_age_ratio"] = (tyre / lap.clip(lower=1)).astype("float32")
    out["lap_per_tyre_life"] = (lap / (tyre + 1)).astype("float32")
    out["lap_minus_tyre"] = (lap - tyre).astype("float32")
    out["pit_window_pressure"] = (tyre * progress).astype("float32")
    out["tyre_to_laps_remaining"] = (tyre / (laps_remaining + 1)).astype("float32")
    out["deg_per_tyre_lap"] = (deg / tyre.clip(lower=1)).astype("float32")
    out["abs_deg_per_tyre_lap"] = (deg.abs() / tyre.clip(lower=1)).astype("float32")
    out["delta_per_tyre_lap"] = (delta / tyre.clip(lower=1)).astype("float32")
    out["abs_delta_per_tyre_lap"] = (delta.abs() / tyre.clip(lower=1)).astype("float32")
    out["abs_delta"] = delta.abs().astype("float32")
    out["abs_deg"] = deg.abs().astype("float32")
    out["abs_pos_change"] = pos_chg.abs().astype("float32")
    out["pos_pressure"] = (pos * progress).astype("float32")
    out["stint_pressure"] = (stint * tyre).astype("float32")
    out["lap_time_delta_ratio"] = (delta / lap_time.abs().clip(lower=1)).astype("float32")
    out["lap_time_degradation_ratio"] = (deg / lap_time.abs().clip(lower=1)).astype("float32")
    out["pitstop_progress"] = (pit * progress).astype("float32")
    out["late_stint_pressure"] = (tyre * progress * (stint + 1)).astype("float32")
    out["position_recovery_pressure"] = (pos_chg * progress).astype("float32")
    return out


class BaseFeatureState:
    def __init__(self):
        self.cat_cols: List[str] = []
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.count_maps: Dict[str, pd.Series] = {}
        self.rank_values: Dict[str, np.ndarray] = {}

    def fit(self, df_raw: pd.DataFrame) -> "BaseFeatureState":
        df = add_keys(df_raw)
        extra_keys = [
            "Race_Compound_", "Driver_Compound_", "Compound_Stint_", "Race_Year_", "Race_Driver_",
            "RacePhase_", "TyreLifeBin_", "LapBin_", "PosBin_",
        ]
        round_keys = [c for c in df.columns if c.startswith(("ROUND_", "FLOOR_"))]
        self.cat_cols = [c for c in CAT_COLS + extra_keys + round_keys if c in df.columns]
        for c in self.cat_cols:
            vals = df[c].astype(str).fillna("missing")
            uniq = vals.drop_duplicates().tolist()
            self.cat_maps[c] = {v: i + 1 for i, v in enumerate(uniq)}
            self.count_maps[c] = vals.value_counts(dropna=False)
        for c in STATE_NUM_COLS:
            vals = np.sort(to_numeric(df[c]).to_numpy(dtype=float))
            self.rank_values[c] = vals
        return self

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = add_keys(df_raw)
        out = add_domain_numeric(df)
        for c in self.cat_cols:
            vals = df[c].astype(str).fillna("missing")
            out[f"cat_code_{clean_name(c)}"] = vals.map(self.cat_maps[c]).fillna(0).astype("int32")
            cnt = vals.map(self.count_maps[c]).fillna(0).astype(float)
            out[f"freq_{clean_name(c)}"] = np.log1p(cnt).astype("float32")
        for c, sorted_vals in self.rank_values.items():
            vals = to_numeric(df[c]).to_numpy(dtype=float)
            out[f"rank_{clean_name(c)}"] = (np.searchsorted(sorted_vals, vals, side="right") / max(1, len(sorted_vals))).astype("float32")
        return out.reset_index(drop=True)


class EmpiricalHazardFrailty:
    """Piecewise-constant hazard over TyreLife bins, smoothed toward the global mean (baseline) or the baseline hazard
    (per compound), plus smoothed group 'frailties'."""

    def __init__(self, alpha: float = 40.0):
        self.alpha = float(alpha)
        self.edges = np.array([-np.inf, 0, 1, 2, 3, 5, 8, 12, 16, 22, 30, np.inf])
        self.n_bins = len(self.edges) - 1
        self.global_mean = 0.0
        self.baseline = np.zeros(self.n_bins, dtype=np.float32)
        self.compound_hazards: Dict[str, np.ndarray] = {}
        self.group_maps: Dict[str, Dict[str, float]] = {}
        self.compounds: List[str] = []

    def _bin(self, s: pd.Series) -> np.ndarray:
        return np.digitize(to_numeric(s).to_numpy(dtype=float), self.edges[1:-1], right=True)

    def _smooth_mean(self, y, prior):
        y = np.asarray(y, dtype=float)
        return (y.sum() + self.alpha * prior) / (len(y) + self.alpha)

    def fit(self, df_raw: pd.DataFrame, y: np.ndarray) -> "EmpiricalHazardFrailty":
        df = add_keys(df_raw)
        y = np.asarray(y, dtype=float)
        self.global_mean = float(y.mean())
        bins = self._bin(df["TyreLife"])
        for b in range(self.n_bins):
            mask = bins == b
            self.baseline[b] = self._smooth_mean(y[mask], self.global_mean) if mask.any() else self.global_mean
        self.baseline = np.clip(self.baseline, 1e-5, 1 - 1e-5)

        self.compounds = sorted(df["Compound"].astype(str).unique().tolist())
        for comp in self.compounds:
            comp_mask = df["Compound"].astype(str).to_numpy() == comp
            vec = np.zeros(self.n_bins, dtype=np.float32)
            for b in range(self.n_bins):
                mask = comp_mask & (bins == b)
                vec[b] = self._smooth_mean(y[mask], self.baseline[b]) if mask.any() else self.baseline[b]
            self.compound_hazards[comp] = np.clip(vec, 1e-5, 1 - 1e-5)

        for key in ["Race_Compound_", "Driver_Compound_", "Compound_Stint_", "Race_Driver_", "TyreLifeBin_", "RacePhase_"]:
            vals = df[key].astype(str)
            stats = {}
            for val, idx in vals.groupby(vals).groups.items():
                stats[val] = float(self._smooth_mean(y[list(idx)], self.global_mean))
            self.group_maps[key] = stats
        return self

    def _survival_from_hazard(self, hazard_vec: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        surv_curve = np.cumprod(1.0 - np.clip(hazard_vec, 1e-5, 1 - 1e-5))
        cur = surv_curve[np.clip(bins, 0, self.n_bins - 1)]
        nxt = surv_curve[np.clip(bins + 1, 0, self.n_bins - 1)]
        vel = cur - nxt
        return cur, nxt, vel

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = add_keys(df_raw)
        out = pd.DataFrame(index=df.index)
        bins = self._bin(df["TyreLife"])
        base_h = self.baseline[np.clip(bins, 0, self.n_bins - 1)]
        next_h = self.baseline[np.clip(bins + 1, 0, self.n_bins - 1)]
        surv_cur, surv_next, surv_vel = self._survival_from_hazard(self.baseline, bins)
        out["surv_base_hazard"] = base_h.astype("float32")
        out["surv_base_hazard_next"] = next_h.astype("float32")
        out["surv_base_churn_prob"] = (1 - surv_cur).astype("float32")
        out["surv_base_next_churn_prob"] = (1 - surv_next).astype("float32")
        out["surv_base_velocity"] = surv_vel.astype("float32")

        comp_h = np.zeros(len(df), dtype=np.float32)
        comp_next_h = np.zeros(len(df), dtype=np.float32)
        comp_surv = np.zeros(len(df), dtype=np.float32)
        comp_vel = np.zeros(len(df), dtype=np.float32)
        comp_vals = df["Compound"].astype(str).to_numpy()
        for i, comp in enumerate(comp_vals):
            vec = self.compound_hazards.get(comp, self.baseline)
            b = int(np.clip(bins[i], 0, self.n_bins - 1))
            comp_h[i] = vec[b]
            comp_next_h[i] = vec[min(b + 1, self.n_bins - 1)]
            s_cur, _, s_vel = self._survival_from_hazard(vec, np.array([b]))
            comp_surv[i] = s_cur[0]
            comp_vel[i] = s_vel[0]
        out["surv_compound_hazard"] = comp_h
        out["surv_compound_hazard_next"] = comp_next_h
        out["surv_compound_churn_prob"] = 1 - comp_surv
        out["surv_compound_velocity"] = comp_vel
        out["surv_hazard_ratio_compound_base"] = (comp_h / np.clip(base_h, 1e-5, None)).astype("float32")
        out["surv_hazard_delta_compound_base"] = (comp_h - base_h).astype("float32")

        for key, stats in self.group_maps.items():
            vals = df[key].astype(str)
            mapped = vals.map(stats).fillna(self.global_mean).astype("float32")
            safe = clean_name(key)
            out[f"surv_frailty_{safe}"] = mapped
            out[f"surv_frailty_ratio_{safe}"] = (mapped / np.clip(base_h, 1e-5, None)).astype("float32")
            out[f"surv_frailty_delta_{safe}"] = (mapped - base_h).astype("float32")
        return out.reset_index(drop=True)


class OriginalPriorState:
    def __init__(self, alpha: float = 20.0):
        self.alpha = float(alpha)
        self.global_mean = 0.0
        self.prior_maps: Dict[str, pd.Series] = {}
        self.orig_counts: Dict[str, pd.Series] = {}
        self.train_counts: Dict[str, pd.Series] = {}
        self.keys: List[str] = []

    def fit(self, orig_raw: pd.DataFrame, y_orig: pd.Series, train_ref: pd.DataFrame) -> "OriginalPriorState":
        orig = add_keys(orig_raw).reset_index(drop=True)
        train = add_keys(train_ref).reset_index(drop=True)
        y = pd.Series(y_orig).reset_index(drop=True).astype(float)
        self.global_mean = float(y.mean()) if len(y) else 0.0
        candidate_keys = CAT_COLS + [
            "Race_Compound_", "Driver_Compound_", "Compound_Stint_", "Race_Year_", "Race_Driver_",
            "RacePhase_", "TyreLifeBin_", "LapBin_", "PosBin_",
        ] + [c for c in orig.columns if c.startswith(("ROUND_", "FLOOR_"))]
        self.keys = [c for c in candidate_keys if c in orig.columns and c in train.columns]
        tmp = orig.copy()
        tmp["_y_"] = y
        for key in self.keys:
            vals = tmp[key].astype(str)
            grp = tmp.groupby(vals, observed=False)["_y_"].agg(["sum", "count"])
            prior = (grp["sum"] + self.alpha * self.global_mean) / (grp["count"] + self.alpha)
            self.prior_maps[key] = prior
            self.orig_counts[key] = vals.value_counts(dropna=False)
            self.train_counts[key] = train[key].astype(str).value_counts(dropna=False)
        return self

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = add_keys(df_raw)
        out = pd.DataFrame(index=df.index)
        for key in self.keys:
            vals = df[key].astype(str)
            safe = clean_name(key)
            oc = vals.map(self.orig_counts[key]).fillna(0).astype(float)
            tc = vals.map(self.train_counts[key]).fillna(0).astype(float)
            out[f"orig_prior_{safe}"] = vals.map(self.prior_maps[key]).fillna(self.global_mean).astype("float32")
            out[f"orig_count_{safe}"] = np.log1p(oc).astype("float32")
            out[f"train_count_{safe}"] = np.log1p(tc).astype("float32")
            out[f"train_orig_drift_{safe}"] = np.log1p((tc + 1.0) / (oc + 1.0)).astype("float32")
        return out.reset_index(drop=True)


class InteractionBuilder:
    def __init__(self, q: int = 12):
        self.q = int(q)
        self.cat_cols = [
            "Driver", "Compound", "Race", "Year", "PitStop", "Stint", "Position", "Position_Change",
            "Race_Compound_", "Driver_Compound_", "Compound_Stint_", "RacePhase_", "TyreLifeBin_", "LapBin_", "PosBin_",
        ]
        self.num_cols = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "TyreLife", "LapNumber"]
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.combo_cols: List[str] = []
        self.maps: Dict[str, Dict[str, int]] = {}

    def fit(self, df_raw: pd.DataFrame) -> "InteractionBuilder":
        df = add_keys(df_raw)
        for c in self.num_cols:
            vals = to_numeric(df[c]).to_numpy(dtype=float)
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, self.q + 1)))
            if len(edges) <= 2:
                edges = np.array([-np.inf, np.inf])
            else:
                edges[0] = -np.inf
                edges[-1] = np.inf
            self.bin_edges[c] = edges
        str_df = self._strings(df)
        cols = list(str_df.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                name = f"I_{clean_name(c1)}__{clean_name(c2)}"
                combo = str_df[c1].astype(str) + "__" + str_df[c2].astype(str)
                uniq = combo.drop_duplicates().tolist()
                self.maps[name] = {v: k for k, v in enumerate(uniq)}
                self.combo_cols.append(name)
        return self

    def _strings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_keys(df)
        out = pd.DataFrame(index=df.index)
        for c in self.cat_cols:
            if c in df.columns:
                out[c] = df[c].astype(str).fillna("missing")
        for c, edges in self.bin_edges.items():
            vals = to_numeric(df[c]).to_numpy(dtype=float)
            out[f"BIN_{clean_name(c)}"] = np.digitize(vals, edges[1:-1], right=True).astype(str)
        return out

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        str_df = self._strings(df_raw)
        out = pd.DataFrame(index=df_raw.index)
        cols = list(str_df.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                name = f"I_{clean_name(c1)}__{clean_name(c2)}"
                combo = str_df[c1].astype(str) + "__" + str_df[c2].astype(str)
                out[name] = combo.map(self.maps[name]).fillna(-1).astype("int32")
        return out.reset_index(drop=True)


def assemble_fold_features(df_tr_synth_raw, y_tr_synth, df_val_raw, df_test_raw, df_orig_tr_raw, y_orig_tr, fold_seed):
    df_tr_synth = drop_model_cols(df_tr_synth_raw).reset_index(drop=True)
    df_val = drop_model_cols(df_val_raw).reset_index(drop=True)
    df_test = drop_model_cols(df_test_raw).reset_index(drop=True)
    df_orig_tr = drop_model_cols(df_orig_tr_raw).reset_index(drop=True)

    df_train_model = pd.concat([df_tr_synth, df_orig_tr], axis=0, ignore_index=True)
    y_train_model = pd.concat([y_tr_synth.reset_index(drop=True), y_orig_tr.reset_index(drop=True)], axis=0,
                              ignore_index=True).astype(int)
    sample_weight = np.concatenate([
        np.full(len(df_tr_synth), SYNTH_WEIGHT, dtype=np.float32),
        np.full(len(df_orig_tr), ORIGINAL_WEIGHT, dtype=np.float32),
    ])

    base_state = BaseFeatureState().fit(df_train_model)
    hazard_state = EmpiricalHazardFrailty(alpha=45.0).fit(df_train_model, y_train_model.to_numpy())
    prior_state = OriginalPriorState(alpha=25.0).fit(df_orig_tr, y_orig_tr.reset_index(drop=True), df_train_model)
    interaction_state = InteractionBuilder(q=12).fit(df_train_model)

    def build(df):
        parts = [base_state.transform(df), hazard_state.transform(df), prior_state.transform(df)]
        inter = interaction_state.transform(df)
        return pd.concat(parts + [inter], axis=1), inter

    X_tr, inter_tr = build(df_train_model)
    X_val, inter_val = build(df_val)
    X_test, inter_test = build(df_test)

    te_cols = list(inter_tr.columns)
    print(f"  target encoding {len(te_cols)} interaction columns", flush=True)
    te = TargetEncoder(target_type="binary", smooth="auto", cv=5, shuffle=True, random_state=fold_seed)
    te_tr = te.fit_transform(inter_tr[te_cols], y_train_model)
    te_val = te.transform(inter_val[te_cols])
    te_test = te.transform(inter_test[te_cols])
    te_names = [f"TE_{c}" for c in te_cols]

    X_tr = X_tr.drop(columns=te_cols)
    X_val = X_val.drop(columns=te_cols)
    X_test = X_test.drop(columns=te_cols)
    X_tr[te_names] = te_tr.astype("float32")
    X_val[te_names] = te_val.astype("float32")
    X_test[te_names] = te_test.astype("float32")

    for frame in [X_tr, X_val, X_test]:
        frame["te_mean"] = frame[te_names].mean(axis=1).astype("float32")
        frame["te_std"] = frame[te_names].std(axis=1).fillna(0).astype("float32")
        frame["te_min"] = frame[te_names].min(axis=1).astype("float32")
        frame["te_max"] = frame[te_names].max(axis=1).astype("float32")
        frame.replace([np.inf, -np.inf], 0, inplace=True)
        frame.fillna(0, inplace=True)

    X_tr = sanitize_columns(X_tr).astype("float32")
    X_val = sanitize_columns(X_val).astype("float32")
    X_test = sanitize_columns(X_test).astype("float32")
    X_val = X_val.reindex(columns=X_tr.columns, fill_value=0).astype("float32")
    X_test = X_test.reindex(columns=X_tr.columns, fill_value=0).astype("float32")
    return X_tr, X_val, X_test, y_train_model.to_numpy(), sample_weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    out = model_dir(FOLDER)
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].astype(int).reset_index(drop=True)
    fold_id = folds(y.to_numpy())
    orig_fold_id = orig_folds(orig[TARGET].astype(int).to_numpy())

    for k in fold_list(args.folds):
        if have_part(out, PAIR, k):
            print(f'{PAIR} fold {k}: done', flush=True)
            continue
        fold = k + 1
        tr_idx, val_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        orig_tr_idx = np.where(orig_fold_id != k)[0]   # fold-matched original rows
        df_tr_raw = train.iloc[tr_idx].reset_index(drop=True).copy()
        df_val_raw = train.iloc[val_idx].reset_index(drop=True).copy()
        df_test_raw = test.reset_index(drop=True).copy()
        df_orig_tr_raw = orig.iloc[orig_tr_idx].reset_index(drop=True).copy()
        y_tr_synth = df_tr_raw[TARGET].astype(int).reset_index(drop=True)
        y_val = df_val_raw[TARGET].astype(int).reset_index(drop=True).to_numpy()
        y_orig_tr = df_orig_tr_raw[TARGET].astype(int).reset_index(drop=True)

        X_tr, X_val, X_test, y_tr_model, sample_weight = assemble_fold_features(
            df_tr_raw, y_tr_synth, df_val_raw, df_test_raw, df_orig_tr_raw, y_orig_tr, SEED + fold)
        print(f'{PAIR} fold {k}: train {X_tr.shape}, valid {X_val.shape}, test {X_test.shape}', flush=True)

        params = dict(XGB_PARAMS)
        params["random_state"] = SEED + fold
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr_model, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=250)
        val_pred = np.clip(model.predict_proba(X_val)[:, 1].astype(np.float32), 1e-7, 1 - 1e-7)
        fold_test = np.clip(model.predict_proba(X_test)[:, 1].astype(np.float32), 1e-7, 1 - 1e-7)
        save_part(out, PAIR, k, val_pred, fold_test)
        print(f'{PAIR} fold {k}: AUC {roc_auc_score(y_val, val_pred):.6f}, best iteration '
              f'{getattr(model, "best_iteration", params["n_estimators"])}', flush=True)
        del X_tr, X_val, X_test, y_tr_model, sample_weight, model
        gc.collect()
    assemble(out, PAIR, fold_id, y.to_numpy())


if __name__ == '__main__':
    main()
