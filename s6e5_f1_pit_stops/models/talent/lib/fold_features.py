"""Per-fold feature builder shared by every model in this folder.

FoldFeatureBuilder(cat_profile).fit(train_rows, original_rows, test_rows) learns, for one outer fold:
  - the categorical keys: `low` = 20 keys (Driver, Compound, Race, 7 integer-cast numerics, 4 bins, 6 crosses),
    `rich` = those + 14 (rounded lap time / delta / degradation / progress values and 7 more crosses);
  - the frequency of every key value over train rows + original rows + test rows (normalized value counts);
  - original-data priors: mean target and log1p count of 10 keys in the original rows (unseen values get the original
    mean and 0).
transform(df) returns (float32 numeric matrix, string categorical matrix): 11 raw numerics, 18 derived columns, one
frequency column per key and the 20 prior columns (69 numerics + 20 keys for `low`, 83 numerics + 34 keys for `rich`).
"""
import numpy as np
import pandas as pd

from common.data import TARGET

EPS = 1e-6

BASE_NUMS = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change",
]
BASE_CATS = ["Driver", "Compound", "Race"]
LOW_CARD_CATS = [
    "Driver", "Compound", "Race", "Year_cat", "PitStop_cat", "LapNumber_cat",
    "Stint_cat", "TyreLife_cat", "Position_cat", "Position_Change_cat",
    "TyreLifeBin", "LapBin", "ProgressBin", "PositionBin", "Race_x_Compound",
    "Race_x_Year", "Race_x_Stint", "Compound_x_TyreLifeBin", "LapBin_x_Race",
    "ProgressBin_x_Compound",
]
RICH_EXTRA_CATS = [
    "LapTime_r1", "LapTime_r2", "LapTime_Delta_r2", "LapTime_Delta_r3",
    "Cumulative_Degradation_r2", "Cumulative_Degradation_r3", "RaceProgress_r3",
    "Driver_x_Compound", "Driver_x_Race", "Driver_x_Race_x_Stint",
    "Race_x_Compound_x_TyreLifeBin", "Race_x_Compound_x_LapBin",
    "Compound_x_Delta_r2", "Race_x_LapTime_r1",
]
PRIOR_KEYS = [
    "Race", "Compound", "Driver", "Race_x_Compound", "Race_x_Year", "Race_x_Stint",
    "Compound_x_TyreLifeBin", "LapBin_x_Race", "ProgressBin_x_Compound",
    "Race_x_Compound_x_TyreLifeBin",
]


def to_num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series(default, index=df.index, dtype="float64")


def safe_div(a, b):
    return a / (np.abs(b) + EPS)


def add_basic_features(df):
    out = df.copy()
    for c in BASE_NUMS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    lap = to_num(out, "LapNumber")
    tyre = to_num(out, "TyreLife")
    stint = to_num(out, "Stint")
    pit = to_num(out, "PitStop")
    pos = to_num(out, "Position")
    pos_chg = to_num(out, "Position_Change")
    laptime = to_num(out, "LapTime (s)")
    delta = to_num(out, "LapTime_Delta")
    deg = to_num(out, "Cumulative_Degradation")
    progress = to_num(out, "RaceProgress")

    out["TyreAgeRatio"] = safe_div(tyre, lap + 1.0)
    out["StintLapRatio"] = safe_div(stint, lap + 1.0)
    out["ProgressLeft"] = 1.0 - progress
    out["LapNumberProgressResidual"] = lap - progress * 100.0
    out["DegPerTyreLap"] = safe_div(deg, tyre + 1.0)
    out["DeltaPerTyreLap"] = safe_div(delta, tyre + 1.0)
    out["LapTimePerProgress"] = safe_div(laptime, progress + 0.02)
    out["LapTimePerTyreLife"] = safe_div(laptime, tyre + 1.0)
    out["PositionPressure"] = safe_div(1.0, pos + 1.0) + 0.05 * pos_chg
    out["LateRaceOldTyre"] = progress * np.log1p(np.maximum(tyre, 0))
    out["DegDeltaProduct"] = deg * delta
    out["LapTimeMinusDelta"] = laptime - delta
    out["LapTimePlusDelta"] = laptime + delta
    out["PitWindowCore"] = ((progress > 0.18) & (progress < 0.95)).astype("float32") * safe_div(tyre, lap + 1.0)
    out["PaceStress"] = np.abs(delta) * np.log1p(np.maximum(tyre, 0)) + np.abs(deg)
    out["StintProgress"] = stint + progress
    out["PitStopLapProduct"] = pit * lap
    out["DegradationPressure"] = deg * progress * np.log1p(np.maximum(tyre, 0))
    return out


def int_cat(s):
    return pd.to_numeric(s, errors="coerce").round(0).astype("Int64").astype(str).fillna("nan")


def round_cat(s, decimals):
    return pd.to_numeric(s, errors="coerce").round(decimals).astype(str).fillna("nan")


def key_from(cols):
    out = cols[0].astype(str).fillna("nan")
    for c in cols[1:]:
        out = out.str.cat(c.astype(str).fillna("nan"), sep="|")
    return out


class FoldFeatureBuilder:
    def __init__(self, cat_profile="low", use_original_priors=True):
        self.cat_profile = cat_profile
        self.use_original_priors = use_original_priors
        self.freq_maps = {}
        self.prior_maps = {}
        self.prior_count_maps = {}
        self.global_prior = None
        self.cat_cols = None
        self.num_cols = None

    def make_categories(self, df):
        f = add_basic_features(df)
        cats = pd.DataFrame(index=f.index)
        for c in BASE_CATS:
            cats[c] = f[c].astype(str).fillna("nan") if c in f.columns else "nan"
        for c in ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "Position_Change"]:
            cats[f"{c}_cat"] = int_cat(f[c]) if c in f.columns else "nan"

        tyre = to_num(f, "TyreLife")
        lap = to_num(f, "LapNumber")
        pos = to_num(f, "Position")
        progress = to_num(f, "RaceProgress")
        cats["TyreLifeBin"] = np.floor(np.clip(tyre, 0, 200) / 3).astype("Int64").astype(str)
        cats["LapBin"] = np.floor(np.clip(lap, 0, 250) / 3).astype("Int64").astype(str)
        cats["ProgressBin"] = np.floor(np.clip(progress, 0, 1.5) * 30).astype("Int64").astype(str)
        cats["PositionBin"] = np.floor(np.clip(pos, 0, 100) / 2).astype("Int64").astype(str)

        cats["LapTime_r1"] = round_cat(f["LapTime (s)"], 1)
        cats["LapTime_r2"] = round_cat(f["LapTime (s)"], 2)
        cats["LapTime_Delta_r2"] = round_cat(f["LapTime_Delta"], 2)
        cats["LapTime_Delta_r3"] = round_cat(f["LapTime_Delta"], 3)
        cats["Cumulative_Degradation_r2"] = round_cat(f["Cumulative_Degradation"], 2)
        cats["Cumulative_Degradation_r3"] = round_cat(f["Cumulative_Degradation"], 3)
        cats["RaceProgress_r3"] = round_cat(f["RaceProgress"], 3)

        cats["Race_x_Compound"] = key_from([cats["Race"], cats["Compound"]])
        cats["Race_x_Year"] = key_from([cats["Race"], cats["Year_cat"]])
        cats["Race_x_Stint"] = key_from([cats["Race"], cats["Stint_cat"]])
        cats["Compound_x_TyreLifeBin"] = key_from([cats["Compound"], cats["TyreLifeBin"]])
        cats["LapBin_x_Race"] = key_from([cats["LapBin"], cats["Race"]])
        cats["ProgressBin_x_Compound"] = key_from([cats["ProgressBin"], cats["Compound"]])
        cats["Driver_x_Compound"] = key_from([cats["Driver"], cats["Compound"]])
        cats["Driver_x_Race"] = key_from([cats["Driver"], cats["Race"]])
        cats["Driver_x_Race_x_Stint"] = key_from([cats["Driver"], cats["Race"], cats["Stint_cat"]])
        cats["Race_x_Compound_x_TyreLifeBin"] = key_from([cats["Race"], cats["Compound"], cats["TyreLifeBin"]])
        cats["Race_x_Compound_x_LapBin"] = key_from([cats["Race"], cats["Compound"], cats["LapBin"]])
        cats["Compound_x_Delta_r2"] = key_from([cats["Compound"], cats["LapTime_Delta_r2"]])
        cats["Race_x_LapTime_r1"] = key_from([cats["Race"], cats["LapTime_r1"]])
        return cats

    def fit(self, train_df, orig_df, test_df):
        stat_df = pd.concat([train_df.drop(columns=[TARGET], errors="ignore"),
                             orig_df.drop(columns=[TARGET], errors="ignore"),
                             test_df.drop(columns=[TARGET], errors="ignore")], ignore_index=True)
        stat_cats = self.make_categories(stat_df)
        all_cats = LOW_CARD_CATS + (RICH_EXTRA_CATS if self.cat_profile == "rich" else [])
        self.cat_cols = [c for c in all_cats if c in stat_cats.columns]

        for c in self.cat_cols:
            vc = stat_cats[c].value_counts(normalize=True, dropna=False)
            self.freq_maps[c] = vc.to_dict()

        self.global_prior = float(orig_df[TARGET].mean()) if TARGET in orig_df.columns and len(orig_df) else float(train_df[TARGET].mean())
        if self.use_original_priors and TARGET in orig_df.columns and len(orig_df):
            orig_cats = self.make_categories(orig_df)
            y = orig_df[TARGET].astype(float).reset_index(drop=True)
            for key in PRIOR_KEYS:
                if key not in orig_cats.columns:
                    continue
                tmp = pd.DataFrame({"key": orig_cats[key].astype(str).values, "y": y.values})
                stats = tmp.groupby("key")["y"].agg(["mean", "count"])
                self.prior_maps[key] = stats["mean"].to_dict()
                self.prior_count_maps[key] = stats["count"].to_dict()

        probe = self.transform(train_df.drop(columns=[TARGET], errors="ignore"), return_frame=True)
        self.num_cols = probe["num_cols"]
        return self

    def transform(self, df, return_frame=False):
        f = add_basic_features(df)
        cats = self.make_categories(f)
        num_cols = [c for c in BASE_NUMS if c in f.columns]
        engineered = [
            "TyreAgeRatio", "StintLapRatio", "ProgressLeft", "LapNumberProgressResidual",
            "DegPerTyreLap", "DeltaPerTyreLap", "LapTimePerProgress", "LapTimePerTyreLife",
            "PositionPressure", "LateRaceOldTyre", "DegDeltaProduct", "LapTimeMinusDelta",
            "LapTimePlusDelta", "PitWindowCore", "PaceStress", "StintProgress",
            "PitStopLapProduct", "DegradationPressure",
        ]
        num_cols += [c for c in engineered if c in f.columns]

        for c, mp in self.freq_maps.items():
            name = f"freq_{c}"
            f[name] = cats[c].astype(str).map(mp).fillna(0.0).astype("float32")
            num_cols.append(name)

        for key, mp in self.prior_maps.items():
            prior_name = f"orig_prior_{key}"
            cnt_name = f"orig_logcount_{key}"
            f[prior_name] = cats[key].astype(str).map(mp).fillna(self.global_prior).astype("float32")
            f[cnt_name] = np.log1p(cats[key].astype(str).map(self.prior_count_maps[key]).fillna(0.0)).astype("float32")
            num_cols += [prior_name, cnt_name]

        X_num_df = f[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        X_cat_df = cats[self.cat_cols].fillna("nan").astype(str)
        if return_frame:
            return {"num": X_num_df, "cat": X_cat_df, "num_cols": list(X_num_df.columns), "cat_cols": list(X_cat_df.columns)}
        return X_num_df.to_numpy(np.float32), X_cat_df.to_numpy(dtype=str)


def stratified_cap(df, y, cap, seed):
    """Stratified subsample of `cap` rows (each class keeps its share), sorted back into the original row order."""
    if cap is None or len(df) <= cap:
        return df, y
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pieces = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        n = max(1, int(round(cap * len(idx) / len(y))))
        pieces.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    keep = np.sort(np.concatenate(pieces))
    return df.iloc[keep].reset_index(drop=True), y[keep]
