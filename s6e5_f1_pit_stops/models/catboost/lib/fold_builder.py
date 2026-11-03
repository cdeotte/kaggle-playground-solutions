"""Per-fold feature builder of ../train_fold_builder.py.

FoldFeatureBuilder.fit(X_train, X_orig, y_orig, X_test) learns, from the outer fold's training rows + its original-row
subset + the test rows: numeric medians, quantile edges, category frequencies, anchor-group statistics,
and (from the original subset and its labels only) smoothed original priors. transform(df) then builds the same columns
for any frame:
- the raw columns (numerics median-filled), low-cardinality numerics as categories (<col>_cat);
- row features: tyre-age / stint / degradation / delta ratios, progress left, pit-window score, interaction products;
- tyre-life / lap / progress / position bins; rounded (1-3 decimals), fractional-part and 50 / 100-quantile categories
  of lap time, lap-time delta, degradation and race progress; 10 categorical crosses and, with use_pair_grid, all 36
  pairs of 9 keys;
- frequency of every categorical column, mean / std / count of 7 numerics per 9 anchor keys and the residual to the
  group mean, original prior (smoothing 40) and log count of 16 keys, and a source flag (1 on original rows).
The string columns of cat_cols_ are CatBoost-native categoricals."""
import numpy as np
import pandas as pd


def clean_name(name):
    return (
        str(name)
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def to_num(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")
    return pd.to_numeric(df[col], errors="coerce").astype("float32")


def safe_div(a, b, eps=1e-6):
    return np.asarray(a, dtype="float32") / (np.asarray(b, dtype="float32") + eps)


def make_key(df, cols):
    if isinstance(cols, str):
        return df[cols].astype("string").fillna("NA").astype(str)
    out = df[cols[0]].astype("string").fillna("NA").astype(str)
    for col in cols[1:]:
        out = out + "|" + df[col].astype("string").fillna("NA").astype(str)
    return out


def fit_smooth_target_map(keys, y, smooth=20.0):
    keys = pd.Series(keys).astype("string").fillna("NA").astype(str)
    y = pd.Series(y).astype("float32").reset_index(drop=True)
    global_mean = float(y.mean())
    stats = pd.DataFrame({"key": keys.reset_index(drop=True), "y": y}).groupby("key")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
    return enc.astype("float32"), global_mean


class FoldFeatureBuilder:
    def __init__(self, feature_cols, object_cats, low_card_numeric_cols, numeric_like_cols, use_pair_grid):
        self.feature_cols = feature_cols
        self.object_cats = object_cats
        self.low_card_numeric_cols = low_card_numeric_cols
        self.numeric_like_cols = numeric_like_cols
        self.use_pair_grid = use_pair_grid
        self.core_nums = [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"] if c in feature_cols]
        self.cat_cols_ = []

    def fit(self, X_train, X_orig, y_orig, X_test):
        ref = pd.concat([X_train[self.feature_cols], X_orig[self.feature_cols], X_test[self.feature_cols]], axis=0, ignore_index=True)
        self.medians_ = {c: float(to_num(ref, c).median()) for c in self.numeric_like_cols}
        for c, v in list(self.medians_.items()):
            if not np.isfinite(v):
                self.medians_[c] = 0.0

        self.q_edges_ = {}
        q_cols = list(dict.fromkeys(self.core_nums + ["TyreLife", "LapNumber", "Position"]))
        for col in q_cols:
            if col not in ref.columns:
                continue
            values = to_num(ref, col).replace([np.inf, -np.inf], np.nan).dropna().values
            if len(values) < 10:
                continue
            for bins in [20, 50, 100]:
                qs = np.linspace(0, 1, bins + 1)
                edges = np.unique(np.nanquantile(values, qs))
                if len(edges) > 2:
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    self.q_edges_[(col, bins)] = edges.astype("float32")

        pre = self._make_core(ref)
        self.base_cat_cols_ = [c for c in pre.columns if c.endswith("_cat") or c in self.object_cats]
        standalone_bins = {"TyreLifeBin", "LapBin", "ProgressBin", "PositionBin"}
        combo_candidates = [
            c for c in pre.columns
            if c in standalone_bins or c.startswith("Pair_") or any(token in c for token in ["_x_", "_r", "_frac", "_q", "_combo"])
        ]
        self.cat_cols_ = list(dict.fromkeys([c for c in self.base_cat_cols_ + combo_candidates if c in pre.columns and pre[c].dtype == "object"]))

        self.freq_maps_ = {}
        for col in self.cat_cols_:
            vc = pre[col].astype("string").fillna("NA").astype(str).value_counts(dropna=False)
            self.freq_maps_[col] = (vc / max(len(pre), 1)).astype("float32")

        self.group_stat_maps_ = {}
        stat_keys = [c for c in [
            "Race", "Compound", "Driver", "Race_x_Compound", "Race_x_Year",
            "Compound_x_TyreLifeBin", "Race_x_Compound_x_TyreLifeBin",
            "ProgressBin_x_Compound", "LapBin_x_Race",
        ] if c in pre.columns]
        stat_nums = [c for c in self.core_nums + ["TyreLife", "LapNumber", "Position"] if c in pre.columns]
        for key in stat_keys:
            for num in stat_nums:
                values = to_num(pre, num).fillna(self.medians_.get(num, 0.0))
                tmp = pd.DataFrame({"key": pre[key].astype(str), "v": values})
                grp = tmp.groupby("key")["v"].agg(["mean", "std", "count"])
                self.group_stat_maps_[(key, num, "mean")] = grp["mean"].astype("float32")
                self.group_stat_maps_[(key, num, "std")] = grp["std"].fillna(0).astype("float32")
                self.group_stat_maps_[(key, num, "count")] = grp["count"].astype("float32")

        self.orig_prior_maps_ = {}
        if len(X_orig):
            orig_pre = self._make_core(X_orig[self.feature_cols])
            prior_keys = [c for c in [
                "Race", "Compound", "Driver", "Year_cat", "Stint_cat", "TyreLifeBin",
                "LapBin", "ProgressBin", "Race_x_Compound", "Driver_x_Compound",
                "Race_x_Year", "Race_x_Stint", "Compound_x_TyreLifeBin",
                "Race_x_Compound_x_TyreLifeBin", "ProgressBin_x_Compound", "LapBin_x_Race",
            ] if c in orig_pre.columns]
            y_orig = pd.Series(y_orig).astype("int8").reset_index(drop=True)
            for key in prior_keys:
                enc, global_mean = fit_smooth_target_map(orig_pre[key], y_orig, smooth=40.0)
                counts = orig_pre[key].astype(str).value_counts(dropna=False).astype("float32")
                self.orig_prior_maps_[key] = (enc, counts, global_mean)
        return self

    def _add_qbin(self, f, col, bins, name):
        if (col, bins) in self.q_edges_:
            values = to_num(f, col).fillna(self.medians_.get(col, 0.0)).values
            f[name] = np.digitize(values, self.q_edges_[(col, bins)][1:-1]).astype("int16").astype(str)
        else:
            f[name] = "NA"

    def _make_core(self, df):
        f = df[self.feature_cols].copy()
        for col in self.numeric_like_cols:
            f[col] = to_num(f, col).fillna(self.medians_.get(col, 0.0))
        for col in self.object_cats:
            if col in f.columns:
                f[col] = f[col].astype("string").fillna("NA").astype(str)
        for col in self.low_card_numeric_cols:
            if col in f.columns:
                f[f"{clean_name(col)}_cat"] = to_num(f, col).round(0).astype("Int64").astype("string").fillna("NA").astype(str)

        lap = to_num(f, "LapNumber").fillna(0)
        tyre = to_num(f, "TyreLife").fillna(0)
        stint = to_num(f, "Stint").fillna(0)
        pos = to_num(f, "Position").fillna(0)
        delta = to_num(f, "LapTime_Delta").fillna(0)
        deg = to_num(f, "Cumulative_Degradation").fillna(0)
        laptime = to_num(f, "LapTime (s)").fillna(0)
        progress = to_num(f, "RaceProgress").fillna(0)

        f["TyreAgeRatio"] = safe_div(tyre, lap + 1)
        f["StintLapRatio"] = safe_div(tyre, stint + 1)
        f["ProgressLeft"] = 1.0 - progress
        f["DegPerTyreLap"] = safe_div(deg, tyre + 1)
        f["DeltaPerTyreLap"] = safe_div(delta, tyre + 1)
        f["LapTimePerProgress"] = safe_div(laptime, progress + 0.02)
        f["PositionPressure"] = pos + to_num(f, "Position_Change").fillna(0)
        f["PitWindowCore"] = (tyre >= 8).astype("int8") + (tyre >= 14).astype("int8") + (progress >= 0.35).astype("int8") + (progress >= 0.65).astype("int8")
        f["LateRaceOldTyre"] = f["ProgressLeft"] * np.log1p(np.maximum(tyre, 0))
        f["DegDeltaProduct"] = deg * delta
        f["LapTimeMinusDelta"] = laptime - delta

        f["TyreLifeBin"] = pd.cut(tyre, bins=[-1, 2, 5, 8, 12, 16, 22, 30, 1000], labels=False).astype("Int64").astype("string").fillna("NA").astype(str)
        f["LapBin"] = pd.cut(lap, bins=[-1, 5, 10, 15, 25, 35, 50, 75, 1000], labels=False).astype("Int64").astype("string").fillna("NA").astype(str)
        f["ProgressBin"] = pd.cut(progress, bins=[-0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01], labels=False).astype("Int64").astype("string").fillna("NA").astype(str)
        f["PositionBin"] = pd.cut(pos, bins=[-1, 3, 6, 10, 14, 18, 25, 1000], labels=False).astype("Int64").astype("string").fillna("NA").astype(str)

        for col in self.core_nums:
            cname = clean_name(col)
            values = to_num(f, col).fillna(self.medians_.get(col, 0.0))
            f[f"{cname}_r1"] = values.round(1).astype("string").fillna("NA").astype(str)
            f[f"{cname}_r2"] = values.round(2).astype("string").fillna("NA").astype(str)
            f[f"{cname}_r3"] = values.round(3).astype("string").fillna("NA").astype(str)
            f[f"{cname}_frac100"] = np.floor((np.abs(values) % 1) * 100).astype("int16").astype(str)
            self._add_qbin(f, col, 50, f"{cname}_q50")
            self._add_qbin(f, col, 100, f"{cname}_q100")

        combos = [
            ("Race_x_Compound", ["Race", "Compound"]),
            ("Driver_x_Compound", ["Driver", "Compound"]),
            ("Driver_x_Race", ["Driver", "Race"]),
            ("Race_x_Year", ["Race", "Year_cat"]),
            ("Race_x_Stint", ["Race", "Stint_cat"]),
            ("Compound_x_TyreLifeBin", ["Compound", "TyreLifeBin"]),
            ("Race_x_Compound_x_TyreLifeBin", ["Race", "Compound", "TyreLifeBin"]),
            ("ProgressBin_x_Compound", ["ProgressBin", "Compound"]),
            ("LapBin_x_Race", ["LapBin", "Race"]),
            ("Driver_x_Race_x_Stint", ["Driver", "Race", "Stint_cat"]),
        ]
        for name, cols in combos:
            if all(c in f.columns for c in cols):
                f[name] = make_key(f, cols)
        if self.use_pair_grid:
            pair_cols = [c for c in ["Driver", "Compound", "Race", "Year_cat", "Stint_cat", "TyreLifeBin", "LapBin", "ProgressBin", "PositionBin"] if c in f.columns]
            for i, a in enumerate(pair_cols):
                for b in pair_cols[i + 1:]:
                    f[f"Pair_{a}_x_{b}"] = make_key(f, [a, b])
        return f

    def transform(self, df, source_flag=0):
        f = self._make_core(df)
        for col in self.cat_cols_:
            if col in f.columns and col in self.freq_maps_:
                vals = f[col].astype("string").fillna("NA").astype(str)
                f[f"freq_{clean_name(col)}"] = vals.map(self.freq_maps_[col]).fillna(0).astype("float32")
        for (key, num, stat), mp in self.group_stat_maps_.items():
            if key not in f.columns or num not in f.columns:
                continue
            vals = f[key].astype("string").fillna("NA").astype(str)
            out_name = f"grp_{clean_name(key)}_{clean_name(num)}_{stat}"
            f[out_name] = vals.map(mp).fillna(0).astype("float32")
            if stat == "mean":
                f[f"resid_{clean_name(key)}_{clean_name(num)}"] = to_num(f, num).fillna(self.medians_.get(num, 0.0)) - f[out_name]
        for key, (enc, counts, global_mean) in self.orig_prior_maps_.items():
            if key not in f.columns:
                continue
            vals = f[key].astype("string").fillna("NA").astype(str)
            f[f"orig_prior_{clean_name(key)}"] = vals.map(enc).fillna(global_mean).astype("float32")
            f[f"orig_logcount_{clean_name(key)}"] = np.log1p(vals.map(counts).fillna(0).astype("float32"))
        f["SourceFlag"] = np.float32(source_flag)
        for col in self.cat_cols_:
            if col in f.columns:
                f[col] = f[col].astype("string").fillna("NA").astype(str)
        return f


def align_columns(X_train, X_val, X_test):
    cols = list(X_train.columns)
    for X in [X_val, X_test]:
        for c in cols:
            if c not in X.columns:
                X[c] = 0
    return X_train[cols], X_val[cols], X_test[cols]


def finalize_numeric(X_train, X_val, X_test, cat_cols):
    """Non-categorical columns -> float32, non-finite values filled with the training median."""
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    for c in X_train.columns:
        if c in cat_cols:
            continue
        tr = pd.to_numeric(X_train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = float(tr.median()) if np.isfinite(tr.median()) else 0.0
        X_train[c] = tr.fillna(med).astype("float32")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")
    return X_train, X_val, X_test, cat_cols


def prepare_cat(X_train, X_val, X_test, cat_cols):
    X_train, X_val, X_test, cat_cols = finalize_numeric(X_train, X_val, X_test, cat_cols)
    for c in cat_cols:
        X_train[c] = X_train[c].astype("string").fillna("NA").astype(str)
        X_val[c] = X_val[c].astype("string").fillna("NA").astype(str)
        X_test[c] = X_test[c].astype("string").fillna("NA").astype(str)
    return X_train, X_val, X_test, cat_cols
