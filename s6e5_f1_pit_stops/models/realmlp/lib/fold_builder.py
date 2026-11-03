"""Per-fold feature builder (FoldFeatureBuilder), nested target encodings and the RealMLP fold loop (run).

FoldFeatureBuilder.fit learns on the union of the fold's training rows, the fold's original rows and the test rows:
medians of the numeric columns (fill value), quantile edges (20 / 50 / 100 bins) of LapTime, LapTime_Delta,
Cumulative_Degradation, RaceProgress, TyreLife, LapNumber, Position, and on the core frame of that union:
- the relative frequency of every categorical column;
- anchor statistics: mean / std / count of LapTime, LapTime_Delta, Cumulative_Degradation, RaceProgress, TyreLife,
  LapNumber, Position per Race, Compound, Driver and 6 combination keys (+ the residual value - group mean);
- original priors: smoothed (m = 40) original target rate and log count of 16 keys, from the fold's original rows.
Core frame (_make_core): the 14 columns (low-cardinality numerics also as `<col>_cat` strings), 11 ratios / products /
pit-window counters, tyre-life / lap / progress / position bins, rounded values (1-3 decimals), hundredths and 50 / 100
quantile bins of the 4 continuous columns, 10 combination keys; optional race-stint sequence features (rows sorted by
Year, Race, Driver, Stint, LapNumber: position and size of the group, previous / next value and differences of 6
columns, two slopes) and optional pair grid (every pair of 9 categorical keys).
Nested target encodings (add_cv_target_encodings): the first te_limit candidate keys, each at every smoothing in
te_smooths; training rows get 5-fold inner out-of-fold encodings (inner seed seed + 1000 + fold + smoothing), the
validation and test rows the map of all training rows.

run(spec): per outer fold (the 5 folds of common.data), training rows = the fold's train rows + the original rows
outside the same fold of the original data (common.data.orig_folds); RealMLP fitted with the fold's validation rows as
validation set (early stopping on 1-AUC), seed 42 + fold (1-based). OOF = held-out predictions, test = mean of the 5 fold
models, both clipped to [1e-7, 1 - 1e-7]."""
import gc
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common.data import ID, N_FOLDS, TARGET, folds, orig_folds, read_original, read_test, read_train

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

SEED = 42
INNER_FOLDS = 5
CARDINALITY_CAT_THRESHOLD = 100


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def add_cv_target_encodings(X_train, y_train, X_val, X_test, key_cols, smooths, n_splits=5, seed=42):
    created = []
    y_train = pd.Series(y_train).astype("int8").reset_index(drop=True)
    for col in key_cols:
        if col not in X_train.columns:
            continue
        tr_key_all = make_key(X_train, col)
        val_key = make_key(X_val, col)
        test_key = make_key(X_test, col)
        for smooth in smooths:
            name = f"TE_{clean_name(col)}_s{smooth}"
            oof_enc = np.zeros(len(X_train), dtype="float32")
            inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + int(smooth))
            for inner_tr, inner_val in inner.split(X_train, y_train):
                enc_map, global_mean = fit_smooth_target_map(tr_key_all.iloc[inner_tr], y_train.iloc[inner_tr], smooth=smooth)
                oof_enc[inner_val] = tr_key_all.iloc[inner_val].map(enc_map).fillna(global_mean).astype("float32").values
            full_map, global_mean = fit_smooth_target_map(tr_key_all, y_train, smooth=smooth)
            X_train[name] = oof_enc
            X_val[name] = val_key.map(full_map).fillna(global_mean).astype("float32").values
            X_test[name] = test_key.map(full_map).fillna(global_mean).astype("float32").values
            created.append(name)
    return created


class FoldFeatureBuilder:
    def __init__(self, feature_cols, object_cats, low_card_numeric_cols, numeric_like_cols, use_sequence=False,
                 use_pair_grid=False, te_limit=14):
        self.feature_cols = feature_cols
        self.object_cats = object_cats
        self.low_card_numeric_cols = low_card_numeric_cols
        self.numeric_like_cols = numeric_like_cols
        self.use_sequence = use_sequence
        self.use_pair_grid = use_pair_grid
        self.te_limit = te_limit
        self.core_nums = [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"] if c in feature_cols]
        self.cat_cols_ = []
        self.te_candidates_ = []

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

        self.te_candidates_ = [c for c in [
            "Race_x_Compound", "Driver_x_Compound", "Race_x_Year", "Race_x_Stint",
            "Compound_x_TyreLifeBin", "Race_x_Compound_x_TyreLifeBin",
            "ProgressBin_x_Compound", "LapBin_x_Race", "Driver_x_Race_x_Stint",
            "LapTime_s_r2", "LapTime_Delta_r2", "Cumulative_Degradation_r2",
            "LapTime_s_q50", "LapTime_Delta_q50", "Cumulative_Degradation_q50",
        ] if c in pre.columns]
        if self.use_pair_grid:
            self.te_candidates_.extend([c for c in pre.columns if c.startswith("Pair_")])
        self.te_candidates_ = list(dict.fromkeys(self.te_candidates_))[: self.te_limit]
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
        if self.use_sequence:
            f = self._add_sequence_features(f)
        return f

    def _add_sequence_features(self, f):
        keys = [c for c in ["Year", "Race", "Driver", "Stint"] if c in f.columns]
        if len(keys) < 3 or "LapNumber" not in f.columns:
            return f
        sort_cols = [c for c in keys + ["LapNumber", "TyreLife", "RaceProgress"] if c in f.columns]
        order = f.sort_values(sort_cols, kind="mergesort").index
        tmp = f.loc[order, keys + [c for c in ["LapNumber", "TyreLife", "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position"] if c in f.columns]].copy()
        g = tmp.groupby(keys, sort=False, observed=True)
        f.loc[order, "SeqPosInStint"] = g.cumcount().astype("float32").values
        f.loc[order, "SeqGroupSize"] = g["LapNumber"].transform("size").astype("float32").values
        f["SeqFracInStint"] = safe_div(f["SeqPosInStint"], f["SeqGroupSize"])
        for col in [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position", "TyreLife"] if c in tmp.columns]:
            cname = clean_name(col)
            prev_v = g[col].shift(1)
            next_v = g[col].shift(-1)
            f.loc[order, f"{cname}_prev"] = prev_v.values
            f.loc[order, f"{cname}_next"] = next_v.values
            f[f"{cname}_dprev"] = to_num(f, col) - pd.to_numeric(f[f"{cname}_prev"], errors="coerce").fillna(to_num(f, col))
            f[f"{cname}_dnext"] = pd.to_numeric(f[f"{cname}_next"], errors="coerce").fillna(to_num(f, col)) - to_num(f, col)
        f["DegSlopePrev"] = safe_div(f.get("Cumulative_Degradation_dprev", 0), np.abs(f.get("TyreLife_dprev", 0)) + 1)
        f["LapDeltaSlopePrev"] = safe_div(f.get("LapTime_Delta_dprev", 0), np.abs(f.get("TyreLife_dprev", 0)) + 1)
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


def prepare_nn(X_train, X_val, X_test, cat_cols):
    """Numeric columns: inf -> NaN -> median of the training rows, float32. Categorical columns: strings."""
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    for c in X_train.columns:
        if c in cat_cols:
            continue
        tr = pd.to_numeric(X_train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = float(tr.median()) if np.isfinite(tr.median()) else 0.0
        X_train[c] = tr.fillna(med).astype("float32")
        X_val[c] = pd.to_numeric(X_val[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")
    for c in cat_cols:
        X_train[c] = X_train[c].astype("string").fillna("NA").astype(str)
        X_val[c] = X_val[c].astype("string").fillna("NA").astype(str)
        X_test[c] = X_test[c].astype("string").fillna("NA").astype(str)
    return X_train, X_val, X_test, cat_cols


def run(spec):
    """spec: model_params (RealMLP_TD_Classifier arguments), use_sequence, use_pair_grid, te_limit, te_smooths.
    Returns the (oof, test) probabilities."""
    from pytabkit import RealMLP_TD_Classifier

    seed_everything(SEED)
    train, test, orig = read_train(), read_test(), read_original()
    feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
    orig = orig[feature_cols + [TARGET]].copy()
    y = train[TARGET].astype("int8").reset_index(drop=True)
    y_orig = orig[TARGET].astype("int8").reset_index(drop=True)

    object_cats = [c for c in feature_cols if train[c].dtype == "object" or str(train[c].dtype).startswith("string")]
    numeric_like_cols = [c for c in feature_cols if c not in object_cats]
    low_card_numeric_cols = [c for c in numeric_like_cols if train[c].nunique(dropna=False) < CARDINALITY_CAT_THRESHOLD]
    fold_id, orig_fold_id = folds(y.to_numpy()), orig_folds(y_orig.to_numpy())

    oof = np.zeros(len(train), dtype="float32")
    test_pred = np.zeros(len(test), dtype="float32")
    for k in range(N_FOLDS):
        fold = k + 1
        tr_idx, val_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        orig_tr_idx = np.where(orig_fold_id != k)[0]
        seed_everything(SEED + fold)

        X_tr_raw = train.iloc[tr_idx][feature_cols].reset_index(drop=True)
        y_tr_syn = y.iloc[tr_idx].reset_index(drop=True)
        X_val_raw = train.iloc[val_idx][feature_cols].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        X_test_raw = test[feature_cols].reset_index(drop=True)
        X_orig_raw = orig.iloc[orig_tr_idx][feature_cols].reset_index(drop=True)
        y_orig_tr = y_orig.iloc[orig_tr_idx].reset_index(drop=True)

        builder = FoldFeatureBuilder(feature_cols, object_cats, low_card_numeric_cols, numeric_like_cols,
                                     use_sequence=spec["use_sequence"], use_pair_grid=spec["use_pair_grid"],
                                     te_limit=spec["te_limit"]).fit(X_tr_raw, X_orig_raw, y_orig_tr, X_test_raw)
        X_tr = builder.transform(X_tr_raw, source_flag=0)
        X_val = builder.transform(X_val_raw, source_flag=0)
        X_test_fold = builder.transform(X_test_raw, source_flag=0)
        X_orig = builder.transform(X_orig_raw, source_flag=1)

        X_train = pd.concat([X_tr, X_orig], axis=0, ignore_index=True)
        y_train = pd.concat([y_tr_syn, y_orig_tr], axis=0, ignore_index=True).astype("int8")
        te_created = []
        if builder.te_candidates_:
            te_created = add_cv_target_encodings(X_train, y_train, X_val, X_test_fold, key_cols=builder.te_candidates_,
                                                 smooths=spec["te_smooths"], n_splits=INNER_FOLDS, seed=SEED + 1000 + fold)
        X_train, X_val, X_test_fold = align_columns(X_train, X_val, X_test_fold)
        cat_cols = [c for c in builder.cat_cols_ if c in X_train.columns]
        if fold == 1:
            print(f"{X_train.shape[1]} features, {len(cat_cols)} categorical, {len(te_created)} target encodings", flush=True)

        X_fit, X_eval, X_tst, cat_cols = prepare_nn(X_train, X_val, X_test_fold, cat_cols)
        params = dict(spec["model_params"])
        params["random_state"] = SEED + fold
        model = RealMLP_TD_Classifier(**params)
        model.fit(X_fit, y_train, X_eval, y_val)
        val_pred = model.predict_proba(X_eval)[:, 1].astype("float32")
        fold_test = model.predict_proba(X_tst)[:, 1].astype("float32")

        oof[val_idx] = val_pred
        test_pred += fold_test / N_FOLDS
        print(f"fold {fold}: AUC {roc_auc_score(y_val, val_pred):.6f}", flush=True)
        del X_tr, X_val, X_test_fold, X_orig, X_train, model
        gc.collect()

    print(f"OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    return np.clip(oof, 1e-7, 1 - 1e-7).astype("float32"), np.clip(test_pred, 1e-7, 1 - 1e-7).astype("float32")
