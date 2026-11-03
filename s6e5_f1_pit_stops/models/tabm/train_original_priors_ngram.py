#!/usr/bin/env python
"""TabM-mini (pytabkit) on fold-local original-data priors, distribution and digit features with nested target encodings
(pair tabm_original_priors_ngram_te).

Columns: categoricals = Driver, Compound, Race and the numerics with < 100 values in train (10); numerics = LapTime (s),
LapTime_Delta, Cumulative_Degradation, RaceProgress. Per outer fold, with the fold-matched original subset (the original rows
outside the same fold of StratifiedKFold(5, seed 42) on the original target) as the reference; the original rows are not
training rows:
  - frequency of every numeric value over the fold's training rows + original subset + test rows
  - degradation / lap-time relations (degradation minus TyreLife x LapTime_Delta, ratios) and race-context flags
  - original target mean of every categorical and numeric value
  - Cumulative_Degradation against the original rows: percentile ranks among all / positive / negative rows (and
    within Compound, Race), z-scores, distances to the positive / negative quartiles; LapTime and degradation residuals
    to the original Compound / Race means
  - digit features of LapTime (s), LapTime_Delta, Cumulative_Degradation (first / second / last digit, mod 10 / 100,
    number of digits, rounding deviations, fractional part, sign)
  - nested target encodings (target mean; 5-fold inner out-of-fold values on the training rows, seeds 42 + fold and
    142 + fold; validation / test rows from all training rows) of the 14 base columns (numerics as exact-value strings) and
    of the 15 bigrams and 4 trigrams of Driver, Race, Compound, Year, PitStop, Stint
Numerics are standardized on the fold's training rows; categoricals ordinal-encoded (unseen -1).
Model: TabM_D_Classifier, tabm-mini-normal, k 32, 3 blocks x 256, pwl embeddings d 24, dropout 0.2, lr 1e-3, weight decay
1e-3, batch 512, up to 50 epochs with patience 10 on the validation fold; seed 42 + fold. Test = mean of the 5 fold models.
Usage: python train_original_priors_ngram.py"""
import gc
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, orig_folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "tabm_original_priors_ngram_te"
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
CARDINALITY_CAT_THRESHOLD = 100
HIGH_CARD_NUMS = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation"]
PRIMARY_DIST_COL = "Cumulative_Degradation"

TABM_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbosity": 0,
    "arch_type": "tabm-mini-normal",
    "tabm_k": 32,
    "num_emb_type": "pwl",
    "d_embedding": 24,
    "batch_size": 512,
    "lr": 1e-3,
    "n_epochs": 50,
    "dropout": 0.2,
    "d_block": 256,
    "n_blocks": 3,
    "patience": 10,
    "weight_decay": 1e-3,
    "random_state": 42,
}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def clean_name(name):
    out = str(name).replace(" ", "_").replace("(", "").replace(")", "")
    out = out.replace("/", "_").replace("-", "m").replace("+", "p").replace(".", "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")


def safe_div(num, den, eps=1e-6):
    num = safe_num(num)
    den = safe_num(den)
    safe_den = den.where(den.abs() > eps, np.sign(den).replace(0, 1) * eps)
    out = num / safe_den
    return out.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")


def pctrank_against(values, reference):
    reference = np.asarray(reference, dtype="float32")
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0:
        return np.zeros(len(values), dtype="float32")
    ref_sorted = np.sort(reference)
    values = np.asarray(values, dtype="float32")
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def zscore_against(values, reference):
    reference = np.asarray(reference, dtype="float32")
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0:
        return np.zeros(len(values), dtype="float32")
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((np.asarray(values, dtype="float32") - mu) / sigma).astype("float32")


def first_digit_from_abs(values):
    arr = np.floor(np.abs(np.asarray(values, dtype="float32"))).astype("int64")
    out = np.zeros(len(arr), dtype="int16")
    for i, v in enumerate(arr):
        s = str(v)
        out[i] = int(s[0]) if s else 0
    return out


def second_digit_from_abs(values):
    arr = np.floor(np.abs(np.asarray(values, dtype="float32"))).astype("int64")
    out = np.zeros(len(arr), dtype="int16")
    for i, v in enumerate(arr):
        s = str(v)
        out[i] = int(s[1]) if len(s) > 1 else 0
    return out


def num_digits_from_abs(values):
    arr = np.floor(np.abs(np.asarray(values, dtype="float32"))).astype("int64")
    return np.array([len(str(v)) for v in arr], dtype="int16")


def add_digit_features(df, cols):
    new_cols = []
    for col in cols:
        safe = clean_name(col)
        v = safe_num(df[col])
        abs_v = v.abs()
        floor_abs = np.floor(abs_v).astype("int64")
        nd = num_digits_from_abs(v)

        specs = {
            f"{safe}_first_digit": first_digit_from_abs(v),
            f"{safe}_last_digit": (floor_abs % 10).astype("int16"),
            f"{safe}_second_digit": second_digit_from_abs(v),
            f"{safe}_mod10": (floor_abs % 10).astype("float32"),
            f"{safe}_mod100": (floor_abs % 100).astype("float32"),
            f"{safe}_num_digits": nd.astype("float32"),
            f"{safe}_is_multiple_10": ((floor_abs % 10) == 0).astype("float32"),
            f"{safe}_is_multiple_100": ((floor_abs % 100) == 0).astype("float32"),
            f"{safe}_rounded_10": (np.round(v / 10) * 10).astype("float32"),
            f"{safe}_rounded_100": (np.round(v / 100) * 100).astype("float32"),
            f"{safe}_fractional": (abs_v - np.floor(abs_v)).astype("float32"),
            f"{safe}_dev_from_round10": np.abs(v - np.round(v / 10) * 10).astype("float32"),
            f"{safe}_dev_from_round100": np.abs(v - np.round(v / 100) * 100).astype("float32"),
            f"{safe}_per_digit": safe_div(v, pd.Series(nd, index=df.index) + 0.001),
            f"{safe}_sign": np.sign(v).astype("float32"),
        }
        for name, values in specs.items():
            df[name] = np.asarray(values, dtype="float32")
            new_cols.append(name)
    return new_cols


def add_ngram_columns(df, top_cols):
    bigram_cols, trigram_cols = [], []
    for c1, c2 in combinations(top_cols, 2):
        name = f"BG_{c1}_{c2}"
        df[name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        bigram_cols.append(name)
    for c1, c2, c3 in combinations(top_cols[:4], 3):
        name = f"TG_{c1}_{c2}_{c3}"
        df[name] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
        trigram_cols.append(name)
    return bigram_cols + trigram_cols


def add_fold_base_features(frames, orig_ref, y_orig_ref, cats, nums, core_nums, top_ngram_cols, primary_col):
    """Adds the fold's features to the frames (train / val / test) with the fold-matched original subset as reference."""
    new_nums = []
    y_orig_ref = pd.Series(np.asarray(y_orig_ref), index=orig_ref.index, name=TARGET)
    orig_work = orig_ref.copy()
    orig_work[TARGET] = y_orig_ref
    fallback_orig = float(y_orig_ref.mean()) if len(y_orig_ref) else 0.5

    # [1/7] frequency encoding over the fold's training rows + original subset + test rows
    for col in nums:
        ref_values = pd.concat([frames["train"][col], orig_ref[col], frames["test"][col]], axis=0)
        freq = ref_values.value_counts(normalize=True, dropna=False)
        for df in frames.values():
            df[f"FREQ_{clean_name(col)}"] = df[col].map(freq).fillna(0).astype("float32")
        new_nums.append(f"FREQ_{clean_name(col)}")

    # [2/7] degradation / lap-time relations
    for df in frames.values():
        lap = safe_num(df["LapTime (s)"])
        delta = safe_num(df["LapTime_Delta"])
        deg = safe_num(df["Cumulative_Degradation"])
        tyre = safe_num(df["TyreLife"]).clip(lower=1)
        race_progress = safe_num(df["RaceProgress"])
        lap_number = safe_num(df["LapNumber"]).clip(lower=1)

        df["degradation_deviation"] = (deg - tyre * delta).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
        df["lap_to_degradation_ratio"] = safe_div(lap, deg.abs() + 1)
        df["avg_degradation_per_tyre"] = safe_div(deg, tyre + 1)
        df["lap_delta_ratio"] = safe_div(delta, lap.abs() + 1)
        df["degradation_progress_ratio"] = safe_div(deg, race_progress + 1e-3)
        df["lap_per_lapnumber"] = safe_div(lap, lap_number)
    new_nums += [
        "degradation_deviation", "lap_to_degradation_ratio", "avg_degradation_per_tyre",
        "lap_delta_ratio", "degradation_progress_ratio", "lap_per_lapnumber",
    ]

    # [3/7] race-context counts
    for df in frames.values():
        tyre = safe_num(df["TyreLife"])
        lap_number = safe_num(df["LapNumber"])
        position_change = safe_num(df["Position_Change"])
        race_progress = safe_num(df["RaceProgress"])
        df["pit_context_count"] = (
            (safe_num(df["PitStop"]) > 0).astype("int8")
            + (safe_num(df["Stint"]) > 1).astype("int8")
            + (tyre > 20).astype("int8")
            + (position_change != 0).astype("int8")
            + (race_progress > 0.5).astype("int8")
            + (lap_number > 30).astype("int8")
        ).astype("float32")
        df["has_position_change"] = (position_change != 0).astype("float32")
        df["is_late_race"] = (race_progress > 0.65).astype("float32")
    new_nums += ["pit_context_count", "has_position_change", "is_late_race"]

    # [4/7] original target mean of every value (fold-matched original subset)
    for col in cats + nums:
        name = f"ORIG_proba_{clean_name(col)}"
        tmp = orig_work.groupby(col, observed=False)[TARGET].mean()
        for df in frames.values():
            df[name] = df[col].map(tmp).fillna(fallback_orig).astype("float32")
        new_nums.append(name)

    # [5/7] position of the primary numeric within the original distributions
    orig_pos = orig_work.loc[orig_work[TARGET] == 1, primary_col].astype("float32").values
    orig_neg = orig_work.loc[orig_work[TARGET] == 0, primary_col].astype("float32").values
    orig_all = orig_work[primary_col].astype("float32").values
    compound_lap_mean = orig_work.groupby("Compound", observed=False)["LapTime (s)"].mean()
    race_deg_mean = orig_work.groupby("Race", observed=False)[primary_col].mean()

    for df in frames.values():
        vals = safe_num(df[primary_col]).values
        df[f"pctrank_negative_{clean_name(primary_col)}"] = pctrank_against(vals, orig_neg)
        df[f"pctrank_positive_{clean_name(primary_col)}"] = pctrank_against(vals, orig_pos)
        df[f"pctrank_orig_{clean_name(primary_col)}"] = pctrank_against(vals, orig_all)
        df[f"zscore_positive_gap_{clean_name(primary_col)}"] = (
            np.abs(zscore_against(vals, orig_pos)) - np.abs(zscore_against(vals, orig_neg))
        ).astype("float32")
        df[f"zscore_negative_{clean_name(primary_col)}"] = zscore_against(vals, orig_neg)
        df[f"pctrank_positive_gap_{clean_name(primary_col)}"] = (
            pctrank_against(vals, orig_pos) - pctrank_against(vals, orig_neg)
        ).astype("float32")
        df["resid_Compound_LapTime"] = (
            safe_num(df["LapTime (s)"]) - df["Compound"].map(compound_lap_mean).fillna(0).astype("float32")
        ).astype("float32")
        df["resid_Race_Degradation"] = (
            safe_num(df[primary_col]) - df["Race"].map(race_deg_mean).fillna(0).astype("float32")
        ).astype("float32")

        for cat_col, out_col in [("Compound", "cond_pctrank_Compound_Degradation"), ("Race", "cond_pctrank_Race_Degradation")]:
            cond_vals = np.zeros(len(df), dtype="float32")
            for cat_value in orig_work[cat_col].dropna().unique():
                mask = df[cat_col] == cat_value
                ref = orig_work.loc[orig_work[cat_col] == cat_value, primary_col].astype("float32").values
                if mask.sum() and len(ref):
                    cond_vals[mask.values] = pctrank_against(df.loc[mask, primary_col].astype("float32").values, ref)
            df[out_col] = cond_vals

    dist_cols = [
        f"pctrank_negative_{clean_name(primary_col)}", f"zscore_positive_gap_{clean_name(primary_col)}",
        f"pctrank_positive_gap_{clean_name(primary_col)}", "resid_Compound_LapTime", "resid_Race_Degradation",
        "cond_pctrank_Compound_Degradation", f"zscore_negative_{clean_name(primary_col)}",
        f"pctrank_orig_{clean_name(primary_col)}", f"pctrank_positive_{clean_name(primary_col)}",
        "cond_pctrank_Race_Degradation",
    ]
    new_nums += dist_cols

    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        pos_q = np.quantile(orig_pos, q_val) if len(orig_pos) else 0.0
        neg_q = np.quantile(orig_neg, q_val) if len(orig_neg) else 0.0
        for df in frames.values():
            df[f"dist_To_pos_{q_label}"] = np.abs(safe_num(df[primary_col]) - pos_q).astype("float32")
            df[f"dist_To_neg_{q_label}"] = np.abs(safe_num(df[primary_col]) - neg_q).astype("float32")
            df[f"qdist_gap_To_{q_label}"] = (df[f"dist_To_neg_{q_label}"] - df[f"dist_To_pos_{q_label}"]).astype("float32")
        new_nums += [f"qdist_gap_To_{q_label}", f"dist_To_pos_{q_label}", f"dist_To_neg_{q_label}"]

    # [6/7] digit features of the three high-cardinality numerics
    digit_cols = None
    for df in frames.values():
        cols_added = add_digit_features(df, core_nums)
        if digit_cols is None:
            digit_cols = cols_added
    new_nums += digit_cols

    # [7/7] n-gram keys and numeric-as-string keys for the target encodings
    ngram_cols = None
    for df in frames.values():
        cols_added = add_ngram_columns(df, top_ngram_cols)
        if ngram_cols is None:
            ngram_cols = cols_added
        for col in nums:
            df[f"CAT_{clean_name(col)}"] = df[col].astype(str)

    te_columns = [f"CAT_{clean_name(col)}" for col in nums] + cats
    new_nums = list(dict.fromkeys(new_nums))
    return frames, new_nums, te_columns, ngram_cols


def add_inner_cv_mean_encoding(X_tr, y_tr, X_val, X_test, columns, prefix, n_splits=5, seed=42):
    y_tr = np.asarray(y_tr)
    global_mean = float(np.mean(y_tr))
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    feature_names = [f"{prefix}_{clean_name(col)}" for col in columns]

    for df in [X_tr, X_val, X_test]:
        for name in feature_names:
            df[name] = global_mean

    X_tr_target = X_tr.copy()
    X_tr_target[TARGET] = y_tr

    for inner_tr_idx, inner_val_idx in inner.split(X_tr, y_tr):
        ref = X_tr_target.iloc[inner_tr_idx]
        hold = X_tr.iloc[inner_val_idx]
        for col, name in zip(columns, feature_names):
            mapping = ref.groupby(col, observed=False)[TARGET].mean()
            X_tr.loc[X_tr.index[inner_val_idx], name] = hold[col].map(mapping).fillna(global_mean).values

    full_ref = X_tr.copy()
    full_ref[TARGET] = y_tr
    for col, name in zip(columns, feature_names):
        mapping = full_ref.groupby(col, observed=False)[TARGET].mean()
        X_val[name] = X_val[col].map(mapping).fillna(global_mean).astype("float32")
        X_test[name] = X_test[col].map(mapping).fillna(global_mean).astype("float32")
        X_tr[name] = X_tr[name].fillna(global_mean).astype("float32")

    return feature_names


def load_data():
    train, test, orig = read_train(), read_test(), read_original()
    orig[TARGET] = pd.to_numeric(orig[TARGET]).astype("int8")
    train[TARGET] = pd.to_numeric(train[TARGET]).astype("int8")
    feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
    orig = orig[feature_cols + [TARGET]].copy()
    for df in [train, test, orig]:
        for col in feature_cols:
            if col not in df.columns:
                continue
            if col in train.select_dtypes(exclude=["object"]).columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return train, test, orig, feature_cols


def main():
    seed_everything(SEED)
    train, test, orig, feature_cols = load_data()
    y = train[TARGET].astype("int8").reset_index(drop=True)
    y_orig = orig[TARGET].astype("int8").reset_index(drop=True)

    base_cat_cols, base_num_cols = [], []
    for c in feature_cols:
        if train[c].dtype == "object":
            base_cat_cols.append(c)
        elif train[c].nunique(dropna=False) < CARDINALITY_CAT_THRESHOLD:
            base_cat_cols.append(c)
        else:
            base_num_cols.append(c)
    core_nums = [c for c in HIGH_CARD_NUMS if c in base_num_cols]
    num_cols = core_nums + [c for c in base_num_cols if c not in core_nums]
    top_cats_for_ngram = [c for c in ["Driver", "Race", "Compound", "Year", "PitStop", "Stint"] if c in base_cat_cols]
    primary_dist_col = PRIMARY_DIST_COL if PRIMARY_DIST_COL in feature_cols else core_nums[-1]
    print(f"categorical {base_cat_cols}\nnumeric {num_cols}", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    orig_fold_ids = orig_folds(y_orig.to_numpy())
    oof = np.zeros(len(train), dtype="float32")
    pred = np.zeros(len(test), dtype="float32")

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y), start=1):
        seed_everything(SEED + fold_i)
        orig_train_idx = np.where(orig_fold_ids != fold_i - 1)[0]
        X_tr = train.iloc[train_idx][feature_cols].reset_index(drop=True).copy()
        y_tr = y.iloc[train_idx].to_numpy()
        X_val = train.iloc[val_idx][feature_cols].reset_index(drop=True).copy()
        y_val = y.iloc[val_idx].to_numpy()
        X_te = test[feature_cols].reset_index(drop=True).copy()
        orig_tr = orig.iloc[orig_train_idx][feature_cols].reset_index(drop=True).copy()
        y_orig_tr = y_orig.iloc[orig_train_idx].to_numpy()

        frames = {"train": X_tr, "val": X_val, "test": X_te}
        frames, new_nums, te_columns, ngram_cols = add_fold_base_features(
            frames=frames, orig_ref=orig_tr, y_orig_ref=y_orig_tr, cats=base_cat_cols, nums=num_cols, core_nums=core_nums,
            top_ngram_cols=top_cats_for_ngram, primary_col=primary_dist_col)
        X_tr, X_val, X_te = frames["train"], frames["val"], frames["test"]

        te_feat_names = add_inner_cv_mean_encoding(X_tr, y_tr, X_val, X_te, te_columns, prefix="TE1", n_splits=INNER_FOLDS,
                                                   seed=SEED + fold_i)
        ng_te_feat_names = add_inner_cv_mean_encoding(X_tr, y_tr, X_val, X_te, ngram_cols, prefix="TE_ng",
                                                      n_splits=INNER_FOLDS, seed=SEED + 100 + fold_i)

        all_nums_final = list(dict.fromkeys(num_cols + new_nums + te_feat_names + ng_te_feat_names))
        all_cats_final = list(base_cat_cols)
        if fold_i == 1:
            print(f"numeric features {len(all_nums_final)}, categorical features {len(all_cats_final)}, "
                  f"base TE keys {len(te_columns)}, n-gram TE keys {len(ngram_cols)}", flush=True)

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(X_tr[all_cats_final].astype(str))
        X_tr_cat = encoder.transform(X_tr[all_cats_final].astype(str))
        X_val_cat = encoder.transform(X_val[all_cats_final].astype(str))
        X_te_cat = encoder.transform(X_te[all_cats_final].astype(str))

        for df in [X_tr, X_val, X_te]:
            df[all_nums_final] = df[all_nums_final].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr[all_nums_final])
        X_val_num = scaler.transform(X_val[all_nums_final])
        X_te_num = scaler.transform(X_te[all_nums_final])

        all_cols = all_nums_final + all_cats_final
        X_tr_final = pd.DataFrame(np.hstack([X_tr_num, X_tr_cat]), columns=all_cols)
        X_val_final = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=all_cols)
        X_te_final = pd.DataFrame(np.hstack([X_te_num, X_te_cat]), columns=all_cols)
        for c in all_cats_final:
            X_tr_final[c] = X_tr_final[c].astype(int)
            X_val_final[c] = X_val_final[c].astype(int)
            X_te_final[c] = X_te_final[c].astype(int)

        params = dict(TABM_PARAMS)
        params["random_state"] = SEED + fold_i
        model = TabM_D_Classifier(**params)
        model.fit(X_tr_final, y_tr, X_val=X_val_final, y_val=y_val, cat_col_names=all_cats_final)

        val_probs = model.predict_proba(X_val_final)[:, 1].astype("float32")
        test_probs = model.predict_proba(X_te_final)[:, 1].astype("float32")
        oof[val_idx] = val_probs
        pred += test_probs / N_FOLDS
        print(f"fold {fold_i - 1}: AUC {roc_auc_score(y_val, val_probs):.6f}", flush=True)

        del model, X_tr, X_val, X_te, X_tr_final, X_val_final, X_te_final
        del X_tr_num, X_val_num, X_te_num, X_tr_cat, X_val_cat, X_te_cat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, PAIR, np.clip(oof, 1e-7, 1 - 1e-7), np.clip(pred, 1e-7, 1 - 1e-7))


if __name__ == "__main__":
    main()
