#!/usr/bin/env python
"""Extra-trees LightGBM with a two-stage target encoding of category pairs, digit keys, roundings and bins
(pair lgbm_extra_trees_pairs_digits_two_stage_te).

Adapts masayakawamata's public LightGBM notebook of an earlier Playground competition:
https://www.kaggle.com/code/masayakawamata/s5e11-single-lgbm-tuned

MonthlyCharges / TotalCharges are read as their CSV text. Label-free features (train, test and original rows):
  - base: the 15 string categoricals, SeniorCitizen, tenure and the two charges as text
  - MonthlyCharges rounded to 1 / 10, TotalCharges rounded to 1 / 10 / 100
  - floor(x / b) and round(x / b): tenure b = 3, 6, 12; MonthlyCharges b = 5, 10; TotalCharges b = 100, 500, 1000
  - digits: 2 of tenure, 5 of round(100 x MonthlyCharges), 7 of round(100 x TotalCharges) (zero-padded); all digit
    pairs within each column and all digit triples of the two charges as keys
  - the 136 pairs of the 17 base columns other than the charge texts, and 35 crosses of 7 digits (the first 2 of
    MonthlyCharges, the first 3 of TotalCharges, both of tenure) with Contract, InternetService, PaymentMethod,
    OnlineSecurity, TechSupport as keys
  - original churn rate and original row count of each of the 19 base columns (unseen: -1)
Per fold: stage 1 replaces the 248 pair / digit-pair / digit-triple / cross keys by their mean target (out-of-fold over
an inner KFold(5, shuffled, seed 42) of the training rows; the validation and test rows get the means of all training
rows; unseen values the global mean) and drops the keys; stage 2 adds the same encoding of the charge texts, tenure,
SeniorCitizen, the digits, roundings and bins (39 columns, kept). The 17 remaining string columns are LightGBM
categoricals (one vocabulary over the fold's training, validation and test rows).
Model: extra_trees, 141 leaves, max_depth 7, lr 0.01, feature_fraction 0.17, colsample_bynode 0.925,
bagging_fraction 0.7, min_data_in_leaf 63, lambda_l1 3.3, lambda_l2 5.7, max_bin 1008, seed 6394 (bagging 6395,
feature 6396), 32 threads; up to 20,000 rounds with early stopping (200) on the validation fold.
Usage: python train_extra_trees_two_stage_te.py"""
import gc
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ORIGINAL_FILE, TARGET, folds  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.lightgbm.lib.kfold import stratified_fold_ids  # noqa: E402

warnings.simplefilter("ignore")

PAIR = "lgbm_extra_trees_pairs_digits_two_stage_te"
SEED = 42
N_SPLITS = 5
TEST_FOLDS = 15
N_SEEDS_PER_FOLD = 1
THREADS = 32
STRING_DTYPES = {"MonthlyCharges": "string", "TotalCharges": "string"}

PARAMS = {
    "objective": "binary",
    "n_estimators": 20000,
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 141,
    "max_depth": 7,
    "learning_rate": 0.01,
    "bagging_fraction": 0.7,
    "colsample_bynode": 0.925,
    "feature_fraction": 0.17,
    "min_data_in_leaf": 63,
    "lambda_l1": 3.3,
    "lambda_l2": 5.7,
    "n_jobs": THREADS,
    "verbose": -1,
    "extra_trees": True,
    "max_bin": 1008,
}


# ============================================================
# Utilities
# ============================================================
def clean_charge_string(s):
    s = s.astype("string")
    s = s.fillna("")
    s = s.str.strip()
    s = s.replace("", pd.NA)
    return s


def charge_str_to_float(s):
    s = clean_charge_string(s)
    return pd.to_numeric(s, errors="coerce").astype("float32")


def ensure_binary_target(df, target_col):
    if str(df[target_col].dtype) in ["object", "string", "category"]:
        df[target_col] = (
            df[target_col]
            .astype("string")
            .str.strip()
            .map({"Yes": 1, "No": 0, "1": 1, "0": 0})
            .astype("float32")
        )
    return df


def safe_str_cat(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing").str.strip()
    return df


def make_digit_columns(df, src_col, multiplier, max_len, out_prefix):
    if multiplier == "direct":
        temp_int = pd.to_numeric(df[src_col], errors="coerce").fillna(-1).round(0).astype(np.int64)
    else:
        temp_int = (pd.to_numeric(df[src_col], errors="coerce").fillna(-1) * multiplier).round(0).astype(np.int64)
    temp_int = temp_int.clip(lower=0)
    temp_str = temp_int.astype(str).str.zfill(max_len)
    created = []
    for i in range(max_len):
        new_col = f"{out_prefix}_DIGIT_{i + 1}"
        df[new_col] = temp_str.str[i].astype("int16")
        created.append(new_col)
    return created


def unique_keep_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def make_lgb_compatible(train_df, val_df, test_df, base_cat_cols):
    """String columns -> pandas categories with one vocabulary over train / val / test; the rest numeric."""
    cat_cols = [c for c in base_cat_cols if c in train_df.columns]
    extra_cat_cols = [c for c in train_df.columns if str(train_df[c].dtype) in ["object", "string", "category"]]
    cat_cols = sorted(set(cat_cols + extra_cat_cols))

    for c in cat_cols:
        all_vals = pd.concat([
            train_df[c].astype("string"),
            val_df[c].astype("string"),
            test_df[c].astype("string"),
        ], axis=0).fillna("missing")
        cats = pd.Index(pd.unique(all_vals))
        train_df[c] = pd.Categorical(train_df[c].astype("string").fillna("missing"), categories=cats)
        val_df[c] = pd.Categorical(val_df[c].astype("string").fillna("missing"), categories=cats)
        test_df[c] = pd.Categorical(test_df[c].astype("string").fillna("missing"), categories=cats)

    for df_ in [train_df, val_df, test_df]:
        for c in df_.columns:
            if c not in cat_cols:
                if str(df_[c].dtype) == "bool":
                    df_[c] = df_[c].astype("int8")
                elif not pd.api.types.is_numeric_dtype(df_[c]):
                    df_[c] = pd.to_numeric(df_[c], errors="coerce").astype("float32")
    return train_df, val_df, test_df, cat_cols


class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    """TE_<col>_<agg>: fit_transform encodes each row out-of-fold over KFold(cv, shuffle, seed 42) (the mean smoothed
    toward the fold mean with weight `smooth`; 'auto' = within / between variance ratio); transform uses the plain
    statistics of all fitted rows. Unseen values get the global statistic."""

    def __init__(self, cols_to_encode, aggs=("mean",), cv=5, smooth="auto", drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        temp_df = X.copy()
        temp_df["target"] = y.values if isinstance(y, pd.Series) else y
        y_series = pd.Series(y)
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = getattr(y_series, agg_func)()
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                self.mappings_[col][agg_func] = temp_df.groupby(col, dropna=False)["target"].agg(agg_func)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f"TE_{col}_{agg_func}"
                X_transformed[new_col_name] = X_transformed[col].map(self.mappings_[col][agg_func])
                X_transformed[new_col_name] = (
                    X_transformed[new_col_name].fillna(self.global_stats_[agg_func]).astype("float32")
                )
        if self.drop_original:
            X_transformed = X_transformed.drop(columns=self.cols_to_encode)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df_train = X_train.copy()
            temp_df_train["target"] = y_train.values

            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f"TE_{col}_{agg_func}"
                    fold_global_stat = getattr(pd.Series(y_train), agg_func)()
                    mapping = temp_df_train.groupby(col, dropna=False)["target"].agg(agg_func)

                    if agg_func == "mean":
                        counts = temp_df_train.groupby(col, dropna=False)["target"].count()
                        m = self.smooth
                        if self.smooth == "auto":
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col, dropna=False)["target"].var().mean()
                            if pd.notna(variance_between) and variance_between > 0 and pd.notna(avg_variance_within):
                                m = avg_variance_within / variance_between
                            else:
                                m = 0
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)

                    encoded_features.loc[X_val.index, new_col_name] = (
                        encoded_values.fillna(fold_global_stat).astype("float32")
                    )

        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
        if self.drop_original:
            X_transformed = X_transformed.drop(columns=self.cols_to_encode)
        return X_transformed


# ============================================================
# Features
# ============================================================
def build_features():
    train = pd.read_csv(DATA_DIR / "train.csv", dtype=STRING_DTYPES)
    test = pd.read_csv(DATA_DIR / "test.csv", dtype=STRING_DTYPES)
    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, dtype=STRING_DTYPES)
    print(f"train: {train.shape} | test: {test.shape} | original: {orig.shape}", flush=True)

    train = ensure_binary_target(train, TARGET)
    orig = ensure_binary_target(orig, TARGET)
    train[TARGET] = train[TARGET].astype("int8")
    orig[TARGET] = orig[TARGET].astype("int8")

    for df in [train, test, orig]:
        df["MonthlyCharges"] = clean_charge_string(df["MonthlyCharges"])
        df["TotalCharges"] = clean_charge_string(df["TotalCharges"])

    cats = [
        c for c in [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod",
        ] if c in train.columns
    ]
    for df in [train, test, orig]:
        safe_str_cat(df, cats)

    for df in [train, test, orig]:
        df["MonthlyCharges_num"] = charge_str_to_float(df["MonthlyCharges"])
        df["MonthlyCharges_str"] = df["MonthlyCharges"].fillna("missing")
        df["TotalCharges_num"] = charge_str_to_float(df["TotalCharges"])
        df["TotalCharges_str"] = df["TotalCharges"].fillna("missing")
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(-1).astype("int16")
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype("int8")

    base = list(cats) + ["SeniorCitizen", "tenure", "MonthlyCharges_str", "TotalCharges_str"]

    round_feats = []
    for col, levels in {
        "MonthlyCharges_num": {"1s": 0, "10s": -1},
        "TotalCharges_num": {"1s": 0, "10s": -1, "100s": -2},
    }.items():
        for suffix, level in levels.items():
            new_col = f"{col}_ROUND_{suffix}"
            round_feats.append(new_col)
            for df in [train, test, orig]:
                df[new_col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).round(level).astype("int32")

    binning = []
    bin_specs = {
        "tenure": [3, 6, 12],
        "MonthlyCharges_num": [5, 10],
        "TotalCharges_num": [100, 500, 1000],
    }
    for col, bins in bin_specs.items():
        for b in bins:
            floor_col = f"{col}_BIN_FLOOR_{b}"
            binning.append(floor_col)
            for df in [train, test, orig]:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(-1)
                df[floor_col] = np.floor_divide(vals.astype("float64"), b).astype("int32")
            round_col = f"{col}_BIN_ROUND_{b}"
            binning.append(round_col)
            for df in [train, test, orig]:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(-1)
                df[round_col] = np.round(vals / b).astype("int32")

    digit = []
    digit_specs = [
        ("tenure", "direct", 2, "tenure"),
        ("MonthlyCharges_num", 100, 5, "MonthlyCharges"),
        ("TotalCharges_num", 100, 7, "TotalCharges"),
    ]
    for src_col, multiplier, max_len, prefix in digit_specs:
        created_train = make_digit_columns(train, src_col, multiplier, max_len, prefix)
        make_digit_columns(test, src_col, multiplier, max_len, prefix)
        make_digit_columns(orig, src_col, multiplier, max_len, prefix)
        digit.extend(created_train)

    digit_inter, digit_inter_3way = [], []
    digit_groups = {
        "MonthlyCharges": [c for c in digit if c.startswith("MonthlyCharges_DIGIT_")],
        "TotalCharges": [c for c in digit if c.startswith("TotalCharges_DIGIT_")],
        "tenure": [c for c in digit if c.startswith("tenure_DIGIT_")],
    }
    for _, cols in digit_groups.items():
        for comb in combinations(cols, 2):
            new_col = "_".join(comb)
            digit_inter.append(new_col)
            for df in [train, test, orig]:
                df[new_col] = df[comb[0]].astype("string") + "_" + df[comb[1]].astype("string")
    for prefix in ["MonthlyCharges", "TotalCharges"]:
        cols = digit_groups[prefix]
        if len(cols) >= 3:
            for comb in combinations(cols, 3):
                new_col = "_".join(comb)
                digit_inter_3way.append(new_col)
                for df in [train, test, orig]:
                    df[new_col] = (
                        df[comb[0]].astype("string") + "_" +
                        df[comb[1]].astype("string") + "_" +
                        df[comb[2]].astype("string")
                    )

    inter = []
    te_base = [c for c in base if c not in ["MonthlyCharges_str", "TotalCharges_str"]]
    for col1, col2 in combinations(te_base, 2):
        new_col = f"{col1}_{col2}"
        inter.append(new_col)
        for df in [train, test, orig]:
            df[new_col] = df[col1].astype("string") + "_" + df[col2].astype("string")

    digit_cats = []
    selected_digit_cols = (
        [c for c in digit if c.startswith("MonthlyCharges_DIGIT_")][:2] +
        [c for c in digit if c.startswith("TotalCharges_DIGIT_")][:3] +
        [c for c in digit if c.startswith("tenure_DIGIT_")]
    )
    selected_cat_cols = [c for c in ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport"] if c in cats]
    for dcol in selected_digit_cols:
        for ccol in selected_cat_cols:
            new_col = f"{dcol}__{ccol}"
            digit_cats.append(new_col)
            for df in [train, test, orig]:
                df[new_col] = df[dcol].astype("string") + "_" + df[ccol].astype("string")

    orig_feats = []
    for col in base:
        mean_map = orig.groupby(col, dropna=False)[TARGET].mean()
        mean_name = f"orig_mean_{col}"
        mean_map.name = mean_name
        train = train.merge(mean_map, on=col, how="left")
        test = test.merge(mean_map, on=col, how="left")
        orig_feats.append(mean_name)

        count_name = f"orig_count_{col}"
        count_map = orig.groupby(col, dropna=False).size().reset_index(name=count_name)
        train = train.merge(count_map, on=col, how="left")
        test = test.merge(count_map, on=col, how="left")
        orig_feats.append(count_name)
    for df in [train, test]:
        for col in orig_feats:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("float32")

    features = base + orig_feats + inter + round_feats + digit + digit_inter + digit_inter_3way + digit_cats + binning
    te_feat = inter + digit_inter + digit_inter_3way + digit_cats
    base_te_col = ["MonthlyCharges_str", "TotalCharges_str", "tenure", "SeniorCitizen"] + digit + round_feats + binning

    features = unique_keep_order([c for c in features if c in train.columns and c in test.columns])
    te_feat = unique_keep_order([c for c in te_feat if c in features])
    base_te_col = unique_keep_order([c for c in base_te_col if c in features])
    print(f"features: {len(features)} | stage-1 keys: {len(te_feat)} | stage-2 columns: {len(base_te_col)}", flush=True)

    X = train[features].copy()
    y = train[TARGET].copy().astype("int8")
    X_test_full = test[features].copy()
    return X, y, X_test_full, cats, te_feat, base_te_col


# ============================================================
# Cross-validation
# ============================================================
def cross_validate(X, y, X_test_full, cats, te_feat, base_te_col, fold_ids, n_splits, seeds):
    oof_preds = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test_full), dtype=np.float64)

    for k in range(n_splits):
        train_idx = np.where(fold_ids != k)[0]
        val_idx = np.where(fold_ids == k)[0]
        X_train_fold = X.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()
        X_test_fold = X_test_full.copy()

        te1 = MeanTargetEncoder(cols_to_encode=te_feat, cv=5, smooth=0, aggs=("mean",), drop_original=True)
        X_train_fold_te = te1.fit_transform(X_train_fold, y_train_fold)
        X_val_fold_te = te1.transform(X_val_fold)
        X_test_fold_te = te1.transform(X_test_fold)

        te2_cols = [c for c in base_te_col if c in X_train_fold_te.columns]
        te2 = MeanTargetEncoder(cols_to_encode=te2_cols, cv=5, smooth=0, aggs=("mean",), drop_original=False)
        X_train_fold_te = te2.fit_transform(X_train_fold_te, y_train_fold)
        X_val_fold_te = te2.transform(X_val_fold_te)
        X_test_fold_te = te2.transform(X_test_fold_te)

        X_train_fold_te, X_val_fold_te, X_test_fold_te, cat_in_matrix = make_lgb_compatible(
            X_train_fold_te, X_val_fold_te, X_test_fold_te, cats)

        fold_seed_preds = []
        for seed_value in seeds:
            current_params = PARAMS.copy()
            current_params["random_state"] = seed_value
            current_params["bagging_seed"] = seed_value + 1
            current_params["feature_fraction_seed"] = seed_value + 2

            model = lgb.LGBMClassifier(**current_params)
            model.fit(
                X_train_fold_te,
                y_train_fold,
                eval_set=[(X_val_fold_te, y_val_fold)],
                eval_metric="auc",
                categorical_feature=cat_in_matrix,
                callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=1000)],
            )
            fold_seed_preds.append(model.predict_proba(X_val_fold_te)[:, 1])
            test_preds += model.predict_proba(X_test_fold_te)[:, 1]
            print(f"  seed {seed_value}: best_iteration {model.best_iteration_}", flush=True)
            del model
            gc.collect()

        fold_val_avg = np.mean(fold_seed_preds, axis=0)
        oof_preds[val_idx] = fold_val_avg
        print(f"fold {k + 1}/{n_splits}: AUC {roc_auc_score(y_val_fold, fold_val_avg):.6f} | "
              f"{X_train_fold_te.shape[1]} columns, {len(cat_in_matrix)} categorical", flush=True)

        del X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold
        del X_train_fold_te, X_val_fold_te, X_test_fold_te
        gc.collect()

    test_preds /= (n_splits * len(seeds))
    return oof_preds, test_preds


def main():
    X, y, X_test_full, cats, te_feat, base_te_col = build_features()

    random.seed(SEED)
    np.random.seed(SEED)
    seeds = random.choices(range(10000), k=N_SEEDS_PER_FOLD)
    print(f"seeds per fold: {seeds}", flush=True)

    oof, test_pred = cross_validate(X, y, X_test_full, cats, te_feat, base_te_col, folds(y.values), N_SPLITS, seeds)
    print(f"{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    if TEST_FOLDS != N_SPLITS:
        _, test_pred = cross_validate(X, y, X_test_full, cats, te_feat, base_te_col,
                                      stratified_fold_ids(y.values, TEST_FOLDS), TEST_FOLDS, seeds)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == "__main__":
    main()
