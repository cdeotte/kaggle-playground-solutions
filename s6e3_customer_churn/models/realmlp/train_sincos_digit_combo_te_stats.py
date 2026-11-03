#!/usr/bin/env python
"""RealMLP on periodic, square and digit features with mean / std / skew target encodings of curated column pairs and
digit pairs, original rows added to every fold's training rows (pair realmlp_sincos_digit_combo_te_stats).

Columns: NUMS = SeniorCitizen, tenure, MonthlyCharges, TotalCharges; the 15 other raw columns are categoricals.
Features (train, test and all 7,043 original rows):
- sin / cos of tenure and MonthlyCharges with periods 12, 14, 20; the square and a copy of each of NUMS;
- digits ((x * 10^k) mod 10) of SeniorCitizen (k 0), tenure (k 0, 1, 2), MonthlyCharges and TotalCharges
  (k -1, 0, 1, 2), used as categories;
- 8 pairs (tenure, MonthlyCharges, TotalCharges, SeniorCitizen x Contract, InternetService, PaymentMethod, Partner,
  Dependents) and the 6 pairs of the first 4 digit features, factorized over train + test + original (a pair with more
  distinct values than half the rows is dropped).
Per fold (the 5 folds of common/data.py): the fold's training rows followed by all original rows; the pair codes are
replaced by their target mean / std / skew (TargetEncoder_: 5-fold out-of-fold encodings on the training rows,
smoothing 'auto' for the mean; validation / test rows use the maps of all training rows); NUMS are factorized and,
with the digits and the categoricals, used as categories; NaN in continuous columns -> median.
Model: pytabkit RealMLP_TD_Classifier (3 x 384 mish, n_ens 8, lr 0.03, batch 1024, up to 60 epochs with early stopping
on the validation fold's AUC, random_state 42). Test = mean of the 5 fold models.
Usage: python train_sincos_digit_combo_te_stats.py"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit import RealMLP_TD_Classifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, N_FOLDS, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.simplefilter("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_sincos_digit_combo_te_stats"

PARAMS = {
    "device": "cuda",
    "random_state": 42,
    "verbosity": 2,
    "val_metric_name": "1-auc_ovr",
    "n_epochs": 60,
    "batch_size": 1024,
    "n_ens": 8,
    "use_early_stopping": True,
    "early_stopping_additive_patience": 20,
    "early_stopping_multiplicative_patience": 1,
    "act": "mish",
    "embedding_size": 8,
    "first_layer_lr_factor": 0.5962121993798933,
    "hidden_sizes": "rectangular",
    "hidden_width": 384,
    "lr": 0.03,
    "ls_eps": 0.011498317194338772,
    "ls_eps_sched": "coslog4",
    "max_one_hot_cat_size": 18,
    "n_hidden_layers": 3,
    "p_drop": 0.07301419697186451,
    "p_drop_sched": "flat_cos",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_lr_factor": 0.1151437622270563,
    "plr_sigma": 2.3316811282666916,
    "scale_lr_factor": 2.244801835541429,
    "sq_mom": 1.0 - 0.011834054955582318,
    "wd": 0.02369230879235962,
}


class TargetEncoder_(BaseEstimator, TransformerMixin):
    """Target statistics (aggs) of each column in cols_to_encode. fit_transform: out-of-fold encodings
    (KFold(cv, shuffle=True, random_state=42)); the mean is smoothed toward the fold mean (smooth='auto': within-group
    variance / between-group variance). transform: unsmoothed maps of all fitted rows."""

    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        temp_df = X.copy()
        temp_df['target'] = y
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                self.mappings_[col][agg_func] = temp_df.groupby(col)['target'].agg(agg_func)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name] = X_transformed[new_col_name].fillna(self.global_stats_[agg_func])
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        m = self.smooth
                        if self.smooth == 'auto':
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            m = (avg_variance_within / variance_between) if (variance_between and variance_between > 0) else 0
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)

                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed


def to01(s):
    if s.dtype == "O":
        return s.map({"Yes": 1, "No": 0}).astype("int8")
    return s.astype("int8")


def add_features(train, test, orig, NUMS, CATS):
    # sin / cos of tenure and MonthlyCharges
    for p in [12, 14, 20]:
        for df in (train, test, orig):
            df[f"_tenure_sin_{p}"] = np.sin(2 * np.pi * df["tenure"] / p).astype("float32")
            df[f"_tenure_cos_{p}"] = np.cos(2 * np.pi * df["tenure"] / p).astype("float32")
        for df in (train, test, orig):
            df[f"_MonthlyCharges_sin_{p}"] = np.sin(2 * np.pi * df["MonthlyCharges"] / p).astype("float32")
            df[f"_MonthlyCharges_cos_{p}"] = np.cos(2 * np.pi * df["MonthlyCharges"] / p).astype("float32")

    # squares and copies
    for c in NUMS:
        for df in (train, test, orig):
            df[f"{c}_sq"] = (df[c].astype("float32") ** 2).astype("float32")
            df[f"{c}_2"] = df[c].copy()

    # digit extraction
    DIGITS = []
    digit_plan = {"SeniorCitizen": [0], "tenure": [0, 1, 2], "MonthlyCharges": [-1, 0, 1, 2],
                  "TotalCharges": [-1, 0, 1, 2]}
    for c, ks in digit_plan.items():
        for k in ks:
            n = f"{c}_d{k}"
            for df in (train, test, orig):
                x = df[c].astype("float64")
                df[n] = ((x * (10 ** k)) % 10).fillna(-1).astype("int8")
            DIGITS.append(n)

    # curated pairs for target encoding
    TE_columns = []
    wanted_pairs = [
        ("tenure", "Contract"),
        ("tenure", "InternetService"),
        ("MonthlyCharges", "Contract"),
        ("MonthlyCharges", "InternetService"),
        ("TotalCharges", "Contract"),
        ("TotalCharges", "PaymentMethod"),
        ("SeniorCitizen", "Partner"),
        ("SeniorCitizen", "Dependents"),
    ]
    columns = NUMS + CATS
    for a, b in wanted_pairs:
        if (a not in columns) or (b not in columns):
            continue
        name = f"{a}-{b}"
        for df in (train, test, orig):
            df[name] = df[a].astype(str) + "_" + df[b].astype(str)

        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        combined, _ = combined.factorize()

        if pd.Series(combined).nunique() > len(combined) // 2:
            for df in (train, test, orig):
                df.drop(columns=[name], inplace=True)
            continue

        ntr, nts = len(train), len(test)
        train[name] = combined[:ntr]
        test[name] = combined[ntr:ntr + nts]
        orig[name] = combined[ntr + nts:]
        TE_columns.append(name)

    # digit-digit pairs
    for cols in list(combinations(DIGITS[:4], 2)):
        name = "-".join(cols)
        for df in (train, test, orig):
            df[name] = df[cols[0]].astype(str) + "_" + df[cols[1]].astype(str)

        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        combined, _ = combined.factorize()

        ntr, nts = len(train), len(test)
        train[name] = combined[:ntr]
        test[name] = combined[ntr:ntr + nts]
        orig[name] = combined[ntr + nts:]
        TE_columns.append(name)

    return train, test, orig, TE_columns, DIGITS


def predict_proba_1(model, X):
    p = np.asarray(model.predict_proba(X))
    if p.ndim == 2 and p.shape[1] >= 2:
        return p[:, 1]
    return p.reshape(-1)


def main():
    train = read_train()
    test = read_test()
    orig = read_original()
    for df in (train, test, orig):
        if ID in df.columns:
            df.drop(columns=[ID], inplace=True)

    train[TARGET] = to01(train[TARGET])
    orig[TARGET] = to01(orig[TARGET])

    NUMS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for df in (train, test, orig):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    CATS = [c for c in train.columns if c not in (NUMS + [TARGET])]
    print(f"train: {train.shape}  test: {test.shape}  orig: {orig.shape}")

    train, test, orig, TE_columns, DIGITS = add_features(train, test, orig, NUMS, CATS)
    FEATURES_ALL = [c for c in train.columns if c != TARGET]
    FEATURES = [c for c in FEATURES_ALL if (c in test.columns) and (c in orig.columns)]
    print(f"digits: {len(DIGITS)}  target-encoded pairs: {len(TE_columns)}  features: {len(FEATURES)}")

    y = train[TARGET].astype("int8").values
    fold_id = folds(y)
    oof = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test), dtype=np.float32)

    for fold in range(N_FOLDS):
        tr_idx, va_idx = np.where(fold_id != fold)[0], np.where(fold_id == fold)[0]
        X_tr = train.iloc[tr_idx][FEATURES].copy()
        y_tr = train.iloc[tr_idx][TARGET].copy()
        X_va = train.iloc[va_idx][FEATURES].copy()
        y_va = train.iloc[va_idx][TARGET].copy()
        X_te = test[FEATURES].copy()

        # original rows added to the fold's training rows
        X_tr = pd.concat([X_tr, orig[FEATURES]], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y_tr, orig[TARGET]], axis=0).reset_index(drop=True)

        te_cols_fold = [c for c in TE_columns if (c in X_tr.columns) and (c in X_va.columns) and (c in X_te.columns)]
        if len(te_cols_fold) > 0:
            enc = TargetEncoder_(cols_to_encode=te_cols_fold, cv=5, smooth="auto", aggs=["mean", "std", "skew"],
                                 drop_original=True)
            X_tr = enc.fit_transform(X_tr, y_tr)
            X_va = enc.transform(X_va)
            X_te = enc.transform(X_te)

        common_cols = sorted(set(X_tr.columns) & set(X_va.columns) & set(X_te.columns))
        X_tr = X_tr[common_cols].copy()
        X_va = X_va[common_cols].copy()
        X_te = X_te[common_cols].copy()

        combined = pd.concat([X_tr, X_va, X_te], axis=0, ignore_index=True)
        for c in NUMS:
            if c in combined.columns:
                combined[c], _ = combined[c].factorize()

        cat_block = [c for c in (NUMS + CATS + DIGITS) if c in combined.columns]
        combined[cat_block] = combined[cat_block].astype("category")

        # continuous NaN -> median, categorical NaN -> "__MISSING__"
        cont_cols = [c for c in combined.columns if pd.api.types.is_float_dtype(combined[c])]
        if cont_cols:
            med = combined[cont_cols].median(numeric_only=True)
            combined[cont_cols] = combined[cont_cols].fillna(med)
        for c in cat_block:
            if combined[c].isna().any():
                combined[c] = combined[c].astype(object).fillna("__MISSING__").astype("category")

        X_tr2 = combined.iloc[:len(X_tr)].copy()
        X_va2 = combined.iloc[len(X_tr):len(X_tr) + len(X_va)].copy()
        X_te2 = combined.iloc[len(X_tr) + len(X_va):].copy()

        model = RealMLP_TD_Classifier(**PARAMS)
        model.fit(X_tr2, y_tr, X_va2, y_va, cat_col_names=cat_block)

        va_pred = predict_proba_1(model, X_va2).astype(np.float32)
        te_pred = predict_proba_1(model, X_te2).astype(np.float32)
        oof[va_idx] = va_pred
        pred += te_pred / N_FOLDS
        print(f"Fold {fold + 1}/{N_FOLDS} | AUC={roc_auc_score(y_va, va_pred):.6f}", flush=True)

        del X_tr, X_va, X_te, X_tr2, X_va2, X_te2, combined, model
        gc.collect()

    print(f"OOF AUC: {roc_auc_score(train[TARGET].values, oof):.6f}")
    save_pair(model_dir(FOLDER), PAIR, oof, pred)


if __name__ == "__main__":
    main()
