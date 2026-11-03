#!/usr/bin/env python
"""LightGBM on original-data priors, value frequencies and nested target statistics (pair lgbm_te_stats_orig_priors).

Adapts the LightGBM part of badalkrsharma's public notebook of this competition:
https://www.kaggle.com/code/badalkrsharma/rank-blend-cv-0-91865-xgb-lgb-cat

Features of the train / test rows (the original rows, TotalCharges blanks filled with the median, give frequencies and
priors):
  - tenure, MonthlyCharges, TotalCharges (float32), TotalCharges / (tenure + 1), tenure x MonthlyCharges
  - frequency of tenure, MonthlyCharges and TotalCharges over train + original + test
  - original churn rate of each value of the 16 categoricals and the 3 numerics (unseen: 0.5)
Per fold, for the 19 keys (the 16 categoricals and the 3 numerics as categories): std / min / max of the target per
value (training rows out-of-fold over an inner StratifiedKFold(10, shuffled, seed 42); validation / test rows from all
training rows; missing: 0) and the scikit-learn TargetEncoder mean (binary, inner 10 folds, smooth "auto", seed 42);
the keys themselves are dropped (103 model columns).
Model: lr 0.03, 63 leaves, max_depth 7, subsample 0.8, colsample_bytree 0.8, reg_alpha 0.1, reg_lambda 1,
min_child_samples 20, seed 42, 32 threads; up to 50,000 rounds with early stopping (500) on the validation fold.
Usage: python train_te_stats_orig_priors.py"""
import gc
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "lgbm_te_stats_orig_priors"
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 10
STATS = ["std", "min", "max"]

LGB_PARAMS = {
    "n_estimators": 50000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "objective": "binary",
    "metric": "auc",
    "n_jobs": 32,
    "verbose": -1,
}


def load_data():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = labels(train[TARGET])
    orig[TARGET] = labels(orig[TARGET])
    orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())
    print(f"train {len(train):,} rows (churn {train[TARGET].mean() * 100:.2f}%) | original {len(orig):,} rows | "
          f"test {len(test):,} rows", flush=True)
    return train, test, orig


def build_features(train, test, orig):
    cats = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]
    nums = ["tenure", "MonthlyCharges", "TotalCharges"]
    new_nums, num_as_cat = [], []

    for df in (train, test, orig):
        for c in cats:
            df[c] = df[c].astype(str).fillna("missing").str.strip()
        for c in nums:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    for df in (train, test, orig):
        df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["tenure_x_MonthlyCharges"] = df["tenure"] * df["MonthlyCharges"]
    new_nums.extend(["AvgChargesPerMonth", "tenure_x_MonthlyCharges"])

    for col in nums:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in (train, test, orig):
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
        new_nums.append(f"FREQ_{col}")

    for col in nums:
        new_col = f"CAT_{col}"
        num_as_cat.append(new_col)
        for df in (train, test, orig):
            df[new_col] = df[col].astype(str).astype("category")

    for col in cats + nums:
        tmp = orig.groupby(col)[TARGET].mean()
        new_col = f"ORIG_proba_{col}"
        tmp.name = new_col
        train = train.merge(tmp, on=col, how="left")
        test = test.merge(tmp, on=col, how="left")
        new_nums.append(new_col)
        train[new_col] = train[new_col].fillna(0.5)
        test[new_col] = test[new_col].fillna(0.5)

    features = nums + cats + new_nums + num_as_cat
    seen = set()
    features = [f for f in features if f not in seen and not seen.add(f)]
    te_columns = num_as_cat + cats
    print(f"features before encoding: {len(features)} | keys: {len(te_columns)}", flush=True)
    return train, test, features, cats, num_as_cat, te_columns


def add_nested_te(X_tr, y_tr, X_val, X_te, features, te_columns, cats, num_as_cat, seed):
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()

    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for in_tr, in_va in skf_inner.split(X_tr, y_tr):
        X_tr2 = X_tr.loc[in_tr, features + [TARGET]].copy()
        X_va2 = X_tr.loc[in_va, features].copy()
        for col in te_columns:
            tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            X_va2 = X_va2.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")

    for col in te_columns:
        tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
        tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
        tmp = tmp.astype("float32")
        X_val = X_val.merge(tmp, on=col, how="left")
        X_te = X_te.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            for df_ in (X_tr, X_val, X_te):
                df_[c] = df_[c].fillna(0)

    te_mean_cols = [f"TE_{col}" for col in te_columns]
    te = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth="auto", target_type="binary", random_state=seed)
    X_tr[te_mean_cols] = te.fit_transform(X_tr[te_columns], y_tr).astype("float32")
    X_val[te_mean_cols] = te.transform(X_val[te_columns]).astype("float32")
    X_te[te_mean_cols] = te.transform(X_te[te_columns]).astype("float32")

    to_remove = num_as_cat + cats
    cat_cols_present = [c for c in (cats + num_as_cat) if c in X_tr.columns]
    for df_ in (X_tr, X_val, X_te):
        if cat_cols_present:
            df_[cat_cols_present] = df_[cat_cols_present].astype(str).astype("category")
        drop_cols = [c for c in to_remove if c in df_.columns]
        if drop_cols:
            df_.drop(columns=drop_cols, inplace=True)

    X_tr.drop(columns=[TARGET], inplace=True)
    cols = list(X_tr.columns)
    X_val = X_val[cols].copy()
    X_te = X_te[cols].copy()
    return X_tr, X_val, X_te, cols


def main():
    train, test, orig = load_data()
    train, test, features, cats, num_as_cat, te_columns = build_features(train, test, orig)
    params = {**LGB_PARAMS, "random_state": SEED}
    y = train[TARGET].values
    fold_ids = folds(y)

    oof = np.zeros(len(train), dtype=np.float64)
    pred = np.zeros(len(test), dtype=np.float64)
    for k in range(N_FOLDS):
        train_idx = np.where(fold_ids != k)[0]
        val_idx = np.where(fold_ids == k)[0]
        X_tr = train.loc[train_idx, features + [TARGET]].reset_index(drop=True).copy()
        y_tr = train.loc[train_idx, TARGET].values.copy()
        X_val = train.loc[val_idx, features].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, TARGET].values.copy()
        X_te = test[features].reset_index(drop=True).copy()

        X_tr, X_val, X_te, cols = add_nested_te(X_tr, y_tr, X_val, X_te, features, te_columns, cats, num_as_cat, SEED)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr.values, y_tr,
            eval_set=[(X_val.values, y_val)],
            callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(500)],
        )
        oof_p = model.predict_proba(X_val.values)[:, 1]
        oof[val_idx] = oof_p
        pred += model.predict_proba(X_te[cols].values)[:, 1] / N_FOLDS
        print(f"fold {k + 1}/{N_FOLDS}: AUC {roc_auc_score(y_val, oof_p):.6f} | best_iteration {model.best_iteration_}",
              flush=True)

        del X_tr, X_val, X_te, y_tr, y_val, model
        gc.collect()

    print(f"{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, PAIR, oof, pred)


if __name__ == "__main__":
    main()
