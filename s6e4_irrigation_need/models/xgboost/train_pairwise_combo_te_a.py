#!/usr/bin/env python
"""GPU XGBoost on class target encodings and counts of all 171 pairwise combinations of the 19 columns, with the
original rows appended to every training fold (pair xgb_pairwise_combo_te_original_a).

Features: the 11 numerics (float32) and the 8 categoricals (integer codes over train + test + original, native
categoricals); the 171 pairwise combinations of the 19 columns (string concatenation, factorized over train + test +
original). Per outer fold, the training side = the competition rows of the 4 other folds + all 10,000 original rows;
every combination gets 3 smoothed (10) class-rate target encodings, out-of-fold over a random 5-way split of the
training side (numpy default_rng(42 + fold), fold = 1..5), fitted on the whole training side for the validation / test
rows, and its count on the training side; the raw combination codes are dropped (19 + 4 x 171 = 703 columns).
Model: XGBoost on GPU, depth 8, lr 0.01, colsample_bytree 0.341, subsample 0.899, lambda 4.06, alpha 2.92, up to 10,000
rounds, early stopping 200 (mlogloss) on the validation fold, random_state 42. Test = mean of the 5 fold models.
About 2 hours in total on one GPU (mostly the per-fold encodings).
Usage: python train_pairwise_combo_te_a.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.simplefilter(action="ignore", category=PerformanceWarning)

PAIR = "xgb_pairwise_combo_te_original_a"
DROP_COLS = [ID]

N_SPLITS = 5
RANDOM_STATE = 42
PAIR_R = 2

TE_KFOLD = 5
TE_SMOOTH = 10.0

LABELS = list(CLASSES)
NUM_CLASSES = 3

XGB_PARAM_GRID = {
    "colsample_bytree": 0.3413131357034518,
    "subsample": 0.8991973526634904,
    "reg_lambda": 4.063399595895267,
    "reg_alpha": 2.917406632455954,
    "max_depth": 8,
}

XGB_FIXED = dict(
    n_estimators=10000,
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=NUM_CLASSES,
    learning_rate=0.01,
    early_stopping_rounds=200,
    random_state=RANDOM_STATE,
    enable_categorical=True,
    device="cuda",
    n_jobs=-1,
)


def encode_target(series: pd.Series) -> pd.Series:
    y = series.map(TARGET_MAP)
    if y.isna().any():
        raise ValueError("Unexpected target labels encountered.")
    return y.astype(np.int8)


def target_encode_multiclass(train_with_y, valid, test, col, target, kfold=5, smooth=10.0, seed=42):
    train = train_with_y.copy().reset_index(drop=True)
    valid = valid.copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    fold_id = np.arange(len(train)) % kfold
    rng.shuffle(fold_id)
    train["kfold"] = fold_id

    out_cols = [f"TE_{'_'.join(col)}_c{cls}" for cls in range(NUM_CLASSES)]
    for out_col in out_cols:
        train[out_col] = 0.0

    priors = {cls: float((train[target] == cls).mean()) for cls in range(NUM_CLASSES)}

    for i in range(kfold):
        df_tmp = train[train["kfold"] != i]
        df_va = train[train["kfold"] == i]
        for cls in range(NUM_CLASSES):
            grp = (
                pd.DataFrame({**{c: df_tmp[c].values for c in col}, "_y_": (df_tmp[target].values == cls).astype(np.float32)})
                .groupby(col)["_y_"]
                .agg(["sum", "count"])
                .reset_index()
            )
            grp["TE_tmp"] = ((grp["sum"] + smooth * priors[cls]) / (grp["count"] + smooth)).astype(np.float32)
            merged = df_va[col].merge(grp[col + ["TE_tmp"]], how="left", on=col)
            train.loc[train["kfold"] == i, out_cols[cls]] = merged["TE_tmp"].fillna(priors[cls]).astype(np.float32).values

    for cls in range(NUM_CLASSES):
        grp = (
            pd.DataFrame({**{c: train[c].values for c in col}, "_y_": (train[target].values == cls).astype(np.float32)})
            .groupby(col)["_y_"]
            .agg(["sum", "count"])
            .reset_index()
        )
        grp["TE_tmp"] = ((grp["sum"] + smooth * priors[cls]) / (grp["count"] + smooth)).astype(np.float32)
        valid[out_cols[cls]] = valid[col].merge(grp[col + ["TE_tmp"]], how="left", on=col)["TE_tmp"].fillna(priors[cls]).astype(np.float32).values
        test[out_cols[cls]] = test[col].merge(grp[col + ["TE_tmp"]], how="left", on=col)["TE_tmp"].fillna(priors[cls]).astype(np.float32).values

    train = train.drop(columns=["kfold"])
    return train.drop(columns=[target]), valid, test


def count_encode(train, valid, test, col):
    counts = train[col].value_counts()
    train[f"CE_{col}"] = train[col].map(counts).astype(np.float32)
    valid[f"CE_{col}"] = valid[col].map(counts).fillna(0).astype(np.float32)
    test[f"CE_{col}"] = test[col].map(counts).fillna(0).astype(np.float32)
    return train, valid, test


def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()

    for c in DROP_COLS:
        if c in train.columns:
            train.drop(columns=[c], inplace=True)
        if c in test.columns:
            test.drop(columns=[c], inplace=True)
        if c in orig.columns:
            orig.drop(columns=[c], inplace=True)

    train[TARGET] = encode_target(train[TARGET])
    orig[TARGET] = encode_target(orig[TARGET])
    y = train[TARGET].values.astype(np.int8)

    all_cols = [c for c in train.columns if c != TARGET]
    CATS = [c for c in all_cols if train[c].dtype == object or str(train[c].dtype) == "boolean" or train[c].dtype == bool]
    NUMS = [c for c in all_cols if c not in CATS]

    for c in NUMS:
        train[c] = pd.to_numeric(train[c], errors="coerce").astype(np.float32)
        test[c] = pd.to_numeric(test[c], errors="coerce").astype(np.float32)
        orig[c] = pd.to_numeric(orig[c], errors="coerce").astype(np.float32)

    for c in CATS:
        combined = pd.concat([train[c].astype(str), test[c].astype(str), orig[c].astype(str)], ignore_index=True)
        codes, _ = pd.factorize(combined, sort=False)
        train[c] = codes[: len(train)].astype(np.int32)
        test[c] = codes[len(train): len(train) + len(test)].astype(np.int32)
        orig[c] = codes[len(train) + len(test):].astype(np.int32)

    print(f"NUMS ({len(NUMS)}): {NUMS}")
    print(f"CATS ({len(CATS)}): {CATS}")

    TE_COLUMNS = []
    columns = NUMS + CATS
    for cols in combinations(columns, PAIR_R):
        name = "-".join(cols)
        for df in (train, test, orig):
            s = df[cols[0]].astype(str)
            for cc in cols[1:]:
                s = s + "_" + df[cc].astype(str)
            df[name] = s
        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        codes, _ = pd.factorize(combined, sort=False)
        train[name] = codes[: len(train)].astype(np.int32)
        test[name] = codes[len(train): len(train) + len(test)].astype(np.int32)
        orig[name] = codes[len(train) + len(test):].astype(np.int32)
        TE_COLUMNS.append(name)

    FEATURES = [c for c in train.columns if c != TARGET]
    BASE_FEATURES = FEATURES.copy()
    CAT_FEATURES = set(CATS + TE_COLUMNS)
    print(f"Base+combo feature count (pre-encoding): {len(FEATURES)}")
    return train, test, orig, y, BASE_FEATURES, TE_COLUMNS, CAT_FEATURES


def run_fold(fold, tr_idx, va_idx, train, test, orig, BASE_FEATURES, TE_COLUMNS, CAT_FEATURES):
    """fold: 1..5 (seeds the inner target-encoding split)."""
    print(f"\n### Fold {fold}/{N_SPLITS}")

    train_fold = train.iloc[tr_idx].reset_index(drop=True)
    valid_fold = train.iloc[va_idx].reset_index(drop=True)
    test_fold = test.copy().reset_index(drop=True)
    orig_fold = orig.copy().reset_index(drop=True)

    train_fold_plus_orig = pd.concat([train_fold, orig_fold], axis=0, ignore_index=True)

    Xtr = train_fold_plus_orig.copy()
    Xva = valid_fold[BASE_FEATURES].copy()
    Xte = test_fold[BASE_FEATURES].copy()

    for col in TE_COLUMNS:
        train_with_y = Xtr.copy()
        train_with_y[TARGET] = train_fold_plus_orig[TARGET].values
        Xtr, Xva, Xte = target_encode_multiclass(
            train_with_y,
            Xva,
            Xte,
            [col],
            TARGET,
            kfold=TE_KFOLD,
            smooth=TE_SMOOTH,
            seed=RANDOM_STATE + fold,
        )
        Xtr, Xva, Xte = count_encode(Xtr, Xva, Xte, col)
        if col in Xtr.columns:
            Xtr.drop(columns=[col], inplace=True)
        if col in Xva.columns:
            Xva.drop(columns=[col], inplace=True)
        if col in Xte.columns:
            Xte.drop(columns=[col], inplace=True)
        gc.collect()

    y_tr = train_fold_plus_orig[TARGET].values.astype(np.int8)
    y_va = valid_fold[TARGET].values.astype(np.int8)

    cols = list(Xtr.columns)
    feature_types = ["c" if c in CAT_FEATURES else "q" for c in cols]
    model = XGBClassifier(**XGB_FIXED, **XGB_PARAM_GRID, feature_types=feature_types)
    model.fit(
        Xtr[cols],
        y_tr,
        eval_set=[(Xva[cols], y_va)],
        verbose=200,
    )

    va_pred = model.predict_proba(Xva[cols]).astype(np.float32)
    te_pred = model.predict_proba(Xte[cols]).astype(np.float32)

    fold_bac = balanced_accuracy_score(y_va, va_pred.argmax(axis=1))
    print(f"{PAIR} fold {fold - 1}: BAC {fold_bac:.8f}", flush=True)
    del Xtr, Xva, Xte, model, train_fold, valid_fold, test_fold, orig_fold, train_fold_plus_orig
    gc.collect()
    return va_pred, te_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train, test, orig, y, BASE_FEATURES, TE_COLUMNS, CAT_FEATURES = build_features()
        fold_id = folds(y)
        for k in todo:
            va_pred, te_pred = run_fold(k + 1, np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k), train, test,
                                        orig, BASE_FEATURES, TE_COLUMNS, CAT_FEATURES)
            save_part(HERE, PAIR, k, va_pred, te_pred)
    y = read_train()[TARGET].map(TARGET_MAP).values.astype(np.int8)
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
