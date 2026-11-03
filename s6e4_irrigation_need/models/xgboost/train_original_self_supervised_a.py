#!/usr/bin/env python
"""GPU XGBoost on the raw columns plus self-supervised predictions of auxiliary models trained on the original rows
(pair xgb_original_self_supervised_a).

Auxiliary stage: for each of the 19 columns, XGBoost models trained on the 10,000 original rows predict that column
from the 18 others (native categoricals; KFold(5) for numeric targets, StratifiedKFold(5) for categorical targets, both
shuffle 42, early stopping 200 on the held-out original fold; depth 4, lr 0.05, subsample / colsample_bytree 0.9, up to
50,000 rounds). The train / test predictions are the mean of the 5 auxiliary models: the predicted value for the 10
numeric columns with > 20 distinct values (PRED_<col>), the max class probability and the entropy for the categoricals
(plus P(class 1) for the 2-level Mulching_Used): 28 columns.
Final model: the 19 raw columns (native categoricals) + the 28 predictions; XGBoost hist on GPU, depth 3, lr 0.1,
up to 100,000 rounds, min_child_weight 5, subsample / colsample_bytree 0.85, early stopping 200 (mlogloss) on the
validation fold, balanced sample weights, random_state 42. Test = mean of the 5 fold models. The original rows are only
used by the auxiliary models.
The auxiliary stage takes about 2 minutes, each fold about 1 minute on one GPU.
Usage: python train_original_self_supervised_a.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "xgb_original_self_supervised_a"
DROP_COLS = [ID]

N_SPLITS = 5
RANDOM_STATE = 42
DEVICE = "cuda"

AUX_N_ESTIMATORS = 50000
AUX_LR = 0.05
AUX_EARLY_STOP = 200

FINAL_N_ESTIMATORS = 100000
FINAL_LR = 0.1
FINAL_EARLY_STOP = 200

xgb_params = dict(
    n_estimators=FINAL_N_ESTIMATORS,
    learning_rate=FINAL_LR,
    max_depth=3,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    enable_categorical=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=FINAL_EARLY_STOP,
    device=DEVICE,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _drop_cols_that_exist(df, cols):
    return [c for c in cols if c in df.columns]


def _entropy_from_probs(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)


def _is_numeric(s):
    return pd.api.types.is_numeric_dtype(s)


def _as_category_inplace(df):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype("category")


def build_features():
    log("Loading data...")
    orig = read_original()
    train = read_train()
    test = read_test()

    assert TARGET in orig.columns
    assert TARGET in train.columns

    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(np.int32)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(np.int32)

    drop_existing_orig = _drop_cols_that_exist(orig, DROP_COLS)
    drop_existing_train = _drop_cols_that_exist(train, DROP_COLS)
    drop_existing_test = _drop_cols_that_exist(test, DROP_COLS)

    orig_feats = [c for c in orig.columns if c not in ([TARGET] + drop_existing_orig)]
    train_feats = [c for c in train.columns if c not in ([TARGET] + drop_existing_train)]
    test_feats = [c for c in test.columns if c not in drop_existing_test]
    base_feats = [c for c in orig_feats if c in train_feats and c in test_feats]

    if len(base_feats) == 0:
        raise ValueError("No overlapping base features among orig/train/test after dropping target/IDs.")

    log(f"Base feature count (intersection): {len(base_feats)}")

    orig_X_base = orig[base_feats].copy()
    train_X_base = train[base_feats].copy()
    test_X_base = test[base_feats].copy()

    all_base = pd.concat([orig_X_base, train_X_base, test_X_base], axis=0, ignore_index=True)
    _as_category_inplace(all_base)

    n0 = len(orig_X_base)
    n1 = len(train_X_base)
    orig_X_base = all_base.iloc[:n0].copy()
    train_X_base = all_base.iloc[n0:n0 + n1].copy()
    test_X_base = all_base.iloc[n0 + n1:].copy()
    del all_base
    gc.collect()

    aux_train_cols = {}
    aux_test_cols = {}

    common_tree_params = dict(
        n_estimators=AUX_N_ESTIMATORS,
        learning_rate=AUX_LR,
        max_depth=4,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device=DEVICE,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    log("Training aux models on ORIG and inferring PRED_* on train/test ...")
    for j, tgt_col in enumerate(base_feats, 1):
        Xo = orig_X_base.drop(columns=[tgt_col], errors="ignore")
        Xt_train = train_X_base.drop(columns=[tgt_col], errors="ignore")
        Xt_test = test_X_base.drop(columns=[tgt_col], errors="ignore")
        y_raw = orig_X_base[tgt_col]

        task = "reg" if _is_numeric(y_raw) and y_raw.nunique(dropna=True) > 20 else "clf"
        nunq = int(y_raw.nunique(dropna=True))

        if nunq <= 1:
            const_val = float(pd.to_numeric(y_raw, errors="coerce").dropna().iloc[0]) if y_raw.dropna().shape[0] else 0.0
            colname = f"PRED_{tgt_col}"
            aux_train_cols[colname] = np.full(len(train_X_base), const_val, dtype=np.float32)
            aux_test_cols[colname] = np.full(len(test_X_base), const_val, dtype=np.float32)
            log(f"[{j:02d}/{len(base_feats)}] {tgt_col}: degenerate -> {colname}")
            continue

        if task == "reg":
            y = pd.to_numeric(y_raw, errors="coerce").astype("float32").values
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            pred_train = np.zeros(len(train_X_base), dtype=np.float32)
            pred_test = np.zeros(len(test_X_base), dtype=np.float32)

            for tr_idx, va_idx in kf.split(Xo, y):
                X_tr, X_va = Xo.iloc[tr_idx], Xo.iloc[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                m_tr = ~np.isnan(y_tr)
                m_va = ~np.isnan(y_va)
                if m_tr.sum() < 50 or m_va.sum() < 20:
                    continue
                model = XGBRegressor(
                    **common_tree_params,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    early_stopping_rounds=AUX_EARLY_STOP,
                    enable_categorical=True,
                )
                model.fit(X_tr.iloc[m_tr], y_tr[m_tr], eval_set=[(X_va.iloc[m_va], y_va[m_va])], verbose=False)
                pred_train += model.predict(Xt_train).astype(np.float32) / N_SPLITS
                pred_test += model.predict(Xt_test).astype(np.float32) / N_SPLITS

            colname = f"PRED_{tgt_col}"
            aux_train_cols[colname] = pred_train
            aux_test_cols[colname] = pred_test
            log(f"[{j:02d}/{len(base_feats)}] {tgt_col}: REG -> {colname}")
            continue

        y_cat = y_raw.astype("category")
        y_codes = y_cat.cat.codes.values
        if (y_codes == -1).any():
            y_codes = y_codes.copy()
            y_codes[y_codes == -1] = y_codes.max() + 1
        n_classes = int(np.max(y_codes) + 1)

        if n_classes > 50:
            y = y_codes.astype(np.float32)
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            pred_train = np.zeros(len(train_X_base), dtype=np.float32)
            pred_test = np.zeros(len(test_X_base), dtype=np.float32)
            for tr_idx, va_idx in kf.split(Xo, y):
                X_tr, X_va = Xo.iloc[tr_idx], Xo.iloc[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                model = XGBRegressor(
                    **common_tree_params,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    early_stopping_rounds=AUX_EARLY_STOP,
                    enable_categorical=True,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                pred_train += model.predict(Xt_train).astype(np.float32) / N_SPLITS
                pred_test += model.predict(Xt_test).astype(np.float32) / N_SPLITS
            colname = f"PRED_{tgt_col}_code"
            aux_train_cols[colname] = pred_train
            aux_test_cols[colname] = pred_test
            log(f"[{j:02d}/{len(base_feats)}] {tgt_col}: MANY-CLASS -> {colname}")
            continue

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        probs_train = np.zeros((len(train_X_base), n_classes), dtype=np.float32)
        probs_test = np.zeros((len(test_X_base), n_classes), dtype=np.float32)
        for tr_idx, va_idx in skf.split(Xo, y_codes):
            X_tr, X_va = Xo.iloc[tr_idx], Xo.iloc[va_idx]
            y_tr, y_va = y_codes[tr_idx], y_codes[va_idx]
            if n_classes == 2:
                model = XGBClassifier(
                    **common_tree_params,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    early_stopping_rounds=AUX_EARLY_STOP,
                    enable_categorical=True,
                )
            else:
                model = XGBClassifier(
                    **common_tree_params,
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    early_stopping_rounds=AUX_EARLY_STOP,
                    enable_categorical=True,
                )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            probs_train += model.predict_proba(Xt_train).astype(np.float32) / N_SPLITS
            probs_test += model.predict_proba(Xt_test).astype(np.float32) / N_SPLITS

        aux_train_cols[f"PRED_{tgt_col}_maxprob"] = probs_train.max(axis=1).astype(np.float32)
        aux_test_cols[f"PRED_{tgt_col}_maxprob"] = probs_test.max(axis=1).astype(np.float32)
        aux_train_cols[f"PRED_{tgt_col}_entropy"] = _entropy_from_probs(probs_train)
        aux_test_cols[f"PRED_{tgt_col}_entropy"] = _entropy_from_probs(probs_test)
        if n_classes == 2:
            aux_train_cols[f"PRED_{tgt_col}_p1"] = probs_train[:, 1].astype(np.float32)
            aux_test_cols[f"PRED_{tgt_col}_p1"] = probs_test[:, 1].astype(np.float32)
        log(f"[{j:02d}/{len(base_feats)}] {tgt_col}: CLF({n_classes})")

    aux_train = pd.DataFrame(aux_train_cols)
    aux_test = pd.DataFrame(aux_test_cols)
    log(f"Aux features created: {aux_train.shape[1]}")

    y = train[TARGET].values.astype(np.int32)
    X_train = pd.concat([train_X_base.reset_index(drop=True), aux_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_X_base.reset_index(drop=True), aux_test.reset_index(drop=True)], axis=1)
    _as_category_inplace(X_train)
    _as_category_inplace(X_test)

    log(f"Final matrices: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X_train, X_test, y = build_features()
        fold_id = folds(y)
        log("Training final multiclass XGBoost with aux features...")
        for k in todo:
            trn_idx, val_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
            y_trn, y_val = y[trn_idx], y[val_idx]
            sw = compute_sample_weight(class_weight="balanced", y=y_trn)
            model = XGBClassifier(**xgb_params)
            model.fit(X_trn, y_trn, sample_weight=sw, eval_set=[(X_val, y_val)], verbose=500)
            p_val = model.predict_proba(X_val).astype(np.float32)
            p_test = model.predict_proba(X_test).astype(np.float32)
            save_part(HERE, PAIR, k, p_val, p_test)
            bac = balanced_accuracy_score(y_val, p_val.argmax(axis=1))
            log(f"{PAIR} fold {k}: BAC = {bac:.8f}")
            del model
            gc.collect()
    y = read_train()[TARGET].map(TARGET_MAP).astype(np.int32).values
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
