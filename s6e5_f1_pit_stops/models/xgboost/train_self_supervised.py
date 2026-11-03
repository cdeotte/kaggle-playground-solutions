#!/usr/bin/env python
"""GPU XGBoost with self-supervised feature predictions learned on the original data (pair xgb_original_self_supervised).

Auxiliary models (no target used): for each of the 14 base features, XGBoost models trained on the original rows predict
that feature from the 13 others (5-fold split of the original rows, early stopping 200 on the held-out original fold; the
5 models' predictions for train and test are averaged). Numerics with >= 100 levels are regressed (PRED_<col>); the
other columns are classified (PRED_<col>_maxprob, PRED_<col>_entropy, and PRED_<col>_p1 for binary columns).
Aux params: depth 4, lr 0.05, up to 50,000 rounds, subsample 0.9, colsample_bytree 0.9, native categoricals.
Final model on the 14 base features (columns with < 100 levels as categories) + 25 PRED_* columns: depth 3, lr 0.1,
min_child_weight 5, subsample 0.85, colsample_bytree 0.85, native categoricals, early stopping 200 on the validation fold,
random_state 42. Test = mean of the 5 fold models.
Usage: python train_self_supervised.py [--folds 0,1,2,3,4] [--force]
The auxiliary stage (70 models) runs before the folds in every process."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = "xgb_original_self_supervised"
ID_COL = ID
# Feature order of the model = the column order of the original csv.
ORIGINAL_COLUMN_ORDER = ['Driver', 'LapNumber', 'Compound', 'Stint', 'TyreLife', 'Position', 'LapTime (s)', 'Race', 'Year',
                         'LapTime_Delta', 'Cumulative_Degradation', 'PitStop', TARGET, 'RaceProgress', 'Position_Change']

N_SPLITS = 5
RANDOM_STATE = 42
CAT_CARDINALITY_THRESHOLD = 100

AUX_N_ESTIMATORS = 50_000
AUX_LR = 0.05
AUX_EARLY_STOP = 200
FINAL_N_ESTIMATORS = 100_000
FINAL_LR = 0.1
FINAL_EARLY_STOP = 200
DEVICE = 'cuda'
N_JOBS = 32

AUX_COMMON_PARAMS = dict(
    n_estimators=AUX_N_ESTIMATORS,
    learning_rate=AUX_LR,
    max_depth=4,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method='hist',
    device=DEVICE,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    enable_categorical=True,
)

FINAL_XGB_PARAMS = dict(
    n_estimators=FINAL_N_ESTIMATORS,
    learning_rate=FINAL_LR,
    max_depth=3,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    enable_categorical=True,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    early_stopping_rounds=FINAL_EARLY_STOP,
    device=DEVICE,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def to_binary_target(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype('int8')
    return s.astype(str).str.strip().str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0}).astype('int8')


def safe_feature_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in str(name)).strip('_')


def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return (-np.sum(p * np.log(p), axis=1)).astype('float32')


def is_cat_col(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True
    return s.nunique(dropna=False) < CAT_CARDINALITY_THRESHOLD


def prepare_base_frames(orig: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame):
    orig[TARGET] = to_binary_target(orig[TARGET])
    train[TARGET] = to_binary_target(train[TARGET])

    orig_feats = [c for c in orig.columns if c not in [TARGET, ID_COL]]
    train_feats = [c for c in train.columns if c not in [TARGET, ID_COL]]
    test_feats = [c for c in test.columns if c not in [ID_COL]]
    base_feats = [c for c in orig_feats if c in train_feats and c in test_feats]
    if not base_feats:
        raise RuntimeError('No common base features found.')

    log(f'Base features ({len(base_feats)}): {base_feats}')
    orig_X = orig[base_feats].copy()
    train_X = train[base_feats].copy()
    test_X = test[base_feats].copy()

    combined = pd.concat([orig_X, train_X, test_X], axis=0, ignore_index=True)
    cat_cols = [c for c in base_feats if is_cat_col(combined[c])]
    num_cols = [c for c in base_feats if c not in cat_cols]
    log(f'Categorical base cols ({len(cat_cols)}): {cat_cols}')
    log(f'Numeric base cols ({len(num_cols)}): {num_cols}')

    for c in cat_cols:
        combined[c] = combined[c].astype('string').fillna('__NA__').astype('category')
    for c in num_cols:
        combined[c] = pd.to_numeric(combined[c], errors='coerce').astype('float32')

    n0, n1 = len(orig_X), len(train_X)
    orig_X = combined.iloc[:n0].reset_index(drop=True).copy()
    train_X = combined.iloc[n0:n0 + n1].reset_index(drop=True).copy()
    test_X = combined.iloc[n0 + n1:].reset_index(drop=True).copy()
    return orig, train, test, orig_X, train_X, test_X, base_feats, cat_cols, num_cols


def split_for_aux(y: np.ndarray, task: str):
    if task == 'clf':
        vc = pd.Series(y).value_counts()
        if len(vc) > 1 and vc.min() >= N_SPLITS:
            return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def fit_aux_classifier_fold(
    X_fit: pd.DataFrame,
    y_fit_raw: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval_raw: np.ndarray,
    X_pred_train: pd.DataFrame,
    X_pred_test: pd.DataFrame,
    n_global_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    # A rare class can be missing from an auxiliary training fold: train on the fold's own classes and put the
    # probabilities back into the global class columns.
    from xgboost import XGBClassifier

    fold_classes = np.sort(np.unique(y_fit_raw)).astype('int32')
    out_train = np.zeros((len(X_pred_train), n_global_classes), dtype='float32')
    out_test = np.zeros((len(X_pred_test), n_global_classes), dtype='float32')

    if len(fold_classes) == 1:
        out_train[:, fold_classes[0]] = 1.0
        out_test[:, fold_classes[0]] = 1.0
        return out_train, out_test

    class_to_local = {int(cls): i for i, cls in enumerate(fold_classes)}
    y_fit = np.array([class_to_local[int(v)] for v in y_fit_raw], dtype='int32')
    eval_mask = np.isin(y_eval_raw, fold_classes)

    params = dict(AUX_COMMON_PARAMS)
    if len(fold_classes) == 2:
        params.update(objective='binary:logistic', eval_metric='logloss')
    else:
        params.update(objective='multi:softprob', num_class=len(fold_classes), eval_metric='mlogloss')

    fit_kwargs = {'verbose': False}
    if eval_mask.any():
        y_eval = np.array([class_to_local[int(v)] for v in y_eval_raw[eval_mask]], dtype='int32')
        fit_kwargs['eval_set'] = [(X_eval.iloc[eval_mask], y_eval)]
        params['early_stopping_rounds'] = AUX_EARLY_STOP

    model = XGBClassifier(**params)
    model.fit(X_fit, y_fit, **fit_kwargs)
    p_tr_local = model.predict_proba(X_pred_train).astype('float32')
    p_te_local = model.predict_proba(X_pred_test).astype('float32')
    if len(fold_classes) == 2 and p_tr_local.ndim == 1:
        p_tr_local = np.column_stack([1 - p_tr_local, p_tr_local]).astype('float32')
        p_te_local = np.column_stack([1 - p_te_local, p_te_local]).astype('float32')

    out_train[:, fold_classes] = p_tr_local
    out_test[:, fold_classes] = p_te_local
    del model
    gc.collect()
    return out_train, out_test


def make_aux_features(orig_X: pd.DataFrame, train_X: pd.DataFrame, test_X: pd.DataFrame, base_feats: list[str]):
    from xgboost import XGBRegressor

    aux_train_cols = {}
    aux_test_cols = {}

    log('Training original-feature auxiliary models')
    for j, tgt_col in enumerate(base_feats, start=1):
        safe = safe_feature_name(tgt_col)
        Xo = orig_X.drop(columns=[tgt_col], errors='ignore')
        Xt_train = train_X.drop(columns=[tgt_col], errors='ignore')
        Xt_test = test_X.drop(columns=[tgt_col], errors='ignore')
        y_raw = orig_X[tgt_col]
        nunq = int(y_raw.nunique(dropna=True))

        if nunq <= 1:
            const_val = float(pd.to_numeric(y_raw, errors='coerce').dropna().iloc[0]) if pd.to_numeric(y_raw, errors='coerce').dropna().shape[0] else 0.0
            cname = f'PRED_{safe}'
            aux_train_cols[cname] = np.full(len(train_X), const_val, dtype='float32')
            aux_test_cols[cname] = np.full(len(test_X), const_val, dtype='float32')
            log(f'[{j:02d}/{len(base_feats)}] {tgt_col}: constant')
            continue

        if pd.api.types.is_numeric_dtype(y_raw) and nunq >= CAT_CARDINALITY_THRESHOLD:
            task = 'reg'
        else:
            task = 'clf'

        if task == 'reg':
            y = pd.to_numeric(y_raw, errors='coerce').astype('float32').to_numpy()
            pred_train = np.zeros(len(train_X), dtype='float32')
            pred_test = np.zeros(len(test_X), dtype='float32')
            kf = split_for_aux(y, task)
            for fold, (tr_idx, va_idx) in enumerate(kf.split(Xo, y), start=1):
                y_tr = y[tr_idx]
                y_va = y[va_idx]
                m_tr = np.isfinite(y_tr)
                m_va = np.isfinite(y_va)
                model = XGBRegressor(
                    **AUX_COMMON_PARAMS,
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    early_stopping_rounds=AUX_EARLY_STOP,
                )
                model.fit(
                    Xo.iloc[tr_idx].iloc[m_tr],
                    y_tr[m_tr],
                    eval_set=[(Xo.iloc[va_idx].iloc[m_va], y_va[m_va])],
                    verbose=False,
                )
                pred_train += model.predict(Xt_train).astype('float32') / N_SPLITS
                pred_test += model.predict(Xt_test).astype('float32') / N_SPLITS
                del model
                gc.collect()
            cname = f'PRED_{safe}'
            aux_train_cols[cname] = pred_train
            aux_test_cols[cname] = pred_test
            log(f'[{j:02d}/{len(base_feats)}] {tgt_col}: REG -> {cname}')
        else:
            y_codes, uniques = pd.factorize(y_raw.astype('string').fillna('__NA__'), sort=True)
            n_classes = int(len(uniques))
            probs_train = np.zeros((len(train_X), n_classes), dtype='float32')
            probs_test = np.zeros((len(test_X), n_classes), dtype='float32')
            splitter = split_for_aux(y_codes, task)
            for fold, (tr_idx, va_idx) in enumerate(splitter.split(Xo, y_codes), start=1):
                p_tr, p_te = fit_aux_classifier_fold(
                    Xo.iloc[tr_idx],
                    y_codes[tr_idx],
                    Xo.iloc[va_idx],
                    y_codes[va_idx],
                    Xt_train,
                    Xt_test,
                    n_classes,
                )
                probs_train += p_tr / N_SPLITS
                probs_test += p_te / N_SPLITS
            new_cols = []
            cname = f'PRED_{safe}_maxprob'
            aux_train_cols[cname] = probs_train.max(axis=1).astype('float32')
            aux_test_cols[cname] = probs_test.max(axis=1).astype('float32')
            new_cols.append(cname)
            cname = f'PRED_{safe}_entropy'
            aux_train_cols[cname] = entropy_from_probs(probs_train)
            aux_test_cols[cname] = entropy_from_probs(probs_test)
            new_cols.append(cname)
            if n_classes == 2:
                cname = f'PRED_{safe}_p1'
                aux_train_cols[cname] = probs_train[:, 1].astype('float32')
                aux_test_cols[cname] = probs_test[:, 1].astype('float32')
                new_cols.append(cname)
            log(f'[{j:02d}/{len(base_feats)}] {tgt_col}: CLF({n_classes}) -> {len(new_cols)} cols')

    aux_train = pd.DataFrame(aux_train_cols)
    aux_test = pd.DataFrame(aux_test_cols)
    log(f'Aux feature frames: train={aux_train.shape}, test={aux_test.shape}')
    return aux_train, aux_test


def build_features():
    orig = read_original()[ORIGINAL_COLUMN_ORDER]
    train = read_train()
    test = read_test()
    orig, train, test, orig_X, train_X, test_X, base_feats, _, _ = prepare_base_frames(orig, train, test)
    aux_train, aux_test = make_aux_features(orig_X, train_X, test_X, base_feats)
    X_train = pd.concat([train_X.reset_index(drop=True), aux_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_X.reset_index(drop=True), aux_test.reset_index(drop=True)], axis=1)
    y = train[TARGET].astype('int8').reset_index(drop=True)
    log(f'Final matrices: X_train={X_train.shape}, X_test={X_test.shape}')
    return X_train, X_test, y


def main():
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X_train, X_test, y = build_features()
        f = folds(y.values)
        for k in todo:
            tr_idx, va_idx = np.flatnonzero(f != k), np.flatnonzero(f == k)
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            model = XGBClassifier(**FINAL_XGB_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
            va_pred = model.predict_proba(X_va)[:, 1].astype('float32')
            te_pred = model.predict_proba(X_test)[:, 1].astype('float32')
            save_part(HERE, PAIR, k, va_pred, te_pred)
            log(f'{PAIR} fold {k}: AUC {roc_auc_score(y_va, va_pred):.6f}, best_iteration {getattr(model, "best_iteration", None)}')
            del model, X_tr, X_va
            gc.collect()
    y = read_train()[TARGET].values.astype('int8')
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
