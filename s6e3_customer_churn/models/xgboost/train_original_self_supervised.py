#!/usr/bin/env python
"""GPU XGBoost with self-supervised column predictions learned on the original data (pair xgb_original_self_supervised).

Auxiliary models (the target is never used): for each of the 19 columns shared by train, test and the original data
(ids dropped), XGBoost models trained on the 7,043 original rows predict that column from the 18 others (5-fold split
of the original rows, random_state 42; early stopping 200 on the held-out original fold); the 5 models' predictions for
the train and test rows are averaged. Numeric columns with > 20 levels (tenure, MonthlyCharges, TotalCharges) are
regressed (PRED_<col>); the other columns are classified on their category codes (StratifiedKFold split):
PRED_<col>_maxprob, PRED_<col>_entropy and, for the binary columns, PRED_<col>_p1 (41 PRED_* columns).
Aux params: depth 4, lr 0.05, min_child_weight 1, subsample 0.9, colsample_bytree 0.9, up to 50,000 rounds, native
categoricals (string columns as pandas categories aligned over original + train + test).
Final model on the 19 columns + the PRED_* columns: depth 3, lr 0.1, min_child_weight 5, subsample 0.85,
colsample_bytree 0.85, native categoricals, up to 100,000 rounds, early stopping 200 on the validation fold,
random_state 42. Competition rows only. Test = mean of the 5 fold models.
Usage: python train_original_self_supervised.py"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

PAIR = 'xgb_original_self_supervised'
DROP_COLS = ['customerID', 'id']
N_SPLITS = 5
RANDOM_STATE = 42
DEVICE = 'cuda'

AUX_N_ESTIMATORS = 50_000
AUX_LR = 0.05
AUX_EARLY_STOP = 200

FINAL_N_ESTIMATORS = 100_000
FINAL_LR = 0.1
FINAL_EARLY_STOP = 200

AUX_PARAMS = dict(
    n_estimators=AUX_N_ESTIMATORS,
    learning_rate=AUX_LR,
    max_depth=4,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method='hist',
    device=DEVICE,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

FINAL_PARAMS = dict(
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
    n_jobs=-1,
    early_stopping_rounds=FINAL_EARLY_STOP,
    device=DEVICE,
)


def log(msg):
    print(msg, flush=True)


def coerce_totalcharges(df):
    tc = df['TotalCharges']
    if tc.dtype == 'object':
        tc = tc.replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(tc, errors='coerce').astype('float32')
    return df


def entropy_from_probs(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)


def as_category_inplace(df):
    for c in df.select_dtypes(include=['object']).columns.tolist():
        df[c] = df[c].astype('category')


def make_aux_features(orig_X, train_X, test_X, base_feats):
    aux_train, aux_test = {}, {}
    log('Training auxiliary models on the original rows ...')
    for j, tgt_col in enumerate(base_feats, 1):
        Xo = orig_X.drop(columns=[tgt_col])
        Xt_train = train_X.drop(columns=[tgt_col])
        Xt_test = test_X.drop(columns=[tgt_col])
        y_raw = orig_X[tgt_col]
        task = 'reg' if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > 20 else 'clf'

        nunq = int(y_raw.nunique(dropna=True))
        if nunq <= 1:
            const_val = float(y_raw.dropna().iloc[0]) if y_raw.dropna().shape[0] else 0.0
            aux_train[f'PRED_{tgt_col}'] = np.full(len(train_X), const_val, dtype=np.float32)
            aux_test[f'PRED_{tgt_col}'] = np.full(len(test_X), const_val, dtype=np.float32)
            log(f'[{j:02d}/{len(base_feats)}] {tgt_col}: constant')
            continue

        if task == 'reg':
            y = pd.to_numeric(y_raw, errors='coerce').astype('float32').values
            pred_train = np.zeros(len(train_X), dtype=np.float32)
            pred_test = np.zeros(len(test_X), dtype=np.float32)
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            for tr_idx, va_idx in kf.split(Xo, y):
                X_tr, X_va = Xo.iloc[tr_idx], Xo.iloc[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]
                m_tr, m_va = ~np.isnan(y_tr), ~np.isnan(y_va)
                if m_tr.sum() < 50 or m_va.sum() < 20:
                    continue
                model = XGBRegressor(**AUX_PARAMS, objective='reg:squarederror', eval_metric='rmse',
                                     early_stopping_rounds=AUX_EARLY_STOP, enable_categorical=True)
                model.fit(X_tr.iloc[m_tr], y_tr[m_tr], eval_set=[(X_va.iloc[m_va], y_va[m_va])], verbose=False)
                pred_train += model.predict(Xt_train).astype(np.float32) / N_SPLITS
                pred_test += model.predict(Xt_test).astype(np.float32) / N_SPLITS
            aux_train[f'PRED_{tgt_col}'] = pred_train
            aux_test[f'PRED_{tgt_col}'] = pred_test
            log(f'[{j:02d}/{len(base_feats)}] {tgt_col}: regression -> PRED_{tgt_col}')
            continue

        y_codes = y_raw.astype('category').cat.codes.values
        if (y_codes == -1).any():
            y_codes = y_codes.copy()
            y_codes[y_codes == -1] = y_codes.max() + 1
        n_classes = int(np.max(y_codes) + 1)

        if n_classes > 50:
            y = y_codes.astype(np.float32)
            pred_train = np.zeros(len(train_X), dtype=np.float32)
            pred_test = np.zeros(len(test_X), dtype=np.float32)
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            for tr_idx, va_idx in kf.split(Xo, y):
                model = XGBRegressor(**AUX_PARAMS, objective='reg:squarederror', eval_metric='rmse',
                                     early_stopping_rounds=AUX_EARLY_STOP, enable_categorical=True)
                model.fit(Xo.iloc[tr_idx], y[tr_idx], eval_set=[(Xo.iloc[va_idx], y[va_idx])], verbose=False)
                pred_train += model.predict(Xt_train).astype(np.float32) / N_SPLITS
                pred_test += model.predict(Xt_test).astype(np.float32) / N_SPLITS
            aux_train[f'PRED_{tgt_col}_code'] = pred_train
            aux_test[f'PRED_{tgt_col}_code'] = pred_test
            log(f'[{j:02d}/{len(base_feats)}] {tgt_col}: many classes -> PRED_{tgt_col}_code')
            continue

        probs_train = np.zeros((len(train_X), n_classes), dtype=np.float32)
        probs_test = np.zeros((len(test_X), n_classes), dtype=np.float32)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for tr_idx, va_idx in skf.split(Xo, y_codes):
            if n_classes == 2:
                task_params = dict(objective='binary:logistic', eval_metric='logloss')
            else:
                task_params = dict(objective='multi:softprob', num_class=n_classes, eval_metric='mlogloss')
            model = XGBClassifier(**AUX_PARAMS, **task_params, early_stopping_rounds=AUX_EARLY_STOP,
                                  enable_categorical=True)
            model.fit(Xo.iloc[tr_idx], y_codes[tr_idx], eval_set=[(Xo.iloc[va_idx], y_codes[va_idx])], verbose=False)
            probs_train += model.predict_proba(Xt_train).astype(np.float32) / N_SPLITS
            probs_test += model.predict_proba(Xt_test).astype(np.float32) / N_SPLITS

        aux_train[f'PRED_{tgt_col}_maxprob'] = probs_train.max(axis=1).astype(np.float32)
        aux_test[f'PRED_{tgt_col}_maxprob'] = probs_test.max(axis=1).astype(np.float32)
        aux_train[f'PRED_{tgt_col}_entropy'] = entropy_from_probs(probs_train)
        aux_test[f'PRED_{tgt_col}_entropy'] = entropy_from_probs(probs_test)
        if n_classes == 2:
            aux_train[f'PRED_{tgt_col}_p1'] = probs_train[:, 1].astype(np.float32)
            aux_test[f'PRED_{tgt_col}_p1'] = probs_test[:, 1].astype(np.float32)
        log(f'[{j:02d}/{len(base_feats)}] {tgt_col}: {n_classes} classes')
        gc.collect()

    return pd.DataFrame(aux_train), pd.DataFrame(aux_test)


def build_features():
    orig = coerce_totalcharges(read_original())
    train = coerce_totalcharges(read_train())
    test = coerce_totalcharges(read_test())

    orig_feats = [c for c in orig.columns if c not in [TARGET] + DROP_COLS]
    train_feats = [c for c in train.columns if c not in [TARGET] + DROP_COLS]
    test_feats = [c for c in test.columns if c not in DROP_COLS]
    base_feats = [c for c in orig_feats if c in train_feats and c in test_feats]
    log(f'Base features ({len(base_feats)}): {base_feats}')

    all_base = pd.concat([orig[base_feats], train[base_feats], test[base_feats]], axis=0, ignore_index=True)
    as_category_inplace(all_base)
    n0, n1 = len(orig), len(train)
    orig_X = all_base.iloc[:n0].copy()
    train_X = all_base.iloc[n0:n0 + n1].copy()
    test_X = all_base.iloc[n0 + n1:].copy()
    del all_base
    gc.collect()

    aux_train, aux_test = make_aux_features(orig_X, train_X, test_X, base_feats)
    log(f'Auxiliary features: {aux_train.shape[1]}')

    y = (train[TARGET].astype(str).str.strip().str.lower() == 'yes').astype(np.int32).values
    X_train = pd.concat([train_X.reset_index(drop=True), aux_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_X.reset_index(drop=True), aux_test.reset_index(drop=True)], axis=1)
    log(f'X_train {X_train.shape}, X_test {X_test.shape}')
    return X_train, X_test, y


def run_cv(X_train, X_test, y, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]
        model = XGBClassifier(**FINAL_PARAMS)
        model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=500)
        oof[val_idx] = model.predict_proba(X_val)[:, 1].astype(np.float32)
        test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32) / n_folds
        log(f'{PAIR} fold {fold}/{n_folds}: AUC {roc_auc_score(y_val, oof[val_idx]):.6f}, best_iteration '
            f'{getattr(model, "best_iteration", None)}')
        del model
        gc.collect()
    log(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}')
    return oof, test_pred


def main():
    X_train, X_test, y = build_features()
    oof, test_pred = run_cv(X_train, X_test, y, N_SPLITS)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
