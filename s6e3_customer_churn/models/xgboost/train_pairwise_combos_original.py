#!/usr/bin/env python
"""GPU XGBoost on target / count encodings of all pairwise combinations of the 19 columns, with the original rows
appended to every training fold (pair xgb_pairwise_combo_te_original).

Features: the 4 numeric columns (SeniorCitizen, tenure, MonthlyCharges, TotalCharges) and the 15 string columns as
pandas categories (native XGBoost categoricals); the 171 pairwise combinations of these 19 columns (string
concatenation, factorized over train + test + original rows). Per outer fold, the training side = the competition rows
of the other folds + all 7,043 original rows; every combination gets a smoothed (10) target mean, out-of-fold over a
random 5-way split of the training side (numpy default_rng(42)) and fitted on the whole training side for the
validation / test rows (unseen: the training-side mean), and its count on the training side; the raw combination codes
are then dropped (361 features).
Model: depth 8, lr 0.01, colsample_bytree 0.341, subsample 0.899, lambda 4.06, alpha 2.92, up to 10,000 rounds, early
stopping 200 on the validation fold, random_state 42.
Usage: python train_pairwise_combos_original.py"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.simplefilter(action='ignore', category=PerformanceWarning)

PAIR = 'xgb_pairwise_combo_te_original'
DROP_COLS = ['id', 'customerID']
N_SPLITS = 5
TE_KFOLD = 5
TEST_FOLDS = 15
TEST_TE_KFOLD = 10
RANDOM_STATE = 42
PAIR_R = 2
TE_SMOOTH = 10

XGB_PARAM_GRID = {
    'colsample_bytree': 0.3413131357034518,
    'subsample': 0.8991973526634904,
    'reg_lambda': 4.063399595895267,
    'reg_alpha': 2.917406632455954,
    'max_depth': 8,
}
XGB_FIXED = dict(
    n_estimators=10000,
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.01,
    early_stopping_rounds=200,
    random_state=RANDOM_STATE,
    enable_categorical=True,
    device='cuda',
    n_jobs=-1,
)


def log(msg):
    print(msg, flush=True)


def to_binary_yesno(series):
    return (series.astype(str).str.strip().str.lower() == 'yes').astype(np.int32)


def clean_total_charges(df):
    out = df.copy()
    if out['TotalCharges'].dtype == 'object':
        x = out['TotalCharges'].astype(str)
        out['TotalCharges'] = pd.to_numeric(x.where(x.str.strip() != '', np.nan), errors='coerce')
    out['TotalCharges'] = out['TotalCharges'].astype('float32')
    return out


def target_encode(train_with_y, valid, test, col, kfold, smooth, seed):
    """Smoothed target mean of `col`: out-of-fold over a shuffled kfold split of the training side, fitted on the whole
    training side for the validation / test rows."""
    train = train_with_y.copy()
    rng = np.random.default_rng(seed)
    fold_id = np.arange(len(train)) % kfold
    rng.shuffle(fold_id)
    train['kfold'] = fold_id
    out_col = f'TE_MEAN_{col}'
    train[out_col] = 0.0
    mn = float(train[TARGET].mean())

    for i in range(kfold):
        df_tmp = train[train['kfold'] != i]
        grp = df_tmp[[col, TARGET]].groupby([col]).agg(['mean', 'count']).reset_index()
        grp.columns = [col, 'mean', 'count']
        grp['TE_tmp'] = ((grp['mean'] * grp['count']) + (mn * smooth)) / (grp['count'] + smooth)
        merged = train[[col, 'kfold']].merge(grp, how='left', on=[col])
        m = merged['TE_tmp'].fillna(mn).values
        train.loc[train['kfold'] == i, out_col] = m[train['kfold'].values == i]

    grp = train[[col, TARGET]].groupby([col]).agg(['mean', 'count']).reset_index()
    grp.columns = [col, 'mean', 'count']
    grp['TE_tmp'] = ((grp['mean'] * grp['count']) + (mn * smooth)) / (grp['count'] + smooth)
    va = valid[[col]].merge(grp, how='left', on=[col])['TE_tmp'].fillna(mn).astype('float32').values
    te = test[[col]].merge(grp, how='left', on=[col])['TE_tmp'].fillna(mn).astype('float32').values
    return train[out_col].astype('float32').values, va, te


def count_encode(train, valid, test, col):
    counts = train[col].value_counts()
    return (train[col].map(counts).astype('float32').values,
            valid[col].map(counts).fillna(0).astype('float32').values,
            test[col].map(counts).fillna(0).astype('float32').values)


def build_features():
    train = clean_total_charges(read_train())
    test = clean_total_charges(read_test())
    orig = clean_total_charges(read_original())
    for df in (train, test, orig):
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    train[TARGET] = to_binary_yesno(train[TARGET])
    orig[TARGET] = to_binary_yesno(orig[TARGET])

    all_cols = [c for c in train.columns if c != TARGET]
    cats = [c for c in all_cols if train[c].dtype == 'object' or str(train[c].dtype) == 'boolean']
    nums = [c for c in all_cols if c not in cats]
    for df in (train, test, orig):
        for c in cats:
            df[c] = df[c].astype('category')
    log(f'NUMS ({len(nums)}): {nums}')
    log(f'CATS ({len(cats)}): {cats}')

    te_columns = []
    for cols in combinations(nums + cats, PAIR_R):
        name = '-'.join(cols)
        for df in (train, test, orig):
            s = df[cols[0]].astype(str)
            for cc in cols[1:]:
                s = s + '_' + df[cc].astype(str)
            df[name] = s
        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        codes, _ = combined.factorize()
        train[name] = codes[:len(train)]
        test[name] = codes[len(train):len(train) + len(test)]
        orig[name] = codes[len(train) + len(test):]
        te_columns.append(name)

    features = [c for c in train.columns if c != TARGET]
    log(f'{len(features)} base + combination columns, {len(te_columns)} combinations')
    return train, test, orig, features, te_columns


def run_cv(train, test, orig, features, te_columns, n_splits, te_kfold):
    oof = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_STATE, shuffle=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train[TARGET]), 1):
        X_train = train.iloc[tr_idx][features].copy()
        y_train = train.iloc[tr_idx][TARGET].copy()
        X_val = train.iloc[va_idx][features].copy()
        y_val = train.iloc[va_idx][TARGET].copy()
        X_test = test[features].copy()

        X_train = pd.concat([X_train, orig[features]], axis=0, ignore_index=True)
        y_train = pd.concat([y_train, orig[TARGET]], axis=0, ignore_index=True)

        enc_tr, enc_va, enc_te = {}, {}, {}
        for col in te_columns:
            Xy = pd.concat([X_train[[col]], y_train.rename(TARGET)], axis=1)
            tr_te, va_te, te_te = target_encode(Xy, X_val[[col]], X_test[[col]], col, te_kfold, TE_SMOOTH, RANDOM_STATE)
            enc_tr[f'TE_MEAN_{col}'], enc_va[f'TE_MEAN_{col}'], enc_te[f'TE_MEAN_{col}'] = tr_te, va_te, te_te
            tr_ce, va_ce, te_ce = count_encode(X_train, X_val, X_test, col)
            enc_tr[f'CE_{col}'], enc_va[f'CE_{col}'], enc_te[f'CE_{col}'] = tr_ce, va_ce, te_ce

        X_train = pd.concat([X_train.drop(columns=te_columns), pd.DataFrame(enc_tr, index=X_train.index)], axis=1)
        X_val = pd.concat([X_val.drop(columns=te_columns), pd.DataFrame(enc_va, index=X_val.index)], axis=1)
        X_test = pd.concat([X_test.drop(columns=te_columns), pd.DataFrame(enc_te, index=X_test.index)], axis=1)

        model = XGBClassifier(**XGB_PARAM_GRID, **XGB_FIXED)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
        p_val = model.predict_proba(X_val)[:, 1].astype(np.float32)
        oof[va_idx] = p_val
        pred += model.predict_proba(X_test)[:, 1].astype(np.float32)
        log(f'{n_splits} folds, fold {fold}: AUC {roc_auc_score(y_val, p_val):.6f}, best_iteration '
            f'{getattr(model, "best_iteration", None)}')
        del model, X_train, X_val, y_train, y_val, X_test
        gc.collect()

    pred /= n_splits
    log(f'{n_splits} folds: OOF AUC {roc_auc_score(train[TARGET].values, oof):.6f}')
    return oof, pred


def main():
    data = build_features()
    oof, _ = run_cv(*data, N_SPLITS, TE_KFOLD)
    _, test_pred = run_cv(*data, TEST_FOLDS, TEST_TE_KFOLD)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
