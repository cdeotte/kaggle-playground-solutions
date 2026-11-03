#!/usr/bin/env python
"""GPU XGBoost on target / count encodings of all pairwise combinations of the 14 columns, with the original rows appended
to every training fold (pair xgb_pairwise_combo_te_original).

Features: the 4 numerics with >= 100 levels (float32) and the 10 other columns as integer categories (codes over
train + test + original); the 91 pairwise combinations of the 14 columns (string concatenation, factorized over
train + test + original). Per outer fold, the training side = the competition rows of the 4 other folds + all 101,371
original rows (weight 1); every combination gets a smoothed (10) target mean, out-of-fold over a random 5-way split of
the training side (numpy default_rng(42)), fitted on the whole training side for the validation / test rows, and its
count on the training side; the raw combination codes are dropped.
Model: depth 8, lr 0.01, colsample_bytree 0.341, subsample 0.899, lambda 4.06, alpha 2.92, up to 10,000 rounds, early
stopping 200 on the validation fold, random_state 42. Test = mean of the 5 fold models.
Usage: python train_pairwise_combos.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.simplefilter(action='ignore', category=PerformanceWarning)

PAIR = "xgb_pairwise_combo_te_original"
ID_COL = ID
RANDOM_STATE = 42
PAIR_R = 2
TE_KFOLD = 5
TE_SMOOTH = 10
CARDINALITY_CAT_THRESHOLD = 100

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
    tree_method='hist',
    n_jobs=32,
)


def log(msg):
    print(msg, flush=True)


def to_binary_target(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype('int8')
    return s.astype(str).str.strip().str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0}).astype('int8')


def infer_groups(train, test):
    cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
    both = pd.concat([train[cols], test[cols]], axis=0, ignore_index=True)
    cats, nums = [], []
    for c in cols:
        if pd.api.types.is_object_dtype(both[c]) or pd.api.types.is_string_dtype(both[c]) or both[c].nunique(dropna=False) < CARDINALITY_CAT_THRESHOLD:
            cats.append(c)
        else:
            nums.append(c)
    return cats, nums


def target_encode_combo(train_with_y, valid, test, col, target, kfold=5, smooth=20.0, seed=42):
    train_local = train_with_y.copy()
    rng = np.random.default_rng(seed)
    fold_id = np.arange(len(train_local)) % kfold
    rng.shuffle(fold_id)
    train_local['kfold'] = fold_id
    out_col = f'TE_MEAN_{col}'
    train_local[out_col] = 0.0
    prior = float(train_local[target].mean())
    for i in range(kfold):
        df_tmp = train_local[train_local['kfold'] != i]
        grp = df_tmp.groupby(col)[target].agg(['mean', 'count']).reset_index()
        grp['TE_tmp'] = ((grp['mean'] * grp['count']) + (prior * smooth)) / (grp['count'] + smooth)
        merged = train_local[[col, 'kfold']].merge(grp[[col, 'TE_tmp']], how='left', on=col)
        m = merged['TE_tmp'].fillna(prior).values
        mask = train_local['kfold'].values == i
        train_local.loc[mask, out_col] = m[mask]
    grp = train_local.groupby(col)[target].agg(['mean', 'count']).reset_index()
    grp['TE_tmp'] = ((grp['mean'] * grp['count']) + (prior * smooth)) / (grp['count'] + smooth)
    valid[out_col] = valid[[col]].merge(grp[[col, 'TE_tmp']], how='left', on=col)['TE_tmp'].fillna(prior).astype('float32').values
    test[out_col] = test[[col]].merge(grp[[col, 'TE_tmp']], how='left', on=col)['TE_tmp'].fillna(prior).astype('float32').values
    train_local[out_col] = train_local[out_col].astype('float32')
    return train_local[[out_col]], valid[[out_col]], test[[out_col]]


def count_encode(train, valid, test, col):
    out_col = f'CE_{col}'
    counts = train[col].value_counts()
    train[out_col] = train[col].map(counts).astype('float32')
    valid[out_col] = valid[col].map(counts).fillna(0).astype('float32')
    test[out_col] = test[col].map(counts).fillna(0).astype('float32')
    return train[[out_col]], valid[[out_col]], test[[out_col]]


def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = to_binary_target(train[TARGET])
    orig[TARGET] = to_binary_target(orig[TARGET])

    feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL] and c in test.columns and c in orig.columns]
    train = train[[TARGET] + feature_cols].copy()
    test = test[feature_cols].copy()
    orig = orig[[TARGET] + feature_cols].copy()
    cats, nums = infer_groups(train, test)
    log(f'NUMS ({len(nums)}): {nums}')
    log(f'CATS ({len(cats)}): {cats}')

    for df in [train, test, orig]:
        for c in nums:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        for c in cats:
            df[c] = df[c].astype('string').fillna('__NA__').astype(str)

    te_columns = []
    combo_sources = nums + cats
    log(f'Building pairwise combos over {len(combo_sources)} columns')
    for cols in combinations(combo_sources, PAIR_R):
        name = '-'.join(cols).replace(' ', '_').replace('(', '').replace(')', '')
        for df in [train, test, orig]:
            s = df[cols[0]].astype(str)
            for cc in cols[1:]:
                s = s + '_' + df[cc].astype(str)
            df[name] = s
        combined = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
        codes, _ = combined.factorize(sort=False)
        train[name] = codes[:len(train)].astype('int32')
        test[name] = codes[len(train):len(train) + len(test)].astype('int32')
        orig[name] = codes[len(train) + len(test):].astype('int32')
        te_columns.append(name)

    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]], ignore_index=True)
        codes, _ = combined.factorize(sort=False)
        train[c] = pd.Series(codes[:len(train)], index=train.index).astype('int32').astype('category')
        test[c] = pd.Series(codes[len(train):len(train) + len(test)], index=test.index).astype('int32').astype('category')
        orig[c] = pd.Series(codes[len(train) + len(test):], index=orig.index).astype('int32').astype('category')

    features = [c for c in train.columns if c != TARGET]
    log(f'Base+combo feature count before encoding: {len(features)}; combos={len(te_columns)}')
    return train, test, orig, features, te_columns


def run_fold(tr_idx, va_idx, train, test, orig, features, te_columns):
    from xgboost import XGBClassifier

    X_train = train.iloc[tr_idx][features].copy()
    y_train = train.iloc[tr_idx][TARGET].copy()
    X_val = train.iloc[va_idx][features].copy()
    y_val = train.iloc[va_idx][TARGET].copy()
    X_test = test[features].copy()

    # Original rows join the training side of every fold.
    X_train = pd.concat([X_train, orig[features]], axis=0, ignore_index=True)
    y_train = pd.concat([y_train, orig[TARGET]], axis=0, ignore_index=True)

    encoded_train, encoded_val, encoded_test = {}, {}, {}
    for j, col in enumerate(te_columns, start=1):
        Xy = pd.concat([X_train[[col]].copy(), y_train.rename(TARGET)], axis=1)
        tr_te, va_te, te_te = target_encode_combo(Xy, X_val[[col]].copy(), X_test[[col]].copy(), col, TARGET, kfold=TE_KFOLD, smooth=TE_SMOOTH, seed=RANDOM_STATE)
        encoded_train[tr_te.columns[0]] = tr_te.iloc[:, 0].values
        encoded_val[va_te.columns[0]] = va_te.iloc[:, 0].values
        encoded_test[te_te.columns[0]] = te_te.iloc[:, 0].values
        tr_ce, va_ce, te_ce = count_encode(X_train[[col]].copy(), X_val[[col]].copy(), X_test[[col]].copy(), col)
        encoded_train[tr_ce.columns[0]] = tr_ce.iloc[:, 0].values
        encoded_val[va_ce.columns[0]] = va_ce.iloc[:, 0].values
        encoded_test[te_ce.columns[0]] = te_ce.iloc[:, 0].values
        if j % 20 == 0:
            log(f'encoded {j}/{len(te_columns)} combos')

    X_train = pd.concat([X_train.drop(columns=te_columns).reset_index(drop=True), pd.DataFrame(encoded_train).reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_val.drop(columns=te_columns).reset_index(drop=True), pd.DataFrame(encoded_val).reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.drop(columns=te_columns).reset_index(drop=True), pd.DataFrame(encoded_test).reset_index(drop=True)], axis=1)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    # The categories are global integer codes already; train on the integer labels (the pandas category indexes of the
    # training / validation / test frames can differ).
    for frame in (X_train, X_val, X_test):
        for c in frame.columns:
            if isinstance(frame[c].dtype, pd.CategoricalDtype):
                frame[c] = frame[c].astype('int32')
            elif not pd.api.types.is_numeric_dtype(frame[c]):
                frame[c] = pd.to_numeric(frame[c], errors='coerce').fillna(0).astype('float32')

    model = XGBClassifier(**XGB_PARAM_GRID, **XGB_FIXED)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
    p_val = model.predict_proba(X_val)[:, 1].astype('float32')
    p_tst = model.predict_proba(X_test)[:, 1].astype('float32')
    best_iter = getattr(model, 'best_iteration', None)
    del model, X_train, X_val, X_test, y_train
    gc.collect()
    return p_val, p_tst, y_val, best_iter


def main():
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train, test, orig, features, te_columns = build_features()
        f = folds(train[TARGET].values)
        for k in todo:
            p_val, p_tst, y_val, best_iter = run_fold(np.flatnonzero(f != k), np.flatnonzero(f == k), train, test, orig,
                                                      features, te_columns)
            save_part(HERE, PAIR, k, p_val, p_tst)
            log(f'{PAIR} fold {k}: AUC {roc_auc_score(y_val, p_val):.6f}, best_iteration {best_iter}')
    y = read_train()[TARGET].values.astype('int8')
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
