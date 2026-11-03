#!/usr/bin/env python
"""GPU XGBoost with inner-fold target statistics around a raw TyreLife anchor (pair xgb_tyrelife_anchor_te).
Competition rows only (no original data).

Base features (86): the 10 columns with < 100 levels (strings and low-cardinality numerics) as integer category codes
(factorized over train + test), the 4 other numerics, the raw TyreLife anchor; category x anchor keys
(code * 1e6 + TyreLife), 6 anchor digits and the 10 pairs of the first 5 digits; the 45 pairs of the 10 categories.
Per outer fold, out-of-fold over an inner KFold(5, shuffle=True, random_state=42) of the training rows (all training rows
for the validation / test rows): target mean / std / count / nunique / median / min / max / skew and the 5, 10, 40, 45,
55, 60, 90, 95 % quantiles per anchor value, the target mean of every category x anchor key, digit pair and category pair
(fallback: the fold prior), count / nunique and std / count ratios. The category-pair ids are dropped after encoding.
Model: categories and digits as XGBoost categoricals, depth 6, colsample_bynode 0.3, subsample 0.8, min_child_weight 10,
lr 0.01, up to 50,000 rounds, early stopping 500 on the validation fold, random_state 42. Test = mean of the 5 fold models.
Usage: python train_tyrelife_anchor.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = "xgb_tyrelife_anchor_te"
ID_COL = ID
INNER_FOLDS = 5
RANDOM_STATE = 42
CARDINALITY_CAT_THRESHOLD = 100
ANCHOR_SOURCE_COL = 'TyreLife'
ANCHOR_COL = 'ANCHOR_TyreLife_raw'
STATS_TE1 = ['mean', 'std', 'count', 'nunique', 'median', 'min', 'max', 'skew']
QUANTILES = [5, 10, 40, 45, 55, 60, 90, 95]
MAX_PAIR_CATS = 1000

XGB_PARAMS = dict(
    device='cuda',
    tree_method='hist',
    enable_categorical=True,
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=6,
    colsample_bynode=0.3,
    subsample=0.8,
    min_child_weight=10,
    learning_rate=0.01,
    n_estimators=50_000,
    early_stopping_rounds=500,
    random_state=RANDOM_STATE,
    n_jobs=32,
)


def log(msg):
    print(msg, flush=True)


def to_binary_target(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype('int8')
    return s.astype(str).str.strip().str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0}).astype('int8')


def safe_map(keys: pd.Series, mapping: pd.Series, default: float) -> np.ndarray:
    return keys.map(mapping).fillna(default).astype('float32').to_numpy()


def factorize_int_codes(train_s: pd.Series, test_s: pd.Series):
    comb = pd.concat([train_s, test_s], axis=0, ignore_index=True).astype(str).fillna('__NA__')
    codes, _ = pd.factorize(comb, sort=False)
    codes = codes.astype('int32')
    return pd.Series(codes[:len(train_s)], index=train_s.index), pd.Series(codes[len(train_s):], index=test_s.index)


def infer_feature_groups(train, test):
    feats = [c for c in train.columns if c not in [TARGET, ID_COL]]
    both = pd.concat([train[feats], test[feats]], axis=0, ignore_index=True)
    cat_cols, num_cols = [], []
    for c in feats:
        if pd.api.types.is_object_dtype(both[c]) or pd.api.types.is_string_dtype(both[c]) or both[c].nunique(dropna=False) < CARDINALITY_CAT_THRESHOLD:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return cat_cols, num_cols


def make_ratio_block(df: pd.DataFrame, te_count: str, te_nuniq: str, te_std: str) -> pd.DataFrame:
    out = {}
    if te_count in df.columns and te_nuniq in df.columns:
        denom = df[te_nuniq].replace(0, np.nan)
        out['TE1_count_per_nunique'] = (df[te_count] / denom).astype('float32').to_numpy()
    if te_std in df.columns and te_count in df.columns:
        denom = df[te_count].replace(0, np.nan)
        out['TE1_std_per_count'] = (df[te_std] / denom).astype('float32').to_numpy()
    if out:
        df = pd.concat([df, pd.DataFrame(out)], axis=1)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_features():
    train = read_train()
    test = read_test()
    y = to_binary_target(train[TARGET]).reset_index(drop=True)
    train = train.drop(columns=[TARGET])

    cat_cols, num_cols = infer_feature_groups(pd.concat([train, y.rename(TARGET)], axis=1), test)
    train[ANCHOR_COL] = pd.to_numeric(train[ANCHOR_SOURCE_COL], errors='coerce').astype('float32')
    test[ANCHOR_COL] = pd.to_numeric(test[ANCHOR_SOURCE_COL], errors='coerce').astype('float32')
    if ANCHOR_COL not in num_cols:
        num_cols = num_cols + [ANCHOR_COL]

    for c in num_cols:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
            test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')

    log(f'Base cat cols ({len(cat_cols)}): {cat_cols}')
    log(f'Base num cols ({len(num_cols)}): {num_cols}')

    for c in cat_cols:
        tr_c, te_c = factorize_int_codes(train[c], test[c])
        train[c] = tr_c
        test[c] = te_c

    new_train, new_test = {}, {}
    combo, digits, pairs = [], [], []
    anchor_tr = train[ANCHOR_COL].astype('float32').fillna(0)
    anchor_te = test[ANCHOR_COL].astype('float32').fillna(0)

    for c in cat_cols:
        n = f'{c}_anc'
        new_train[n] = (train[c].astype('float32') * 1_000_000.0 + anchor_tr).astype('float32')
        new_test[n] = (test[c].astype('float32') * 1_000_000.0 + anchor_te).astype('float32')
        combo.append(n)

    for k in range(1, 7):
        dn = f'digit{k}'
        new_train[dn] = (((anchor_tr * (10 ** k)) % 10)).astype('int8')
        new_test[dn] = (((anchor_te * (10 ** k)) % 10)).astype('int8')
        digits.append(dn)

    for i in range(4):
        for j in range(i + 1, 5):
            n = f'digit_{i+1}_{j+1}'
            new_train[n] = ((new_train[f'digit{i+1}'].astype('int16') + 1) * 11 + (new_train[f'digit{j+1}'].astype('int16') + 1)).astype('int16')
            new_test[n] = ((new_test[f'digit{i+1}'].astype('int16') + 1) * 11 + (new_test[f'digit{j+1}'].astype('int16') + 1)).astype('int16')
            combo.append(n)

    cat_cols_for_pairs = cat_cols[:MAX_PAIR_CATS]
    for i, c1 in enumerate(cat_cols_for_pairs[:-1]):
        c1_tr = train[c1].astype('int64')
        c1_te = test[c1].astype('int64')
        for c2 in cat_cols_for_pairs[i + 1:]:
            n = f'{c1}__{c2}'
            c2_tr = train[c2].astype('int64')
            c2_te = test[c2].astype('int64')
            m2 = int(max(c2_tr.max(), c2_te.max()) + 2)
            new_train[n] = ((c1_tr + 1) * m2 + (c2_tr + 1)).astype('int32')
            new_test[n] = ((c1_te + 1) * m2 + (c2_te + 1)).astype('int32')
            combo.append(n)
            pairs.append(n)

    train = pd.concat([train, pd.DataFrame(new_train)], axis=1).copy()
    test = pd.concat([test, pd.DataFrame(new_test)], axis=1).copy()
    features = cat_cols + num_cols + combo + digits
    cat_for_xgb = set(cat_cols + digits)
    log(f'Final base feature count before TE: {len(features)}')
    return train, test, y, features, combo, pairs, cat_for_xgb


def run_fold(tr_idx, va_idx, train, test, y, features, combo, pairs, cat_for_xgb):
    from xgboost import XGBClassifier

    Xtr_base = train.iloc[tr_idx][features].reset_index(drop=True)
    ytr = y.iloc[tr_idx].reset_index(drop=True)
    Xva_base = train.iloc[va_idx][features].reset_index(drop=True)
    yva = y.iloc[va_idx].reset_index(drop=True)
    Xte_base = test[features].reset_index(drop=True)
    ntr = len(Xtr_base)

    inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    te1_names = [f'TE1_{ANCHOR_COL}_{s}' for s in STATS_TE1]
    teq_names = [f'quantile_{q}' for q in QUANTILES]
    te2_names = [f'TE2_{col}_mean' for col in combo]
    te1_store = {n: np.full(ntr, np.nan, dtype='float32') for n in te1_names}
    teq_store = {n: np.full(ntr, np.nan, dtype='float32') for n in teq_names}
    te2_store = {n: np.full(ntr, np.nan, dtype='float32') for n in te2_names}
    prior = float(ytr.mean())

    for tr2_idx, va2_idx in inner.split(Xtr_base):
        Xtr2 = Xtr_base.iloc[tr2_idx]
        ytr2 = ytr.iloc[tr2_idx]
        g = pd.DataFrame({ANCHOR_COL: Xtr2[ANCHOR_COL], '_y_': ytr2.values})
        agg = g.groupby(ANCHOR_COL)['_y_'].agg(STATS_TE1)
        keys_va = Xtr_base.iloc[va2_idx][ANCHOR_COL]
        for s in STATS_TE1:
            te1_store[f'TE1_{ANCHOR_COL}_{s}'][va2_idx] = safe_map(keys_va, agg[s], prior)
        for q in QUANTILES:
            qmap = g.groupby(ANCHOR_COL)['_y_'].quantile(q / 100.0)
            teq_store[f'quantile_{q}'][va2_idx] = safe_map(keys_va, qmap, prior)
        for col in combo:
            m = pd.DataFrame({col: Xtr2[col], '_y_': ytr2.values}).groupby(col)['_y_'].mean()
            te2_store[f'TE2_{col}_mean'][va2_idx] = safe_map(Xtr_base.iloc[va2_idx][col], m, prior)

    Xtr = pd.concat([Xtr_base, pd.DataFrame(te1_store), pd.DataFrame(teq_store), pd.DataFrame(te2_store)], axis=1)

    gfull = pd.DataFrame({ANCHOR_COL: Xtr_base[ANCHOR_COL], '_y_': ytr.values})
    agg_full = gfull.groupby(ANCHOR_COL)['_y_'].agg(STATS_TE1)
    va_te, te_te = {}, {}
    for s in STATS_TE1:
        name = f'TE1_{ANCHOR_COL}_{s}'
        va_te[name] = safe_map(Xva_base[ANCHOR_COL], agg_full[s], prior)
        te_te[name] = safe_map(Xte_base[ANCHOR_COL], agg_full[s], prior)
    for q in QUANTILES:
        qmap = gfull.groupby(ANCHOR_COL)['_y_'].quantile(q / 100.0)
        name = f'quantile_{q}'
        va_te[name] = safe_map(Xva_base[ANCHOR_COL], qmap, prior)
        te_te[name] = safe_map(Xte_base[ANCHOR_COL], qmap, prior)
    for col in combo:
        m = pd.DataFrame({col: Xtr_base[col], '_y_': ytr.values}).groupby(col)['_y_'].mean()
        name = f'TE2_{col}_mean'
        va_te[name] = safe_map(Xva_base[col], m, prior)
        te_te[name] = safe_map(Xte_base[col], m, prior)

    Xva = pd.concat([Xva_base, pd.DataFrame(va_te)], axis=1)
    Xte = pd.concat([Xte_base, pd.DataFrame(te_te)], axis=1)
    Xtr = make_ratio_block(Xtr, f'TE1_{ANCHOR_COL}_count', f'TE1_{ANCHOR_COL}_nunique', f'TE1_{ANCHOR_COL}_std')
    Xva = make_ratio_block(Xva, f'TE1_{ANCHOR_COL}_count', f'TE1_{ANCHOR_COL}_nunique', f'TE1_{ANCHOR_COL}_std')
    Xte = make_ratio_block(Xte, f'TE1_{ANCHOR_COL}_count', f'TE1_{ANCHOR_COL}_nunique', f'TE1_{ANCHOR_COL}_std')

    if pairs:
        drop_cols = [c for c in pairs if c in Xtr.columns]
        Xtr = Xtr.drop(columns=drop_cols)
        Xva = Xva.drop(columns=drop_cols)
        Xte = Xte.drop(columns=drop_cols)

    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xva = Xva.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(0)
    cols = list(Xtr.columns)
    feature_types = ['c' if c in cat_for_xgb else 'q' for c in cols]
    model = XGBClassifier(**XGB_PARAMS, feature_types=feature_types)
    model.fit(Xtr[cols], ytr, eval_set=[(Xva[cols], yva)], verbose=500)
    oof_fold = model.predict_proba(Xva[cols])[:, 1].astype('float32')
    test_fold = model.predict_proba(Xte[cols])[:, 1].astype('float32')
    best_iter = getattr(model, 'best_iteration', None)
    del Xtr_base, Xva_base, Xte_base, Xtr, Xva, Xte, model
    gc.collect()
    return oof_fold, test_fold, yva, best_iter


def main():
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train, test, y, features, combo, pairs, cat_for_xgb = build_features()
        f = folds(y.values)
        for k in todo:
            oof_fold, test_fold, yva, best_iter = run_fold(np.flatnonzero(f != k), np.flatnonzero(f == k), train, test, y,
                                                           features, combo, pairs, cat_for_xgb)
            save_part(HERE, PAIR, k, oof_fold, test_fold)
            log(f'{PAIR} fold {k}: AUC {roc_auc_score(yva, oof_fold):.6f}, best_iteration {best_iter}')
    y = read_train()[TARGET].values.astype('int8')
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
