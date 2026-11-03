#!/usr/bin/env python
"""GPU XGBoost with inner-fold target statistics around a TotalCharges or tenure anchor (2 presets).

Presets:
  xgb_total_round1_anchor_te  anchor TotalCharges; target statistics keyed by round(TotalCharges, 1), and that key is
                              also a numeric feature
  xgb_tenure_anchor_te        anchor tenure; target statistics keyed by the raw tenure value
Base features: the 15 string columns as integer codes (factorized over train + test), the 5 numeric columns (id,
SeniorCitizen, tenure, MonthlyCharges, TotalCharges), category x anchor keys (code x 1e6 + anchor), 6 anchor digits
(the k-th digit of anchor x 10^k, k = 1..6) and the 10 pairs of the first 5 digits, the 105 pairs of the 15 categories.
Per outer fold, out-of-fold over an inner KFold(5, shuffle=True, random_state=42) of the training rows (all training
rows for the validation / test rows): target mean / std / count / nunique / median / min / max / skew and the 5, 10, 40,
45, 55, 60, 90, 95 % quantiles per key value, the target mean of every category x anchor key, digit pair and category
pair (unseen: missing), count / nunique and std / count ratios. The category-pair ids are dropped after encoding.
Competition rows only.
Model: categories and digits as XGBoost categoricals (feature_types), depth 6, colsample_bynode 0.3, subsample 0.8,
min_child_weight 10, lr 0.01, up to 50,000 rounds, early stopping 500 on the validation fold.
Usage: python train_total_tenure_anchor_te.py {xgb_total_round1_anchor_te,xgb_tenure_anchor_te}"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, labels, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

PRESETS = {
    'xgb_total_round1_anchor_te': dict(anchor='TotalCharges', round_anchor=1, test_folds=15, test_inner_folds=10),
    'xgb_tenure_anchor_te': dict(anchor='tenure', round_anchor=None),
}

OUTER_FOLDS = 5
INNER_FOLDS = 5
RANDOM_STATE = 42

STATS_TE1 = ['mean', 'std', 'count', 'nunique', 'median', 'min', 'max', 'skew']
QUANTILES = [5, 10, 40, 45, 55, 60, 90, 95]

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
)


def coerce_numeric_inplace(df, col):
    if df[col].dtype == object or str(df[col].dtype).startswith('string'):
        df[col] = pd.to_numeric(df[col].replace(' ', np.nan), errors='coerce')
    df[col] = df[col].astype('float32')


def factorize_int_codes(train_s, test_s):
    comb = pd.concat([train_s, test_s], axis=0, ignore_index=True)
    codes, _ = pd.factorize(comb, sort=False)
    codes = codes.astype('int32')
    return pd.Series(codes[:len(train_s)], index=train_s.index), pd.Series(codes[len(train_s):], index=test_s.index)


def safe_map(keys, mapping):
    return keys.map(mapping).astype('float32').to_numpy()


def build_features(anchor_col, round_anchor):
    train = read_train()
    test = read_test()
    y = pd.Series(labels(train[TARGET]))
    train = train.drop(columns=[TARGET])
    coerce_numeric_inplace(train, anchor_col)
    coerce_numeric_inplace(test, anchor_col)

    cat_cols = [c for c in train.columns
                if train[c].dtype == object or str(train[c].dtype) == 'category' or train[c].dtype == bool]
    num_cols = [c for c in train.columns if c not in cat_cols]
    for c in cat_cols:
        train[c], test[c] = factorize_int_codes(train[c], test[c])

    new_train, new_test = {}, {}
    anchor_tr = train[anchor_col].astype('float32')
    anchor_te = test[anchor_col].astype('float32')

    combo = []
    for c in cat_cols:
        n = f'{c}_anc'
        new_train[n] = (train[c].astype('float32') * 1_000_000.0 + anchor_tr).astype('float32')
        new_test[n] = (test[c].astype('float32') * 1_000_000.0 + anchor_te).astype('float32')
        combo.append(n)

    digits = []
    anchor_fill_tr = anchor_tr.fillna(0.0)
    anchor_fill_te = anchor_te.fillna(0.0)
    for k in range(1, 7):
        dn = f'digit{k}'
        new_train[dn] = ((anchor_fill_tr * (10 ** k)) % 10).astype('int8')
        new_test[dn] = ((anchor_fill_te * (10 ** k)) % 10).astype('int8')
        digits.append(dn)

    for i in range(4):
        for j in range(i + 1, 5):
            n = f'digit_{i + 1}_{j + 1}'
            new_train[n] = ((new_train[f'digit{i + 1}'].astype('int16') + 1) * 11
                            + (new_train[f'digit{j + 1}'].astype('int16') + 1)).astype('int16')
            new_test[n] = ((new_test[f'digit{i + 1}'].astype('int16') + 1) * 11
                           + (new_test[f'digit{j + 1}'].astype('int16') + 1)).astype('int16')
            combo.append(n)

    pairs = []
    for i, c1 in enumerate(cat_cols[:-1]):
        c1_tr = train[c1].astype('int64')
        c1_te = test[c1].astype('int64')
        for c2 in cat_cols[i + 1:]:
            n = f'{c1}__{c2}'
            c2_tr = train[c2].astype('int64')
            c2_te = test[c2].astype('int64')
            m2 = int(c2_tr.max() + 2)
            new_train[n] = ((c1_tr + 1) * m2 + (c2_tr + 1)).astype('int32')
            new_test[n] = ((c1_te + 1) * m2 + (c2_te + 1)).astype('int32')
            combo.append(n)
            pairs.append(n)

    train = pd.concat([train, pd.DataFrame(new_train)], axis=1)
    test = pd.concat([test, pd.DataFrame(new_test)], axis=1)

    features = cat_cols + num_cols + combo + digits
    if round_anchor is None:
        key_col = anchor_col
    else:
        key_col = f'{anchor_col}__te1key'
        train[key_col] = train[anchor_col].round(int(round_anchor)).astype('float32')
        test[key_col] = test[anchor_col].round(int(round_anchor)).astype('float32')
        features = features + [key_col]
    print(f'{len(features)} base features, anchor {anchor_col}, key {key_col}', flush=True)
    return train.copy(), test.copy(), y, features, combo, pairs, set(cat_cols + digits), key_col


def add_ratio_block(df, key_col):
    te_count, te_nuniq, te_std = f'TE1_{key_col}_count', f'TE1_{key_col}_nunique', f'TE1_{key_col}_std'
    out = {}
    if te_count in df.columns and te_nuniq in df.columns:
        out['TE1_count_per_nunique'] = (df[te_count] / df[te_nuniq].replace(0, np.nan)).astype('float32').to_numpy()
    if te_std in df.columns and te_count in df.columns:
        out['TE1_std_per_count'] = (df[te_std] / df[te_count].replace(0, np.nan)).astype('float32').to_numpy()
    if out:
        df = pd.concat([df, pd.DataFrame(out)], axis=1)
    return df


def run_cv(train, test, y, features, combo, pairs, cat_for_xgb, key_col, outer_folds, inner_folds):
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(train), dtype='float32')
    pred = np.zeros(len(test), dtype='float32')

    for fold, (tr_idx, va_idx) in enumerate(outer.split(train, y), start=1):
        Xtr_base = train.iloc[tr_idx][features].reset_index(drop=True)
        ytr = y.iloc[tr_idx].reset_index(drop=True)
        Xva_base = train.iloc[va_idx][features].reset_index(drop=True)
        yva = y.iloc[va_idx].reset_index(drop=True)
        Xte_base = test[features].reset_index(drop=True)
        ntr = len(Xtr_base)

        inner = KFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE)
        te1_store = {f'TE1_{key_col}_{s}': np.full(ntr, np.nan, dtype='float32') for s in STATS_TE1}
        teq_store = {f'quantile_{q}': np.full(ntr, np.nan, dtype='float32') for q in QUANTILES}
        te2_store = {f'TE2_{col}_mean': np.full(ntr, np.nan, dtype='float32') for col in combo}
        for tr2_idx, va2_idx in inner.split(Xtr_base):
            Xtr2 = Xtr_base.iloc[tr2_idx]
            ytr2 = ytr.iloc[tr2_idx]
            g = pd.DataFrame({key_col: Xtr2[key_col], '_y_': ytr2.values})
            agg = g.groupby(key_col)['_y_'].agg(STATS_TE1)
            keys_va = Xtr_base.iloc[va2_idx][key_col]
            for s in STATS_TE1:
                te1_store[f'TE1_{key_col}_{s}'][va2_idx] = safe_map(keys_va, agg[s])
            for q in QUANTILES:
                qmap = g.groupby(key_col)['_y_'].quantile(q / 100.0)
                teq_store[f'quantile_{q}'][va2_idx] = safe_map(keys_va, qmap)
            for col in combo:
                m = pd.DataFrame({col: Xtr2[col], '_y_': ytr2.values}).groupby(col)['_y_'].mean()
                te2_store[f'TE2_{col}_mean'][va2_idx] = safe_map(Xtr_base.iloc[va2_idx][col], m)
            del g, agg
            gc.collect()
        Xtr = pd.concat([Xtr_base, pd.DataFrame(te1_store), pd.DataFrame(teq_store), pd.DataFrame(te2_store)], axis=1)

        gfull = pd.DataFrame({key_col: Xtr_base[key_col], '_y_': ytr.values})
        agg_full = gfull.groupby(key_col)['_y_'].agg(STATS_TE1)
        va_te, te_te = {}, {}
        for s in STATS_TE1:
            name = f'TE1_{key_col}_{s}'
            va_te[name] = safe_map(Xva_base[key_col], agg_full[s])
            te_te[name] = safe_map(Xte_base[key_col], agg_full[s])
        for q in QUANTILES:
            qmap = gfull.groupby(key_col)['_y_'].quantile(q / 100.0)
            va_te[f'quantile_{q}'] = safe_map(Xva_base[key_col], qmap)
            te_te[f'quantile_{q}'] = safe_map(Xte_base[key_col], qmap)
        for col in combo:
            m = pd.DataFrame({col: Xtr_base[col], '_y_': ytr.values}).groupby(col)['_y_'].mean()
            va_te[f'TE2_{col}_mean'] = safe_map(Xva_base[col], m)
            te_te[f'TE2_{col}_mean'] = safe_map(Xte_base[col], m)
        Xva = pd.concat([Xva_base, pd.DataFrame(va_te)], axis=1)
        Xte = pd.concat([Xte_base, pd.DataFrame(te_te)], axis=1)

        Xtr = add_ratio_block(Xtr, key_col)
        Xva = add_ratio_block(Xva, key_col)
        Xte = add_ratio_block(Xte, key_col)

        drop_cols = [c for c in pairs if c in Xtr.columns]
        Xtr = Xtr.drop(columns=drop_cols)
        Xva = Xva.drop(columns=drop_cols)
        Xte = Xte.drop(columns=drop_cols)

        cols = list(Xtr.columns)
        feature_types = ['c' if c in cat_for_xgb else 'q' for c in cols]
        model = XGBClassifier(**XGB_PARAMS, feature_types=feature_types)
        model.fit(Xtr[cols], ytr, eval_set=[(Xva[cols], yva)], verbose=500)
        oof_fold = model.predict_proba(Xva[cols])[:, 1].astype('float32')
        test_fold = model.predict_proba(Xte[cols])[:, 1].astype('float32')
        oof[va_idx] = oof_fold
        pred += test_fold
        print(f'{outer_folds} folds, fold {fold}: AUC {roc_auc_score(yva, oof_fold):.6f}, best_iteration '
              f'{getattr(model, "best_iteration", None)}', flush=True)

        del Xtr_base, Xva_base, Xte_base, Xtr, Xva, Xte, ytr, yva, model, gfull, agg_full
        gc.collect()

    pred /= outer_folds
    print(f'{outer_folds} folds: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    return oof, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    args = ap.parse_args()
    cfg = PRESETS[args.preset]

    data = build_features(cfg['anchor'], cfg['round_anchor'])
    oof, test_pred = run_cv(*data, OUTER_FOLDS, INNER_FOLDS)
    if 'test_folds' in cfg:
        _, test_pred = run_cv(*data, cfg['test_folds'], cfg['test_inner_folds'])
    save_pair(HERE, args.preset, oof, test_pred)


if __name__ == '__main__':
    main()
