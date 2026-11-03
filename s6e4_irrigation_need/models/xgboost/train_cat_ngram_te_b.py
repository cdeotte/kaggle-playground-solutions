#!/usr/bin/env python
"""GPU XGBoost with tuned slow-learning parameters on categorical n-gram target encodings (pair
xgb_cat_ngram_nested_te_b).

Label-free features (84, built on train, test and the original rows):
- the 19 raw columns (categoricals integer-coded per fold);
- the rule formula score = 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
  - 2 [Crop_Growth_Stage = Harvest] - 2 [Crop_Growth_Stage = Sowing] - [Mulching_Used = Yes], is_low, is_high;
- the frequency of every value of the 19 columns over train + test + original;
- per-class original-data priors of the 8 categoricals;
- 15 bigrams of 6 categoricals (Crop_Growth_Stage, Mulching_Used, Season, Crop_Type, Soil_Type, Region) and 4 trigrams of
  the first 4 of them (string keys, dropped after encoding).
Per fold: the bigrams, trigrams and categoricals (27 keys) get 3 binary target encodings each (one per class, sklearn
TargetEncoder, smooth 'auto'), out-of-fold over an inner StratifiedKFold(5, seed 42 + fold, fold = 1..5) of the training
rows and fitted on all training rows for the validation / test rows.
Model: XGBoost multi:softprob, depth 5, lr 0.0063, subsample 0.81, colsample_bytree 0.32, min_child_weight 6, alpha 3.5017,
lambda 1.2925, gamma 0.79, balanced class weights, up to 50,000 rounds, early stopping 500 (mlogloss) on the validation
fold, seed 42. Competition rows only. Test = mean of the 5 fold models. A fold takes about 8-9 minutes on one GPU.
Usage: python train_cat_ngram_te_b.py [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_cat_ngram_nested_te_b'
SEED = 42
INNER_FOLDS = 5
LABEL = 'target'

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture',
        'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']
TOP_CATS = ['Crop_Growth_Stage', 'Mulching_Used', 'Season', 'Crop_Type', 'Soil_Type', 'Region']

XGB_PARAMS = dict(
    n_estimators=50000,
    learning_rate=0.0063,
    max_depth=5,
    subsample=0.81,
    colsample_bytree=0.32,
    min_child_weight=6,
    reg_alpha=3.5017,
    reg_lambda=1.2925,
    gamma=0.790,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    enable_categorical=False,
    device='cuda',
    tree_method='hist',
    random_state=SEED,
    early_stopping_rounds=500,
    verbosity=0,
)


def build_features():
    train, test, orig = read_train(), read_test(), read_original()
    train[LABEL] = labels(train[TARGET]).astype(np.int32)
    orig[LABEL] = labels(orig[TARGET]).astype(np.int32)
    y = train[LABEL].values
    y_orig = orig[LABEL].values
    for c in [ID, TARGET]:
        for df in [train, test]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    # formula
    for df in [train, test, orig]:
        SM = df['Soil_Moisture']; RM = df['Rainfall_mm']; TC = df['Temperature_C']
        WS = df['Wind_Speed_kmh']; CGS = df['Crop_Growth_Stage']; MU = df['Mulching_Used']
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
                 - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes'))
        df['formula_score'] = score.astype(np.float32)
        df['is_low'] = (score <= 0).astype(np.float32)
        df['is_high'] = (score > 3).astype(np.float32)

    # frequency over train + test + original
    for c in NUMS + CATS:
        freq = pd.concat([train[c], test[c], orig[c]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{c}'] = df[c].map(freq).fillna(0).astype(np.float32)

    # per-class original priors
    for c in CATS:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            gm = float((y_orig == k).mean())
            for df in [train, test, orig]:
                df[f'ORIG_k{k}_{c}'] = df[c].map(m).fillna(gm).astype(np.float32)

    # bigrams / trigrams
    bigram_cols = []
    for c1, c2 in combinations(TOP_CATS, 2):
        name = f'BG_{c1}_{c2}'
        for df in [train, test, orig]:
            df[name] = df[c1].astype(str) + '_' + df[c2].astype(str)
        bigram_cols.append(name)
    trigram_cols = []
    for c1, c2, c3 in combinations(TOP_CATS[:4], 3):
        name = f'TG_{c1}_{c2}_{c3}'
        for df in [train, test, orig]:
            df[name] = df[c1].astype(str) + '_' + df[c2].astype(str) + '_' + df[c3].astype(str)
        trigram_cols.append(name)

    for c in CATS:
        for df in [train, test]:
            df[c] = df[c].astype('category')

    features = [c for c in train.columns if c != LABEL]
    te_cols = [c for c in bigram_cols + trigram_cols + CATS if c in train.columns]
    print(f'Features: {len(features)}, target-encoded keys: {len(te_cols)}', flush=True)
    return train[features].copy(), test[features].copy(), y, te_cols


def _te_to_1d(x):
    if isinstance(x, np.ndarray):
        return x[:, 0].astype(np.float32) if x.ndim == 2 else x.astype(np.float32)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.iloc[:, 0].astype(np.float32).values if x.ndim == 2 else x.astype(np.float32).values
    return np.asarray(x).astype(np.float32)


def nested_multiclass_te(X_tr, y_tr, X_va, X_te, te_cols, inner_folds=5, seed=42):
    """3 binary target encodings per column (one per class), nested inside the outer fold."""
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    made = []
    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    for c in te_cols:
        if c not in X_tr.columns:
            continue
        tr_col = X_tr[c].astype('string').fillna('<NA>').to_frame(name=c)
        va_col = X_va[c].astype('string').fillna('<NA>').to_frame(name=c)
        te_col = X_te[c].astype('string').fillna('<NA>').to_frame(name=c)
        for k in range(3):
            y_k = (y_tr == k).astype(np.int32)
            oof_enc = np.zeros(len(X_tr), dtype=np.float32)
            for itr, iva in skf_inner.split(tr_col, y_k):
                enc = TargetEncoder(target_type='binary', smooth='auto')
                enc.fit(tr_col.iloc[itr], y_k.iloc[itr])
                oof_enc[iva] = _te_to_1d(enc.transform(tr_col.iloc[iva]))
            enc_full = TargetEncoder(target_type='binary', smooth='auto')
            enc_full.fit(tr_col, y_k)
            newc = f'TE_k{k}_{c}'
            X_tr[newc] = oof_enc
            X_va[newc] = _te_to_1d(enc_full.transform(va_col))
            X_te[newc] = _te_to_1d(enc_full.transform(te_col))
            made.append(newc)
    return X_tr, X_va, X_te, made


def factorize_tvt(tr, va, te, cols):
    tr, va, te = tr.copy(), va.copy(), te.copy()
    for c in cols:
        if c not in tr.columns:
            continue
        allv = pd.concat([tr[c].astype(str), va[c].astype(str), te[c].astype(str)], ignore_index=True)
        codes, _ = pd.factorize(allv, sort=False)
        n1, n2 = len(tr), len(va)
        tr[c] = codes[:n1].astype('int32')
        va[c] = codes[n1:n1 + n2].astype('int32')
        te[c] = codes[n1 + n2:].astype('int32')
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X, X_test_all, y, te_cols = build_features()
        cls_counts = np.bincount(y)
        cls_weights = cls_counts.sum() / (3 * cls_counts)
        fold_id = folds(y)
        for k in todo:
            fold = k + 1
            t0 = time.time()
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr = X.iloc[tr_idx].reset_index(drop=True)
            X_va = X.iloc[va_idx].reset_index(drop=True)
            y_tr = pd.Series(y[tr_idx])
            y_va = y[va_idx]
            X_te = X_test_all.copy()

            X_tr, X_va, X_te, te_made = nested_multiclass_te(X_tr, y_tr, X_va, X_te, te_cols, INNER_FOLDS, SEED + fold)
            # the string n-grams are replaced by their encodings; the categoricals are integer-coded
            str_cols = [c for c in X_tr.columns if X_tr[c].dtype.name in ['object', 'string']]
            for df in [X_tr, X_va, X_te]:
                df.drop(columns=str_cols, inplace=True, errors='ignore')
            cat_remaining = [c for c in X_tr.columns if X_tr[c].dtype.name == 'category']
            X_tr, X_va, X_te = factorize_tvt(X_tr, X_va, X_te, cat_remaining)
            features = X_tr.columns.tolist()
            for df in [X_tr, X_va, X_te]:
                for c in features:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(np.float32)

            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr[features], y_tr, sample_weight=cls_weights[y_tr.values],
                      eval_set=[(X_va[features], y_va)], verbose=1000)
            p_va = model.predict_proba(X_va[features]).astype(np.float32)
            p_te = model.predict_proba(X_te[features]).astype(np.float32)
            save_part(HERE, PAIR, k, p_va, p_te)
            bac = balanced_accuracy_score(y_va, np.argmax(p_va, axis=1))
            print(f'{PAIR} fold {k}: balanced accuracy {bac:.5f}, best iteration {model.best_iteration} '
                  f'({time.time() - t0:.0f}s)', flush=True)
            del X_tr, X_va, X_te, model
            gc.collect()
    y = labels(read_train()[TARGET])
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
