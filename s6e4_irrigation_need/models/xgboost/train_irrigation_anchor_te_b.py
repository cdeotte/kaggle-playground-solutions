#!/usr/bin/env python
"""GPU XGBoost with target statistics around a rounded Previous_Irrigation_mm anchor (pair
xgb_prev_irrigation_anchor_te_b).

Base features (101, label-free, built on train + test; the original rows only provide the priors):
- the 8 categoricals (integer codes over train + test, XGBoost categoricals) and the 11 numerics (float32);
- the anchor key round(Previous_Irrigation_mm);
- the 6 decimal digits of the anchor value (XGBoost categoricals) and the 10 pairs of its first 5 digits;
- category x anchor keys code x 1e6 + Previous_Irrigation_mm (8) and the 28 pairs of the 8 categoricals (dropped after
  encoding);
- the numeric part of the rule formula, 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] +
  [Wind_Speed_kmh > 10], its flag score > 3, Soil_Moisture - 25, Rainfall_mm - 300, Soil_Moisture / (Rainfall_mm + 1);
- per-class original-data priors of the 8 categoricals.
Per fold, out-of-fold over an inner KFold(5, shuffle, seed 42) of the training rows (all training rows for the
validation / test rows): per anchor key the label (0, 1, 2) mean / std / count / nunique / median / min / max / skew and
its 5, 10, 40, 45, 55, 60, 90, 95 % quantiles; the 3 class shares of every category x anchor key, digit pair and
categorical pair; count / nunique and std / count ratios.
Model: XGBoost multi:softprob, depth 6, colsample_bynode 0.3, subsample 0.8, min_child_weight 10, lr 0.01, balanced
class weights, up to 50,000 rounds, early stopping 500 (mlogloss) on the validation fold, seed 42. Competition rows only.
Test = mean of the 5 fold models.
Usage: python train_irrigation_anchor_te_b.py [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_prev_irrigation_anchor_te_b'
SEED = 42
INNER_FOLDS = 5
N_CLASSES = 3

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture',
        'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']

ANCHOR_COL = 'Previous_Irrigation_mm'
ROUND_ANCHOR = 0
STATS_TE1 = ['mean', 'std', 'count', 'nunique', 'median', 'min', 'max', 'skew']
QUANTILES = [5, 10, 40, 45, 55, 60, 90, 95]
MAX_PAIR_CATS = 8

XGB_PARAMS = dict(
    device='cuda',
    tree_method='hist',
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    max_depth=6,
    colsample_bynode=0.3,
    subsample=0.8,
    min_child_weight=10,
    learning_rate=0.01,
    n_estimators=50000,
    early_stopping_rounds=500,
    random_state=SEED,
    enable_categorical=True,
    n_jobs=-1,
)


def safe_map(keys, mapping):
    return keys.map(mapping).astype('float32').to_numpy()


def build_features():
    train, test, orig = read_train(), read_test(), read_original()
    for df in (train, test, orig):
        for c in NUMS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        for c in CATS:
            if c in df.columns:
                df[c] = df[c].astype(str)
    train[TARGET] = labels(train[TARGET]).astype('int32')
    orig[TARGET] = labels(orig[TARGET]).astype('int32')
    y = train[TARGET].values
    print(f'train={len(train):,}  original={len(orig):,}', flush=True)

    # 1. categoricals -> integer codes (train + test); keep the string -> code map for the priors
    cat_to_int = {}
    for c in CATS:
        combined = pd.concat([train[c], test[c]], ignore_index=True)
        codes, uniques = combined.factorize(sort=False)
        train[c] = codes[:len(train)].astype('int32')
        test[c] = codes[len(train):].astype('int32')
        cat_to_int[c] = {v: i for i, v in enumerate(uniques)}

    # 2. anchor key
    anchor_tr = train[ANCHOR_COL].astype('float32')
    anchor_te = test[ANCHOR_COL].astype('float32')
    te1_key_col = f'{ANCHOR_COL}__te1key'
    train[te1_key_col] = anchor_tr.round(int(ROUND_ANCHOR)).astype('float32')
    test[te1_key_col] = anchor_te.round(int(ROUND_ANCHOR)).astype('float32')

    # 3. anchor digits
    anc_fill_tr = anchor_tr.fillna(0.0)
    anc_fill_te = anchor_te.fillna(0.0)
    digits = []
    new_tr, new_te = {}, {}
    for k in range(1, 7):
        dn = f'digit{k}'
        new_tr[dn] = ((anc_fill_tr * (10 ** k)) % 10).astype('int8')
        new_te[dn] = ((anc_fill_te * (10 ** k)) % 10).astype('int8')
        digits.append(dn)
    # 4. digit pairs
    for i in range(4):
        for j in range(i + 1, 5):
            n = f'digit_{i + 1}_{j + 1}'
            new_tr[n] = ((new_tr[f'digit{i + 1}'].astype('int16') + 1) * 11
                         + (new_tr[f'digit{j + 1}'].astype('int16') + 1)).astype('int16')
            new_te[n] = ((new_te[f'digit{i + 1}'].astype('int16') + 1) * 11
                         + (new_te[f'digit{j + 1}'].astype('int16') + 1)).astype('int16')
    # 5. category x anchor keys
    for c in CATS:
        n = f'{c}_anc'
        new_tr[n] = (train[c].astype('float32') * 1_000_000.0 + anchor_tr).astype('float32')
        new_te[n] = (test[c].astype('float32') * 1_000_000.0 + anchor_te).astype('float32')
    # 6. categorical pairs
    pairs = []
    cat_for_pairs = CATS[:MAX_PAIR_CATS]
    for i, c1 in enumerate(cat_for_pairs[:-1]):
        for c2 in cat_for_pairs[i + 1:]:
            n = f'{c1}__{c2}'
            m2 = int(train[c2].max() + 2)
            new_tr[n] = ((train[c1].astype('int64') + 1) * m2 + (train[c2].astype('int64') + 1)).astype('int32')
            new_te[n] = ((test[c1].astype('int64') + 1) * m2 + (test[c2].astype('int64') + 1)).astype('int32')
            pairs.append(n)
    train = pd.concat([train, pd.DataFrame(new_tr)], axis=1)
    test = pd.concat([test, pd.DataFrame(new_te)], axis=1)

    # 7. numeric part of the formula (the stage / mulching columns are integer codes here)
    for df in (train, test):
        SM = df['Soil_Moisture'].astype('float32')
        RM = df['Rainfall_mm'].astype('float32')
        TC = df['Temperature_C'].astype('float32')
        WS = df['Wind_Speed_kmh'].astype('float32')
        score = ((SM < 25).astype(int) * 2 + (RM < 300).astype(int) * 2 + (TC > 30).astype(int) + (WS > 10).astype(int))
        df['_formula_score'] = score.astype('float32')
        df['_is_high'] = (score > 3).astype('float32')
        df['_SM_thr'] = (SM - 25.0).astype('float32')
        df['_RM_thr'] = (RM - 300.0).astype('float32')
        df['_SM_div_RM_p1'] = (SM / (RM + 1.0)).astype('float32')
    formula_cols = ['_formula_score', '_is_high', '_SM_thr', '_RM_thr', '_SM_div_RM_p1']

    # 8. per-class original priors, mapped through the integer codes
    orig_prior_cols = []
    for c in CATS:
        for k in range(N_CLASSES):
            col_name = f'ORIG_k{k}_{c}'
            gm = (orig[TARGET] == k).mean()
            m = orig.groupby(c, observed=False).apply(lambda g: (g[TARGET] == k).mean())
            int_to_prior = {cat_to_int[c].get(str_val, -1): float_val for str_val, float_val in m.items()}
            train[col_name] = train[c].astype('int32').map(int_to_prior).fillna(gm).astype('float32')
            test[col_name] = test[c].astype('int32').map(int_to_prior).fillna(gm).astype('float32')
            orig_prior_cols.append(col_name)

    combo_cols = ([f'{c}_anc' for c in CATS]
                  + [n for n in train.columns if n.startswith('digit_') and '__' not in n]
                  + pairs)
    features = CATS + NUMS + formula_cols + orig_prior_cols + digits + combo_cols + [te1_key_col]
    seen = set()
    features = [f for f in features if f in train.columns and not (f in seen or seen.add(f))]
    print(f'Features before target encoding: {len(features)}', flush=True)
    cat_for_xgb = set(CATS + digits)
    return train.copy(), test.copy(), y, features, combo_cols, pairs, te1_key_col, cat_for_xgb


def add_ratio_block(df, te1_key_col):
    tc = f'TE1_{te1_key_col}_count'
    tn = f'TE1_{te1_key_col}_nunique'
    ts = f'TE1_{te1_key_col}_std'
    out = {}
    if tc in df.columns and tn in df.columns:
        out['TE1_count_per_nunique'] = (df[tc] / df[tn].replace(0, np.nan)).astype('float32').values
    if ts in df.columns and tc in df.columns:
        out['TE1_std_per_count'] = (df[ts] / df[tc].replace(0, np.nan)).astype('float32').values
    if out:
        df = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
    return df


def run_fold(tr_idx, va_idx, train, test, y, features, combo_cols, pairs, te1_key_col, cat_for_xgb, cls_weights):
    Xtr_base = train.iloc[tr_idx][features].reset_index(drop=True)
    ytr = y[tr_idx]
    Xva_base = train.iloc[va_idx][features].reset_index(drop=True)
    yva = y[va_idx]
    Xte_base = test[features].reset_index(drop=True)
    ntr = len(Xtr_base)

    te1_names = [f'TE1_{te1_key_col}_{s}' for s in STATS_TE1]
    teq_names = [f'quantile_{q}' for q in QUANTILES]
    te2_names = [f'TE2_{col}_k{k}' for col in combo_cols for k in range(N_CLASSES)]
    te1_store = {n: np.full(ntr, np.nan, 'float32') for n in te1_names}
    teq_store = {n: np.full(ntr, np.nan, 'float32') for n in teq_names}
    te2_store = {n: np.full(ntr, np.nan, 'float32') for n in te2_names}

    # training rows: statistics of the other inner folds
    inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    for tr2_idx, va2_idx in inner.split(Xtr_base):
        Xtr2 = Xtr_base.iloc[tr2_idx]
        ytr2 = ytr[tr2_idx]
        g = pd.DataFrame({te1_key_col: Xtr2[te1_key_col], '_y_': ytr2.astype(float)})
        agg = g.groupby(te1_key_col)['_y_'].agg(STATS_TE1)
        keys_va2 = Xtr_base.iloc[va2_idx][te1_key_col]
        for s in STATS_TE1:
            te1_store[f'TE1_{te1_key_col}_{s}'][va2_idx] = safe_map(keys_va2, agg[s])
        for q in QUANTILES:
            qmap = g.groupby(te1_key_col)['_y_'].quantile(q / 100.0)
            teq_store[f'quantile_{q}'][va2_idx] = safe_map(keys_va2, qmap)
        for k in range(N_CLASSES):
            y_bin = (ytr2 == k).astype(float)
            for col in combo_cols:
                m = pd.Series(y_bin, name='_y_').groupby(Xtr2[col].values).mean()
                te2_store[f'TE2_{col}_k{k}'][va2_idx] = safe_map(Xtr_base.iloc[va2_idx][col], m)
        del g, agg
        gc.collect()
    Xtr = pd.concat([Xtr_base, pd.DataFrame(te1_store), pd.DataFrame(teq_store), pd.DataFrame(te2_store)], axis=1)

    # validation / test rows: statistics of all training rows
    gfull = pd.DataFrame({te1_key_col: Xtr_base[te1_key_col], '_y_': ytr.astype(float)})
    agg_full = gfull.groupby(te1_key_col)['_y_'].agg(STATS_TE1)
    va_te, te_te = {}, {}
    for s in STATS_TE1:
        name = f'TE1_{te1_key_col}_{s}'
        va_te[name] = safe_map(Xva_base[te1_key_col], agg_full[s])
        te_te[name] = safe_map(Xte_base[te1_key_col], agg_full[s])
    for q in QUANTILES:
        qmap = gfull.groupby(te1_key_col)['_y_'].quantile(q / 100.0)
        name = f'quantile_{q}'
        va_te[name] = safe_map(Xva_base[te1_key_col], qmap)
        te_te[name] = safe_map(Xte_base[te1_key_col], qmap)
    for k in range(N_CLASSES):
        y_bin_full = (ytr == k).astype(float)
        for col in combo_cols:
            m = pd.Series(y_bin_full, name='_y_').groupby(Xtr_base[col].values).mean()
            name = f'TE2_{col}_k{k}'
            va_te[name] = safe_map(Xva_base[col], m)
            te_te[name] = safe_map(Xte_base[col], m)
    Xva = pd.concat([Xva_base, pd.DataFrame(va_te, index=Xva_base.index)], axis=1)
    Xte = pd.concat([Xte_base, pd.DataFrame(te_te, index=Xte_base.index)], axis=1)

    Xtr = add_ratio_block(Xtr, te1_key_col)
    Xva = add_ratio_block(Xva, te1_key_col)
    Xte = add_ratio_block(Xte, te1_key_col)

    # the raw categorical pair ids are dropped (their encodings stay)
    if pairs:
        drop_p = [c for c in pairs if c in Xtr.columns]
        Xtr.drop(columns=drop_p, inplace=True, errors='ignore')
        Xva.drop(columns=drop_p, inplace=True, errors='ignore')
        Xte.drop(columns=drop_p, inplace=True, errors='ignore')

    cols = list(Xtr.columns)
    feature_types = ['c' if c in cat_for_xgb else 'q' for c in cols]
    model = XGBClassifier(**XGB_PARAMS, feature_types=feature_types)
    model.fit(Xtr[cols], ytr, sample_weight=cls_weights[ytr], eval_set=[(Xva[cols], yva)], verbose=500)
    p_va = model.predict_proba(Xva[cols]).astype(np.float32)
    p_te = model.predict_proba(Xte[cols]).astype(np.float32)
    best_it = getattr(model, 'best_iteration', None)
    del Xtr_base, Xva_base, Xte_base, Xtr, Xva, Xte, model, gfull, agg_full
    gc.collect()
    return p_va, p_te, yva, best_it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train, test, y, features, combo_cols, pairs, te1_key_col, cat_for_xgb = build_features()
        cls_counts = np.bincount(y, minlength=N_CLASSES)
        cls_weights = cls_counts.sum() / (N_CLASSES * cls_counts)
        fold_id = folds(y)
        for k in todo:
            print(f'fold {k}', flush=True)
            p_va, p_te, yva, best_it = run_fold(np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k), train, test,
                                                y, features, combo_cols, pairs, te1_key_col, cat_for_xgb, cls_weights)
            save_part(HERE, PAIR, k, p_va, p_te)
            print(f'{PAIR} fold {k}: balanced accuracy {balanced_accuracy_score(yva, p_va.argmax(axis=1)):.6f}, '
                  f'best iteration {best_it}', flush=True)
    y = labels(read_train()[TARGET])
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
