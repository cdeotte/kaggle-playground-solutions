#!/usr/bin/env python
"""Public notebook "0.979 CV Single XGB: Pairwise TE + Bias Tuning" run on our 5 folds (the split the notebook uses),
without its bias tuning.
https://www.kaggle.com/code/utaazu/0-979-cv-single-xgb-pairwise-te-bias-tuning

Features: the 19 raw columns (the 8 categoricals as pandas categories, XGBoost native categorical splits); ET_proxy,
water_deficit and soil_quality; TE_ORIG_<col> for the 19 columns from the original data (categoricals: target mean per
category; numerics: target mean in 10 equal-width bins of the original column); per fold, a multiclass TargetEncoder
(cv=3, seed 42) of every pairwise combination of the 19 columns that has at most half as many distinct values as there
are train + test + original rows (3 columns per pair), fitted on the fold's training rows + original rows.
Original rows: all 10,000 are appended to each fold's training rows; balanced sample weights, original rows x 0.35.
Model: XGBClassifier (GPU, hist) multi:softprob, max_depth 8, lr 0.03, subsample 0.8, colsample_bytree 0.6, up to 3,000
trees, early stopping 100 on the validation fold's mlogloss (predictions use the best iteration), seed 42.
Test = mean of the 5 fold models. Usage: python train.py [--folds 0,1,2,3,4]"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, TARGET_MAP, read_original, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'utaazu_xgb'
SEED = 42
ORIG_ROW_WEIGHT = 0.35

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
ALL_BASE = NUMS + CATS

XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'tree_method': 'hist',
    'device': 'cuda',
    'enable_categorical': True,
    'n_estimators': 3000,
    'random_state': SEED,
    'verbosity': 0,
    'early_stopping_rounds': 100,
}


def add_physical_features(df):
    d = df.copy()
    d['ET_proxy'] = (d['Temperature_C'] * d['Wind_Speed_kmh'] * d['Sunlight_Hours']) / (d['Humidity'] + 1)
    d['water_deficit'] = d['Soil_Moisture'] - (d['Rainfall_mm'] + d['Previous_Irrigation_mm']) * 0.1
    d['soil_quality'] = d['Organic_Carbon'] / (d['Electrical_Conductivity'] + 0.1)
    return d


def build_features():
    train = add_physical_features(read_train().dropna(subset=[TARGET]))
    test = add_physical_features(read_test())
    orig = add_physical_features(read_original())

    # target-mean priors from the original data
    orig_target_numeric = orig[TARGET].map(TARGET_MAP)
    for col in ALL_BASE:
        name = f'TE_ORIG_{col}'
        if orig[col].dtype == 'object':
            mapping = orig_target_numeric.groupby(orig[col]).mean().to_dict()
            fallback = orig_target_numeric.mean()
            for df in (train, test, orig):
                df[name] = df[col].map(mapping).fillna(fallback).astype(np.float32)
        else:
            edges = np.histogram_bin_edges(orig[col].dropna(), bins=10)
            means = orig_target_numeric.groupby(pd.cut(orig[col], bins=edges, include_lowest=True)).mean().values

            def map_bins(val, e=edges, m=means):
                idx = np.digitize(val, e) - 1
                return m[min(max(idx, 0), 9)]

            for df in (train, test, orig):
                df[name] = df[col].apply(map_bins).astype(np.float32)

    # C(19, 2) pairwise combinations, factorized over train + test + original rows
    pair_cols = []
    for left, right in combinations(ALL_BASE, 2):
        name = f'{left}___{right}'
        tr_token = train[left].astype(str) + '_' + train[right].astype(str)
        te_token = test[left].astype(str) + '_' + test[right].astype(str)
        og_token = orig[left].astype(str) + '_' + orig[right].astype(str)
        combined = pd.concat([tr_token, te_token, og_token], ignore_index=True)
        codes, _ = pd.factorize(combined)
        if pd.Series(codes).nunique() <= (len(combined) // 2):
            train[name] = codes[:len(train)]
            test[name] = codes[len(train):len(train) + len(test)]
            orig[name] = codes[len(train) + len(test):]
            pair_cols.append(name)

    for c in CATS:
        for df in (train, test, orig):
            df[c] = df[c].astype('category')

    y_train_full = train[TARGET].map(TARGET_MAP).values.astype(int)
    y_orig_full = orig[TARGET].map(TARGET_MAP).values.astype(int)
    features = [c for c in train.columns if c not in [TARGET, ID]]
    pair_features = [c for c in features if '___' in c]
    print(f'Pairwise combinations: {len(pair_cols)} | Total features: {len(features)}', flush=True)
    return train, test, orig, y_train_full, y_orig_full, features, pair_features


def main():
    args = fold_args()
    train, test, orig, y_train_full, y_orig_full, features, pair_features = build_features()

    def fit_fold(fold, tr_idx, val_idx):
        x_tr_fold, y_tr_fold = train.iloc[tr_idx][features], y_train_full[tr_idx]
        x_va_fold, y_va_fold = train.iloc[val_idx][features], y_train_full[val_idx]

        # all original rows are appended to the fold's training rows
        x_tr_merged = pd.concat([x_tr_fold, orig[features]], axis=0).reset_index(drop=True)
        y_tr_merged = np.concatenate([y_tr_fold, y_orig_full])

        encoder = TargetEncoder(target_type='multiclass', cv=3, random_state=SEED)
        enc_tr = encoder.fit_transform(x_tr_merged[pair_features], y_tr_merged).astype(np.float32)
        enc_va = encoder.transform(x_va_fold[pair_features]).astype(np.float32)
        enc_te = encoder.transform(test[features][pair_features]).astype(np.float32)

        # the raw pair codes are replaced by their target encodings
        x_tr_final = x_tr_merged.drop(columns=pair_features).copy()
        x_va_final = x_va_fold.drop(columns=pair_features).copy()
        x_te_final = test[features].drop(columns=pair_features).copy()
        te_names = [f'TE_pair_{i}' for i in range(enc_tr.shape[1])]
        x_tr_final[te_names] = enc_tr
        x_va_final[te_names] = enc_va
        x_te_final[te_names] = enc_te

        # balanced sample weights for all rows, then the original rows x 0.35
        sw = compute_sample_weight('balanced', y_tr_merged).astype(np.float32)
        sw[len(tr_idx):] *= ORIG_ROW_WEIGHT

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(x_tr_final, y_tr_merged, sample_weight=sw, eval_set=[(x_va_final, y_va_fold)], verbose=False)
        valid_pred = model.predict_proba(x_va_final).astype(np.float32)
        test_pred = model.predict_proba(x_te_final).astype(np.float32)
        print(f'fold {fold}: best_iter={model.best_iteration}', flush=True)
        del x_tr_final, x_va_final, x_te_final, model
        gc.collect()
        return valid_pred, test_pred

    run_folds(FOLDER, fit_fold, args.folds, y=y_train_full)


if __name__ == '__main__':
    main()
