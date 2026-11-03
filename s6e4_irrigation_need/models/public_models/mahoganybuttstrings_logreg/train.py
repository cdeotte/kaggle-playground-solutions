#!/usr/bin/env python
"""Public notebook "PG S6E4 - LR - [CV 0.97652 LB 0.97487]" run on our 5 folds (the split the notebook uses).
https://www.kaggle.com/code/mahoganybuttstrings/pg-s6e4-lr-cv-0-97652-lb-0-97487

Features: 4 binary formula indicators (Soil_Moisture < 25, Temperature_C > 30, Rainfall_mm < 300, Wind_Speed_kmh > 10),
one-hot Crop_Growth_Stage and Mulching_Used, and every pairwise combination of the 19 raw columns that has at most half
as many distinct values as there are train + test + original rows, factorized over those rows (all other raw columns
are dropped). Per fold: a multiclass TargetEncoder (cv=5, seed 42) fitted on the fold's training rows replaces each
pair by its 3 class encodings; StandardScaler fitted on the training rows.
Model: scikit-learn LogisticRegression (multinomial, lbfgs, max_iter 10,000, balanced class weights, seed 42).
Test = mean of the 5 fold models. Needs a scikit-learn that still accepts multi_class (removed in 1.8).
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, TARGET_MAP, read_original, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'mahoganybuttstrings_logreg'
NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
FORMULA = ['soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10']


def build_features():
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    orig = read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP)

    for df in [train, test]:
        df['soil_lt_25'] = (df['Soil_Moisture'] < 25).astype(int)
        df['temp_gt_30'] = (df['Temperature_C'] > 30).astype(int)
        df['rain_lt_300'] = (df['Rainfall_mm'] < 300).astype(int)
        df['wind_gt_10'] = (df['Wind_Speed_kmh'] > 10).astype(int)

    # all C(19, 2) pairs, factorized over train + test + original rows, kept if at most half of the rows are distinct
    te_columns = []
    total_rows = len(train) + len(test) + len(orig)
    for cols in combinations(NUMS + CATS, 2):
        name = '-'.join(cols)
        tr_pair = train[cols[0]].astype(str) + '_' + train[cols[1]].astype(str)
        te_pair = test[cols[0]].astype(str) + '_' + test[cols[1]].astype(str)
        og_pair = orig[cols[0]].astype(str) + '_' + orig[cols[1]].astype(str)
        combined, _ = pd.concat([tr_pair, te_pair, og_pair], ignore_index=True).factorize()
        if pd.Series(combined).nunique() > total_rows // 2:
            continue
        train[name] = combined[:len(train)]
        test[name] = combined[len(train):len(train) + len(test)]
        te_columns.append(name)
    print(f'Pairwise TE columns kept: {len(te_columns)}', flush=True)

    # keep only the formula indicators, 2 categoricals (one-hot over train + test) and the pairs
    train = train[FORMULA + ['Crop_Growth_Stage', 'Mulching_Used', TARGET] + te_columns]
    test = test[FORMULA + ['Crop_Growth_Stage', 'Mulching_Used'] + te_columns]
    combined = pd.get_dummies(pd.concat([train, test]), columns=['Crop_Growth_Stage', 'Mulching_Used'])
    train = combined[:len(train)]
    test = combined[len(train):].drop(TARGET, axis=1)
    features = [c for c in train.columns if c != TARGET]
    print(f'Total features after trim+OHE (before TE replacement): {len(features)}', flush=True)
    return train, test, features, te_columns


def main():
    args = fold_args()
    train, test, features, te_columns = build_features()
    y = train[TARGET]

    def fit_fold(fold, train_idx, val_idx):
        x_train, x_val, x_test = train.iloc[train_idx][features], train.iloc[val_idx][features], test.copy()
        y_train = y.iloc[train_idx]

        encoder = TargetEncoder(target_type='multiclass', cv=5, random_state=42)
        enc_tr = pd.DataFrame(encoder.fit_transform(x_train[te_columns], y_train)).reset_index(drop=True)
        enc_va = pd.DataFrame(encoder.transform(x_val[te_columns])).reset_index(drop=True)
        enc_te = pd.DataFrame(encoder.transform(x_test[te_columns])).reset_index(drop=True)
        x_train = pd.concat([enc_tr, x_train.reset_index(drop=True)], axis=1).drop(te_columns, axis=1)
        x_val = pd.concat([enc_va, x_val.reset_index(drop=True)], axis=1).drop(te_columns, axis=1)
        x_test = pd.concat([enc_te, x_test.reset_index(drop=True)], axis=1).drop(te_columns, axis=1)
        x_train.columns = x_train.columns.astype(str)
        x_val.columns = x_val.columns.astype(str)
        x_test.columns = x_test.columns.astype(str)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000, random_state=42,
                                   class_weight='balanced', n_jobs=-1)
        model.fit(x_train, y_train)
        return model.predict_proba(x_val), model.predict_proba(x_test)

    run_folds(FOLDER, fit_fold, args.folds, y=y.to_numpy().astype(np.int64))


if __name__ == '__main__':
    main()
