#!/usr/bin/env python
"""Public notebook "s6e4-catboost-robust-0-978-cv" run on our 5 folds, without its threshold tuning.
https://www.kaggle.com/code/blamerx/s6e4-catboost-robust-0-978-cv

Features (80 + 405): the 19 raw columns (categoricals label-encoded over train + test + original rows, CatBoost
categorical features); formula indicators (Soil_Moisture < 25, Rainfall_mm < 300, Temperature_C > 30,
Wind_Speed_kmh > 10, Harvest, Sowing, Mulching = Yes), the formula score 2 soil + 2 rain + temp + wind - 2 harvest -
2 sowing - mulch, its distances to the class boundaries 0 and 3, and the signed distances to the 4 thresholds;
logit_Low / logit_Medium / logit_High, a fixed logistic regression of those indicators fitted on the original data; the
first decimal digit of the 11 numerics; 32 agronomy features (water input, ET / evaporation / VPD / heat /
dryness / aridity / drought proxies, soil health, salinity, pH deviation, per-hectare ratios, mulch effects, logs); per
fold, a multiclass TargetEncoder (cv=5, seed 42) of every pairwise combination of the 19 columns that has at most half
as many distinct values as there are train + test + original rows (3 columns per pair), fitted on the fold's training
rows + original rows.
Original rows: all 10,000 are appended to each fold's training rows.
Model: CatBoost (GPU) MultiClass with the notebook's tuned parameters (depth 6, lr 0.0447, l2_leaf_reg 4.41,
random_strength 2.21, bagging_temperature 0.318), up to 5,000 iterations, balanced class weights, early stopping 200 on
TotalF1:average=Macro of the validation fold with the best model kept, seed 42. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder, TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, TARGET_MAP, read_original, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds, work_dir, working_directory  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'blamerx_catboost'
SEED = 42

CAT_COLS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
            'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUM_COLS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
            'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
            'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
ALL_COLS = NUM_COLS + CAT_COLS

# physical thresholds
SOIL_THRESH = 25
RAIN_THRESH = 300
TEMP_THRESH = 30
WIND_THRESH = 10

# logistic-regression coefficients fitted on the original data (100% accurate there)
LOGIT_COEFS = {
    'Low':    {'intercept': 16.3173, 'soil_lt_25': -11.0237, 'temp_gt_30': -5.8559,
               'rain_lt_300': -10.8500, 'wind_gt_10': -5.8284,
               'Flowering': -5.4155, 'Harvest': 5.5073, 'Sowing': 5.2299, 'Vegetative': -5.4617,
               'Mulch_No': -3.0014, 'Mulch_Yes': 2.8613},
    'Medium': {'intercept': 4.6524, 'soil_lt_25': 0.3290, 'temp_gt_30': -0.0204,
               'rain_lt_300': 0.1542, 'wind_gt_10': 0.0841,
               'Flowering': 0.3586, 'Harvest': -0.1348, 'Sowing': -0.3547, 'Vegetative': 0.3334,
               'Mulch_No': 0.1883, 'Mulch_Yes': 0.0142},
    'High':   {'intercept': -20.9697, 'soil_lt_25': 10.6947, 'temp_gt_30': 5.8763,
               'rain_lt_300': 10.6958, 'wind_gt_10': 5.7444,
               'Flowering': 5.0569, 'Harvest': -5.3725, 'Sowing': -4.8752, 'Vegetative': 5.1283,
               'Mulch_No': 2.8131, 'Mulch_Yes': -2.8755},
}

CAT_PARAMS = dict(
    depth=6,
    learning_rate=0.04465386288353782,
    l2_leaf_reg=4.4142886840444255,
    random_strength=2.205133902609163,
    bagging_temperature=0.3178262056913819,
    iterations=5000,
    task_type='GPU',
    devices='0',
    loss_function='MultiClass',
    eval_metric='TotalF1:average=Macro',
    random_seed=SEED,
    verbose=500,
    early_stopping_rounds=200,
    auto_class_weights='Balanced',
)


def compute_logit_scores(df):
    """Logit scores of the 3 classes from the fixed coefficients."""
    soil = (df['Soil_Moisture'] < SOIL_THRESH).astype(float)
    temp = (df['Temperature_C'] > TEMP_THRESH).astype(float)
    rain = (df['Rainfall_mm'] < RAIN_THRESH).astype(float)
    wind = (df['Wind_Speed_kmh'] > WIND_THRESH).astype(float)

    stage = df['Crop_Growth_Stage'].astype(str)
    is_flowering = (stage == 'Flowering').astype(float)
    is_harvest = (stage == 'Harvest').astype(float)
    is_sowing = (stage == 'Sowing').astype(float)
    is_vegetative = (stage == 'Vegetative').astype(float)

    mulch = df['Mulching_Used'].astype(str)
    mulch_no = (mulch == 'No').astype(float)
    mulch_yes = (mulch == 'Yes').astype(float)

    d = df.copy()
    for cls_name, coefs in LOGIT_COEFS.items():
        d[f'logit_{cls_name}'] = (
            coefs['intercept']
            + coefs['soil_lt_25'] * soil + coefs['temp_gt_30'] * temp
            + coefs['rain_lt_300'] * rain + coefs['wind_gt_10'] * wind
            + coefs['Flowering'] * is_flowering + coefs['Harvest'] * is_harvest
            + coefs['Sowing'] * is_sowing + coefs['Vegetative'] * is_vegetative
            + coefs['Mulch_No'] * mulch_no + coefs['Mulch_Yes'] * mulch_yes
        )
    return d


def engineer_features(df):
    df = df.copy()

    # binary thresholds
    df['soil_lt_25'] = (df['Soil_Moisture'] < SOIL_THRESH).astype(int)
    df['rain_lt_300'] = (df['Rainfall_mm'] < RAIN_THRESH).astype(int)
    df['temp_gt_30'] = (df['Temperature_C'] > TEMP_THRESH).astype(int)
    df['wind_gt_10'] = (df['Wind_Speed_kmh'] > WIND_THRESH).astype(int)

    df['is_harvest'] = (df['Crop_Growth_Stage'] == 'Harvest').astype(int)
    df['is_sowing'] = (df['Crop_Growth_Stage'] == 'Sowing').astype(int)
    df['mulching_yes'] = (df['Mulching_Used'] == 'Yes').astype(int)

    # formula score
    high_score = 2 * df['soil_lt_25'] + 2 * df['rain_lt_300'] + df['temp_gt_30'] + df['wind_gt_10']
    low_score = 2 * df['is_harvest'] + 2 * df['is_sowing'] + df['mulching_yes']
    df['magic_score'] = high_score - low_score

    # first decimal digit of the 11 numerics
    for col in NUM_COLS:
        vals = df[col].values
        df[f'{col}_dec'] = np.floor((vals - np.floor(vals)) * 10).astype(int)

    # distances to the score boundaries and signed distances to the thresholds
    df['dist_boundary_0'] = np.abs(df['magic_score'] - 0)
    df['dist_boundary_3'] = np.abs(df['magic_score'] - 3)
    df['dist_boundary_min'] = np.minimum(df['dist_boundary_0'], df['dist_boundary_3'])
    df['soil_dist_25'] = df['Soil_Moisture'] - SOIL_THRESH
    df['rain_dist_300'] = df['Rainfall_mm'] - RAIN_THRESH
    df['temp_dist_30'] = df['Temperature_C'] - TEMP_THRESH
    df['wind_dist_10'] = df['Wind_Speed_kmh'] - WIND_THRESH

    df = compute_logit_scores(df)

    # agronomy features
    df['Total_Water_Input'] = df['Rainfall_mm'] + df['Previous_Irrigation_mm']
    df['Moisture_Deficit'] = 100 - df['Soil_Moisture']
    df['Irrigation_Ratio'] = df['Previous_Irrigation_mm'] / (df['Rainfall_mm'] + 1)
    df['ET_Proxy'] = (df['Temperature_C'] * df['Sunlight_Hours']) / (df['Humidity'] + 1)
    df['Evap_Stress'] = (df['Temperature_C'] * df['Wind_Speed_kmh']) / (df['Humidity'] + 1)
    df['Wind_ET'] = df['Wind_Speed_kmh'] * (1 - df['Humidity'] / 100)
    df['Net_Water_Need'] = df['ET_Proxy'] - df['Rainfall_mm'] / 10
    df['VPD_Proxy'] = df['Temperature_C'] * (1 - df['Humidity'] / 100)
    df['Heat_Stress'] = df['Temperature_C'] * (100 - df['Humidity']) / 100
    df['Dryness_Index'] = df['Temperature_C'] * df['Sunlight_Hours'] / (df['Rainfall_mm'] + 1)
    df['Aridity_Index'] = df['Rainfall_mm'] / (df['ET_Proxy'] + 0.1)
    df['Drought_Risk'] = df['Dryness_Index'] * df['Moisture_Deficit'] / 100
    df['Soil_Health'] = df['Organic_Carbon'] * df['Soil_Moisture'] / (df['Electrical_Conductivity'] + 0.1)
    df['Salinity_Risk'] = df['Electrical_Conductivity'] * df['Temperature_C'] / (df['Rainfall_mm'] + 1)
    df['pH_Deviation'] = np.abs(df['Soil_pH'] - 6.5)
    df['Moisture_Retention'] = df['Soil_Moisture'] * df['Organic_Carbon']
    df['EC_pH_Interaction'] = df['Electrical_Conductivity'] * df['Soil_pH']
    df['Irrig_Per_Ha'] = df['Previous_Irrigation_mm'] / (df['Field_Area_hectare'] + 0.1)
    df['Rain_Per_Ha'] = df['Rainfall_mm'] / (df['Field_Area_hectare'] + 0.1)
    df['Area_Log'] = np.log1p(df['Field_Area_hectare'])
    df['Water_Per_Ha'] = df['Total_Water_Input'] / (df['Field_Area_hectare'] + 0.1)

    mulch_map = {'Yes': 1.0, 'No': 0.0}
    df['Mulch_Flag'] = df['Mulching_Used'].map(mulch_map).fillna(0.0)
    df['Mulch_Moisture'] = df['Soil_Moisture'] * (1 + 0.5 * df['Mulch_Flag'])
    df['Mulch_ET_Saving'] = df['ET_Proxy'] * (1 - 0.3 * df['Mulch_Flag'])

    df['Moisture_ET_Ratio'] = df['Soil_Moisture'] / (df['ET_Proxy'] + 0.1)
    df['Rain_ET_Balance'] = df['Rainfall_mm'] - df['ET_Proxy'] * 2
    df['Sunlight_Temp_Ratio'] = df['Sunlight_Hours'] / (df['Temperature_C'] + 1)
    df['Water_Stress_Index'] = (df['ET_Proxy'] - df['Total_Water_Input'] / 10) / (df['Soil_Moisture'] + 1)

    for col in ['Rainfall_mm', 'Previous_Irrigation_mm', 'Field_Area_hectare', 'Wind_Speed_kmh']:
        df[f'log_{col}'] = np.log1p(df[col])
    return df


def build_features():
    train = engineer_features(read_train().dropna(subset=[TARGET]))
    test = engineer_features(read_test())
    orig = engineer_features(read_original())

    # label encoding of the categoricals over train + test + original rows
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(pd.concat([train[col].astype(str), test[col].astype(str), orig[col].astype(str)]))
        train[col] = le.transform(train[col].astype(str)).astype('int32')
        test[col] = le.transform(test[col].astype(str)).astype('int32')
        orig[col] = le.transform(orig[col].astype(str)).astype('int32')

    # C(19, 2) pairwise combinations (raw codes; target-encoded per fold)
    te_columns = []
    n_train, n_test = len(train), len(test)
    for col_a, col_b in combinations(ALL_COLS, 2):
        name = f'{col_a}|{col_b}'
        tr_s = train[col_a].astype(str) + '_' + train[col_b].astype(str)
        te_s = test[col_a].astype(str) + '_' + test[col_b].astype(str)
        og_s = orig[col_a].astype(str) + '_' + orig[col_b].astype(str)
        combined = pd.concat([tr_s, te_s, og_s], ignore_index=True)
        codes, _ = combined.factorize()
        if pd.Series(codes).nunique() > len(combined) // 2:
            continue
        train[name] = codes[:n_train]
        test[name] = codes[n_train:n_train + n_test]
        orig[name] = codes[n_train + n_test:]
        te_columns.append(name)

    y_train = train[TARGET].map(TARGET_MAP).values.astype(int)
    y_orig = orig[TARGET].map(TARGET_MAP).values.astype(int)
    base_features = [c for c in train.columns
                     if c not in [ID, TARGET] and c not in te_columns and train[c].dtype != object]
    cat_feature_indices = [i for i, c in enumerate(base_features) if c in CAT_COLS]
    print(f'Base features: {len(base_features)} | Cat indices: {len(cat_feature_indices)} | '
          f'Pairwise TE: {len(te_columns)}', flush=True)
    return train, test, orig, y_train, y_orig, base_features, cat_feature_indices, te_columns


def main():
    args = fold_args()
    train, test, orig, y_train, y_orig, base_features, cat_feature_indices, te_columns = build_features()

    def fit_fold(fold, tr_idx, val_idx):
        # fold-train = the fold's competition rows + all original rows; fold-valid = competition rows only
        x_va, y_va = train.iloc[val_idx], y_train[val_idx]
        x_tr = pd.concat([train.iloc[tr_idx], orig], ignore_index=True)
        y_tr = np.concatenate([y_train[tr_idx], y_orig])

        # target encoding of the pair codes, fitted on the fold-train rows
        enc = TargetEncoder(target_type='multiclass', cv=5, random_state=SEED)
        tr_enc_arr = enc.fit_transform(x_tr[te_columns].values, y_tr)
        va_enc_arr = enc.transform(x_va[te_columns].values)
        te_enc_arr = enc.transform(test[te_columns].values)
        tr_enc = pd.DataFrame(tr_enc_arr, columns=[f'te_{i}' for i in range(tr_enc_arr.shape[1])])
        va_enc = pd.DataFrame(va_enc_arr, columns=tr_enc.columns)
        te_enc = pd.DataFrame(te_enc_arr, columns=tr_enc.columns)

        xt = pd.concat([x_tr[base_features].reset_index(drop=True).fillna(0).astype('float32'), tr_enc], axis=1)
        xv = pd.concat([x_va[base_features].reset_index(drop=True).fillna(0).astype('float32'), va_enc], axis=1)
        xe = pd.concat([test[base_features].reset_index(drop=True).fillna(0).astype('float32'), te_enc], axis=1)
        for c in CAT_COLS:
            if c in xt.columns:
                xt[c] = xt[c].astype(int)
                xv[c] = xv[c].astype(int)
                xe[c] = xe[c].astype(int)

        train_pool = Pool(xt, y_tr, cat_features=cat_feature_indices)
        val_pool = Pool(xv, y_va, cat_features=cat_feature_indices)
        with working_directory(work_dir(FOLDER)):
            model = CatBoostClassifier(**CAT_PARAMS)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        valid_pred = model.predict_proba(xv).astype(np.float32)
        test_pred = model.predict_proba(xe).astype(np.float32)
        print(f'fold {fold}: best_iter={model.best_iteration_}', flush=True)
        del xt, xv, xe, train_pool, val_pool, model
        gc.collect()
        return valid_pred, test_pred

    run_folds(FOLDER, fit_fold, args.folds, y=y_train)


if __name__ == '__main__':
    main()
