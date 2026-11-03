#!/usr/bin/env python
"""Public notebook "S6E4 🌾| XGBoost Adv FE | 0.979 CV" run on our 5 folds, without its pseudo-labelling and threshold
tuning.
https://www.kaggle.com/code/blamerx/s6e4-xgboost-adv-fe-0-979-cv

Features (80 + 405): the 19 raw columns (categoricals label-encoded over train + original + test rows); formula
indicators (Soil_Moisture < 25, Rainfall_mm < 300, Temperature_C > 30, Wind_Speed_kmh > 10, Harvest, Sowing,
Mulching = Yes), the formula score 2 soil + 2 rain + temp + wind - 2 harvest - 2 sowing - mulch, its distances to the
class boundaries 0 and 3, and the signed distances to the 4 thresholds; logit_Low / logit_Medium / logit_High, a fixed
logistic regression of those indicators fitted on the original data; the first decimal digit of the 11 numerics;
32 agronomy features (water input, ET / evaporation / VPD / heat / dryness / aridity / drought proxies, soil
health, salinity, pH deviation, per-hectare ratios, mulch effects, logs); per fold, a multiclass TargetEncoder (cv=5,
seed 42) of every pairwise combination of the 19 columns that has at most half as many distinct values as there are
train + original + test rows (3 columns per pair), fitted on the fold's training rows + original rows.
Original rows: all 10,000 are appended to each fold's training rows; balanced sample weights over those rows.
Model: xgb.train (GPU, hist) multi:softprob with the notebook's tuned parameters (max_depth 4, lr 0.0305), up to
10,000 rounds, early stopping 300 on the validation fold's balanced accuracy (predictions use all trees grown), seed 42.
Test = mean of the 5 fold models. Usage: python train.py [--folds 0,1,2,3,4]"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, TARGET_MAP, read_original, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

warnings.filterwarnings('ignore')

FOLDER = 'blamerx_xgb'
SEED = 42

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']

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

XGB_PARAMS = dict(
    max_depth=4,
    learning_rate=0.030495387759654796,
    min_child_weight=2.333941903991847,
    subsample=0.9766412297733108,
    colsample_bytree=0.535324419516146,
    gamma=4.258489082295074,
    reg_alpha=4.082875850185249e-08,
    reg_lambda=0.00013528868091784412,
    objective='multi:softprob',
    num_class=3,
    tree_method='hist',
    device='cuda',
    eval_metric='mlogloss',
    seed=SEED,
)


def compute_logit_scores(df):
    """Logit scores of the 3 classes from the fixed coefficients."""
    soil = (df['Soil_Moisture'] < SOIL_THRESH).astype(float)
    temp = (df['Temperature_C'] > TEMP_THRESH).astype(float)
    rain = (df['Rainfall_mm'] < RAIN_THRESH).astype(float)
    wind = (df['Wind_Speed_kmh'] > WIND_THRESH).astype(float)
    stage = df['Crop_Growth_Stage'].astype(str)
    mulch = df['Mulching_Used'].astype(str)

    is_fl = (stage == 'Flowering').astype(float)
    is_ha = (stage == 'Harvest').astype(float)
    is_so = (stage == 'Sowing').astype(float)
    is_ve = (stage == 'Vegetative').astype(float)
    m_no = (mulch == 'No').astype(float)
    m_yes = (mulch == 'Yes').astype(float)

    for cls_name, c in LOGIT_COEFS.items():
        df[f'logit_{cls_name}'] = (
            c['intercept']
            + c['soil_lt_25'] * soil + c['temp_gt_30'] * temp
            + c['rain_lt_300'] * rain + c['wind_gt_10'] * wind
            + c['Flowering'] * is_fl + c['Harvest'] * is_ha
            + c['Sowing'] * is_so + c['Vegetative'] * is_ve
            + c['Mulch_No'] * m_no + c['Mulch_Yes'] * m_yes
        ).astype('float32')
    return df


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
    high_s = 2 * df['soil_lt_25'] + 2 * df['rain_lt_300'] + df['temp_gt_30'] + df['wind_gt_10']
    low_s = 2 * df['is_harvest'] + 2 * df['is_sowing'] + df['mulching_yes']
    df['magic_score'] = high_s - low_s

    # first decimal digit of the 11 numerics
    for col in NUMS:
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

    mulch_flag = (df['Mulching_Used'].astype(str) == 'Yes').astype(float)
    df['Mulch_Flag'] = mulch_flag
    df['Mulch_Moisture'] = df['Soil_Moisture'] * (1 + 0.5 * mulch_flag)
    df['Mulch_ET_Saving'] = df['ET_Proxy'] * (1 - 0.3 * mulch_flag)

    df['Moisture_ET_Ratio'] = df['Soil_Moisture'] / (df['ET_Proxy'] + 0.1)
    df['Rain_ET_Balance'] = df['Rainfall_mm'] - df['ET_Proxy'] * 2
    df['Sunlight_Temp_Ratio'] = df['Sunlight_Hours'] / (df['Temperature_C'] + 1)
    df['Water_Stress_Index'] = (df['ET_Proxy'] - df['Total_Water_Input'] / 10) / (df['Soil_Moisture'] + 1)

    for col in ['Rainfall_mm', 'Previous_Irrigation_mm', 'Field_Area_hectare', 'Wind_Speed_kmh']:
        df[f'log_{col}'] = np.log1p(df[col])
    return df


def build_features():
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    orig = read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(int)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(int)
    y_train, y_orig = train[TARGET].values, orig[TARGET].values

    train_eng = engineer_features(train.drop(columns=[TARGET]))
    orig_eng = engineer_features(orig.drop(columns=[TARGET]))
    test_eng = engineer_features(test)

    # label encoding of the categoricals over train + original + test rows
    for col in CATS:
        le = LabelEncoder()
        le.fit(pd.concat([train_eng[col], orig_eng[col], test_eng[col]]).astype(str))
        train_eng[col] = le.transform(train_eng[col].astype(str)).astype('int32')
        orig_eng[col] = le.transform(orig_eng[col].astype(str)).astype('int32')
        test_eng[col] = le.transform(test_eng[col].astype(str)).astype('int32')

    # C(19, 2) pairwise combinations, factorized over train + original + test rows
    n_tr, n_or = len(train_eng), len(orig_eng)
    combined_all = pd.concat([train_eng, orig_eng, test_eng], ignore_index=True)
    te_columns = []
    for c1, c2 in combinations(NUMS + CATS, 2):
        name = f'{c1}|{c2}'
        codes, _ = (combined_all[c1].astype(str) + '_' + combined_all[c2].astype(str)).factorize()
        if pd.Series(codes).nunique() > len(codes) // 2:
            continue
        combined_all[name] = codes
        te_columns.append(name)

    train_fe = combined_all.iloc[:n_tr].reset_index(drop=True)
    orig_fe = combined_all.iloc[n_tr:n_tr + n_or].reset_index(drop=True)
    test_fe = combined_all.iloc[n_tr + n_or:].reset_index(drop=True)
    base_features = [c for c in train_fe.columns if c not in te_columns and train_fe[c].dtype != object]
    print(f'Base features: {len(base_features)}, Interaction features: {len(te_columns)}', flush=True)
    return train_fe, orig_fe, test_fe, y_train, y_orig, base_features, te_columns


def feval_bal_acc(preds, dmatrix):
    """Balanced accuracy, the early-stopping metric."""
    labels = dmatrix.get_label().astype(int)
    y_pred = preds.reshape(-1, 3).argmax(axis=1)
    return 'bal_acc', balanced_accuracy_score(labels, y_pred)


def main():
    args = fold_args()
    train_fe, orig_fe, test_fe, y_train, y_orig, base_features, te_columns = build_features()

    def fit_fold(fold, tr_idx, val_idx):
        # fold-train = the fold's competition rows + all original rows
        x_tr_base = pd.concat([train_fe.iloc[tr_idx][base_features].reset_index(drop=True), orig_fe[base_features]],
                              ignore_index=True).fillna(0).astype('float32')
        y_tr = np.concatenate([y_train[tr_idx], y_orig])
        x_va_base = train_fe.iloc[val_idx][base_features].reset_index(drop=True).fillna(0).astype('float32')
        y_va = y_train[val_idx]
        x_te_base = test_fe[base_features].reset_index(drop=True).fillna(0).astype('float32')

        # target encoding of the pair codes (as strings), fitted on the fold-train rows
        x_tr_pair = pd.concat([train_fe.iloc[tr_idx][te_columns].reset_index(drop=True).astype(str),
                               orig_fe[te_columns].astype(str)], ignore_index=True)
        x_va_pair = train_fe.iloc[val_idx][te_columns].reset_index(drop=True).astype(str)
        x_te_pair = test_fe[te_columns].astype(str)
        enc = TargetEncoder(target_type='multiclass', cv=5, random_state=SEED)
        te_tr = enc.fit_transform(x_tr_pair, y_tr).astype(np.float32)
        te_va = enc.transform(x_va_pair).astype(np.float32)
        te_te = enc.transform(x_te_pair).astype(np.float32)

        te_names = [f'TE_{i}' for i in range(te_tr.shape[1])]
        xt = pd.concat([x_tr_base, pd.DataFrame(te_tr, columns=te_names)], axis=1).astype('float32')
        xv = pd.concat([x_va_base, pd.DataFrame(te_va, columns=te_names)], axis=1).astype('float32')
        xe = pd.concat([x_te_base, pd.DataFrame(te_te, columns=te_names)], axis=1).astype('float32')

        sw = compute_sample_weight('balanced', y_tr)
        dtrain = xgb.DMatrix(xt, label=y_tr, weight=sw)
        dval = xgb.DMatrix(xv, label=y_va)
        dtest = xgb.DMatrix(xe)
        model = xgb.train(params=XGB_PARAMS, dtrain=dtrain, num_boost_round=10000,
                          evals=[(dtrain, 'train'), (dval, 'val')], custom_metric=feval_bal_acc, maximize=True,
                          early_stopping_rounds=300, verbose_eval=500)
        valid_pred = model.predict(dval).reshape(-1, 3).astype(np.float32)
        test_pred = model.predict(dtest).reshape(-1, 3).astype(np.float32)
        print(f'fold {fold}: best_iter={model.best_iteration}', flush=True)
        del xt, xv, xe, dtrain, dval, dtest, model
        gc.collect()
        return valid_pred, test_pred

    run_folds(FOLDER, fit_fold, args.folds, y=y_train)


if __name__ == '__main__':
    main()
