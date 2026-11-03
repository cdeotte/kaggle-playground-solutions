#!/usr/bin/env python
"""GPU XGBoost boosting from a base margin given by the deterministic formula of the original data (2 pairs).

The original data follows the rule score = 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30]
+ [Wind_Speed_kmh > 10] - 2 [Crop_Growth_Stage = Harvest] - 2 [Crop_Growth_Stage = Sowing] - [Mulching_Used = Yes]
(Low if score <= 0, Medium if 0 < score <= 3, High if score > 3). Every row (train, validation, test) gets a 3-class base
margin from it, and XGBoost (native API, multi:softprob) learns the correction on the 19 raw columns (categoricals as
native categoricals): depth 4, lr 0.05, subsample / colsample_bytree 0.8, balanced class weights, up to 50,000 rounds,
early stopping 500 (mlogloss) on the validation fold, seed 42; predictions use all trees of the booster returned by
xgb.train (the best iteration + 500). Competition rows only. Test = mean of the 5 fold models.

Recipes:
  xgb_rule_formula_margin_b    margin 2 for the formula class and -1 for the two others; max_bin 1024
  xgb_logit_formula_margin_b   margin = 3 linear logits (one per class) of the 4 threshold flags and the
                               Crop_Growth_Stage / Mulching_Used indicators; max_bin 256
Usage: python train_formula_margin_b.py <pair> [--folds 0,1,2,3,4] [--force]   (finished folds are kept in parts/)"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, labels, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore')

NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
FEATURES = NUMS + CATS


def rule_margin(df):
    """Logit 2 for the class given by the formula, -1 for the two other classes."""
    score = (df['Soil_Moisture'] < 25) * 2 + \
            (df['Rainfall_mm'] < 300) * 2 + \
            (df['Temperature_C'] > 30) * 1 + \
            (df['Wind_Speed_kmh'] > 10) * 1 - \
            (df['Crop_Growth_Stage'] == 'Harvest') * 2 - \
            (df['Crop_Growth_Stage'] == 'Sowing') * 2 - \
            (df['Mulching_Used'] == 'Yes') * 1
    margin = np.full((len(df), 3), -1.0)
    margin[score <= 0, 0] = 2.0
    margin[(score > 0) & (score <= 3), 1] = 2.0
    margin[score > 3, 2] = 2.0
    return margin


def logit_margin(df):
    """3 linear logits (Low, Medium, High) of the formula flags."""
    soil_lt_25 = (df['Soil_Moisture'] < 25).astype(float)
    temp_gt_30 = (df['Temperature_C'] > 30).astype(float)
    rain_lt_300 = (df['Rainfall_mm'] < 300).astype(float)
    wind_gt_10 = (df['Wind_Speed_kmh'] > 10).astype(float)
    flowering = (df['Crop_Growth_Stage'] == 'Flowering').astype(float)
    harvest = (df['Crop_Growth_Stage'] == 'Harvest').astype(float)
    sowing = (df['Crop_Growth_Stage'] == 'Sowing').astype(float)
    vegetative = (df['Crop_Growth_Stage'] == 'Vegetative').astype(float)
    mulch_no = (df['Mulching_Used'] == 'No').astype(float)
    mulch_yes = (df['Mulching_Used'] == 'Yes').astype(float)

    logit_low = 16.3173 + (-11.0237 * soil_lt_25) + (-5.8559 * temp_gt_30) + \
        (-10.8500 * rain_lt_300) + (-5.8284 * wind_gt_10) + \
        (-5.4155 * flowering) + (5.5073 * harvest) + (5.2299 * sowing) + \
        (-5.4617 * vegetative) + (-3.0014 * mulch_no) + (2.8613 * mulch_yes)
    logit_med = 4.6524 + (0.3290 * soil_lt_25) + (-0.0204 * temp_gt_30) + \
        (0.1542 * rain_lt_300) + (0.0841 * wind_gt_10) + \
        (0.3586 * flowering) + (-0.1348 * harvest) + (-0.3547 * sowing) + \
        (0.3334 * vegetative) + (0.1883 * mulch_no) + (0.0142 * mulch_yes)
    logit_high = -20.9697 + (10.6947 * soil_lt_25) + (5.8763 * temp_gt_30) + \
        (10.6958 * rain_lt_300) + (5.7444 * wind_gt_10) + \
        (5.0569 * flowering) + (-5.3725 * harvest) + (-4.8752 * sowing) + \
        (5.1283 * vegetative) + (2.8131 * mulch_no) + (-2.8755 * mulch_yes)
    return np.column_stack([logit_low, logit_med, logit_high])


BASE_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'random_state': 42,
    'device': 'cuda',
}
RECIPES = {
    'xgb_rule_formula_margin_b': dict(margin=rule_margin, params={**BASE_PARAMS, 'max_bin': 1024, 'nthread': -1},
                                      watch_train=True),
    'xgb_logit_formula_margin_b': dict(margin=logit_margin, params=dict(BASE_PARAMS), watch_train=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--force', action='store_true', help='rerun folds that are already done')
    args = ap.parse_args()
    pair, recipe = args.pair, RECIPES[args.pair]
    compute_margin = recipe['margin']
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, pair, k)]
    if todo:
        train = read_train()
        test = read_test().set_index(ID)
        train[TARGET] = train[TARGET].map(TARGET_MAP)
        for col in CATS:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
        fold_id = folds(train[TARGET].values)
        for k in todo:
            train_idx, val_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr, y_tr = train.iloc[train_idx][FEATURES], train.iloc[train_idx][TARGET]
            X_val, y_val = train.iloc[val_idx][FEATURES], train.iloc[val_idx][TARGET]
            margin_tr = compute_margin(train.iloc[train_idx])
            margin_val = compute_margin(train.iloc[val_idx])
            margin_te = compute_margin(test)

            classes = np.unique(y_tr)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
            class_weight_dict = dict(zip(classes, weights))
            sample_weights = np.array([class_weight_dict[label] for label in y_tr])

            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weights, enable_categorical=True)
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
            dtest = xgb.DMatrix(test[FEATURES], enable_categorical=True)
            dtrain.set_base_margin(margin_tr)
            dval.set_base_margin(margin_val)
            dtest.set_base_margin(margin_te)

            evals = [(dtrain, 'train'), (dval, 'valid')] if recipe['watch_train'] else [(dval, 'valid')]
            model = xgb.train(recipe['params'], dtrain, num_boost_round=50_000, evals=evals,
                              early_stopping_rounds=500, verbose_eval=500)
            val_pred = model.predict(dval)
            test_pred = model.predict(dtest)
            save_part(HERE, pair, k, val_pred, test_pred)
            acc = balanced_accuracy_score(y_val, np.argmax(val_pred, axis=1))
            print(f'{pair} fold {k}: balanced accuracy {acc:.5f}, best iteration {model.best_iteration}', flush=True)
            del dtrain, dval, dtest, X_tr, X_val, y_tr, y_val, model
            gc.collect()
    y = labels(read_train()[TARGET])
    assemble(HERE, pair, folds(y), y)


if __name__ == '__main__':
    main()
