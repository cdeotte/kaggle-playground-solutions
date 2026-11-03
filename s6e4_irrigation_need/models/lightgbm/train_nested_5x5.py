#!/usr/bin/env python
"""Class-weighted LightGBM in nested 5x5 cross-validation on category pairs, counts, original priors, bins and digits
(pair lgbm_nested5x5_cat_pairs_te_b).

The recipe adapts angelosmar1's public LightGBM notebook (2nd place, an earlier Playground competition):
https://www.kaggle.com/code/angelosmar1/s5e11-lgbm-cv-0-92813-2nd-place

Label-free features (train + test rows; the original rows give the priors):
  - the 11 numerics and the 8 categoricals (pandas categories, native in LightGBM)
  - count over train + test of every categorical value
  - per-class original frequency of every categorical value (missing: 0.33)
  - the last integer digit and the first decimal digit of soil moisture, rainfall, temperature
Encoded keys: the 8 categoricals, their 28 pairs, 100 quantile bins of soil moisture and 50 of rainfall.
Nested CV: for outer fold k (the shared 5 folds), an inner StratifiedKFold(5, seed 1000 + k + 1) of its training rows.
Per inner fold: scikit-learn TargetEncoder (multiclass, smooth 10, seed 42 + inner fold number) fitted on the inner
training rows and applied to all rows; LightGBM multiclass with balanced class sample weights, lr 0.01, max_depth 4,
subsample 0.5, colsample_bytree 0.2, lambda_l2 15, lambda_l1 10, seed 42, 32 threads, up to 7500 rounds with early
stopping (500) on the inner validation rows. Each inner model predicts its inner validation rows, the outer validation
rows and the test rows. OOF slice k = the inner out-of-fold predictions on outer fold k's training rows and the mean
of the 5 inner models on its validation rows; test = mean of all 25 models.
Writes an OOF of shape (5, n_train, 3).
Usage: python train_nested_5x5.py"""
import gc
import itertools
import sys
import warnings
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

PAIR = "lgbm_nested5x5_cat_pairs_te_b"
NUMS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 'Temperature_C', 'Humidity',
        'Rainfall_mm', 'Sunlight_Hours', 'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type', 'Water_Source', 'Mulching_Used',
        'Region']
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
RANDOM_STATE = 42

LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    n_estimators=7500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.5,
    colsample_bytree=0.2,
    lambda_l2=15.0,
    lambda_l1=10.0,
    random_state=RANDOM_STATE,
    verbosity=-1,
    n_jobs=32,
)


def create_categorical_combs(X, list_of_col_tuples, delimiter="_"):
    combs = {}
    for cols in list_of_col_tuples:
        combs[delimiter.join(cols)] = X[list(cols)].astype(str).agg(delimiter.join, axis=1)
    return pd.DataFrame(combs, index=X.index)


def create_count_features(X, list_of_inputs):
    list_of_inputs = [[col] if isinstance(col, str) else col for col in list_of_inputs]
    new_features = {}
    for cols in list_of_inputs:
        name = f"count({','.join(cols)})"
        new_features[name] = X.groupby(list(cols), dropna=False)[cols[0]].transform('size')
    return pd.DataFrame(new_features, index=X.index)


def create_groupby_features_original(X, list_of_inputs, original_data):
    """Per-class original frequency of every value (the 3 classes as 3 binary targets)."""
    new_features = {}
    for col in list_of_inputs:
        for class_idx in [0, 1, 2]:
            mapping = original_data.groupby(col, dropna=False)[f"target_{class_idx}"].agg("mean")
            name = f"groupby({col})_mean_class{class_idx}_original"
            new_features[name] = X[col].map(mapping).astype('float32').fillna(0.33)
    return pd.DataFrame(new_features, index=X.index)


def apply_te(df, enc, cols_in, cols_out):
    res = pd.DataFrame(enc.transform(df[cols_in]), columns=cols_out, index=df.index)
    return pd.concat([df.drop(columns=cols_in), res], axis=1)


def build_features():
    print("Loading data...", flush=True)
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    orig = read_original()
    if ID in orig.columns:
        orig = orig.drop(columns=[ID])
    train[TARGET] = train[TARGET].map(TARGET_MAP)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP)
    for c in [0, 1, 2]:
        orig[f'target_{c}'] = (orig[TARGET] == c).astype(int)

    y = train[TARGET].copy()
    combined = pd.concat([train.drop(columns=TARGET), test])

    print("Engineering features...", flush=True)
    high_cardinality = CATS.copy()
    combs = create_categorical_combs(combined, itertools.combinations(CATS, 2)).add_suffix("_te")
    ce1 = create_count_features(combined, high_cardinality)
    te1_orig = create_groupby_features_original(combined, high_cardinality, orig)
    sm_quantiled = pd.qcut(combined["Soil_Moisture"], q=100, duplicates='drop').astype(str).to_frame().add_suffix("_quantiled_te")
    rain_quantiled = pd.qcut(combined["Rainfall_mm"], q=50, duplicates='drop').astype(str).to_frame().add_suffix("_quantiled_te")

    digits = {}
    for col in ['Soil_Moisture', 'Rainfall_mm', 'Temperature_C']:
        splitted = combined[col].astype(str).str.split(".", expand=True)
        non_decimal = splitted[0].str.zfill(3)
        decimal = splitted[1].str.ljust(2, '0').fillna('00')
        digits[f"{col}_digit_last_int"] = non_decimal.str[-1].astype(int)
        digits[f"{col}_digit_first_dec"] = decimal.str[0].astype(int)
    digits_df = pd.DataFrame(digits, index=combined.index)

    te_base = combined[high_cardinality].astype(str).add_suffix("_te")
    combined = pd.concat([combined, te_base, ce1, te1_orig, sm_quantiled, rain_quantiled, digits_df, combs], axis=1)

    # string columns that are not target-encoded become native LightGBM categories
    cat_cols = combined.select_dtypes(["object", "string"]).columns.tolist()
    cat_cols = [c for c in cat_cols if not c.endswith("_te")]
    for c in cat_cols:
        combined[c] = combined[c].astype("category")

    X_train_full = combined.loc[train.index].copy()
    X_test_full = combined.loc[test.index].copy()
    return X_train_full, X_test_full, y


def main():
    X_train_full, X_test_full, y = build_features()
    n_train, n_test = len(X_train_full), len(X_test_full)

    fold_ids = folds(y.values)
    oof_matrix = np.zeros((N_OUTER_SPLITS, n_train, 3), dtype=np.float32)
    test_pred = np.zeros((n_test, 3), dtype=np.float32)
    for k in range(N_OUTER_SPLITS):
        outer_fold = k + 1
        outer_tr_idx = np.where(fold_ids != k)[0]
        outer_va_idx = np.where(fold_ids == k)[0]
        print(f"Outer fold {k}", flush=True)

        X_outer_tr, y_outer_tr = X_train_full.iloc[outer_tr_idx].copy(), y.iloc[outer_tr_idx].copy()
        X_outer_va, y_outer_va = X_train_full.iloc[outer_va_idx].copy(), y.iloc[outer_va_idx].copy()
        outer_full_pred = np.zeros((n_train, 3), dtype=np.float32)
        outer_va_pred_sum = np.zeros((len(outer_va_idx), 3), dtype=np.float32)

        skf_inner = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=1000 + outer_fold)
        for inner_fold, (inner_tr_loc, inner_va_loc) in enumerate(skf_inner.split(X_outer_tr, y_outer_tr), start=1):
            X_tr, y_tr = X_outer_tr.iloc[inner_tr_loc].copy(), y_outer_tr.iloc[inner_tr_loc].copy()
            X_va, y_va = X_outer_tr.iloc[inner_va_loc].copy(), y_outer_tr.iloc[inner_va_loc].copy()
            X_ova_inner = X_outer_va.copy()
            X_te_inner = X_test_full.copy()

            to_target_encode = [col for col in X_tr.columns if col.endswith("_te")]
            if len(to_target_encode) > 0:
                encoder = TargetEncoder(target_type='multiclass', cv=5, random_state=RANDOM_STATE + inner_fold, smooth=10.0)
                encoder.fit(X_tr[to_target_encode], y_tr)
                te_cols_out = encoder.get_feature_names_out(to_target_encode)
                X_tr = apply_te(X_tr, encoder, to_target_encode, te_cols_out)
                X_va = apply_te(X_va, encoder, to_target_encode, te_cols_out)
                X_ova_inner = apply_te(X_ova_inner, encoder, to_target_encode, te_cols_out)
                X_te_inner = apply_te(X_te_inner, encoder, to_target_encode, te_cols_out)

            classes = np.unique(y_tr)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
            class_weight_dict = dict(zip(classes, weights))
            sample_weights = np.array([class_weight_dict[label] for label in y_tr])

            model = lgbm.LGBMClassifier(**LGBM_PARAMS)
            model.fit(
                X_tr, y_tr,
                sample_weight=sample_weights,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgbm.early_stopping(stopping_rounds=500, verbose=False),
                    lgbm.log_evaluation(period=0),
                ],
            )

            p_inner_va = model.predict_proba(X_va).astype(np.float32)
            outer_full_pred[outer_tr_idx[inner_va_loc]] = p_inner_va
            inner_acc = balanced_accuracy_score(y_va, np.argmax(p_inner_va, axis=1))
            print(f"  outer {k} inner {inner_fold}: balanced accuracy {inner_acc:.5f}", flush=True)

            outer_va_pred_sum += model.predict_proba(X_ova_inner).astype(np.float32)
            test_pred += model.predict_proba(X_te_inner).astype(np.float32) / (N_OUTER_SPLITS * N_INNER_SPLITS)

            del X_tr, X_va, y_tr, y_va, X_ova_inner, X_te_inner, model, p_inner_va
            gc.collect()

        outer_va_pred = outer_va_pred_sum / N_INNER_SPLITS
        outer_full_pred[outer_va_idx] = outer_va_pred
        oof_matrix[k] = outer_full_pred
        outer_va_acc = balanced_accuracy_score(y_outer_va, np.argmax(outer_va_pred, axis=1))
        print(f"Outer fold {k}: validation balanced accuracy {outer_va_acc:.5f}", flush=True)

        del X_outer_tr, y_outer_tr, X_outer_va, y_outer_va, outer_full_pred, outer_va_pred_sum, outer_va_pred
        gc.collect()

    oof_mean = oof_matrix.mean(axis=0)
    print(f"{PAIR}: balanced accuracy of the slice mean {balanced_accuracy_score(y, np.argmax(oof_mean, axis=1)):.6f}",
          flush=True)
    save_pair(HERE, PAIR, oof_matrix, test_pred)


if __name__ == "__main__":
    main()
