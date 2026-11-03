#!/usr/bin/env python
"""Public notebook "PS S6E5: CatBoost 10-Fold CV" (version 6, published then as ps-s6e5-catboost) run on our 5 folds.
https://www.kaggle.com/code/rohit8527kmr7518/ps-s6e5-catboost-10-fold-cv

Features (the notebook's): an IsOriginalData flag; 38 domain features (estimated total laps, laps remaining, tyre-age
ratios, degradation / lap-time-delta per lap, pressures, products, sign flags, race-phase / lap / tyre-life / position
bins) and 9 string crosses (Race_Year, Driver_Race, Race_Compound_Stint, ...); integer and decimal digits of 22 numeric
columns; 5 scaled-digit signatures of 13 float columns (kept per frame where they vary); rounded-string copies of
RaceProgress, EstimatedTotalLaps and TyreAgeRatio. Per fold: counts / frequencies of 15 categorical columns and the
mean, std and difference-from-mean of LapTime_Delta, Position_Change, RaceProgress and TyreLife within Race_Year,
Race_Compound_Stint, Driver_Race and Compound_Stint, computed on the fold's training rows + test + all original rows;
missing values filled with the mode / median of those rows. 301 features; the 77 string columns are CatBoost
categorical features.
Original rows: all 101,371 original rows are appended to each fold's training rows.
Model (the notebook's): CatBoost on one GPU, Logloss, 9,000 iterations, lr 0.033, depth 8, l2 7, random strength 0.8,
Bayesian bagging temperature 0.55, balanced class weights, early stopping 350 on the validation AUC (best model kept),
seed 43 + fold. Predictions clipped to [1e-6, 1 - 1e-6]; test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

FOLDER = 'rohit8527kmr7518_catboost'
SEED = 42
CLIP_LOW, CLIP_HIGH = 1e-6, 1 - 1e-6
PARAMS = dict(iterations=9000, learning_rate=0.033, depth=8, l2_leaf_reg=7.0, random_strength=0.8,
              bootstrap_type='Bayesian', bagging_temperature=0.55, loss_function='Logloss', eval_metric='AUC',
              auto_class_weights='Balanced', task_type='GPU', devices='0', early_stopping_rounds=350,
              allow_writing_files=False, verbose=100)

BASE_CAT_COLS = ['Driver', 'Compound', 'Race']
DIGIT_SOURCE_COLS = ['Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position', 'LapTime (s)', 'LapTime_Delta',
                     'Cumulative_Degradation', 'RaceProgress', 'Position_Change', 'EstimatedTotalLaps', 'LapsRemaining',
                     'TyreAgeRatio', 'DegPerTyreLap', 'DegPerRaceLap', 'DeltaPerTyreLap', 'DeltaAbs',
                     'PositionPressure', 'StintPressure', 'PitWindowPressure', 'LapMinusTyreLife']
FREQ_COLS = ['Driver', 'Race', 'Compound', 'Race_Year', 'Compound_Stint', 'Driver_Race', 'Driver_Compound',
             'Race_Compound', 'Race_Compound_Stint', 'Compound_RacePhase', 'Compound_TyreLifeBin',
             'RacePhase_TyreLifeBin', 'LapBin', 'TyreLifeBin', 'PositionBin']
GROUP_COLS_LIST = [['Race_Year'], ['Race_Compound_Stint'], ['Driver_Race'], ['Compound_Stint']]
GROUP_VALUE_COLS = ['LapTime_Delta', 'Position_Change', 'RaceProgress', 'TyreLife']
SIGNATURE_COLS = ['RaceProgress', 'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'TyreAgeRatio',
                  'DegPerTyreLap', 'DegPerRaceLap', 'DeltaPerTyreLap', 'DeltaAbs', 'PitWindowPressure',
                  'EstimatedTotalLaps', 'LapsRemaining', 'LapMinusTyreLife']


def reduce_mem_usage(df):
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            c_min, c_max = out[col].min(), out[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                out[col] = out[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                out[col] = out[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                out[col] = out[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(dtype):
            out[col] = pd.to_numeric(out[col], downcast='float')
    return out


def safe_divide(a, b, eps=1e-6):
    return a / (b + eps)


def add_domain_features(df):
    out = df.copy()
    eps = 1e-6

    for col in BASE_CAT_COLS:
        if col in out.columns:
            out[col] = out[col].astype('string').fillna('__MISSING__').astype(str)

    if {'LapNumber', 'RaceProgress'}.issubset(out.columns):
        out['EstimatedTotalLaps'] = safe_divide(
            out['LapNumber'], out['RaceProgress'].clip(lower=eps), eps).replace([np.inf, -np.inf], np.nan).clip(1, 120)
        out['LapsRemaining'] = (out['EstimatedTotalLaps'] - out['LapNumber']).clip(lower=0)
        out['LapProgress_x_LapNumber'] = out['LapNumber'] * out['RaceProgress']

    if {'TyreLife', 'LapNumber'}.issubset(out.columns):
        out['TyreAgeRatio'] = safe_divide(out['TyreLife'], out['LapNumber'].clip(lower=1), eps)
        out['LapPerTyreLife'] = safe_divide(out['LapNumber'], out['TyreLife'] + 1, eps)
        out['LapMinusTyreLife'] = out['LapNumber'] - out['TyreLife']
        out['TyreLifeMinusLap'] = out['TyreLife'] - out['LapNumber']

    if {'TyreLife', 'EstimatedTotalLaps'}.issubset(out.columns):
        out['TyreAgeVsRace'] = safe_divide(out['TyreLife'], out['EstimatedTotalLaps'].clip(lower=1), eps)

    if {'TyreLife', 'RaceProgress'}.issubset(out.columns):
        out['PitWindowPressure'] = out['TyreLife'] * out['RaceProgress']
        out['TyreLife_x_RaceProgress'] = out['TyreLife'] * out['RaceProgress']

    if {'TyreLife', 'LapsRemaining'}.issubset(out.columns):
        out['TyreLife_to_LapsRemaining'] = safe_divide(out['TyreLife'], out['LapsRemaining'] + 1, eps)
        out['LapsRemaining_to_TyreLife'] = safe_divide(out['LapsRemaining'], out['TyreLife'] + 1, eps)

    if {'Cumulative_Degradation', 'TyreLife'}.issubset(out.columns):
        out['DegPerTyreLap'] = safe_divide(out['Cumulative_Degradation'], out['TyreLife'].clip(lower=1), eps)
        out['AbsDegPerTyreLap'] = safe_divide(out['Cumulative_Degradation'].abs(), out['TyreLife'].clip(lower=1), eps)

    if {'Cumulative_Degradation', 'LapNumber'}.issubset(out.columns):
        out['DegPerRaceLap'] = safe_divide(out['Cumulative_Degradation'], out['LapNumber'].clip(lower=1), eps)

    if {'LapTime_Delta', 'TyreLife'}.issubset(out.columns):
        out['DeltaPerTyreLap'] = safe_divide(out['LapTime_Delta'], out['TyreLife'].clip(lower=1), eps)
        out['AbsDeltaPerTyreLap'] = safe_divide(out['LapTime_Delta'].abs(), out['TyreLife'].clip(lower=1), eps)

    if 'LapTime_Delta' in out.columns:
        out['DeltaAbs'] = out['LapTime_Delta'].abs()
        out['LapTimeDeltaPositive'] = (out['LapTime_Delta'] > 0).astype(np.int8)
        out['LapTimeDeltaNegative'] = (out['LapTime_Delta'] < 0).astype(np.int8)

    if 'Cumulative_Degradation' in out.columns:
        out['Abs_Cumulative_Degradation'] = out['Cumulative_Degradation'].abs()
        out['Positive_Degradation'] = (out['Cumulative_Degradation'] > 0).astype(np.int8)

    if 'Position_Change' in out.columns:
        out['Abs_Position_Change'] = out['Position_Change'].abs()
        out['Gained_Position'] = (out['Position_Change'] > 0).astype(np.int8)
        out['Lost_Position'] = (out['Position_Change'] < 0).astype(np.int8)

    if {'Position', 'RaceProgress'}.issubset(out.columns):
        out['PositionPressure'] = out['Position'] * out['RaceProgress']

    if {'Stint', 'TyreLife'}.issubset(out.columns):
        out['StintPressure'] = out['Stint'] * out['TyreLife']
        out['TyreLife_x_Stint'] = out['TyreLife'] * out['Stint']

    if {'Stint', 'LapNumber'}.issubset(out.columns):
        out['Stint_x_LapNumber'] = out['Stint'] * out['LapNumber']
        out['Is_First_Stint'] = (out['Stint'] == 1).astype(np.int8)
        out['Is_Late_Stint'] = (out['Stint'] >= 3).astype(np.int8)

    if 'RaceProgress' in out.columns:
        out['Early_Race'] = (out['RaceProgress'] <= 0.25).astype(np.int8)
        out['Mid_Race'] = ((out['RaceProgress'] > 0.25) & (out['RaceProgress'] <= 0.65)).astype(np.int8)
        out['Late_Race'] = (out['RaceProgress'] > 0.65).astype(np.int8)
        out['RacePhase'] = pd.cut(out['RaceProgress'], bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
                                  labels=['P1', 'P2', 'P3', 'P4', 'P5']).astype(str)

    if 'LapNumber' in out.columns:
        out['LapBin'] = pd.cut(out['LapNumber'], bins=[-np.inf, 5, 10, 20, 35, 50, np.inf],
                               labels=['L_000_005', 'L_006_010', 'L_011_020', 'L_021_035', 'L_036_050',
                                       'L_051_plus']).astype(str)

    if 'TyreLife' in out.columns:
        out['TyreLifeBin'] = pd.cut(out['TyreLife'], bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
                                    labels=['T_000_003', 'T_004_007', 'T_008_012', 'T_013_020', 'T_021_030',
                                            'T_031_plus']).astype(str)

    if 'Position' in out.columns:
        out['PositionBin'] = pd.cut(out['Position'], bins=[-np.inf, 3, 8, 14, np.inf],
                                    labels=['front', 'upper_mid', 'lower_mid', 'back']).astype(str)

    def make_cross(name, cols):
        if set(cols).issubset(out.columns):
            val = out[cols[0]].astype(str)
            for c in cols[1:]:
                val = val + '_' + out[c].astype(str)
            out[name] = val

    make_cross('Race_Year', ['Race', 'Year'])
    make_cross('Compound_Stint', ['Compound', 'Stint'])
    make_cross('Driver_Race', ['Driver', 'Race'])
    make_cross('Driver_Compound', ['Driver', 'Compound'])
    make_cross('Race_Compound', ['Race', 'Compound'])
    make_cross('Race_Compound_Stint', ['Race', 'Compound', 'Stint'])
    make_cross('Compound_RacePhase', ['Compound', 'RacePhase'])
    make_cross('Compound_TyreLifeBin', ['Compound', 'TyreLifeBin'])
    make_cross('RacePhase_TyreLifeBin', ['RacePhase', 'TyreLifeBin'])

    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.select_dtypes(include=['float64']).columns:
        out[col] = out[col].astype(np.float32)
    return out


def add_digit_features(df, numeric_cols, int_digit_limit=3, decimal_digit_limit=2):
    out = df.copy()
    for col in numeric_cols:
        if col not in out.columns:
            continue
        abs_s = out[col].fillna(0).astype(float).abs()
        for i in range(int_digit_limit):
            out[f'{col}_int_digit_{i + 1}'] = ((abs_s // (10 ** i)) % 10).astype(np.int8)
        if pd.api.types.is_float_dtype(out[col]):
            for i in range(1, decimal_digit_limit + 1):
                out[f'{col}_dec_digit_{i}'] = ((abs_s * (10 ** i)).round().astype(int) % 10).astype(np.int8)
    return out


def add_float_signature_features(df):
    """Digits 1-5 of round(|x| * 100) as string categories; a digit column is kept only if it varies in this frame."""
    out = df.copy()
    for col in [c for c in SIGNATURE_COLS if c in out.columns]:
        scaled = (out[col].fillna(0).astype(float) * 100).round().astype(int).abs()
        for i in range(5):
            digit = ((scaled // (10 ** i)) % 10).astype(np.int8)
            if digit.nunique() > 1:
                out[f'{col}_sig_{i + 1}'] = digit.astype(str)
    return out


def add_string_precision_features(df):
    out = df.copy()
    if 'RaceProgress' in out.columns:
        out['RaceProgress_str'] = out['RaceProgress'].round(4).astype(str)
    if 'EstimatedTotalLaps' in out.columns:
        out['EstimatedTotalLaps_str'] = out['EstimatedTotalLaps'].round(1).astype(str)
    if 'TyreAgeRatio' in out.columns:
        out['TyreAgeRatio_str'] = out['TyreAgeRatio'].round(3).astype(str)
    return out


def add_frequency_features_fit_apply(fit_frames, apply_frames):
    """In place: <col>_count / <col>_freq of every FREQ_COLS value, counted over the fit frames."""
    freq_cols = [c for c in FREQ_COLS if all(c in f.columns for f in fit_frames)]
    base = pd.concat([f[freq_cols] for f in fit_frames], axis=0, ignore_index=True)
    total = len(base)
    for col in freq_cols:
        counts = base[col].astype(str).value_counts(dropna=False)
        for frame in apply_frames:
            frame[f'{col}_count'] = frame[col].astype(str).map(counts).fillna(0).astype(np.float32)
            frame[f'{col}_freq'] = (frame[f'{col}_count'] / total).astype(np.float32)
    del base
    gc.collect()
    return apply_frames


def add_light_group_stats_fit_apply(fit_frames, apply_frames):
    """Mean / std of each GROUP_VALUE_COLS column within each GROUP_COLS_LIST group over the fit frames, merged into the
    apply frames, plus the difference value - group mean."""
    for group_cols in GROUP_COLS_LIST:
        group_name = '_'.join(group_cols)
        for value_col in GROUP_VALUE_COLS:
            base = pd.concat([f[group_cols + [value_col]] for f in fit_frames], axis=0, ignore_index=True)
            stats = base.groupby(group_cols, dropna=False)[value_col].agg(['mean', 'std']).reset_index()
            stats.columns = group_cols + [f'{value_col}_mean_by_{group_name}', f'{value_col}_std_by_{group_name}']
            mean_col = f'{value_col}_mean_by_{group_name}'
            diff_col = f'{value_col}_diff_mean_by_{group_name}'
            for idx, frame in enumerate(apply_frames):
                apply_frames[idx] = frame.merge(stats, on=group_cols, how='left')
                apply_frames[idx][diff_col] = (apply_frames[idx][value_col] - apply_frames[idx][mean_col]).astype(
                    np.float32)
            del base, stats
            gc.collect()
    return apply_frames


def align_and_fill_fold(train_df, val_df, test_df, original_df):
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    feature_cols = [c for c in common_cols if c != TARGET]

    for frame in [train_df, val_df, test_df, original_df]:
        for col in feature_cols:
            if col not in frame.columns:
                frame[col] = np.nan

    train_df = train_df[feature_cols + [TARGET]].copy()
    val_df = val_df[feature_cols + [TARGET]].copy()
    test_df = test_df[feature_cols].copy()
    original_df = original_df[feature_cols + [TARGET]].copy()
    fill_fit_frames = [train_df, test_df, original_df]

    cat_cols = []
    for col in feature_cols:
        if any(frame[col].dtype == 'object' or str(frame[col].dtype).startswith('category')
               or str(frame[col].dtype).startswith('string') for frame in fill_fit_frames if col in frame.columns):
            cat_cols.append(col)
    num_cols = [c for c in feature_cols if c not in cat_cols]
    all_apply_frames = [train_df, val_df, test_df, original_df]

    for col in cat_cols:
        values = pd.concat([frame[col].astype('string') for frame in fill_fit_frames if col in frame.columns], axis=0)
        mode = values.mode(dropna=True)
        fill_value = mode.iloc[0] if len(mode) else '__MISSING__'
        for frame in all_apply_frames:
            frame[col] = frame[col].astype('string').fillna(fill_value).astype(str)

    for col in num_cols:
        values = pd.concat([frame[col] for frame in fill_fit_frames if col in frame.columns], axis=0)
        fill_value = values.replace([np.inf, -np.inf], np.nan).median()
        if pd.isna(fill_value):
            fill_value = 0
        for frame in all_apply_frames:
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
            if frame[col].dtype == 'float64':
                frame[col] = frame[col].astype(np.float32)

    train_df, val_df, test_df, original_df = (reduce_mem_usage(f) for f in (train_df, val_df, test_df, original_df))
    return train_df, val_df, test_df, original_df, cat_cols, feature_cols


def add_public_row_features(train_df, val_df, test_df, original_df):
    train_df, val_df, test_df, original_df = (f.copy() for f in (train_df, val_df, test_df, original_df))
    train_df['IsOriginalData'] = 0
    val_df['IsOriginalData'] = 0
    test_df['IsOriginalData'] = 0
    original_df['IsOriginalData'] = 1

    frames = [add_domain_features(f) for f in (train_df, val_df, test_df, original_df)]
    digit_cols = [c for c in DIGIT_SOURCE_COLS if c in frames[0].columns and c in frames[2].columns]
    frames = [add_digit_features(f, digit_cols) for f in frames]
    frames = [add_float_signature_features(f) for f in frames]
    frames = [add_string_precision_features(f) for f in frames]
    return frames


def prepare_fold_features(train_df, val_df, test_df, original_df):
    train_df, val_df, test_df, original_df = add_public_row_features(train_df, val_df, test_df, original_df)

    # frequency / group statistics over the fold's training rows + test + original rows
    fit_frames = [train_df, test_df, original_df]
    apply_frames = [train_df, val_df, test_df, original_df]
    apply_frames = add_frequency_features_fit_apply(fit_frames, apply_frames)
    apply_frames = add_light_group_stats_fit_apply(fit_frames, apply_frames)
    train_df, val_df, test_df, original_df = apply_frames

    train_df, val_df, test_df, original_df, cat_cols, feature_cols = align_and_fill_fold(
        train_df, val_df, test_df, original_df)

    model_features = [c for c in feature_cols if c != ID]
    cat_cols = [c for c in cat_cols if c in model_features]
    return (train_df[model_features].copy(), train_df[TARGET].astype(int).copy(),
            val_df[model_features].copy(), val_df[TARGET].astype(int).copy(),
            test_df[model_features].copy(),
            original_df[model_features].copy(), original_df[TARGET].astype(int).copy(),
            cat_cols)


def main():
    args = fold_args()
    train_raw, test_raw, original_raw = read_train(), read_test(), read_original()
    y = train_raw[TARGET].astype(int).to_numpy()

    def fit_fold(fold, tr_idx, val_idx):
        x_tr_comp, y_tr_comp, x_val, y_val, x_test, x_orig, y_orig, cat_cols = prepare_fold_features(
            train_raw.iloc[tr_idx].reset_index(drop=True), train_raw.iloc[val_idx].reset_index(drop=True),
            test_raw.copy().reset_index(drop=True), original_raw.copy().reset_index(drop=True))
        x_tr = pd.concat([x_tr_comp, x_orig], axis=0, ignore_index=True)
        y_tr = pd.concat([y_tr_comp.reset_index(drop=True), y_orig.reset_index(drop=True)], axis=0, ignore_index=True)
        print(f'fold {fold}: train {x_tr.shape}, valid {x_val.shape}, {len(cat_cols)} categorical', flush=True)

        model = CatBoostClassifier(**PARAMS, random_seed=SEED + fold + 1)
        model.fit(x_tr, y_tr, eval_set=(x_val, y_val), cat_features=cat_cols, use_best_model=True)
        print(f'fold {fold}: best iteration {model.get_best_iteration()}', flush=True)
        valid_pred = np.clip(model.predict_proba(x_val)[:, 1], CLIP_LOW, CLIP_HIGH)
        test_pred = np.clip(model.predict_proba(x_test)[:, 1], CLIP_LOW, CLIP_HIGH)
        del x_tr_comp, y_tr_comp, x_val, y_val, x_test, x_orig, y_orig, x_tr, y_tr, model
        gc.collect()
        return valid_pred, test_pred

    run_folds(FOLDER, FOLDER, fit_fold, args.folds, y=y)


if __name__ == '__main__':
    main()
