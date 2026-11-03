#!/usr/bin/env python
"""Public notebook "Mobile Addiction || LGBM" (version 17) run on our 5 folds instead of its 10.
https://www.kaggle.com/code/vh10935cse20/mobile-addiction-lgbm

Features (the notebook's): missing indicators of the 9 numeric predictors, ~30 sleep / weekly / activity-share /
engagement / interaction / log features and 3 categorical combinations (stress x impact, gender x stress, all three).
Per fold: normalized frequencies of the 12 raw predictors and z-scores / differences of 5 usage columns against their
means within 3 categorical groups (all from the fold's training rows), and smoothed target encodings (smoothing 15) of
the 3 combinations and the 3 categoricals. The notebook fitted the target encodings on the fold's training rows and
applied them to those same rows; in our run the training rows get 5-fold inner out-of-fold encodings (validation and
test rows: the fold's training-row map, as in the notebook). The combination strings are then dropped and the
categoricals use the sorted levels of all frames. Columns are built in the same way, in a slightly different order.
Model (the notebook's): LGBMClassifier, lr 0.008, 45 leaves, depth 7, feature_fraction 0.7, bagging 0.85 every round,
min_child_samples 35, alpha 1.5, lambda 3.5, seed 42, up to 10,000 rounds with early stopping 300 on the validation
AUC. The notebook averaged rank-transformed fold predictions; our run averages the probabilities of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

NAME = 'vh10935cse20_lgbm'
THREADS = 24
NUM = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
       'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
CAT = ['stress_level', 'academic_work_impact', 'gender']
PARAMS = dict(objective='binary', metric='auc', boosting_type='gbdt', learning_rate=0.008, num_leaves=45, max_depth=7,
              feature_fraction=0.70, bagging_fraction=0.85, bagging_freq=1, min_child_samples=35, reg_alpha=1.5,
              reg_lambda=3.5, random_state=42, n_jobs=THREADS, verbose=-1, n_estimators=10000)


def engineer(df):
    df = df.copy()
    screen, sleep = df.daily_screen_time_hours, df.sleep_hours
    social, gaming, work = df.social_media_hours, df.gaming_hours, df.work_study_hours
    opens, notifs, weekend = df.app_opens_per_day, df.notifications_per_day, df.weekend_screen_time
    for col in NUM:
        df[f'{col}_isna'] = df[col].isnull().astype(int)
    df['awake_hours'] = (24.0 - sleep.clip(0, 23.5)).clip(lower=0.5)
    df['screen_awake_share'] = screen / df.awake_hours
    df['non_screen_awake_hours'] = (df.awake_hours - screen).clip(lower=0)
    df['sleep_deficit'] = (7.5 - sleep).clip(lower=0)
    df['sleep_deficit_screen_impact'] = df.sleep_deficit * screen
    df['est_weekly_screen_hours'] = screen * 5.0 + weekend
    df['weekend_vs_weekday_ratio'] = weekend / (screen * 2.0 + 1e-5)
    df['weekend_weekday_diff'] = weekend / 2.0 - screen
    df['activity_sum'] = social.fillna(0) + gaming.fillna(0) + work.fillna(0)
    df['unaccounted_screen_time'] = (screen - df.activity_sum).clip(lower=0)
    df['leisure_hours'] = social.fillna(0) + gaming.fillna(0)
    df['social_share'] = social / (screen + 1e-5)
    df['gaming_share'] = gaming / (screen + 1e-5)
    df['work_share'] = work / (screen + 1e-5)
    df['screen_to_work_ratio'] = screen / (work + 1e-5)
    df['leisure_to_work_ratio'] = df.leisure_hours / (work + 1e-5)
    df['screen_sleep_ratio'] = screen / (sleep + 1e-5)
    df['screen_sleep_diff'] = screen - sleep
    df['avg_session_len_min'] = screen * 60.0 / (opens + 1.0)
    df['opens_per_notification'] = opens / (notifs + 1.0)
    df['opens_per_screen_hour'] = opens / (screen + 1e-5)
    df['notifs_per_screen_hour'] = notifs / (screen + 1e-5)
    df['screen_per_open_min'] = screen * 60.0 / (opens + 1e-5)
    df['intensity_score'] = opens * notifs / (sleep + 1e-5)
    df['compulsion_index'] = opens * screen / (work + 1.0)
    df['digital_overload_index'] = opens * notifs * screen / (sleep + work + 1.0)
    df['night_screen_estimate'] = (screen - work - social - gaming).clip(lower=0) * (notifs / (opens + 1.0))
    df['log_app_opens'] = np.log1p(opens.clip(lower=0))
    df['log_notifications'] = np.log1p(notifs.clip(lower=0))
    df['log_screen_time'] = np.log1p(screen.clip(lower=0))
    df['stress_impact_combo'] = df.stress_level.astype(str) + '_' + df.academic_work_impact.astype(str)
    df['gender_stress_combo'] = df.gender.astype(str) + '_' + df.stress_level.astype(str)
    df['triple_combo'] = (df.gender.astype(str) + '_' + df.stress_level.astype(str) + '_'
                          + df.academic_work_impact.astype(str))
    for col in CAT:
        df[col] = df[col].astype(str)
    return df


def add_predictor_aggregates(xtr, xva, xte):
    """Differences / z-scores of 5 usage columns against their group means (groups and statistics from xtr)."""
    groups = [['stress_level', 'academic_work_impact'], ['gender', 'stress_level'], ['triple_combo']]
    measures = ['daily_screen_time_hours', 'app_opens_per_day', 'sleep_hours', 'social_media_hours',
                'notifications_per_day']
    for group in groups:
        name = '_'.join(group)
        keys = [frame[group].astype(str).agg('_'.join, axis=1) for frame in (xtr, xva, xte)]
        for measure in measures:
            means = xtr.groupby(keys[0])[measure].mean().to_dict()
            stds = xtr.groupby(keys[0])[measure].std().to_dict()
            default_mean = xtr[measure].mean()
            for key, frame in zip(keys, (xtr, xva, xte)):
                group_mean = key.map(means).fillna(default_mean)
                group_std = key.map(stds).fillna(1.0)
                frame[f'{measure}_diff_{name}_mean'] = frame[measure] - group_mean
                frame[f'{measure}_zscore_{name}'] = (frame[measure] - group_mean) / (group_std + 1e-5)


def add_nested_te(xtr, ytr, xva, xte):
    """In place: <col>_te (smoothing 15) of the 3 combinations and 3 categoricals; inner 5-fold out-of-fold on the
    training rows, the training-row map for validation / test."""
    columns = ['stress_impact_combo', 'gender_stress_combo', 'triple_combo', *CAT]
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_mean = float(ytr.mean())
    for col in columns:
        encoded = np.full(len(xtr), outer_mean, dtype=np.float64)
        for fit, hold in inner.split(xtr, ytr):
            fit_y, fit_key = ytr.iloc[fit], xtr.iloc[fit][col]
            mean = float(fit_y.mean())
            stats = pd.DataFrame({'key': fit_key.to_numpy(), 'y': fit_y.to_numpy()}).groupby('key').y.agg(
                ['count', 'mean'])
            mapping = (stats['count'] * stats['mean'] + 15.0 * mean) / (stats['count'] + 15.0)
            encoded[hold] = xtr.iloc[hold][col].map(mapping).fillna(mean)
        stats = pd.DataFrame({'key': xtr[col].to_numpy(), 'y': ytr.to_numpy()}).groupby('key').y.agg(['count', 'mean'])
        mapping = (stats['count'] * stats['mean'] + 15.0 * outer_mean) / (stats['count'] + 15.0)
        xtr[f'{col}_te'] = encoded
        xva[f'{col}_te'] = xva[col].map(mapping).fillna(outer_mean).astype(float)
        xte[f'{col}_te'] = xte[col].map(mapping).fillna(outer_mean).astype(float)


def main():
    fold_ids = fold_args()
    train, test = read_train(), read_test()
    y = train['addicted_label'].to_numpy(np.int8)
    base, base_test = engineer(train), engineer(test)
    features = [c for c in base if c not in ['id', 'addicted_label']]
    x, xt = base[features], base_test[features]

    def fit_fold(fold, tr_idx, va_idx):
        xtr, xva, xte = x.iloc[tr_idx].copy(), x.iloc[va_idx].copy(), xt.copy()
        ytr, yva = pd.Series(y[tr_idx], index=xtr.index), y[va_idx]
        for col in [*NUM, *CAT]:
            mapping = xtr[col].value_counts(normalize=True).to_dict()
            for frame in (xtr, xva, xte):
                frame[f'{col}_freq'] = frame[col].map(mapping).fillna(0).astype(float)
        add_predictor_aggregates(xtr, xva, xte)
        xtr.reset_index(drop=True, inplace=True)
        xva.reset_index(drop=True, inplace=True)
        xte.reset_index(drop=True, inplace=True)
        ytr.reset_index(drop=True, inplace=True)
        add_nested_te(xtr, ytr, xva, xte)
        drop = ['stress_impact_combo', 'gender_stress_combo', 'triple_combo']
        xtr.drop(columns=drop, inplace=True)
        xva.drop(columns=drop, inplace=True)
        xte.drop(columns=drop, inplace=True)
        for col in CAT:
            levels = sorted(set(xtr[col]) | set(xva[col]) | set(xte[col]))
            xtr[col] = pd.Categorical(xtr[col], categories=levels)
            xva[col] = pd.Categorical(xva[col], categories=levels)
            xte[col] = pd.Categorical(xte[col], categories=levels)
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(xtr, ytr, eval_set=[(xva, yva)],
                  callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False), lgb.log_evaluation(period=0)])
        print(f'fold {fold}: best iteration {model.best_iteration_}', flush=True)
        return model.predict_proba(xva)[:, 1], model.predict_proba(xte)[:, 1]

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
