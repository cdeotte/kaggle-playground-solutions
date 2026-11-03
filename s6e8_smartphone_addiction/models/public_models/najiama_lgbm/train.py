#!/usr/bin/env python
"""Public notebook "Single LGBM Model LB 0.96988 CV 0.96863" (version 2) run on our 5 folds.
https://www.kaggle.com/code/najiama/single-lgbm-model-lb-0-96988-cv-0-96863

Features (the notebook's, label-free part built once on train + test): the 9 numeric predictors; stress_level mapped
to 0/1/2 and gender / academic_work_impact label-encoded (missing -> 'Unknown'), all three and ci_case as categories;
missing counts, active / passive screen time, 8 ratios, 5 threshold flags, constrained-imputation features (bound of a
missing daily total, bound / midpoint of one missing component, exact slack, case id); the frequency of every numeric
value over train + test. The 10 columns the notebook found least important are dropped.
Per fold: a smoothed target encoding (smoothing 30) of every numeric predictor. The notebook fitted it on the fold's
training rows, applied it to those same rows and smoothed towards the mean of all training rows; in our run the training
rows get 5-fold inner out-of-fold encodings and the smoothing uses the means of the fold's own training rows.
The categories use the sorted levels of train + test in every frame.
Model (the notebook's): LGBMClassifier, lr 0.01, 127 leaves, min_child_samples 200, feature_fraction 0.34,
bagging 0.75 / 5, alpha 0.1, lambda 1, max_bin 1023, seed 42, up to 10,000 rounds with early stopping 500 on the
validation AUC. Test = mean of the 5 fold models.
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

NAME = 'najiama_lgbm'
THREADS = 24
TARGET, ID = 'addicted_label', 'id'
NUM = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
       'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
CAT = ['gender', 'stress_level', 'academic_work_impact']
PARAMS = dict(objective='binary', metric='auc', boosting_type='gbdt', n_estimators=10000, learning_rate=0.01,
              num_leaves=127, max_depth=-1, min_child_samples=200, feature_fraction=0.34, bagging_fraction=0.75,
              bagging_freq=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=THREADS, verbose=-1, max_bin=1023)
# the notebook's 10 least important columns
LOW_IMPORTANCE = ['ci_case', 'very_sleep_deprived', 'missing_cat_count', 'heavy_screen_user', 'high_notifications',
                  'sleep_deprived', 'heavy_social_user', 'missing_num_count', 'gender', 'missing_count']


def features(df):
    df = df.copy()
    df['missing_count'] = df[NUM + CAT].isnull().sum(axis=1)
    df['missing_num_count'] = df[NUM].isnull().sum(axis=1)
    df['missing_cat_count'] = df[CAT].isnull().sum(axis=1)
    active = ['social_media_hours', 'gaming_hours', 'work_study_hours']
    df['total_active_hours'] = df[active].sum(axis=1, min_count=1)
    df['passive_screen_time'] = df.daily_screen_time_hours - df.total_active_hours
    df['social_ratio'] = df.social_media_hours / (df.daily_screen_time_hours + 1e-9)
    df['gaming_ratio'] = df.gaming_hours / (df.daily_screen_time_hours + 1e-9)
    df['work_ratio'] = df.work_study_hours / (df.daily_screen_time_hours + 1e-9)
    df['screen_to_sleep'] = df.daily_screen_time_hours / (df.sleep_hours + 1e-9)
    df['weekend_vs_daily'] = df.weekend_screen_time / (df.daily_screen_time_hours + 1e-9)
    df['app_opens_per_hour'] = df.app_opens_per_day / (df.daily_screen_time_hours + 1e-9)
    df['notif_per_hour'] = df.notifications_per_day / (df.daily_screen_time_hours + 1e-9)
    df['notif_per_app_open'] = df.notifications_per_day / (df.app_opens_per_day + 1e-9)
    df['sleep_deprived'] = np.where(df.sleep_hours.isna(), np.nan, (df.sleep_hours < 6).astype(float))
    df['very_sleep_deprived'] = np.where(df.sleep_hours.isna(), np.nan, (df.sleep_hours < 5).astype(float))
    df['heavy_screen_user'] = np.where(df.daily_screen_time_hours.isna(), np.nan,
                                       (df.daily_screen_time_hours > 8).astype(float))
    df['heavy_social_user'] = np.where(df.social_media_hours.isna(), np.nan, (df.social_media_hours > 3).astype(float))
    df['high_notifications'] = np.where(df.notifications_per_day.isna(), np.nan,
                                        (df.notifications_per_day > 150).astype(float))
    comp, daily = ['social_media_hours', 'gaming_hours', 'work_study_hours'], df.daily_screen_time_hours
    df['total_breakdown_hours'] = df[comp].sum(axis=1, min_count=1)
    slack = daily - df.total_breakdown_hours
    missing_comp, missing_daily = df[comp].isna().sum(axis=1), daily.isna()
    case_a = missing_daily & (missing_comp == 0)
    case_b = (~missing_daily) & (missing_comp == 1)
    case_d = (~missing_daily) & (missing_comp == 0)
    df['ci_daily_lb'] = np.where(case_a, df.total_breakdown_hours, np.nan)
    df['ci_missing_comp_ub'] = np.where(case_b, slack, np.nan)
    df['ci_missing_comp_mid'] = np.where(case_b, np.clip(slack / 2, 0, None), np.nan)
    df['ci_slack_exact'] = np.where(case_d, slack, np.nan)
    df['ci_case'] = np.select([case_a, case_b], [1.0, 2.0], default=0.0).astype('float32')
    return df


def nested_te(xtr, ytr, xva, xte):
    """In place: <col>_target_enc of every numeric predictor (smoothing 30); inner 5-fold out-of-fold on the training
    rows, the training-row map for validation / test."""
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mean = float(ytr.mean())
    for col in NUM:
        output = np.full(len(xtr), mean, dtype=np.float64)
        for fit, hold in inner.split(xtr, ytr):
            fit_mean = float(ytr.iloc[fit].mean())
            key = xtr.iloc[fit][col].astype(str)
            stats = pd.DataFrame({'key': key.to_numpy(), 'y': ytr.iloc[fit].to_numpy()}).groupby('key').y.agg(
                ['mean', 'count'])
            mapping = (stats['mean'] * stats['count'] + fit_mean * 30) / (stats['count'] + 30)
            output[hold] = xtr.iloc[hold][col].astype(str).map(mapping).fillna(fit_mean)
        key = xtr[col].astype(str)
        stats = pd.DataFrame({'key': key.to_numpy(), 'y': ytr.to_numpy()}).groupby('key').y.agg(['mean', 'count'])
        mapping = (stats['mean'] * stats['count'] + mean * 30) / (stats['count'] + 30)
        xtr[f'{col}_target_enc'] = output
        xva[f'{col}_target_enc'] = xva[col].astype(str).map(mapping).fillna(mean)
        xte[f'{col}_target_enc'] = xte[col].astype(str).map(mapping).fillna(mean)


def main():
    fold_ids = fold_args()
    train, test = read_train(), read_test()
    y = train[TARGET].to_numpy(np.int8)
    train, test = features(train), features(test)
    stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
    train.stress_level, test.stress_level = train.stress_level.map(stress_map), test.stress_level.map(stress_map)
    for col in ['gender', 'academic_work_impact']:
        combined = pd.concat([train[col], test[col]]).fillna('Unknown').astype(str)
        codes = {value: i for i, value in enumerate(combined.unique())}
        train[col] = train[col].fillna('Unknown').astype(str).map(codes)
        test[col] = test[col].fillna('Unknown').astype(str).map(codes)
    for col in [*CAT, 'ci_case']:
        levels = sorted(set(train[col].dropna()) | set(test[col].dropna()))
        train[col] = pd.Categorical(train[col], categories=levels)
        test[col] = pd.Categorical(test[col], categories=levels)
    for col in NUM:
        counts = pd.concat([train[col].astype(str), test[col].astype(str)]).value_counts()
        train[f'{col}_freq'] = train[col].astype(str).map(counts).fillna(0).astype(int)
        test[f'{col}_freq'] = test[col].astype(str).map(counts).fillna(0).astype(int)
        train[f'{col}_target_enc'] = np.nan
        test[f'{col}_target_enc'] = np.nan
    columns = [c for c in train if c not in [ID, TARGET, *LOW_IMPORTANCE]]
    x, xt = train[columns], test[columns]

    def fit_fold(fold, tr_idx, va_idx):
        xtr, xva, xte = x.iloc[tr_idx].copy(), x.iloc[va_idx].copy(), xt.copy()
        ytr, yva = pd.Series(y[tr_idx]), y[va_idx]
        xtr.reset_index(drop=True, inplace=True)
        xva.reset_index(drop=True, inplace=True)
        xte.reset_index(drop=True, inplace=True)
        nested_te(xtr, ytr, xva, xte)
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(xtr, ytr, eval_set=[(xva, yva)],
                  callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(300)])
        print(f'fold {fold}: best iteration {model.best_iteration_}', flush=True)
        return model.predict_proba(xva)[:, 1], model.predict_proba(xte)[:, 1]

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
