"""Features of the two "S6E8: LightGBM | SAP" / "S6E8: XGBoost | SAP" public notebooks (same author, same feature code):
the 9 numeric columns (float32), 8 domain features (unexplained screen time, total activity, work / social / gaming
share of the daily screen time, max and min of daily / weekend / social screen time, awake hours), digit features (ones
digit of every numeric column, tenths and hundredths digits of the 6 fractional columns) and the 3 categoricals as
pandas categories (levels = sorted values of train + test). No target-dependent features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data import read_test, read_train  # noqa: E402

CAT = ['gender', 'stress_level', 'academic_work_impact']
NUM = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
       'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
# age, notifications_per_day and app_opens_per_day are whole numbers: decimal digits only for the other 6
FRACTIONAL = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
              'weekend_screen_time']


def sap_features(df):
    x = df[NUM].astype('float32').copy()
    daily = df.daily_screen_time_hours
    social, gaming = df.social_media_hours, df.gaming_hours
    work, sleep = df.work_study_hours, df.sleep_hours
    x['a_unexplained_screen'] = daily - social - gaming
    x['a_all_activity'] = social + gaming + work
    x['r_work_daily'] = work / (daily + 1e-3)
    x['r_soc_daily'] = social / (daily + 1e-3)
    x['r_game_daily'] = gaming / (daily + 1e-3)
    x['x_screen3_max'] = df[['daily_screen_time_hours', 'weekend_screen_time', 'social_media_hours']].max(axis=1)
    x['x_screen3_min'] = df[['daily_screen_time_hours', 'weekend_screen_time', 'social_media_hours']].min(axis=1)
    x['a_awake'] = 24.0 - sleep
    for col in NUM:
        value = df[col].astype('float32')
        x[col + '_digit_ones'] = np.floor(np.abs(value)) % 10
    for col in FRACTIONAL:
        value = df[col].astype('float32')
        x[col + '_digit_tenths'] = np.floor(np.abs(value) * 10) % 10
        x[col + '_digit_hundredths'] = np.floor(np.abs(value) * 100) % 10
    for col in CAT:
        x[col] = df[col]
    return x


def load():
    """Feature frames of train / test (categoricals as pandas categories) and the target."""
    train, test = read_train(), read_test()
    x, xt = sap_features(train), sap_features(test)
    for col in CAT:
        levels = sorted(set(x[col].dropna().unique()) | set(xt[col].dropna().unique()))
        x[col] = pd.Categorical(x[col], categories=levels)
        xt[col] = pd.Categorical(xt[col], categories=levels)
    return x, xt, train['addicted_label'].to_numpy(np.int8)
