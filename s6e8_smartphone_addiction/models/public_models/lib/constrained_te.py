"""Features of the two "S6E8: LGBM | LB 0.96965" / "S6E8: HistGradientBoosting | LB 0.96945" public notebooks (same
author, same feature code):

  * the 12 raw columns (3 native categoricals, NaN kept), missing_count and the 9 screen-time ratio features;
  * constrained-imputation features from the generator identity daily >= social + gaming + work_study: lower bound of a
    missing daily total, upper bound / midpoint / interval width of a single missing component, exact slack, case id,
    number of missing screen values;
  * frequency of every raw value (as a string) over train + test;
  * per fold: a scikit-learn TargetEncoder (cv=5, smooth 'auto', shuffle, seed 42) of the 12 raw columns as strings,
    fitted on the fold's training rows (cross-fitted for those rows), applied to the validation and test rows.
The notebooks used the same 5 folds as we do, so nothing had to be changed to run them on our folds."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data import read_test, read_train  # noqa: E402

CAT = ['gender', 'stress_level', 'academic_work_impact']


def constrained_features(train, test):
    raw = [c for c in train.columns if c not in ('id', 'addicted_label')]
    comp = ['social_media_hours', 'gaming_hours', 'work_study_hours']

    def add(df):
        d = df[raw].copy()
        sm, gm, wk = d.social_media_hours, d.gaming_hours, d.work_study_hours
        ds, sl, ws = d.daily_screen_time_hours, d.sleep_hours, d.weekend_screen_time
        ao, nt = d.app_opens_per_day, d.notifications_per_day
        d['missing_count'] = df[raw].isna().sum(axis=1).astype(float)
        d['total_breakdown_hours'] = sm.fillna(0) + gm.fillna(0) + wk.fillna(0)
        d['social_ratio'] = sm / (ds + 1e-5)
        d['gaming_ratio'] = gm / (ds + 1e-5)
        d['work_ratio'] = wk / (ds + 1e-5)
        d['unaccounted_screen_time'] = ds - d.total_breakdown_hours
        d['screen_to_sleep_ratio'] = ds / (sl + 1e-5)
        d['weekend_vs_daily_ratio'] = ws / (ds + 1e-5)
        d['app_opens_per_hour'] = ao / (ds + 1e-5)
        d['notifications_per_hour'] = nt / (ds + 1e-5)
        # constrained imputation: a missing daily total is bounded below by the known components, a single missing
        # component above by the slack
        nmc, dm = df[comp].isna().sum(axis=1), ds.isna()
        ksum, slack = df[comp].sum(axis=1, min_count=1), ds - df[comp].sum(axis=1, min_count=1)
        c_a, c_b, c_d = dm & (nmc == 0), (~dm) & (nmc == 1), (~dm) & (nmc == 0)
        d['ci_daily_lb'] = np.where(c_a, ksum, np.nan)
        d['ci_missing_comp_ub'] = np.where(c_b, slack, np.nan)
        d['ci_missing_comp_mid'] = np.where(c_b, np.clip(slack / 2, 0, None), np.nan)
        d['ci_interval_width'] = np.where(c_b, slack, np.nan)
        d['ci_slack_exact'] = np.where(c_d, slack, np.nan)
        d['ci_case'] = np.select([c_a, c_b], [1.0, 2.0], default=0.0)
        d['ci_n_missing'] = (dm.astype(int) + nmc).astype(float)
        return d

    x, xt = add(train), add(test)
    for col in CAT:
        x[col] = x[col].astype('category')
        xt[col] = xt[col].astype('category').cat.set_categories(x[col].cat.categories)
    xs, xst = train[raw].astype(str).fillna('na'), test[raw].astype(str).fillna('na')
    for col in raw:
        frequency = pd.concat([xs[col], xst[col]]).value_counts()
        x[f'{col}_freq'] = xs[col].map(frequency).astype(float)
        xt[f'{col}_freq'] = xst[col].map(frequency).astype(float)
    return x, xt, xs, xst, raw


def load():
    """Feature frames of train / test, the string views to target-encode, the raw column names and the target."""
    train, test = read_train(), read_test()
    x, xt, xs, xst, raw = constrained_features(train, test)
    return x, xt, xs, xst, raw, train['addicted_label'].to_numpy(np.int8)


def fold_frames(x, xt, xs, xst, raw, y, train_index, valid_index):
    """Training / validation / test matrices of one fold: the features plus the fold's target encodings."""
    te_names = [f'te_{c}' for c in raw]
    encoder = TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=42, target_type='binary')
    ztr = encoder.fit_transform(xs.iloc[train_index], y[train_index])
    zva, zte = encoder.transform(xs.iloc[valid_index]), encoder.transform(xst)
    atr = pd.concat([x.iloc[train_index].reset_index(drop=True), pd.DataFrame(ztr, columns=te_names)], axis=1)
    ava = pd.concat([x.iloc[valid_index].reset_index(drop=True), pd.DataFrame(zva, columns=te_names)], axis=1)
    ate = pd.concat([xt.reset_index(drop=True), pd.DataFrame(zte, columns=te_names)], axis=1)
    return atr, ava, ate
