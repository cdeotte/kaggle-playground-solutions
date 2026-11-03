#!/usr/bin/env python
"""Public notebook "S6E8: CatBoost" (version 3) run on our 5 folds (the notebook's own split is the same:
StratifiedKFold(5, shuffle=True, random_state=42)).
https://www.kaggle.com/code/donmarch14/s6e8-catboost

Features (the notebook's make_features, label-free): the 12 raw columns (categoricals as strings, missing ->
'__MISSING__'), missing flags and count, 8 domain relations (leisure / other / activity screen time, weekend minus
daily, sleep deficit, notifications per open, social / gaming share), ones / tenths / hundredths digits of 10 columns,
the floor / round(0.1) / mod-10 / 20ths-of-fraction values of those columns as CatBoost categoricals, activity
statistics (mean, std, min, max, range, argmax) and 2 floor-pair categoricals.
Per fold (the notebook's add_fold_safe_te): smoothed target encodings (smoothing 20) of the 3 categoricals, their 3
pairs and 20-quantile bins (edges from the fold's training rows) of the 9 numeric predictors; inner 4-fold out-of-fold
(seed 143) for the training rows, the fold's training-row map for validation and test.
Model (the notebook's GPU branch): CatBoostClassifier, Logloss, 4,000 iterations, lr 0.06, depth 8, l2 5,
random_strength 0.5, border_count 128, Bernoulli subsample 0.8, seed 42, early stopping (od_wait 300) on the validation
AUC. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

NAME = 'donmarch14_catboost'
CAT = ['gender', 'stress_level', 'academic_work_impact']
NUM = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
       'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']
PARAMS = dict(loss_function='Logloss', eval_metric='AUC', iterations=4000, learning_rate=0.06, depth=8, l2_leaf_reg=5.0,
              random_seed=42, random_strength=0.5, border_count=128, od_type='Iter', od_wait=300, verbose=250,
              allow_writing_files=False, task_type='GPU', devices='0', bootstrap_type='Bernoulli', subsample=0.8)


def catboost_features(df):
    out = df.copy()
    raw = [c for c in out.columns if c != 'id']
    cats = out.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cats:
        out[col] = out[col].fillna('__MISSING__').astype(str)
    for col in raw:
        if out[col].isna().any():
            out[f'{col}_was_missing'] = out[col].isna().astype('int8')
    out['missing_count'] = out[raw].isna().sum(axis=1).astype('int8')
    out['leisure_hours'] = out.social_media_hours + out.gaming_hours
    out['other_screen'] = out.daily_screen_time_hours - (out.social_media_hours + out.gaming_hours
                                                         + out.work_study_hours)
    out['screen_activity_hours'] = out.social_media_hours + out.gaming_hours + out.work_study_hours
    out['weekend_minus_daily_screen'] = out.weekend_screen_time - out.daily_screen_time_hours
    out['sleep_deficit_from_8'] = 8.0 - out.sleep_hours
    out['notifications_per_open'] = out.notifications_per_day / (out.app_opens_per_day + 1.0)
    out['social_share_of_screen'] = out.social_media_hours / (out.daily_screen_time_hours + 0.1)
    out['gaming_share_of_screen'] = out.gaming_hours / (out.daily_screen_time_hours + 0.1)
    digits = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
              'weekend_screen_time', 'notifications_per_day', 'app_opens_per_day', 'age', 'other_screen']
    for col in digits:
        values = out[col].astype('float32')
        for position, multiplier in [('ones', 1), ('tenths', 10), ('hundredths', 100)]:
            out[f'{col}_digit_{position}'] = (np.floor(np.abs(values) * multiplier).astype('Int32') % 10).astype(
                'float32')
    for col in digits:
        values = out[col].astype('float32')
        for suffix, encoded in [('floor_cat', np.floor(values)), ('round1_cat', values.round(1) * 10),
                                ('mod10_cat', np.floor(np.abs(values)) % 10),
                                ('frac20_cat', np.floor((np.abs(values) % 1) * 20))]:
            name = f'{col}_{suffix}'
            out[name] = encoded.fillna(-999999).astype('Int32').astype(str)
            cats.append(name)
    activity = ['daily_screen_time_hours', 'weekend_screen_time', 'social_media_hours', 'gaming_hours',
                'work_study_hours', 'other_screen']
    av = out[activity]
    out['activity_mean'], out['activity_std'] = av.mean(axis=1), av.std(axis=1)
    out['activity_min'], out['activity_max'] = av.min(axis=1), av.max(axis=1)
    out['activity_range'] = out.activity_max - out.activity_min
    out['activity_argmax_cat'] = av.fillna(-np.inf).to_numpy().argmax(axis=1).astype(str)
    cats.append('activity_argmax_cat')
    for left, right in [('daily_screen_time_hours_floor_cat', 'weekend_screen_time_floor_cat'),
                        ('social_media_hours_floor_cat', 'gaming_hours_floor_cat')]:
        name = f'PAIR_{left}__{right}'
        out[name] = out[left] + '__' + out[right]
        cats.append(name)
    return out, cats


def catboost_te(xtr, ytr, xva, xte, smooth=20):
    keytr, keyva, keyte = pd.DataFrame(index=xtr.index), pd.DataFrame(index=xva.index), pd.DataFrame(index=xte.index)
    for col in CAT:
        keytr[col], keyva[col], keyte[col] = xtr[col].astype(str), xva[col].astype(str), xte[col].astype(str)
    for left, right in [('gender', 'stress_level'), ('gender', 'academic_work_impact'),
                        ('stress_level', 'academic_work_impact')]:
        name = f'{left}__{right}'
        keytr[name] = xtr[left].astype(str) + '__' + xtr[right].astype(str)
        keyva[name] = xva[left].astype(str) + '__' + xva[right].astype(str)
        keyte[name] = xte[left].astype(str) + '__' + xte[right].astype(str)
    for col in NUM:
        _, bins = pd.qcut(xtr[col], q=20, retbins=True, duplicates='drop')
        bins[0], bins[-1] = -np.inf, np.inf
        for frame, keys in [(xtr, keytr), (xva, keyva), (xte, keyte)]:
            keys[f'{col}_qbin'] = pd.cut(frame[col], bins=bins, labels=False,
                                         include_lowest=True).fillna(-1).astype(str)
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=143)
    mean = float(ytr.mean())
    for col in keytr:
        oof = np.full(len(keytr), mean, dtype='float32')
        for fit, hold in inner.split(keytr, ytr):
            stats = pd.DataFrame({'key': keytr.iloc[fit][col].to_numpy(), 'y': ytr.iloc[fit].to_numpy()}).groupby(
                'key').y.agg(['sum', 'count'])
            mapping = (stats['sum'] + smooth * mean) / (stats['count'] + smooth)
            oof[hold] = keytr.iloc[hold][col].map(mapping).fillna(mean)
        stats = pd.DataFrame({'key': keytr[col].to_numpy(), 'y': ytr.to_numpy()}).groupby('key').y.agg(['sum', 'count'])
        mapping = (stats['sum'] + smooth * mean) / (stats['count'] + smooth)
        xtr[f'TE_{col}'] = oof
        xva[f'TE_{col}'] = keyva[col].map(mapping).fillna(mean).astype('float32')
        xte[f'TE_{col}'] = keyte[col].map(mapping).fillna(mean).astype('float32')
    return xtr, xva, xte


def main():
    fold_ids = fold_args()
    train, test = read_train(), read_test()
    y = train['addicted_label'].to_numpy(np.int8)
    x, cats = catboost_features(train.drop(columns='addicted_label'))
    xt, test_cats = catboost_features(test)
    assert cats == test_cats
    features = [c for c in x if c != 'id']
    cat_indices = [features.index(c) for c in cats]
    x, xt = x[features], xt[features]

    def fit_fold(fold, tr, va):
        xtr, xva = x.iloc[tr].reset_index(drop=True), x.iloc[va].reset_index(drop=True)
        ytr, yva = pd.Series(y[tr]), pd.Series(y[va])
        xtr, xva, xte = catboost_te(xtr, ytr, xva, xt.reset_index(drop=True))
        model = CatBoostClassifier(**PARAMS)
        model.fit(xtr, ytr, cat_features=cat_indices, eval_set=(xva, yva), verbose=250)
        valid_pred, test_pred = model.predict_proba(xva)[:, 1], model.predict_proba(xte)[:, 1]
        print(f'fold {fold}: best iteration {model.get_best_iteration()}', flush=True)
        del model
        gc.collect()
        return valid_pred, test_pred

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
