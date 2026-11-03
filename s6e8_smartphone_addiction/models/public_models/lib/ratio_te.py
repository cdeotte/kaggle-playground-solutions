"""Features shared by the RealMLP, TabM, FT-Transformer, ResNet, CatBoost, TabNet, XGBoost and FLAML-XGBoost public
notebooks of one author (same template):

  * every raw predictor is cast to a string category;
  * missing_count (number of 'nan' predictors) and 9 screen-time ratio features;
  * per raw predictor a smoothed target encoding (smoothing 10) and a frequency encoding.

The notebooks computed the encodings once on all training rows before their CV loop. In our run they are computed inside
each of our folds: the training rows get 5-fold inner out-of-fold target encodings (StratifiedKFold(5, shuffle, 42)),
validation and test rows the map fitted on all training rows of the fold; frequencies are counted on the fold's
training rows."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data import read_test, read_train  # noqa: E402

TARGET, ID = 'addicted_label', 'id'


def categorical_frames():
    """train / test with every raw predictor cast to a string category (levels = sorted values of train + test), as the
    notebooks do before their feature engineering. Returns train, test and the 12 predictor names."""
    train, test = read_train(), read_test()
    initial = [c for c in train if c not in [TARGET, ID]]
    for col in initial:
        levels = sorted(set(train[col].astype(str)) | set(test[col].astype(str)))
        train[col] = pd.Categorical(train[col].astype(str), categories=levels)
        test[col] = pd.Categorical(test[col].astype(str), categories=levels)
    return train, test, initial


def _numbers(out):
    def number(name):
        return pd.to_numeric(out[name].astype(str), errors='coerce')
    return (number('social_media_hours'), number('gaming_hours'), number('work_study_hours'),
            number('daily_screen_time_hours'), number('sleep_hours'), number('weekend_screen_time'),
            number('app_opens_per_day'), number('notifications_per_day'))


def simple_features(df, initial):
    """missing_count and the 9 ratio features (the XGBoost / FLAML notebooks: NaNs kept)."""
    out = df.copy()
    # vectorized form of the notebooks' row-wise string membership count
    out['missing_count'] = out[initial].astype(str).isin(['nan', 'None', '<NA>']).sum(axis=1)
    social, gaming, work, daily, sleep, weekend, opens, notifications = _numbers(out)
    out['total_breakdown_hours'] = social.fillna(0) + gaming.fillna(0) + work.fillna(0)
    out['social_ratio'] = social / (daily + 1e-5)
    out['gaming_ratio'] = gaming / (daily + 1e-5)
    out['work_ratio'] = work / (daily + 1e-5)
    out['unaccounted_screen_time'] = daily - out.total_breakdown_hours
    out['screen_to_sleep_ratio'] = daily / (sleep + 1e-5)
    out['weekend_vs_daily_ratio'] = weekend / (daily + 1e-5)
    out['app_opens_per_hour'] = opens / (daily + 1e-5)
    out['notifications_per_hour'] = notifications / (daily + 1e-5)
    return out


def engineer(df, initial):
    """missing_count (float) and the 9 ratio features with NaN -> 0 (the neural-network / CatBoost notebooks)."""
    out = simple_features(df, initial)
    out['missing_count'] = out['missing_count'].astype(float)
    continuous = ['total_breakdown_hours', 'social_ratio', 'gaming_ratio', 'work_ratio', 'unaccounted_screen_time',
                  'screen_to_sleep_ratio', 'weekend_vs_daily_ratio', 'app_opens_per_hour', 'notifications_per_hour']
    out[continuous] = out[continuous].fillna(0.0)
    return out


def add_fold_encodings(xtr, ytr, xva, xte, columns):
    """In place: <col>_te (smoothing 10; inner 5-fold out-of-fold on the training rows) and <col>_freq (counts on the
    training rows) for every column, te before freq as in the notebooks."""
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_mean = float(ytr.mean())
    for col in columns:
        encoded = np.full(len(xtr), outer_mean, dtype=np.float64)
        for fit, hold in inner.split(xtr, ytr):
            fit_y, fit_key = ytr.iloc[fit], xtr.iloc[fit][col]
            mean = float(fit_y.mean())
            stats = pd.DataFrame({'key': fit_key.astype(str).to_numpy(), 'y': fit_y.to_numpy()}).groupby('key').y.agg(
                ['count', 'mean'])
            mapping = (stats['count'] * stats['mean'] + 10.0 * mean) / (stats['count'] + 10.0)
            encoded[hold] = xtr.iloc[hold][col].astype(str).map(mapping).fillna(mean)
        stats = pd.DataFrame({'key': xtr[col].astype(str).to_numpy(), 'y': ytr.to_numpy()}).groupby('key').y.agg(
            ['count', 'mean'])
        mapping = (stats['count'] * stats['mean'] + 10.0 * outer_mean) / (stats['count'] + 10.0)
        counts = xtr[col].value_counts()
        xtr[f'{col}_te'] = encoded
        xva[f'{col}_te'] = xva[col].astype(str).map(mapping).fillna(outer_mean).astype(float)
        xte[f'{col}_te'] = xte[col].astype(str).map(mapping).fillna(outer_mean).astype(float)
        xtr[f'{col}_freq'] = xtr[col].map(counts).astype(float).fillna(0.0)
        xva[f'{col}_freq'] = xva[col].map(counts).astype(float).fillna(0.0)
        xte[f'{col}_freq'] = xte[col].map(counts).astype(float).fillna(0.0)


def nested_encodings(tr, ytr, va, te, columns, smoothing=10.0):
    """Returns new frames (train, valid, test) with <col>_te and <col>_freq (float32) for every column: inner 5-fold
    out-of-fold target encoding on the training rows, the full training-row map for valid / test."""
    ytr = np.asarray(ytr)
    tr_out, va_out, te_out = pd.DataFrame(index=tr.index), pd.DataFrame(index=va.index), pd.DataFrame(index=te.index)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outer_mean = float(ytr.mean())
    for col in columns:
        tr_key, va_key, te_key = tr[col].astype(str), va[col].astype(str), te[col].astype(str)
        encoded = np.full(len(tr), outer_mean, dtype=np.float32)
        for fit_pos, hold_pos in inner.split(np.zeros(len(tr)), ytr):
            fit_key, fit_y = tr_key.iloc[fit_pos], ytr[fit_pos]
            inner_mean = float(fit_y.mean())
            stats = pd.DataFrame({'key': fit_key.to_numpy(), 'y': fit_y}).groupby('key').y.agg(['count', 'mean'])
            mapping = (stats['count'] * stats['mean'] + smoothing * inner_mean) / (stats['count'] + smoothing)
            encoded[hold_pos] = tr_key.iloc[hold_pos].map(mapping).fillna(inner_mean).to_numpy(dtype=np.float32)
        stats = pd.DataFrame({'key': tr_key.to_numpy(), 'y': ytr}).groupby('key').y.agg(['count', 'mean'])
        mapping = (stats['count'] * stats['mean'] + smoothing * outer_mean) / (stats['count'] + smoothing)
        outer_counts = tr_key.value_counts()
        tr_out[f'{col}_te'] = encoded
        va_out[f'{col}_te'] = va_key.map(mapping).fillna(outer_mean).astype('float32')
        te_out[f'{col}_te'] = te_key.map(mapping).fillna(outer_mean).astype('float32')
        tr_out[f'{col}_freq'] = tr_key.map(outer_counts).fillna(0).astype('float32')
        va_out[f'{col}_freq'] = va_key.map(outer_counts).fillna(0).astype('float32')
        te_out[f'{col}_freq'] = te_key.map(outer_counts).fillna(0).astype('float32')
    return tr_out.reset_index(drop=True), va_out.reset_index(drop=True), te_out.reset_index(drop=True)


def pytabkit_frames():
    """Engineered train / test frames of the pytabkit notebooks, their feature list, the 12 predictors and the
    target."""
    train, test, initial = categorical_frames()
    x, xt = engineer(train, initial), engineer(test, initial)
    features = [c for c in x if c not in [TARGET, ID]]
    y = pd.Series(train[TARGET].to_numpy(np.int8))
    return x, xt, features, initial, y


def pytabkit_fold(x, xt, y, features, initial, train_index, valid_index):
    """One fold of the pytabkit notebooks: fold-local encodings, numeric NaN -> 0."""
    xtr = x.iloc[train_index][features].copy().reset_index(drop=True)
    xva = x.iloc[valid_index][features].copy().reset_index(drop=True)
    xte = xt[features].copy().reset_index(drop=True)
    ytr, yva = y.iloc[train_index].reset_index(drop=True), y.iloc[valid_index].reset_index(drop=True)
    add_fold_encodings(xtr, ytr, xva, xte, initial)
    numeric = xtr.select_dtypes(include=[np.number]).columns
    xtr[numeric] = xtr[numeric].fillna(0.0)
    xva[numeric] = xva[numeric].fillna(0.0)
    xte[numeric] = xte[numeric].fillna(0.0)
    return xtr, ytr, xva, yva, xte


def run_pytabkit(name, make_model, fold_ids):
    """The fold loop of the pytabkit notebooks: model = make_model(), fit(train, y, valid, y_valid) (the validation rows
    drive pytabkit's early stopping / best-epoch selection), probabilities of the validation and test rows."""
    import torch
    from .runner import run_folds, work_dir, working_directory

    x, xt, features, initial, y = pytabkit_frames()

    def fit_fold(fold, train_index, valid_index):
        xtr, ytr, xva, yva, xte = pytabkit_fold(x, xt, y, features, initial, train_index, valid_index)
        with working_directory(work_dir(name)):  # pytabkit may write checkpoints to the working directory
            model = make_model()
            model.fit(xtr, ytr, xva, yva)
            valid_pred, test_pred = model.predict_proba(xva)[:, 1], model.predict_proba(xte)[:, 1]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return valid_pred, test_pred

    return run_folds(name, fit_fold, fold_ids, y=y.to_numpy())
