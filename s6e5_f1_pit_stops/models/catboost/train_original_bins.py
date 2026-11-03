#!/usr/bin/env python
"""GPU CatBoost with 4 continuous numerics and every other feature as a native categorical, incl. original-value bins.

Features (31 columns): lap time, lap-time delta, degradation and race progress stay numeric; the other 10 raw columns
and 9 race-state features (lap minus tyre life, tyre / lap, degradation and delta per tyre lap, lap time / progress,
pit-stop x stint and position x stint codes, tyre-life and lap tens) are label-encoded over train + test + original
and passed as categoricals, together with
- original-value bins of the 4 continuous numerics: the sorted unique values of the original data split into 2048
  (512 for progress) equal-count chunks; a value seen in the original data gets its chunk, any other value the first
  chunk whose upper end is >= the value;
- the first decimal digit of the 4 continuous numerics.
The original rows are used only for the encodings and the bins (they are not appended to the training side).
CatBoost: depth 6, lr 0.03, l2 8, up to 50,000 iterations, early stopping 200 on the validation fold
(use_best_model=True), seed 42. Test = mean of the 5 fold models (probabilities clipped to [1e-7, 1 - 1e-7]).

Usage: python train_original_bins.py [--folds 0,1,2,3,4] [--force]"""
import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_part  # noqa: E402
from models.catboost.lib.runner import FOLDER, finish, fold_parser, todo_folds  # noqa: E402

PAIR = 'catboost_original_value_bins'
SEED = 42

CONT_NUMERIC_COLS = ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress']
NUMERIC_BIN_CARDINALITY = {
    'LapTime (s)': 2048,
    'LapTime_Delta': 2048,
    'Cumulative_Degradation': 2048,
    'RaceProgress': 512,
}
DECIMAL_DIGIT_COLS = ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress']

CB_PARAMS = dict(
    loss_function='Logloss',
    eval_metric='AUC',
    iterations=50000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=8.0,
    random_seed=SEED,
    early_stopping_rounds=200,
    use_best_model=True,
    verbose=500,
    task_type='GPU',
    devices='0',
)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def safe_name(col: str) -> str:
    out = str(col)
    for a, b in [(' ', '_'), ('(', ''), (')', ''), ('/', '_'), ('-', '_'), ('.', '_'), ('%', 'pct')]:
        out = out.replace(a, b)
    return out


def first_decimal_digit(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float32, copy=False)
    out = np.full(len(arr), -1, dtype=np.int32)
    mask = np.isfinite(arr)
    out[mask] = (np.floor(np.abs(arr[mask]) * 10.0 + 1e-6).astype(np.int64) % 10).astype(np.int32)
    return pd.Series(out, index=s.index, dtype='int32')


def make_apply_bins_fn(value_to_bin, edges, col_name):
    ends = np.array([b for a, b in edges], dtype=np.float32)

    def apply_bins(s):
        arr = pd.to_numeric(s, errors='coerce').to_numpy(dtype=np.float32, copy=False)
        out = np.full(len(arr), -1, dtype=np.int32)
        finite = np.isfinite(arr)
        for i, v in enumerate(arr):
            if not finite[i]:
                continue
            key = float(v)
            if key in value_to_bin:
                out[i] = value_to_bin[key]
            else:
                b = int(np.searchsorted(ends, v, side='left'))
                if b < 0:
                    b = 0
                elif b >= len(ends):
                    b = len(ends) - 1
                if b >= 0 and len(ends) > 0:
                    out[i] = b
        return pd.Series(out, index=s.index, name=f'{safe_name(col_name)}_bin', dtype='int32')
    return apply_bins


def add_f1_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lap = pd.to_numeric(df.get('LapNumber', 0), errors='coerce').fillna(0).astype('float32')
    tyre = pd.to_numeric(df.get('TyreLife', 0), errors='coerce').fillna(0).astype('float32')
    stint = pd.to_numeric(df.get('Stint', 0), errors='coerce').fillna(0).astype('float32')
    pit = pd.to_numeric(df.get('PitStop', 0), errors='coerce').fillna(0).astype('float32')
    pos = pd.to_numeric(df.get('Position', 0), errors='coerce').fillna(0).astype('float32')
    prog = pd.to_numeric(df.get('RaceProgress', 0), errors='coerce').fillna(0).astype('float32')
    lap_time = pd.to_numeric(df.get('LapTime (s)', 0), errors='coerce').fillna(0).astype('float32')
    lap_delta = pd.to_numeric(df.get('LapTime_Delta', 0), errors='coerce').fillna(0).astype('float32')
    deg = pd.to_numeric(df.get('Cumulative_Degradation', 0), errors='coerce').fillna(0).astype('float32')

    df['state_lap_minus_tyre'] = (lap - tyre).astype('float32')
    df['state_tyre_per_lap'] = (tyre / (lap + 1.0)).astype('float32')
    df['state_degradation_per_tyre'] = (deg / (tyre + 1.0)).astype('float32')
    df['state_delta_per_tyre'] = (lap_delta / (tyre + 1.0)).astype('float32')
    df['state_laptime_per_progress'] = (lap_time / (prog + 1e-3)).astype('float32')
    df['state_pit_stint'] = (pit * 10 + stint).round().astype('int32')
    df['state_pos_stint'] = (pos * 10 + stint).round().astype('int32')
    df['state_tyre_bin10'] = np.floor(tyre / 10).astype('int32')
    df['state_lap_bin10'] = np.floor(lap / 10).astype('int32')
    return df


def main():
    from catboost import CatBoostClassifier

    args = fold_parser().parse_args()
    todo = todo_folds(PAIR, args.folds, args.force)
    if todo:
        seed_everything(SEED)
        train_df, test_df, orig_df = read_train(), read_test(), read_original()
        keep_orig = [c for c in train_df.columns if c != ID and c in orig_df.columns]
        orig_df = orig_df[keep_orig].copy()

        X = train_df.drop([ID, TARGET], axis=1, errors='ignore').copy()
        y = train_df[TARGET].astype(np.int8).values
        X_test = test_df.drop([ID], axis=1, errors='ignore').copy()
        X_orig = orig_df.drop([TARGET], axis=1, errors='ignore').copy()

        for df in (X, X_test, X_orig):
            for c in CONT_NUMERIC_COLS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')

        X = add_f1_state_features(X)
        X_test = add_f1_state_features(X_test)
        X_orig = add_f1_state_features(X_orig)

        feature_cols = X.columns.tolist()
        cont_numeric_cols = [c for c in CONT_NUMERIC_COLS if c in feature_cols]
        base_cat_cols = [c for c in feature_cols if c not in cont_numeric_cols]

        # label-encode the categorical-like columns over train + test + original
        for col in base_cat_cols:
            combined = pd.concat([X[col], X_test[col], X_orig[col]], axis=0, ignore_index=True).astype(str).fillna('__MISSING__')
            uniques = pd.Index(combined.unique()).sort_values()
            mapping = {v: i for i, v in enumerate(uniques)}
            for df in (X, X_test, X_orig):
                df[col] = df[col].astype(str).fillna('__MISSING__').map(mapping).fillna(-1).astype('int32')

        # bins of the original data's unique values, with an interval fallback for values not in the original data
        for col in cont_numeric_cols:
            desired_card = NUMERIC_BIN_CARDINALITY.get(col, 512)
            uniq = np.sort(pd.to_numeric(X_orig[col], errors='coerce').dropna().unique().astype(np.float32))
            actual_card = int(min(desired_card, max(1, len(uniq))))
            chunks = np.array_split(uniq, actual_card)
            value_to_bin = {}
            edges = []
            for bin_id, arr in enumerate(chunks):
                if len(arr) == 0:
                    continue
                for v in arr:
                    value_to_bin[float(v)] = int(bin_id)
                edges.append((float(arr[0]), float(arr[-1])))
            apply_bins = make_apply_bins_fn(value_to_bin, edges, col)
            bin_col = f'{safe_name(col)}_bin'
            for df in (X, X_test, X_orig):
                df[bin_col] = apply_bins(df[col]).astype('int32')

        for col in DECIMAL_DIGIT_COLS:
            if col in X.columns:
                dcol = f'{safe_name(col)}_d1'
                for df in (X, X_test, X_orig):
                    df[dcol] = first_decimal_digit(df[col]).astype('int32')

        features = X.columns.tolist()
        cat_feature_names = [c for c in features if c not in cont_numeric_cols]
        for c in cat_feature_names:
            X[c] = X[c].astype('int32')
            X_test[c] = X_test[c].astype('int32')
        for c in cont_numeric_cols:
            X[c] = X[c].astype('float32')
            X_test[c] = X_test[c].astype('float32')
        print(f'features: {len(features)}; categorical: {len(cat_feature_names)}; continuous: {len(cont_numeric_cols)}',
              flush=True)

        fold_id = folds(y)
        for k in todo:
            tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            X_tr = X.iloc[tr_idx].copy()
            y_tr = y[tr_idx]
            X_va = X.iloc[va_idx].copy()
            y_va = y[va_idx]
            model = CatBoostClassifier(**CB_PARAMS)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                cat_features=cat_feature_names,
                verbose=CB_PARAMS.get('verbose', 500),
            )
            p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
            p_te = model.predict_proba(X_test)[:, 1].astype(np.float32)
            save_part(FOLDER, PAIR, k, p_va, p_te)
            print(f'{PAIR} fold {k}: AUC {roc_auc_score(y_va, p_va):.6f} best_iter {model.get_best_iteration()}', flush=True)
            del X_tr, X_va, y_tr, y_va, model, p_va, p_te
            gc.collect()
    finish(PAIR, clip=1e-7)


if __name__ == '__main__':
    main()
