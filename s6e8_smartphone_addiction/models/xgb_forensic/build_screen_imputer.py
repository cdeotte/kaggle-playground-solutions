#!/usr/bin/env python
"""Label-free LightGBM imputations of the 5 screen fields, one model set per outer fold.

Input matrix (train and test rows): the 9 numerics, their missing flags, one-hot levels (original levels) and a
missing flag per categorical. For each screen field and outer fold: fit on up to 480,000 random outer-training rows
where the field is observed (all other inputs, the field's own value and flag removed), two LightGBM regressors
(L1 and L2 objective; 700 trees, lr 0.025, 63 leaves, min_child_samples 100, max_bin 511, subsample 0.9,
colsample 0.95, lambda 15), predictions clipped to the field's outer-training 0.1% / 99.9% quantiles. Columns per field
and objective: imputed value (observed value, else the prediction) and value-if-missing (prediction, 0 if observed).
Writes cache/screen_imputer_v2/fold_{k}_train.npy (all train rows), fold_{k}_test.npy and manifest.json.
Usage: python build_screen_imputer.py [--folds 0,1,2,3,4]"""
import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import TARGET, folds as fold_ids, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from models.xgb_forensic.lib.external import xgb_cache  # noqa: E402

NUMERIC = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
           "notifications_per_day", "app_opens_per_day", "weekend_screen_time"]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
IMPUTED = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "weekend_screen_time"]


def matrix_pair(train, test, source):
    outputs, names = [], []
    for frame in (train, test):
        numeric = frame[NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        parts = [numeric, np.isnan(numeric).astype(np.float32)]
        if not names:
            names += NUMERIC + [f"missing__{column}" for column in NUMERIC]
        for column in CATEGORICAL:
            levels = sorted(source[column].astype(str).unique())
            values = frame[column].astype("string").fillna("<NA>").astype(str).to_numpy()
            parts.append(np.column_stack([(values == level) for level in levels]).astype(np.float32))
            parts.append((values == "<NA>").astype(np.float32)[:, None])
            if not outputs:
                names += [f"{column}__{level}" for level in levels] + [f"missing__{column}"]
        outputs.append(np.column_stack(parts).astype(np.float32))
    return outputs[0], outputs[1], names


def model(objective, seed):
    return lgb.LGBMRegressor(objective=objective, n_estimators=700, learning_rate=0.025, num_leaves=63,
                             min_child_samples=100, max_bin=511, subsample=0.9, subsample_freq=1, colsample_bytree=0.95,
                             reg_lambda=15.0, random_state=seed, deterministic=True, force_col_wise=True, verbosity=-1,
                             n_jobs=8)


def one_feature(feature, feature_index, fold, train, test, train_matrix, test_matrix, schema, outer_train):
    keep = np.asarray([name not in {feature, f"missing__{feature}"} for name in schema], dtype=bool)
    observed_global = outer_train[train.iloc[outer_train][feature].notna().to_numpy()]
    rng = np.random.default_rng(8100 + 100 * fold + feature_index)
    fit = np.sort(rng.choice(observed_global, size=min(480_000, len(observed_global)), replace=False))
    truth = train[feature].to_numpy(np.float64)
    raw_train = truth.copy()
    raw_test = pd.to_numeric(test[feature], errors="coerce").to_numpy(np.float64)
    train_parts, test_parts, names = [], [], []
    for objective_index, (token, objective) in enumerate((("l1", "regression_l1"), ("l2", "regression_l2"))):
        estimator = model(objective, 8200 + 1000 * objective_index + 100 * fold + feature_index)
        estimator.fit(train_matrix[fit][:, keep], truth[fit])
        prediction_train = estimator.predict(train_matrix[:, keep]).astype(np.float64)
        prediction_test = estimator.predict(test_matrix[:, keep]).astype(np.float64)
        lower = float(train.iloc[outer_train][feature].quantile(0.001))
        upper = float(train.iloc[outer_train][feature].quantile(0.999))
        np.clip(prediction_train, lower, upper, out=prediction_train)
        np.clip(prediction_test, lower, upper, out=prediction_test)
        missing_train, missing_test = ~np.isfinite(raw_train), ~np.isfinite(raw_test)
        filled_train, filled_test = raw_train.copy(), raw_test.copy()
        filled_train[missing_train] = prediction_train[missing_train]
        filled_test[missing_test] = prediction_test[missing_test]
        train_parts += [filled_train[:, None], np.where(missing_train, prediction_train, 0.0)[:, None]]
        test_parts += [filled_test[:, None], np.where(missing_test, prediction_test, 0.0)[:, None]]
        names += [f"imputer_v2_{token}_imputed__{feature}", f"imputer_v2_{token}_if_missing__{feature}"]
    return feature_index, train_parts, test_parts, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    train, test, source = read_train(), read_test(), read_original()
    outer = fold_ids(labels(train[TARGET]))
    predictors = train.drop(columns=[TARGET])  # the imputers never see the target
    train_matrix, test_matrix, schema = matrix_pair(predictors, test, source)
    out = xgb_cache() / "screen_imputer_v2"
    out.mkdir(parents=True, exist_ok=True)
    final_names = None
    for fold in fold_list(args.folds):
        outer_train = np.flatnonzero(outer != fold)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(one_feature, feature, index, fold, predictors, test, train_matrix, test_matrix,
                                   schema, outer_train) for index, feature in enumerate(IMPUTED)]
            results = sorted((future.result() for future in futures), key=lambda row: row[0])
        train_parts, test_parts, names = [], [], []
        for _, current_train, current_test, current_names in results:
            train_parts += current_train
            test_parts += current_test
            names += current_names
        final_names = names
        np.save(out / f"fold_{fold}_train.npy", np.column_stack(train_parts).astype(np.float32))
        np.save(out / f"fold_{fold}_test.npy", np.column_stack(test_parts).astype(np.float32))
        print(f"screen imputer fold {fold}: {len(names)} columns", flush=True)
    (out / "manifest.json").write_text(json.dumps({"feature_names": final_names}, indent=1) + "\n")


if __name__ == "__main__":
    main()
