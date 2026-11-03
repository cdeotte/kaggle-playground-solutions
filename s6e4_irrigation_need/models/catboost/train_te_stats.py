#!/usr/bin/env python
"""Class-weighted CatBoost on frequencies, original priors and target statistics
(pair catboost_freq_priors_te_stats_balanced_a).

Label-free features (column names lower-cased):
  - the 11 numerics and 8 categoricals; (previous irrigation + rainfall) / (area + 1), soil moisture x temperature
  - frequency of every numeric value over the train + original + test rows
  - rainfall, previous irrigation and soil moisture rounded to 3 decimals, as strings
  - original prior of the 8 categoricals and 11 numerics: the expected class index (0 Low, 1 Medium, 2 High) of the
    original rows with that value
Per fold, for the 8 categoricals and the 3 numeric strings (the class index used as a continuous target):
  - std / min / max of the class index per value, out-of-fold over an inner StratifiedKFold(10, seed 42) of the
    training rows, from all training rows for the validation / test rows
  - scikit-learn TargetEncoder (continuous, cv 10, shuffle, seed 42, smooth 'auto') mean encoding
The 11 encoded string columns are then dropped (no native categories are left).
Model: CatBoost GPU, MultiClass, depth 6, lr 0.05, l2 3, min_data_in_leaf 20, auto_class_weights Balanced, up to 5000
iterations with early stopping (150) on the validation fold, seed 42. Test = mean of the 5 fold models.
Usage: python train_te_stats.py"""
import gc
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "catboost_freq_priors_te_stats_balanced_a"
SEED = 42
N_SPLITS = 5
INNER_FOLDS = 10
VERBOSE_EVAL = 500

BASE_CATS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
BASE_NUMS = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]

CAT_PARAMS = dict(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    iterations=5000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    min_data_in_leaf=20,
    random_seed=SEED,
    early_stopping_rounds=150,
    verbose=VERBOSE_EVAL,
    auto_class_weights="Balanced",
    task_type="GPU",
)


def log(msg: str):
    print(msg, flush=True)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"[^\w\s]", "", c.strip().lower()).replace(" ", "_") for c in out.columns]
    return out


def to_expected_value_map(df: pd.DataFrame, col: str) -> dict:
    grp = df.groupby(col, dropna=False)["_target_int"].value_counts(normalize=True).unstack(fill_value=0.0)
    grp = grp.reindex(columns=range(3), fill_value=0.0)
    expected = grp.to_numpy() @ np.arange(3, dtype=np.float32)
    return dict(zip(grp.index.tolist(), expected.astype(np.float32)))


def load_data():
    train = normalize_cols(read_train())
    test = normalize_cols(read_test())
    orig = normalize_cols(read_original())

    target = TARGET.lower()
    train[target] = train[target].map(TARGET_MAP).astype("int8")
    orig[target] = orig[target].map(TARGET_MAP).astype("int8")

    if ID in train.columns:
        train = train.drop(columns=[ID])
    if ID in test.columns:
        test = test.drop(columns=[ID])

    for df in (train, test, orig):
        for c in BASE_CATS:
            c2 = c.lower()
            df[c2] = df[c2].astype(str).fillna("missing").str.strip()
        for c in BASE_NUMS:
            c2 = c.lower()
            df[c2] = pd.to_numeric(df[c2], errors="coerce").astype("float32")

    return train, test, orig, target


def build_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, target: str):
    cats = [c.lower() for c in BASE_CATS]
    nums = [c.lower() for c in BASE_NUMS]

    new_nums = []
    num_as_cat = []

    for df in (train, test, orig):
        df["avg_water_per_area"] = (df["previous_irrigation_mm"] + df["rainfall_mm"]) / (df["field_area_hectare"] + 1.0)
        df["moisture_x_temp"] = df["soil_moisture"] * df["temperature_c"]
    new_nums += ["avg_water_per_area", "moisture_x_temp"]

    for col in nums:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True, dropna=False)
        new_col = f"freq_{col}"
        for df in (train, test, orig):
            df[new_col] = pd.to_numeric(df[col].map(freq), errors="coerce").fillna(0).astype("float32")
        new_nums.append(new_col)

    for col in ["rainfall_mm", "previous_irrigation_mm", "soil_moisture"]:
        new_col = f"cat_{col}"
        for df in (train, test, orig):
            df[new_col] = df[col].round(3).astype(str)
        num_as_cat.append(new_col)

    orig_tmp = orig.copy()
    orig_tmp["_target_int"] = orig_tmp[target].astype("int8")
    prior = np.bincount(orig_tmp["_target_int"], minlength=3) / len(orig_tmp)
    prior_ev = float(prior @ np.arange(3, dtype=np.float32))
    for col in cats + nums:
        mapping = to_expected_value_map(orig_tmp, col)
        new_col = f"orig_proba_{col}"
        train[new_col] = pd.to_numeric(train[col].map(mapping), errors="coerce").fillna(prior_ev).astype("float32")
        test[new_col] = pd.to_numeric(test[col].map(mapping), errors="coerce").fillna(prior_ev).astype("float32")
        new_nums.append(new_col)

    features = nums + cats + new_nums + num_as_cat
    te_columns = cats + num_as_cat
    drop_cols = te_columns.copy()
    return train, test, features, cats, te_columns, drop_cols


def add_nested_te(x_tr, y_tr, x_val, x_te, features, te_columns, cats, drop_cols, seed):
    x_tr = x_tr.copy()
    x_val = x_val.copy()
    x_te = x_te.copy()

    y_ev = y_tr.astype(np.float32)
    x_tr["_target_ev"] = y_ev
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for in_tr, in_va in inner.split(x_tr, y_tr):
        tr2 = x_tr.loc[in_tr, features + ["_target_ev"]].copy()
        va2 = x_tr.loc[in_va, features].copy()
        for col in te_columns:
            agg = tr2.groupby(col, dropna=False)["_target_ev"].agg(["std", "min", "max"])
            agg.columns = [f"te1_{col}_{s}" for s in ["std", "min", "max"]]
            va2 = va2.merge(agg, on=col, how="left")
            for c in agg.columns:
                x_tr.loc[in_va, c] = pd.to_numeric(va2[c], errors="coerce").fillna(0).astype("float32").values

    for col in te_columns:
        agg = x_tr.groupby(col, dropna=False)["_target_ev"].agg(["std", "min", "max"])
        agg.columns = [f"te1_{col}_{s}" for s in ["std", "min", "max"]]
        x_val = x_val.merge(agg, on=col, how="left")
        x_te = x_te.merge(agg, on=col, how="left")
        for c in agg.columns:
            x_tr[c] = pd.to_numeric(x_tr[c], errors="coerce").fillna(0).astype("float32")
            x_val[c] = pd.to_numeric(x_val[c], errors="coerce").fillna(0).astype("float32")
            x_te[c] = pd.to_numeric(x_te[c], errors="coerce").fillna(0).astype("float32")

    te = TargetEncoder(
        cv=INNER_FOLDS,
        shuffle=True,
        smooth="auto",
        target_type="continuous",
        random_state=seed,
    )
    mean_cols = [f"te_{c}" for c in te_columns]
    x_tr[mean_cols] = te.fit_transform(x_tr[te_columns], y_ev).astype("float32")
    x_val[mean_cols] = te.transform(x_val[te_columns]).astype("float32")
    x_te[mean_cols] = te.transform(x_te[te_columns]).astype("float32")

    cat_cols_present = [c for c in cats if c in x_tr.columns]
    for df_ in (x_tr, x_val, x_te):
        if cat_cols_present:
            df_[cat_cols_present] = df_[cat_cols_present].astype(str)
        df_.drop(columns=[c for c in drop_cols if c in df_.columns], inplace=True)

    x_tr.drop(columns=["_target_ev"], inplace=True)
    cols = list(x_tr.columns)
    x_val = x_val[cols].copy()
    x_te = x_te[cols].copy()
    return x_tr, x_val, x_te, cols


def main():
    train, test, orig, target = load_data()
    train, test, features, cats, te_columns, drop_cols = build_features(train, test, orig, target)

    y = train[target].to_numpy()
    fold_ids = folds(y)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    pred = np.zeros((len(test), 3), dtype=np.float32)
    for k in range(N_SPLITS):
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        t0 = time.time()
        x_tr = train.loc[tr_idx, features].reset_index(drop=True).copy()
        y_tr = y[tr_idx].copy()
        x_val = train.loc[va_idx, features].reset_index(drop=True).copy()
        y_val = y[va_idx].copy()
        x_te = test[features].reset_index(drop=True).copy()

        x_tr, x_val, x_te, cols = add_nested_te(
            x_tr=x_tr,
            y_tr=y_tr,
            x_val=x_val,
            x_te=x_te,
            features=features,
            te_columns=te_columns,
            cats=cats,
            drop_cols=drop_cols,
            seed=SEED,
        )

        cat_features = [i for i, c in enumerate(cols) if x_tr[c].dtype == object]
        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(
            x_tr,
            y_tr,
            cat_features=cat_features,
            eval_set=(x_val, y_val),
            verbose=VERBOSE_EVAL,
        )

        val_pred = model.predict_proba(x_val)
        test_pred = model.predict_proba(x_te[cols])
        oof[va_idx] = val_pred.astype(np.float32)
        pred += test_pred.astype(np.float32) / N_SPLITS

        bac = balanced_accuracy_score(y_val, val_pred.argmax(axis=1))
        log(f"fold {k} bac={bac:.6f} time={time.time() - t0:.1f}s")

        del x_tr, x_val, x_te, y_tr, y_val, model
        gc.collect()

    log(f"{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, oof.argmax(axis=1)):.6f}")
    save_pair(HERE, PAIR, oof, pred)


if __name__ == "__main__":
    main()
