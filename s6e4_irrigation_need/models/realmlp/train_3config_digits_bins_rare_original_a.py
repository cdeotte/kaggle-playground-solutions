#!/usr/bin/env python
"""RealMLP, three tuned configurations averaged, on digit / bin / rounded-category / rare-flag features, original rows
appended (pair realmlp_3config_digits_bins_rare_original_a).

Training rows of fold k: the competition rows of the 4 other folds followed by all 10,000 original rows.
Features (the 19 raw columns plus, for each of the 11 numerics):
- a NaN flag, the base-10 digits of the value rounded to its step (ROUND_MAP; x10 when the step is < 1), a sign flag;
- quantile bins (4, 5, 10, 20), equal-width bins (5, 10, 20) and the value rounded to its step (integer codes);
- every raw column as a string category (numerics rounded to their step);
- a rare flag (< 25 rows) of each rounded numeric category.
Bins, digit counts and rare counts are computed over the rows they are built on: the training rows' features over the
fold's training + original + test rows; the validation and test rows' features in a second pass over the training +
original + validation + test rows.
Model: pytabkit RealMLP_TD_Classifier in three tuned configurations (mish, PBLD numeric embeddings, n_ens 8, up to 32
epochs, batch 1024, random_state 42; the first two with early stopping), fitted with the validation fold as validation
set; the three probabilities are averaged. Test = mean of the 5 fold models.
Usage: python train_3config_digits_bins_rare_original_a.py"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import (CLASSES, ID, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_original,  # noqa: E402
                         read_test, read_train)
from common.pairs import save_pair  # noqa: E402

warnings.simplefilter("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_3config_digits_bins_rare_original_a"
LABELS = list(CLASSES)

BASE_FEATS = [
    "Soil_Type", "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
    "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type", "Water_Source",
    "Field_Area_hectare", "Mulching_Used", "Previous_Irrigation_mm", "Region",
]

NUM_DIGIT_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]

ROUND_MAP = {
    "Soil_pH": 0.1,
    "Soil_Moisture": 1.0,
    "Organic_Carbon": 0.1,
    "Electrical_Conductivity": 0.1,
    "Temperature_C": 1.0,
    "Humidity": 1.0,
    "Rainfall_mm": 1.0,
    "Sunlight_Hours": 0.5,
    "Wind_Speed_kmh": 0.5,
    "Field_Area_hectare": 0.5,
    "Previous_Irrigation_mm": 1.0,
}

RARE_MIN_COUNT = 25

common_fixed_params = {
    "act": "mish",
    "ls_eps_sched": "coslog4",
    "p_drop_sched": "flat_cos",
    "wd_sched": "flat_cos",
    "embedding_size": 8,
    "bias_wd_factor": 0.0,
    "weight_param": "ntk",
    "bias_lr_factor": 0.1,
    "use_parametric_act": True,
    "add_front_scale": True,
    "act_lr_factor": 0.1,
    "block_str": "w-b-a-d",
    "max_one_hot_cat_size": 9,
    "hidden_sizes": "rectangular",
    "bias_init_mode": "he+5",
    "weight_init_mode": "std",
    "val_metric_name": "class_error",
    "num_emb_type": "pbld",
    "verbosity": 2,
    "batch_size": 1024,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_state": 42,
    "n_epochs": 32,
    "early_stopping_additive_patience": 40,
    "early_stopping_multiplicative_patience": 3,
    "n_ens": 8,
    "ens_av_before_softmax": False,
}

top3_params = [
    {
        **common_fixed_params,
        "n_hidden_layers": 4, "hidden_width": 512,
        "embedding_size": 4,
        "lr": 0.1301052035689966,
        "wd": 0.04914969178803988,
        "p_drop": 0.02682248660489578,
        "scale_lr_factor": 2.1896500647074553,
        "use_ls": False,
        "ls_eps": 0.04088400565558735,
        "plr_sigma": 1.668950443527603,
        "plr_lr_factor": 0.27098419375438565,
        "plr_hidden_1": 8, "plr_hidden_2": 8,
        "sq_mom": 0.9855686567493073,
        "first_layer_lr_factor": 1.3750227365258307,
        "use_early_stopping": True,
        "max_one_hot_cat_size": 15,
    },
    {
        **common_fixed_params,
        "n_hidden_layers": 2, "hidden_width": 512,
        "embedding_size": 8,
        "lr": 0.03865822397961398,
        "wd": 0.00851274626094165,
        "p_drop": 0.2739396730231609,
        "scale_lr_factor": 3.6178064047128555,
        "use_ls": False,
        "ls_eps": 0.012194703961296842,
        "plr_sigma": 1.3478060390217743,
        "plr_lr_factor": 0.10119556825550731,
        "plr_hidden_1": 8, "plr_hidden_2": 8,
        "sq_mom": 0.9865749129374782,
        "first_layer_lr_factor": 0.6175393224001581,
        "use_early_stopping": True,
        "max_one_hot_cat_size": 8,
    },
    {
        **common_fixed_params,
        "n_hidden_layers": 2, "hidden_width": 384,
        "embedding_size": 16,
        "lr": 0.029186039688349992,
        "wd": 0.009702038034899564,
        "p_drop": 0.028921980435491934,
        "scale_lr_factor": 5.433094297291475,
        "use_ls": False,
        "ls_eps": 0.035742411185025844,
        "plr_sigma": 48.89920447537555,
        "plr_lr_factor": 0.21112939343489906,
        "plr_hidden_1": 16, "plr_hidden_2": 4,
        "sq_mom": 0.9947345074902412,
        "first_layer_lr_factor": 1.0025819443139303,
        "use_early_stopping": False,
        "max_one_hot_cat_size": 4,
    },
]


def encode_target(s):
    y = s.map(TARGET_MAP)
    if y.isna().any():
        raise ValueError("Unexpected target labels")
    return y.astype(np.int32)


def round_to_step(x, step):
    x = pd.to_numeric(x, errors="coerce")
    if step == 0:
        return x
    return (x / step).round() * step


def add_features(combined_df):
    """Digit, bin, rounded-category and rare-flag features, computed over all rows of combined_df (modified in place).
    Returns the frame, the feature list and the categorical columns (string columns, cast to category)."""
    digit_cols = []
    for col in NUM_DIGIT_COLS:
        nan_flag = f"{col}__is_nan"
        combined_df[nan_flag] = pd.to_numeric(combined_df[col], errors="coerce").isna().astype(np.int8)
        digit_cols.append(nan_flag)

        s = pd.to_numeric(combined_df[col], errors="coerce").fillna(0.0)
        step = ROUND_MAP.get(col, 1.0)
        s_rounded = round_to_step(s, step)
        scale = 10 if step < 1.0 else 1
        s_scaled = (s_rounded * scale).round().astype(np.int64)
        max_val = int(np.abs(s_scaled).max()) if len(s_scaled) else 0
        max_int_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1
        for i in range(max_int_digits):
            new_col = f"{col}_digit_pos_{i+1}"
            combined_df[new_col] = ((np.abs(s_scaled) // (10 ** i)) % 10).astype(np.int8)
            digit_cols.append(new_col)
        sign_col = f"{col}__neg"
        combined_df[sign_col] = (s_scaled < 0).astype(np.int8)
        digit_cols.append(sign_col)

    binning_cols = []
    for col in NUM_DIGIT_COLS:
        x = pd.to_numeric(combined_df[col], errors="coerce")
        for q in [4, 5, 10, 20]:
            new_col = f"{col}_bin_q{q}"
            try:
                combined_df[new_col] = pd.qcut(x, q=q, labels=False, duplicates="drop")
                binning_cols.append(new_col)
            except Exception:
                pass
        for b in [5, 10, 20]:
            new_col = f"{col}_bin_cut{b}"
            combined_df[new_col] = pd.cut(x, bins=b, labels=False)
            binning_cols.append(new_col)
        step = ROUND_MAP.get(col, 1.0)
        new_col = f"{col}_bin_round_{str(step).replace('.', 'p')}"
        combined_df[new_col] = round_to_step(x, step=step)
        binning_cols.append(new_col)

    for col in binning_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce").fillna(-1).astype(np.int32)

    all_cats = []
    for c in BASE_FEATS:
        nc = f"{c}_cat"
        if c in NUM_DIGIT_COLS:
            step = ROUND_MAP.get(c, 1.0)
            combined_df[nc] = round_to_step(combined_df[c], step=step).astype("float32")
            combined_df[nc] = combined_df[nc].fillna(-9999).astype(str)
        else:
            combined_df[nc] = combined_df[c].astype(str)
        all_cats.append(nc)

    rare_flags = []
    for c in NUM_DIGIT_COLS:
        rounded_str = combined_df[f"{c}_cat"]
        vc = rounded_str.value_counts(dropna=False)
        rare_mask = rounded_str.map(vc).fillna(0) < RARE_MIN_COUNT
        rf = f"{c}_cat__is_rare"
        combined_df[rf] = rare_mask.astype(np.int8)
        rare_flags.append(rf)

    features = BASE_FEATS + digit_cols + binning_cols + all_cats + rare_flags
    cat_cols = [c for c in combined_df.columns if combined_df[c].dtype == "object"]
    for col in cat_cols:
        combined_df[col] = combined_df[col].astype(str).astype("category")
    return combined_df, features, cat_cols


def main():
    train = read_train()
    test = read_test()
    orig = read_original()

    if ID in train.columns:
        train = train.drop(columns=[ID])
    if ID in test.columns:
        test = test.drop(columns=[ID])
    if ID in orig.columns:
        orig = orig.drop(columns=[ID])

    train[TARGET] = encode_target(train[TARGET])
    orig[TARGET] = encode_target(orig[TARGET])
    y = train[TARGET].values.astype(np.int32)

    for df in (train, test, orig):
        for c in NUM_DIGIT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    pred = np.zeros((len(test), len(LABELS)), dtype=np.float32)
    fold_id = folds(y)
    test_base = test[BASE_FEATS].copy().reset_index(drop=True)

    print("Starting ensemble training (3 configurations averaged per fold)...")
    for fold in range(1, N_FOLDS + 1):
        tr_idx, va_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        train_plus_orig = pd.concat(
            [train.iloc[tr_idx][BASE_FEATS + [TARGET]].copy().reset_index(drop=True),
             orig[BASE_FEATS + [TARGET]].copy().reset_index(drop=True)],
            axis=0, ignore_index=True,
        )
        n_tr = len(train_plus_orig)
        y_tr = train_plus_orig[TARGET].values.astype(np.int32)

        # training rows: features computed over the fold's training + original + test rows
        train_fe, features, cat_cols = add_features(
            pd.concat([train_plus_orig[BASE_FEATS], test_base.copy()], axis=0, ignore_index=True))
        X_tr = train_fe.iloc[:n_tr].reset_index(drop=True)[features].copy()

        # validation and test rows: the same features rebuilt with the validation rows included
        valid_base = train.iloc[va_idx][BASE_FEATS].copy().reset_index(drop=True)
        combo, val_features, _ = add_features(
            pd.concat([train_plus_orig[BASE_FEATS], valid_base, test_base.copy()], axis=0, ignore_index=True))
        X_va = combo.iloc[n_tr: n_tr + len(va_idx)][val_features].reset_index(drop=True)
        X_te = combo.iloc[n_tr + len(va_idx):][val_features].reset_index(drop=True)
        y_va = y[va_idx].copy()

        fold_ens_va = np.zeros((len(va_idx), len(LABELS)), dtype=np.float32)
        fold_ens_te = np.zeros((len(X_te), len(LABELS)), dtype=np.float32)
        for model_idx, params in enumerate(top3_params, 1):
            model = RealMLP_TD_Classifier(**params)
            model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, cat_col_names=cat_cols)

            va_pred = model.predict_proba(X_va).astype(np.float32)
            fold_ens_va += va_pred / len(top3_params)
            te_pred = model.predict_proba(X_te).astype(np.float32)
            fold_ens_te += te_pred / len(top3_params)

            model_bac = balanced_accuracy_score(y_va, va_pred.argmax(axis=1))
            print(f"Fold {fold} | #{model_idx} | BAC {model_bac:.8f}")

            del model, va_pred, te_pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        oof[va_idx] = fold_ens_va
        pred += fold_ens_te / N_FOLDS

        fold_bac = balanced_accuracy_score(y_va, fold_ens_va.argmax(axis=1))
        print(f"Fold {fold} | Ens. | BAC {fold_bac:.8f}")

        del train_fe, X_tr, X_va, X_te, y_tr, y_va, fold_ens_va, fold_ens_te, combo, train_plus_orig, valid_base
        gc.collect()

    overall_bac = balanced_accuracy_score(y, oof.argmax(axis=1))
    print(f"\nOverall OOF BAC: {overall_bac:.8f}")
    save_pair(model_dir(FOLDER), PAIR, oof, pred)


if __name__ == "__main__":
    main()
