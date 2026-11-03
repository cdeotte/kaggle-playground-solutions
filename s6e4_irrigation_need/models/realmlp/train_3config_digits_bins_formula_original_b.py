#!/usr/bin/env python
"""RealMLP, three tuned configurations averaged, on formula / digit / bin / rounded-category / rare-flag features,
original rows appended (pair realmlp_3config_digits_bins_formula_original_b).

The 10,000 original rows are appended to the 630,000 competition rows and the folds are a
StratifiedKFold(5, shuffle=True, random_state=42) over these 640,000 rows: a fold trains on the rows (competition and
original) of the 4 other folds and is validated on its own competition rows; the OOF covers the competition rows.
Features, computed once over the competition + original + test rows (the 19 raw columns plus):
- formula score (Soil_Moisture < 25, Rainfall_mm < 300, Temperature_C > 30, Wind_Speed_kmh > 10, growth stage
  Harvest / Sowing, mulching) and the flags score <= 0 and score > 3;
- for each of the 11 numerics: a NaN flag, the base-10 digits of the value rounded to its step (ROUND_MAP; x10 when the
  step is < 1) and a sign flag; quantile bins (4, 5, 10, 20), equal-width bins (5, 10, 20) and the value rounded to its
  step (integer codes);
- every raw column as a string category (numerics rounded to their step);
- a rare flag (< 25 rows) of the rounded Soil_Moisture, Humidity, Previous_Irrigation_mm and Rainfall_mm categories.
Model: pytabkit RealMLP_TD_Classifier in three tuned configurations (mish, PBLD numeric embeddings, n_ens 8, up to 32
epochs, batch 1024, random_state 42; the first two with early stopping), fitted with the fold's validation rows as
validation set; the three probabilities are averaged. Test = mean of the 5 fold models.
Usage: python train_3config_digits_bins_formula_original_b.py"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.simplefilter("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_3config_digits_bins_formula_original_b"
N_SPLITS = 5
SEED = 42
N_CLASSES = 3
USE_ORIG = True

NUMS = [
    "Soil_pH", "Organic_Carbon", "Electrical_Conductivity", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Temperature_C",
    "Soil_Moisture", "Humidity", "Previous_Irrigation_mm", "Rainfall_mm",
]
CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
FEATS = NUMS + CATS

# rounding steps for the numeric -> category strings
ROUND_MAP = {
    "Soil_pH": 0.1, "Organic_Carbon": 0.1, "Electrical_Conductivity": 0.1,
    "Sunlight_Hours": 0.1, "Wind_Speed_kmh": 0.5, "Field_Area_hectare": 0.5,
    "Temperature_C": 0.5, "Soil_Moisture": 1.0, "Humidity": 1.0,
    "Previous_Irrigation_mm": 1.0, "Rainfall_mm": 5.0,
}
RARE_MIN_COUNT = 25
FORMULA_FEATS = ["formula_score", "is_low", "is_high"]

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


def add_formula(df):
    df = df.copy()
    SM = df["Soil_Moisture"]; RM = df["Rainfall_mm"]
    TC = df["Temperature_C"]; WS = df["Wind_Speed_kmh"]
    CGS = df["Crop_Growth_Stage"]; MU = df["Mulching_Used"]
    score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
             - (CGS == "Harvest") * 2 - (CGS == "Sowing") * 2 - (MU == "Yes"))
    df["formula_score"] = score.astype(np.float32)
    df["is_low"] = (score <= 0).astype(np.float32)
    df["is_high"] = (score > 3).astype(np.float32)
    return df


def round_to_step(x, step):
    x = pd.to_numeric(x, errors="coerce")
    return (x / step).round() * step if step else x


def build_features(train, test):
    """All features over the training (+ original) and test rows; returns the frame, FEATURES and CAT_COLS."""
    combined_df = pd.concat([train[FEATS], test[FEATS]], axis=0, ignore_index=True)
    combined_df = add_formula(combined_df)

    # 1. digit / NaN / sign features of the 11 numerics
    digit = []
    for col in NUMS:
        nan_flag = f"{col}__is_nan"
        combined_df[nan_flag] = pd.to_numeric(combined_df[col], errors="coerce").isna().astype(np.int8)
        digit.append(nan_flag)

        s = pd.to_numeric(combined_df[col], errors="coerce").fillna(0.0)
        step = ROUND_MAP.get(col, 1.0)
        s_r = round_to_step(s, step)
        scale = 10 if step < 1.0 else 1
        s_sc = (s_r * scale).round().astype(np.int64)
        max_v = int(np.abs(s_sc).max()) if len(s_sc) else 0
        n_dig = 1 if max_v == 0 else int(np.log10(max_v)) + 1

        for i in range(n_dig):
            nc = f"{col}_digit_pos_{i+1}"
            combined_df[nc] = ((np.abs(s_sc) // (10 ** i)) % 10).astype(np.int8)
            digit.append(nc)

        sign_col = f"{col}__neg"
        combined_df[sign_col] = (s_sc < 0).astype(np.int8)
        digit.append(sign_col)
    print(f"{len(digit)} digit/nan/sign features")

    # 2. binning features of the 11 numerics
    binning = []
    for col in NUMS:
        x = pd.to_numeric(combined_df[col], errors="coerce")
        for q in [4, 5, 10, 20]:
            nc = f"{col}_bin_q{q}"
            try:
                combined_df[nc] = pd.qcut(x, q=q, labels=False, duplicates="drop")
                binning.append(nc)
            except Exception:
                pass
        for b in [5, 10, 20]:
            nc = f"{col}_bin_cut{b}"
            combined_df[nc] = pd.cut(x, bins=b, labels=False)
            binning.append(nc)
        step = ROUND_MAP.get(col, 1.0)
        nc = f"{col}_bin_round_{str(step).replace('.', 'p')}"
        combined_df[nc] = round_to_step(x, step=step)
        binning.append(nc)

    for col in binning:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce").fillna(-1).astype(np.int32)
    print(f"{len(binning)} binning features")

    # 3. *_cat string categories of all 19 columns
    all_cats = []
    for c in FEATS:
        nc = f"{c}_cat"
        if c in NUMS:
            step = ROUND_MAP.get(c, 1.0)
            combined_df[nc] = round_to_step(combined_df[c], step=step).astype("float32")
            combined_df[nc] = combined_df[nc].fillna(-9999).astype(str)
        else:
            combined_df[nc] = combined_df[c].astype(str)
        all_cats.append(nc)

    # 4. rare flags of the high-cardinality numerics
    rare_flags = []
    for c in ["Soil_Moisture", "Humidity", "Previous_Irrigation_mm", "Rainfall_mm"]:
        rounded_str = combined_df[f"{c}_cat"]
        vc = rounded_str.value_counts(dropna=False)
        rf = f"{c}_cat__is_rare"
        combined_df[rf] = (rounded_str.map(vc).fillna(0) < RARE_MIN_COUNT).astype(np.int8)
        rare_flags.append(rf)
    print(f"{len(rare_flags)} rare-flag features")

    features = FEATS + FORMULA_FEATS + digit + binning + all_cats + rare_flags

    # string columns -> category
    cat_cols = [c for c in combined_df.columns if combined_df[c].dtype == "object"]
    for col in cat_cols:
        combined_df[col] = combined_df[col].astype(str).astype("category")
    return combined_df, features, cat_cols


def main():
    train = read_train()
    test = read_test()
    orig = read_original()
    print(f"train={train.shape}  test={test.shape}  orig={orig.shape}")

    train = train.drop(columns=[ID], errors="ignore")
    test = test.drop(columns=[ID], errors="ignore")

    train["target"] = train[TARGET].map(TARGET_MAP).astype(np.int32)
    orig["target"] = orig[TARGET].map(TARGET_MAP).astype(np.int32)
    train = train.drop(columns=[TARGET])
    orig = orig.drop(columns=[TARGET])

    n_comp = len(train)  # competition rows come first; the OOF covers only them
    if USE_ORIG:
        orig_use = orig[FEATS + ["target"]].copy()
        train = pd.concat([train[FEATS + ["target"]], orig_use], axis=0, ignore_index=True)
        print(f"train after orig-append: {train.shape}")

    y_full = train["target"].values.astype(np.int32)
    n_train_all = len(train)

    combined_df, features, cat_cols = build_features(train, test)
    train_fe = combined_df.iloc[:n_train_all].reset_index(drop=True)
    test_fe = combined_df.iloc[n_train_all:].reset_index(drop=True)
    print(f"train_fe={train_fe.shape}  test_fe={test_fe.shape}")
    print(f"FEATURES={len(features)}  CAT_COLS={len(cat_cols)}")

    oof = np.zeros((n_comp, N_CLASSES), dtype=np.float32)
    test_preds = np.zeros((len(test_fe), N_CLASSES), dtype=np.float32)

    # the split runs over the competition + original rows; validation = the fold's competition rows
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_bacs = []

    print("Starting ensemble training (3 configurations x n_ens=8 per fold)...")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_fe, y_full), 1):
        va_idx_comp = va_idx[va_idx < n_comp]

        X_tr = train_fe.loc[tr_idx, features].copy()
        y_tr = y_full[tr_idx]
        X_va = train_fe.loc[va_idx_comp, features].copy()
        y_va = y_full[va_idx_comp]

        fold_ens_va = np.zeros((len(va_idx_comp), N_CLASSES), dtype=np.float32)
        fold_ens_te = np.zeros((len(test_fe), N_CLASSES), dtype=np.float32)

        print(f"--- Fold {fold}/{N_SPLITS}  tr={len(X_tr):,}  va={len(X_va):,} ---")

        for model_idx, params in enumerate(top3_params, 1):
            model = RealMLP_TD_Classifier(**params)
            model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, cat_col_names=cat_cols)

            va_pred = model.predict_proba(X_va).astype(np.float32)
            fold_ens_va += va_pred / len(top3_params)
            bac_m = balanced_accuracy_score(y_va, va_pred.argmax(1))
            print(f"  Model {model_idx}: BAC={bac_m:.5f}")

            te_pred = model.predict_proba(test_fe[features]).astype(np.float32)
            fold_ens_te += te_pred / len(top3_params)

            del model, va_pred, te_pred
            torch.cuda.empty_cache()
            gc.collect()

        oof[va_idx_comp] = fold_ens_va
        fold_bac = balanced_accuracy_score(y_va, fold_ens_va.argmax(1))
        fold_bacs.append(fold_bac)
        print(f"  Fold {fold} ensemble BAC={fold_bac:.5f}")
        test_preds += fold_ens_te / N_SPLITS

        del X_tr, X_va, y_tr, y_va, fold_ens_va, fold_ens_te
        gc.collect()

    oof_bac = balanced_accuracy_score(y_full[:n_comp], oof.argmax(1))
    print("=" * 50)
    print(f"Fold BACs: {[f'{b:0.6f}' for b in fold_bacs]}")
    print(f"OOF Balanced Accuracy: {oof_bac:.6f}")
    print("=" * 50)
    save_pair(model_dir(FOLDER), PAIR, oof, test_preds)


if __name__ == "__main__":
    main()
