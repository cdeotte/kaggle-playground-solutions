"""Shared code of the RealMLP models that append the original rows to the competition rows before the split
(train_3config_digits_bins_rare_original.py, train_crosses_priors_drift_original.py): the digit / bin /
rounded-category / rare-flag block, the three tuned RealMLP configurations and the cross-validation loop."""
import gc

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

FEATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

NUM_DIGIT_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
# rounding steps of the numeric -> category strings
ROUND_MAP = {"tenure": 1.0, "MonthlyCharges": 0.5, "TotalCharges": 5.0}
# rare flag threshold (rounded value counts over the rows of combined_df)
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
    "val_metric_name": "1-auc_ovr",
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

THREE_CONFIGS = [
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


def round_to_step(x, step):
    x = pd.to_numeric(x, errors="coerce")
    if step == 0:
        return x
    return (x / step).round() * step


def add_digit_bin_cat_rare_features(combined_df, feats=FEATS):
    """Adds to combined_df (in place), computed over all its rows:
    - for tenure / MonthlyCharges / TotalCharges a NaN flag, the base-10 digits of the value rounded to its step
      (x10 when the step is < 1) and a sign flag;
    - quantile bins (4, 5, 10, 20), equal-width bins (5, 10, 20) and the value rounded to its step (integer codes);
    - every column of feats as a string category (numerics rounded to their step);
    - rare flags (< RARE_MIN_COUNT rows) of the rounded MonthlyCharges / TotalCharges categories.
    Returns the lists (digit, binning, all_cats, rare_flags)."""
    digit = []
    for col in NUM_DIGIT_COLS:
        nan_flag = f"{col}__is_nan"
        combined_df[nan_flag] = pd.to_numeric(combined_df[col], errors="coerce").isna().astype(np.int8)
        digit.append(nan_flag)

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
            digit.append(new_col)

        sign_col = f"{col}__neg"
        combined_df[sign_col] = (s_scaled < 0).astype(np.int8)
        digit.append(sign_col)
    print(f"{len(digit)} digit / NaN / sign features")

    binning = []
    for col in NUM_DIGIT_COLS:
        x = pd.to_numeric(combined_df[col], errors="coerce")
        for q in [4, 5, 10, 20]:
            new_col = f"{col}_bin_q{q}"
            try:
                combined_df[new_col] = pd.qcut(x, q=q, labels=False, duplicates="drop")
                binning.append(new_col)
            except Exception:
                pass
        for b in [5, 10, 20]:
            new_col = f"{col}_bin_cut{b}"
            combined_df[new_col] = pd.cut(x, bins=b, labels=False)
            binning.append(new_col)
        step = ROUND_MAP.get(col, 1.0)
        new_col = f"{col}_bin_round_{str(step).replace('.', 'p')}"
        combined_df[new_col] = round_to_step(x, step=step)
        binning.append(new_col)

    for col in binning:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce").fillna(-1).astype(np.int32)
    print(f"{len(binning)} binning features")

    all_cats = []
    for c in feats:
        nc = f"{c}_cat"
        if c in NUM_DIGIT_COLS:
            step = ROUND_MAP.get(c, 1.0)
            combined_df[nc] = round_to_step(combined_df[c], step=step).astype("float32")
            combined_df[nc] = combined_df[nc].fillna(-9999).astype(str)
        else:
            combined_df[nc] = combined_df[c].astype(str)
        all_cats.append(nc)

    rare_flags = []
    for c in ["MonthlyCharges", "TotalCharges"]:
        rounded_str = combined_df[f"{c}_cat"]
        vc = rounded_str.value_counts(dropna=False)
        rare_mask = rounded_str.map(vc).fillna(0) < RARE_MIN_COUNT
        rf = f"{c}_cat__is_rare"
        combined_df[rf] = rare_mask.astype(np.int8)
        rare_flags.append(rf)
    print(f"{len(rare_flags)} rare-flag features: {rare_flags}")
    return digit, binning, all_cats, rare_flags


def cross_validate(train_fe, test_fe, y, features, cat_cols, configs, n_splits, seed=42):
    """StratifiedKFold(n_splits, shuffle=True, random_state=seed) over all rows of train_fe (competition + original
    rows). Per fold every configuration in configs is fitted on the other folds' rows with the fold's rows as validation
    set; their probabilities are averaged. Returns the OOF over all rows of train_fe and the mean test prediction of the
    n_splits fold models."""
    oof_preds = np.zeros(len(train_fe), dtype=np.float32)
    test_preds = np.zeros(len(test_fe), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_fe, y), 1):
        X_tr = train_fe.loc[tr_idx, features].copy()
        y_tr = y[tr_idx].copy()
        X_va = train_fe.loc[va_idx, features].copy()
        y_va = y[va_idx].copy()

        fold_ens_va = np.zeros(len(va_idx), dtype=np.float32)
        fold_ens_te = np.zeros(len(test_fe), dtype=np.float32)

        for model_idx, params in enumerate(configs, 1):
            model = RealMLP_TD_Classifier(**params)
            model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, cat_col_names=cat_cols)

            va_pred = model.predict_proba(X_va)[:, 1].astype(np.float32)
            fold_ens_va += va_pred / len(configs)
            print(f"fold {fold}/{n_splits} | config {model_idx} | AUC {roc_auc_score(y_va, va_pred):.5f}", flush=True)

            te_pred = model.predict_proba(test_fe[features])[:, 1].astype(np.float32)
            fold_ens_te += te_pred / len(configs)

            del model, va_pred, te_pred
            torch.cuda.empty_cache()
            gc.collect()

        oof_preds[va_idx] = fold_ens_va
        if len(configs) > 1:
            print(f"fold {fold}/{n_splits} | mean  | AUC {roc_auc_score(y_va, fold_ens_va):.5f}", flush=True)
        test_preds += fold_ens_te / n_splits

        del X_tr, X_va, y_tr, y_va, fold_ens_va, fold_ens_te
        gc.collect()

    return oof_preds, test_preds
