#!/usr/bin/env python
"""RealMLP with nested target encodings of the categoricals, the numeric values and bigrams / trigrams, plus
original-data priors and distribution features (pair realmlp_nested_te_stats_ngram).

Training rows = competition rows only; the original data (7,043 rows, blank TotalCharges -> median) only feeds features.
Features:
- frequencies of tenure / MonthlyCharges / TotalCharges over train + original + test; TotalCharges - tenure x
  MonthlyCharges, MonthlyCharges / (TotalCharges + 1), TotalCharges / (tenure + 1); service count, has_internet,
  has_phone;
- original churn rate of every categorical and numeric value (0.5 if unseen);
- TotalCharges against the original data: percentile ranks among all / churned / retained customers and within the
  row's InternetService and Contract, z-scores, the churned-vs-retained gaps, distances to the original quartiles of
  both classes and their gaps; MonthlyCharges minus the original mean of its InternetService;
- 35 digit features (first / second / last digit, mod 10 / 12 / 100, number of digits, round-number flags, rounded
  values, fractional parts, tenure years / months) and the three numerics as string categories;
- 15 bigrams of Contract, InternetService, PaymentMethod, OnlineSecurity, TechSupport, PaperlessBilling and the 4
  trigrams of the first four.
Per fold (the 5 folds of common/data.py): label std / min / max of the 16 categoricals and the 3 numeric-value
categories, out-of-fold over StratifiedKFold(5, shuffle=True, random_state=42) on the fold's training rows (validation /
test rows: maps of all training rows); label means of the 19 n-grams the same way (0.5 if unseen); scikit-learn
TargetEncoder (cv 5, smooth auto, seed 42) of the 19 categories. Categoricals, numeric-value categories and n-grams are
passed as categories.
Model: pytabkit RealMLP_TD_Classifier (4 x 512 mish, PBLD embeddings, n_ens 1, lr 0.13, batch 1024, up to 64 epochs
with early stopping on the validation fold's AUC, random_state 42). Test = mean of the 5 fold models.
Usage: python train_nested_te_stats_ngram.py"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import N_FOLDS, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_nested_te_stats_ngram"
INNER_FOLDS = 5
RANDOM_SEED = 42

REALMLP_PARAMS = {
    "act": "mish",
    "ls_eps_sched": "coslog4",
    "p_drop_sched": "flat_cos",
    "wd_sched": "flat_cos",
    "embedding_size": 4,
    "bias_wd_factor": 0.0,
    "weight_param": "ntk",
    "bias_lr_factor": 0.1,
    "use_parametric_act": True,
    "add_front_scale": True,
    "act_lr_factor": 0.1,
    "block_str": "w-b-a-d",
    "hidden_sizes": "rectangular",
    "bias_init_mode": "he+5",
    "weight_init_mode": "std",
    "val_metric_name": "1-auc_ovr",
    "num_emb_type": "pbld",
    "verbosity": 2,
    "batch_size": 1024,
    "random_state": RANDOM_SEED,
    "n_epochs": 64,
    "early_stopping_additive_patience": 24,
    "early_stopping_multiplicative_patience": 2,
    "n_ens": 1,
    "ens_av_before_softmax": False,
    "n_hidden_layers": 4,
    "hidden_width": 512,
    "lr": 0.1301052035689966,
    "wd": 0.04914969178803988,
    "p_drop": 0.02682248660489578,
    "scale_lr_factor": 2.1896500647074553,
    "use_ls": False,
    "ls_eps": 0.04088400565558735,
    "plr_sigma": 1.668950443527603,
    "plr_lr_factor": 0.27098419375438565,
    "plr_hidden_1": 8,
    "plr_hidden_2": 8,
    "sq_mom": 0.9855686567493073,
    "first_layer_lr_factor": 1.3750227365258307,
    "use_early_stopping": True,
    "max_one_hot_cat_size": 15,
}

TOP_CATS_FOR_NGRAM = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling",
]
CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
STATS = ["std", "min", "max"]


def load_data():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = train[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    orig[TARGET] = orig[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())
    print(f"train {train.shape}  test {test.shape}  orig {orig.shape}  target mean {train[TARGET].mean():.6f}")
    return train, test, orig


def cast_base_types(train, test, orig):
    for df in [train, test, orig]:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype("float32")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").astype("float32")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").astype("float32")
        for c in CATS:
            df[c] = df[c].astype(str)
    return train, test, orig


def add_block(df, cols_dict):
    block = pd.DataFrame(cols_dict, index=df.index)
    return pd.concat([df, block], axis=1)


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(reference)).astype("float32")


def zscore_against(values, reference):
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")


def build_features(train, test, orig):
    new_nums = []
    num_as_cat = []

    # frequency encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
        new_nums.append(f"FREQ_{col}")

    # arithmetic interactions
    for name, df in [("train", train), ("test", test), ("orig", orig)]:
        df2 = add_block(df, {
            "charges_deviation": (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32"),
            "monthly_to_total_ratio": (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32"),
            "avg_monthly_charges": (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32"),
        })
        if name == "train":
            train = df2
        elif name == "test":
            test = df2
        else:
            orig = df2
    new_nums += ["charges_deviation", "monthly_to_total_ratio", "avg_monthly_charges"]

    # service counts
    for name, df in [("train", train), ("test", test), ("orig", orig)]:
        df2 = add_block(df, {
            "service_count": (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32"),
            "has_internet": (df["InternetService"] != "No").astype("float32"),
            "has_phone": (df["PhoneService"] == "Yes").astype("float32"),
        })
        if name == "train":
            train = df2
        elif name == "test":
            test = df2
        else:
            orig = df2
    new_nums += ["service_count", "has_internet", "has_phone"]

    # original churn rate of every value
    for col in CATS + NUMS:
        tmp = orig.groupby(col, observed=False)[TARGET].mean()
        new_col = f"ORIG_proba_{col}"
        train[new_col] = train[col].map(tmp).fillna(0.5).astype("float32")
        test[new_col] = test[col].map(tmp).fillna(0.5).astype("float32")
        new_nums.append(new_col)

    # TotalCharges against the original distributions
    orig_churner_tc = orig.loc[orig[TARGET] == 1, "TotalCharges"].values
    orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, "TotalCharges"].values
    orig_tc = orig["TotalCharges"].values
    orig_is_mc_mean = orig.groupby("InternetService", observed=False)["MonthlyCharges"].mean()

    for name, df in [("train", train), ("test", test)]:
        tc = df["TotalCharges"].values
        vals = {
            "pctrank_nonchurner_TC": pctrank_against(tc, orig_nonchurner_tc),
            "pctrank_churner_TC": pctrank_against(tc, orig_churner_tc),
            "pctrank_orig_TC": pctrank_against(tc, orig_tc),
            "zscore_churn_gap_TC": (
                np.abs(zscore_against(tc, orig_churner_tc))
                - np.abs(zscore_against(tc, orig_nonchurner_tc))
            ).astype("float32"),
            "zscore_nonchurner_TC": zscore_against(tc, orig_nonchurner_tc),
            "pctrank_churn_gap_TC": (
                pctrank_against(tc, orig_churner_tc)
                - pctrank_against(tc, orig_nonchurner_tc)
            ).astype("float32"),
            "resid_IS_MC": (
                df["MonthlyCharges"].values
                - df["InternetService"].map(orig_is_mc_mean).fillna(0).values
            ).astype("float32"),
        }

        cond_is = np.zeros(len(df), dtype="float32")
        for cat_val in orig["InternetService"].unique():
            mask = (df["InternetService"] == cat_val).values
            ref = orig.loc[orig["InternetService"] == cat_val, "TotalCharges"].values
            if len(ref) > 0 and mask.sum() > 0:
                cond_is[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
        vals["cond_pctrank_IS_TC"] = cond_is

        cond_c = np.zeros(len(df), dtype="float32")
        for cat_val in orig["Contract"].unique():
            mask = (df["Contract"] == cat_val).values
            ref = orig.loc[orig["Contract"] == cat_val, "TotalCharges"].values
            if len(ref) > 0 and mask.sum() > 0:
                cond_c[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
        vals["cond_pctrank_C_TC"] = cond_c

        df2 = add_block(df, vals)
        if name == "train":
            train = df2
        else:
            test = df2

    new_nums += [
        "pctrank_nonchurner_TC", "zscore_churn_gap_TC", "pctrank_churn_gap_TC",
        "resid_IS_MC", "cond_pctrank_IS_TC", "zscore_nonchurner_TC",
        "pctrank_orig_TC", "pctrank_churner_TC", "cond_pctrank_C_TC",
    ]

    # distances to the original quartiles
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for name, df in [("train", train), ("test", test)]:
            vals = {
                f"dist_To_ch_{q_label}": np.abs(df["TotalCharges"] - ch_q).astype("float32"),
                f"dist_To_nc_{q_label}": np.abs(df["TotalCharges"] - nc_q).astype("float32"),
                f"qdist_gap_To_{q_label}": (
                    np.abs(df["TotalCharges"] - nc_q) - np.abs(df["TotalCharges"] - ch_q)
                ).astype("float32"),
            }
            df2 = add_block(df, vals)
            if name == "train":
                train = df2
            else:
                test = df2

    new_nums += [
        "qdist_gap_To_q50", "dist_To_ch_q50", "dist_To_nc_q50",
        "dist_To_nc_q25", "qdist_gap_To_q25",
        "dist_To_nc_q75", "dist_To_ch_q75", "qdist_gap_To_q75",
    ]

    # numerics as categories
    for col in NUMS:
        new_col = f"CAT_{col}"
        train[new_col] = train[col].astype(str).astype("category")
        test[new_col] = test[col].astype(str).astype("category")
        num_as_cat.append(new_col)

    # digit features
    digit_features = [
        "tenure_first_digit", "tenure_last_digit", "tenure_second_digit",
        "tenure_mod10", "tenure_mod12", "tenure_num_digits",
        "tenure_is_multiple_10", "tenure_rounded_10", "tenure_dev_from_round10",
        "mc_first_digit", "mc_last_digit", "mc_second_digit",
        "mc_mod10", "mc_mod100", "mc_num_digits",
        "mc_is_multiple_10", "mc_is_multiple_50",
        "mc_rounded_10", "mc_fractional", "mc_dev_from_round10",
        "tc_first_digit", "tc_last_digit", "tc_second_digit",
        "tc_mod10", "tc_mod100", "tc_num_digits",
        "tc_is_multiple_10", "tc_is_multiple_100",
        "tc_rounded_100", "tc_fractional", "tc_dev_from_round100",
        "tenure_years", "tenure_months_in_year",
        "mc_per_digit", "tc_per_digit",
    ]

    for name, df in [("train", train), ("test", test)]:
        t_str = df["tenure"].fillna(0).astype(int).astype(str)
        mc_floor = np.floor(df["MonthlyCharges"].fillna(0)).astype(int)
        tc_floor = np.floor(df["TotalCharges"].fillna(0)).astype(int)
        mc_str = df["MonthlyCharges"].fillna(0).astype(str).str.replace(".", "", regex=False)
        tc_str = df["TotalCharges"].fillna(0).astype(str).str.replace(".", "", regex=False)

        vals = {
            "tenure_first_digit": t_str.str[0].astype(int),
            "tenure_last_digit": t_str.str[-1].astype(int),
            "tenure_second_digit": t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0).astype("int16"),
            "tenure_mod10": (df["tenure"] % 10).astype("float32"),
            "tenure_mod12": (df["tenure"] % 12).astype("float32"),
            "tenure_num_digits": t_str.str.len().astype("int16"),
            "tenure_is_multiple_10": (df["tenure"] % 10 == 0).astype("float32"),
            "tenure_rounded_10": (np.round(df["tenure"] / 10) * 10).astype("float32"),
            "tenure_dev_from_round10": np.abs(df["tenure"] - np.round(df["tenure"] / 10) * 10).astype("float32"),
            "mc_first_digit": mc_str.str[0].replace("-", "0").fillna("0").astype(int),
            "mc_last_digit": mc_str.str[-1].replace("-", "0").fillna("0").astype(int),
            "mc_second_digit": mc_str.apply(lambda x: int(x[1]) if len(x) > 1 and x[1].isdigit() else 0).astype("int16"),
            "mc_mod10": (mc_floor % 10).astype("float32"),
            "mc_mod100": (mc_floor % 100).astype("float32"),
            "mc_num_digits": mc_floor.astype(str).str.len().astype("int16"),
            "mc_is_multiple_10": (mc_floor % 10 == 0).astype("float32"),
            "mc_is_multiple_50": (mc_floor % 50 == 0).astype("float32"),
            "mc_rounded_10": (np.round(df["MonthlyCharges"] / 10) * 10).astype("float32"),
            "mc_fractional": (df["MonthlyCharges"] - np.floor(df["MonthlyCharges"])).astype("float32"),
            "mc_dev_from_round10": np.abs(df["MonthlyCharges"] - np.round(df["MonthlyCharges"] / 10) * 10).astype("float32"),
            "tc_first_digit": tc_str.str[0].replace("-", "0").fillna("0").astype(int),
            "tc_last_digit": tc_str.str[-1].replace("-", "0").fillna("0").astype(int),
            "tc_second_digit": tc_str.apply(lambda x: int(x[1]) if len(x) > 1 and x[1].isdigit() else 0).astype("int16"),
            "tc_mod10": (tc_floor % 10).astype("float32"),
            "tc_mod100": (tc_floor % 100).astype("float32"),
            "tc_num_digits": tc_floor.astype(str).str.len().astype("int16"),
            "tc_is_multiple_10": (tc_floor % 10 == 0).astype("float32"),
            "tc_is_multiple_100": (tc_floor % 100 == 0).astype("float32"),
            "tc_rounded_100": (np.round(df["TotalCharges"] / 100) * 100).astype("float32"),
            "tc_fractional": (df["TotalCharges"] - np.floor(df["TotalCharges"])).astype("float32"),
            "tc_dev_from_round100": np.abs(df["TotalCharges"] - np.round(df["TotalCharges"] / 100) * 100).astype("float32"),
            "tenure_years": (df["tenure"] // 12).astype("float32"),
            "tenure_months_in_year": (df["tenure"] % 12).astype("float32"),
            "mc_per_digit": (df["MonthlyCharges"] / (mc_floor.astype(str).str.len().astype(float) + 0.001)).astype("float32"),
            "tc_per_digit": (df["TotalCharges"] / (tc_floor.astype(str).str.len().astype(float) + 0.001)).astype("float32"),
        }
        df2 = add_block(df, vals)
        if name == "train":
            train = df2
        else:
            test = df2
    new_nums += digit_features

    # bigrams / trigrams
    bigram_cols = []
    trigram_cols = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        train[col_name] = (train[c1].astype(str) + "_" + train[c2].astype(str)).astype("category")
        test[col_name] = (test[c1].astype(str) + "_" + test[c2].astype(str)).astype("category")
        bigram_cols.append(col_name)

    top4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(top4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        train[col_name] = (
            train[c1].astype(str) + "_" + train[c2].astype(str) + "_" + train[c3].astype(str)
        ).astype("category")
        test[col_name] = (
            test[c1].astype(str) + "_" + test[c2].astype(str) + "_" + test[c3].astype(str)
        ).astype("category")
        trigram_cols.append(col_name)

    ngram_cols = bigram_cols + trigram_cols
    features = NUMS + CATS + new_nums + num_as_cat + ngram_cols
    te_columns = num_as_cat + CATS
    print(f"engineered numerics {len(new_nums)}, n-grams {len(ngram_cols)}, features {len(features)}")
    return train, test, orig, features, te_columns, ngram_cols


def get_cat_cols_for_realmlp(features, te_ngram_columns):
    """The raw categoricals, the numeric-value categories and the n-grams, in this order."""
    out = []
    seen = set()
    for c in (CATS + [f"CAT_{x}" for x in NUMS] + te_ngram_columns):
        if c in features and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def finalize_fold_frames(X_tr, X_val, X_te, features, cat_cols):
    use_cols = []
    seen = set()
    for c in features:
        if c not in seen:
            use_cols.append(c)
            seen.add(c)

    X_tr = X_tr[use_cols].copy()
    X_val = X_val[use_cols].copy()
    X_te = X_te[use_cols].copy()

    for c in use_cols:
        if c in cat_cols:
            X_tr[c] = X_tr[c].astype(str).fillna("missing").astype("category")
            X_val[c] = X_val[c].astype(str).fillna("missing").astype("category")
            X_te[c] = X_te[c].astype(str).fillna("missing").astype("category")
        else:
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").astype("float32").fillna(0)
            X_val[c] = pd.to_numeric(X_val[c], errors="coerce").astype("float32").fillna(0)
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce").astype("float32").fillna(0)
    return X_tr, X_val, X_te


def train_model(train, test, features, te_columns, te_ngram_columns):
    np.random.seed(RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cat_cols = get_cat_cols_for_realmlp(features, te_ngram_columns)
    te_mean_cols = [f"TE_{col}" for col in te_columns]
    te_stat_cols = [f"TE1_{col}_{s}" for col in te_columns for s in STATS]
    te_ng_cols = [f"TE_ng_{col}" for col in te_ngram_columns]
    features_final = list(features) + te_mean_cols + te_stat_cols + te_ng_cols

    y = train[TARGET].values.astype(np.int32)
    fold_id = folds(y)
    oof = np.zeros(len(train), dtype="float32")
    pred = np.zeros(len(test), dtype="float32")
    params = dict(REALMLP_PARAMS)
    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    for fold in range(N_FOLDS):
        train_idx, val_idx = np.where(fold_id != fold)[0], np.where(fold_id == fold)[0]
        X_tr = train.loc[train_idx, features + [TARGET]].reset_index(drop=True).copy()
        y_tr = y[train_idx]
        X_val = train.loc[val_idx, features].reset_index(drop=True).copy()
        y_val = y[val_idx]
        X_te = test[features].reset_index(drop=True).copy()

        # inner-fold label std / min / max of the categoricals and numeric-value categories
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr, features + [TARGET]].copy()
            X_va2 = X_tr.loc[in_va, features].copy()
            for col in te_columns:
                tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_va2 = X_va2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")

        for col in te_columns:
            tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_te = X_te.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").fillna(0).astype("float32")
                X_val[c] = pd.to_numeric(X_val[c], errors="coerce").fillna(0).astype("float32")
                X_te[c] = pd.to_numeric(X_te[c], errors="coerce").fillna(0).astype("float32")

        # inner-fold label means of the n-grams
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr].copy()
            X_va2 = X_tr.loc[in_va].copy()
            for col in te_ngram_columns:
                ng_te = X_tr2.groupby(col, observed=False)[TARGET].mean()
                ng_name = f"TE_ng_{col}"
                mapped = X_va2[col].astype(str).map(ng_te)
                X_tr.loc[in_va, ng_name] = pd.to_numeric(mapped, errors="coerce").fillna(0.5).astype("float32").values

        for col in te_ngram_columns:
            ng_te = X_tr.groupby(col, observed=False)[TARGET].mean()
            ng_name = f"TE_ng_{col}"
            X_val[ng_name] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32")
            X_te[ng_name] = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32")
            if ng_name in X_tr.columns:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors="coerce").fillna(0.5).astype("float32")
            else:
                X_tr[ng_name] = np.float32(0.5)

        # scikit-learn TargetEncoder means of the categoricals and numeric-value categories
        te = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth="auto", target_type="binary",
                           random_state=RANDOM_SEED)
        X_tr[te_mean_cols] = te.fit_transform(X_tr[te_columns], y_tr)
        X_val[te_mean_cols] = te.transform(X_val[te_columns])
        X_te[te_mean_cols] = te.transform(X_te[te_columns])

        X_tr, X_val, X_te = finalize_fold_frames(X_tr, X_val, X_te, features_final, cat_cols)
        print(f"fold {fold + 1}/{N_FOLDS}: X_tr {X_tr.shape}, X_val {X_val.shape}, X_te {X_te.shape}", flush=True)

        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, cat_col_names=cat_cols)

        val_pred = model.predict_proba(X_val)[:, 1].astype("float32")
        test_pred = model.predict_proba(X_te)[:, 1].astype("float32")
        oof[val_idx] = val_pred
        pred += test_pred / N_FOLDS
        print(f"fold {fold + 1} AUC = {roc_auc_score(y_val, val_pred):.6f}", flush=True)

        del model, X_tr, X_val, X_te, val_pred, test_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"OOF AUC = {roc_auc_score(y, oof):.6f}")
    return oof, pred


def main():
    train, test, orig = load_data()
    train, test, orig = cast_base_types(train, test, orig)
    train, test, orig, features, te_columns, ngram_cols = build_features(train, test, orig)
    oof, pred = train_model(train, test, features, te_columns, ngram_cols)
    save_pair(model_dir(FOLDER), PAIR, oof, pred)


if __name__ == "__main__":
    main()
