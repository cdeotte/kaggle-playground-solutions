#!/usr/bin/env python
"""RealMLP, three tuned configurations averaged, on digit / bin / rounded-category / rare-flag features, original rows
appended before the split (pair realmlp_3config_digits_bins_rare_original).

Rows: the 594,194 competition rows followed by the 7,032 original rows with a TotalCharges. Folds:
StratifiedKFold(5, shuffle=True, random_state=42) over these 601,226 rows; a fold trains on the competition and original
rows of the 4 other folds and is validated on all its own rows. The pair keeps the OOF of the competition rows.
Features (computed once over the 601,226 rows + test; the 19 raw columns plus, from lib/appended_original.py):
- for tenure, MonthlyCharges, TotalCharges: a NaN flag, the base-10 digits of the value rounded to its step (1, 0.5, 5;
  x10 for MonthlyCharges) and a sign flag; quantile bins (4, 5, 10, 20), equal-width bins (5, 10, 20) and the value
  rounded to its step (integer codes);
- every raw column as a string category (numerics rounded to their step);
- rare flags (< 25 rows) of the rounded MonthlyCharges / TotalCharges categories.
Model: pytabkit RealMLP_TD_Classifier in three tuned configurations (mish, PBLD numeric embeddings, n_ens 8, up to 32
epochs, batch 1024, random_state 42; the first two with early stopping), fitted with the fold's rows as validation set;
the three probabilities are averaged. Test = mean of the 5 fold models.
Usage: python train_3config_digits_bins_rare_original.py"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from lib.appended_original import FEATS, THREE_CONFIGS, add_digit_bin_cat_rare_features, cross_validate  # noqa: E402

warnings.simplefilter("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_3config_digits_bins_rare_original"
N_SPLITS = 5
SEED = 42


def coerce_totalcharges(df):
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").astype("float32")
    return df


def main():
    train = read_train()
    test = read_test()
    orig = read_original()
    orig = orig.loc[orig.TotalCharges.notna()].copy()  # the 11 customers with a blank TotalCharges
    orig["TotalCharges"] = orig["TotalCharges"].astype("float32")
    print("train:", train.shape, "test:", test.shape, "orig:", orig.shape)

    train = coerce_totalcharges(train)
    test = coerce_totalcharges(test)
    for col in ["MonthlyCharges", "tenure"]:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        orig[col] = pd.to_numeric(orig[col], errors="coerce")

    train[TARGET] = train[TARGET].map({"No": 0, "Yes": 1}).astype(np.int32)
    n_comp = len(train)

    # append the original rows
    orig_use = orig.loc[orig[TARGET].isin(["Yes", "No"])].copy()
    orig_use[TARGET] = orig_use[TARGET].map({"No": 0, "Yes": 1}).astype(np.int32)
    train = pd.concat([train[FEATS + [TARGET]], orig_use[FEATS + [TARGET]]], axis=0, ignore_index=True)
    test = test[FEATS].copy()
    print("train after appending orig:", train.shape)

    combined_df = pd.concat([train[FEATS], test[FEATS]], axis=0, ignore_index=True)
    digit, binning, all_cats, rare_flags = add_digit_bin_cat_rare_features(combined_df, FEATS)
    features = FEATS + digit + binning + all_cats + rare_flags

    # string columns -> category
    cat_cols = [c for c in combined_df.columns if combined_df[c].dtype == "object"]
    for col in cat_cols:
        combined_df[col] = combined_df[col].astype(str).astype("category")

    train_fe = combined_df.iloc[: len(train)].reset_index(drop=True)
    test_fe = combined_df.iloc[len(train):].reset_index(drop=True)
    y = train[TARGET].values.astype(np.int32)
    print("train_fe:", train_fe.shape, "test_fe:", test_fe.shape, "features:", len(features), "cat cols:", len(cat_cols))

    oof, test_pred = cross_validate(train_fe, test_fe, y, features, cat_cols, THREE_CONFIGS, N_SPLITS, SEED)
    print(f"OOF AUC (competition rows): {roc_auc_score(y[:n_comp], oof[:n_comp]):.6f}")
    save_pair(model_dir(FOLDER), PAIR, oof[:n_comp], test_pred)


if __name__ == "__main__":
    main()
