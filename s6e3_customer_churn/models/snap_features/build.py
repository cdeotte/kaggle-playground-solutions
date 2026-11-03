#!/usr/bin/env python
"""Snap and decimal-string features of the train + test rows (label-free), read by many gradient-boosting and
neural-net models.

MonthlyCharges and TotalCharges are read exactly as written in the CSV files (strings). Writes to cache/:
  combo.csv    848,849 rows: the train rows (train.csv order) followed by the test rows (test.csv order, Churn empty);
               the 21 columns of train.csv with 11 three-decimal values cut to two decimals (MonthlyCharges of id
               151415, TotalCharges of 10 ids), the decimal strings of both charges (tc, tc_len, tc0, mc, mc0, mc1,
               mc_len, tc1) and the snap features <col>_snap (the value itself when the original data contains the same
               string, else the numerically closest original value) and <col>_snap_diff = snap - value.
  combo_b.csv  combo.csv + orig_MonthlyCharges / orig_TotalCharges: the churn rate (Yes = 1) of the original rows whose
               value equals the row's snap value.
  orig.csv     the 7,032 original rows with a TotalCharges (customerID replaced by id 848,849 + row number) and their
               decimal strings (mc, mc0, mc1, mc_len, tc, tc0, tc1, tc_len).
Usage: python build.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ORIGINAL_FILE, cache_dir  # noqa: E402

AS_STRINGS = {"TotalCharges": "string", "MonthlyCharges": "string"}

# rows (position in train + test = id) whose value has three decimals
MONTHLY_FIX = {151415: "115.33"}
TOTAL_CUT_LAST_CHAR = [19542, 305615, 319208, 450690, 466283, 492101, 581362, 653927, 727684, 796527]

# column order of the decimal-string block in combo.csv
COMBO_DECIMAL_COLS = ["tc", "tc_len", "tc0", "mc", "mc0", "mc1", "mc_len", "tc1"]


def add_decimal_strings(df, col, p):
    """p = the decimals of col as written, p0 / p1 = first / second decimal ("-1" if there is one decimal),
    p_len = number of decimals."""
    df[p] = df[col].str.split(".").str[1]
    df[f"{p}0"] = df[p].str[0]
    df[f"{p}1"] = "-1"
    df[f"{p}_len"] = df[p].str.len()
    two = df[f"{p}_len"] == 2
    df.loc[two, f"{p}1"] = df.loc[two, p].str[1]


def build_snap_map(combo_s, orig_s, colname):
    """Map every combo string value that is not an original string value to the numerically closest original value."""
    combo_str = combo_s.astype("string")
    orig_str = orig_s.astype("string")

    orig_unique = pd.Series(orig_str.dropna().unique(), dtype="string")
    orig_num = pd.to_numeric(orig_unique, errors="coerce").astype("float64")
    keep = orig_num.notna()
    orig_unique = orig_unique[keep].reset_index(drop=True)
    orig_num = orig_num[keep].to_numpy()

    combo_unique = pd.Series(combo_str.dropna().unique(), dtype="string")
    rare = combo_unique[~combo_unique.isin(orig_unique)].reset_index(drop=True)
    rare_num = pd.to_numeric(rare, errors="coerce").astype("float64")
    ok = rare_num.notna()
    rare = rare[ok].reset_index(drop=True)
    rare_num = rare_num[ok].to_numpy()

    snap_map = {}
    if len(rare) == 0 or len(orig_num) == 0:
        return snap_map
    for s_val, x in zip(rare.tolist(), rare_num):
        j = int(np.argmin(np.abs(orig_num - x)))
        snap_map[s_val] = orig_unique.iloc[j]
    print(f"{colname}: snapping {len(snap_map):,} values not in the original data to the closest original value")
    return snap_map


def apply_snap(combo_df, orig_df, col):
    snap_col, diff_col = f"{col}_snap", f"{col}_snap_diff"
    combo_df[snap_col] = combo_df[col].astype("string")
    mp = build_snap_map(combo_df[col], orig_df[col], col)
    if mp:
        combo_df[snap_col] = combo_df[snap_col].replace(mp)
    orig_num = pd.to_numeric(combo_df[col], errors="coerce")
    snap_num = pd.to_numeric(combo_df[snap_col], errors="coerce")
    combo_df[diff_col] = (snap_num - orig_num).astype("float32")


def main():
    out = cache_dir("snap_features")

    train = pd.read_csv(DATA_DIR / "train.csv", dtype=AS_STRINGS)
    test = pd.read_csv(DATA_DIR / "test.csv", dtype=AS_STRINGS)
    test["Churn"] = ""
    raw_cols = list(train.columns)
    combo = pd.concat([train, test], axis=0, ignore_index=True)
    print("combo:", combo.shape)

    for i, v in MONTHLY_FIX.items():
        combo.loc[i, "MonthlyCharges"] = v
    combo.loc[TOTAL_CUT_LAST_CHAR, "TotalCharges"] = combo.loc[TOTAL_CUT_LAST_CHAR, "TotalCharges"].str[:-1]

    add_decimal_strings(combo, "MonthlyCharges", "mc")
    add_decimal_strings(combo, "TotalCharges", "tc")
    combo = combo[raw_cols + COMBO_DECIMAL_COLS].copy()

    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, dtype=AS_STRINGS)
    orig = orig.loc[orig.TotalCharges != " "]
    orig = orig.rename({"customerID": "id"}, axis=1)
    orig["id"] = np.arange(len(orig)) + len(combo)
    orig["id"] = orig["id"].astype("int64")
    add_decimal_strings(orig, "MonthlyCharges", "mc")
    add_decimal_strings(orig, "TotalCharges", "tc")
    print("orig:", orig.shape)

    apply_snap(combo, orig, "MonthlyCharges")
    apply_snap(combo, orig, "TotalCharges")

    combo.to_csv(out / "combo.csv", index=False)
    orig.to_csv(out / "orig.csv", index=False)

    orig["y"] = orig.Churn.map({"Yes": 1, "No": 0}).astype("float32")
    for c in ["MonthlyCharges", "TotalCharges"]:
        tmp = orig.groupby(c).y.mean()
        tmp.name = f"orig_{c}"
        tmp.index.name = f"{c}_snap"
        combo = combo.merge(tmp, on=f"{c}_snap", how="left")
    print("rows without an original TotalCharges rate:", int(combo.orig_TotalCharges.isna().sum()))
    combo.to_csv(out / "combo_b.csv", index=False)
    print(f"wrote {out / 'combo.csv'}, {out / 'combo_b.csv'}, {out / 'orig.csv'}")


if __name__ == "__main__":
    main()
