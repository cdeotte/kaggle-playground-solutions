#!/usr/bin/env python
"""TabM-mini-normal in nested 5 x 5 cross-validation on frequency / ratio / original-prior features and per-class
target encodings of categoricals, numeric values and n-grams (pair tabm_nested_binary_te_ngram_b).

Features (competition rows only):
- the 11 numerics; value frequency of the 11 numerics and 8 categoricals over train + original;
- moisture / rain, temperature / humidity, moisture / temperature, high temperature (>= 30), low moisture (< 20),
  no rain flags;
- per-class original target rate of the 8 categoricals and the 11 numeric values (57 columns, unseen -> 0.33);
- percentile rank of Soil_Moisture within the original Low / Medium / High rows;
- per outer fold: per-class target means (one-vs-rest label mean per key, unseen -> the class rate) of the
  8 categoricals, the 11 numerics as exact value strings, 15 bigrams of Soil_Type, Crop_Type, Crop_Growth_Stage,
  Season, Irrigation_Type, Water_Source and 4 trigrams of the first 4 (114 columns): out-of-fold over
  StratifiedKFold(5, shuffle=True, random_state=42 + outer fold) for the outer training rows, fitted on all outer
  training rows for the outer validation and test rows.
All numeric columns are standardized on the outer training rows; the 8 categoricals are ordinal-encoded (unseen -> -1).
Nested CV: outer folds = the 5 folds of common/data.py; inside each outer training set an inner
StratifiedKFold(5, shuffle=True, random_state=1000 + outer fold) trains 5 models.
Model: pytabkit TabM_D_Classifier, tabm-mini-normal, k 32, 3 blocks x 256, piecewise-linear numeric embeddings
(d 24), dropout 0.2, lr 1e-3, weight decay 1e-3, batch 512, up to 50 epochs, patience 10 on the inner validation rows,
random_state 42.
OOF slice k (shape (5, 630000, 3)): the inner-validation predictions for outer fold k's training rows and the mean of
the 5 inner models for its held-out rows. Test = mean of the 25 models.
Usage: python train_nested_binary_te_ngram_b.py"""
import gc
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "tabm"
PAIR = "tabm_nested_binary_te_ngram_b"

SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5

TABM_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbosity": 0,
    "arch_type": "tabm-mini-normal",
    "tabm_k": 32,
    "num_emb_type": "pwl",
    "d_embedding": 24,
    "batch_size": 512,
    "lr": 1e-3,
    "n_epochs": 50,
    "dropout": 0.2,
    "d_block": 256,
    "n_blocks": 3,
    "patience": 10,
    "weight_decay": 1e-3,
    "random_state": SEED,
}

CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity", "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
TOP_CATS_FOR_NGRAM = CATS[:6]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / (len(ref_sorted) + 1e-6)).astype("float32")


def load_data():
    print("Loading datasets...")
    train = read_train()
    test = read_test()
    orig = read_original()

    train[TARGET] = train[TARGET].map(TARGET_MAP)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP)

    for df in [train, test]:
        if ID in df.columns:
            df.set_index(ID, inplace=True)

    print(f"Train: {train.shape}")
    print(f"Test : {test.shape}")
    return train, test, orig


def add_base_features(train, test, orig):
    new_nums = []

    print("[1/6] Frequency Encoding...")
    for col in NUMS + CATS:
        freq = pd.concat([train[col], orig[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
        new_nums.append(f"FREQ_{col}")

    print("[2/6] Arithmetic Interactions...")
    for df in [train, test, orig]:
        df["moisture_rain_ratio"] = (df["Soil_Moisture"] / (df["Rainfall_mm"] + 1)).astype("float32")
        df["temp_humidity_ratio"] = (df["Temperature_C"] / (df["Humidity"] + 1)).astype("float32")
        df["moisture_temp_ratio"] = (df["Soil_Moisture"] / (df["Temperature_C"] + 1)).astype("float32")
        df["is_high_temp"] = (df["Temperature_C"] >= 30).astype("float32")
        df["is_low_moisture"] = (df["Soil_Moisture"] < 20).astype("float32")
        df["is_no_rain"] = (df["Rainfall_mm"] == 0).astype("float32")
    new_nums += ["moisture_rain_ratio", "temp_humidity_ratio", "moisture_temp_ratio", "is_high_temp", "is_low_moisture", "is_no_rain"]

    print("[3/6] Orig Target Probability Mapping...")
    for class_idx in [0, 1, 2]:
        orig[f"target_{class_idx}"] = (orig[TARGET] == class_idx).astype(int)

    for col in CATS + NUMS:
        for class_idx in [0, 1, 2]:
            tmp = orig.groupby(col)[f"target_{class_idx}"].mean().astype("float32")
            _name = f"ORIG_proba_c{class_idx}_{col}"
            train = train.merge(tmp.rename(_name), on=col, how="left")
            test = test.merge(tmp.rename(_name), on=col, how="left")
            for df in [train, test]:
                df[_name] = df[_name].fillna(0.33).astype("float32")
            new_nums.append(_name)

    print("[4/6] Distribution and Quantile features...")
    orig_sm_c0 = orig.loc[orig[TARGET] == 0, "Soil_Moisture"].values
    orig_sm_c1 = orig.loc[orig[TARGET] == 1, "Soil_Moisture"].values
    orig_sm_c2 = orig.loc[orig[TARGET] == 2, "Soil_Moisture"].values

    for df in [train, test]:
        sm = df["Soil_Moisture"].values
        df["pctrank_c0_SM"] = pctrank_against(sm, orig_sm_c0)
        df["pctrank_c1_SM"] = pctrank_against(sm, orig_sm_c1)
        df["pctrank_c2_SM"] = pctrank_against(sm, orig_sm_c2)
    new_nums += ["pctrank_c0_SM", "pctrank_c1_SM", "pctrank_c2_SM"]

    print("[5/6] Numericals as Categories...")
    num_as_cat = []
    for col in NUMS:
        _new = f"CAT_{col}"
        num_as_cat.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str)

    print("[6/6] N-gram Features...")
    ngram_cols = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        ngram_cols.append(col_name)

    top4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(top4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
        ngram_cols.append(col_name)

    # keys to target-encode
    te_columns = CATS + num_as_cat + ngram_cols

    return train, test, new_nums, te_columns, CATS


def run_nested_training(train, test, new_nums, te_columns, base_cats):
    print("=" * 80)
    print(f"TRAINING TABM WITH {N_OUTER_FOLDS}x{N_INNER_FOLDS} NESTED CV")
    print("=" * 80)

    y_all = train[TARGET].values
    n_train = len(train)
    n_test = len(test)

    oof_matrix = np.zeros((N_OUTER_FOLDS, n_train, 3), dtype=np.float32)
    test_pred = np.zeros((n_test, 3), dtype=np.float32)

    outer_va_scores = []
    outer_full_scores = []

    tabm_cats = base_cats
    fold_id = folds(y_all)

    for outer_fold in range(1, N_OUTER_FOLDS + 1):
        outer_tr_idx = np.where(fold_id != outer_fold - 1)[0]
        outer_va_idx = np.where(fold_id == outer_fold - 1)[0]
        print(f"\n{'-' * 80}")
        print(f"OUTER FOLD {outer_fold}/{N_OUTER_FOLDS}")
        print(f"outer_tr size = {len(outer_tr_idx):,}")
        print(f"outer_va size = {len(outer_va_idx):,}")
        print(f"{'-' * 80}")

        X_outer_tr = train.iloc[outer_tr_idx].reset_index(drop=True).copy()
        X_outer_va = train.iloc[outer_va_idx].reset_index(drop=True).copy()
        X_te = test.copy()

        y_outer_tr = X_outer_tr[TARGET].values
        y_outer_va = X_outer_va[TARGET].values

        # 1. per-class target means (one-vs-rest label mean of each key)
        te_feat_names = []
        for col in te_columns:
            for c_idx in [0, 1, 2]:
                te_feat_names.append(f"TE_c{c_idx}_{col}")

        for df in [X_outer_tr, X_outer_va, X_te]:
            for c in te_feat_names:
                df[c] = np.nan

        # outer training rows: out-of-fold over 5 folds
        skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + outer_fold)
        X_outer_tr["TMP_TARGET"] = y_outer_tr

        for te_tr_idx, te_va_idx in skf_te.split(X_outer_tr, y_outer_tr):
            X_te_tr = X_outer_tr.iloc[te_tr_idx]
            for col in te_columns:
                for c_idx in [0, 1, 2]:
                    y_bin = (X_te_tr["TMP_TARGET"] == c_idx).astype(int)
                    ng_te = y_bin.groupby(X_te_tr[col]).mean()
                    name = f"TE_c{c_idx}_{col}"
                    mapped = X_outer_tr.iloc[te_va_idx][col].map(ng_te)
                    X_outer_tr.loc[te_va_idx, name] = mapped.values

        # outer validation and test rows: fitted on all outer training rows
        for col in te_columns:
            for c_idx in [0, 1, 2]:
                y_bin = (X_outer_tr["TMP_TARGET"] == c_idx).astype(int)
                ng_te = y_bin.groupby(X_outer_tr[col]).mean()
                name = f"TE_c{c_idx}_{col}"

                # unseen keys -> the class rate of the outer training rows
                global_mean = y_bin.mean()
                X_outer_tr[name] = X_outer_tr[name].fillna(global_mean).astype("float32")

                X_outer_va[name] = X_outer_va[col].map(ng_te).fillna(global_mean).astype("float32")
                X_te[name] = X_te[col].map(ng_te).fillna(global_mean).astype("float32")

        X_outer_tr.drop(columns=["TMP_TARGET"], inplace=True)

        # 2. scaling and categorical encoding
        tabm_nums = NUMS + new_nums + te_feat_names

        scaler = StandardScaler()
        X_outer_tr[tabm_nums] = scaler.fit_transform(X_outer_tr[tabm_nums].fillna(0))
        X_outer_va[tabm_nums] = scaler.transform(X_outer_va[tabm_nums].fillna(0))
        X_te[tabm_nums] = scaler.transform(X_te[tabm_nums].fillna(0))

        for df in [X_outer_tr, X_outer_va, X_te]:
            df[tabm_nums] = df[tabm_nums].astype("float32")

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_outer_tr[tabm_cats] = encoder.fit_transform(X_outer_tr[tabm_cats].astype(str))
        X_outer_va[tabm_cats] = encoder.transform(X_outer_va[tabm_cats].astype(str))
        X_te[tabm_cats] = encoder.transform(X_te[tabm_cats].astype(str))

        for df in [X_outer_tr, X_outer_va, X_te]:
            df[tabm_cats] = df[tabm_cats].astype(int)

        all_cols = tabm_nums + tabm_cats
        X_outer_va_final = X_outer_va[all_cols]
        X_te_final = X_te[all_cols]

        # 3. inner models
        outer_full_pred = np.zeros((n_train, 3), dtype=np.float32)
        outer_va_pred_sum = np.zeros((len(outer_va_idx), 3), dtype=np.float32)

        skf_inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=1000 + outer_fold)
        inner_scores = []

        for inner_fold, (inner_tr_idx, inner_va_idx) in enumerate(skf_inner.split(X_outer_tr, y_outer_tr), start=1):

            X_in_tr = X_outer_tr.iloc[inner_tr_idx][all_cols]
            y_in_tr = y_outer_tr[inner_tr_idx]
            X_in_va = X_outer_tr.iloc[inner_va_idx][all_cols]
            y_in_va = y_outer_tr[inner_va_idx]

            model = TabM_D_Classifier(**TABM_PARAMS)
            model.fit(
                X_in_tr, y_in_tr,
                X_val=X_in_va, y_val=y_in_va,
                cat_col_names=tabm_cats,
            )

            # inner validation rows (outer training rows of this slice)
            p_inner_va = model.predict_proba(X_in_va).astype(np.float32)
            global_inner_va_idx = outer_tr_idx[inner_va_idx]
            outer_full_pred[global_inner_va_idx] = p_inner_va

            inner_acc = balanced_accuracy_score(y_in_va, np.argmax(p_inner_va, axis=1))
            inner_scores.append(inner_acc)
            print(f"   -> Outer {outer_fold} | Inner {inner_fold}/{N_INNER_FOLDS} Balanced ACC = {inner_acc:.5f}")

            # outer validation rows (averaged over the 5 inner models)
            p_outer_va = model.predict_proba(X_outer_va_final).astype(np.float32)
            outer_va_pred_sum += p_outer_va

            # test rows (averaged over all 25 models)
            p_test = model.predict_proba(X_te_final).astype(np.float32)
            test_pred += p_test / (N_OUTER_FOLDS * N_INNER_FOLDS)

            del model, X_in_tr, X_in_va
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 4. outer fold slice
        outer_va_pred = outer_va_pred_sum / N_INNER_FOLDS
        outer_full_pred[outer_va_idx] = outer_va_pred
        oof_matrix[outer_fold - 1, :, :] = outer_full_pred

        outer_va_acc = balanced_accuracy_score(y_outer_va, np.argmax(outer_va_pred, axis=1))
        outer_full_acc = balanced_accuracy_score(y_all, np.argmax(outer_full_pred, axis=1))

        outer_va_scores.append(outer_va_acc)
        outer_full_scores.append(outer_full_acc)

        print(f"{'-' * 80}")
        print(f"Outer {outer_fold} Mean Inner ACC = {np.mean(inner_scores):.5f}")
        print(f"Outer {outer_fold} OUTER-VA ACC   = {outer_va_acc:.5f}")
        print(f"Outer {outer_fold} FULL-ROW ACC   = {outer_full_acc:.5f}")

    oof_mean = oof_matrix.mean(axis=0)
    oof_mean_acc = balanced_accuracy_score(y_all, np.argmax(oof_mean, axis=1))

    print("\n" + "=" * 80)
    print("NESTED TRAINING FINISHED")
    print(f"Mean OUTER-VA ACC        = {np.mean(outer_va_scores):.5f} +/- {np.std(outer_va_scores):.5f}")
    print(f"Mean-row OOF ACC         = {oof_mean_acc:.5f}")
    print("=" * 80)

    return oof_matrix, test_pred


def main():
    seed_everything(SEED)
    train, test, orig = load_data()
    train, test, new_nums, te_columns, base_cats = add_base_features(train, test, orig)
    oof_matrix, test_pred = run_nested_training(train, test, new_nums, te_columns, base_cats)
    save_pair(model_dir(FOLDER), PAIR, oof_matrix, test_pred)


if __name__ == "__main__":
    main()
