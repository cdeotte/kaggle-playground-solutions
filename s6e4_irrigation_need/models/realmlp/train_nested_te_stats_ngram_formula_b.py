#!/usr/bin/env python
"""RealMLP on formula / original-prior / percentile / digit features and nested target encodings (label std / min /
max, n-gram means) (pair realmlp_nested_te_stats_ngram_formula_b).

Features (competition rows only in training; the original rows feed priors and percentile ranks):
- the 19 raw columns; frequency of the 11 numeric values over train + original + test;
- moisture / rain, temperature / humidity, rain - moisture, and the formula score (Soil_Moisture < 25,
  Rainfall_mm < 300, Temperature_C > 30, Wind_Speed_kmh > 10, growth stage Harvest / Sowing, mulching);
- per-class original target rate of the 8 categoricals (24 columns);
- percentile rank of the 11 numerics within the original Low / Medium / High rows (33 columns);
- mod 10, mod 100 and fraction of the 11 numerics;
- the 11 numerics as categories (exact value strings); 15 bigrams of Crop_Type, Season, Region, Crop_Growth_Stage,
  Mulching_Used, Soil_Type and 4 trigrams of the first 4, as categories.
Per fold, target encodings fitted out-of-fold over 5 inner folds (seed 42) for the training rows and on all training
rows for the validation and test rows: std / min / max of the label (0 / 1 / 2) for the 11 numeric categories and the
8 categoricals, label mean for the 19 n-grams (76 columns).
Model: pytabkit RealMLP_TD_Classifier, 4 x 512 mish, PBLD numeric embeddings, n_ens 1, lr 0.13, batch 1024, up to
64 epochs with early stopping on the validation fold (cross-entropy), random_state 42.
Test = mean of the 5 fold models.
Usage: python train_nested_te_stats_ngram_formula_b.py"""
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_nested_te_stats_ngram_formula_b"

SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5

NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

TOP_CATS_FOR_NGRAM = ["Crop_Type", "Season", "Region", "Crop_Growth_Stage", "Mulching_Used", "Soil_Type"]
STATS = ["std", "min", "max"]

REALMLP_PARAMS = dict(
    act="mish",
    ls_eps_sched="coslog4",
    p_drop_sched="flat_cos",
    wd_sched="flat_cos",
    embedding_size=4,
    bias_wd_factor=0.0,
    weight_param="ntk",
    bias_lr_factor=0.1,
    use_parametric_act=True,
    add_front_scale=True,
    act_lr_factor=0.1,
    block_str="w-b-a-d",
    hidden_sizes="rectangular",
    bias_init_mode="he+5",
    weight_init_mode="std",
    val_metric_name="cross_entropy",
    num_emb_type="pbld",
    verbosity=2,
    batch_size=1024,
    random_state=SEED,
    n_epochs=64,
    early_stopping_additive_patience=24,
    early_stopping_multiplicative_patience=2,
    n_ens=1,
    ens_av_before_softmax=False,
    n_hidden_layers=4,
    hidden_width=512,
    lr=0.1301052035689966,
    wd=0.04914969178803988,
    p_drop=0.02682248660489578,
    scale_lr_factor=2.1896500647074553,
    use_ls=False,
    ls_eps=0.04088400565558735,
    plr_sigma=1.668950443527603,
    plr_lr_factor=0.27098419375438565,
    plr_hidden_1=8,
    plr_hidden_2=8,
    sq_mom=0.9855686567493073,
    first_layer_lr_factor=1.3750227365258307,
    use_early_stopping=True,
    max_one_hot_cat_size=15,
)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def pctrank(values, reference):
    ref_s = np.sort(reference)
    return (np.searchsorted(ref_s, values) / max(len(ref_s), 1)).astype("float32")


def build_prefold_features(train_df, test_df, orig_df, orig_y_arr):
    """Row-level and original-data features (no fold dependence)."""
    new_nums = []
    num_as_cat_cols = []

    # 1. frequency encoding
    for col in NUMS:
        freq = pd.concat([train_df[col], orig_df[col], test_df[col]], axis=0).value_counts(normalize=True)
        for df in [train_df, test_df, orig_df]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
        new_nums.append(f"FREQ_{col}")

    # 2. arithmetic interactions
    for df in [train_df, test_df, orig_df]:
        sm = df["Soil_Moisture"].values
        rf = df["Rainfall_mm"].values
        tc = df["Temperature_C"].values
        df["_Moisture_div_Rain"] = (sm / (rf + 1e-6)).astype("float32")
        df["_Temp_div_Humid"] = (tc / (df["Humidity"].values + 1e-6)).astype("float32")
        df["_Rain_minus_Moist"] = (rf - sm).astype("float32")
    new_nums += ["_Moisture_div_Rain", "_Temp_div_Humid", "_Rain_minus_Moist"]

    # 3. formula score
    for df in [train_df, test_df, orig_df]:
        sm = df["Soil_Moisture"].values
        rf = df["Rainfall_mm"].values
        tc = df["Temperature_C"].values
        ws = df["Wind_Speed_kmh"].values
        df["_formula_score"] = (
            (sm < 25).astype(int) * 2 + (rf < 300).astype(int) * 2 +
            (tc > 30).astype(int) + (ws > 10).astype(int) -
            (df["Crop_Growth_Stage"].values == "Harvest").astype(int) * 2 -
            (df["Crop_Growth_Stage"].values == "Sowing").astype(int) * 2 -
            (df["Mulching_Used"].values == "Yes").astype(int)
        ).astype("float32")
    new_nums.append("_formula_score")

    # 4. original per-class rate of each categorical
    for col in CATS:
        for k in range(3):
            grp = orig_df.copy()
            grp["_k"] = (orig_y_arr == k).astype(float)
            tmp = grp.groupby(col)["_k"].mean()
            overall = (orig_y_arr == k).mean()
            for df in [train_df, test_df]:
                df[f"ORIG_p{k}_{col}"] = df[col].map(tmp).fillna(overall).astype("float32")
            new_nums.append(f"ORIG_p{k}_{col}")

    # 5. percentile rank against the original per-class distributions
    for k in range(3):
        for col in NUMS:
            orig_k = orig_df.loc[orig_y_arr == k, col].values
            for df in [train_df, test_df]:
                df[f"pctrank_orig{k}_{col}"] = pctrank(df[col].values, orig_k)
            new_nums.append(f"pctrank_orig{k}_{col}")

    # 6. digit features of all numerics
    for col in NUMS:
        for df in [train_df, test_df, orig_df]:
            v = df[col].values.astype("float32")
            vf = np.floor(v).astype("int32")
            df[f"{col}_mod10"] = (vf % 10).astype("float32")
            df[f"{col}_mod100"] = (vf % 100).astype("float32")
            df[f"{col}_frac"] = (v - np.floor(v)).astype("float32")
        new_nums += [f"{col}_mod10", f"{col}_mod100", f"{col}_frac"]

    # 7. numerics as categories (string of the value)
    for col in NUMS:
        nc = f"CAT_{col}"
        for df in [train_df, test_df, orig_df]:
            df[nc] = df[col].astype(str).astype("category")
        num_as_cat_cols.append(nc)

    # 8. bigrams and trigrams
    bigram_cols, trigram_cols = [], []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col = f"BG_{c1}_{c2}"
        for df in [train_df, test_df, orig_df]:
            df[col] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype("category")
        bigram_cols.append(col)
    for c1, c2, c3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        col = f"TG_{c1}_{c2}_{c3}"
        for df in [train_df, test_df, orig_df]:
            df[col] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype("category")
        trigram_cols.append(col)

    ngram_cols = bigram_cols + trigram_cols
    features = NUMS + CATS + new_nums + num_as_cat_cols + ngram_cols
    te_base_cols = num_as_cat_cols + CATS   # for the std / min / max stats

    log(f"Base features before TE: {len(features)}")
    log(f"  nums={len(NUMS)}, cats={len(CATS)}, derived_nums={len(new_nums)}, "
        f"num_as_cat={len(num_as_cat_cols)}, bigrams={len(bigram_cols)}, trigrams={len(trigram_cols)}")

    return train_df, test_df, orig_df, features, te_base_cols, ngram_cols


def main():
    log("Loading data...")
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    orig = read_original()

    y = train[TARGET].map(TARGET_MAP).astype(int).values
    train = train.drop(columns=[TARGET])
    orig_y = orig[TARGET].map(TARGET_MAP).astype(int).values
    orig = orig.drop(columns=[TARGET])

    for df in [train, test, orig]:
        for c in NUMS:
            df[c] = df[c].astype("float32")
        for c in CATS:
            df[c] = df[c].astype(str).fillna("missing")

    N = len(train)
    M = len(test)
    log(f"Train: {N:,} | Test: {M:,}")

    log("Building pre-fold features...")
    train, test, orig, features, te_base_cols, ngram_cols = build_prefold_features(train, test, orig, orig_y)

    cat_col_names = CATS + [f"CAT_{c}" for c in NUMS] + ngram_cols

    fold_id = folds(y)
    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros((N, 3), dtype="float32")
    test_preds = np.zeros((M, 3), dtype="float32")
    fold_accs = []

    te_stat_names = [f"TE1_{col}_{s}" for col in te_base_cols for s in STATS]
    te_ng_names = [f"TE_ng_{col}" for col in ngram_cols]
    features_final = list(features) + te_stat_names + te_ng_names

    for fold in range(1, N_FOLDS + 1):
        tr_idx, val_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        t0 = time.time()
        log("=" * 60)
        log(f"Fold {fold}/{N_FOLDS}")

        X_tr = train.iloc[tr_idx][features].reset_index(drop=True).copy()
        X_tr[TARGET] = y[tr_idx]
        y_tr = y[tr_idx]
        X_val = train.iloc[val_idx][features].reset_index(drop=True).copy()
        y_val = y[val_idx]
        X_te = test[features].reset_index(drop=True).copy()

        # inner out-of-fold label std / min / max of the categoricals and numeric categories
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.iloc[in_tr].copy()
            for col in te_base_cols:
                tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                for c in tmp.columns:
                    fill = float(X_tr2[TARGET].agg(c.split("_")[-1]))
                    mapped = X_tr.iloc[in_va][col].map(tmp[c]).astype(float)
                    X_tr.loc[X_tr.index[in_va], c] = mapped.fillna(fill).values.astype("float32")

        # validation / test rows: statistics of all training rows
        for col in te_base_cols:
            tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            for s in STATS:
                col_name = f"TE1_{col}_{s}"
                fill = float(X_tr[TARGET].agg(s))
                X_val[col_name] = X_val[col].map(tmp[col_name]).astype(float).fillna(fill).astype("float32")
                X_te[col_name] = X_te[col].map(tmp[col_name]).astype(float).fillna(fill).astype("float32")
                X_tr[col_name] = pd.to_numeric(X_tr[col_name], errors="coerce").fillna(fill).astype("float32")

        # inner out-of-fold label mean of the n-grams
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.iloc[in_tr].copy()
            for col in ngram_cols:
                ng_mean = X_tr2.groupby(col, observed=False)[TARGET].mean()
                ng_name = f"TE_ng_{col}"
                mapped = X_tr.iloc[in_va][col].map(ng_mean).astype(float)
                X_tr.loc[X_tr.index[in_va], ng_name] = mapped.fillna(y_tr.mean()).values.astype("float32")

        # validation / test rows: n-gram means of all training rows
        for col in ngram_cols:
            ng_mean = X_tr.groupby(col, observed=False)[TARGET].mean().astype("float32")
            ng_name = f"TE_ng_{col}"
            fill = float(y_tr.mean())
            X_val[ng_name] = X_val[col].map(ng_mean).astype(float).fillna(fill).astype("float32")
            X_te[ng_name] = X_te[col].map(ng_mean).astype(float).fillna(fill).astype("float32")
            X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors="coerce").fillna(fill).astype("float32")

        use_cols = [c for c in features_final if c != TARGET]
        for c in use_cols:
            if c not in X_tr.columns:
                X_tr[c] = 0.0
            if c not in X_val.columns:
                X_val[c] = 0.0
            if c not in X_te.columns:
                X_te[c] = 0.0

        for c in use_cols:
            if c in cat_col_names:
                for df in [X_tr, X_val, X_te]:
                    df[c] = df[c].astype(str).fillna("missing").astype("category")
            else:
                for df in [X_tr, X_val, X_te]:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

        X_tr_final = X_tr[use_cols].reset_index(drop=True)
        X_val_final = X_val[use_cols].reset_index(drop=True)
        X_te_final = X_te[use_cols].reset_index(drop=True)

        log(f"  Features: {len(use_cols)} | cat_cols: {len([c for c in use_cols if c in cat_col_names])}")

        params = dict(REALMLP_PARAMS)
        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr_final, y_tr, X_val_final, y_val, cat_col_names=cat_col_names)

        val_prob = model.predict_proba(X_val_final)
        test_prob = model.predict_proba(X_te_final)

        oof_preds[val_idx] = val_prob.astype("float32")
        test_preds += test_prob.astype("float32") / N_FOLDS

        fold_acc = balanced_accuracy_score(y_val, np.argmax(val_prob, axis=1))
        fold_accs.append(fold_acc)
        log(f"  Fold {fold} Balanced ACC: {fold_acc:.5f} | time: {time.time() - t0:.1f}s")

        torch.cuda.empty_cache()
        gc.collect()

    overall = balanced_accuracy_score(y, np.argmax(oof_preds, axis=1))
    log(f"Mean fold ACC : {np.mean(fold_accs):.5f} +/- {np.std(fold_accs):.5f}")
    log(f"Overall OOF Balanced ACC: {overall:.5f}")
    save_pair(model_dir(FOLDER), PAIR, oof_preds, test_preds)


if __name__ == "__main__":
    main()
