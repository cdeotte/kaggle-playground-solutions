#!/usr/bin/env python
"""RealMLP on the raw columns, a formula score and per-class original priors (pair realmlp_formula_original_priors_b).

Features (46 columns, competition rows only): the 19 raw columns; the formula score (Soil_Moisture < 25,
Rainfall_mm < 300, Temperature_C > 30, Wind_Speed_kmh > 10, growth stage Harvest / Sowing, mulching) and the flags
score <= 0 and score > 3; the per-class original target rate of the 8 categoricals (24 columns).
Model: pytabkit RealMLP_TD_Classifier, 3 x 384 mish, n_ens 8, lr 0.03, wd 0.02, p_drop 0.08, batch 1024, up to
80 epochs with early stopping on the validation fold (additive patience 20), random_state 42.
Test = mean of the 5 fold models.
Usage: python train_formula_original_priors_b.py"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_formula_original_priors_b"
SEED = 42
LABEL = "target"  # integer label column (0 / 1 / 2)

CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

REALMLP_PARAMS = {
    "device": "cuda",
    "random_state": SEED,
    "verbosity": 2,
    "n_epochs": 80,
    "batch_size": 1024,
    "n_ens": 8,
    "use_early_stopping": True,
    "early_stopping_additive_patience": 20,
    "early_stopping_multiplicative_patience": 1,
    "act": "mish",
    "hidden_sizes": "rectangular",
    "hidden_width": 384,
    "n_hidden_layers": 3,
    "embedding_size": 8,
    "p_drop": 0.08,
    "p_drop_sched": "flat_cos",
    "lr": 0.03,
    "wd": 0.02,
    "use_ls": False,
}


def main():
    train = read_train()
    test = read_test()
    orig = read_original()

    train[LABEL] = train[TARGET].map(TARGET_MAP).astype(np.int32)
    orig[LABEL] = orig[TARGET].map(TARGET_MAP).astype(np.int32)
    y = train[LABEL].values
    y_orig = orig[LABEL].values

    for c in [ID, TARGET]:
        for df in [train, test]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    for df in [train, test]:
        SM = df["Soil_Moisture"]; RM = df["Rainfall_mm"]; TC = df["Temperature_C"]
        WS = df["Wind_Speed_kmh"]; CGS = df["Crop_Growth_Stage"]; MU = df["Mulching_Used"]
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10) - (CGS == "Harvest") * 2 - (CGS == "Sowing") * 2 - (MU == "Yes"))
        df["formula_score"] = score.astype(np.float32)
        df["is_low"] = (score <= 0).astype(np.float32)
        df["is_high"] = (score > 3).astype(np.float32)

    for c in CATS:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            gm = float((y_orig == k).mean())
            for df in [train, test]:
                df[f"ORIG_k{k}_{c}"] = df[c].map(m).fillna(gm).astype(np.float32)

    features = [c for c in train.columns if c not in [LABEL]]
    cat_cols = [c for c in features if train[c].dtype.name in ["object", "string", "category"]]
    print(f"Features: {len(features)}, Cat: {len(cat_cols)}")

    fold_id = folds(y)
    oof_preds = np.zeros((len(train), 3), dtype=np.float32)
    test_preds = np.zeros((len(test), 3), dtype=np.float32)
    test_fe = test[features].copy()

    for fold in range(1, N_FOLDS + 1):
        tr_idx, va_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        print(f"\n=== Fold {fold}/{N_FOLDS} ===")
        t0 = time.time()

        X_tr = train.iloc[tr_idx][features].copy()
        X_va = train.iloc[va_idx][features].copy()

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        model.fit(X_tr, y[tr_idx], X_val=X_va, y_val=y[va_idx], cat_col_names=cat_cols)

        va_probs = model.predict_proba(X_va).astype(np.float32)
        oof_preds[va_idx] = va_probs
        test_preds += model.predict_proba(test_fe).astype(np.float32) / N_FOLDS

        bac = balanced_accuracy_score(y[va_idx], np.argmax(va_probs, axis=1))
        print(f"  Fold {fold} BAC={bac:.5f} ({time.time() - t0:.0f}s)")
        del model
        torch.cuda.empty_cache()
        gc.collect()

    oof_bac = balanced_accuracy_score(y, np.argmax(oof_preds, axis=1))
    print(f"\nOOF BAC: {oof_bac:.5f}")
    save_pair(model_dir(FOLDER), PAIR, oof_preds, test_preds)


if __name__ == "__main__":
    main()
