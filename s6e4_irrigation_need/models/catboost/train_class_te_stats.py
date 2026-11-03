#!/usr/bin/env python
"""Class-weighted CatBoost on per-class target statistics and category n-grams (pair catboost_class_te_stats_ngrams_b).

Label-free features (99):
  - the 19 columns; a threshold score 2 (moisture < 25) + 2 (rainfall < 300) + (temperature > 30) + (wind > 10)
    - 2 (Harvest) - 2 (Sowing) - (mulching) and the flags score <= 0 / score > 3
  - frequency of every numeric value over the train + test + original rows
  - per-class original frequency of the 8 categoricals (24)
  - percentile rank of soil moisture, rainfall, temperature and wind speed among the original rows of each class (12)
  - every numeric rounded to an integer, as a string (11)
  - the 15 pairs of Crop_Growth_Stage, Mulching_Used, Season, Crop_Type, Soil_Type, Region and the 4 triples of the
    first four, as strings
The string columns (38) are native CatBoost categories.
Per fold, for each class k (indicator target), out-of-fold over an inner StratifiedKFold(5, seed 42) of the training
rows and from all training rows for the validation / test rows:
  - mean / std / min / max of the indicator for the 8 categoricals and the 11 rounded numerics (228)
  - mean of the indicator for the 19 n-grams (57)
Missing encodings: 0.33 for means, 0 otherwise.
Model: CatBoost GPU, MultiClass with eval_metric TotalF1, auto_class_weights Balanced, depth 7, lr 0.00984,
l2 5.333, random_strength 2.877, bagging_temperature 0.264, 254 borders, min_data_in_leaf 14, up to 20,000 iterations
with early stopping (200) on the validation fold, seed 42. Test = mean of the 5 fold models.
Usage: python train_class_te_stats.py"""
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "catboost_class_te_stats_ngrams_b"
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
LABEL = "target"

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture',
        'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']
FORMULA_NUMS = ['Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh']
TOP_CATS = ['Crop_Growth_Stage', 'Mulching_Used', 'Season', 'Crop_Type', 'Soil_Type', 'Region']
STATS = ['mean', 'std', 'min', 'max']

CAT_PARAMS = dict(
    iterations=20000,
    learning_rate=0.00984,
    depth=7,
    l2_leaf_reg=5.333,
    random_strength=2.877,
    bagging_temperature=0.264,
    border_count=254,
    min_data_in_leaf=14,
    grow_policy='SymmetricTree',
    random_seed=SEED,
    early_stopping_rounds=200,
    loss_function='MultiClass',
    eval_metric='TotalF1',
    auto_class_weights='Balanced',
    task_type='GPU',
    verbose=200,
    use_best_model=True,
)


def build_features():
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
    orig.drop(columns=[ID], inplace=True, errors='ignore')
    print(f'Train: {train.shape}, Test: {test.shape}', flush=True)

    # threshold score
    for df in [train, test, orig]:
        SM = df['Soil_Moisture']; RM = df['Rainfall_mm']; TC = df['Temperature_C']
        WS = df['Wind_Speed_kmh']; CGS = df['Crop_Growth_Stage']; MU = df['Mulching_Used']
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10)
                 - (CGS == 'Harvest') * 2 - (CGS == 'Sowing') * 2 - (MU == 'Yes'))
        df['formula_score'] = score.astype(np.float32)
        df['is_low'] = (score <= 0).astype(np.float32)
        df['is_high'] = (score > 3).astype(np.float32)

    # frequency encoding
    for c in NUMS:
        freq = pd.concat([train[c], test[c], orig[c]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{c}'] = df[c].map(freq).fillna(0).astype(np.float32)

    # per-class original priors
    for c in CATS:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            gm = float((y_orig == k).mean())
            for df in [train, test, orig]:
                df[f'ORIG_k{k}_{c}'] = df[c].map(m).fillna(gm).astype(np.float32)

    # percentile rank among the original rows of each class
    for c in FORMULA_NUMS:
        for k in range(3):
            ref = np.sort(orig.loc[y_orig == k, c].values)
            if len(ref) > 0:
                for df in [train, test]:
                    df[f'pctrank_k{k}_{c}'] = (np.searchsorted(ref, df[c].values) / (len(ref) + 1e-6)).astype(np.float32)

    # numerics as categories
    num_as_cat = []
    for c in NUMS:
        name = f'CAT_{c}'
        for df in [train, test]:
            df[name] = df[c].round(0).astype(int).astype(str)
        num_as_cat.append(name)

    bigram_cols = []
    for c1, c2 in combinations(TOP_CATS, 2):
        name = f'BG_{c1}_{c2}'
        for df in [train, test]:
            df[name] = df[c1].astype(str) + '_' + df[c2].astype(str)
        bigram_cols.append(name)
    trigram_cols = []
    for c1, c2, c3 in combinations(TOP_CATS[:4], 3):
        name = f'TG_{c1}_{c2}_{c3}'
        for df in [train, test]:
            df[name] = df[c1].astype(str) + '_' + df[c2].astype(str) + '_' + df[c3].astype(str)
        trigram_cols.append(name)

    features = [c for c in train.columns if c != LABEL]
    te_columns = CATS + num_as_cat               # mean / std / min / max per class
    te_ngram_columns = bigram_cols + trigram_cols  # mean per class
    print(f'Features: {len(features)}, TE cols: {len(te_columns)}, n-gram TE cols: {len(te_ngram_columns)}', flush=True)
    return train, test, y, features, te_columns, te_ngram_columns


def add_target_stats(X_tr, X_va, X_te, y_tr, te_columns, te_ngram_columns, skf_inner):
    for k in range(3):
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.iloc[in_tr]
            X_va2 = X_tr.iloc[in_va]
            y_k = (X_tr2[LABEL] == k).astype(float)
            for col in te_columns:
                if col not in X_tr2.columns:
                    continue
                tmp = y_k.groupby(X_tr2[col]).agg(STATS)
                for s in STATS:
                    te_name = f'TE_k{k}_{s}_{col}'
                    mapped = X_va2[col].map(tmp[s]).astype(np.float32)
                    X_tr.loc[X_va2.index, te_name] = mapped.values

        y_k_full = (X_tr[LABEL] == k).astype(float)
        for col in te_columns:
            if col not in X_tr.columns:
                continue
            tmp = y_k_full.groupby(X_tr[col]).agg(STATS)
            for s in STATS:
                te_name = f'TE_k{k}_{s}_{col}'
                X_tr[te_name] = X_tr[te_name].fillna(0.33 if s == 'mean' else 0).astype(np.float32)
                X_va[te_name] = X_va[col].map(tmp[s]).fillna(0.33 if s == 'mean' else 0).astype(np.float32)
                X_te[te_name] = X_te[col].map(tmp[s]).fillna(0.33 if s == 'mean' else 0).astype(np.float32)

    for k in range(3):
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.iloc[in_tr]
            X_va2 = X_tr.iloc[in_va]
            y_k = (X_tr2[LABEL] == k).astype(float)
            for col in te_ngram_columns:
                if col not in X_tr2.columns:
                    continue
                ng_te = y_k.groupby(X_tr2[col]).mean()
                te_name = f'TE_ng_k{k}_{col}'
                X_tr.loc[X_va2.index, te_name] = X_va2[col].map(ng_te).astype(np.float32).values

        y_k_full = (X_tr[LABEL] == k).astype(float)
        for col in te_ngram_columns:
            if col not in X_tr.columns:
                continue
            ng_te = y_k_full.groupby(X_tr[col]).mean()
            te_name = f'TE_ng_k{k}_{col}'
            X_tr[te_name] = X_tr[te_name].fillna(0.33).astype(np.float32)
            X_va[te_name] = X_va[col].map(ng_te).fillna(0.33).astype(np.float32)
            X_te[te_name] = X_te[col].map(ng_te).fillna(0.33).astype(np.float32)
    return X_tr, X_va, X_te


def main():
    train, test, y, features, te_columns, te_ngram_columns = build_features()
    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)

    fold_ids = folds(y)
    oof_preds = np.zeros((len(train), 3), dtype=np.float32)
    test_preds = np.zeros((len(test), 3), dtype=np.float32)
    for k in range(N_FOLDS):
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        print(f'Fold {k}', flush=True)
        t0 = time.time()

        X_tr = train.loc[tr_idx, features + [LABEL]].reset_index(drop=True).copy()
        y_tr = y[tr_idx]
        X_va = train.loc[va_idx, features].reset_index(drop=True).copy()
        y_va = y[va_idx]
        X_te = test[features].reset_index(drop=True).copy()

        X_tr, X_va, X_te = add_target_stats(X_tr, X_va, X_te, y_tr, te_columns, te_ngram_columns, skf_inner)
        X_tr = X_tr.drop(columns=[LABEL], errors='ignore')

        all_feats = X_tr.columns.tolist()
        cat_feat_names = [c for c in all_feats if X_tr[c].dtype == 'object']
        for c in cat_feat_names:
            for df in [X_tr, X_va, X_te]:
                df[c] = df[c].fillna('__NA__').astype(str)
        for c in all_feats:
            if c not in cat_feat_names:
                for df in [X_tr, X_va, X_te]:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)
        cat_indices = [all_feats.index(c) for c in cat_feat_names]

        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_indices, feature_names=all_feats)
        val_pool = Pool(X_va, label=y_va, cat_features=cat_indices, feature_names=all_feats)
        test_pool = Pool(X_te, cat_features=cat_indices, feature_names=all_feats)

        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool)

        p_va = model.predict_proba(val_pool).astype(np.float32)
        p_te = model.predict_proba(test_pool).astype(np.float32)
        oof_preds[va_idx] = p_va
        test_preds += p_te / N_FOLDS

        bac = balanced_accuracy_score(y_va, np.argmax(p_va, axis=1))
        print(f'  Fold {k} BAC={bac:.5f} best_iter={model.best_iteration_} ({time.time() - t0:.0f}s)', flush=True)

        del model, train_pool, val_pool, test_pool, X_tr, X_va, X_te
        gc.collect()

    print(f'{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, np.argmax(oof_preds, axis=1)):.6f}', flush=True)
    save_pair(HERE, PAIR, oof_preds, test_preds)


if __name__ == '__main__':
    main()
