#!/usr/bin/env python
"""LightGBM on value fingerprints: counts, pair counts, a numeric signature, distances to the original rows and k-means
prototypes of the original rows (pair lgbm_synthetic_fingerprint).

The original rows (CSV as written; the 11 rows with a blank TotalCharges dropped, 7,032 rows) are a reference, not
training rows. Columns: the 19 predictors; categoricals = the 15 string columns, numerics = SeniorCitizen, tenure,
MonthlyCharges, TotalCharges. Features of the train / test rows:
  - the numerics median-imputed and standardized over train + test + original; the categoricals ordinal-encoded (fitted
    on train + test + original)
  - row statistics of the numerics: missing count, mean, std, min, max, range, mean / std of the fractional parts,
    near-integer counts (tolerance 1e-3, 1e-2), count of quarter-like fractions
  - count and log count of every categorical value over train + test + original; counts of every pair of the 8
    lowest-cardinality categoricals
  - count (and log count) of the row's numeric signature: the 16-quantile bins of MonthlyCharges and tenure, joined (the
    numeric columns of train + test + original by variance; TotalCharges is text in the original CSV and SeniorCitizen
    has fewer than 4 distinct quantiles)
  - Euclidean distance to the 1st / 3rd / 5th / 10th nearest original row, mean distance to the k nearest, ratio of the
    nearest distance to the 5-mean (standardized numerics)
  - MiniBatchKMeans (128 clusters, batch 4096, seed 42) of the original rows: cluster id, its original row count (and
    log), distance to the center
  - target encodings over the same 5 folds (each fold's rows encoded with the other 4 folds, test rows with all train
    rows) of the cluster id (smoothing 30) and of the 6 categoricals with the fewest levels (smoothing 20)
110 features. Model: lr 0.03, 127 leaves, min_data_in_leaf 200, feature_fraction 0.8, bagging 0.8 every round,
lambda_l2 3, seed 42, 32 threads; up to 5,000 rounds with early stopping (200) on the validation fold.
Usage: python train_synthetic_fingerprint.py"""
import gc
import sys
import warnings
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ID, ORIGINAL_FILE, TARGET, folds, labels, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "lgbm_synthetic_fingerprint"
N_SPLITS = 5
RANDOM_STATE = 42
MAX_CAT_PAIR_COLS = 8
MAX_TE_COLS = 6
TOP_NUMERIC_SIGNATURE_COLS = 6
KNN_K_LIST = [1, 3, 5, 10]
KMEANS_CLUSTERS = 128
KMEANS_BATCH_SIZE = 4096

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,
    min_data_in_leaf=200,
    feature_fraction=0.80,
    bagging_fraction=0.80,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=3.0,
    verbosity=-1,
    seed=RANDOM_STATE,
    n_estimators=5000,
    n_jobs=32,
)


def is_categorical_dtype(s):
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(s)
    )


def choose_cat_num_cols(df):
    cat_cols = [c for c in df.columns if is_categorical_dtype(df[c])]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, num_cols


def add_missing_and_basic_numeric_features(df, num_cols):
    out = pd.DataFrame(index=df.index)
    if len(num_cols) == 0:
        out["num_missing_count"] = 0
        return out

    num_df = df[num_cols].copy()
    out["num_missing_count"] = num_df.isna().sum(axis=1).astype(np.int16)
    out["num_mean"] = num_df.mean(axis=1)
    out["num_std"] = num_df.std(axis=1)
    out["num_min"] = num_df.min(axis=1)
    out["num_max"] = num_df.max(axis=1)
    out["num_range"] = out["num_max"] - out["num_min"]

    abs_num = num_df.abs()
    frac = (abs_num - np.floor(abs_num)).fillna(0.0)
    out["frac_mean"] = frac.mean(axis=1)
    out["frac_std"] = frac.std(axis=1)
    out["intlike_count_1e3"] = (frac < 1e-3).sum(axis=1).astype(np.int16)
    out["intlike_count_1e2"] = (frac < 1e-2).sum(axis=1).astype(np.int16)
    out["quarterlike_count"] = (np.isclose((frac * 4) % 1, 0, atol=1e-3)).sum(axis=1).astype(np.int16)
    return out


def add_count_features_from_combined(train_df, test_df, orig_df, cat_cols):
    out_train = pd.DataFrame(index=train_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if len(cat_cols) == 0:
        return out_train, out_test

    comb = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)
    for c in cat_cols:
        vc = comb[c].astype("string").fillna("__MISSING__").value_counts(dropna=False)
        out_train[f"{c}__count"] = train_df[c].astype("string").fillna("__MISSING__").map(vc).fillna(0).astype(np.float32)
        out_test[f"{c}__count"] = test_df[c].astype("string").fillna("__MISSING__").map(vc).fillna(0).astype(np.float32)
        out_train[f"{c}__logcount"] = np.log1p(out_train[f"{c}__count"])
        out_test[f"{c}__logcount"] = np.log1p(out_test[f"{c}__count"])
    return out_train, out_test


def add_pair_count_features(train_df, test_df, orig_df, cat_cols, max_pair_cols=8):
    out_train = pd.DataFrame(index=train_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if len(cat_cols) < 2:
        return out_train, out_test

    nunq = []
    for c in cat_cols:
        n = pd.concat([train_df[c], test_df[c], orig_df[c]], axis=0).astype("string").nunique(dropna=False)
        nunq.append((c, n))
    nunq = sorted(nunq, key=lambda x: x[1])
    chosen = [c for c, n in nunq if n > 1][:max_pair_cols]
    if len(chosen) < 2:
        return out_train, out_test

    comb = pd.concat([train_df[chosen], test_df[chosen], orig_df[chosen]], axis=0, ignore_index=True).astype("string").fillna("__MISSING__")
    for c1, c2 in combinations(chosen, 2):
        key = comb[c1] + "||" + comb[c2]
        vc = key.value_counts(dropna=False)
        tr_key = train_df[c1].astype("string").fillna("__MISSING__") + "||" + train_df[c2].astype("string").fillna("__MISSING__")
        te_key = test_df[c1].astype("string").fillna("__MISSING__") + "||" + test_df[c2].astype("string").fillna("__MISSING__")
        feat = f"{c1}__{c2}__paircount"
        out_train[feat] = tr_key.map(vc).fillna(0).astype(np.float32)
        out_test[feat] = te_key.map(vc).fillna(0).astype(np.float32)
    return out_train, out_test


def add_numeric_signature_count_features(train_df, test_df, orig_df, num_cols, top_k=6):
    out_train = pd.DataFrame(index=train_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if len(num_cols) == 0:
        return out_train, out_test

    comb_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True)
    var_order = comb_num.var(numeric_only=True).sort_values(ascending=False)
    chosen = list(var_order.head(min(top_k, len(var_order))).index)
    if len(chosen) == 0:
        return out_train, out_test

    bin_info = {}
    for c in chosen:
        s = comb_num[c].astype(float)
        vals = s.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(vals) < 100:
            continue
        qs = np.unique(np.quantile(vals, np.linspace(0, 1, 17)))
        if len(qs) < 4:
            continue
        bin_info[c] = qs
    if len(bin_info) == 0:
        return out_train, out_test

    def make_sig(df):
        parts = []
        for c, edges in bin_info.items():
            x = df[c].astype(float).replace([np.inf, -np.inf], np.nan)
            b = pd.cut(x, bins=edges, include_lowest=True, labels=False)
            b = b.astype("float").fillna(-1).astype(int).astype(str)
            parts.append(b.rename(c))
        mat = pd.concat(parts, axis=1)
        return mat.astype(str).agg("||".join, axis=1)

    comb_sig = make_sig(pd.concat([train_df, test_df, orig_df], axis=0, ignore_index=True))
    vc = comb_sig.value_counts(dropna=False)
    tr_sig = make_sig(train_df)
    te_sig = make_sig(test_df)
    out_train["num_signature_count"] = tr_sig.map(vc).fillna(0).astype(np.float32)
    out_test["num_signature_count"] = te_sig.map(vc).fillna(0).astype(np.float32)
    out_train["num_signature_logcount"] = np.log1p(out_train["num_signature_count"])
    out_test["num_signature_logcount"] = np.log1p(out_test["num_signature_count"])
    return out_train, out_test


def prepare_numeric_arrays(train_df, test_df, orig_df, num_cols):
    """Median-impute + standardize the numerics over train + test + original."""
    comb = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True)
    comb = comb.replace([np.inf, -np.inf], np.nan)
    comb_imp = SimpleImputer(strategy="median").fit_transform(comb)
    comb_std = StandardScaler().fit_transform(comb_imp).astype(np.float32)
    n_tr, n_te = len(train_df), len(test_df)
    return comb_std[:n_tr], comb_std[n_tr:n_tr + n_te], comb_std[n_tr + n_te:]


def add_knn_to_original_features(Xtr_num, Xte_num, Xorig_num, k_list=(1, 3, 5, 10)):
    out_train = pd.DataFrame(index=np.arange(Xtr_num.shape[0]))
    out_test = pd.DataFrame(index=np.arange(Xte_num.shape[0]))
    max_k = max(k_list)
    nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean", n_jobs=-1)
    nn.fit(Xorig_num)
    dtr, _ = nn.kneighbors(Xtr_num)
    dte, _ = nn.kneighbors(Xte_num)
    for k in k_list:
        out_train[f"orig_knn_dist_{k}"] = dtr[:, k - 1].astype(np.float32)
        out_test[f"orig_knn_dist_{k}"] = dte[:, k - 1].astype(np.float32)
        out_train[f"orig_knn_mean_{k}"] = dtr[:, :k].mean(axis=1).astype(np.float32)
        out_test[f"orig_knn_mean_{k}"] = dte[:, :k].mean(axis=1).astype(np.float32)
    out_train["orig_knn_ratio_1_5"] = out_train["orig_knn_dist_1"] / (out_train["orig_knn_mean_5"] + 1e-6)
    out_test["orig_knn_ratio_1_5"] = out_test["orig_knn_dist_1"] / (out_test["orig_knn_mean_5"] + 1e-6)
    return out_train, out_test


def add_kmeans_prototype_features(Xtr_num, Xte_num, Xorig_num, n_clusters=128, batch_size=4096, random_state=42):
    out_train = pd.DataFrame(index=np.arange(Xtr_num.shape[0]))
    out_test = pd.DataFrame(index=np.arange(Xte_num.shape[0]))

    n_clusters = min(n_clusters, max(2, Xorig_num.shape[0] // 10))
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state, n_init=10)
    km.fit(Xorig_num)
    tr_cluster = km.predict(Xtr_num)
    te_cluster = km.predict(Xte_num)
    orig_cluster = km.predict(Xorig_num)
    centers = km.cluster_centers_

    def row_dist_to_center(X, cid):
        return np.sqrt(((X - centers[cid]) ** 2).sum(axis=1))

    orig_counts = pd.Series(orig_cluster).value_counts().sort_index()
    orig_counts = orig_counts.reindex(np.arange(n_clusters), fill_value=0)

    out_train["orig_cluster_id"] = tr_cluster.astype(np.int32)
    out_test["orig_cluster_id"] = te_cluster.astype(np.int32)
    out_train["orig_cluster_size"] = pd.Series(tr_cluster).map(orig_counts).to_numpy(dtype=np.float32)
    out_test["orig_cluster_size"] = pd.Series(te_cluster).map(orig_counts).to_numpy(dtype=np.float32)
    out_train["orig_cluster_logsize"] = np.log1p(out_train["orig_cluster_size"])
    out_test["orig_cluster_logsize"] = np.log1p(out_test["orig_cluster_size"])
    out_train["orig_cluster_center_dist"] = row_dist_to_center(Xtr_num, tr_cluster).astype(np.float32)
    out_test["orig_cluster_center_dist"] = row_dist_to_center(Xte_num, te_cluster).astype(np.float32)
    return out_train, out_test


def make_oof_target_encoding(train_key, test_key, y, fold_list, smoothing=20.0):
    """Each fold's rows encoded with the other folds, test rows with all train rows; smoothed toward the prior."""
    train_key = pd.Series(train_key).reset_index(drop=True)
    test_key = pd.Series(test_key).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    prior = float(y.mean())
    oof = np.zeros(len(train_key), dtype=np.float32)

    for tr_idx, va_idx in fold_list:
        x_tr = train_key.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        x_va = train_key.iloc[va_idx]
        stats = pd.DataFrame({"key": x_tr.astype(str), "y": y_tr.values}).groupby("key")["y"].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * prior) / (stats["count"] + smoothing)
        oof[va_idx] = x_va.astype(str).map(smooth).fillna(prior).to_numpy(dtype=np.float32)

    stats_full = pd.DataFrame({"key": train_key.astype(str), "y": y.values}).groupby("key")["y"].agg(["mean", "count"])
    smooth_full = (stats_full["count"] * stats_full["mean"] + smoothing * prior) / (stats_full["count"] + smoothing)
    test_enc = test_key.astype(str).map(smooth_full).fillna(prior).to_numpy(dtype=np.float32)
    return oof, test_enc


def ordinal_encode_from_combined(train_df, test_df, orig_df, cat_cols):
    comb = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)
    comb = comb.astype("string").fillna("__MISSING__")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(comb)
    tr = enc.transform(train_df[cat_cols].astype("string").fillna("__MISSING__"))
    te = enc.transform(test_df[cat_cols].astype("string").fillna("__MISSING__"))
    tr = pd.DataFrame(tr, columns=[f"{c}__ord" for c in cat_cols], index=train_df.index).astype(np.float32)
    te = pd.DataFrame(te, columns=[f"{c}__ord" for c in cat_cols], index=test_df.index).astype(np.float32)
    return tr, te


def choose_te_cols(train_df, test_df, orig_df, cat_cols, max_te_cols=6):
    stats = []
    for c in cat_cols:
        nunq = pd.concat([train_df[c], test_df[c], orig_df[c]], axis=0).astype("string").nunique(dropna=False)
        if nunq > 1:
            stats.append((c, nunq))
    stats = sorted(stats, key=lambda x: (x[1] > 1000, x[1]))
    return [c for c, _ in stats[:max_te_cols]]


def build_all_features(train_df, test_df, orig_df, y, fold_list):
    common = [c for c in train_df.columns if c in test_df.columns and c in orig_df.columns]
    feature_cols = [c for c in common if c not in [TARGET, ID]]
    trainX = train_df[feature_cols].copy()
    testX = test_df[feature_cols].copy()
    origX = orig_df[feature_cols].copy()
    cat_cols, num_cols = choose_cat_num_cols(trainX)
    print(f"{len(feature_cols)} shared columns: {len(cat_cols)} categorical, {len(num_cols)} numeric", flush=True)

    f_tr_basic_num = add_missing_and_basic_numeric_features(trainX, num_cols)
    f_te_basic_num = add_missing_and_basic_numeric_features(testX, num_cols)
    f_tr_cat_ord, f_te_cat_ord = ordinal_encode_from_combined(trainX, testX, origX, cat_cols)
    f_tr_count, f_te_count = add_count_features_from_combined(trainX, testX, origX, cat_cols)
    f_tr_pair, f_te_pair = add_pair_count_features(trainX, testX, origX, cat_cols, max_pair_cols=MAX_CAT_PAIR_COLS)
    f_tr_sig, f_te_sig = add_numeric_signature_count_features(trainX, testX, origX, num_cols, top_k=TOP_NUMERIC_SIGNATURE_COLS)

    Xtr_num, Xte_num, Xorig_num = prepare_numeric_arrays(trainX, testX, origX, num_cols)
    f_tr_knn, f_te_knn = add_knn_to_original_features(Xtr_num, Xte_num, Xorig_num, k_list=KNN_K_LIST)
    f_tr_km, f_te_km = add_kmeans_prototype_features(Xtr_num, Xte_num, Xorig_num, n_clusters=KMEANS_CLUSTERS,
                                                     batch_size=KMEANS_BATCH_SIZE, random_state=RANDOM_STATE)

    te_train = pd.DataFrame(index=train_df.index)
    te_test = pd.DataFrame(index=test_df.index)
    oof_te, tst_te = make_oof_target_encoding(f_tr_km["orig_cluster_id"], f_te_km["orig_cluster_id"], y, fold_list,
                                              smoothing=30.0)
    te_train["te_orig_cluster_id"] = oof_te
    te_test["te_orig_cluster_id"] = tst_te
    for c in choose_te_cols(trainX, testX, origX, cat_cols, max_te_cols=MAX_TE_COLS):
        oof_te, tst_te = make_oof_target_encoding(trainX[c].astype("string").fillna("__MISSING__"),
                                                  testX[c].astype("string").fillna("__MISSING__"), y, fold_list,
                                                  smoothing=20.0)
        te_train[f"te_{c}"] = oof_te
        te_test[f"te_{c}"] = tst_te

    raw_num_train = pd.DataFrame(Xtr_num, columns=[f"{c}__z" for c in num_cols], index=train_df.index)
    raw_num_test = pd.DataFrame(Xte_num, columns=[f"{c}__z" for c in num_cols], index=test_df.index)

    X_train = pd.concat([raw_num_train, f_tr_cat_ord, f_tr_basic_num, f_tr_count, f_tr_pair, f_tr_sig, f_tr_knn, f_tr_km,
                         te_train], axis=1)
    X_test = pd.concat([raw_num_test, f_te_cat_ord, f_te_basic_num, f_te_count, f_te_pair, f_te_sig, f_te_knn, f_te_km,
                        te_test], axis=1)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(-999.0).astype(np.float32)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-999.0).astype(np.float32)
    return X_train, X_test


def main():
    train = read_train()
    test = read_test()
    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE)
    orig = orig.loc[orig.TotalCharges != " "]
    y = labels(train[TARGET])
    print(f"train {train.shape} | test {test.shape} | original {orig.shape} | positive rate {y.mean():.6f}", flush=True)

    fold_ids = folds(y)
    fold_list = [(np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]) for k in range(N_SPLITS)]

    X_train, X_test = build_all_features(train, test, orig, y, fold_list)
    print(f"X_train {X_train.shape} | X_test {X_test.shape}", flush=True)

    oof = np.zeros(len(X_train), dtype=np.float64)
    test_pred_folds = np.zeros((len(X_test), N_SPLITS), dtype=np.float64)
    for k, (tr_idx, va_idx) in enumerate(fold_list):
        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y[va_idx]
        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)],
        )
        va_pred = clf.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        test_pred_folds[:, k] = clf.predict_proba(X_test)[:, 1]
        print(f"fold {k + 1}/{N_SPLITS}: AUC {roc_auc_score(y_va, va_pred):.6f}", flush=True)
        del X_tr, y_tr, X_va, y_va, clf, va_pred
        gc.collect()

    print(f"{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, PAIR, oof, test_pred_folds.mean(axis=1))


if __name__ == "__main__":
    main()
