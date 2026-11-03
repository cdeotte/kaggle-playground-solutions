#!/usr/bin/env python
"""LightGBM on value-fingerprint features: counts, pair counts, numeric signatures, distances to the original rows and k-means
prototypes of the original rows (pair lgbm_fingerprint_knn_kmeans).

Columns: the 14 shared columns; categoricals = string columns and numerics with < 100 values in train + test (10),
numerics = the rest (LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress). Features of the train / test rows
(the original rows are a reference, not training rows):
  - the numerics standardized over train + test + original (median-imputed), the categoricals ordinal-encoded
  - row statistics of the numerics (mean, std, min, max, range, fractional-part mean / std, near-integer counts)
  - count and log count of every categorical value over train + test + original; counts of every pair of the 10
    lowest-cardinality categoricals
  - count of the row's numeric signature (16-quantile bins of the 8 highest-variance numerics, joined)
  - Euclidean distance to the 1st / 3rd / 5th / 10th nearest original row and the mean distance to the k nearest
    (standardized numerics), ratio of the nearest distance to the 5-mean
  - MiniBatchKMeans (128 clusters, seed 42) of the original rows: cluster id, its original row count (and log), distance to
    the center, the cluster's original target rate
  - target encodings over the same 5 folds (each fold's rows encoded with the other 4 folds, test rows with all train rows)
    of the cluster id (smoothing 30) and of the 10 categoricals with the fewest levels (smoothing 20)
Model: LightGBM (32 threads), lr 0.03, 127 leaves, min_data_in_leaf 200, feature_fraction 0.8, bagging 0.8 every round,
lambda_l2 3, seed 42; up to 5,000 rounds with early stopping (200) on the validation fold. Test = mean of the 5 fold models.
Usage: python train_fingerprint.py"""
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'lgbm_fingerprint_knn_kmeans'
N_SPLITS = 5
RANDOM_STATE = 42
MAX_CAT_PAIR_COLS = 10
MAX_TE_COLS = 10
TOP_NUMERIC_SIGNATURE_COLS = 8
KNN_K_LIST = [1, 3, 5, 10]
KMEANS_CLUSTERS = 128
KMEANS_BATCH_SIZE = 4096

LGB_PARAMS = dict(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
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
    num_threads=32,
)


def to_binary_y(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce').fillna(0).astype(np.int8).to_numpy()
    return s.astype(str).str.lower().map({'false': 0, 'true': 1, '0': 0, '1': 1, 'no': 0, 'yes': 1}).fillna(0).astype(np.int8).to_numpy()


def is_categorical_series(s, train_s=None, test_s=None):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s):
        return True
    if train_s is not None and test_s is not None:
        nunq = pd.concat([train_s, test_s], ignore_index=True).nunique(dropna=False)
        return nunq < 100
    return False


def choose_cat_num_cols(train_df, test_df):
    cat_cols, num_cols = [], []
    for c in train_df.columns:
        if is_categorical_series(train_df[c], train_df[c], test_df[c]):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return cat_cols, num_cols


def add_missing_and_basic_numeric_features(df, num_cols):
    out = pd.DataFrame(index=df.index)
    if len(num_cols) == 0:
        out['num_missing_count'] = 0
        return out
    num_df = df[num_cols].apply(pd.to_numeric, errors='coerce')
    out['num_missing_count'] = num_df.isna().sum(axis=1).astype(np.int16)
    out['num_mean'] = num_df.mean(axis=1).astype('float32')
    out['num_std'] = num_df.std(axis=1).fillna(0).astype('float32')
    out['num_min'] = num_df.min(axis=1).astype('float32')
    out['num_max'] = num_df.max(axis=1).astype('float32')
    out['num_range'] = (out['num_max'] - out['num_min']).astype('float32')
    abs_num = num_df.abs()
    frac = (abs_num - np.floor(abs_num)).fillna(0.0)
    out['frac_mean'] = frac.mean(axis=1).astype('float32')
    out['frac_std'] = frac.std(axis=1).fillna(0).astype('float32')
    out['intlike_count_1e3'] = (frac < 1e-3).sum(axis=1).astype(np.int16)
    out['intlike_count_1e2'] = (frac < 1e-2).sum(axis=1).astype(np.int16)
    out['quarterlike_count'] = (np.isclose((frac * 4) % 1, 0, atol=1e-3)).sum(axis=1).astype(np.int16)
    return out


def add_count_features_from_combined(train_df, test_df, orig_df, cat_cols):
    out_train = pd.DataFrame(index=train_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if len(cat_cols) == 0:
        return out_train, out_test
    comb = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)
    for c in cat_cols:
        vc = comb[c].astype('string').fillna('__MISSING__').value_counts(dropna=False)
        tr = train_df[c].astype('string').fillna('__MISSING__').map(vc).fillna(0).astype(np.float32)
        te = test_df[c].astype('string').fillna('__MISSING__').map(vc).fillna(0).astype(np.float32)
        out_train[f'{c}__count'] = tr
        out_test[f'{c}__count'] = te
        out_train[f'{c}__logcount'] = np.log1p(tr)
        out_test[f'{c}__logcount'] = np.log1p(te)
    return out_train, out_test


def add_pair_count_features(train_df, test_df, orig_df, cat_cols, max_pair_cols=10):
    out_train = pd.DataFrame(index=train_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if len(cat_cols) < 2:
        return out_train, out_test
    nunq = []
    for c in cat_cols:
        n = pd.concat([train_df[c], test_df[c], orig_df[c]], axis=0).astype('string').nunique(dropna=False)
        nunq.append((c, n))
    chosen = [c for c, n in sorted(nunq, key=lambda x: x[1]) if n > 1][:max_pair_cols]
    if len(chosen) < 2:
        return out_train, out_test
    comb = pd.concat([train_df[chosen], test_df[chosen], orig_df[chosen]], axis=0, ignore_index=True).astype('string').fillna('__MISSING__')
    for c1, c2 in combinations(chosen, 2):
        key = comb[c1] + '||' + comb[c2]
        vc = key.value_counts(dropna=False)
        tr_key = train_df[c1].astype('string').fillna('__MISSING__') + '||' + train_df[c2].astype('string').fillna('__MISSING__')
        te_key = test_df[c1].astype('string').fillna('__MISSING__') + '||' + test_df[c2].astype('string').fillna('__MISSING__')
        feat = f'{c1}__{c2}__paircount'
        out_train[feat] = tr_key.map(vc).fillna(0).astype(np.float32)
        out_test[feat] = te_key.map(vc).fillna(0).astype(np.float32)
    return out_train, out_test


def add_numeric_signature_count_features(train_df, test_df, orig_df, num_cols, top_k=8):
    out_train = pd.DataFrame(index=train_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if len(num_cols) == 0:
        return out_train, out_test
    comb_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True).apply(pd.to_numeric, errors='coerce')
    var_order = comb_num.var(numeric_only=True).sort_values(ascending=False)
    chosen = list(var_order.head(min(top_k, len(var_order))).index)
    bin_info = {}
    for c in chosen:
        vals = comb_num[c].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype='float64')
        if len(vals) < 100:
            continue
        qs = np.unique(np.quantile(vals, np.linspace(0, 1, 17)))
        if len(qs) >= 4:
            bin_info[c] = qs
    if not bin_info:
        return out_train, out_test

    def make_sig(df):
        parts = []
        for c, edges in bin_info.items():
            x = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            b = pd.cut(x, bins=edges, include_lowest=True, labels=False).astype('float').fillna(-1).astype(int).astype(str)
            parts.append(b.rename(c))
        return pd.concat(parts, axis=1).astype(str).agg('||'.join, axis=1)

    comb_sig = make_sig(pd.concat([train_df, test_df, orig_df], axis=0, ignore_index=True))
    vc = comb_sig.value_counts(dropna=False)
    tr_sig = make_sig(train_df)
    te_sig = make_sig(test_df)
    out_train['num_signature_count'] = tr_sig.map(vc).fillna(0).astype(np.float32)
    out_test['num_signature_count'] = te_sig.map(vc).fillna(0).astype(np.float32)
    out_train['num_signature_logcount'] = np.log1p(out_train['num_signature_count'])
    out_test['num_signature_logcount'] = np.log1p(out_test['num_signature_count'])
    return out_train, out_test


def prepare_numeric_arrays(train_df, test_df, orig_df, num_cols):
    if len(num_cols) == 0:
        return (np.zeros((len(train_df), 0), dtype=np.float32), np.zeros((len(test_df), 0), dtype=np.float32), np.zeros((len(orig_df), 0), dtype=np.float32))
    comb = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True).apply(pd.to_numeric, errors='coerce')
    comb = comb.replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    comb_std = scl.fit_transform(imp.fit_transform(comb)).astype(np.float32)
    n_tr = len(train_df)
    n_te = len(test_df)
    return comb_std[:n_tr], comb_std[n_tr:n_tr + n_te], comb_std[n_tr + n_te:]


def add_knn_to_original_features(Xtr_num, Xte_num, Xorig_num, k_list=(1, 3, 5, 10)):
    out_train = pd.DataFrame(index=np.arange(Xtr_num.shape[0]))
    out_test = pd.DataFrame(index=np.arange(Xte_num.shape[0]))
    if Xorig_num.shape[1] == 0 or Xorig_num.shape[0] == 0:
        return out_train, out_test
    max_k = min(max(k_list), Xorig_num.shape[0])
    nn = NearestNeighbors(n_neighbors=max_k, metric='euclidean', n_jobs=32)
    nn.fit(Xorig_num)
    dtr, _ = nn.kneighbors(Xtr_num)
    dte, _ = nn.kneighbors(Xte_num)
    for k in k_list:
        kk = min(k, max_k)
        out_train[f'orig_knn_dist_{k}'] = dtr[:, kk - 1].astype(np.float32)
        out_test[f'orig_knn_dist_{k}'] = dte[:, kk - 1].astype(np.float32)
        out_train[f'orig_knn_mean_{k}'] = dtr[:, :kk].mean(axis=1).astype(np.float32)
        out_test[f'orig_knn_mean_{k}'] = dte[:, :kk].mean(axis=1).astype(np.float32)
    out_train['orig_knn_ratio_1_5'] = out_train['orig_knn_dist_1'] / (out_train['orig_knn_mean_5'] + 1e-6)
    out_test['orig_knn_ratio_1_5'] = out_test['orig_knn_dist_1'] / (out_test['orig_knn_mean_5'] + 1e-6)
    return out_train, out_test


def add_kmeans_prototype_features(Xtr_num, Xte_num, Xorig_num, y_orig, n_clusters=128, batch_size=4096, random_state=42):
    out_train = pd.DataFrame(index=np.arange(Xtr_num.shape[0]))
    out_test = pd.DataFrame(index=np.arange(Xte_num.shape[0]))
    if Xorig_num.shape[1] == 0 or Xorig_num.shape[0] == 0:
        return out_train, out_test
    n_clusters = min(n_clusters, max(2, Xorig_num.shape[0] // 10))
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state, n_init=10)
    km.fit(Xorig_num)
    tr_cluster = km.predict(Xtr_num)
    te_cluster = km.predict(Xte_num)
    orig_cluster = km.predict(Xorig_num)
    centers = km.cluster_centers_
    tr_dist = np.sqrt(((Xtr_num - centers[tr_cluster]) ** 2).sum(axis=1)).astype(np.float32)
    te_dist = np.sqrt(((Xte_num - centers[te_cluster]) ** 2).sum(axis=1)).astype(np.float32)
    orig_counts = pd.Series(orig_cluster).value_counts().sort_index().reindex(np.arange(n_clusters), fill_value=0)
    orig_rate = pd.DataFrame({'c': orig_cluster, 'y': y_orig}).groupby('c')['y'].mean().reindex(np.arange(n_clusters)).fillna(float(np.mean(y_orig)))
    out_train['orig_cluster_id'] = tr_cluster.astype(np.int32)
    out_test['orig_cluster_id'] = te_cluster.astype(np.int32)
    out_train['orig_cluster_size'] = pd.Series(tr_cluster).map(orig_counts).to_numpy(dtype=np.float32)
    out_test['orig_cluster_size'] = pd.Series(te_cluster).map(orig_counts).to_numpy(dtype=np.float32)
    out_train['orig_cluster_logsize'] = np.log1p(out_train['orig_cluster_size'])
    out_test['orig_cluster_logsize'] = np.log1p(out_test['orig_cluster_size'])
    out_train['orig_cluster_center_dist'] = tr_dist
    out_test['orig_cluster_center_dist'] = te_dist
    out_train['orig_cluster_orig_rate'] = pd.Series(tr_cluster).map(orig_rate).to_numpy(dtype=np.float32)
    out_test['orig_cluster_orig_rate'] = pd.Series(te_cluster).map(orig_rate).to_numpy(dtype=np.float32)
    return out_train, out_test


def make_oof_target_encoding_binary(train_key, test_key, y, folds, smoothing=20.0):
    train_key = pd.Series(train_key).reset_index(drop=True).astype(str)
    test_key = pd.Series(test_key).reset_index(drop=True).astype(str)
    y = np.asarray(y, dtype=np.float32)
    prior = float(y.mean())
    oof = np.zeros(len(train_key), dtype=np.float32)
    for tr_idx, va_idx in folds:
        key_tr = train_key.iloc[tr_idx]
        key_va = train_key.iloc[va_idx]
        stats = pd.DataFrame({'k': key_tr.values, 'y': y[tr_idx]}).groupby('k')['y'].agg(['mean', 'count'])
        enc = (stats['mean'] * stats['count'] + prior * smoothing) / (stats['count'] + smoothing)
        oof[va_idx] = key_va.map(enc).fillna(prior).astype(np.float32).values
    stats_full = pd.DataFrame({'k': train_key.values, 'y': y}).groupby('k')['y'].agg(['mean', 'count'])
    enc_full = (stats_full['mean'] * stats_full['count'] + prior * smoothing) / (stats_full['count'] + smoothing)
    test_enc = test_key.map(enc_full).fillna(prior).astype(np.float32).values
    return oof, test_enc


def ordinal_encode_from_combined(train_df, test_df, orig_df, cat_cols):
    if len(cat_cols) == 0:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)
    comb = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True).astype('string').fillna('__MISSING__')
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    enc.fit(comb)
    tr = enc.transform(train_df[cat_cols].astype('string').fillna('__MISSING__'))
    te = enc.transform(test_df[cat_cols].astype('string').fillna('__MISSING__'))
    return (pd.DataFrame(tr, columns=[f'{c}__ord' for c in cat_cols], index=train_df.index).astype(np.float32),
            pd.DataFrame(te, columns=[f'{c}__ord' for c in cat_cols], index=test_df.index).astype(np.float32))


def choose_te_cols(train_df, test_df, orig_df, cat_cols, max_te_cols=10):
    stats = []
    for c in cat_cols:
        nunq = pd.concat([train_df[c], test_df[c], orig_df[c]], axis=0).astype('string').nunique(dropna=False)
        if nunq > 1:
            stats.append((c, nunq))
    stats = sorted(stats, key=lambda x: (x[1] > 2000, x[1]))
    return [c for c, _ in stats[:max_te_cols]]


def build_all_features(train_df, test_df, orig_df, y, y_orig, folds):
    common = [c for c in train_df.columns if c in test_df.columns and c in orig_df.columns]
    feature_cols = [c for c in common if c not in [TARGET, ID]]
    trainX = train_df[feature_cols].copy()
    testX = test_df[feature_cols].copy()
    origX = orig_df[feature_cols].copy()
    cat_cols, num_cols = choose_cat_num_cols(trainX, testX)
    print(f'{len(feature_cols)} columns: {len(cat_cols)} categorical, {len(num_cols)} numeric', flush=True)
    f_tr_basic_num = add_missing_and_basic_numeric_features(trainX, num_cols)
    f_te_basic_num = add_missing_and_basic_numeric_features(testX, num_cols)
    f_tr_cat_ord, f_te_cat_ord = ordinal_encode_from_combined(trainX, testX, origX, cat_cols)
    f_tr_count, f_te_count = add_count_features_from_combined(trainX, testX, origX, cat_cols)
    f_tr_pair, f_te_pair = add_pair_count_features(trainX, testX, origX, cat_cols, max_pair_cols=MAX_CAT_PAIR_COLS)
    f_tr_sig, f_te_sig = add_numeric_signature_count_features(trainX, testX, origX, num_cols, top_k=TOP_NUMERIC_SIGNATURE_COLS)
    Xtr_num, Xte_num, Xorig_num = prepare_numeric_arrays(trainX, testX, origX, num_cols)
    f_tr_knn, f_te_knn = add_knn_to_original_features(Xtr_num, Xte_num, Xorig_num, k_list=KNN_K_LIST)
    f_tr_km, f_te_km = add_kmeans_prototype_features(Xtr_num, Xte_num, Xorig_num, y_orig, n_clusters=KMEANS_CLUSTERS, batch_size=KMEANS_BATCH_SIZE, random_state=RANDOM_STATE)
    te_train = pd.DataFrame(index=train_df.index)
    te_test = pd.DataFrame(index=test_df.index)
    if 'orig_cluster_id' in f_tr_km.columns:
        te_train['te_orig_cluster_id'], te_test['te_orig_cluster_id'] = make_oof_target_encoding_binary(f_tr_km['orig_cluster_id'], f_te_km['orig_cluster_id'], y, folds, smoothing=30.0)
    for c in choose_te_cols(trainX, testX, origX, cat_cols, max_te_cols=MAX_TE_COLS):
        te_train[f'te_{c}'], te_test[f'te_{c}'] = make_oof_target_encoding_binary(trainX[c].astype('string').fillna('__MISSING__'), testX[c].astype('string').fillna('__MISSING__'), y, folds, smoothing=20.0)
    raw_num_train = pd.DataFrame(Xtr_num, columns=[f'{c}__z' for c in num_cols], index=train_df.index) if len(num_cols) else pd.DataFrame(index=train_df.index)
    raw_num_test = pd.DataFrame(Xte_num, columns=[f'{c}__z' for c in num_cols], index=test_df.index) if len(num_cols) else pd.DataFrame(index=test_df.index)
    X_train = pd.concat([raw_num_train, f_tr_cat_ord, f_tr_basic_num, f_tr_count, f_tr_pair, f_tr_sig, f_tr_knn, f_tr_km, te_train], axis=1)
    X_test = pd.concat([raw_num_test, f_te_cat_ord, f_te_basic_num, f_te_count, f_te_pair, f_te_sig, f_te_knn, f_te_km, te_test], axis=1)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(-999.0).astype(np.float32)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-999.0).astype(np.float32)
    return X_train, X_test


def fit_lightgbm_oof(X_train, y, X_test, folds, params):
    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred = np.zeros(len(X_test), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y[va_idx]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
                callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)])
        va_pred = clf.predict_proba(X_va)[:, 1].astype(np.float32)
        te_pred = clf.predict_proba(X_test)[:, 1].astype(np.float32)
        oof[va_idx] = va_pred
        test_pred += te_pred / len(folds)
        print(f'fold {fold}: AUC {roc_auc_score(y_va, va_pred):.6f}, best_iteration {clf.best_iteration_}', flush=True)
        del X_tr, y_tr, X_va, y_va, clf, va_pred, te_pred
        gc.collect()
    return oof, test_pred


def main():
    train, test, orig = read_train(), read_test(), read_original()
    y = to_binary_y(train[TARGET])
    y_orig = to_binary_y(orig[TARGET])
    train[TARGET] = y
    orig[TARGET] = y_orig
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(train), dtype=np.int8), y))

    X_train, X_test = build_all_features(train, test, orig, y, y_orig, folds)
    print(f'X_train {X_train.shape}, X_test {X_test.shape}', flush=True)
    oof_pred, test_pred = fit_lightgbm_oof(X_train, y, X_test, folds, LGB_PARAMS)
    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof_pred):.6f}', flush=True)
    save_pair(HERE, PAIR, np.clip(oof_pred, 1e-7, 1 - 1e-7), np.clip(test_pred, 1e-7, 1 - 1e-7))


if __name__ == '__main__':
    main()
