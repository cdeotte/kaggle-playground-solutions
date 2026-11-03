#!/usr/bin/env python
"""GPU CatBoost with fold-local distribution / density features and CatBoost-native categorical expansions.

Column groups: numerics with < 100 distinct train + test values and object columns are categorical (cast to strings),
the other numerics are continuous; every numeric column is numeric-like.
- Row features (per row): estimated total laps / laps remaining, tyre / stint / pit-stop per lap, lap time / delta /
  degradation per tyre lap, lap time minus delta, |delta|, |position change|, position / pit-window / late-old-tyre /
  degradation pressures; log1p, sqrt, 1 / (1 + x), |x| and string length of every numeric-like column; 4 decimal
  digits and integer mod 10 / 100 of lap time, delta and degradation.
- CatBoost-native categoricals: the raw categorical columns; per numeric-like column the exact value, rounding to
  0 / 1 / 3 decimals, floor, floor to even numbers and a 50-quantile bin; all pairs of 8 keys and 4 triples; a flag
  for original rows.
- Fold-local features, with the outer-fold training rows + all original rows as the reference for the training,
  validation and test rows: frequency, count and percentile rank of every numeric-like value;
  joint frequency / count / PMI of all pairs of 6 numerics; frequency / count / singleton flag of the exact
  (lap time, delta, degradation) triple; frequency of each of 4 numerics given 6 keys and its ratio to the marginal;
  counts of every column with > 7 distinct values; counts and rarity flags of the categorical columns; ball counts
  (radius 0.05 / 0.2 / 0.5) and 5 / 10 / 20-th neighbour distances in the standardized (lap time, delta, degradation,
  progress) space; group means of the 4 numerics per categorical key.
All original rows are appended to the training side of every fold (weight 1). CatBoost: depth 8, lr 0.03, l2 3,
random_strength 0.5, Bayesian bagging_temperature 0.7, border_count 254, one_hot_max_size 4, max_ctr_complexity 2,
5000 iterations, early stopping 300 on the validation fold (use_best_model=True), seed 42.
Test = mean of the 5 fold models (probabilities clipped to [1e-7, 1 - 1e-7]).

Usage: python train_fold_density.py [--folds 0,1,2,3,4] [--force]"""
import gc
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_part  # noqa: E402
from models.catboost.lib.runner import FOLDER, finish, fold_parser, todo_folds  # noqa: E402

PAIR = "catboost_fold_density_knn"
RANDOM_SEED = 42
CAT_CARDINALITY_THRESHOLD = 100
HIGH_CARD_THRESHOLD = 7

CB_PARAMS = dict(
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    random_strength=0.5,
    bagging_temperature=0.7,
    border_count=254,
    grow_policy='SymmetricTree',
    bootstrap_type='Bayesian',
    eval_metric='AUC',
    loss_function='Logloss',
    task_type='GPU',
    devices='0',
    random_seed=RANDOM_SEED,
    verbose=200,
    early_stopping_rounds=300,
    use_best_model=True,
    one_hot_max_size=4,
    max_ctr_complexity=2,
)

ARTIFACT_NUMS = ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation']
KNN_NUMS = ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress']
TOP_CATS_FOR_NGRAM = ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife']


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def safe_name(col):
    return str(col).replace(' ', '').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_').replace('.', '_')


def finite_float32(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')


def sanitize_object_series(s, fill='missing'):
    s = s.astype('object')
    return s.where(pd.notna(s), fill).astype(str)


def align_df_to_columns(df, target_columns, fill_value=0):
    missing = [c for c in target_columns if c not in df.columns]
    for c in missing:
        df[c] = fill_value
    extra = [c for c in df.columns if c not in target_columns]
    if extra:
        df = df.drop(columns=extra)
    return df[target_columns]


def infer_groups(train, test, feature_cols):
    both = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
    cats, nums, numeric_like = [], [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(both[c]):
            numeric_like.append(c)
            if both[c].nunique(dropna=False) < CAT_CARDINALITY_THRESHOLD:
                cats.append(c)
            else:
                nums.append(c)
        else:
            cats.append(c)
    return cats, nums, numeric_like


def cast_base_types(train, test, orig, cats, nums, numeric_like):
    for df in [train, test, orig]:
        for c in numeric_like:
            if c in df.columns:
                df[c] = finite_float32(df[c])
        for c in cats:
            if c in df.columns:
                df[c] = sanitize_object_series(df[c])
    return train, test, orig


def build_safe_features(df, cats, nums, numeric_like):
    feats = {}
    lap = finite_float32(df.get('LapNumber', 0)).to_numpy()
    tyre = finite_float32(df.get('TyreLife', 0)).to_numpy()
    pitstop = finite_float32(df.get('PitStop', 0)).to_numpy()
    stint = finite_float32(df.get('Stint', 0)).to_numpy()
    pos = finite_float32(df.get('Position', 0)).to_numpy()
    pos_chg = finite_float32(df.get('Position_Change', 0)).to_numpy()
    progress = finite_float32(df.get('RaceProgress', 0)).clip(0, 1).to_numpy()
    lap_time = finite_float32(df.get('LapTime (s)', 0)).to_numpy()
    delta = finite_float32(df.get('LapTime_Delta', 0)).to_numpy()
    deg = finite_float32(df.get('Cumulative_Degradation', 0)).to_numpy()

    est_total = np.clip(lap / (progress + 1e-3), 1, 400).astype('float32')
    laps_remaining = np.clip(est_total - lap, -20, 400).astype('float32')

    feats['est_total_laps'] = est_total
    feats['laps_remaining'] = laps_remaining
    feats['tyre_to_lap_ratio'] = (tyre / (lap + 1)).astype('float32')
    feats['tyre_to_remaining_ratio'] = (tyre / (np.abs(laps_remaining) + 1)).astype('float32')
    feats['stint_to_lap_ratio'] = (stint / (lap + 1)).astype('float32')
    feats['pitstop_to_lap_ratio'] = (pitstop / (lap + 1)).astype('float32')
    feats['lap_time_per_tyre'] = (lap_time / (tyre + 1)).astype('float32')
    feats['delta_per_tyre'] = (delta / (tyre + 1)).astype('float32')
    feats['deg_per_tyre'] = (deg / (tyre + 1)).astype('float32')
    feats['lap_time_minus_delta'] = (lap_time - delta).astype('float32')
    feats['abs_delta'] = np.abs(delta).astype('float32')
    feats['abs_position_change'] = np.abs(pos_chg).astype('float32')
    feats['position_pressure'] = (np.clip(21 - pos, 0, 25) + np.clip(pos_chg, -10, 10)).astype('float32')
    feats['pit_window_pressure'] = ((tyre >= 8).astype('float32') * progress * (1 + pitstop)).astype('float32')
    feats['late_old_tyre'] = (progress * np.log1p(np.clip(tyre, 0, None))).astype('float32')
    feats['degradation_pressure'] = (deg * np.log1p(np.clip(tyre, 0, None))).astype('float32')

    for col in numeric_like:
        if col not in df.columns:
            continue
        v = finite_float32(df[col]).to_numpy()
        feats[f'LOG1P_{safe_name(col)}'] = np.log1p(np.clip(v, 0, None)).astype('float32')
        feats[f'SQRT_{safe_name(col)}'] = np.sqrt(np.clip(v, 0, None)).astype('float32')
        feats[f'INV1P_{safe_name(col)}'] = (1.0 / (1.0 + np.clip(v, 0, None))).astype('float32')
        feats[f'ABS_{safe_name(col)}'] = np.abs(v).astype('float32')
        str_vals = pd.Series(v).astype(str)
        feats[f'STRLEN_{safe_name(col)}'] = str_vals.str.len().astype('int8').to_numpy()

    for col in ARTIFACT_NUMS:
        if col not in df.columns:
            continue
        v = finite_float32(df[col]).to_numpy()
        for k in range(1, 5):
            feats[f'DEC{k}_{safe_name(col)}'] = (np.floor(np.abs(v) * (10 ** k)) % 10).astype('int8')
        feats[f'FLOOR_MOD10_{safe_name(col)}'] = (np.floor(np.abs(v)) % 10).astype('float32')
        feats[f'FLOOR_MOD100_{safe_name(col)}'] = (np.floor(np.abs(v)) % 100).astype('float32')

    return {k: np.asarray(v) for k, v in feats.items()}


def build_cat_native(df, cats, numeric_like):
    cat_native = {}
    for c in cats:
        if c in df.columns:
            cat_native[f'RAW_{safe_name(c)}'] = df[c].astype(str).fillna('__NA__').to_numpy()

    for c in numeric_like:
        if c not in df.columns:
            continue
        v = finite_float32(df[c]).to_numpy()
        cat_native[f'NUMCAT_{safe_name(c)}_exact'] = pd.Series(v).astype(str).to_numpy()
        cat_native[f'NUMCAT_{safe_name(c)}_round0'] = np.round(v, 0).astype(str)
        cat_native[f'NUMCAT_{safe_name(c)}_round1'] = np.round(v, 1).astype(str)
        cat_native[f'NUMCAT_{safe_name(c)}_round3'] = np.round(v, 3).astype(str)
        cat_native[f'NUMCAT_{safe_name(c)}_int'] = np.floor(v).astype('int64').astype(str)
        cat_native[f'NUMCAT_{safe_name(c)}_div2'] = (np.floor(v / 2) * 2).astype('int64').astype(str)
        try:
            qbin = pd.qcut(v, q=50, duplicates='drop')
        except Exception:
            qbin = pd.cut(v, bins=50)
        cat_native[f'NUMCAT_{safe_name(c)}_qbin50'] = np.asarray(qbin.astype(str))

    available = [c for c in TOP_CATS_FOR_NGRAM if c in df.columns]
    for c1, c2 in combinations(available, 2):
        cat_native[f'BI_{safe_name(c1)}_{safe_name(c2)}'] = (df[c1].astype(str) + '_' + df[c2].astype(str)).to_numpy()
    for tri in [('Driver', 'Compound', 'Race'), ('Race', 'Compound', 'LapNumber'), ('Driver', 'Compound', 'Stint'), ('Compound', 'TyreLife', 'PitStop')]:
        if all(c in df.columns for c in tri):
            cat_native[f'TRI_{"_".join(safe_name(c) for c in tri)}'] = (df[tri[0]].astype(str) + '_' + df[tri[1]].astype(str) + '_' + df[tri[2]].astype(str)).to_numpy()
    return pd.DataFrame(cat_native, index=df.index)


def build_fold_features(train_fold_df, apply_df, train_fold_y, cats, numeric_like):
    """Features of apply_df measured against the reference rows train_fold_df."""
    feats = {}
    ref_df = train_fold_df.copy()
    n_apply = len(apply_df)

    for col in numeric_like:
        if col not in ref_df.columns:
            continue
        vc = ref_df[col].value_counts(dropna=False)
        freq = ref_df[col].value_counts(normalize=True, dropna=False)
        feats[f'FREQ_{safe_name(col)}'] = apply_df[col].map(freq).fillna(0).astype('float32').to_numpy()
        feats[f'COUNT_{safe_name(col)}'] = apply_df[col].map(vc).fillna(0).astype('float32').to_numpy()
        ref_sorted = np.sort(finite_float32(ref_df[col]).to_numpy())
        apply_vals = finite_float32(apply_df[col]).to_numpy()
        feats[f'RANK_{safe_name(col)}'] = (np.searchsorted(ref_sorted, apply_vals) / max(len(ref_sorted), 1)).astype('float32')

    pair_nums = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'TyreLife', 'LapNumber'] if c in ref_df.columns]
    for c1, c2 in combinations(pair_nums[:6], 2):
        ref_pair = ref_df[c1].astype(str) + '|' + ref_df[c2].astype(str)
        vc_pair = ref_pair.value_counts(normalize=True)
        vc_count = ref_pair.value_counts()
        apply_pair = apply_df[c1].astype(str) + '|' + apply_df[c2].astype(str)
        obs = apply_pair.map(vc_pair).fillna(0).to_numpy()
        f1 = apply_df[c1].map(ref_df[c1].value_counts(normalize=True)).fillna(1e-10).to_numpy()
        f2 = apply_df[c2].map(ref_df[c2].value_counts(normalize=True)).fillna(1e-10).to_numpy()
        feats[f'JOINT_FREQ_{safe_name(c1)}_{safe_name(c2)}'] = obs.astype('float32')
        feats[f'JOINT_COUNT_{safe_name(c1)}_{safe_name(c2)}'] = apply_pair.map(vc_count).fillna(0).astype('float32').to_numpy()
        feats[f'JOINT_PMI_{safe_name(c1)}_{safe_name(c2)}'] = np.log((obs + 1e-10) / (f1 * f2 + 1e-10)).astype('float32')

    if all(c in ref_df.columns for c in ARTIFACT_NUMS):
        ref_triple = ref_df[ARTIFACT_NUMS[0]].round(3).astype(str) + '|' + ref_df[ARTIFACT_NUMS[1]].round(3).astype(str) + '|' + ref_df[ARTIFACT_NUMS[2]].round(3).astype(str)
        app_triple = apply_df[ARTIFACT_NUMS[0]].round(3).astype(str) + '|' + apply_df[ARTIFACT_NUMS[1]].round(3).astype(str) + '|' + apply_df[ARTIFACT_NUMS[2]].round(3).astype(str)
        vc_tri = ref_triple.value_counts(normalize=True)
        cnt_tri = ref_triple.value_counts()
        tri_count = app_triple.map(cnt_tri).fillna(0).astype('float32')
        feats['JOINT_FREQ_TRIPLE_ARTIFACT'] = app_triple.map(vc_tri).fillna(0).astype('float32').to_numpy()
        feats['JOINT_COUNT_TRIPLE_ARTIFACT'] = tri_count.to_numpy()
        feats['IS_SINGLETON_TRIPLE_ARTIFACT'] = (tri_count <= 1).astype('float32').to_numpy()

    for num_col in [c for c in ARTIFACT_NUMS + ['RaceProgress'] if c in ref_df.columns]:
        marginal = apply_df[num_col].map(ref_df[num_col].value_counts(normalize=True)).fillna(0).to_numpy()
        for cat_col in [c for c in ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'Stint'] if c in ref_df.columns]:
            ref_key = ref_df[cat_col].astype(str) + '|' + ref_df[num_col].round(3).astype(str)
            vc = ref_key.value_counts(normalize=True)
            apply_key = apply_df[cat_col].astype(str) + '|' + apply_df[num_col].round(3).astype(str)
            cond = apply_key.map(vc).fillna(0).to_numpy()
            feats[f'CONDF_{safe_name(num_col)}_given_{safe_name(cat_col)}'] = cond.astype('float32')
            feats[f'CONDF_ratio_{safe_name(num_col)}_given_{safe_name(cat_col)}'] = (cond / (marginal + 1e-10)).astype('float32')

    for col in list(apply_df.columns):
        if col == ID or col not in ref_df.columns:
            continue
        if ref_df[col].nunique(dropna=False) > HIGH_CARD_THRESHOLD:
            vc = ref_df[col].value_counts(dropna=False)
            feats[f'count({safe_name(col)})'] = apply_df[col].map(vc).fillna(0).astype('float32').to_numpy()

    for c in cats:
        if c not in ref_df.columns:
            continue
        vc = sanitize_object_series(ref_df[c]).value_counts(dropna=False)
        cnt = sanitize_object_series(apply_df[c]).map(vc).fillna(0).astype('float32')
        feats[f'CAT_CNT_{safe_name(c)}'] = cnt.to_numpy()
        feats[f'CAT_RARE_{safe_name(c)}'] = (cnt <= 50).astype('float32').to_numpy()

    knn_cols = [c for c in KNN_NUMS if c in ref_df.columns]
    if len(knn_cols) >= 2 and len(ref_df) > 25:
        ref_mat = np.column_stack([finite_float32(ref_df[c]).to_numpy() for c in knn_cols]).astype('float64')
        app_mat = np.column_stack([finite_float32(apply_df[c]).to_numpy() for c in knn_cols]).astype('float64')
        mean = ref_mat.mean(axis=0)
        std = ref_mat.std(axis=0) + 1e-8
        ref_pts = (ref_mat - mean) / std
        app_pts = (app_mat - mean) / std
        tree = cKDTree(ref_pts)
        for radius_label, radius in [('tight', 0.05), ('medium', 0.20), ('wide', 0.50)]:
            feats[f'DENSITY_{radius_label}'] = np.array(tree.query_ball_point(app_pts, r=radius, workers=1, return_length=True), dtype='float32')
        k_actual = min(21, len(ref_pts))
        nn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', n_jobs=8)
        nn.fit(ref_pts)
        dists, _ = nn.kneighbors(app_pts)
        for k in [5, 10, 20]:
            feats[f'KNN_dist_{k}'] = dists[:, k].astype('float32') if k < dists.shape[1] else np.full(n_apply, np.nan, dtype='float32')
        feats['KNN_mean_dist_5'] = dists[:, 1:min(6, dists.shape[1])].mean(axis=1).astype('float32')
        feats['KNN_mean_dist_20'] = dists[:, 1:min(21, dists.shape[1])].mean(axis=1).astype('float32')
        del tree, nn, dists

    # group means of lap time, delta, degradation and progress per key
    ref_with_y = ref_df.copy()
    ref_with_y[TARGET] = np.asarray(train_fold_y, dtype=np.float32)
    for g in [c for c in cats if c in ref_with_y.columns] + [c for c in ['Race', 'Compound', 'Driver', 'Year', 'PitStop', 'Stint', 'TyreLife'] if c in ref_with_y.columns]:
        for num_col in [c for c in ARTIFACT_NUMS + ['RaceProgress'] if c in ref_with_y.columns]:
            m = ref_with_y.groupby(g, dropna=False)[num_col].mean()
            fillv = float(np.nanmean(finite_float32(ref_with_y[num_col]).to_numpy()))
            feats[f'B_MEAN_{safe_name(num_col)}_BY_{safe_name(g)}'] = apply_df[g].map(m).fillna(fillv).astype('float32').to_numpy()

    return feats


def main():
    from catboost import CatBoostClassifier, Pool

    args = fold_parser().parse_args()
    todo = todo_folds(PAIR, args.folds, args.force)
    if todo:
        seed_everything(RANDOM_SEED)
        train_raw, test_raw, orig_raw = read_train(), read_test(), read_original()
        train_raw[TARGET] = train_raw[TARGET].astype('int8')
        orig_raw[TARGET] = orig_raw[TARGET].astype('int8')
        feature_cols = [c for c in train_raw.columns if c not in [ID, TARGET]]
        cats, nums, numeric_like = infer_groups(train_raw, test_raw, feature_cols)
        train_raw, test_raw, orig_raw = cast_base_types(train_raw, test_raw, orig_raw, cats, nums, numeric_like)
        print(f'cats={cats}\nnums={nums}', flush=True)

        y_train = train_raw[TARGET].to_numpy(dtype=np.int8)
        y_orig = orig_raw[TARGET].to_numpy(dtype=np.int8)
        train_base = train_raw.drop(columns=[TARGET, ID], errors='ignore').copy()
        test_base = test_raw.drop(columns=[ID], errors='ignore').copy()
        orig_base = orig_raw.drop(columns=[TARGET, ID], errors='ignore').copy()

        train_safe = pd.DataFrame(build_safe_features(train_base, cats, nums, numeric_like))
        test_safe = pd.DataFrame(build_safe_features(test_base, cats, nums, numeric_like))
        orig_safe = pd.DataFrame(build_safe_features(orig_base, cats, nums, numeric_like))
        all_safe_cols = sorted(set(train_safe.columns) | set(test_safe.columns) | set(orig_safe.columns))
        train_safe = align_df_to_columns(train_safe, all_safe_cols, 0)
        test_safe = align_df_to_columns(test_safe, all_safe_cols, 0)
        orig_safe = align_df_to_columns(orig_safe, all_safe_cols, 0)

        train_cat = build_cat_native(train_base, cats, numeric_like)
        test_cat = build_cat_native(test_base, cats, numeric_like)
        orig_cat = build_cat_native(orig_base, cats, numeric_like)
        all_cat_cols = sorted(set(train_cat.columns) | set(test_cat.columns) | set(orig_cat.columns))
        train_cat = align_df_to_columns(train_cat, all_cat_cols, '__NA__')
        test_cat = align_df_to_columns(test_cat, all_cat_cols, '__NA__')
        orig_cat = align_df_to_columns(orig_cat, all_cat_cols, '__NA__')

        fold_id = folds(y_train)
        canonical_fold_features = None
        for k in todo:
            train_idx, val_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            fold_val_base = train_base.iloc[val_idx].reset_index(drop=True)
            fold_val_y = y_train[val_idx]
            fold_train_base = train_base.iloc[train_idx].reset_index(drop=True)
            # reference and training side: the fold's training rows + every original row
            fold_train_full = pd.concat([fold_train_base, orig_base], axis=0, ignore_index=True)
            fold_train_y_full = np.concatenate([y_train[train_idx], y_orig])

            print(f'{PAIR} fold {k}: fold features with reference size {len(fold_train_full)}', flush=True)
            fold_feats_train = pd.DataFrame(build_fold_features(fold_train_full, fold_train_full, fold_train_y_full, cats, numeric_like))
            fold_feats_val = pd.DataFrame(build_fold_features(fold_train_full, fold_val_base, fold_train_y_full, cats, numeric_like))
            fold_feats_test = pd.DataFrame(build_fold_features(fold_train_full, test_base, fold_train_y_full, cats, numeric_like))

            all_fold_cols = sorted(set(fold_feats_train.columns) | set(fold_feats_val.columns) | set(fold_feats_test.columns))
            canonical_fold_features = all_fold_cols if canonical_fold_features is None else sorted(set(canonical_fold_features) | set(all_fold_cols))
            fold_feats_train = align_df_to_columns(fold_feats_train, canonical_fold_features, 0)
            fold_feats_val = align_df_to_columns(fold_feats_val, canonical_fold_features, 0)
            fold_feats_test = align_df_to_columns(fold_feats_test, canonical_fold_features, 0)

            train_safe_full = pd.concat([train_safe.iloc[train_idx].reset_index(drop=True), orig_safe.reset_index(drop=True)], axis=0, ignore_index=True)
            train_cat_full = pd.concat([train_cat.iloc[train_idx].reset_index(drop=True), orig_cat.reset_index(drop=True)], axis=0, ignore_index=True)
            train_cat_full['_is_original_'] = np.r_[np.zeros(len(train_idx), dtype=np.int8), np.ones(len(orig_cat), dtype=np.int8)].astype(str)
            val_cat = train_cat.iloc[val_idx].reset_index(drop=True).copy()
            test_cat_fold = test_cat.reset_index(drop=True).copy()
            val_cat['_is_original_'] = '0'
            test_cat_fold['_is_original_'] = '0'

            cat_features = list(all_cat_cols) + ['_is_original_']
            safe_num_features = list(all_safe_cols)
            fold_num_features = list(canonical_fold_features)
            all_features = safe_num_features + fold_num_features + cat_features

            X_tr = pd.concat([train_safe_full, fold_feats_train, train_cat_full], axis=1)
            X_val = pd.concat([train_safe.iloc[val_idx].reset_index(drop=True), fold_feats_val, val_cat], axis=1)
            X_te = pd.concat([test_safe.reset_index(drop=True), fold_feats_test, test_cat_fold], axis=1)
            X_tr = align_df_to_columns(X_tr, all_features, 0)
            X_val = align_df_to_columns(X_val, all_features, 0)
            X_te = align_df_to_columns(X_te, all_features, 0)

            for c in cat_features:
                X_tr[c] = X_tr[c].astype(str).fillna('__NA__')
                X_val[c] = X_val[c].astype(str).fillna('__NA__')
                X_te[c] = X_te[c].astype(str).fillna('__NA__')
            for c in safe_num_features + fold_num_features:
                X_tr[c] = finite_float32(X_tr[c])
                X_val[c] = finite_float32(X_val[c])
                X_te[c] = finite_float32(X_te[c])

            cat_indices = [all_features.index(c) for c in cat_features]
            print(f'{PAIR} fold {k}: {len(all_features)} features (row {len(safe_num_features)}, '
                  f'fold {len(fold_num_features)}, categorical {len(cat_features)})', flush=True)

            train_pool = Pool(X_tr, label=fold_train_y_full, cat_features=cat_indices, feature_names=all_features)
            val_pool = Pool(X_val, label=fold_val_y, cat_features=cat_indices, feature_names=all_features)
            test_pool = Pool(X_te, cat_features=cat_indices, feature_names=all_features)
            model = CatBoostClassifier(**CB_PARAMS)
            model.fit(train_pool, eval_set=val_pool)
            val_pred = model.predict_proba(val_pool)[:, 1].astype(np.float32)
            test_pred = model.predict_proba(test_pool)[:, 1].astype(np.float32)
            save_part(FOLDER, PAIR, k, val_pred, test_pred)
            print(f'{PAIR} fold {k}: AUC {roc_auc_score(fold_val_y, val_pred):.6f} best_iter {model.get_best_iteration()}', flush=True)

            del X_tr, X_val, X_te, train_pool, val_pool, test_pool, fold_feats_train, fold_feats_val, fold_feats_test, model
            gc.collect()
    finish(PAIR, clip=1e-7)


if __name__ == '__main__':
    main()
