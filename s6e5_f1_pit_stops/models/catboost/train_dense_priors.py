#!/usr/bin/env python
"""GPU CatBoost (native categorical crosses, max_ctr_complexity 4) on a dense frame with original-data priors.

Features, built once on train + test + original (514 columns, 210 categorical):
- the raw columns (low-cardinality numerics also as categories), 18 race-state relationships (tyre / lap / stint /
  pit-stop ratios, degradation and delta ratios, lap time minus delta, products with progress, position pressure,
  a pit-window score, years since the first year);
- log1p, sqrt, 1 / (1 + |x|), percentile rank and value frequency of up to 36 numerics; rounding to 0 / 1 / 2 decimals,
  floor, fractional part and 5 digit categories of 6 numerics and 2 relationships;
- 12-quantile bins of 10 numerics, 12 pair and 4 triple categorical crosses;
- for the first 80 categorical candidates: log count over train + test + original and log1p of the ratio of their
  train frequency to their original frequency; for the first 50: smoothed original-label mean (smoothing 25) and log
  original count;
- distances to the two original class centroids (10 standardized numerics) and the softmax of their negatives.
The original rows are used only for these priors (they are not appended to the training side). CatBoost: depth 8,
lr 0.02, l2 6, min_data_in_leaf 20, Bernoulli subsample 0.85, random_strength 0.6, max_ctr_complexity 4, up to
50,000 iterations, early stopping 600 on the validation fold (use_best_model=True), seed 42 + fold number (1-5).
Test = mean of the 5 fold models (probabilities).

Usage: python train_dense_priors.py [--folds 0,1,2,3,4] [--force]"""
import gc
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_part  # noqa: E402
from models.catboost.lib.runner import FOLDER, finish, fold_parser, todo_folds  # noqa: E402

PAIR = 'catboost_dense_priors_crosses'
RANDOM_SEED = 42

CB_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 50000,
    'learning_rate': 0.02,
    'depth': 8,
    'l2_leaf_reg': 6.0,
    'min_data_in_leaf': 20,
    'task_type': 'GPU',
    'devices': '0',
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.85,
    'random_strength': 0.6,
    'grow_policy': 'SymmetricTree',
    'max_ctr_complexity': 4,
    'verbose': 250,
}


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def safe_name(c):
    return ''.join(ch if ch.isalnum() else '_' for ch in str(c)).strip('_')


def dedup_keep_order(seq):
    out = []
    seen = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def finite_float32(s):
    arr = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    return arr


def infer_groups(train, test, feature_cols):
    cats, nums, numeric_like = [], [], []
    both = pd.concat([train[feature_cols], test[feature_cols]], axis=0, ignore_index=True)
    for c in feature_cols:
        if pd.api.types.is_object_dtype(both[c]) or str(both[c].dtype).startswith('string') or str(both[c].dtype) == 'category':
            cats.append(c)
        else:
            numeric_like.append(c)
            nun = both[c].nunique(dropna=False)
            if nun < 100:
                cats.append(c)
            else:
                nums.append(c)
    return cats, nums, numeric_like


def qcut_labels(s, q=12):
    s = pd.to_numeric(s, errors='coerce').astype('float32')
    try:
        return pd.qcut(s.rank(method='first'), q=q, labels=False, duplicates='drop').astype('float32').fillna(-1).astype('int16').astype(str)
    except Exception:
        return pd.Series(['-1'] * len(s), index=s.index, dtype='string')


def add_round_digit_features(all_df, cols):
    out = pd.DataFrame(index=all_df.index)
    for c in cols:
        if c not in all_df.columns:
            continue
        sn = safe_name(c)
        s = finite_float32(all_df[c])
        out[f'ROUND0_{sn}'] = s.round(0).astype('float32')
        out[f'ROUND1_{sn}'] = s.round(1).astype('float32')
        out[f'ROUND2_{sn}'] = s.round(2).astype('float32')
        out[f'FLOOR_{sn}'] = np.floor(s).astype('float32')
        out[f'FRAC_{sn}'] = (s - np.floor(s)).astype('float32')
        sf = s.map(lambda x: f'{float(x):.3f}')
        parts = sf.str.split('.', n=1, expand=True)
        left = parts[0].str.replace('-', '', regex=False).str.zfill(5)
        right = parts[1].str.pad(3, side='right', fillchar='0')
        out[f'DIG_{sn}_i0'] = left.str[-1].astype('string')
        out[f'DIG_{sn}_i1'] = left.str[-2].astype('string')
        out[f'DIG_{sn}_f0'] = right.str[0].astype('string')
        out[f'DIG_{sn}_f1'] = right.str[1].astype('string')
        out[f'DIG_{sn}_bucket'] = (left.str[-2:] + '_' + right.str[:1]).astype('string')
    return out


def add_numeric_transforms(all_df, cols):
    out = pd.DataFrame(index=all_df.index)
    for c in cols:
        if c not in all_df.columns:
            continue
        sn = safe_name(c)
        v = finite_float32(all_df[c])
        out[f'LOG1P_{sn}'] = np.log1p(v.clip(lower=0)).astype('float32')
        out[f'SQRT_{sn}'] = np.sqrt(v.clip(lower=0)).astype('float32')
        out[f'INV1P_{sn}'] = (1.0 / (1.0 + np.abs(v))).astype('float32')
        out[f'RANK_{sn}'] = v.rank(method='average', pct=True).astype('float32')
        vc = v.value_counts(normalize=True, dropna=False)
        out[f'FREQ_{sn}'] = v.map(vc).fillna(0).astype('float32')
    return out


def add_f1_relationships(all_df):
    def num(c):
        return finite_float32(all_df[c]) if c in all_df.columns else pd.Series(0, index=all_df.index, dtype='float32')
    lap = num('LapNumber')
    tyre = num('TyreLife')
    stint = num('Stint')
    pit = num('PitStop')
    pos = num('Position')
    poschg = num('Position_Change')
    lt = num('LapTime (s)')
    delta = num('LapTime_Delta')
    degr = num('Cumulative_Degradation')
    prog = num('RaceProgress')
    year = num('Year')
    out = pd.DataFrame(index=all_df.index)
    out['tyre_per_lap'] = (tyre / (lap + 1)).astype('float32')
    out['lap_per_stint'] = (lap / (stint + 1)).astype('float32')
    out['tyre_per_stint'] = (tyre / (stint + 1)).astype('float32')
    out['pit_adjusted_tyre'] = (tyre / (pit + 1)).astype('float32')
    out['degr_per_tyre'] = (degr / (tyre + 1)).astype('float32')
    out['degr_per_lap'] = (degr / (lap + 1)).astype('float32')
    out['delta_per_laptime'] = (delta / (np.abs(lt) + 1)).astype('float32')
    out['delta_per_degr'] = (delta / (np.abs(degr) + 1)).astype('float32')
    out['laptime_per_tyre'] = (lt / (tyre + 1)).astype('float32')
    out['laptime_minus_delta'] = (lt - delta).astype('float32')
    out['degr_x_progress'] = (degr * prog).astype('float32')
    out['tyre_x_progress'] = (tyre * prog).astype('float32')
    out['laptime_x_progress'] = (lt * prog).astype('float32')
    out['position_pressure'] = (pos + poschg.abs()).astype('float32')
    out['position_gain_risk'] = (-poschg / (pos + 1)).astype('float32')
    out['late_old_tyre'] = (prog * tyre).astype('float32')
    out['pit_window_proxy'] = ((tyre > 12).astype('float32') + (prog > 0.45).astype('float32') + (degr > degr.median()).astype('float32')).astype('float32')
    out['modern_year'] = (year - year.min()).astype('float32') if len(year) else 0
    return out


def build_feature_frames(train_x, test_x, orig_x, y_orig, base_cats, base_nums, numeric_like):
    n_tr, n_te, n_or = len(train_x), len(test_x), len(orig_x)
    all_df = pd.concat([train_x, test_x, orig_x], axis=0, ignore_index=True)
    for c in base_cats:
        if c in all_df.columns:
            all_df[c] = all_df[c].astype(str).fillna('NA')
    for c in numeric_like:
        if c in all_df.columns:
            all_df[c] = finite_float32(all_df[c])

    blocks = [all_df.copy()]
    artifact_nums = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'TyreLife', 'LapNumber'] if c in all_df.columns]
    rel = add_f1_relationships(all_df)
    blocks.append(rel)
    aug = pd.concat([all_df, rel], axis=1)
    trans_cols = dedup_keep_order(base_nums + artifact_nums + list(rel.columns))[:36]
    blocks.append(add_numeric_transforms(aug, trans_cols))
    blocks.append(add_round_digit_features(aug, artifact_nums + ['degr_per_tyre', 'late_old_tyre']))

    bin_cols = dedup_keep_order(base_nums + ['TyreLife', 'LapNumber', 'Position', 'Position_Change', 'degr_per_tyre', 'late_old_tyre'])
    bins = pd.DataFrame(index=all_df.index)
    for c in bin_cols:
        if c in aug.columns:
            bins[f'{safe_name(c)}_bin'] = qcut_labels(aug[c], q=12).astype('string')
    blocks.append(bins)

    cat_src = pd.concat([all_df, bins], axis=1)
    pair_specs = [
        ('Driver', 'Race'), ('Driver', 'Compound'), ('Race', 'Compound'), ('Year', 'Race'),
        ('Stint', 'Compound'), ('PitStop', 'Stint'), ('Position', 'Position_Change'),
        ('TyreLife_bin', 'Compound'), ('LapNumber_bin', 'Driver'), ('RaceProgress_bin', 'Compound'),
        ('Cumulative_Degradation_bin', 'Compound'), ('LapTime_s_bin', 'Race'), ('degr_per_tyre_bin', 'Compound'),
    ]
    # the lap-time bin column is named 'LapTime__s_bin' (safe_name), so the ('LapTime_s_bin', 'Race') pair is skipped
    crosses = pd.DataFrame(index=all_df.index)
    for a, b in pair_specs:
        if a in cat_src.columns and b in cat_src.columns:
            crosses[f'{a}__{b}'] = cat_src[a].astype(str).fillna('NA') + '__' + cat_src[b].astype(str).fillna('NA')
    triple_specs = [('Driver', 'Race', 'Compound'), ('Year', 'Race', 'Driver'), ('Race', 'Compound', 'Stint'), ('Compound', 'TyreLife_bin', 'RaceProgress_bin')]
    for a, b, c in triple_specs:
        if a in cat_src.columns and b in cat_src.columns and c in cat_src.columns:
            crosses[f'{a}__{b}__{c}'] = cat_src[a].astype(str).fillna('NA') + '__' + cat_src[b].astype(str).fillna('NA') + '__' + cat_src[c].astype(str).fillna('NA')
    blocks.append(crosses)

    all_feat = pd.concat(blocks, axis=1)
    all_feat = all_feat.loc[:, ~all_feat.columns.duplicated()].copy()

    train_feat = all_feat.iloc[:n_tr].reset_index(drop=True).copy()
    test_feat = all_feat.iloc[n_tr:n_tr + n_te].reset_index(drop=True).copy()
    orig_feat = all_feat.iloc[n_tr + n_te:n_tr + n_te + n_or].reset_index(drop=True).copy()

    # counts over train + test + original and train-vs-original frequency drift
    cat_candidates = dedup_keep_order(base_cats + list(bins.columns) + list(crosses.columns) + [c for c in all_feat.columns if c.startswith('DIG_')])
    count_cols = cat_candidates[:80]
    for c in count_cols:
        vc_all = all_feat[c].astype(str).value_counts(dropna=False)
        train_feat[f'CNT_{safe_name(c)}'] = np.log1p(train_feat[c].astype(str).map(vc_all).fillna(0)).astype('float32')
        test_feat[f'CNT_{safe_name(c)}'] = np.log1p(test_feat[c].astype(str).map(vc_all).fillna(0)).astype('float32')
        orig_feat[f'CNT_{safe_name(c)}'] = np.log1p(orig_feat[c].astype(str).map(vc_all).fillna(0)).astype('float32')
        vc_tr = train_feat[c].astype(str).value_counts(normalize=True, dropna=False)
        vc_or = orig_feat[c].astype(str).value_counts(normalize=True, dropna=False)
        denom_tr = train_feat[c].astype(str).map(vc_or).fillna(1e-6) + 1e-6
        denom_te = test_feat[c].astype(str).map(vc_or).fillna(1e-6) + 1e-6
        denom_or = orig_feat[c].astype(str).map(vc_or).fillna(1e-6) + 1e-6
        train_feat[f'DRIFT_{safe_name(c)}'] = np.log1p(train_feat[c].astype(str).map(vc_tr).fillna(0) / denom_tr).astype('float32')
        test_feat[f'DRIFT_{safe_name(c)}'] = np.log1p(test_feat[c].astype(str).map(vc_tr).fillna(0) / denom_te).astype('float32')
        orig_feat[f'DRIFT_{safe_name(c)}'] = np.log1p(orig_feat[c].astype(str).map(vc_tr).fillna(0) / denom_or).astype('float32')

    # smoothed original-label priors
    global_mean = float(np.mean(y_orig)) if y_orig is not None and len(y_orig) else 0.5
    for c in count_cols[:50]:
        stat = pd.DataFrame({'key': orig_feat[c].astype(str), 'y': y_orig}).groupby('key')['y'].agg(['mean', 'count'])
        smooth = (stat['mean'] * stat['count'] + global_mean * 25.0) / (stat['count'] + 25.0)
        for df in (train_feat, test_feat, orig_feat):
            df[f'ORIG_mean_{safe_name(c)}'] = df[c].astype(str).map(smooth).fillna(global_mean).astype('float32')
            df[f'ORIG_logcnt_{safe_name(c)}'] = np.log1p(df[c].astype(str).map(stat['count']).fillna(0)).astype('float32')

    # distances to the original class centroids
    aux_nums = [c for c in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'TyreLife', 'LapNumber', 'Stint', 'PitStop', 'Position', 'Position_Change'] if c in orig_feat.columns]
    if aux_nums and y_orig is not None and len(y_orig):
        ref = orig_feat[aux_nums].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
        sd = ref.std().replace(0, 1).fillna(1).values.astype('float32')
        mus = []
        for cls in [0, 1]:
            if np.any(y_orig == cls):
                mus.append(ref.loc[y_orig == cls, aux_nums].mean().values.astype('float32'))
            else:
                mus.append(np.zeros(len(aux_nums), dtype='float32'))
        for df in (train_feat, test_feat, orig_feat):
            vals = df[aux_nums].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32').values
            d0 = np.sqrt((((vals - mus[0]) / sd) ** 2).sum(axis=1))
            d1 = np.sqrt((((vals - mus[1]) / sd) ** 2).sum(axis=1))
            logits = np.column_stack([-d0, -d1]).astype('float32')
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            df['AUX_dist0'] = d0.astype('float32')
            df['AUX_dist1'] = d1.astype('float32')
            df['AUX_prob_like_1'] = probs[:, 1].astype('float32')
            df['AUX_gap10'] = (d1 - d0).astype('float32')

    train_feat = train_feat.loc[:, ~train_feat.columns.duplicated()].copy()
    test_feat = test_feat.loc[:, train_feat.columns].copy()
    orig_feat = orig_feat.loc[:, train_feat.columns].copy()
    cat_candidates = [c for c in cat_candidates if c in train_feat.columns]
    return train_feat, test_feat, orig_feat, cat_candidates


def main():
    from catboost import CatBoostClassifier

    args = fold_parser().parse_args()
    todo = todo_folds(PAIR, args.folds, args.force)
    if todo:
        seed_everything(RANDOM_SEED)
        train, test, orig = read_train(), read_test(), read_original()
        y = train[TARGET].to_numpy().astype(np.int8)
        y_orig = orig[TARGET].to_numpy().astype(np.int8)
        feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
        base_cats, base_nums, numeric_like = infer_groups(train, test, feature_cols)
        base_cats = dedup_keep_order([c for c in ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position', 'Position_Change'] if c in feature_cols] + base_cats)
        base_nums = [c for c in feature_cols if c in base_nums]
        for df in (train, test, orig):
            for c in numeric_like:
                if c in df.columns:
                    df[c] = finite_float32(df[c])
            for c in base_cats:
                if c in df.columns:
                    df[c] = df[c].astype(str).fillna('NA')

        train_fe, test_fe, orig_fe, cat_candidates = build_feature_frames(
            train[feature_cols].copy(), test[feature_cols].copy(), orig[feature_cols].copy(), y_orig, base_cats, base_nums, numeric_like
        )
        del orig_fe
        keep_cols = [c for c in dict.fromkeys(train_fe.columns) if c != TARGET]
        train_fe = train_fe.loc[:, keep_cols].copy()
        test_fe = test_fe.loc[:, keep_cols].copy()

        use_cols = list(train_fe.columns)
        cat_cols = []
        for c in use_cols:
            dt = str(train_fe[c].dtype)
            if c in cat_candidates or c in base_cats or c.endswith('_bin') or c.startswith('DIG_') or '__' in c or dt in ('object', 'string', 'category'):
                cat_cols.append(c)
        cat_cols = dedup_keep_order([c for c in cat_cols if c in use_cols])
        num_cols = [c for c in use_cols if c not in cat_cols]
        for df in (train_fe, test_fe):
            for c in cat_cols:
                df[c] = df[c].astype(str).fillna('NA')
            for c in num_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
        cat_idx = [train_fe.columns.get_loc(c) for c in cat_cols]
        print(f'feature frames: train={train_fe.shape}, test={test_fe.shape}, cat_cols={len(cat_cols)}, num_cols={len(num_cols)}', flush=True)

        fold_id = folds(y)
        for k in todo:
            tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            X_tr = train_fe.iloc[tr_idx].copy()
            X_va = train_fe.iloc[va_idx].copy()
            y_tr = y[tr_idx]
            y_va = y[va_idx]
            params = dict(CB_PARAMS)
            params['random_seed'] = RANDOM_SEED + k + 1
            model = CatBoostClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_idx,
                eval_set=(X_va, y_va),
                use_best_model=True,
                early_stopping_rounds=600,
                verbose=250,
            )
            va_pred = model.predict_proba(X_va)[:, 1].astype('float32')
            te_pred = model.predict_proba(test_fe)[:, 1].astype('float32')
            save_part(FOLDER, PAIR, k, va_pred, te_pred)
            print(f'{PAIR} fold {k}: AUC {roc_auc_score(y_va, va_pred):.6f} best_iter {model.get_best_iteration()}', flush=True)
            del X_tr, X_va, model
            gc.collect()
    finish(PAIR)


if __name__ == '__main__':
    main()
