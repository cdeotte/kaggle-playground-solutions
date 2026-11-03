#!/usr/bin/env python
"""Three tuned RealMLP configurations per fold on original-data rank / frequency / probability / drift features.

Features (computed once on the train, test and original rows; numerics are float32, categories strings):
- the 14 columns; Driver, Compound, Race and the 7 numerics with < 100 distinct values (Year, PitStop, LapNumber, Stint,
  TyreLife, Position, Position_Change) become string categories;
- 16 race-state numerics (estimated total laps, laps remaining, tyre / stint / pit-stop ratios, lap time, delta and
  degradation per tyre lap, lap time - delta, |delta|, |position change|, position, pit-window and degradation pressure,
  late-race old tyre) and RacePhase / TyreAgePhase / PitWindow categories;
- relative frequency and percentile rank over train + test + original of the 4 continuous columns and the 16 state
  numerics;
- log1p, sqrt, 1 / (1 + x), |x|, values rounded to 1 / 2 / 3 decimals and floor(|x|) mod 100 of LapTime, LapTime_Delta,
  Cumulative_Degradation and RaceProgress;
- 64 and 256 quantile bins of those 4 columns, 7 pair keys and 3 triple keys (categories; 31 categories in all);
- per category: log count over train + test + original and a rare flag (count <= 50), smoothed original target rate
  (m = 25) and the original log count, log1p(train frequency / original frequency) (drift). 278 columns.
Training rows = the fold's train rows + all original rows, plus an is-original category. Each fold trains the three
configurations (pytabkit RealMLP_TD_Classifier, n_ens 8, up to 32 epochs, batch 1024; the first two with early stopping
on the validation 1-AUC; seed 42 + 100 fold + config index, fold 1-based) and averages their probabilities.
Usage: python train_three_configs.py"""
import gc
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, N_FOLDS, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.simplefilter('ignore')
pd.options.mode.chained_assignment = None

PAIR = 'realmlp_3config_rank_freq_drift'
SEED = 42
CAT_CARDINALITY_THRESHOLD = 100
DRIFT_EPS = 1e-6

BASE_CAT_CANDIDATES = ['Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position', 'Position_Change']
BASE_NUMS = ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress']
ARTIFACT_NUMS = ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation']
CROSS_PAIRS = [
    ('Driver', 'Compound'),
    ('Race', 'Compound'),
    ('Driver', 'Race'),
    ('Compound', 'TyreLife'),
    ('Race', 'LapNumber'),
    ('Driver', 'Stint'),
    ('Compound', 'PitStop'),
]
TRIPLES = [('Driver', 'Compound', 'Race'), ('Race', 'Compound', 'LapNumber'), ('Driver', 'Compound', 'Stint')]

COMMON_FIXED_PARAMS = {
    'act': 'mish',
    'ls_eps_sched': 'coslog4',
    'p_drop_sched': 'flat_cos',
    'wd_sched': 'flat_cos',
    'embedding_size': 8,
    'bias_wd_factor': 0.0,
    'weight_param': 'ntk',
    'bias_lr_factor': 0.1,
    'use_parametric_act': True,
    'add_front_scale': True,
    'act_lr_factor': 0.1,
    'block_str': 'w-b-a-d',
    'max_one_hot_cat_size': 9,
    'hidden_sizes': 'rectangular',
    'bias_init_mode': 'he+5',
    'weight_init_mode': 'std',
    'val_metric_name': '1-auc_ovr',
    'num_emb_type': 'pbld',
    'verbosity': 2,
    'batch_size': 1024,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_state': 42,
    'n_epochs': 32,
    'early_stopping_additive_patience': 40,
    'early_stopping_multiplicative_patience': 3,
    'n_ens': 8,
    'ens_av_before_softmax': False,
}

THREE_CONFIGS = [
    {**COMMON_FIXED_PARAMS, 'n_hidden_layers': 4, 'hidden_width': 512, 'embedding_size': 4, 'lr': 0.1301052035689966, 'wd': 0.04914969178803988, 'p_drop': 0.02682248660489578, 'scale_lr_factor': 2.1896500647074553, 'use_ls': False, 'ls_eps': 0.04088400565558735, 'plr_sigma': 1.668950443527603, 'plr_lr_factor': 0.27098419375438565, 'plr_hidden_1': 8, 'plr_hidden_2': 8, 'sq_mom': 0.9855686567493073, 'first_layer_lr_factor': 1.3750227365258307, 'use_early_stopping': True, 'max_one_hot_cat_size': 15},
    {**COMMON_FIXED_PARAMS, 'n_hidden_layers': 2, 'hidden_width': 512, 'embedding_size': 8, 'lr': 0.03865822397961398, 'wd': 0.00851274626094165, 'p_drop': 0.2739396730231609, 'scale_lr_factor': 3.6178064047128555, 'use_ls': False, 'ls_eps': 0.012194703961296842, 'plr_sigma': 1.3478060390217743, 'plr_lr_factor': 0.10119556825550731, 'plr_hidden_1': 8, 'plr_hidden_2': 8, 'sq_mom': 0.9865749129374782, 'first_layer_lr_factor': 0.6175393224001581, 'use_early_stopping': True, 'max_one_hot_cat_size': 8},
    {**COMMON_FIXED_PARAMS, 'n_hidden_layers': 2, 'hidden_width': 384, 'embedding_size': 16, 'lr': 0.029186039688349992, 'wd': 0.009702038034899564, 'p_drop': 0.028921980435491934, 'scale_lr_factor': 5.433094297291475, 'use_ls': False, 'ls_eps': 0.035742411185025844, 'plr_sigma': 48.89920447537555, 'plr_lr_factor': 0.21112939343489906, 'plr_hidden_1': 16, 'plr_hidden_2': 4, 'sq_mom': 0.9947345074902412, 'first_layer_lr_factor': 1.0025819443139303, 'use_early_stopping': False, 'max_one_hot_cat_size': 4},
]


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dedup_keep_order(seq):
    return list(dict.fromkeys(seq))


def safe_name(col):
    return str(col).replace(' ', '').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_').replace('.', '_')


def finite_float32(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')


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


def add_state_features(df):
    df = df.copy()
    lap = finite_float32(df.get('LapNumber', 0))
    tyre = finite_float32(df.get('TyreLife', 0))
    pitstop = finite_float32(df.get('PitStop', 0))
    stint = finite_float32(df.get('Stint', 0))
    pos = finite_float32(df.get('Position', 0))
    pos_chg = finite_float32(df.get('Position_Change', 0))
    progress = finite_float32(df.get('RaceProgress', 0)).clip(0, 1)
    lap_time = finite_float32(df.get('LapTime (s)', 0))
    delta = finite_float32(df.get('LapTime_Delta', 0))
    deg = finite_float32(df.get('Cumulative_Degradation', 0))
    est_total = (lap / (progress + 1e-3)).clip(1, 400).astype('float32')
    laps_remaining = (est_total - lap).clip(-20, 400).astype('float32')

    df['_est_total_laps'] = est_total
    df['_laps_remaining'] = laps_remaining
    df['_tyre_to_lap_ratio'] = (tyre / (lap + 1)).astype('float32')
    df['_tyre_to_remaining_ratio'] = (tyre / (laps_remaining.abs() + 1)).astype('float32')
    df['_stint_to_lap_ratio'] = (stint / (lap + 1)).astype('float32')
    df['_pitstop_to_lap_ratio'] = (pitstop / (lap + 1)).astype('float32')
    df['_lap_time_per_tyre'] = (lap_time / (tyre + 1)).astype('float32')
    df['_delta_per_tyre'] = (delta / (tyre + 1)).astype('float32')
    df['_deg_per_tyre'] = (deg / (tyre + 1)).astype('float32')
    df['_lap_time_minus_delta'] = (lap_time - delta).astype('float32')
    df['_abs_delta'] = delta.abs().astype('float32')
    df['_abs_position_change'] = pos_chg.abs().astype('float32')
    df['_position_pressure'] = ((21 - pos).clip(0, 25) + pos_chg.clip(-10, 10)).astype('float32')
    df['_pit_window_pressure'] = ((tyre >= 8).astype('float32') * progress * (1 + pitstop)).astype('float32')
    df['_degradation_pressure'] = (deg * np.log1p(tyre.clip(lower=0))).astype('float32')
    df['_late_old_tyre'] = (progress * np.log1p(tyre.clip(lower=0))).astype('float32')
    df['RacePhase'] = pd.cut(progress, bins=[-0.01, .2, .4, .6, .8, 1.01], labels=False).astype('int8')
    df['TyreAgePhase'] = pd.cut(tyre, bins=[-1, 2, 5, 9, 13, 18, 25, 1000], labels=False).astype('int8')
    df['PitWindow'] = (((tyre >= 8).astype('int8') + (progress >= .55).astype('int8') + (pitstop >= 1).astype('int8')).astype('int8'))
    return df


def add_global_rank_and_freq(train_df, test_df, orig_df, num_cols):
    all_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], ignore_index=True)
    add_tr, add_te, add_or = {}, {}, {}
    n_tr, n_te = len(train_df), len(test_df)
    for c in num_cols:
        freq = all_num[c].value_counts(normalize=True, dropna=False)
        add_tr[f'FREQ_{safe_name(c)}'] = train_df[c].map(freq).fillna(0).astype('float32').to_numpy()
        add_te[f'FREQ_{safe_name(c)}'] = test_df[c].map(freq).fillna(0).astype('float32').to_numpy()
        add_or[f'FREQ_{safe_name(c)}'] = orig_df[c].map(freq).fillna(0).astype('float32').to_numpy()
        r = all_num[c].rank(method='average', pct=True)
        add_tr[f'RANK_{safe_name(c)}'] = r.iloc[:n_tr].astype('float32').to_numpy()
        add_te[f'RANK_{safe_name(c)}'] = r.iloc[n_tr:n_tr + n_te].astype('float32').to_numpy()
        add_or[f'RANK_{safe_name(c)}'] = r.iloc[n_tr + n_te:].astype('float32').to_numpy()
    return (pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1),
            pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1),
            pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1))


def add_num_transforms(train_df, test_df, orig_df, num_cols):
    def build(df):
        out = {}
        for c in num_cols:
            v = finite_float32(df[c])
            nm = safe_name(c)
            out[f'LOG1P_{nm}'] = np.log1p(v.clip(lower=0)).astype('float32')
            out[f'SQRT_{nm}'] = np.sqrt(v.clip(lower=0)).astype('float32')
            out[f'INV1P_{nm}'] = (1.0 / (1.0 + v.clip(lower=0))).astype('float32')
            out[f'ABS_{nm}'] = v.abs().astype('float32')
            for r in [1, 2, 3]:
                out[f'{nm}_round{r}'] = v.round(r).astype(str)
            out[f'{nm}_mod100'] = (np.floor(np.abs(v)) % 100).astype('int16').astype(str)
        return pd.DataFrame(out, index=df.index)
    return (pd.concat([train_df, build(train_df)], axis=1),
            pd.concat([test_df, build(test_df)], axis=1),
            pd.concat([orig_df, build(orig_df)], axis=1))


def add_bins_and_cross_cats(train_df, test_df, orig_df, num_cols):
    all_df = pd.concat([train_df, test_df, orig_df], ignore_index=True)
    n_tr, n_te = len(train_df), len(test_df)
    new_cat_cols = []
    for c in [x for x in num_cols if x in all_df.columns]:
        vals = finite_float32(all_df[c])
        for n_bins in [64, 256]:
            name = f'{safe_name(c)}_qbin{n_bins}'
            try:
                binned = pd.qcut(vals, q=min(n_bins, vals.nunique()), duplicates='drop', labels=False)
            except Exception:
                binned = pd.cut(vals, bins=min(n_bins, max(2, vals.nunique())), labels=False)
            all_df[name] = pd.Series(binned).fillna(-1).astype('int32').astype(str)
            new_cat_cols.append(name)
    for a, b in CROSS_PAIRS:
        if a in all_df.columns and b in all_df.columns:
            name = f'{safe_name(a)}__{safe_name(b)}'
            all_df[name] = all_df[a].astype(str) + '_' + all_df[b].astype(str)
            new_cat_cols.append(name)
    for tri in TRIPLES:
        if all(c in all_df.columns for c in tri):
            name = '__'.join(safe_name(c) for c in tri)
            all_df[name] = all_df[tri[0]].astype(str) + '_' + all_df[tri[1]].astype(str) + '_' + all_df[tri[2]].astype(str)
            new_cat_cols.append(name)
    return (all_df.iloc[:n_tr].reset_index(drop=True),
            all_df.iloc[n_tr:n_tr + n_te].reset_index(drop=True),
            all_df.iloc[n_tr + n_te:].reset_index(drop=True),
            dedup_keep_order(new_cat_cols))


def add_cat_counts_and_rare(train_df, test_df, orig_df, cat_cols, rare_thresh=50):
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], ignore_index=True)
    add_tr, add_te, add_or = {}, {}, {}
    for c in cat_cols:
        vc = all_cat[c].astype(str).value_counts(dropna=False)
        for df, out in [(train_df, add_tr), (test_df, add_te), (orig_df, add_or)]:
            cnt = df[c].astype(str).map(vc).fillna(0).astype('float32')
            out[f'CNT_{safe_name(c)}'] = np.log1p(cnt).astype('float32').to_numpy()
            out[f'RARE_{safe_name(c)}'] = (cnt <= rare_thresh).astype('float32').to_numpy()
    return (pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1),
            pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1),
            pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1))


def add_orig_proba_features(train_df, test_df, orig_df, cat_cols, y_orig):
    tmp = orig_df[cat_cols].copy()
    tmp['_y_'] = np.asarray(y_orig, dtype=np.float32)
    add_tr, add_te, add_or = {}, {}, {}
    global_mean = float(np.mean(y_orig))
    for c in cat_cols:
        stat = tmp.groupby(c, observed=False)['_y_'].agg(['mean', 'count'])
        smooth = (stat['mean'] * stat['count'] + global_mean * 25.0) / (stat['count'] + 25.0)
        name = f'ORIG_proba_{safe_name(c)}'
        cnt_name = f'ORIG_logcnt_{safe_name(c)}'
        for df, out in [(train_df, add_tr), (test_df, add_te), (orig_df, add_or)]:
            out[name] = df[c].map(smooth).fillna(global_mean).astype('float32').to_numpy()
            out[cnt_name] = np.log1p(df[c].map(stat['count']).fillna(0).astype('float32')).astype('float32').to_numpy()
    return (pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1),
            pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1),
            pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1))


def add_drift_ratio(train_df, test_df, orig_df, cat_cols, eps=1e-6):
    add_tr, add_te, add_or = {}, {}, {}
    for c in cat_cols:
        p_tr = train_df[c].astype(str).value_counts(normalize=True)
        p_or = orig_df[c].astype(str).value_counts(normalize=True)
        name = f'DRIFT_{safe_name(c)}'
        add_tr[name] = np.log1p(train_df[c].astype(str).map(p_tr).fillna(0) / (train_df[c].astype(str).map(p_or).fillna(eps) + eps)).astype('float32').to_numpy()
        add_te[name] = np.log1p(test_df[c].astype(str).map(p_tr).fillna(0) / (test_df[c].astype(str).map(p_or).fillna(eps) + eps)).astype('float32').to_numpy()
        add_or[name] = np.log1p(orig_df[c].astype(str).map(p_tr).fillna(0) / (orig_df[c].astype(str).map(p_or).fillna(eps) + eps)).astype('float32').to_numpy()
    return (pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1),
            pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1),
            pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1))


def build_features():
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].astype('int8')
    orig[TARGET] = orig[TARGET].astype('int8')
    y = train[TARGET].to_numpy(dtype=np.int8)
    y_orig = orig[TARGET].to_numpy(dtype=np.int8)
    feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
    base_cats, base_nums, numeric_like = infer_groups(train, test, feature_cols)
    base_cats = dedup_keep_order([c for c in BASE_CAT_CANDIDATES if c in feature_cols] + base_cats)
    base_nums = dedup_keep_order([c for c in BASE_NUMS if c in feature_cols] + base_nums)

    for df in [train, test, orig]:
        for c in numeric_like:
            if c in df.columns:
                df[c] = finite_float32(df[c])
        for c in base_cats:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna('missing')

    train_fe = add_state_features(train[feature_cols].copy())
    test_fe = add_state_features(test[feature_cols].copy())
    orig_fe = add_state_features(orig[feature_cols].copy())
    extra_nums = [c for c in train_fe.columns if c.startswith('_')]
    num_cols = dedup_keep_order([c for c in base_nums + extra_nums if c in train_fe.columns])
    artifact_cols = [c for c in ARTIFACT_NUMS + ['RaceProgress'] if c in train_fe.columns]

    train_fe, test_fe, orig_fe = add_global_rank_and_freq(train_fe, test_fe, orig_fe, num_cols)
    train_fe, test_fe, orig_fe = add_num_transforms(train_fe, test_fe, orig_fe, artifact_cols)
    train_fe, test_fe, orig_fe, engineered_cats = add_bins_and_cross_cats(train_fe, test_fe, orig_fe, artifact_cols)

    cat_cols = dedup_keep_order([c for c in base_cats + ['RacePhase', 'TyreAgePhase', 'PitWindow'] + engineered_cats if c in train_fe.columns])
    for df in [train_fe, test_fe, orig_fe]:
        for c in cat_cols:
            df[c] = df[c].astype(str).fillna('missing')

    train_fe, test_fe, orig_fe = add_cat_counts_and_rare(train_fe, test_fe, orig_fe, cat_cols, rare_thresh=50)
    train_fe, test_fe, orig_fe = add_orig_proba_features(train_fe, test_fe, orig_fe, cat_cols, y_orig)
    train_fe, test_fe, orig_fe = add_drift_ratio(train_fe, test_fe, orig_fe, cat_cols, eps=DRIFT_EPS)

    for df in [train_fe, test_fe, orig_fe]:
        for c in cat_cols:
            df[c] = df[c].astype('category')
        for c in df.columns:
            if c not in cat_cols:
                df[c] = finite_float32(df[c])
    print(f'{train_fe.shape[1]} features, {len(cat_cols)} categorical', flush=True)
    return train_fe, test_fe, orig_fe, cat_cols, y, y_orig


def main():
    seed_everything(SEED)
    train_fe, test_fe, orig_fe, cat_cols, y, y_orig = build_features()
    fold_id = folds(y)
    oof = np.zeros(len(train_fe), dtype=np.float32)
    test_preds = np.zeros(len(test_fe), dtype=np.float32)

    for k in range(N_FOLDS):
        fold = k + 1
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        X_tr = pd.concat([train_fe.iloc[tr_idx].reset_index(drop=True), orig_fe.reset_index(drop=True)], ignore_index=True)
        y_tr = np.concatenate([y[tr_idx], y_orig])
        X_va = train_fe.iloc[va_idx].reset_index(drop=True)
        y_va = y[va_idx]
        X_te = test_fe.copy().reset_index(drop=True)
        X_tr['_is_original_'] = pd.Series(np.r_[np.zeros(len(tr_idx), dtype=np.int8), np.ones(len(orig_fe), dtype=np.int8)].astype(str), index=X_tr.index).astype('category')
        X_va['_is_original_'] = pd.Series(['0'] * len(X_va), index=X_va.index).astype('category')
        X_te['_is_original_'] = pd.Series(['0'] * len(X_te), index=X_te.index).astype('category')
        fold_cat_cols = cat_cols + ['_is_original_']

        fold_ens_va = np.zeros(len(va_idx), dtype=np.float32)
        fold_ens_te = np.zeros(len(test_fe), dtype=np.float32)
        for model_idx, p in enumerate(THREE_CONFIGS, 1):
            params = dict(p)
            params['random_state'] = SEED + 100 * fold + model_idx
            model = RealMLP_TD_Classifier(**params)
            model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, cat_col_names=fold_cat_cols)
            va_pred = model.predict_proba(X_va)[:, 1].astype(np.float32)
            te_pred = model.predict_proba(X_te)[:, 1].astype(np.float32)
            print(f'fold {fold} config {model_idx}: AUC {roc_auc_score(y_va, va_pred):.6f}', flush=True)
            fold_ens_va += va_pred / len(THREE_CONFIGS)
            fold_ens_te += te_pred / len(THREE_CONFIGS)
            del model, va_pred, te_pred
            torch.cuda.empty_cache()
            gc.collect()

        oof[va_idx] = fold_ens_va
        test_preds += fold_ens_te / N_FOLDS
        print(f'fold {fold}: ensemble AUC {roc_auc_score(y_va, fold_ens_va):.6f}', flush=True)
        del X_tr, X_va, X_te, y_tr, y_va, fold_ens_va, fold_ens_te
        torch.cuda.empty_cache()
        gc.collect()

    oof = np.clip(oof, 1e-7, 1 - 1e-7).astype(np.float32)
    test_preds = np.clip(test_preds, 1e-7, 1 - 1e-7).astype(np.float32)
    print(f'OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, test_preds)


if __name__ == '__main__':
    main()
