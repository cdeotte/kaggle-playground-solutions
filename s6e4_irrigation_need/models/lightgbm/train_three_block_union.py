#!/usr/bin/env python
"""Class-weighted LightGBM on three feature blocks (value tokens / transforms and crosses / artifact bins and drift)
with nested smoothed per-class target encodings (pair lgbm_three_block_union_nested_te_b).

Label-free features (train + test rows; original rows for frequencies, priors and drift; 511 columns before encoding):
  - the raw columns as read (id included; the 8 string categoricals are coerced to missing values, they enter only
    through the blocks below)
  - block 1: every one of the 19 columns as a string key, its count over train + test, its mean original class index
    (0 Low, 1 Medium, 2 High; the numeric keys fall back to the overall mean) and count / original count; per numeric
    the rounded integer, <= 100 quantile bins and 200 equal-width bins as keys; up to 4 integer and 4 decimal digits
    of rainfall, soil moisture, previous irrigation, humidity, temperature and the 15 pairs of the first 6 rainfall /
    soil-moisture decimal digits as keys
  - block A: frequency (train + test + original), percentile rank, log1p, sqrt and 1 / (1 + x) of every numeric; a
    threshold score 2 (moisture < 25) + 2 (rainfall < 300) + (temperature > 30) + (wind > 10) - 2 (Harvest)
    - 2 (Sowing) - (mulching), its flags, moisture - 25, rainfall - 300, moisture / (rainfall + 1), temperature x wind;
    quantile bins of 5 numerics (edges from train + test + original), 6 category pairs and 1 triple as keys;
    per-class original frequency of the 8 categoricals; count (train + test, plus the original rows for the
    categoricals) and rare flag (<= 50) of the categoricals and keys
  - block B: 100-quantile bins of 6 numerics, roundings (to 10 / 1 / 0.1) of 5 numerics, digits at 100 ... 0.01 of 5
    numerics; log1p(train frequency / original frequency) of these and of the categoricals; per-class original
    frequency and original count of the categoricals and the 6 numerics; train means of the 6 numerics per categorical
Per fold: each of the 103 columns whose name ends in "_te" (the keys above, plus the counts / rare flags of the block A
keys) is replaced, for each class, by the class rate smoothed toward the fold's class rate ((sum + 10 x rate) /
(count + 10)), out-of-fold over 5 random inner folds (seed 42) of the training rows and from all training rows for the
validation / test rows.
Model: LightGBM multiclass, balanced class sample weights, lr 0.01, max_depth 4, 15 leaves, subsample 0.5,
colsample_bytree 0.2, lambda_l2 15, lambda_l1 10, seed 42, 32 threads, up to 7500 rounds with early stopping (500) on
the validation fold. Test = mean of the 5 fold models.
Usage: python train_three_block_union.py"""
import gc
import itertools
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

PAIR = 'lgbm_three_block_union_nested_te_b'
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
N_CLASSES = 3
TE_SMOOTH = 10

CATS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours',
        'Wind_Speed_kmh', 'Field_Area_hectare', 'Temperature_C', 'Soil_Moisture',
        'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']

QCUT_BINS = 100  # block-1 quantile bins of each numeric
DIGIT_SPECS = {
    'Rainfall_mm':            list(range(-2, 3)),
    'Soil_Moisture':          list(range(-2, 3)),
    'Previous_Irrigation_mm': list(range(-2, 3)),
    'Humidity':               list(range(-1, 3)),
    'Temperature_C':          list(range(-1, 3)),
}
ROUND_SPECS = {
    'Rainfall_mm':            [-1, 0, 1],
    'Soil_Moisture':          [-1, 0, 1],
    'Previous_Irrigation_mm': [-1, 0, 1],
    'Temperature_C':          [-1, 0],
    'Wind_Speed_kmh':         [-1, 0],
}
CROSS_PAIRS = [
    ('Crop_Type', 'Season'),
    ('Season', 'Region'),
    ('Irrigation_Type', 'Crop_Type'),
    ('Crop_Growth_Stage', 'Mulching_Used'),
    ('Water_Source', 'Irrigation_Type'),
    ('Crop_Type', 'Region'),
]
TRIPLE = [('Crop_Type', 'Season', 'Region')]
BIN_CONFIG = {'Rainfall_mm': 50, 'Soil_Moisture': 30, 'Previous_Irrigation_mm': 40,
              'Humidity': 20, 'Temperature_C': 20}
B_NUMS = ['Rainfall_mm', 'Soil_Moisture', 'Previous_Irrigation_mm', 'Humidity', 'Temperature_C', 'Wind_Speed_kmh']

LGBM_PARAMS = dict(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    n_estimators=7500,
    learning_rate=0.01,
    max_depth=4,
    num_leaves=15,
    subsample=0.5,
    colsample_bytree=0.2,
    lambda_l2=15.0,
    lambda_l1=10.0,
    random_state=SEED,
    verbose=-1,
    device='cpu',
    n_jobs=32,
)

T0 = time.perf_counter()


def log(msg):
    print(f'[{time.perf_counter() - T0:7.1f}s] {msg}', flush=True)


def _reset(df):
    return df.reset_index(drop=True)


def nested_te_multiclass(X_tr_df, y_tr, X_va_df, X_te_df, cols, n_classes=3, n_folds=5, smooth=10, seed=42):
    """Per string column and class k: smoothed rate of class k, out-of-fold on X_tr_df (random inner folds),
    from all of X_tr_df for X_va_df / X_te_df."""
    rng = np.random.default_rng(seed)
    fold_id = np.arange(len(X_tr_df)) % n_folds
    rng.shuffle(fold_id)

    tr_new, va_new, te_new = {}, {}, {}
    for k in range(n_classes):
        y_bin = (y_tr == k).astype(float)
        gm = y_bin.mean()
        for col in cols:
            out = f'{col}_TE_k{k}'
            tr_enc = np.full(len(X_tr_df), gm, dtype=np.float32)
            for i in range(n_folds):
                mask_in = (fold_id != i)
                mask_oof = (fold_id == i)
                grp = (pd.DataFrame({'key': X_tr_df[col].values[mask_in], '_y_': y_bin[mask_in]})
                       .groupby('key', sort=False)
                       .agg(cnt=('_y_', 'count'), sm=('_y_', 'sum'))
                       .reset_index())
                grp[out] = (grp['sm'] + gm * smooth) / (grp['cnt'] + smooth)
                mp = grp.set_index('key')[out]
                tr_enc[mask_oof] = pd.Series(X_tr_df[col].values[mask_oof]).map(mp).fillna(gm).astype(np.float32).values

            grp_full = (pd.DataFrame({'key': X_tr_df[col].values, '_y_': y_bin})
                        .groupby('key', sort=False)
                        .agg(cnt=('_y_', 'count'), sm=('_y_', 'sum'))
                        .reset_index())
            grp_full[out] = (grp_full['sm'] + gm * smooth) / (grp_full['cnt'] + smooth)
            mp_full = grp_full.set_index('key')[out]

            tr_new[out] = tr_enc
            va_new[out] = pd.Series(X_va_df[col].values).map(mp_full).fillna(gm).astype(np.float32).values
            te_new[out] = pd.Series(X_te_df[col].values).map(mp_full).fillna(gm).astype(np.float32).values

    return pd.DataFrame(tr_new), pd.DataFrame(va_new), pd.DataFrame(te_new)


def add_qcut_bins(train, test, orig, num_cols, n_bins):
    new_cols = []
    for c in num_cols:
        name = f'B_qcut_{c}'
        all_vals = pd.concat([train[c], test[c], orig[c]], ignore_index=True)
        try:
            q = min(n_bins, max(2, all_vals.nunique(dropna=True)))
            _, bins = pd.qcut(all_vals, q=q, retbins=True, duplicates='drop')
            for df in (train, test, orig):
                df[name] = pd.cut(df[c], bins=bins, labels=False, include_lowest=True).astype('float32')
        except Exception:
            for df in (train, test, orig):
                df[name] = 0.0
        new_cols.append(name)
    return train, test, orig, new_cols


def add_round_features(train, test, orig, round_specs):
    out = []
    for c, rr in round_specs.items():
        for r in rr:
            n = f'B_round_{c}_r{r}'
            for df in (train, test, orig):
                df[n] = pd.to_numeric(df[c], errors='coerce').round(r)
            out.append(n)
    return train, test, orig, out


def add_digit_features(train, test, orig, digit_specs):
    out = []
    for c, ks in digit_specs.items():
        for k in ks:
            n = f'B_digit_{c}_d{k}'
            for df in (train, test, orig):
                v = pd.to_numeric(df[c], errors='coerce')
                df[n] = (((v * (10 ** k)) % 10).fillna(-1)).astype('int8')
            out.append(n)
    return train, test, orig, out


def add_drift_ratio(train, test, orig, cols, eps=1e-6):
    new_cols = []
    for c in cols:
        p_tr = train[c].astype(str).value_counts(normalize=True)
        p_or = orig[c].astype(str).value_counts(normalize=True)
        name = f'B_DRIFT_{c}'
        train[name] = np.log1p(train[c].astype(str).map(p_tr).fillna(0) /
                               (train[c].astype(str).map(p_or).fillna(eps) + eps)).astype('float32')
        test[name] = np.log1p(test[c].astype(str).map(p_tr).fillna(0) /
                              (test[c].astype(str).map(p_or).fillna(eps) + eps)).astype('float32')
        new_cols.append(name)
    return train, test, new_cols


def add_orig_priors(train, test, orig, cols, y_orig, n_classes=3):
    new_cols = []
    for c in cols:
        for k in range(n_classes):
            name = f'B_ORIG_k{k}_{c}'
            m = (pd.DataFrame({'key': orig[c].astype(str), '_y_': (y_orig == k).astype(float)})
                 .groupby('key')['_y_'].mean())
            gm = (y_orig == k).mean()
            train[name] = train[c].astype(str).map(m).fillna(gm).astype('float32')
            test[name] = test[c].astype(str).map(m).fillna(gm).astype('float32')
            new_cols.append(name)
    return train, test, new_cols


def add_orig_ce(train, test, orig, cols):
    new_cols = []
    for c in cols:
        name = f'B_CEORIG_{c}'
        vc = orig[c].astype(str).value_counts(dropna=False)
        train[name] = train[c].astype(str).map(vc).fillna(0).astype('float32')
        test[name] = test[c].astype(str).map(vc).fillna(0).astype('float32')
        new_cols.append(name)
    return train, test, new_cols


def add_group_means(train, test, group_cols, num_cols):
    new_cols = []
    for g in group_cols:
        for n in num_cols:
            if g == n:
                continue
            name = f'B_MEAN_{n}_BY_{g}'
            m = train.groupby(g, dropna=False)[n].mean()
            train[name] = train[g].map(m).astype('float32')
            test[name] = test[g].map(m).fillna(float(m.mean())).astype('float32')
            new_cols.append(name)
    return train, test, new_cols


def build_features():
    log('Loading data...')
    train = read_train()
    test = read_test()
    orig = read_original()
    for df in (train, test, orig):
        for c in NUMS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        for c in CATS:
            if c in df.columns:
                df[c] = df[c].astype(str)
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype('int32')
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype('int32')
    y = train[TARGET].values
    y_orig = orig[TARGET].values

    X_base = train.drop(columns=[TARGET]).copy()
    T_base = test.copy()
    N_train, N_test = len(X_base), len(T_base)

    # block 1: value keys, counts, original priors, numeric bins, digits
    log('Block 1: value keys, counts, original priors, bins, digits')
    combined = pd.concat([_reset(X_base), _reset(T_base)], axis=0, ignore_index=True)
    orig_fe = orig.drop(columns=[TARGET]).copy()
    te1_cols = CATS + NUMS
    te1 = _reset(combined[te1_cols].astype(str).rename(columns={c: f'{c}_te' for c in te1_cols}))

    ce1 = {}
    for col in te1_cols:
        ce1[f'count({col})'] = combined.groupby(col, dropna=False)[col].transform('size')
    ce1 = _reset(pd.DataFrame(ce1))

    orig_tmp = orig_fe.copy()
    orig_tmp['_y_'] = y_orig.astype('float32')
    te1_orig = {}
    for col in te1_cols:
        m = orig_tmp.groupby(col, dropna=False)['_y_'].mean()
        te1_orig[f'ORIG_mean_{col}'] = combined[col].astype(str).map(m.astype('float32')).fillna(float(np.nanmean(y_orig)))
    te1_orig = _reset(pd.DataFrame(te1_orig))

    orig_count = {}
    for col in te1_cols:
        oc = orig_fe[col].value_counts(dropna=False)
        name = f'count_ratio({col})'
        synth_cnt = ce1[f'count({col})'].astype('float32')
        orig_cnt = combined[col].map(oc).fillna(0).astype('float32')
        orig_count[name] = (synth_cnt / (orig_cnt.replace(0, np.nan))).fillna(0).astype('float32')
    orig_count = _reset(pd.DataFrame(orig_count))

    num_bins = {}
    for col in NUMS:
        s = pd.to_numeric(combined[col], errors='coerce')
        num_bins[f'{col}_int_te'] = s.round(0).fillna(-999).astype('int64').astype(str)
        try:
            q = min(QCUT_BINS, max(2, s.nunique(dropna=True)))
            num_bins[f'{col}_qbin_te'] = pd.qcut(s, q=q, duplicates='drop').astype(str)
        except Exception:
            num_bins[f'{col}_qbin_te'] = pd.Series(['NA'] * len(combined))
        try:
            num_bins[f'{col}_bin_te'] = pd.cut(s, bins=200, duplicates='drop').astype(str)
        except Exception:
            num_bins[f'{col}_bin_te'] = pd.Series(['NA'] * len(combined))
    num_bins = _reset(pd.DataFrame(num_bins))

    digits = {}
    for col in ['Rainfall_mm', 'Soil_Moisture', 'Previous_Irrigation_mm', 'Humidity', 'Temperature_C']:
        s = pd.to_numeric(combined[col], errors='coerce').fillna(0).astype('float64')
        ss = s.map(lambda x: f'{x:.6f}')
        sp = ss.str.split('.', expand=True)
        ndec = sp[0]
        dec = sp[1] if sp.shape[1] > 1 else pd.Series(['0'] * len(ss))
        maxn = int(ndec.str.len().max())
        maxd = int(dec.str.len().max())
        ndec_p = ndec.str.rjust(maxn, '0')
        dec_p = dec.str.ljust(maxd, '0')
        for i in range(min(4, maxn)):
            digits[f'{col}_digit{i}'] = ndec_p.str[i].astype(int)
        for i in range(min(4, maxd)):
            digits[f'{col}_decimal_digit{i}'] = dec_p.str[i].astype(int)
    digits = _reset(pd.DataFrame(digits))

    dec_cands = [c for c in digits.columns if 'decimal_digit' in c and
                 any(nm in c for nm in ['Rainfall', 'Soil_Moisture'])]
    dec_cands = dec_cands[:6]
    digits_combs = pd.DataFrame(index=pd.RangeIndex(len(combined)))
    if len(dec_cands) >= 2:
        for c1, c2 in itertools.combinations(dec_cands, 2):
            digits_combs[f'{c1}__{c2}_te'] = (digits[c1].astype(str) + '|' + digits[c2].astype(str))
    digits_combs = _reset(digits_combs)
    log(f'Block 1: te1={te1.shape[1]} num_bins={num_bins.shape[1]} digits={digits.shape[1]} digit_combs={digits_combs.shape[1]}')

    # block A: frequency / rank / transforms, threshold score, bins and crosses, priors, counts
    log('Block A: frequencies, transforms, crosses')
    all_num = pd.concat([train[NUMS], test[NUMS], orig[NUMS]], axis=0, ignore_index=True)
    A_blocks = {}
    for c in NUMS:
        freq = all_num[c].value_counts(normalize=True)
        tt = pd.concat([train[c], test[c]], axis=0, ignore_index=True)
        A_blocks[f'A_FREQ_{c}'] = tt.map(freq).fillna(0).astype('float32').values
    for c in NUMS:
        r = all_num[c].rank(method='average', pct=True)
        A_blocks[f'A_RANK_{c}'] = r.iloc[:N_train + N_test].astype('float32').values
    for c in NUMS:
        v = pd.concat([train[c], test[c]], axis=0, ignore_index=True).astype('float32')
        A_blocks[f'A_LOG1P_{c}'] = np.log1p(v.clip(lower=0)).values
        A_blocks[f'A_SQRT_{c}'] = np.sqrt(v.clip(lower=0)).values
        A_blocks[f'A_INV1P_{c}'] = (1.0 / (1.0 + v.clip(lower=0))).values

    for df in (train, test):
        SM = df['Soil_Moisture'].astype('float32')
        RM = df['Rainfall_mm'].astype('float32')
        TC = df['Temperature_C'].astype('float32')
        WS = df['Wind_Speed_kmh'].astype('float32')
        CGS = df['Crop_Growth_Stage'].astype(str)
        MU = df['Mulching_Used'].astype(str)
        score = ((SM < 25).astype(int) * 2 + (RM < 300).astype(int) * 2 +
                 (TC > 30).astype(int) + (WS > 10).astype(int) -
                 (CGS == 'Harvest').astype(int) * 2 - (CGS == 'Sowing').astype(int) * 2 -
                 (MU == 'Yes').astype(int))
        df['_formula_score'] = score.astype('float32')
        df['_is_high'] = (score > 3).astype('float32')
        df['_is_low'] = (score <= 0).astype('float32')
        df['_SM_thr'] = (SM - 25.0).astype('float32')
        df['_RM_thr'] = (RM - 300.0).astype('float32')
        df['_SM_div_RM_p1'] = (SM / (RM + 1.0)).astype('float32')
        df['_TC_x_WS'] = (TC * WS).astype('float32')
    formula_cols = ['_formula_score', '_is_high', '_is_low', '_SM_thr', '_RM_thr', '_SM_div_RM_p1', '_TC_x_WS']
    for c in formula_cols:
        v = pd.concat([train[c], test[c]], axis=0, ignore_index=True).astype('float32')
        A_blocks[f'A_{c}'] = v.values
    A_num_df = _reset(pd.DataFrame(A_blocks))

    A_te_extra = {}
    for col, q in BIN_CONFIG.items():
        all_v = pd.concat([train[col], test[col], orig[col]], ignore_index=True)
        try:
            bins = pd.qcut(all_v, q=q, retbins=True, duplicates='drop')[1]
            tt = pd.concat([train[col], test[col]], ignore_index=True)
            A_te_extra[f'A_{col}_{q}bin_te'] = pd.cut(tt, bins=bins, include_lowest=True).astype(str)
        except Exception:
            A_te_extra[f'A_{col}_{q}bin_te'] = pd.Series(['NA'] * (N_train + N_test))
    for a, b in CROSS_PAIRS:
        aa = pd.concat([train[a], test[a]], ignore_index=True).astype(str)
        bb = pd.concat([train[b], test[b]], ignore_index=True).astype(str)
        A_te_extra[f'A_{a}__{b}_te'] = (aa + '|' + bb)
    for a, b, c in TRIPLE:
        aa = pd.concat([train[a], test[a]], ignore_index=True).astype(str)
        bb = pd.concat([train[b], test[b]], ignore_index=True).astype(str)
        cc = pd.concat([train[c], test[c]], ignore_index=True).astype(str)
        A_te_extra[f'A_{a}__{b}__{c}_te'] = (aa + '|' + bb + '|' + cc)
    A_te_df = _reset(pd.DataFrame(A_te_extra))

    A_orig = {}
    orig_tmp2 = orig.copy()
    for c in CATS:
        for k in range(N_CLASSES):
            name = f'A_ORIG_k{k}_{c}'
            m = orig_tmp2.groupby(c, observed=False).apply(lambda g: (g[TARGET] == k).mean())
            tt = pd.concat([train[c], test[c]], ignore_index=True).astype(str)
            A_orig[name] = tt.map(m.astype('float32')).fillna((y_orig == k).mean()).astype('float32').values
    A_orig_df = _reset(pd.DataFrame(A_orig))

    cat_for_counts = pd.DataFrame()
    for c in CATS:
        cat_for_counts[c] = pd.concat([train[c], test[c]], ignore_index=True).astype(str)
    for c in A_te_df.columns:
        cat_for_counts[c] = A_te_df[c].astype(str)
    all_cat = pd.concat([cat_for_counts, orig[CATS].astype(str)], axis=0, ignore_index=True)
    A_cnt = {}
    for c in cat_for_counts.columns:
        vc = all_cat[c].value_counts(dropna=False)
        cnt = cat_for_counts[c].map(vc).fillna(0).astype('float32')
        A_cnt[f'A_CAT_CNT_{c}'] = cnt.values
        A_cnt[f'A_CAT_RARE_{c}'] = (cnt <= 50).astype('float32').values
    A_cnt_df = _reset(pd.DataFrame(A_cnt))
    log(f'Block A: A_num={A_num_df.shape[1]} A_te={A_te_df.shape[1]} A_orig={A_orig_df.shape[1]} A_cnt={A_cnt_df.shape[1]}')

    # block B: bins, roundings, digits, drift, original priors and counts, group means
    log('Block B: bins, roundings, digits, drift, priors, group means')
    trB = train.drop(columns=[TARGET]).copy()
    teB = test.copy()
    orB = orig.drop(columns=[TARGET]).copy()
    trB, teB, orB, B_QCUT = add_qcut_bins(trB, teB, orB, B_NUMS, 100)
    trB, teB, orB, B_ROUND = add_round_features(trB, teB, orB, ROUND_SPECS)
    trB, teB, orB, B_DIGITS = add_digit_features(trB, teB, orB, DIGIT_SPECS)
    for c in CATS:
        combined_c = pd.concat([trB[c], teB[c], orB[c]], ignore_index=True)
        codes, _ = combined_c.factorize(sort=False)
        trB[c] = codes[:N_train].astype('int32')
        teB[c] = codes[N_train:N_train + N_test].astype('int32')
        orB[c] = codes[N_train + N_test:].astype('int32')
    trB, teB, B_DRIFT = add_drift_ratio(trB, teB, orB, CATS + B_QCUT + B_ROUND + B_DIGITS)
    trB, teB, B_ORIG_PRIOR = add_orig_priors(trB, teB, orB, CATS + B_NUMS, y_orig)
    trB, teB, B_CE = add_orig_ce(trB, teB, orB, CATS + B_NUMS)
    trB, teB, B_GM = add_group_means(trB, teB, CATS, B_NUMS)
    B_attach_cols = B_QCUT + B_ROUND + B_DIGITS + B_DRIFT + B_ORIG_PRIOR + B_CE + B_GM
    B_attach_df = pd.concat(
        [trB[B_attach_cols].reset_index(drop=True), teB[B_attach_cols].reset_index(drop=True)],
        axis=0, ignore_index=True,
    )
    log(f'Block B: {len(B_attach_cols)} cols')

    dfs = [combined, te1, ce1, te1_orig, orig_count, num_bins, digits, digits_combs,
           A_num_df, A_te_df, A_orig_df, A_cnt_df, B_attach_df]
    for i, d in enumerate(dfs):
        assert len(d) == N_train + N_test, f'block {i} has length {len(d)}, expected {N_train + N_test}'
    combined_fe = pd.concat(dfs, axis=1)
    log(f'combined_fe shape: {combined_fe.shape}')

    X_fe = combined_fe.iloc[:N_train].copy()
    T_fe = combined_fe.iloc[N_train:].copy()
    te_cols_all = [c for c in X_fe.columns if c.endswith('_te')]
    log(f'*_te columns to encode: {len(te_cols_all)}')
    non_te_cols = [c for c in X_fe.columns if not c.endswith('_te')]
    for col in non_te_cols:
        X_fe[col] = pd.to_numeric(X_fe[col], errors='coerce').astype('float32')
        T_fe[col] = pd.to_numeric(T_fe[col], errors='coerce').astype('float32')
    return X_fe, T_fe, y, te_cols_all


def main():
    X_fe, T_fe, y, te_cols_all = build_features()
    cls_counts = np.bincount(y, minlength=N_CLASSES)
    cls_weights = cls_counts.sum() / (N_CLASSES * cls_counts)
    log(f'class weights: {cls_weights}')

    fold_ids = folds(y)
    oof = np.zeros((len(X_fe), N_CLASSES), dtype=np.float32)
    pred = np.zeros((len(T_fe), N_CLASSES), dtype=np.float32)
    for k in range(N_FOLDS):
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        t_fold = time.perf_counter()
        log(f'--- Fold {k} ---')

        X_tr = X_fe.iloc[tr_idx].reset_index(drop=True).copy()
        X_va = X_fe.iloc[va_idx].reset_index(drop=True).copy()
        X_te = T_fe.reset_index(drop=True).copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        if te_cols_all:
            tr_enc, va_enc, te_enc = nested_te_multiclass(
                X_tr[te_cols_all], y_tr,
                X_va[te_cols_all], X_te[te_cols_all],
                te_cols_all, n_classes=N_CLASSES,
                n_folds=INNER_FOLDS, smooth=TE_SMOOTH, seed=SEED,
            )
            for col in te_enc.columns:
                X_tr[col] = tr_enc[col].values
                X_va[col] = va_enc[col].values
                X_te[col] = te_enc[col].values
            X_tr.drop(columns=te_cols_all, inplace=True)
            X_va.drop(columns=te_cols_all, inplace=True)
            X_te.drop(columns=te_cols_all, inplace=True)

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight=cls_weights[y_tr],
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.log_evaluation(1000),
                lgb.early_stopping(500, verbose=False),
            ],
        )

        p_va = model.predict_proba(X_va).astype(np.float32)
        p_te = model.predict_proba(X_te).astype(np.float32)
        oof[va_idx] = p_va
        pred += p_te / N_FOLDS
        bac = balanced_accuracy_score(y_va, p_va.argmax(axis=1))
        log(f'Fold {k} BAC: {bac:.6f} | best_iter={getattr(model, "best_iteration_", None)} | {time.perf_counter() - t_fold:.1f}s')

        del X_tr, X_va, X_te, model, p_va, p_te
        gc.collect()

    log(f'{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, oof.argmax(axis=1)):.6f}')
    save_pair(HERE, PAIR, oof, pred)


if __name__ == '__main__':
    main()
