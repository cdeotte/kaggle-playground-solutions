"""Per-fold feature matrix of the SVC / RFF / RF models: 213 columns, rich=True adds 133 per-fold channel columns (346).

The target-free blocks are read from the cache of ../density_features: static_{tr,te}.parquet (raw, missingness, budget
geometry, value counts, original-data evidence, pair PMIs, imputations), tok_{tr,te}.parquet (exact-value tokens '%.2f' /
'__NA__' of the 12 columns), mc_*, impq_*, den_*. Columns, in this order:
  raw 12 | missingness 15 | budget 13 (sum of the 3 components, slack, shares, weekend - daily, region coordinates) |
  cnt 12 (value counts) | orig 31 (original-data log-likelihood ratios per exact value and per 0.1-bin, amplifications) |
  pmi 18 (exact-pair PMIs) | bnd 6 (interval bounds implied by the budget identity) | mc 11 (Monte-Carlo posterior mechanism
  features) | impq 9 (imputation quantiles) | imp 19 (imputations, fills, region probabilities) | den 26 (masked-density
  log-probability and entropy per column) | lat 12 (fractional part and first decimal of 6 hour columns) |
  te 15 (target encodings of the 12 tokens + 3 pair keys, smoothing 20) | te2s 12 (the 12 tokens, smoothing 4) | woe 2.
Target encodings are nested per outer fold: outer-train rows are encoded by an inner StratifiedKFold(5, shuffle=True) split
(te: the mean over the inner seeds 777, 1778, 2779; te2s: seed 777), validation and test rows by the map of all
outer-train rows.
rich: + cmp4 23, plift 10, cbank 9, wbank 5, denv4 26, mfill4 27, rrd 19, qfill 14 (lib/channels.py)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.data import CATEGORICALS, NUMERICS, TARGET, cache_dir, folds as fold_ids, read_test, read_train

from .channels import append_channels

NUMS, CATS = list(NUMERICS), list(CATEGORICALS)
FEATS = NUMS + CATS
TE_SMOOTH = 20.0
TE_SEEDS = (777, 1778, 2779)
RICH = ['cmp4', 'plift', 'cbank', 'wbank', 'denv4', 'mfill4', 'rrd', 'qfill']


def _te_map(vals, yy, prior, smooth=TE_SMOOTH):
    g = pd.DataFrame({'v': vals, 'y': yy}).groupby('v')['y'].agg(['mean', 'size'])
    return (g['mean'] * g['size'] + prior * smooth) / (g['size'] + smooth)


def build_te_fold(y, trn_idx, keys, inner_seed=777, n_inner=5, smooth=TE_SMOOTH):
    """Nested TE for one outer fold. keys: list of (name, token Series on the train index).
    Returns (inner-fold encodings of the outer-train rows, maps fitted on all outer-train rows)."""
    prior = y[trn_idx].mean()
    te_trn = {name: np.full(len(trn_idx), np.nan, np.float32) for name, _ in keys}
    inner = StratifiedKFold(n_inner, shuffle=True, random_state=inner_seed)
    for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y[trn_idx]):
        fit_rows = trn_idx[i_fit]
        for name, tk in keys:
            mp = _te_map(tk.iloc[fit_rows].values, y[fit_rows], prior, smooth)
            te_trn[name][i_app] = pd.Series(tk.iloc[trn_idx[i_app]].values).map(mp).fillna(prior).values
    full_maps = {name: (_te_map(tk.iloc[trn_idx].values, y[trn_idx], prior, smooth), prior) for name, tk in keys}
    return te_trn, full_maps


def apply_te(full_maps, keys_series):
    out = {}
    for name, tk in keys_series:
        mp, prior = full_maps[name]
        out[name] = tk.map(mp).fillna(prior).values.astype(np.float32)
    return out


def assemble(fold, rich=False, log=print):
    """dict(A, B, y, trn_idx, val_idx): A = train rows (train.csv order), B = test rows."""
    train, test = read_train(), read_test()
    y = train[TARGET].values
    folds = fold_ids(y)
    trn_idx, val_idx = np.where(folds != fold)[0], np.where(folds == fold)[0]

    C = cache_dir('density_features')
    Xtr, Xte = pd.read_parquet(C / 'static_tr.parquet'), pd.read_parquet(C / 'static_te.parquet')
    tok_tr, tok_te = pd.read_parquet(C / 'tok_tr.parquet'), pd.read_parquet(C / 'tok_te.parquet')

    RAW = NUMS + CATS
    MISSING = [c for c in Xtr.columns if c.startswith('na_')] + ['n_missing', 'na_screen', 'na_beh']
    BUDGET = ['sgw', 'slack', 'slack_frac', 'share_social_media_hours', 'share_gaming_hours', 'share_work_study_hours',
              'wk_minus_d', 'wk_minus_sgw', 'wk_ratio', 'd_m6', 'd_m8', 's_m4', 'region_obs']
    COUNTS = [c for c in Xtr.columns if c.startswith('cnt_')]
    ORIG = [c for c in Xtr.columns if c.startswith(('ollr_', 'ollr1_', 'amp_'))] + ['ollr_sum']
    PMI = [c for c in Xtr.columns if c.startswith('pmi_')]
    IMP = [c for c in Xtr.columns if c.startswith(('imp_', 'fill_'))] + ['p_reg_lo', 'p_reg_mid', 'p_reg_hi', 'reg_exp']
    cols = RAW + MISSING + BUDGET + COUNTS + ORIG + PMI

    # bnd: exact interval bounds implied by the budget identity
    for Df in (Xtr, Xte):
        dd, ss = Df['daily_screen_time_hours'], Df['social_media_hours']
        gg, ww = Df['gaming_hours'], Df['work_study_hours']
        Df['social_ub'] = (dd - gg - ww).astype(np.float32)      # >= social always
        Df['gaming_ub'] = (dd - ss - ww).astype(np.float32)
        Df['work_ub'] = (dd - ss - gg).astype(np.float32)
        Df['daily_lb'] = (ss + gg + ww).astype(np.float32)       # <= daily always
        # region-impossibility proofs usable when social is missing
        hi_imp = dd.notna() & (dd <= 8) & ss.isna() & Df['social_ub'].notna() & (Df['social_ub'] <= 4)
        Df['hi_impossible'] = hi_imp.astype(np.float32)
        Df['lo_impossible'] = (dd.notna() & (dd > 6)).astype(np.float32)
    cols += ['social_ub', 'gaming_ub', 'work_ub', 'daily_lb', 'hi_impossible', 'lo_impossible']

    # mc: Monte-Carlo posterior mechanism features; impq: imputation quantiles (both target-free)
    for nm in ('mc', 'impq'):
        Ptr, Pte = pd.read_parquet(C / f'{nm}_tr.parquet'), pd.read_parquet(C / f'{nm}_te.parquet')
        Xtr, Xte = pd.concat([Xtr, Ptr], axis=1), pd.concat([Xte, Pte], axis=1)
        cols += list(Ptr.columns)
    cols += IMP
    cols = list(dict.fromkeys(cols))
    A, B = Xtr[cols].copy(), Xte[cols].copy()

    # den: masked-density model log-probabilities / entropies
    A = pd.concat([A, pd.read_parquet(C / 'den_tr.parquet')], axis=1)
    B = pd.concat([B, pd.read_parquet(C / 'den_te.parquet')], axis=1)

    # lat: lattice coordinates of the hour columns
    for c in ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
              'weekend_screen_time']:
        for D in (A, B):
            v = D[c].values
            D[f'frac_{c}'] = (v - np.floor(v)).astype(np.float32)
            D[f'd1_{c}'] = (np.floor(v * 10) % 10).astype(np.float32)

    # te: nested target encodings of the 12 exact-value tokens and 3 pair keys, inner-fold noise averaged over 3 seeds
    keys_tr = [(f'te_{c}', tok_tr[c]) for c in FEATS]
    keys_te = [(f'te_{c}', tok_te[c]) for c in FEATS]

    def r1key(sa, sb):
        va = sa.round(1).map(lambda x: f'{x:.1f}' if pd.notna(x) else '__NA__')
        vb = sb.round(1).map(lambda x: f'{x:.1f}' if pd.notna(x) else '__NA__')
        return va + '|' + vb
    keys_tr += [('te2_ds', r1key(train['daily_screen_time_hours'], train['social_media_hours'])),
                ('te2_dw', r1key(train['daily_screen_time_hours'], train['weekend_screen_time'])),
                ('te2_na', tok_tr['notifications_per_day'] + '|' + tok_tr['app_opens_per_day'])]
    keys_te += [('te2_ds', r1key(test['daily_screen_time_hours'], test['social_media_hours'])),
                ('te2_dw', r1key(test['daily_screen_time_hours'], test['weekend_screen_time'])),
                ('te2_na', tok_te['notifications_per_day'] + '|' + tok_te['app_opens_per_day'])]
    te_trn = None
    for sd_ in TE_SEEDS:
        t_, full_maps = build_te_fold(y, trn_idx, keys_tr, inner_seed=sd_)
        if te_trn is None:
            te_trn = {k: v.astype(np.float64) for k, v in t_.items()}
        else:
            for k in te_trn:
                te_trn[k] += t_[k]
    te_trn = {k: (v / len(TE_SEEDS)).astype(np.float32) for k, v in te_trn.items()}
    te_val = apply_te(full_maps, [(n, s.iloc[val_idx].reset_index(drop=True)) for n, s in keys_tr])
    te_test = apply_te(full_maps, keys_te)
    for name, _ in keys_tr:
        col_tr = np.full(len(train), np.nan, np.float32)
        col_tr[trn_idx] = te_trn[name]
        col_tr[val_idx] = te_val[name]
        A[name] = col_tr
        B[name] = te_test[name]

    # te2s: second (sharp) smoothing scale for the 12 single columns
    keys1 = [(f'tes_{c}', tok_tr[c]) for c in FEATS]
    keys1t = [(f'tes_{c}', tok_te[c]) for c in FEATS]
    te_trn2, full_maps2 = build_te_fold(y, trn_idx, keys1, smooth=4.0)
    te_val2 = apply_te(full_maps2, [(n, s.iloc[val_idx].reset_index(drop=True)) for n, s in keys1])
    te_test2 = apply_te(full_maps2, keys1t)
    for name, _ in keys1:
        col_tr = np.full(len(train), np.nan, np.float32)
        col_tr[trn_idx] = te_trn2[name]
        col_tr[val_idx] = te_val2[name]
        A[name] = col_tr
        B[name] = te_test2[name]

    # woe: WOE sums of the te columns (all 9 numerics; daily / social / weekend)
    lp_w = np.log(y[trn_idx].mean() / (1 - y[trn_idx].mean()))

    def _wsum(frame, names):
        w = np.zeros(len(frame))
        for nm in names:
            p = np.clip(frame[nm].values, 1e-4, 1 - 1e-4)
            w += np.where(np.isnan(frame[nm].values), 0, np.log(p / (1 - p)) - lp_w)
        return w.astype(np.float32)
    s9 = [f'te_{c}' for c in NUMS]
    s3 = [f'te_{c}' for c in ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time']]
    for D in (A, B):
        D['woe_sum9'] = _wsum(D, s9)
        D['woe_sum_dsw'] = _wsum(D, s3)
    assert not A.columns.duplicated().any()
    log(f'assembled A{A.shape} fold={fold}')

    if rich:
        A, B = append_channels(A, B, RICH, fold, log)
    return dict(A=A, B=B, y=y, trn_idx=trn_idx, val_idx=val_idx)
