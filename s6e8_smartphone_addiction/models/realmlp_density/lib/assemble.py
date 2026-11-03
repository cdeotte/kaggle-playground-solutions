"""Fold-wise channel matrix shared by the RealMLP, TabM and lookup-transformer runners.

assemble(fold, cfg, te_smooth, te2s_smooth, den) -> dict(A=train frame, B=test frame, y, trn_idx, val_idx, train, test)

Target-free channels and fixed original-data evidence are read from the files of ../density_features (cache):
  static_{tr,te}.parquet  raw values (categoricals as fixed ordinal codes), missing flags / counts, budget geometry,
                          exact-value counts, original-data LLR / amplification, 18 pair PMIs, screen imputers and
                          fills, 3-class region posterior
  tok_{tr,te}.parquet     exact-value tokens of the 12 columns ('%.2f', '__NA__')
  mc_{tr,te}.parquet      Monte-Carlo posterior block, knn_f{fold}.parquet  kNN label responses in the MLM row-embedding
                          space (fold-wise), <den>_{tr,te}.parquet  MLM per-cell log p(x_c | rest), entropy, row PLL
Label-derived channels (target encodings, WOE sums) are nested inside the outer fold: every outer-train row gets the
encoding fitted on the other 4 inner folds (StratifiedKFold(5, shuffle=True, random_state=777)), outer-validation and
test rows get the map fitted on all outer-train rows.

cfg flags (comma-separated; the raw columns, missing flags and budget geometry are always in):
  cnt    exact-value counts                     orig   original-data LLR / amplification
  pmi    18 pair PMIs                           imp    screen imputers, fills, region posterior
  mc     Monte-Carlo posterior block            knn    kNN responses
  den    MLM density channels (<den> files)
  te     TE (smoothing te_smooth) of the 12 exact tokens + 3 pair keys (daily x social and daily x weekend on the 0.1
         grid, notifications x app_opens exact)
  rte    + 5 region x token keys                geo    + 8 quantile-binned geometry keys
  teavg  the te encodings averaged over 3 inner seeds (777, 1778, 2779)
  te2s   a second TE of the 12 tokens at smoothing te2s_smooth
  woe    2 WOE sums of the te_ columns
  slim   drop the per-column missing flags, the ollr1_ columns and the categorical ollr_ / amp_ columns
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.data import CATEGORICALS, NUMERICS, TARGET, cache_dir, folds, labels, read_test, read_train

NUMS, CATS = list(NUMERICS), list(CATEGORICALS)
FEATS = NUMS + CATS
BASE_CFG = 'cnt,te,te2s,orig,pmi,imp,geo,rte,den,knn,mc'
BUDGET = ['sgw', 'slack', 'slack_frac', 'share_social_media_hours', 'share_gaming_hours', 'share_work_study_hours',
          'wk_minus_d', 'wk_minus_sgw', 'wk_ratio', 'd_m6', 'd_m8', 's_m4', 'region_obs']
RTE_COLS = ['notifications_per_day', 'app_opens_per_day', 'age', 'sleep_hours', 'gender']
GEO_KEYS = [('slack', 64), ('share_social_media_hours', 16), ('wk_minus_d', 64), ('fill_slack', 64), ('fill_d_m8', 64),
            ('p_reg_hi', 64), ('imp_daily_screen_time_hours', 64), ('fill_s_m4', 64)]
SLIM_DROP = ('amp_gender', 'amp_stress_level', 'amp_academic_work_impact', 'ollr_gender', 'ollr_stress_level',
             'ollr_academic_work_impact')


def density_file(name):
    return cache_dir('density_features') / name


def te_map(vals, yy, prior, smooth):
    g = pd.DataFrame({'v': vals, 'y': yy}).groupby('v')['y'].agg(['mean', 'size'])
    return (g['mean'] * g['size'] + prior * smooth) / (g['size'] + smooth)


def nested_te(y, trn_idx, keys, smooth, inner_seed=777, n_inner=5):
    """keys: [(name, token Series over the train rows)]. Returns the inner-out-of-fold encoding of the outer-train rows
    and the maps fitted on all outer-train rows (applied to validation / test rows)."""
    prior = y[trn_idx].mean()
    te_trn = {name: np.full(len(trn_idx), np.nan, np.float32) for name, _ in keys}
    inner = StratifiedKFold(n_inner, shuffle=True, random_state=inner_seed)
    for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y[trn_idx]):
        fit_rows = trn_idx[i_fit]
        for name, tk in keys:
            mp = te_map(tk.iloc[fit_rows].values, y[fit_rows], prior, smooth)
            te_trn[name][i_app] = pd.Series(tk.iloc[trn_idx[i_app]].values).map(mp).fillna(prior).values
    full_maps = {name: (te_map(tk.iloc[trn_idx].values, y[trn_idx], prior, smooth), prior) for name, tk in keys}
    return te_trn, full_maps


def apply_te(full_maps, keys):
    out = {}
    for name, tk in keys:
        mp, prior = full_maps[name]
        out[name] = tk.map(mp).fillna(prior).values.astype(np.float32)
    return out


def r1_token(s):
    return s.round(1).map(lambda x: f'{x:.1f}' if pd.notna(x) else '__NA__')


def te_keys(raw, tok, X, cfg):
    """(name, token Series) of the te / rte keys of one frame: raw csv frame, its exact tokens, its static frame."""
    keys = [(f'te_{c}', tok[c]) for c in FEATS]
    keys += [('te2_ds', r1_token(raw['daily_screen_time_hours']) + '|' + r1_token(raw['social_media_hours'])),
             ('te2_dw', r1_token(raw['daily_screen_time_hours']) + '|' + r1_token(raw['weekend_screen_time'])),
             ('te2_na', tok['notifications_per_day'] + '|' + tok['app_opens_per_day'])]
    if 'rte' in cfg:
        region = X['region_obs'].map(lambda v: 'm' if pd.isna(v) else str(int(v)))
        keys += [(f'ter_{c}', region + '|' + tok[c]) for c in RTE_COLS]
    return keys


def add_te_columns(A, B, keys_tr, keys_te, te_trn, full_maps, trn_idx, val_idx):
    te_val = apply_te(full_maps, [(n, s.iloc[val_idx].reset_index(drop=True)) for n, s in keys_tr])
    te_test = apply_te(full_maps, keys_te)
    for name, _ in keys_tr:
        col = np.full(len(A), np.nan, np.float32)
        col[trn_idx] = te_trn[name]
        col[val_idx] = te_val[name]
        A[name] = col
        B[name] = te_test[name]


def woe_sum(frame, names, lp):
    w = np.zeros(len(frame))
    for nm in names:
        p = np.clip(frame[nm].values, 1e-4, 1 - 1e-4)
        w += np.where(np.isnan(frame[nm].values), 0, np.log(p / (1 - p)) - lp)
    return w.astype(np.float32)


def assemble(fold, cfg, te_smooth=20.0, te2s_smooth=4.0, den='den384'):
    """den: stem of the MLM density files read by the den flag (den384 = d384 / 6-layer / 24-epoch MLM,
    den = d320 / 7-layer / 25-epoch MLM)."""
    cfg = set(cfg.split(',')) if isinstance(cfg, str) else set(cfg)
    train, test = read_train(), read_test()
    y = labels(train[TARGET]).astype(np.int64)
    f = folds(y)
    trn_idx, val_idx = np.where(f != fold)[0], np.where(f == fold)[0]

    Xtr = pd.read_parquet(density_file('static_tr.parquet'))
    Xte = pd.read_parquet(density_file('static_te.parquet'))
    tok_tr = pd.read_parquet(density_file('tok_tr.parquet'))
    tok_te = pd.read_parquet(density_file('tok_te.parquet'))

    missing = [c for c in Xtr.columns if c.startswith('na_')] + ['n_missing', 'na_screen', 'na_beh']
    counts = [c for c in Xtr.columns if c.startswith('cnt_')]
    orig = [c for c in Xtr.columns if c.startswith(('ollr_', 'ollr1_', 'amp_'))] + ['ollr_sum']
    pmi = [c for c in Xtr.columns if c.startswith('pmi_')]
    imp = [c for c in Xtr.columns if c.startswith(('imp_', 'fill_'))] + ['p_reg_lo', 'p_reg_mid', 'p_reg_hi', 'reg_exp']

    cols = FEATS + missing + BUDGET
    if 'cnt' in cfg:
        cols += counts
    if 'orig' in cfg:
        cols += orig
    if 'pmi' in cfg:
        cols += pmi
    if 'mc' in cfg:
        Mtr, Mte = pd.read_parquet(density_file('mc_tr.parquet')), pd.read_parquet(density_file('mc_te.parquet'))
        Xtr, Xte = pd.concat([Xtr, Mtr], axis=1), pd.concat([Xte, Mte], axis=1)
        cols += list(Mtr.columns)
    if 'knn' in cfg:   # one file per outer fold: train rows then test rows
        K = pd.read_parquet(density_file(f'knn_f{fold}.parquet'))
        Xtr = pd.concat([Xtr, K.iloc[:len(train)].reset_index(drop=True)], axis=1)
        Xte = pd.concat([Xte, K.iloc[len(train):].reset_index(drop=True)], axis=1)
        cols += ['knn_r16', 'knn_r64', 'knn_rw', 'knn_dist']
    if 'imp' in cfg:
        cols += imp
    cols = list(dict.fromkeys(cols))
    A, B = Xtr[cols].copy(), Xte[cols].copy()

    if 'den' in cfg:
        Dtr = pd.read_parquet(density_file(f'{den}_tr.parquet'))
        Dte = pd.read_parquet(density_file(f'{den}_te.parquet'))
        A, B = pd.concat([A, Dtr], axis=1), pd.concat([B, Dte], axis=1)

    if 'te' in cfg:
        keys_tr, keys_te = te_keys(train, tok_tr, Xtr, cfg), te_keys(test, tok_te, Xte, cfg)
        if 'geo' in cfg:   # quantile bins over train + test
            for gcol, q in GEO_KEYS:
                edges = np.unique(np.nanquantile(pd.concat([Xtr[gcol], Xte[gcol]]), np.linspace(0, 1, q + 1)))
                keys_tr.append((f'teg_{gcol}', pd.cut(Xtr[gcol], edges, labels=False, include_lowest=True).astype(str)))
                keys_te.append((f'teg_{gcol}', pd.cut(Xte[gcol], edges, labels=False, include_lowest=True).astype(str)))
        n_seeds = 3 if 'teavg' in cfg else 1
        acc = None
        for s in range(n_seeds):
            t_, full_maps = nested_te(y, trn_idx, keys_tr, te_smooth, inner_seed=777 + 1001 * s)
            acc = {k: v.astype(np.float64) for k, v in t_.items()} if acc is None else {k: acc[k] + t_[k] for k in acc}
        te_trn = {k: (v / n_seeds).astype(np.float32) for k, v in acc.items()}
        add_te_columns(A, B, keys_tr, keys_te, te_trn, full_maps, trn_idx, val_idx)

        if 'te2s' in cfg:   # the 12 single tokens at a second smoothing
            k1_tr = [(f'tes_{c}', tok_tr[c]) for c in FEATS]
            k1_te = [(f'tes_{c}', tok_te[c]) for c in FEATS]
            t2, maps2 = nested_te(y, trn_idx, k1_tr, te2s_smooth)
            add_te_columns(A, B, k1_tr, k1_te, t2, maps2, trn_idx, val_idx)

        if 'woe' in cfg:   # WOE sums over the 9 numerics and over daily / social / weekend
            prior = y[trn_idx].mean()
            lp = np.log(prior / (1 - prior))
            s9 = [f'te_{c}' for c in NUMS]
            s3 = ['te_daily_screen_time_hours', 'te_social_media_hours', 'te_weekend_screen_time']
            for D in (A, B):
                D['woe_sum9'] = woe_sum(D, s9, lp)
                D['woe_sum_dsw'] = woe_sum(D, s3, lp)

    if 'slim' in cfg:
        drop = [c for c in A.columns if (c.startswith('na_') and c not in ('na_screen', 'na_beh'))
                or c.startswith('ollr1_') or c in SLIM_DROP]
        A, B = A.drop(columns=drop), B.drop(columns=drop)

    assert not A.columns.duplicated().any()
    return dict(A=A, B=B, y=y, trn_idx=trn_idx, val_idx=val_idx, train=train, test=test)
