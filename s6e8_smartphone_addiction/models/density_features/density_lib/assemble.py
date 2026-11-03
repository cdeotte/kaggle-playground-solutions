"""Fold-wise feature assembly shared by the XGBoost / RealMLP / TabM / CatBoost / transformer / cuML runners.

assemble(fold, cfg, par) -> dict(A = train frame (all train rows), B = test frame, y, trn_idx, val_idx, margin_tr,
margin_te (or None), train, test).

Label-derived parts (target encodings, margins) are inner-cross-fitted on the outer-train rows only: outer-train rows get
encodings from maps fit on the other inner groups (StratifiedKFold(5, shuffle, seed 777)), outer-validation and test rows
from maps fit on all outer-train rows. Static parts are label-free (train + test predictors) or fixed original-data
evidence and are read from the cache of this folder.

cfg flags (a set of strings). Always: the 12 raw columns + missing flags + budget geometry. Optional:
  cnt / orig / pmi (pmi10)   value counts / original-data amplification + LLRs / 18 (10) exact pair PMIs
  bnd                        interval bounds implied by the budget identity + region-impossibility flags
  mc / impq / imp            Monte-Carlo posterior block / quantile imputers / point imputers, fills, region posterior
  knn                        kNN label responses in the MLM row-embedding space (knn<par ktag>_f<fold>.parquet)
  den                        MLM density channels (den<par dtag>_{tr,te}.parquet; dtag '' = den_*, '384' = den384_*)
  cllr                       inner-cross-fitted class-conditional MLM LLRs (cllr_f<fold>.parquet)
  lat                        fractional part and first decimal of the 6 hour columns
  te                         nested exact-token TE of the 12 columns + 3 pair keys, smoothing par te_smooth (20);
                             with hier / te3 / ter1 / rte / lat2 / geo2 / geo extra keys; teord (ordered statistics) or
                             teavg (inner encodings averaged over par n_te_seeds (3) inner seeds); te2s (second
                             smoothing par te2s_smooth (4) of the 12 single keys); woe (WOE sums)
  mrg / mrg2                 WOE base margin (scale par mrg_scale 0.4; mrg_src 'tes' uses the te2s encodings)
  slim                       drop the per-column missing flags, ollr1_* and the categorical amp/ollr columns
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .base import CACHE, CATS, FEATS, NUMS, TARGET, get_folds, load_comp
from .features import TE_SMOOTH, apply_te, build_te_fold


def assemble(FOLD, CFG, PAR=None):
    PAR = PAR or {}
    train, test = load_comp()
    y = train[TARGET].values
    folds = get_folds(y)
    trn_idx = np.where(folds != FOLD)[0]
    val_idx = np.where(folds == FOLD)[0]

    Xtr = pd.read_parquet(CACHE / 'static_tr.parquet')
    Xte = pd.read_parquet(CACHE / 'static_te.parquet')
    tok_tr = pd.read_parquet(CACHE / 'tok_tr.parquet')
    tok_te = pd.read_parquet(CACHE / 'tok_te.parquet')

    RAW = NUMS + CATS
    MISSING = [c for c in Xtr.columns if c.startswith('na_')] + ['n_missing', 'na_screen', 'na_beh']
    BUDGET = ['sgw', 'slack', 'slack_frac', 'share_social_media_hours',
              'share_gaming_hours', 'share_work_study_hours', 'wk_minus_d',
              'wk_minus_sgw', 'wk_ratio', 'd_m6', 'd_m8', 's_m4', 'region_obs']
    COUNTS = [c for c in Xtr.columns if c.startswith('cnt_')]
    ORIG = [c for c in Xtr.columns if c.startswith(('ollr_', 'ollr1_', 'amp_'))] + ['ollr_sum']
    PMI = [c for c in Xtr.columns if c.startswith('pmi_')]
    IMP = [c for c in Xtr.columns if c.startswith(('imp_', 'fill_'))] + ['p_reg_lo', 'p_reg_mid', 'p_reg_hi', 'reg_exp']

    cols = RAW + MISSING + BUDGET
    if 'cnt' in CFG: cols += COUNTS
    if 'orig' in CFG: cols += ORIG
    if 'pmi' in CFG: cols += PMI
    elif 'pmi10' in CFG: cols += PMI[:10]

    if 'bnd' in CFG:   # exact interval bounds implied by the budget identity
        d_ = Xtr['daily_screen_time_hours']; d2_ = Xte['daily_screen_time_hours']
        s_ = Xtr['social_media_hours']; s2_ = Xte['social_media_hours']
        g_ = Xtr['gaming_hours']; g2_ = Xte['gaming_hours']
        w_ = Xtr['work_study_hours']; w2_ = Xte['work_study_hours']
        for Df, dd, ss, gg, ww in [(Xtr, d_, s_, g_, w_), (Xte, d2_, s2_, g2_, w2_)]:
            Df['social_ub'] = (dd - gg - ww).astype(np.float32)
            Df['gaming_ub'] = (dd - ss - ww).astype(np.float32)
            Df['work_ub'] = (dd - ss - gg).astype(np.float32)
            Df['daily_lb'] = (ss + gg + ww).astype(np.float32)
            # region-impossibility proofs usable when social is missing
            hi_imp = (dd.notna() & (dd <= 8) & ss.isna() & Df['social_ub'].notna() & (Df['social_ub'] <= 4))
            Df['hi_impossible'] = hi_imp.astype(np.float32)
            lo_imp = (dd.notna() & (dd > 6))
            Df['lo_impossible'] = lo_imp.astype(np.float32)
        cols += ['social_ub', 'gaming_ub', 'work_ub', 'daily_lb', 'hi_impossible', 'lo_impossible']

    if 'mc' in CFG:   # Monte-Carlo posterior mechanism features (label-free)
        Mtr = pd.read_parquet(CACHE / 'mc_tr.parquet')
        Mte = pd.read_parquet(CACHE / 'mc_te.parquet')
        Xtr = pd.concat([Xtr, Mtr], axis=1); Xte = pd.concat([Xte, Mte], axis=1)
        cols += list(Mtr.columns)

    if 'impq' in CFG:
        Qtr = pd.read_parquet(CACHE / 'impq_tr.parquet')
        Qte = pd.read_parquet(CACHE / 'impq_te.parquet')
        Xtr = pd.concat([Xtr, Qtr], axis=1); Xte = pd.concat([Xte, Qte], axis=1)
        cols += list(Qtr.columns)

    if 'knn' in CFG:
        ktag = PAR.get('ktag', '')
        K = pd.read_parquet(CACHE / f'knn{ktag}_f{FOLD}.parquet')
        Xtr = pd.concat([Xtr, K.iloc[:len(train)].reset_index(drop=True)], axis=1)
        Xte = pd.concat([Xte, K.iloc[len(train):].reset_index(drop=True)], axis=1)
        cols += ['knn_r16', 'knn_r64', 'knn_rw', 'knn_dist']
    if 'imp' in CFG: cols += IMP
    cols = list(dict.fromkeys(cols))
    A = Xtr[cols].copy(); B = Xte[cols].copy()

    if 'den' in CFG:
        dtag = PAR.get('dtag', '')
        Dtr = pd.read_parquet(CACHE / f'den{dtag}_tr.parquet')
        Dte = pd.read_parquet(CACHE / f'den{dtag}_te.parquet')
        A = pd.concat([A, Dtr], axis=1); B = pd.concat([B, Dte], axis=1)

    if 'cllr' in CFG:
        C = pd.read_parquet(CACHE / f'cllr_f{FOLD}.parquet')
        Ctr = C.iloc[:len(train)].reset_index(drop=True)
        Cte = C.iloc[len(train):].reset_index(drop=True)
        A = pd.concat([A, Ctr], axis=1); B = pd.concat([B, Cte], axis=1)

    if 'lat' in CFG:
        for c in ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                  'work_study_hours', 'sleep_hours', 'weekend_screen_time']:
            for D in (A, B):
                v = D[c].values
                D[f'frac_{c}'] = (v - np.floor(v)).astype(np.float32)
                D[f'd1_{c}'] = (np.floor(v * 10) % 10).astype(np.float32)

    # ---------------- fold-wise target encodings ----------------
    if 'te' in CFG:
        keys_tr, keys_te = [], []
        for c in FEATS:
            keys_tr.append((f'te_{c}', tok_tr[c]))
            keys_te.append((f'te_{c}', tok_te[c]))

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
        if 'hier' in CFG:   # hierarchical empirical-Bayes TE: exact <- 0.1-rounded <- global
            prior_h = y[trn_idx].mean(); K_EX, K_R1 = 8.0, 40.0
            hcols = ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time',
                     'work_study_hours', 'gaming_hours', 'sleep_hours']

            def hmaps(rows):
                out = {}
                for c in hcols:
                    v = train[c].iloc[rows]
                    ex = v.round(2); r1 = v.round(1)
                    yy = pd.Series(y[rows], index=v.index)
                    g1 = yy.groupby(r1).agg(['mean', 'size'])
                    m1 = (g1['mean'] * g1['size'] + prior_h * K_R1) / (g1['size'] + K_R1)
                    gx = pd.DataFrame({'y': yy, 'ex': ex, 'r1': r1})
                    ge = gx.groupby('ex').agg(n=('y', 'size'), s=('y', 'sum'), r1=('r1', 'first'))
                    base = ge['r1'].map(m1).fillna(prior_h)
                    me = (ge['s'] + base * K_EX) / (ge['n'] + K_EX)
                    out[c] = (me, m1)
                return out

            def happly(maps, df_rows):
                res = {}
                for c in hcols:
                    me, m1 = maps[c]
                    v = df_rows[c]
                    ex = v.round(2).map(me)
                    r1 = v.round(1).map(m1)
                    res[f'teh_{c}'] = ex.fillna(r1).fillna(prior_h).values.astype(np.float32)
                return res
            Htr = {f'teh_{c}': np.full(len(train), np.nan, np.float32) for c in hcols}
            inner_h = StratifiedKFold(5, shuffle=True, random_state=777)
            for i_fit, i_app in inner_h.split(np.zeros(len(trn_idx)), y[trn_idx]):
                mp = hmaps(trn_idx[i_fit])
                r = happly(mp, train.iloc[trn_idx[i_app]])
                for k2, v2 in r.items(): Htr[k2][trn_idx[i_app]] = v2
            mp = hmaps(trn_idx)
            rv = happly(mp, train.iloc[val_idx]); rt = happly(mp, test)
            for c in hcols:
                Htr[f'teh_{c}'][val_idx] = rv[f'teh_{c}']
                A[f'teh_{c}'] = Htr[f'teh_{c}']
                B[f'teh_{c}'] = rt[f'teh_{c}']
        if 'te3' in CFG:   # age-centred pair keys
            a_tr, a_te = tok_tr['age'], tok_te['age']
            sl_tr = train['sleep_hours'].round(1).map(lambda x: f'{x:.1f}' if pd.notna(x) else 'N')
            sl_te = test['sleep_hours'].round(1).map(lambda x: f'{x:.1f}' if pd.notna(x) else 'N')
            keys_tr += [('te3_ag', a_tr + '|' + tok_tr['gender']),
                        ('te3_as', a_tr + '|' + sl_tr),
                        ('te3_an', a_tr + '|' + tok_tr['notifications_per_day'])]
            keys_te += [('te3_ag', a_te + '|' + tok_te['gender']),
                        ('te3_as', a_te + '|' + sl_te),
                        ('te3_an', a_te + '|' + tok_te['notifications_per_day'])]
        if 'ter1' in CFG:   # 0.1-rounded single-column TE for the screen / sleep columns
            for c in ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time',
                      'gaming_hours', 'work_study_hours', 'sleep_hours']:
                k_tr = train[c].round(1).map(lambda x: f'{x:.1f}' if pd.notna(x) else 'N')
                k_te = test[c].round(1).map(lambda x: f'{x:.1f}' if pd.notna(x) else 'N')
                keys_tr.append((f'ter1_{c}', k_tr))
                keys_te.append((f'ter1_{c}', k_te))
        if 'rte' in CFG:   # observed region x exact token
            rtr = Xtr['region_obs'].map(lambda v: 'm' if pd.isna(v) else str(int(v)))
            rte_ = Xte['region_obs'].map(lambda v: 'm' if pd.isna(v) else str(int(v)))
            for c in ['notifications_per_day', 'app_opens_per_day', 'age', 'sleep_hours', 'gender']:
                keys_tr.append((f'ter_{c}', rtr + '|' + tok_tr[c]))
                keys_te.append((f'ter_{c}', rte_ + '|' + tok_te[c]))
        if 'lat2' in CFG:   # all 36 numeric pairs on the 0.1 lattice + the (daily, social, weekend) cell
            def r1tok(s):
                return s.round(1).map(lambda x: f'{x:.1f}' if pd.notna(x) else 'N')
            r1_tr = {c: r1tok(train[c]) for c in NUMS}
            r1_te = {c: r1tok(test[c]) for c in NUMS}
            for i in range(len(NUMS)):
                for jj in range(i + 1, len(NUMS)):
                    a, b = NUMS[i], NUMS[jj]
                    nm = f'tl_{i}_{jj}'
                    keys_tr.append((nm, r1_tr[a] + '|' + r1_tr[b]))
                    keys_te.append((nm, r1_te[a] + '|' + r1_te[b]))
            keys_tr.append(('tl3_dsw', r1_tr['daily_screen_time_hours'] + '|' +
                            r1_tr['social_media_hours'] + '|' + r1_tr['weekend_screen_time']))
            keys_te.append(('tl3_dsw', r1_te['daily_screen_time_hours'] + '|' +
                            r1_te['social_media_hours'] + '|' + r1_te['weekend_screen_time']))
        if 'geo2' in CFG:   # joint key: region x slack bin x share bin
            rtr2 = Xtr['region_obs'].map(lambda v: 'm' if pd.isna(v) else str(int(v)))
            rte2 = Xte['region_obs'].map(lambda v: 'm' if pd.isna(v) else str(int(v)))

            def qb(col, q):
                allv = pd.concat([Xtr[col], Xte[col]])
                edges = np.unique(np.nanquantile(allv, np.linspace(0, 1, q + 1)))
                return (pd.cut(Xtr[col], edges, labels=False, include_lowest=True).astype(str),
                        pd.cut(Xte[col], edges, labels=False, include_lowest=True).astype(str))
            sl_tr, sl_te = qb('fill_slack', 16)
            sh_tr, sh_te = qb('fill_share_social' if 'fill_share_social' in Xtr.columns
                              else 'share_social_media_hours', 8)
            keys_tr.append(('teg2_rss', rtr2 + '|' + sl_tr + '|' + sh_tr))
            keys_te.append(('teg2_rss', rte2 + '|' + sl_te + '|' + sh_te))
        if 'geo' in CFG:   # quantile bins (train + test edges) of 8 geometry / imputer columns
            for gcol, q in [('slack', 64), ('share_social_media_hours', 16),
                            ('wk_minus_d', 64), ('fill_slack', 64), ('fill_d_m8', 64),
                            ('p_reg_hi', 64), ('imp_daily_screen_time_hours', 64),
                            ('fill_s_m4', 64)]:
                if gcol not in Xtr.columns: continue
                allv = pd.concat([Xtr[gcol], Xte[gcol]])
                edges = np.unique(np.nanquantile(allv, np.linspace(0, 1, q + 1)))
                btr = pd.cut(Xtr[gcol], edges, labels=False, include_lowest=True)
                bte = pd.cut(Xte[gcol], edges, labels=False, include_lowest=True)
                keys_tr.append((f'teg_{gcol}', btr.astype(str)))
                keys_te.append((f'teg_{gcol}', bte.astype(str)))
        if 'teord' in CFG:   # ordered target statistics (CatBoost-style), 3 permutations
            _, full_maps = build_te_fold(train, tok_tr, y, trn_idx, keys_tr, n_inner=int(PAR.get('n_inner', 5)))
            prior_o = y[trn_idx].mean()
            rng_o = np.random.default_rng(4242)
            y_trn = y[trn_idx]
            te_trn = {}
            for name, tk in keys_tr:
                kv = tk.iloc[trn_idx].values
                acc = np.zeros(len(trn_idx), np.float64)
                for _p in range(3):
                    order = rng_o.permutation(len(trn_idx))
                    df_o = pd.DataFrame({'k': kv[order], 'y': y_trn[order]})
                    g = df_o.groupby('k', sort=False)['y']
                    cs = g.cumsum() - df_o['y']          # sum of the preceding same-key labels
                    cc = g.cumcount()                     # count of the preceding same-key rows
                    enc = (cs + prior_o * TE_SMOOTH) / (cc + TE_SMOOTH)
                    acc[order] += enc.values
                te_trn[name] = (acc / 3).astype(np.float32)
        elif 'teavg' in CFG:   # average the inner-fold encodings over several inner seeds
            n_seeds = int(PAR.get('n_te_seeds', 3))
            te_trn = None
            for sd_ in tuple(777 + 1001 * k for k in range(n_seeds)):
                t_, full_maps = build_te_fold(train, tok_tr, y, trn_idx, keys_tr, inner_seed=sd_,
                                              n_inner=int(PAR.get('n_inner', 5)),
                                              smooth=float(PAR.get('te_smooth', 20.0)))
                if te_trn is None:
                    te_trn = {k2: v2.astype(np.float64) for k2, v2 in t_.items()}
                else:
                    for k2 in te_trn: te_trn[k2] += t_[k2]
            te_trn = {k2: (v2 / n_seeds).astype(np.float32) for k2, v2 in te_trn.items()}
        else:
            te_trn, full_maps = build_te_fold(train, tok_tr, y, trn_idx, keys_tr,
                                              n_inner=int(PAR.get('n_inner', 5)),
                                              smooth=float(PAR.get('te_smooth', 20.0)))
        te_val = apply_te(full_maps, [(n, s.iloc[val_idx].reset_index(drop=True)) for n, s in keys_tr])
        te_test = apply_te(full_maps, keys_te)
        for name, _ in keys_tr:
            col_tr = np.full(len(train), np.nan, np.float32)
            col_tr[trn_idx] = te_trn[name]
            col_tr[val_idx] = te_val[name]
            A[name] = col_tr
            B[name] = te_test[name]
        if 'te2s' in CFG:   # second smoothing scale for the 12 single columns
            keys1 = [(f'tes_{c}', tok_tr[c]) for c in FEATS]
            keys1t = [(f'tes_{c}', tok_te[c]) for c in FEATS]
            te_trn2, full_maps2 = build_te_fold(train, tok_tr, y, trn_idx, keys1,
                                                smooth=float(PAR.get('te2s_smooth', 4.0)))
            te_val2 = apply_te(full_maps2, [(n, s.iloc[val_idx].reset_index(drop=True)) for n, s in keys1])
            te_test2 = apply_te(full_maps2, keys1t)
            for name, _ in keys1:
                col_tr = np.full(len(train), np.nan, np.float32)
                col_tr[trn_idx] = te_trn2[name]
                col_tr[val_idx] = te_val2[name]
                A[name] = col_tr
                B[name] = te_test2[name]

        if 'woe' in CFG:   # WOE sums: all 9 numerics and daily / social / weekend
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

    margin_tr = margin_te = None
    if 'mrg' in CFG:   # WOE base margin: prior logit + scale * sum of the single-column TE log-odds ratios
        prior = y[trn_idx].mean(); lp = np.log(prior / (1 - prior))
        scale = float(PAR.get('mrg_scale', 0.4))

        def woe_sum(frame, names):
            w = np.zeros(len(frame))
            for nm in names:
                col = frame[nm].values
                p = np.clip(col, 1e-4, 1 - 1e-4)
                w += np.where(np.isnan(col), 0, np.log(p / (1 - p)) - lp)
            return w
        pref = 'tes_' if PAR.get('mrg_src') == 'tes' else 'te_'
        singles = [f'{pref}{c}' for c in FEATS if f'{pref}{c}' in A.columns] or [f'te_{c}' for c in FEATS]
        wtr = woe_sum(A, singles); wte = woe_sum(B, singles)
        margin_tr = (lp + scale * wtr)
        margin_te = (lp + scale * wte)
        if 'mrg2' in CFG:   # + pair / region / geometry WOE at a lower scale
            s2 = float(PAR.get('mrg2_scale', 0.25))
            extra = [c for c in A.columns if c.startswith(('te2_', 'ter_', 'teg_'))]
            margin_tr = margin_tr + s2 * woe_sum(A, extra)
            margin_te = margin_te + s2 * woe_sum(B, extra)
        margin_tr = margin_tr.astype(np.float32)
        margin_te = margin_te.astype(np.float32)

    if 'slim' in CFG:   # drop low-value columns to concentrate column sampling
        drop = [c for c in A.columns
                if (c.startswith('na_') and c not in ('na_screen', 'na_beh'))
                or c.startswith('ollr1_')
                or c in ('amp_gender', 'amp_stress_level', 'amp_academic_work_impact',
                         'ollr_gender', 'ollr_stress_level', 'ollr_academic_work_impact')]
        A = A.drop(columns=drop); B = B.drop(columns=drop)

    assert not A.columns.duplicated().any()
    return dict(A=A, B=B, y=y, trn_idx=trn_idx, val_idx=val_idx,
                margin_tr=margin_tr, margin_te=margin_te, train=train, test=test)
