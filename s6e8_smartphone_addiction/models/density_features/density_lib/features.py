"""The static feature block and the nested target-encoding helpers.

Static channels are label-free (computed on the train + test predictors) or fixed evidence from the original dataset; they
are the same for every fold. Label-derived channels are built per outer fold with an inner cross-fit (assemble.py)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .base import COMP, FEATS, NUMS, TARGET


def tok2(s):
    """Canonical exact-value token: numerics as %.2f, NaN -> __NA__."""
    if not pd.api.types.is_numeric_dtype(s):
        return s.astype(object).where(s.notna(), '__NA__').astype(str)
    return s.round(2).map(lambda x: f'{x:.2f}' if pd.notna(x) else '__NA__')


# the 18 numeric pairs with the strongest target association of their exact-pair PMI
PMI_PAIRS = [
    ('daily_screen_time_hours', 'social_media_hours'),
    ('daily_screen_time_hours', 'app_opens_per_day'),
    ('social_media_hours', 'weekend_screen_time'),
    ('daily_screen_time_hours', 'notifications_per_day'),
    ('app_opens_per_day', 'weekend_screen_time'),
    ('gaming_hours', 'work_study_hours'),
    ('social_media_hours', 'work_study_hours'),
    ('social_media_hours', 'app_opens_per_day'),
    ('notifications_per_day', 'weekend_screen_time'),
    ('social_media_hours', 'notifications_per_day'),
    ('daily_screen_time_hours', 'work_study_hours'),
    ('daily_screen_time_hours', 'gaming_hours'),
    ('daily_screen_time_hours', 'weekend_screen_time'),
    ('social_media_hours', 'gaming_hours'),
    ('gaming_hours', 'weekend_screen_time'),
    ('work_study_hours', 'weekend_screen_time'),
    ('notifications_per_day', 'app_opens_per_day'),
    ('gaming_hours', 'app_opens_per_day'),
]


def build_static(train, test, orig, verbose=True):
    n_tr = len(train)
    both = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
    tok = {c: tok2(both[c]) for c in FEATS}
    X = pd.DataFrame(index=both.index)

    # raw numerics; categoricals as fixed ordinal codes
    for c in NUMS:
        X[c] = both[c].astype(np.float32)
    cat_maps = {'gender': {'Male': 0, 'Female': 1, 'Other': 2},
                'stress_level': {'Low': 0, 'Medium': 1, 'High': 2},
                'academic_work_impact': {'No': 0, 'Yes': 1}}
    for c, mp in cat_maps.items():
        X[c] = both[c].map(mp).astype(np.float32)

    # missingness
    M = both[FEATS].isna()
    for c in FEATS:
        X[f'na_{c}'] = M[c].astype(np.float32)
    X['n_missing'] = M.sum(1).astype(np.float32)
    X['na_screen'] = M[['daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                        'work_study_hours', 'weekend_screen_time']].sum(1).astype(np.float32)
    X['na_beh'] = M[['notifications_per_day', 'app_opens_per_day']].sum(1).astype(np.float32)

    # budget geometry: daily >= social + gaming + work
    d = both['daily_screen_time_hours']; wk = both['weekend_screen_time']
    sgw = both[COMP].sum(1, min_count=3)
    X['sgw'] = sgw.astype(np.float32)
    X['slack'] = (d - sgw).astype(np.float32)
    X['slack_frac'] = ((d - sgw) / d.clip(lower=.1)).astype(np.float32)
    for c in COMP:
        X[f'share_{c}'] = (both[c] / d.clip(lower=.1)).astype(np.float32)
    X['wk_minus_d'] = (wk - d).astype(np.float32)
    X['wk_minus_sgw'] = (wk - sgw).astype(np.float32)
    X['wk_ratio'] = (wk / d.clip(lower=.1)).astype(np.float32)

    # region of the label rule from the observed values (NaN where undeterminable)
    s = both['social_media_hours']
    X['d_m6'] = (d - 6).astype(np.float32)
    X['d_m8'] = (d - 8).astype(np.float32)
    X['s_m4'] = (s - 4).astype(np.float32)
    hi = (d > 8) | (s > 4)
    lo = (d <= 6) & (s <= 4)
    region = pd.Series(np.nan, index=both.index)
    region[hi] = 2.0
    region[lo & ~hi] = 0.0
    region[d.notna() & s.notna() & ~hi & ~lo] = 1.0
    region[(s > 4)] = 2.0   # social > 4 alone determines the high region
    X['region_obs'] = region.astype(np.float32)

    # train+test counts, amplification vs the original data and original label LLRs on the exact tokens
    prior_o = orig[TARGET].mean()
    for c in FEATS:
        t = tok[c]
        obs = t != '__NA__'
        cnt = t[obs].value_counts()
        X[f'cnt_{c}'] = t.map(cnt).astype(np.float32)
        X.loc[~obs, f'cnt_{c}'] = np.nan
        ot = tok2(orig[c]); on = ot.value_counts(normalize=True)
        sn = t[obs].value_counts(normalize=True)
        amp = np.log(sn / on).replace([np.inf, -np.inf], np.nan)
        X[f'amp_{c}'] = t.map(amp).astype(np.float32)
        X.loc[~obs, f'amp_{c}'] = np.nan
        # original class-conditional LLR of the exact token, smoothed (k=5) toward the original prior
        og = pd.DataFrame({'v': ot, 'y': orig[TARGET].values}).groupby('v')['y'].agg(['mean', 'size'])
        k = 5.0
        sm = (og['mean'] * og['size'] + prior_o * k) / (og['size'] + k)
        woe = np.log(sm / (1 - sm)) - np.log(prior_o / (1 - prior_o))
        X[f'ollr_{c}'] = t.map(woe).astype(np.float32)
        X.loc[~obs, f'ollr_{c}'] = np.nan
    # the same LLR on 0.1-rounded values for the screen / sleep columns (denser support in the original)
    for c in ['daily_screen_time_hours', 'social_media_hours', 'weekend_screen_time',
              'gaming_hours', 'work_study_hours', 'sleep_hours']:
        ov = orig[c].round(1); bv = both[c].round(1)
        og = pd.DataFrame({'v': ov, 'y': orig[TARGET].values}).groupby('v')['y'].agg(['mean', 'size'])
        k = 5.0
        sm = (og['mean'] * og['size'] + prior_o * k) / (og['size'] + k)
        woe = np.log(sm / (1 - sm)) - np.log(prior_o / (1 - prior_o))
        X[f'ollr1_{c}'] = bv.map(woe).astype(np.float32)
    X['ollr_sum'] = X[[f'ollr_{c}' for c in FEATS]].sum(1, min_count=1).astype(np.float32)

    # observed-only exact pair PMI
    for a, b in PMI_PAIRS:
        ta, tb = tok[a], tok[b]
        ok = (ta != '__NA__') & (tb != '__NA__')
        pa = ta[ok].value_counts(normalize=True)
        pb = tb[ok].value_counts(normalize=True)
        key = ta[ok] + '|' + tb[ok]
        pj = key.value_counts(normalize=True)
        pmi = (np.log(key.map(pj).values) - np.log(ta[ok].map(pa).values)
               - np.log(tb[ok].map(pb).values))
        col = np.full(len(both), np.nan, np.float32)
        col[ok.values] = pmi
        X[f"pmi_{a.split('_')[0]}_{b.split('_')[0]}"] = col

    Xtr = X.iloc[:n_tr].reset_index(drop=True)
    Xte = X.iloc[n_tr:].reset_index(drop=True)
    if verbose:
        print(f'static features: {X.shape[1]} cols')
    return Xtr, Xte, tok


# ---------------------------------------------------------------- nested target encoding
TE_SMOOTH = 20.0


def _te_map(vals, yy, prior, smooth=TE_SMOOTH):
    g = pd.DataFrame({'v': vals, 'y': yy}).groupby('v')['y'].agg(['mean', 'size'])
    return (g['mean'] * g['size'] + prior * smooth) / (g['size'] + smooth)


def build_te_fold(train, tok_tr, y, trn_idx, keys, inner_seed=777, n_inner=5, smooth=TE_SMOOTH):
    """Nested TE for one outer fold. keys: list of (name, token Series on the train index).
    Returns (te_trn: inner-OOF encodings of the outer-train rows, full_maps: maps fit on all outer-train rows, applied to
    the validation and test rows)."""
    prior = y[trn_idx].mean()
    te_trn = {name: np.full(len(trn_idx), np.nan, np.float32) for name, _ in keys}
    inner = StratifiedKFold(n_inner, shuffle=True, random_state=inner_seed)
    for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y[trn_idx]):
        fit_rows = trn_idx[i_fit]
        for name, tk in keys:
            mp = _te_map(tk.iloc[fit_rows].values, y[fit_rows], prior, smooth)
            te_trn[name][i_app] = pd.Series(tk.iloc[trn_idx[i_app]].values).map(mp).fillna(prior).values
    full_maps = {}
    for name, tk in keys:
        full_maps[name] = (_te_map(tk.iloc[trn_idx].values, y[trn_idx], prior, smooth), prior)
    return te_trn, full_maps


def apply_te(full_maps, keys_series):
    out = {}
    for name, tk in keys_series:
        mp, prior = full_maps[name]
        out[name] = tk.map(mp).fillna(prior).values.astype(np.float32)
    return out
