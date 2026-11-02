#!/usr/bin/env python
"""Target-free static blocks: cache/static/<block>_{train,test}.npy + manifest.json (block_names).

They read train + test predictors together and the labels of the original dataset, never the competition labels.
  raw               numeric values + one-hot categoricals (levels over train + test + original), 24 columns
  smooth            sin / cos of the robust z-score at 3 frequencies and 12 Gaussian bumps at quantiles, per numeric field
  semantic          22 hand-made ratios and products (stations, income per car / commute, concern x subsidy x anxiety, ...)
  source_pairs      for 7 field pairs / triples: original-label rate (alpha 20) as logit, lift over the additive per-field
                    logits, log count and support in the original data
  lineage           nearest original income ("donor"): distance, number of candidate donors, best / runner-up number of
                    matching fields among them, hard and soft donor-label logits, entropy of the donor posterior
  smooth_focus      the smooth columns of income and environmental concern
  source_pair_core  8 columns of source_pairs
Usage: python static_cache.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import read_test, read_train  # noqa: E402
from features import CACHE, CATEGORICAL, FEATURES, NUMERIC, read_original_text, save_arrays, source_labels, update_manifest  # noqa: E402

PAIR_SPECS = (
    (('Environmental_Concern_Level', None), ('Subsidy_Available', None)),
    (('Environmental_Concern_Level', None), ('Range_Anxiety_Level', None)),
    (('Subsidy_Available', None), ('Home_Charging_Possible', None)),
    (('City_Type', None), ('Home_Charging_Possible', None)),
    (('Annual_Income_USD', 10000.0), ('Environmental_Concern_Level', None)),
    (('Annual_Income_USD', 10000.0), ('Subsidy_Available', None)),
    (('Annual_Income_USD', 10000.0), ('Environmental_Concern_Level', None), ('Range_Anxiety_Level', None)),
)
SOURCE_PAIR_CORE = (
    'src_pair_logit__Environmental_Concern_Level__Subsidy_Available',
    'src_pair_logcount__Environmental_Concern_Level__Subsidy_Available',
    'src_pair_lift__Environmental_Concern_Level__Subsidy_Available',
    'src_pair_logit__Environmental_Concern_Level__Range_Anxiety_Level',
    'src_pair_logit__Subsidy_Available__Home_Charging_Possible',
    'src_pair_logit__Annual_Income_USD_w10000__Environmental_Concern_Level',
    'src_pair_logit__Annual_Income_USD_w10000__Subsidy_Available',
    'src_pair_logit__Annual_Income_USD_w10000__Environmental_Concern_Level__Range_Anxiety_Level',
)


def normalize_original(source):
    out = source.copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    for col in CATEGORICAL:
        out[col] = out[col].astype('string').fillna('__MISSING__')
    return out


def raw_onehot(train, test, source):
    joined = pd.concat([train[list(FEATURES)], test[list(FEATURES)], source[list(FEATURES)]], ignore_index=True)
    for c in CATEGORICAL:
        joined[c] = joined[c].astype('string').fillna('__MISSING__')
    joined = pd.get_dummies(joined, columns=list(CATEGORICAL), dtype=np.float32)
    for c in joined.columns:
        joined[c] = pd.to_numeric(joined[c], errors='coerce')
    joined = joined.fillna(joined.median(numeric_only=True)).fillna(0.0)
    n, m = len(train), len(test)
    return joined.iloc[:n].to_numpy(np.float32), joined.iloc[n:n + m].to_numpy(np.float32), list(joined.columns)


def smooth_basis(train, test):
    joined = pd.concat([train[list(NUMERIC)], test[list(NUMERIC)]], ignore_index=True)
    arrays, names = [], []
    for col in NUMERIC:
        values = pd.to_numeric(joined[col], errors='coerce').to_numpy(np.float64)
        med = float(np.nanmedian(values))
        q25, q75 = np.nanquantile(values, [0.25, 0.75])
        scale = max(float(q75 - q25), 1e-6)
        filled = np.nan_to_num(values, nan=med)
        z = (filled - med) / scale
        for freq in (0.5, 1.0, 2.0):
            arrays.extend((np.sin(freq * z), np.cos(freq * z)))
            names.extend((f'sin{freq:g}__{col}', f'cos{freq:g}__{col}'))
        centers = np.unique(np.nanquantile(values, np.linspace(0.03, 0.97, 12)))
        width = max(float(np.median(np.diff(centers))), 1e-6) if len(centers) > 1 else scale
        for i, center in enumerate(centers):
            arrays.append(np.exp(-0.5 * ((filled - center) / width) ** 2))
            names.append(f'rbf{i:02d}__{col}')
    matrix = np.column_stack(arrays).astype(np.float32)
    return matrix[:len(train)], matrix[len(train):], names


def semantic_features(train, test):
    frame = pd.concat([train[list(FEATURES)], test[list(FEATURES)]], ignore_index=True)
    get = lambda c: pd.to_numeric(frame[c], errors='coerce').to_numpy(np.float64)  # noqa: E731
    age, income, commute, cars = get('Age'), get('Annual_Income_USD'), get('Daily_Commute_km'), get('Number_of_Cars_Owned')
    hs, ws, concern = get('Charging_Stations_Near_Home'), get('Charging_Stations_Near_Work'), get('Environmental_Concern_Level')
    yes = lambda c: (frame[c].astype(str).to_numpy() == 'Yes').astype(np.float64)  # noqa: E731
    home, subsidy = yes('Home_Charging_Possible'), yes('Subsidy_Available')
    anxiety = frame['Range_Anxiety_Level'].map({'Low': 0.0, 'Medium': 1.0, 'High': 2.0}).fillna(1.0).to_numpy(float)
    total = hs + ws
    d = {
        'sem_log_income': np.log1p(np.maximum(income, 0)),
        'sem_income_10k': income / 10000.0,
        'sem_age_sq': ((age - 45.0) / 20.0) ** 2,
        'sem_station_total': total,
        'sem_station_diff': hs - ws,
        'sem_station_min': np.minimum(hs, ws),
        'sem_station_max': np.maximum(hs, ws),
        'sem_home_station_share': (hs + 0.5) / (total + 1.0),
        'sem_commute_per_station': commute / (total + 1.0),
        'sem_income_per_car': income / (cars + 1.0),
        'sem_income_per_commute': income / (commute + 5.0),
        'sem_charging_access': home + subsidy + np.log1p(total),
        'sem_concern_subsidy': concern * (1.0 + subsidy),
        'sem_concern_home': concern * (1.0 + home),
        'sem_concern_low_anxiety': concern * (3.0 - anxiety),
        'sem_subsidy_low_anxiety': subsidy * (3.0 - anxiety),
        'sem_home_low_anxiety': home * (3.0 - anxiety),
        'sem_commute_anxiety': commute * (1.0 + anxiety),
        'sem_income_concern': np.log1p(np.maximum(income, 0)) * concern,
        'sem_income_subsidy': np.log1p(np.maximum(income, 0)) * (1.0 + subsidy),
        'sem_core_triple': concern * (1.0 + subsidy) * (3.0 - anxiety),
        'sem_long_commute_low_access': commute / (1.0 + total + 3.0 * home),
    }
    matrix = np.column_stack(list(d.values())).astype(np.float32)
    return matrix[:len(train)], matrix[len(train):], list(d)


def _token(frame, col, width):
    if width is None:
        return frame[col].astype('string').fillna('__MISSING__').to_numpy(str)
    value = pd.to_numeric(frame[col], errors='coerce').to_numpy(float)
    out = np.full(len(frame), '__MISSING__', dtype=object)
    good = np.isfinite(value)
    out[good] = np.floor(value[good] / width).astype(np.int64).astype(str)
    return out.astype(str)


def source_pair_lifts(train, test, source, source_y, alpha=20.0):
    query = pd.concat([train[list(FEATURES)], test[list(FEATURES)]], ignore_index=True)
    prior = float(source_y.mean())
    prior_l = float(logit(prior))
    arrays, names = [], []
    for spec in PAIR_SPECS:
        sq = [_token(source, c, w) for c, w in spec]
        qq = [_token(query, c, w) for c, w in spec]
        skey, qkey = sq[0], qq[0]
        for v in sq[1:]:
            skey = np.char.add(np.char.add(skey, '\x1f'), v)
        for v in qq[1:]:
            qkey = np.char.add(np.char.add(qkey, '\x1f'), v)
        stats = pd.DataFrame({'k': skey, 'y': source_y}).groupby('k')['y'].agg(['sum', 'count'])
        n = pd.Series(qkey).map(stats['count']).fillna(0).to_numpy(float)
        pos = pd.Series(qkey).map(stats['sum']).fillna(0).to_numpy(float)
        p = (pos + alpha * prior) / (n + alpha)
        additive = np.full(len(query), -(len(spec) - 1) * prior_l)
        for c, w in spec:
            st = pd.DataFrame({'k': _token(source, c, w), 'y': source_y}).groupby('k')['y'].agg(['sum', 'count'])
            qtok = _token(query, c, w)
            mn = pd.Series(qtok).map(st['count']).fillna(0).to_numpy(float)
            mp = pd.Series(qtok).map(st['sum']).fillna(0).to_numpy(float)
            additive += logit(np.clip((mp + 10.0 * prior) / (mn + 10.0), 1e-5, 1 - 1e-5))
        lift = logit(np.clip(p, 1e-5, 1 - 1e-5)) - additive
        stem = '__'.join(f'{c}_w{w:g}' if w else c for c, w in spec)
        arrays.extend((lift, logit(np.clip(p, 1e-5, 1 - 1e-5)), np.log1p(n), (n > 0).astype(float)))
        names.extend((f'src_pair_lift__{stem}', f'src_pair_logit__{stem}', f'src_pair_logcount__{stem}',
                      f'src_pair_support__{stem}'))
    matrix = np.column_stack(arrays).astype(np.float32)
    return matrix[:len(train)], matrix[len(train):], names


def lineage_features(train, test, source, source_y):
    """Nearest-original-income donors and how well their other 12 fields match the row."""
    query = pd.concat([train[list(FEATURES)], test[list(FEATURES)]], ignore_index=True)
    src_income = pd.to_numeric(source['Annual_Income_USD'], errors='coerce').to_numpy(float)
    q_income = pd.to_numeric(query['Annual_Income_USD'], errors='coerce').to_numpy(float)
    good = np.isfinite(src_income)
    src_index = np.flatnonzero(good)
    unique = np.unique(src_income[good])
    position = np.searchsorted(unique, q_income)
    lo = unique[np.clip(position - 1, 0, len(unique) - 1)]
    hi = unique[np.clip(position, 0, len(unique) - 1)]
    nearest = np.where(np.abs(q_income - lo) <= np.abs(q_income - hi), lo, hi)
    source_groups = pd.Series(src_index).groupby(src_income[good], sort=False).agg(list)
    group = {float(v): np.asarray(rows, dtype=np.int64) for v, rows in source_groups.items()}
    other = [c for c in FEATURES if c != 'Annual_Income_USD']
    src_cols, q_cols = [], []
    for c in other:
        if c in NUMERIC:
            src_cols.append(pd.to_numeric(source[c], errors='coerce').to_numpy(float))
            q_cols.append(pd.to_numeric(query[c], errors='coerce').to_numpy(float))
        else:
            src_cols.append(source[c].astype('string').fillna('__MISSING__').to_numpy(str))
            q_cols.append(query[c].astype('string').fillna('__MISSING__').to_numpy(str))
    n = len(query)
    candidate_n, best, second, ties = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    hard_p = np.full(n, float(source_y.mean()))
    soft_p = hard_p.copy()
    entropy = np.zeros(n)
    query_order = np.argsort(nearest, kind='stable')
    anchors, starts, counts = np.unique(nearest[query_order], return_index=True, return_counts=True)
    for anchor, start, count in zip(anchors, starts, counts):
        rows = group[float(anchor)]
        qi = query_order[start:start + count]
        candidate_n[qi] = len(rows)
        for b in range(0, len(qi), 2048):
            batch = qi[b:b + 2048]
            scores = np.zeros((len(batch), len(rows)), dtype=np.int16)
            for sv, qv in zip(src_cols, q_cols):
                scores += (qv[batch, None] == sv[rows][None, :])
            order = np.sort(scores, axis=1)
            mx = order[:, -1].astype(float)
            sc = order[:, -2].astype(float) if len(rows) > 1 else np.zeros(len(batch))
            mask = scores == mx[:, None]
            nt = mask.sum(1)
            hard = (mask * source_y[rows][None, :]).sum(1) / nt
            weight = np.exp((scores - mx[:, None]) / 1.5)
            sw = weight.sum(1)
            sp = (weight * source_y[rows][None, :]).sum(1) / sw
            prob = weight / sw[:, None]
            best[batch], second[batch], ties[batch] = mx, sc, nt
            hard_p[batch] = (hard * nt + 2.0 * source_y.mean()) / (nt + 2.0)
            soft_p[batch] = (sp * sw + 2.0 * source_y.mean()) / (sw + 2.0)
            entropy[batch] = -(prob * np.log(np.maximum(prob, 1e-15))).sum(1)
    expected = np.where(candidate_n <= 1, 4.7571, np.where(candidate_n <= 5, 5.5997,
                        np.where(candidate_n >= 100, 9.2183, 5.2 + 0.75 * np.log1p(candidate_n))))
    scale = np.where(candidate_n <= 1, 1.0, np.sqrt(np.log1p(candidate_n)))
    dist = np.abs(q_income - nearest) / (np.nanstd(src_income[good]) + 1e-12)
    d = {
        'lin_income_nearest_dist': dist,
        'lin_candidate_logcount': np.log1p(candidate_n),
        'lin_candidate_multi': (candidate_n > 1).astype(float),
        'lin_boundary_30000': (nearest == 30000).astype(float),
        'lin_best_match': best,
        'lin_runnerup_match': second,
        'lin_match_gap': best - second,
        'lin_tie_logcount': np.log1p(ties),
        'lin_match_excess': best - expected,
        'lin_match_excess_scaled': (best - expected) / scale,
        'lin_hard_donor_logit': logit(np.clip(hard_p, 1e-5, 1 - 1e-5)),
        'lin_soft_donor_logit': logit(np.clip(soft_p, 1e-5, 1 - 1e-5)),
        'lin_donor_entropy': entropy,
    }
    matrix = np.column_stack(list(d.values())).astype(np.float32)
    return matrix[:len(train)], matrix[len(train):], list(d)


def main():
    train, test = read_train(), read_test()
    source = normalize_original(read_original_text())
    source_y = source_labels(source)
    folder = CACHE / 'static'
    blocks = {}
    for name, builder in (('raw', lambda: raw_onehot(train, test, source)),
                          ('smooth', lambda: smooth_basis(train, test)),
                          ('semantic', lambda: semantic_features(train, test)),
                          ('source_pairs', lambda: source_pair_lifts(train, test, source, source_y)),
                          ('lineage', lambda: lineage_features(train, test, source, source_y))):
        blocks[name] = builder()
        print(name, blocks[name][0].shape, flush=True)
    for view, parent, keep in (
            ('smooth_focus', 'smooth', lambda n: n.endswith(('Annual_Income_USD', 'Environmental_Concern_Level'))),
            ('source_pair_core', 'source_pairs', None)):
        tr, te, names = blocks[parent]
        selected = list(SOURCE_PAIR_CORE) if keep is None else [n for n in names if keep(n)]
        idx = [names.index(n) for n in selected]
        blocks[view] = (tr[:, idx], te[:, idx], selected)
    for name, (tr, te, names) in blocks.items():
        assert tr.shape[1] == te.shape[1] == len(names) and np.isfinite(tr).all() and np.isfinite(te).all(), name
        save_arrays(folder, {f'{name}_train': tr.astype(np.float32), f'{name}_test': te.astype(np.float32)})
    update_manifest(folder, {'block_names': {name: names for name, (_, _, names) in blocks.items()}})
    print({name: len(names) for name, (_, _, names) in blocks.items()})


if __name__ == '__main__':
    main()
