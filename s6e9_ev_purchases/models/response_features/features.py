"""Shared constants, cache helpers and the target-free features of the response-feature models.

build_row_local: row-local RealMLP inputs (raw values, every numeric value as a token, fixed-width bins).
build_source_bank: per-field evidence from the 10,000 original rows and their labels (value counts, smoothed original-label
posterior and log-odds, distance to the nearest original value, ECDF, local density log-ratios at 3 bandwidths), plus value
counts over the competition rows ("decoder" evidence). It never reads competition labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import DATA_DIR, ORIGINAL_FILE, TARGET

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'

NUMERIC = ('Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
           'Charging_Stations_Near_Work', 'Environmental_Concern_Level')
CATEGORICAL = ('Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level')
FEATURES = NUMERIC + CATEGORICAL

FIXED_BIN_WIDTHS = {
    'Age': (2.0, 5.0),
    'Annual_Income_USD': (2500.0, 5000.0, 10000.0),
    'Daily_Commute_km': (2.0, 5.0, 10.0),
    'Charging_Stations_Near_Home': (2.0, 4.0),
    'Charging_Stations_Near_Work': (2.0, 4.0),
}


def read_original_text():
    """The original dataset as the cache builders read it (empty cells kept as text)."""
    return pd.read_csv(DATA_DIR / ORIGINAL_FILE, keep_default_na=False)


def normalize_source(source):
    out = source.copy()
    for name in NUMERIC:
        out[name] = pd.to_numeric(out[name], errors='coerce')
    return out


def source_labels(source):
    return (source[TARGET] == 'Yes').to_numpy(np.int8)


def exact_codes(train, test):
    """Code of every exact value (sorted) over train + test, per field."""
    joined = pd.concat([train[list(FEATURES)], test[list(FEATURES)]], ignore_index=True)
    return {name: pd.factorize(joined[name].astype('string').fillna('__MISSING__'), sort=True)[0].astype(np.int32)
            for name in FEATURES}


def save_arrays(folder, arrays):
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(folder / f'{name}.npy', np.ascontiguousarray(value), allow_pickle=False)


def update_manifest(folder, entries):
    """manifest.json of a cache folder: the column names of its blocks (several builders may add keys)."""
    path = folder / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.update(entries)
    path.write_text(json.dumps(manifest, indent=1) + '\n')


def read_manifest(folder):
    return json.loads((folder / 'manifest.json').read_text())


def _token(values, name):
    numeric = pd.to_numeric(values, errors='coerce').to_numpy(np.float64)
    code = np.rint(numeric * 10.0) if name == 'Daily_Commute_km' else np.rint(numeric)
    result = np.full(len(values), '__MISSING__', dtype=object)
    finite = np.isfinite(code)
    result[finite] = np.char.add(f'{name}=', code[finite].astype(np.int64).astype(str))
    return pd.Series(result, index=values.index, dtype=object)


def build_row_local(frame):
    """raw (7 numeric + 6 categorical), tokens (7 categorical) and coarse bins (12 categorical); no labels, no other rows."""
    out = pd.DataFrame(index=frame.index)
    categorical = []
    for name in NUMERIC:
        out[name] = pd.to_numeric(frame[name], errors='coerce').astype(np.float32)
    for name in CATEGORICAL:
        out[name] = frame[name].astype('string').fillna('__MISSING__').astype(object)
        categorical.append(name)
    for name in NUMERIC:
        out[f'tok__{name}'] = _token(frame[name], name)
        categorical.append(f'tok__{name}')
    for name, widths in FIXED_BIN_WIDTHS.items():
        values = pd.to_numeric(frame[name], errors='coerce').to_numpy(np.float64)
        for width in widths:
            code = np.floor(values / width)
            token = np.full(len(values), '__MISSING__', dtype=object)
            finite = np.isfinite(code)
            token[finite] = np.char.add(f'{name}/w{width:g}=', code[finite].astype(np.int64).astype(str))
            out[f'bin{width:g}__{name}'] = token
            categorical.append(f'bin{width:g}__{name}')
    return out.reset_index(drop=True), categorical


def _local_density_llr(query, source_values, source_y, frac):
    finite = np.isfinite(source_values)
    src, sy = source_values[finite], source_y[finite]
    neg, pos = np.sort(src[sy == 0]), np.sort(src[sy == 1])
    scale = float(np.nanstd(src)) or 1.0
    h = max(scale * frac, 1e-7)
    nc = np.searchsorted(neg, query + h, side='right') - np.searchsorted(neg, query - h, side='left')
    pc = np.searchsorted(pos, query + h, side='right') - np.searchsorted(pos, query - h, side='left')
    llr = np.log((pc + 2.0) / (len(pos) + 4.0)) - np.log((nc + 2.0) / (len(neg) + 4.0))
    llr[~np.isfinite(query)] = 0.0
    return llr.astype(np.float32)


def _frequency_power(source_counts, generated_counts):
    common = source_counts.index.intersection(generated_counts.index)
    if len(common) < 3:
        return 1.0
    x = np.log(source_counts.loc[common].to_numpy(float))
    z = np.log(generated_counts.loc[common].to_numpy(float))
    if np.std(x) < 1e-12:
        return 1.0
    return float(np.polyfit(x, z, 1)[0])


def build_source_bank(fit_predictors, query_predictors, source, source_y):
    """Original-label evidence + value counts of `fit_predictors`, applied to `query_predictors`. 113 float32 columns."""
    prior = float(np.mean(source_y))
    arrays, names = [], []
    for name in FEATURES:
        src_col, fit_col, query_col = source[name], fit_predictors[name], query_predictors[name]
        valid = src_col.notna()
        src_count = src_col[valid].value_counts(dropna=False)
        gen_count = fit_col.value_counts(dropna=False)
        grouped = pd.DataFrame({'v': src_col[valid], 'y': source_y[valid.to_numpy()]})
        stat = grouped.groupby('v', dropna=False)['y'].agg(['sum', 'count'])
        pos, neg = stat['sum'], stat['count'] - stat['sum']
        posterior = (stat['sum'] + 5.0 * prior) / (stat['count'] + 5.0)
        llr = np.log((pos + 2.0) / (source_y.sum() + 4.0)) - np.log((neg + 2.0) / ((1 - source_y).sum() + 4.0))
        alpha = _frequency_power(src_count, gen_count)
        src_n = query_col.map(src_count).fillna(0).to_numpy(float)
        gen_n = query_col.map(gen_count).fillna(0).to_numpy(float)
        post = query_col.map(posterior).fillna(prior).to_numpy(float)
        odds = query_col.map(llr).fillna(0.0).to_numpy(float)
        columns = [np.log1p(src_n), np.log1p(gen_n), np.log1p(gen_n) - alpha * np.log1p(src_n), src_n > 0, post, odds]
        stems = ('src_log_count', 'gen_log_count', 'decoder_residual', 'src_support', 'src_posterior', 'src_llr')
        arrays.extend(np.asarray(x, dtype=np.float32) for x in columns)
        names.extend(f'{stem}__{name}' for stem in stems)
        if name in NUMERIC:
            src_values = pd.to_numeric(src_col, errors='coerce').to_numpy(float)
            q = pd.to_numeric(query_col, errors='coerce').to_numpy(float)
            finite_src = src_values[np.isfinite(src_values)]
            unique = np.unique(finite_src)
            idx = np.searchsorted(unique, q)
            lo = unique[np.clip(idx - 1, 0, len(unique) - 1)]
            hi = unique[np.clip(idx, 0, len(unique) - 1)]
            dist = np.minimum(np.abs(q - lo), np.abs(q - hi))
            dist /= float(np.std(finite_src)) or 1.0
            ecdf = np.searchsorted(np.sort(finite_src), q, side='right') / len(finite_src)
            dist[~np.isfinite(q)] = 0.0
            ecdf[~np.isfinite(q)] = 0.5
            arrays.extend([dist.astype(np.float32), ecdf.astype(np.float32)])
            names.extend([f'src_nearest_dist__{name}', f'src_ecdf__{name}'])
            for frac in (0.01, 0.03, 0.10):
                arrays.append(_local_density_llr(q, src_values, source_y, frac))
                names.append(f'src_kde_llr_{frac:.2f}__{name}')
    matrix = np.column_stack(arrays).astype(np.float32)
    assert np.isfinite(matrix).all()
    return matrix, names
