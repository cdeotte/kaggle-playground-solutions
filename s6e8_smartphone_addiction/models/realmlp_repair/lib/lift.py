"""Pair-lift banks: does an exact (or coarsened) pair of values occur with the target more or less often than its two
marginals predict?

For a pair (A, B), with smoothed marginal rates P(y|A), P(y|B) (smoothing 20 toward the prior):
  baseline = sigmoid(logit P(y|A) + logit P(y|B) - logit P(y))
  lift     = (pair_positives - pair_count * baseline) / (pair_count + s)
so an unseen pair has lift 0 and rare pairs are shrunk. Each bank appends its lifts and their row mean.
  pairlift  9 exact two-decimal pairs, s = 20 (notifications x app opens: s = 100)                  -> 10 columns
  cbank     8 coarse-lattice pairs (0.25-hour / 0.1-hour rounding, exact counts), s = 1             ->  9 columns
  wbank     4 pairs of derived coordinates (utilization (parts / daily) rounded to 0.01, weekend - daily rounded
            to 0.25) with exact screen values, s = 1                                                ->  5 columns
Missing values are the token '__NA__'. Fold protocol (crossfit): the outer-training rows get their values from an
inner StratifiedKFold(5, shuffle=True, random_state=777) (fit without the row's inner part); outer-validation and test
rows from a state fit on all outer-training rows."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

MISSING_TOKEN = '__NA__'
MARGINAL_SMOOTHING = 20.0
INNER_FOLDS, INNER_SEED = 5, 777


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _keys(left, right):
    return np.char.add(np.char.add(left.astype(str), '|'), right.astype(str))


def _rates(tokens, y, prior):
    group = pd.DataFrame({'token': tokens, 'target': y}).groupby('token', sort=True)['target'].agg(['sum', 'count'])
    return (group['sum'] + MARGINAL_SMOOTHING * prior) / (group['count'] + MARGINAL_SMOOTHING)


def pair_lift(fit_left, fit_right, fit_y, left, right, smoothing):
    y = np.asarray(fit_y, dtype=np.float64)
    prior = float(y.mean())
    left_rate, right_rate = _rates(fit_left, y, prior), _rates(fit_right, y, prior)
    pair = pd.DataFrame({'key': _keys(fit_left, fit_right), 'target': y}).groupby('key', sort=True)['target'] \
        .agg(['sum', 'count'])
    pl = pd.Series(left, copy=False).map(left_rate).fillna(prior).to_numpy()
    pr = pd.Series(right, copy=False).map(right_rate).fillna(prior).to_numpy()
    baseline = 1.0 / (1.0 + np.exp(-(_logit(pl) + _logit(pr) - float(_logit(prior)))))
    keys = pd.Series(_keys(left, right), copy=False)
    count = keys.map(pair['count']).fillna(0.0).to_numpy(dtype=np.float64)
    positive = keys.map(pair['sum']).fillna(0.0).to_numpy(dtype=np.float64)
    return (positive - count * baseline) / (count + smoothing)


def _format(values, decimals):
    out = np.full(len(values), MISSING_TOKEN, dtype=object)
    finite = np.isfinite(values)
    out[finite] = [f'{v:.{decimals}f}' for v in values[finite]]
    return out.astype(str)


def _coarse(series, kind):
    values = pd.to_numeric(series, errors='coerce').to_numpy(np.float64)
    if kind == 'exact':
        return _format(values, 2)
    if kind == 'q025':  # round half to even on a 0.25 grid
        return _format(np.round(values / 0.25) * 0.25, 2)
    if kind == 'r1':
        return _format(np.round(values / 0.1) * 0.1, 1)
    raise ValueError(kind)


class Bank:
    def __init__(self, name, specs, tokenise):
        self.name, self.specs, self.tokenise = name, specs, tokenise
        self.names = [f'{name}__{a}__{b}' for a, b, _ in specs] + [f'{name}__mean']

    def apply(self, fit_tokens, fit_y, query_tokens):
        cols = [pair_lift(fit_tokens[a], fit_tokens[b], fit_y, query_tokens[a], query_tokens[b], s)
                for a, b, s in self.specs]
        m = np.column_stack(cols)
        m = np.column_stack([m, m.mean(axis=1)]).astype(np.float32)
        assert np.isfinite(m).all()
        return m


def _pairlift_tokens(frame):
    cols = sorted({c for a, b, _ in PAIRLIFT_SPECS for c in (a, b)})
    return {c: frame[c].map(lambda v: f'{v:.2f}' if pd.notna(v) else MISSING_TOKEN).to_numpy(dtype=str) for c in cols}


PAIRLIFT_SPECS = (
    ('app_opens_per_day', 'weekend_screen_time', 20.0),
    ('daily_screen_time_hours', 'app_opens_per_day', 20.0),
    ('notifications_per_day', 'weekend_screen_time', 20.0),
    ('daily_screen_time_hours', 'weekend_screen_time', 20.0),
    ('daily_screen_time_hours', 'notifications_per_day', 20.0),
    ('daily_screen_time_hours', 'gaming_hours', 20.0),
    ('notifications_per_day', 'app_opens_per_day', 100.0),
    ('sleep_hours', 'app_opens_per_day', 20.0),
    ('social_media_hours', 'sleep_hours', 20.0),
)
PAIRLIFT = Bank('pairlift', PAIRLIFT_SPECS, _pairlift_tokens)

# token name -> (column, kind)
_CBANK_TOKENS = {'daily_q025': ('daily_screen_time_hours', 'q025'), 'notif_exact': ('notifications_per_day', 'exact'),
                 'app_exact': ('app_opens_per_day', 'exact'), 'weekend_q025': ('weekend_screen_time', 'q025'),
                 'sleep_q025': ('sleep_hours', 'q025'), 'social_r1': ('social_media_hours', 'r1'),
                 'work_r1': ('work_study_hours', 'r1'), 'daily_r1': ('daily_screen_time_hours', 'r1'),
                 'sleep_r1': ('sleep_hours', 'r1'), 'gaming_r1': ('gaming_hours', 'r1')}
CBANK_SPECS = (('daily_q025', 'notif_exact', 1.0), ('app_exact', 'weekend_q025', 1.0),
               ('notif_exact', 'weekend_q025', 1.0), ('daily_q025', 'app_exact', 1.0),
               ('sleep_q025', 'app_exact', 1.0), ('social_r1', 'work_r1', 1.0), ('daily_r1', 'sleep_r1', 1.0),
               ('social_r1', 'gaming_r1', 1.0))


def _cbank_tokens(frame):
    return {name: _coarse(frame[col], kind) for name, (col, kind) in _CBANK_TOKENS.items()}


CBANK = Bank('cbank', CBANK_SPECS, _cbank_tokens)


def _wbank_tokens(frame):
    x = {c: pd.to_numeric(frame[c], errors='coerce').to_numpy(np.float64)
         for c in ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                   'weekend_screen_time')}
    daily = x['daily_screen_time_hours']
    valid = (np.isfinite(daily) & (daily > 0) & np.isfinite(x['social_media_hours']) & np.isfinite(x['gaming_hours'])
             & np.isfinite(x['work_study_hours']))
    utilization = np.full(len(frame), np.nan)
    utilization[valid] = (x['social_media_hours'][valid] + x['gaming_hours'][valid]
                          + x['work_study_hours'][valid]) / daily[valid]
    utilization = np.round(utilization / 0.01) * 0.01
    gap = np.round((x['weekend_screen_time'] - daily) / 0.25) * 0.25
    return {'util_r01': _format(utilization, 2), 'gap_q025': _format(gap, 2),
            'daily_exact': _coarse(frame['daily_screen_time_hours'], 'exact'),
            'weekend_exact': _coarse(frame['weekend_screen_time'], 'exact'),
            'social_exact': _coarse(frame['social_media_hours'], 'exact'),
            'gaming_exact': _coarse(frame['gaming_hours'], 'exact')}


WBANK_SPECS = (('util_r01', 'social_exact', 1.0), ('gap_q025', 'daily_exact', 1.0), ('gap_q025', 'weekend_exact', 1.0),
               ('util_r01', 'gaming_exact', 1.0))
WBANK = Bank('wbank', WBANK_SPECS, _wbank_tokens)
BANKS = {'pairlift': PAIRLIFT, 'cbank': CBANK, 'wbank': WBANK}


def _subset(tokens, index):
    return {k: v[index] for k, v in tokens.items()}


def crossfit(bank, train_tokens, y, fold_train, fold_valid, test_tokens=None):
    """-> dict train / valid / test matrices for one outer fold."""
    out = {'train': np.empty((len(fold_train), len(bank.names)), dtype=np.float32)}
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=INNER_SEED)
    for fit_pos, apply_pos in inner.split(np.zeros(len(fold_train)), y[fold_train]):
        fit_idx, apply_idx = fold_train[fit_pos], fold_train[apply_pos]
        out['train'][apply_pos] = bank.apply(_subset(train_tokens, fit_idx), y[fit_idx],
                                             _subset(train_tokens, apply_idx))
    outer = _subset(train_tokens, fold_train)
    out['valid'] = bank.apply(outer, y[fold_train], _subset(train_tokens, fold_valid))
    if test_tokens is not None:
        out['test'] = bank.apply(outer, y[fold_train], test_tokens)
    return out
