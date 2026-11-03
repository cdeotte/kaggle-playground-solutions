"""plift: pair lift over log-odds independence on exact tokens -> plift_f<fold>_{tr,te}.parquet (10 columns).

Keys = exact '%.2f' tokens (missing = '__NA__'). Nine pairs (A, B, pair smoothing s): (app opens, weekend, 20),
(daily, app opens, 20), (notifications, weekend, 20), (daily, weekend, 20), (daily, notifications, 20), (daily, gaming,
20), (notifications, app opens, 100), (sleep, app opens, 20), (social, sleep, 20). Singles smoothed with strength 20;
lift = pair rate shrunk toward the independence base - base (lib/lift.py), clip 1e-6 in the logits. Columns
plift_<A[:5]>_<B[:5]> + plift_mean. Cross-fit as in lib/lift.py."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.pairs import fold_list  # noqa: E402
from lib.lift import build_bank, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--folds', default='0,1,2,3,4')
args = ap.parse_args()

t0 = time.time()
train, test, y, folds = load()
PAIRS = [('app_opens_per_day', 'weekend_screen_time', 20), ('daily_screen_time_hours', 'app_opens_per_day', 20),
         ('notifications_per_day', 'weekend_screen_time', 20), ('daily_screen_time_hours', 'weekend_screen_time', 20),
         ('daily_screen_time_hours', 'notifications_per_day', 20), ('daily_screen_time_hours', 'gaming_hours', 20),
         ('notifications_per_day', 'app_opens_per_day', 100), ('sleep_hours', 'app_opens_per_day', 20),
         ('social_media_hours', 'sleep_hours', 20)]


def tok(s):
    return s.map(lambda v: f'{v:.2f}' if pd.notna(v) else '__NA__')


cols = sorted(set(sum([[a, b] for a, b, _ in PAIRS], [])))
tk = {c: np.concatenate([tok(train[c]).values, tok(test[c]).values]) for c in cols}   # train rows, then test rows


def lgt(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def lift_for(rows_fit, q_idx, A, B, s):
    """Maps fit on the train rows rows_fit, applied to the rows q_idx."""
    yy = y[rows_fit]; pri = yy.mean(); lpri = np.log(pri / (1 - pri))
    gA = pd.DataFrame({'t': tk[A][rows_fit], 'y': yy}).groupby('t')['y'].agg(['sum', 'count'])
    gB = pd.DataFrame({'t': tk[B][rows_fit], 'y': yy}).groupby('t')['y'].agg(['sum', 'count'])
    mA = (gA['sum'] + 20 * pri) / (gA['count'] + 20); mB = (gB['sum'] + 20 * pri) / (gB['count'] + 20)
    tkA_q, tkB_q = tk[A][q_idx], tk[B][q_idx]
    pA = pd.Series(tkA_q).map(mA).fillna(pri).values
    pB = pd.Series(tkB_q).map(mB).fillna(pri).values
    base = 1 / (1 + np.exp(-(lgt(pA) + lgt(pB) - lpri)))
    key_fit = pd.Series(tk[A][rows_fit]).values + '|' + pd.Series(tk[B][rows_fit]).values
    g = pd.DataFrame({'k': key_fit, 'y': yy}).groupby('k')['y'].agg(['sum', 'count'])
    kq = pd.Series(np.char.add(np.char.add(tkA_q.astype(str), '|'), tkB_q.astype(str)))
    cnt = kq.map(g['count']).fillna(0).values
    pos = kq.map(g['sum']).fillna(0).values
    return (pos + s * base) / (cnt + s) - base


names = [f'plift_{a[:5]}_{b[:5]}' for a, b, _ in PAIRS]
build_bank('plift', names, 'plift_mean',
           lambda pj, rows_fit, q_idx: lift_for(rows_fit, q_idx, *PAIRS[pj]),
           y, folds, len(test), fold_list(args.folds), t0)
