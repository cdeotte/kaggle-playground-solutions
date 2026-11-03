"""cbank: coarse-lattice, low-smoothing pair-lift bank -> cbank_f<fold>_{tr,te}.parquet (9 columns).

Keys: q025 = floor(round(v, 2) / 0.25) (daily, weekend, sleep), r1 = 0.1-rounded value (daily, sleep, social, work,
gaming), exact = '%.2f' token (notifications, app opens); missing = its own key. Eight channels, singles smoothing 20 and
pair smoothing s = 1 (lib/lift.py): daily_q025 x notif_exact (cb_d25_ntf), app_exact x weekend_q025 (cb_app_wk25),
notif_exact x weekend_q025 (cb_ntf_wk25), daily_q025 x app_exact (cb_d25_app), sleep_q025 x app_exact (cb_sl25_app),
social_r1 x work_r1 (cb_so1_wo1), daily_r1 x sleep_r1 (cb_d1_sl1), social_r1 x gaming_r1 (cb_so1_ga1) + cb_mean."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.pairs import fold_list  # noqa: E402
from lib.lift import build_bank, load, pair_lift  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--folds', default='0,1,2,3,4')
args = ap.parse_args()

t0 = time.time()
train, test, y, folds = load()


def codes_exact(col):
    s = pd.concat([train[col], test[col]]).map(lambda v: f'{v:.2f}' if pd.notna(v) else '__NA__')
    c, u = pd.factorize(s, sort=True)
    return c.astype(np.int64), len(u)


def codes_q(col, step):
    v = pd.concat([train[col], test[col]]).values
    b = np.where(np.isfinite(v), np.floor(np.round(v, 2) / step), -1).astype(np.int64)
    b = b - b.min()
    return b, b.max() + 1


def codes_r1(col):
    s = pd.concat([train[col], test[col]]).map(lambda v: f'{round(v, 1):.1f}' if pd.notna(v) else '__NA__')
    c, u = pd.factorize(s, sort=True)
    return c.astype(np.int64), len(u)


C = {
    'daily_q025': codes_q('daily_screen_time_hours', 0.25),
    'weekend_q025': codes_q('weekend_screen_time', 0.25),
    'sleep_q025': codes_q('sleep_hours', 0.25),
    'notif_exact': codes_exact('notifications_per_day'),
    'app_exact': codes_exact('app_opens_per_day'),
    'daily_r1': codes_r1('daily_screen_time_hours'),
    'sleep_r1': codes_r1('sleep_hours'),
    'social_r1': codes_r1('social_media_hours'),
    'work_r1': codes_r1('work_study_hours'),
    'gaming_r1': codes_r1('gaming_hours'),
}
CHANNELS = [
    ('daily_q025', 'notif_exact', 'cb_d25_ntf'),
    ('app_exact', 'weekend_q025', 'cb_app_wk25'),
    ('notif_exact', 'weekend_q025', 'cb_ntf_wk25'),
    ('daily_q025', 'app_exact', 'cb_d25_app'),
    ('sleep_q025', 'app_exact', 'cb_sl25_app'),
    ('social_r1', 'work_r1', 'cb_so1_wo1'),
    ('daily_r1', 'sleep_r1', 'cb_d1_sl1'),
    ('social_r1', 'gaming_r1', 'cb_so1_ga1'),
]
SJ = 1.0

build_bank('cbank', [nm for _, _, nm in CHANNELS], 'cb_mean',
           lambda pj, rows_fit, q_idx: pair_lift(C, y, rows_fit, q_idx, CHANNELS[pj][0], CHANNELS[pj][1], SJ),
           y, folds, len(test), fold_list(args.folds), t0)
