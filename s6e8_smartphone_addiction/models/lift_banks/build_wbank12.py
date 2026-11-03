"""wbank12: the 4 wbank key pairs at pair smoothings 0.5, 1 and 2 -> wbank12_f<fold>_{tr,te}.parquet (13 columns).

Keys as in build_wbank.py: util_r01 x social exact (utso), wkd_q025 x daily exact (wdda), wkd_q025 x weekend exact
(wdwk), util_r01 x gaming exact (utga); singles smoothing 20 (lib/lift.py). Columns w12_<pair>_<s> (s = 0p5, 1p0, 2p0)
+ w12_mean."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.pairs import fold_list  # noqa: E402
from lib.lift import build_bank, codes_from_series, col_all, load, pair_lift  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--folds', default='0,1,2,3,4')
args = ap.parse_args()

t0 = time.time()
train, test, y, folds = load()
d = col_all(train, test, 'daily_screen_time_hours'); s = col_all(train, test, 'social_media_hours')
g = col_all(train, test, 'gaming_hours'); w = col_all(train, test, 'work_study_hours')
wk = col_all(train, test, 'weekend_screen_time')
util = (s + g + w) / d
wkd = wk - d

C = {
    'util_r01': codes_from_series(util, 'r01'),
    'wkd_q025': codes_from_series(wkd, 'q025'),
    'social_ex': codes_from_series(s, 'exact'),
    'daily_ex': codes_from_series(d, 'exact'),
    'week_ex': codes_from_series(wk, 'exact'),
    'gaming_ex': codes_from_series(g, 'exact'),
}
CHANNELS = [(a, b, sj, f"w12_{t}_{str(sj).replace('.', 'p')}")
            for a, b, t in [('util_r01', 'social_ex', 'utso'), ('wkd_q025', 'daily_ex', 'wdda'),
                            ('wkd_q025', 'week_ex', 'wdwk'), ('util_r01', 'gaming_ex', 'utga')]
            for sj in (0.5, 1.0, 2.0)]

build_bank('wbank12', [ch[3] for ch in CHANNELS], 'w12_mean',
           lambda pj, rows_fit, q_idx: pair_lift(C, y, rows_fit, q_idx, *CHANNELS[pj][:3]),
           y, folds, len(test), fold_list(args.folds), t0)
