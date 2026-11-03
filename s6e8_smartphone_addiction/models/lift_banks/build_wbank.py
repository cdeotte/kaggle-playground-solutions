"""wbank: derived-coarse lift bank -> wbank_f<fold>_{tr,te}.parquet (5 columns).

Derived keys: util_r01 = 0.1-rounded utilization (social + gaming + work) / daily, wkd_q025 = floor((weekend - daily) /
0.25); exact '%.2f' tokens of social, daily, weekend, gaming; missing / undefined = its own key. Four channels (A x B,
pair smoothing s), singles smoothing 20 (lib/lift.py): util_r01 x social (0.5, wb_ut_so), wkd_q025 x daily (0.5,
wb_wd_da), wkd_q025 x weekend (2, wb_wd_wk), util_r01 x gaming (2, wb_ut_ga) + wb_mean."""
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
CHANNELS = [
    ('util_r01', 'social_ex', 0.5, 'wb_ut_so'),
    ('wkd_q025', 'daily_ex', 0.5, 'wb_wd_da'),
    ('wkd_q025', 'week_ex', 2.0, 'wb_wd_wk'),
    ('util_r01', 'gaming_ex', 2.0, 'wb_ut_ga'),
]

build_bank('wbank', [ch[3] for ch in CHANNELS], 'wb_mean',
           lambda pj, rows_fit, q_idx: pair_lift(C, y, rows_fit, q_idx, *CHANNELS[pj][:3]),
           y, folds, len(test), fold_list(args.folds), t0)
