"""ebank: half-hour-lattice screen x exact count lift bank -> ebank_f<fold>_{tr,te}.parquet (6 columns).

Keys: q5 = floor(round(v, 2) / 0.5) of weekend, daily, social; exact '%.2f' tokens of notifications and app opens;
missing = its own key. Five channels (A x B, pair smoothing s), singles smoothing 20 (lib/lift.py): weekend q5 x notif
(2, eb_wk_nt), weekend q5 x app (2, eb_wk_ap), daily q5 x app (1, eb_da_ap), daily q5 x notif (2, eb_da_nt), social q5 x
notif (2, eb_so_nt) + eb_mean."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.pairs import fold_list  # noqa: E402
from lib.lift import build_bank, codes_from_series, codes_q5, col_all, load, pair_lift  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--folds', default='0,1,2,3,4')
args = ap.parse_args()

t0 = time.time()
train, test, y, folds = load()
C = {
    'wk_q5': codes_q5(col_all(train, test, 'weekend_screen_time')),
    'da_q5': codes_q5(col_all(train, test, 'daily_screen_time_hours')),
    'so_q5': codes_q5(col_all(train, test, 'social_media_hours')),
    'notif_ex': codes_from_series(col_all(train, test, 'notifications_per_day'), 'exact'),
    'app_ex': codes_from_series(col_all(train, test, 'app_opens_per_day'), 'exact'),
}
CHANNELS = [
    ('wk_q5', 'notif_ex', 2.0, 'eb_wk_nt'),
    ('wk_q5', 'app_ex', 2.0, 'eb_wk_ap'),
    ('da_q5', 'app_ex', 1.0, 'eb_da_ap'),
    ('da_q5', 'notif_ex', 2.0, 'eb_da_nt'),
    ('so_q5', 'notif_ex', 2.0, 'eb_so_nt'),
]

build_bank('ebank', [ch[3] for ch in CHANNELS], 'eb_mean',
           lambda pj, rows_fit, q_idx: pair_lift(C, y, rows_fit, q_idx, *CHANNELS[pj][:3]),
           y, folds, len(test), fold_list(args.folds), t0)
