"""sbank: social-share lift bank -> sbank_f<fold>_{tr,te}.parquet (5 columns).

share_s = social / max(daily, 0.001). Keys: notifications exact ('%.2f') and q025 (floor(v / 0.25)), share_s r01
(0.1-rounded) and exact, app opens r01, social q5 (floor(v / 0.5)); missing / undefined = its own key. Four channels
(A x B, pair smoothing s), singles smoothing 20 (lib/lift.py): notif exact x share_s r01 (1, sb_nt_shs), app r01 x share_s
r01 (5, sb_ap_shs), social q5 x notif exact (2, sb_so_nt), notif q025 x share_s exact (5, sb_ntq_shs) + sb_mean."""
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
d = col_all(train, test, 'daily_screen_time_hours'); s = col_all(train, test, 'social_media_hours')
share_s = s / d.clip(lower=1e-3)
notif = col_all(train, test, 'notifications_per_day')
app = col_all(train, test, 'app_opens_per_day')

C = {
    'shs_r01': codes_from_series(share_s, 'r01'),
    'shs_ex': codes_from_series(share_s, 'exact'),
    'notif_ex': codes_from_series(notif, 'exact'),
    'notif_q': codes_from_series(notif, 'q025'),
    'app_r01': codes_from_series(app, 'r01'),
    'social_q5': codes_q5(s),
}
CHANNELS = [
    ('notif_ex', 'shs_r01', 1.0, 'sb_nt_shs'),
    ('app_r01', 'shs_r01', 5.0, 'sb_ap_shs'),
    ('social_q5', 'notif_ex', 2.0, 'sb_so_nt'),
    ('notif_q', 'shs_ex', 5.0, 'sb_ntq_shs'),
]

build_bank('sbank', [ch[3] for ch in CHANNELS], 'sb_mean',
           lambda pj, rows_fit, q_idx: pair_lift(C, y, rows_fit, q_idx, *CHANNELS[pj][:3]),
           y, folds, len(test), fold_list(args.folds), t0)
