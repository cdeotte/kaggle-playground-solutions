"""Monte-Carlo posterior features of the budget mechanism -> mc_{tr,te}.parquet (11 columns, label-free).

For each of the 5 screen columns an XGBoost quantile regressor (quantiles .05 .15 .30 .45 .55 .70 .85 .95), fit on the
rows of train + test where the column is observed from the other 11 raw columns, gives an approximate conditional CDF.
K = 48 inverse-CDF draws per row (piecewise linear between the quantiles, independent across columns; observed cells keep
their value), then daily is raised to social + gaming + work + 0.01 where the draw violates the budget. From the draws:
P(region high / low / mid), E[max(daily - 8, 0)], E[max(social - 4, 0)], mean / sd of the slack, P(slack < 0.4),
E[weekend - daily], sd of daily and of social."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from density_lib.base import CACHE, CATS, FEATS, NUMS, load_comp  # noqa: E402

t0 = time.time()
train, test = load_comp()
both = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
Xs = pd.read_parquet(CACHE / 'static_tr.parquet')
Xt = pd.read_parquet(CACHE / 'static_te.parquet')
base_cols = NUMS + CATS
XB = pd.concat([Xs[base_cols], Xt[base_cols]], ignore_index=True)

SC = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'weekend_screen_time']
QS = [0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95]
n = len(both)

qpred = {}
for c in SC:
    obs = both[c].notna().values
    feats = [f for f in base_cols if f != c]
    m = xgb.XGBRegressor(n_estimators=450, learning_rate=0.09, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=40, objective='reg:quantileerror', quantile_alpha=np.array(QS),
                         tree_method='hist', device='cuda', random_state=0, n_jobs=16, verbosity=0)
    m.fit(XB.loc[obs, feats], both.loc[obs, c])
    P = np.sort(m.predict(XB[feats]), axis=1)   # (n, len(QS)), monotone quantiles
    qpred[c] = P.astype(np.float32)
    print(f'quantiles {c} done ({time.time() - t0:.0f}s)', flush=True)

rng = np.random.default_rng(7)
K = 48


def draw(c):
    """K inverse-CDF samples per row; observed rows return their value."""
    P = qpred[c]
    u = rng.random((n, K)).astype(np.float32)
    qs = np.array(QS, np.float32)   # piecewise-linear inverse CDF, tails clipped to the edge quantiles
    idx = np.clip(np.searchsorted(qs, u) - 1, 0, len(QS) - 2)
    q0 = qs[idx]; q1 = qs[idx + 1]
    w = np.clip((u - q0) / (q1 - q0), 0, 1)
    row = np.arange(n)[:, None]
    s = P[row, idx] * (1 - w) + P[row, idx + 1] * w
    v = both[c].values.astype(np.float32)
    obs = ~np.isnan(v)
    s[obs] = v[obs, None]
    return s


Sd = draw('daily_screen_time_hours')
Ss = draw('social_media_hours')
Sg = draw('gaming_hours')
Sw = draw('work_study_hours')
Swk = draw('weekend_screen_time')
sgw = Ss + Sg + Sw
Sd = np.maximum(Sd, sgw + 0.01)   # budget constraint daily >= social + gaming + work
print(f'sampled ({time.time() - t0:.0f}s)')

hi = (Sd > 8) | (Ss > 4)
lo = (Sd <= 6) & (Ss <= 4)
slack = Sd - sgw
out = pd.DataFrame({
    'mc_p_hi': hi.mean(1), 'mc_p_lo': lo.mean(1),
    'mc_p_mid': 1 - hi.mean(1) - lo.mean(1),
    'mc_e_d8': np.maximum(Sd - 8, 0).mean(1),
    'mc_e_s4': np.maximum(Ss - 4, 0).mean(1),
    'mc_e_slack': slack.mean(1), 'mc_v_slack': slack.std(1),
    'mc_p_slack_low': (slack < 0.4).mean(1),
    'mc_e_wkd': (Swk - Sd).mean(1),
    'mc_sd_d': Sd.std(1), 'mc_sd_s': Ss.std(1),
}).astype(np.float32)
out.iloc[:len(train)].reset_index(drop=True).to_parquet(CACHE / 'mc_tr.parquet')
out.iloc[len(train):].reset_index(drop=True).to_parquet(CACHE / 'mc_te.parquet')
print(f'saved mc features {out.shape} ({time.time() - t0:.0f}s)')
