"""Quantile imputers for daily and social -> impq_{tr,te}.parquet (9 columns, label-free).

XGBoost quantile regressors (q 0.1 and 0.9), fit on the rows of train + test where the column is observed from the other
11 raw columns: q10_<col>, q90_<col>, width qw_<col>, and the position of the label-rule thresholds inside the interval,
clip((t - q10) / (q90 - q10), -1, 2), for daily (t = 6, 8) and social (t = 4)."""
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
base_cols = NUMS + CATS
Xs = pd.read_parquet(CACHE / 'static_tr.parquet')
Xt = pd.read_parquet(CACHE / 'static_te.parquet')
XB = pd.concat([Xs[base_cols], Xt[base_cols]], ignore_index=True)

out = {}
for c, thr in [('daily_screen_time_hours', (6.0, 8.0)), ('social_media_hours', (4.0,))]:
    obs = both[c].notna().values
    feats = [f for f in base_cols if f != c]
    qs = {}
    for q in (0.1, 0.9):
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=40, objective='reg:quantileerror', quantile_alpha=q,
                             tree_method='hist', device='cuda', random_state=0, n_jobs=16, verbosity=0)
        m.fit(XB.loc[obs, feats], both.loc[obs, c])
        qs[q] = m.predict(XB[feats]).astype(np.float32)
        print(f'{c} q{q} done ({time.time() - t0:.0f}s)')
    lo, hi = qs[0.1], qs[0.9]
    out[f'q10_{c}'] = lo
    out[f'q90_{c}'] = hi
    out[f'qw_{c}'] = hi - lo
    for t in thr:
        out[f'qpos{int(t)}_{c}'] = np.clip((t - lo) / np.maximum(hi - lo, 1e-3), -1, 2)

D = pd.DataFrame(out).astype(np.float32)
D.iloc[:len(train)].reset_index(drop=True).to_parquet(CACHE / 'impq_tr.parquet')
D.iloc[len(train):].reset_index(drop=True).to_parquet(CACHE / 'impq_te.parquet')
print(f'saved impq {D.shape} ({time.time() - t0:.0f}s)')
