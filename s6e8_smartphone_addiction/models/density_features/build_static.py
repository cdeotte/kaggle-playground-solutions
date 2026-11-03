"""Static block and exact-value tokens -> static_{tr,te}.parquet (120 columns), tok_{tr,te}.parquet (12 columns).

static (density_lib/features.py build_static): the 12 raw columns (categoricals as fixed ordinal codes), 12 missing flags
+ 3 missing counts, budget geometry (social + gaming + work, slack, shares, weekend - daily, ...), the observed label-rule
region, train + test value counts, original-data amplification log(p_synthetic / p_original) and smoothed original label
LLRs (exact tokens and 0.1-rounded screen values), 18 observed-only exact pair PMIs. This script adds:
  imp_<col> / fill_<col>  XGBoost regressors for the 5 screen columns, each fit on the rows of train + test where that
                          column is observed, from the other 11 raw columns (no labels); fill = observed value or the
                          prediction; fill_slack, fill_share_social, fill_wk_minus_d, fill_d_m8, fill_s_m4 from the fills
  p_reg_lo/mid/hi, reg_exp a 3-class XGBoost model of the label-rule region (low: daily <= 6 and social <= 4; high:
                          daily > 8 or social > 4; else mid) fit on the rows where the observed values determine it;
                          those rows keep their determined region as a one-hot.
tok: each column's exact value as a string ('%.2f' for numerics, '__NA__' for missing)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from density_lib.base import CACHE, CATS, FEATS, NUMS, load_comp, load_orig  # noqa: E402
from density_lib.features import build_static  # noqa: E402

t0 = time.time()
train, test = load_comp()
orig = load_orig()
n_tr = len(train)

Xtr, Xte, tok = build_static(train, test, orig)
X = pd.concat([Xtr, Xte], ignore_index=True)
both_raw = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)

# predictor-only screen imputers (fit on the observed rows of train + test)
SCREEN = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'weekend_screen_time']
base_cols = NUMS + CATS
XB = X[base_cols].copy()
for c in SCREEN:
    obs = both_raw[c].notna().values
    feats = [f for f in base_cols if f != c]
    m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=20, tree_method='hist', device='cuda', random_state=0, n_jobs=16,
                         verbosity=0)
    m.fit(XB.loc[obs, feats], both_raw.loc[obs, c])
    imp = m.predict(XB[feats]).astype(np.float32)
    rmse = float(np.sqrt(np.mean((imp[obs] - both_raw.loc[obs, c]) ** 2)))
    print(f'imputer {c:26s} fit on {obs.sum():6d} obs | in-fit RMSE {rmse:.4f} ({time.time() - t0:.0f}s)')
    X[f'imp_{c}'] = imp
    X[f'fill_{c}'] = both_raw[c].fillna(pd.Series(imp, index=both_raw.index)).astype(np.float32)

# geometry of the filled values
fd = X['fill_daily_screen_time_hours']; fs = X['fill_social_media_hours']
fg = X['fill_gaming_hours']; fw = X['fill_work_study_hours']
fwk = X['fill_weekend_screen_time']
fsgw = fs + fg + fw
X['fill_slack'] = (fd - fsgw).astype(np.float32)
X['fill_share_social'] = (fs / fd.clip(lower=.1)).astype(np.float32)
X['fill_wk_minus_d'] = (fwk - fd).astype(np.float32)
X['fill_d_m8'] = (fd - 8).astype(np.float32)
X['fill_s_m4'] = (fs - 4).astype(np.float32)

# region posterior (3 classes, predictors only)
d = both_raw['daily_screen_time_hours']; s = both_raw['social_media_hours']
hi = (d > 8) | (s > 4)
lo = (d <= 6) & (s <= 4)
mid = d.notna() & s.notna() & ~hi & ~lo
known = hi | lo | mid
reg_lab = np.where(hi, 2, np.where(mid, 1, 0))
mr = xgb.XGBClassifier(n_estimators=700, learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                       min_child_weight=20, objective='multi:softprob', num_class=3, tree_method='hist',
                       device='cuda', random_state=0, n_jobs=16, verbosity=0)
mr.fit(XB.loc[known.values], reg_lab[known.values])
P = mr.predict_proba(XB).astype(np.float32)
P[known.values] = np.eye(3, dtype=np.float32)[reg_lab[known.values]]
X['p_reg_lo'] = P[:, 0]; X['p_reg_mid'] = P[:, 1]; X['p_reg_hi'] = P[:, 2]
X['reg_exp'] = (P * np.array([0, 1, 2])).sum(1).astype(np.float32)
print(f'region model fit on {int(known.sum())} determinable rows ({time.time() - t0:.0f}s)')

X.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / 'static_tr.parquet')
X.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / 'static_te.parquet')
tok_df = pd.DataFrame(tok)
tok_df.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / 'tok_tr.parquet')
tok_df.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / 'tok_te.parquet')
print(f'saved static ({X.shape[1]} cols) + tokens ({time.time() - t0:.0f}s)')
