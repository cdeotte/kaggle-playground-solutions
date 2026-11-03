"""dsx: multi-scale (daily, social) response surfaces on the exact 0.01 grid -> dsx_f<fold>_{tr,te}.parquet (3 columns).

Grid = the distinct 0.01-rounded daily values x social values of train + test. On the fit rows, label sums S and counts C
per cell are blurred with a 2-D Gaussian filter of sigma 1, 3 and 8 grid steps; rate = (S_b + 25 p0) / (C_b + 25) with p0
the fit rows' label rate; feature = logit(rate) (clipped to [1e-4, 1 - 1e-4]) of the row's cell: dsx_s1, dsx_s3, dsx_s8.
Rows with daily or social missing get logit(p0). Cross-fit: outer-train rows read surfaces fit on the other 4 inner
groups (StratifiedKFold(5, shuffle, seed 777)), validation and test rows the surfaces fit on all outer-train rows. CPU."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.pairs import fold_list  # noqa: E402
from lib.lift import CACHE, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--folds', default='0,1,2,3,4')
args = ap.parse_args()

train, test, y, folds = load()
n_tr = len(train)
K = 25.0
SIG = [1.0, 3.0, 8.0]

da = np.round(pd.concat([train['daily_screen_time_hours'], test['daily_screen_time_hours']]).values, 2)
so = np.round(pd.concat([train['social_media_hours'], test['social_media_hours']]).values, 2)
ga = np.unique(da[~np.isnan(da)]); gb = np.unique(so[~np.isnan(so)])
ia_all = np.searchsorted(ga, da)
ib_all = np.searchsorted(gb, so)
ia_all = np.where(np.isnan(da), -1, np.clip(ia_all, 0, len(ga) - 1)).astype(np.int64)
ib_all = np.where(np.isnan(so), -1, np.clip(ib_all, 0, len(gb) - 1)).astype(np.int64)

for FOLD in fold_list(args.folds):
    trn_idx = np.where(folds != FOLD)[0]
    val_idx = np.where(folds == FOLD)[0]
    prior = y[trn_idx].mean()

    def field(rows):
        ia, ib = ia_all[rows], ib_all[rows]
        ok = (ia >= 0) & (ib >= 0)
        S = np.zeros((len(ga), len(gb)), np.float64); C = np.zeros_like(S)
        np.add.at(S, (ia[ok], ib[ok]), y[rows][ok].astype(np.float64))
        np.add.at(C, (ia[ok], ib[ok]), 1.0)
        out = []
        for sg in SIG:
            Sb = gaussian_filter(S, sg); Cb = gaussian_filter(C, sg)
            R = (Sb + prior * K) / (Cb + K)
            out.append(np.log(np.clip(R, 1e-4, 1 - 1e-4) / (1 - np.clip(R, 1e-4, 1 - 1e-4))))
        return out

    def apply_fields(Fs, rows):
        ia, ib = ia_all[rows], ib_all[rows]
        ok = (ia >= 0) & (ib >= 0)
        lp = np.log(prior / (1 - prior))
        res = np.full((len(rows), len(SIG)), lp, np.float32)
        for k, F in enumerate(Fs):
            res[ok, k] = F[ia[ok], ib[ok]]
        return res

    cols = np.full((n_tr + len(test), len(SIG)), np.nan, np.float32)
    inner = StratifiedKFold(5, shuffle=True, random_state=777)
    for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y[trn_idx]):
        Fs = field(trn_idx[i_fit])
        cols[trn_idx[i_app]] = apply_fields(Fs, trn_idx[i_app])
    Fs = field(trn_idx)
    cols[val_idx] = apply_fields(Fs, val_idx)
    cols[n_tr:] = apply_fields(Fs, np.arange(n_tr, n_tr + len(test)))
    D = pd.DataFrame({f'dsx_s{int(s)}': cols[:, k] for k, s in enumerate(SIG)})
    D.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / f'dsx_f{FOLD}_tr.parquet')
    D.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / f'dsx_f{FOLD}_te.parquet')
    print(f'dsx fold {FOLD} saved', flush=True)
