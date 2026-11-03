"""Posterior shape of the missing continuous cells -> qfill_{tr,te}.parquet (14 columns, label-free).

For every missing cell of daily / social / gaming / work / sleep / weekend, the den_uncond_v4.pt MLM (d384 / 8 layers /
60 epochs) distribution over the column's exact values gives quantiles q10, q50, q90 (first value whose CDF reaches the
level): qf_<col>_w = q90 - q10 and qf_<col>_sk = (q90 + q10 - 2 q50) / (q90 - q10) (0 where the cell is observed).
Row aggregates over the 5 screen columns: qf_scr_w_mean / qf_scr_sk_mean = mean over the nonzero entries (0 if none)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from density_lib.base import CACHE, FEATS, load_comp  # noqa: E402
from density_lib.mlm import DEVICE, cell_logits, load_mlm, load_vocab  # noqa: E402

t0 = time.time()
train, test = load_comp()
n_tr, n_te = len(train), len(test)
n_all = n_tr + n_te
tok_tr, tok_te, vocabs, IDS = load_vocab()
model = load_mlm('den_uncond_v4.pt', vocabs, 384, 8)
X_all = torch.from_numpy(IDS)
CONT = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
        'weekend_screen_time']

cols = {}
for c in CONT:
    j = FEATS.index(c)
    both = pd.concat([tok_tr[c], tok_te[c]], ignore_index=True)
    dfm = pd.DataFrame({'id': IDS[:, j], 't': both.values}).groupby('id')['t'].first()
    v = np.full(vocabs[c], np.nan, np.float32)
    for i, t in dfm.items():
        if t != '__NA__':
            try: v[i] = float(t)
            except ValueError: pass
    vm = np.isfinite(v)
    vv = torch.tensor(np.nan_to_num(v, nan=0.0), device=DEVICE)
    vmask = torch.tensor(vm, device=DEVICE)
    order = torch.argsort(vv)
    W = np.zeros(n_all, np.float32); SK = np.zeros(n_all, np.float32)
    rows = np.where(IDS[:, j] == 0)[0]
    BS = 16384
    for i in range(0, len(rows), BS):
        r = rows[i:i + BS]
        logits = cell_logits(model, X_all, r, j)
        logits[:, ~vmask] = -1e9
        p = torch.softmax(logits, -1)
        ps = p[:, order]; vs = vv[order]
        cdf = torch.cumsum(ps, 1)

        def q(t_):
            return vs[torch.clamp((cdf < t_).sum(1), max=len(vs) - 1)]
        q10, q50, q90 = q(0.10), q(0.50), q(0.90)
        w = (q90 - q10)
        sk = (q90 + q10 - 2 * q50) / torch.clamp(w, min=1e-6)
        W[r] = w.cpu().numpy(); SK[r] = sk.cpu().numpy()
    cols[f'qf_{c}_w'] = W; cols[f'qf_{c}_sk'] = SK
    print(f'{c}: {len(rows)} missing cells ({time.time() - t0:.0f}s)', flush=True)

D = pd.DataFrame(cols)
scr_w = D[[f'qf_{c}_w' for c in CONT[:4]] + ['qf_weekend_screen_time_w']]
scr_sk = D[[f'qf_{c}_sk' for c in CONT[:4]] + ['qf_weekend_screen_time_sk']]
D['qf_scr_w_mean'] = np.where(scr_w.values.sum(1) > 0,
                              scr_w.values.sum(1) / np.maximum((scr_w.values != 0).sum(1), 1), 0)
D['qf_scr_sk_mean'] = np.where((scr_sk.values != 0).any(1),
                               scr_sk.values.sum(1) / np.maximum((scr_sk.values != 0).sum(1), 1), 0)
D.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / 'qfill_tr.parquet')
D.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / 'qfill_te.parquet')
print(f'saved {D.shape[1]} cols ({time.time() - t0:.0f}s)')
