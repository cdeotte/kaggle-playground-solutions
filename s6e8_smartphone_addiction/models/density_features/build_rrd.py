"""Reconstruction residuals of the observed numeric cells -> rrd_{tr,te}.parquet (19 columns, label-free).

Each observed numeric cell is masked in turn and den_uncond384v3.pt (d384 / 8 layers) gives E[v | rest] and its standard
deviation sd (over the column's exact values): rr_<col> = v - E[v], rrz_<col> = (v - E[v]) / sd (both 0 where the cell
is missing), rrz_abs_mean = mean |rrz| over the 9 numerics."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from density_lib.base import CACHE, FEATS, NUMS, load_comp  # noqa: E402
from density_lib.mlm import DEVICE, cell_logits, load_mlm, load_vocab, token_values  # noqa: E402

t0 = time.time()
train, test = load_comp()
n_tr = len(train); n_all = n_tr + len(test)
tok_tr, tok_te, vocabs, IDS = load_vocab()
model = load_mlm('den_uncond384v3.pt', vocabs, 384, 8)
X_all = torch.from_numpy(IDS)
BS = 16384
out = {}
for j, c in enumerate(FEATS):
    if c not in NUMS: continue
    vv = torch.tensor(token_values(tok_tr, tok_te, c), dtype=torch.float32, device=DEVICE)
    obs_v = pd.concat([train[c], test[c]], ignore_index=True).values.astype(np.float64)
    res = np.zeros(n_all, np.float32); zr = np.zeros(n_all, np.float32)
    rows = np.where(IDS[:, j] != 0)[0]
    for i in range(0, len(rows), BS):
        r = rows[i:i + BS]
        p = torch.softmax(cell_logits(model, X_all, r, j), -1)
        ev = (p @ vv.unsqueeze(1)).squeeze(1).cpu().numpy()
        ev2 = (p @ (vv ** 2).unsqueeze(1)).squeeze(1).cpu().numpy()
        sd = np.sqrt(np.clip(ev2 - ev ** 2, 1e-6, None))
        res[r] = (obs_v[r] - ev).astype(np.float32)
        zr[r] = ((obs_v[r] - ev) / sd).astype(np.float32)
    out[f'rr_{c}'] = res
    out[f'rrz_{c}'] = zr
    print(f'  {c}: {len(rows)} cells ({time.time() - t0:.0f}s)', flush=True)
D = pd.DataFrame(out)
D['rrz_abs_mean'] = np.abs(D[[c for c in D.columns if c.startswith('rrz_')]]).mean(axis=1)
D.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / 'rrd_tr.parquet')
D.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / 'rrd_te.parquet')
print(f'saved {D.shape[1]} cols ({time.time() - t0:.0f}s)')
