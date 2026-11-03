"""MLM expected-value fills + recomputed budget geometry -> <name>_{tr,te}.parquet (27 columns, label-free).

For every missing numeric cell: mf_<col> = E[v | rest of the row] under the MLM's distribution over the column's exact
values ([NA] / [MASK] excluded), mfsd_<col> = its standard deviation (0 where observed; observed cells keep their value).
From the filled daily / social / gaming / work / weekend: mf_slack = daily - (social + gaming + work), mf_share_s / g / w
= component / max(|daily|, 0.25), mf_wk_minus_d, mf_d_m6, mf_d_m8, mf_s_m4 and the label-rule region mf_region (0 low,
1 mid, 2 high).

  mlmfill:  python build_mlmfill.py --checkpoint den_uncond384v3.pt --name mlmfill
  mlmfill4: python build_mlmfill.py --checkpoint den_uncond_v4.pt --name mlmfill4
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', default='den_uncond384v3.pt')
ap.add_argument('--dmodel', type=int, default=384)
ap.add_argument('--layers', type=int, default=8)
ap.add_argument('--name', default='mlmfill')
args = ap.parse_args()

t0 = time.time()
train, test = load_comp()
n_tr, n_te = len(train), len(test)
n_all = n_tr + n_te
tok_tr, tok_te, vocabs, IDS = load_vocab()
model = load_mlm(args.checkpoint, vocabs, args.dmodel, args.layers)
X_all = torch.from_numpy(IDS)

NUMC = [c for c in FEATS if c in NUMS]
fills, sds = {}, {}
BS = 16384
for j, c in enumerate(FEATS):
    if c not in NUMS: continue
    vv = torch.tensor(token_values(tok_tr, tok_te, c), dtype=torch.float32, device=DEVICE)
    obs_col = pd.concat([train[c], test[c]], ignore_index=True).values.astype(np.float64)
    filled = obs_col.copy()
    sd_col = np.zeros(n_all, np.float32)
    rows = np.where(IDS[:, j] == 0)[0]
    for i in range(0, len(rows), BS):
        r = rows[i:i + BS]
        p = torch.softmax(cell_logits(model, X_all, r, j), -1)
        ev = (p @ vv.unsqueeze(1)).squeeze(1)
        ev2 = (p @ (vv ** 2).unsqueeze(1)).squeeze(1)
        filled[r] = ev.cpu().numpy()
        sd_col[r] = torch.sqrt(torch.clamp(ev2 - ev ** 2, min=0)).cpu().numpy()
    fills[c] = filled.astype(np.float32); sds[c] = sd_col
    print(f'  {c}: {len(rows)} cells ({time.time() - t0:.0f}s)', flush=True)

F = {}
for c in NUMC:
    F[f'mf_{c}'] = fills[c]
    F[f'mfsd_{c}'] = sds[c]
d = fills['daily_screen_time_hours']; s = fills['social_media_hours']
g = fills['gaming_hours']; w = fills['work_study_hours']; wk = fills['weekend_screen_time']
F['mf_slack'] = d - (s + g + w)
F['mf_share_s'] = s / np.maximum(np.abs(d), 0.25)
F['mf_share_g'] = g / np.maximum(np.abs(d), 0.25)
F['mf_share_w'] = w / np.maximum(np.abs(d), 0.25)
F['mf_wk_minus_d'] = wk - d
F['mf_d_m6'] = d - 6.0; F['mf_d_m8'] = d - 8.0; F['mf_s_m4'] = s - 4.0
F['mf_region'] = np.where((d <= 6) & (s <= 4), 0, np.where((d > 8) | (s > 4), 2, 1)).astype(np.float32)
D = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})
D.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / f'{args.name}_tr.parquet')
D.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / f'{args.name}_te.parquet')
print(f'saved {D.shape[1]} cols ({time.time() - t0:.0f}s)')
