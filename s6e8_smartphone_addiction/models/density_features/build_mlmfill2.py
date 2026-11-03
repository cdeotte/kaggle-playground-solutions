"""Expected-value fills averaged over two independent MLMs -> mlmfill2_{tr,te}.parquet (15 columns, label-free).

The mean of E[v | rest of the row] from den_uncond384v3.pt (d384 / 8 layers) and den_uncond256v2.pt (d256 / 6 layers)
for every missing numeric cell (observed cells keep their value): mf2_<col> for the 9 numerics, then mf2_slack,
mf2_share_s, mf2_wk_minus_d, mf2_d_m8, mf2_s_m4 and mf2_region from the filled values."""
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

MODELS = [('den_uncond384v3.pt', 384, 8), ('den_uncond256v2.pt', 256, 6)]

t0 = time.time()
train, test = load_comp()
n_tr = len(train); n_all = n_tr + len(test)
tok_tr, tok_te, vocabs, IDS = load_vocab()
X_all = torch.from_numpy(IDS)
models = [load_mlm(ck, vocabs, dd, ll) for ck, dd, ll in MODELS]

fills = {}
BS = 16384
for j, c in enumerate(FEATS):
    if c not in NUMS: continue
    vv = torch.tensor(token_values(tok_tr, tok_te, c), dtype=torch.float32, device=DEVICE)
    filled = pd.concat([train[c], test[c]], ignore_index=True).values.astype(np.float64)
    rows = np.where(IDS[:, j] == 0)[0]
    for i in range(0, len(rows), BS):
        r = rows[i:i + BS]
        evs = [(torch.softmax(cell_logits(m, X_all, r, j), -1) @ vv.unsqueeze(1)).squeeze(1) for m in models]
        filled[r] = ((evs[0] + evs[1]) / 2).cpu().numpy()
    fills[c] = filled.astype(np.float32)
    print(f'  {c} done ({time.time() - t0:.0f}s)', flush=True)

F = {f'mf2_{c}': fills[c] for c in fills}
d = fills['daily_screen_time_hours']; s = fills['social_media_hours']
g = fills['gaming_hours']; w = fills['work_study_hours']; wk = fills['weekend_screen_time']
F['mf2_slack'] = d - (s + g + w)
F['mf2_share_s'] = s / np.maximum(np.abs(d), 0.25)
F['mf2_wk_minus_d'] = wk - d
F['mf2_d_m8'] = d - 8.0; F['mf2_s_m4'] = s - 4.0
F['mf2_region'] = np.where((d <= 6) & (s <= 4), 0, np.where((d > 8) | (s > 4), 2, 1)).astype(np.float32)
D = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})
D.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / 'mlmfill2_tr.parquet')
D.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / 'mlmfill2_te.parquet')
print(f'saved ({time.time() - t0:.0f}s)')
