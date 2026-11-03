"""Token-embedding coordinates -> tokemb_{tr,te}.parquet (36 columns, label-free).

Per column, a 3-component PCA (seed 0) of the learned token-embedding table of den_uncond384v3.pt (d384 / 8 layers; rows
of the [NA] / [MASK] ids excluded from the fit and set to 0); every row gets the coordinates of its token:
tke0_<col>, tke1_<col>, tke2_<col>. Runs on the CPU."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from density_lib.base import CACHE, FEATS, load_comp  # noqa: E402
from density_lib.mlm import load_mlm, load_vocab  # noqa: E402

train, test = load_comp()
n_tr = len(train)
_, _, vocabs, IDS = load_vocab()
model = load_mlm('den_uncond384v3.pt', vocabs, 384, 8, device='cpu')
out = {}
for j, c in enumerate(FEATS):
    E = model.emb[j].weight.detach().numpy()   # (V, 384); rows 0, 1 = [NA], [MASK]
    nc = min(3, E.shape[0] - 2, E.shape[1])
    M = np.zeros((E.shape[0], 3), np.float32)
    if nc >= 1:
        M[2:, :nc] = PCA(nc, random_state=0).fit_transform(E[2:])
    ids_c = IDS[:, j]
    for k in range(3):
        out[f'tke{k}_{c}'] = M[ids_c, k]
D = pd.DataFrame(out)
D.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / 'tokemb_tr.parquet')
D.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / 'tokemb_te.parquet')
print('saved', D.shape)
