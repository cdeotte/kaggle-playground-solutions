"""MLM row embeddings and kNN label responses in that embedding space.

emb: row embedding = mean over the 12 column positions of the final hidden state of a trained unconditional MLM
     (label-free), float16 -> <emb> (row_emb384.npy from the d384 / 6-layer model; also read by other folders).
knn: per outer fold, embeddings L2-normalized (cosine geometry), brute-force cuML nearest neighbours (k = 64, self match
     dropped). Features of a query row from its neighbours' labels: knn_r16 (mean label of the 16 nearest), knn_r64 (of
     the 64), knn_rw (distance-weighted mean, w = 1 / (d + 1e-3)) and knn_dist (mean distance). Outer-train rows query an
     index of the other inner groups (StratifiedKFold(5, shuffle, seed 777)); validation and test rows query an index of
     all outer-train rows -> knn<tag>_f<fold>.parquet (train rows then test rows).

  python build_knn.py emb --checkpoint den_uncond384.pt --dmodel 384 --layers 6 --emb row_emb384.npy
  python build_knn.py knn --emb row_emb384.npy --folds 0,1,2,3,4
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
from common.pairs import fold_list  # noqa: E402
from density_lib.base import CACHE, TARGET, auc, get_folds, load_comp  # noqa: E402
from density_lib.mlm import DEVICE, load_mlm, load_vocab  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('step', choices=['emb', 'knn'])
ap.add_argument('--checkpoint', default='den_uncond384.pt')
ap.add_argument('--dmodel', type=int, default=384)
ap.add_argument('--layers', type=int, default=6)
ap.add_argument('--emb', default='row_emb384.npy')
ap.add_argument('--folds', default='0,1,2,3,4')
ap.add_argument('--k', type=int, default=64)
ap.add_argument('--tag', default='', help='output knn<tag>_f<fold>.parquet')
args = ap.parse_args()

t0 = time.time()
train, test = load_comp()
y_all = train[TARGET].values
n_tr = len(train)

if args.step == 'emb':
    _, _, vocabs, ids = load_vocab()
    model = load_mlm(args.checkpoint, vocabs, args.dmodel, args.layers)
    IDS = torch.from_numpy(ids)
    out = np.zeros((len(ids), args.dmodel), np.float16)
    with torch.no_grad():
        for i in range(0, len(ids), 32768):
            xb = IDS[i:i + 32768].to(DEVICE)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model(xb)
            out[i:i + 32768] = h.mean(1).float().cpu().numpy().astype(np.float16)
    np.save(CACHE / args.emb, out)
    print(f'embeddings saved {out.shape} ({time.time() - t0:.0f}s)')
    sys.exit()

from cuml.neighbors import NearestNeighbors  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

emb = np.load(CACHE / args.emb).astype(np.float32)
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
folds = get_folds(y_all)
K = args.k


def knn_feats(index_rows, query_rows):
    nn_ = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='euclidean')
    nn_.fit(emb[index_rows])
    d, idx = nn_.kneighbors(emb[query_rows])
    d, idx = np.asarray(d), np.asarray(idx)
    same = (index_rows[idx[:, 0]] == query_rows) & (d[:, 0] < 1e-6)
    # drop the self match when a row of the index queries it
    dd = np.where(same[:, None], d[:, 1:K + 1], d[:, :K])
    ii = np.where(same[:, None], idx[:, 1:K + 1], idx[:, :K])
    ny = y_all[index_rows[ii]]
    r16 = ny[:, :16].mean(1)
    r64 = ny.mean(1)
    w = 1.0 / (dd + 1e-3)
    rw = (ny * w).sum(1) / w.sum(1)
    return np.stack([r16, r64, rw, dd.mean(1)], 1).astype(np.float32)


for FOLD in fold_list(args.folds):
    trn_rows = np.where(folds != FOLD)[0]
    val_rows = np.where(folds == FOLD)[0]
    F = np.full((len(emb), 4), np.nan, np.float32)
    inner = StratifiedKFold(5, shuffle=True, random_state=777)
    for k, (i_fit, i_app) in enumerate(inner.split(np.zeros(len(trn_rows)), y_all[trn_rows])):
        F[trn_rows[i_app]] = knn_feats(trn_rows[i_fit], trn_rows[i_app])
        print(f'  inner {k} done ({time.time() - t0:.0f}s)', flush=True)
    qr = np.concatenate([val_rows, np.arange(n_tr, len(emb))])
    F[qr] = knn_feats(trn_rows, qr)
    D = pd.DataFrame(F, columns=['knn_r16', 'knn_r64', 'knn_rw', 'knn_dist'])
    D.to_parquet(CACHE / f'knn{args.tag}_f{FOLD}.parquet')
    a = auc(y_all[val_rows], D['knn_rw'].values[val_rows])
    print(f'knn fold{FOLD} done. knn_rw val AUC {a:.5f} ({time.time() - t0:.0f}s)')
