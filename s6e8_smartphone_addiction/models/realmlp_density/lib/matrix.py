"""The full per-fold input matrix of the pytabkit nets (RealMLP / TabM).

fold_matrix(fold, cfg, feat, te_smooth, te2s_smooth): assemble() channels -> smooth fields (sf) -> MLM embedding PCs
(emb) -> channel blocks below, in this order -> probability-scale response columns to logits -> exact-value tokens as
12 categorical columns (cattok) -> numeric NaN filled with the outer-train median.

Channel blocks (flag: file, repo folder; '{fold}' = outer fold, those files are label-derived and nested like the TE):
  cmp3      cmp3_f{fold}   density_features  completion response: exact-token TE (s 20) marginalized over the d384/L8
                                             MLM's P(value | rest) for the screen cells (23 columns)
  den384v3  den384v3       density_features  per-cell log p / entropy / PLL of the d384/L8 MLM (renamed dn2_)
  mfill     mlmfill        density_features  MLM expected-value fills + sd and the budget geometry recomputed from fills
  mfill2    mlmfill2       density_features  fills averaged over the d384/L8 and d256/L6 MLMs + geometry
  rrd       rrd            density_features  per-cell reconstruction residual and z-score of the observed cells
  bp2       bp2_f{fold}    density_features  budget-mixture posterior fills / sd / repair probability + completion
                                             responses under it
  plift     plift_f{fold}  lift_banks        9 pair lifts over the log-odds independence base + mean
  cbank     cbank_f{fold}  lift_banks        8 coarse-lattice low-smoothing pair lifts + mean
  wbank     wbank_f{fold}  lift_banks        4 derived-coarse pair lifts (utilization, weekend - daily) + mean
  mfill4    mlmfill4       density_features  fills + geometry from the d384/L8 60-epoch MLM (v4)
  denv5     den_v5         density_features  per-cell density channels of the d448 MLM (v5, renamed dn5_)
  qfill     qfill          density_features  q10-q90 width and skew of the v4 MLM posterior of missing screen / sleep
                                             cells + row means
  denv4     den_v4         density_features  per-cell density channels of the v4 MLM (renamed dn4_)
  cmp4      cmp4_f{fold}   density_features  completion response from the v4 MLM (same 23 columns as cmp3)
"""
import time

import numpy as np
import pandas as pd

from common.data import cache_dir
from .assemble import FEATS, assemble, density_file
from .fields import add_embeddings, add_smooth_fields

BLOCKS = {   # flag: (repo folder, file stem, column-prefix rename); appended in this order
    'cmp3': ('density_features', 'cmp3_f{fold}', None),
    'den384v3': ('density_features', 'den384v3', ('den_', 'dn2_')),
    'mfill': ('density_features', 'mlmfill', None),
    'mfill2': ('density_features', 'mlmfill2', None),
    'rrd': ('density_features', 'rrd', None),
    'bp2': ('density_features', 'bp2_f{fold}', None),
    'plift': ('lift_banks', 'plift_f{fold}', None),
    'cbank': ('lift_banks', 'cbank_f{fold}', None),
    'wbank': ('lift_banks', 'wbank_f{fold}', None),
    'mfill4': ('density_features', 'mlmfill4', None),
    'denv5': ('density_features', 'den_v5', ('den_', 'dn5_')),
    'qfill': ('density_features', 'qfill', None),
    'denv4': ('density_features', 'den_v4', ('den_', 'dn4_')),
    'cmp4': ('density_features', 'cmp4_f{fold}', None),
}
LOGIT_PREFIXES = ('te_', 'tes_', 'te2_', 'ter_', 'teg', 'knn_r')


def add_blocks(A, B, fold, flags):
    for flag, (folder, stem, rename) in BLOCKS.items():
        if flag not in flags:
            continue
        d = cache_dir(folder)
        Xtr = pd.read_parquet(d / f'{stem.format(fold=fold)}_tr.parquet')
        Xte = pd.read_parquet(d / f'{stem.format(fold=fold)}_te.parquet')
        if rename:
            Xtr.columns = [c.replace(*rename) for c in Xtr.columns]
            Xte.columns = [c.replace(*rename) for c in Xte.columns]
        A, B = pd.concat([A, Xtr], axis=1), pd.concat([B, Xte], axis=1)
    return A, B


def finish(A, B, trn_idx, cattok):
    """Response columns to logits (clip 1e-4), exact tokens as categoricals, median fill of the numeric columns."""
    for c in [c for c in A.columns if c.startswith(LOGIT_PREFIXES)]:
        for D in (A, B):
            p = np.clip(D[c].astype(np.float64), 1e-4, 1 - 1e-4)
            D[c] = np.log(p / (1 - p)).astype(np.float32)
    cat_cols = []
    if cattok:
        tok_tr = pd.read_parquet(density_file('tok_tr.parquet'))
        tok_te = pd.read_parquet(density_file('tok_te.parquet'))
        for c in FEATS:
            A[f'tk_{c}'] = tok_tr[c].values
            B[f'tk_{c}'] = tok_te[c].values
            cat_cols.append(f'tk_{c}')
    num_cols = [c for c in A.columns if c not in cat_cols]
    med = A.iloc[trn_idx][num_cols].median()
    Af, Bf = A.copy(), B.copy()
    Af[num_cols] = Af[num_cols].fillna(med).astype(np.float32)
    Bf[num_cols] = Bf[num_cols].fillna(med).astype(np.float32)
    return Af, Bf, cat_cols


def fold_matrix(fold, cfg, feat, te_smooth, te2s_smooth, den='den384'):
    """cfg: assemble() flags (cmp3 / cmp4 may also be given here); feat: sf, emb, cattok and block flags."""
    t0 = time.time()
    flags = set(cfg.split(',')) | set(feat.split(','))
    D = assemble(fold, cfg, te_smooth, te2s_smooth, den)
    A, B, y, trn_idx = D['A'], D['B'], D['y'], D['trn_idx']
    if 'sf' in flags:
        A, B = add_smooth_fields(A, B, y, trn_idx, D['val_idx'], D['train'], D['test'])
    if 'emb' in flags:
        A, B = add_embeddings(A, B)
    A, B = add_blocks(A, B, fold, flags)
    Af, Bf, cat_cols = finish(A, B, trn_idx, 'cattok' in flags)
    print(f'fold {fold}: matrix {Af.shape} ncat={len(cat_cols)} ({time.time() - t0:.0f}s)', flush=True)
    return Af, Bf, cat_cols, D
