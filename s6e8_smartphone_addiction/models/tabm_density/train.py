#!/usr/bin/env python
"""TabM (pytabkit TabM_D_Classifier) on the lean fold-wise channel matrix of ../realmlp_density/lib: one runner for the
2 pairs of this folder.

Net: tabm-mini, k = 32 members, piecewise-linear numeric embeddings (d 16), 128 epochs, lr 1e-3, batch 512; the best
epoch is picked on a random 4% of outer-train (1 - AUC) and that checkpoint is kept (no refit); the 12 exact-value
tokens are categorical columns; seed 42; test = mean of the 5 fold models.
Streams (matrix flags as in the original runs; README.md lists the original configuration lines):
  tabm_density_mfill_bp2   lean cfg + cmp3; sf, den384v3, mfill, mfill2, rrd, bp2   -> the pair itself
  tabm_density_mlm4_banks  lean cfg + cmp4; sf, denv4, mfill4, rrd, plift, cbank, wbank, bp2
                           -> per-fold logit standardization (standardize.py) -> pair tabm_density_mlm4_banks_std

Usage: python train.py <pair> [--folds 0,1,2,3,4]
  runs the missing folds of the pair's stream (finished folds are kept in parts/), then writes
  preds/<pair>_{oof,test}.npy"""
import argparse
import sys
import time
from pathlib import Path

from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / 'realmlp_density'))   # shared lib/
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.matrix import fold_matrix  # noqa: E402
from standardize import standardize  # noqa: E402

TABM = dict(device='cuda', random_state=42, verbosity=0, n_threads=8, n_cv=1, n_refit=0, val_fraction=0.04,
            val_metric_name='1-auc_ovr', n_epochs=128, batch_size=512, lr=1e-3, tabm_k=32, arch_type='tabm-mini',
            num_emb_type='pwl', d_embedding=16)
LEAN_TE = dict(te_smooth=1.0, te2s_smooth=20.0)
LEAN_CFG = 'te,teavg,te2s,orig,pmi,imp,geo,woe,slim'
STREAMS = {
    'tabm_density_mfill_bp2': dict(cfg=LEAN_CFG + ',cmp3', feat='sf,cattok,den384v3,mfill,mfill2,rrd,bp2'),
    'tabm_density_mlm4_banks': dict(cfg=LEAN_CFG + ',cmp4', feat='sf,cattok,denv4,mfill4,rrd,plift,cbank,wbank,bp2'),
}
PAIRS = {   # pair: (stream, per-fold logit standardization)
    'tabm_density_mfill_bp2': ('tabm_density_mfill_bp2', False),
    'tabm_density_mlm4_banks_std': ('tabm_density_mlm4_banks', True),
}


def run_fold(stream, k):
    from pytabkit import TabM_D_Classifier
    s = STREAMS[stream]
    t0 = time.time()
    Af, Bf, cat_cols, D = fold_matrix(k, s['cfg'], s['feat'], **LEAN_TE)
    y, trn_idx, val_idx = D['y'], D['trn_idx'], D['val_idx']
    m = TabM_D_Classifier(**TABM)
    m.fit(Af.iloc[trn_idx], y[trn_idx], cat_col_names=cat_cols or None)
    pv = m.predict_proba(Af.iloc[val_idx])[:, 1]
    pt = m.predict_proba(Bf)[:, 1]
    print(f'{stream} fold {k}: nfeat={Af.shape[1]} VAL_AUC={roc_auc_score(y[val_idx], pv):.6f} '
          f'({time.time() - t0:.0f}s)', flush=True)
    return pv, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(PAIRS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    stream, stdz = PAIRS[a.pair]
    for k in fold_list(a.folds):
        if not have_part(HERE, stream, k):
            save_part(HERE, stream, k, *run_fold(stream, k))
    if stdz:
        standardize(stream, a.pair)
    else:
        y = target()
        assemble(HERE, a.pair, folds(y), y)


if __name__ == '__main__':
    main()
