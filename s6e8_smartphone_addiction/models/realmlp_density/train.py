#!/usr/bin/env python
"""RealMLP (pytabkit RealMLP_TD_Classifier) on the fold-wise channel matrix of lib/: one runner for the 8 pairs of this
folder. A recipe = assemble() channel flags (cfg), extra features / blocks (feat), channel system (sys: TE smoothings,
density files) and net settings, exactly as in the original runs (README.md lists the original configuration lines).

Three net regimes:
  fast     1 fit, n_ens 24 (library-internal ensemble), 64 epochs, batch 1024, plr_sigma 0.3, best epoch on a random 4%
           of outer-train (1 - AUC), no refit
  battery  n_cv 5 + n_refit 5 + n_ens 16, 128 epochs, batch 512, plr_sigma 0.3: 5 internal CV fits pick the epochs, 5
           refits on 100% of outer-train
  lean     16-epoch selector on 85% of outer-train (best epoch by 1 - AUC on the other 15%), then 1 refit on 100% of
           outer-train at that epoch; n_ens 48, batch 256, lr 0.0175, plr_sigma 0.94, hidden 768/384/192, mish
All: seed 42; the 12 exact-value tokens are categorical columns (cattok); test = mean of the 5 fold models.

Usage: python train.py <pair> [--folds 0,1,2,3,4]
  runs the missing folds (finished folds are kept in parts/), then writes preds/<pair>_{oof,test}.npy"""
import argparse
import sys
import time
from pathlib import Path

from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.assemble import BASE_CFG  # noqa: E402
from lib.matrix import fold_matrix  # noqa: E402

NET = dict(device='cuda', random_state=42, verbosity=0, n_threads=8)
FAST = dict(n_cv=1, n_refit=0, n_ens=24, val_fraction=0.04, val_metric_name='1-auc_ovr', n_epochs=64, batch_size=1024,
            predict_batch_size=32768, lr=0.04, plr_sigma=0.3, num_emb_type='pblrd')
BATTERY = dict(n_cv=5, n_refit=5, n_ens=16, val_fraction=0.04, val_metric_name='1-auc_ovr', n_epochs=128,
               batch_size=512, lr=0.04, plr_sigma=0.3, num_emb_type='pblrd')
LEAN = dict(n_cv=1, n_refit=1, n_ens=48, val_fraction=0.15, val_metric_name='1-auc_ovr', n_epochs=16, batch_size=256,
            lr=0.0175, lr_sched='lin_cos_log_15', num_emb_type='pblrd', plr_sigma=0.94, plr_hidden_1=16, plr_hidden_2=8,
            plr_lr_factor=0.08, plr_act_name='relu', hidden_sizes=[768, 384, 192], act='mish', p_drop=0.05, wd=0.017,
            ls_eps=0.005, add_front_scale=True, scale_lr_factor=6.0, first_layer_lr_factor=0.125, embedding_size=6,
            max_one_hot_cat_size=18)
# channel systems: TE smoothings (+ the MLM density files of the den flag)
BASE_SYS = dict(te_smooth=20.0, te2s_smooth=4.0, den='den384')   # first system: d384 / 6-layer / 24-epoch MLM channels
LEAN_SYS = dict(te_smooth=1.0, te2s_smooth=20.0)   # lean matrix: sharp TE (3 inner seeds) + smooth second TE
LEAN_CFG = 'te,teavg,te2s,orig,pmi,imp,geo,woe,slim'

RECIPES = {
    'realmlp_density_knn_mc': dict(cfg=BASE_CFG, feat='cattok', sys=BASE_SYS, net=FAST),
    'realmlp_density_battery_sf_emb': dict(cfg=BASE_CFG, feat='sf,emb,cattok', sys=BASE_SYS, net=BATTERY),
    'realmlp_density_lean_cmp3': dict(cfg=LEAN_CFG + ',cmp3', feat='sf,cattok,den384v3', sys=LEAN_SYS, net=LEAN),
    'realmlp_density_lean_mfill_rrd': dict(cfg=LEAN_CFG + ',cmp3', feat='sf,cattok,den384v3,mfill,mfill2,rrd',
                                           sys=LEAN_SYS, net=LEAN),
    'realmlp_density_lean_plift': dict(cfg=LEAN_CFG + ',cmp3', feat='sf,cattok,den384v3,mfill,mfill2,rrd,plift',
                                       sys=LEAN_SYS, net=LEAN),
    'realmlp_density_lean_cbank': dict(cfg=LEAN_CFG + ',cmp3', feat='sf,cattok,den384v3,mfill,mfill2,rrd,plift,cbank',
                                       sys=LEAN_SYS, net=LEAN),
    'realmlp_density_lean_mlm4_wbank': dict(cfg=LEAN_CFG + ',cmp4', feat='sf,cattok,denv4,mfill4,rrd,plift,cbank,wbank',
                                            sys=LEAN_SYS, net=LEAN),
    'realmlp_density_lean_mlm4_qfill': dict(cfg=LEAN_CFG + ',cmp4',
                                            feat='sf,cattok,denv4,denv5,mfill4,rrd,plift,cbank,wbank,qfill',
                                            sys=LEAN_SYS, net=LEAN),
}


def run_fold(pair, k):
    from pytabkit import RealMLP_TD_Classifier
    r = RECIPES[pair]
    t0 = time.time()
    Af, Bf, cat_cols, D = fold_matrix(k, r['cfg'], r['feat'], **r['sys'])
    y, trn_idx, val_idx = D['y'], D['trn_idx'], D['val_idx']
    m = RealMLP_TD_Classifier(**NET, **r['net'])
    m.fit(Af.iloc[trn_idx], y[trn_idx], cat_col_names=cat_cols or None)
    pv = m.predict_proba(Af.iloc[val_idx])[:, 1]
    pt = m.predict_proba(Bf)[:, 1]
    print(f'{pair} fold {k}: nfeat={Af.shape[1]} VAL_AUC={roc_auc_score(y[val_idx], pv):.6f} '
          f'({time.time() - t0:.0f}s)', flush=True)
    return pv, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    for k in fold_list(a.folds):
        if not have_part(HERE, a.pair, k):
            save_part(HERE, a.pair, k, *run_fold(a.pair, k))
    y = target()
    assemble(HERE, a.pair, folds(y), y)


if __name__ == '__main__':
    main()
