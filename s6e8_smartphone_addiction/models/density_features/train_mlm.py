"""Train a masked lattice language model (density_lib/mlm.py) on the predictors of train + test and write its channels.

--mode uncond (default): one model on all train + test rows, no labels. Every observed cell is masked in turn:
    den_lp_<col>   log p(x_col | rest of the row)        (NaN where the cell is missing)
    den_ent_<col>  entropy of that predictive distribution
    den_pll_sum / den_pll_mean   row pseudo-log-likelihood
  -> <name>_{tr,te}.parquet (26 columns) and the weights <checkpoint>.
--mode cond: class-conditional model (an extra label token) for each outer fold in --folds, inner-cross-fitted: the
  outer-train rows get cllr_<col> = log p(x_col | rest, y=1) - log p(x_col | rest, y=0) from 5 models fit on the other
  inner groups (StratifiedKFold(5, shuffle, seed 777)), validation and test rows from a model fit on all outer-train rows
  (+ cllr_sum) -> cllr_f<fold>.parquet (train rows then test rows). Not used by the final models.

The unconditional models of the solution (one command each, see README):
  --dmodel 256 --layers 6 --epochs 16 --name den256   --checkpoint den_uncond256.pt     (lookup-transformer init)
  --dmodel 384 --layers 6 --epochs 24 --name den384   --checkpoint den_uncond384.pt     (den384, row_emb384, knn)
  --dmodel 256 --layers 6 --epochs 20 --name den256v2 --checkpoint den_uncond256v2.pt
  --dmodel 384 --layers 8 --epochs 30 --name den384v3 --checkpoint den_uncond384v3.pt
  --dmodel 320 --layers 7 --epochs 25 --seed 777      (den, den_uncond.pt)
  --dmodel 384 --layers 8 --epochs 60 --name den_v4   --checkpoint den_uncond_v4.pt
  --dmodel 448 --layers 8 --epochs 80 --name den_v5   --checkpoint den_uncond_v5.pt
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
from density_lib.base import CACHE, FEATS, TARGET, auc, get_folds, load_comp  # noqa: E402
from density_lib.mlm import NCOL, extract, load_vocab, train_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='uncond', choices=['uncond', 'cond'])
ap.add_argument('--folds', default='0,1,2,3,4', help='outer folds of --mode cond')
ap.add_argument('--dmodel', type=int, default=192)
ap.add_argument('--layers', type=int, default=4)
ap.add_argument('--epochs', type=int, default=8)
ap.add_argument('--mask-p', type=float, default=0.30)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--name', default='den', help='channel files <name>_tr.parquet / <name>_te.parquet')
ap.add_argument('--checkpoint', default='den_uncond.pt')
args = ap.parse_args()

torch.manual_seed(args.seed)
t0 = time.time()
train, test = load_comp()
y_all = train[TARGET].values
n_tr = len(train)
tok_tr, tok_te, vocabs, ids = load_vocab()
n_all = len(ids)
print(f'vocabs: {sum(vocabs.values())} total | rows {n_all} ({time.time() - t0:.0f}s)')


def fit(rows, y_ctx=None):
    return train_model(ids, vocabs, rows, args.dmodel, args.layers, args.epochs, args.seed, args.mask_p,
                       y_ctx=y_ctx, t0=t0)


if args.mode == 'uncond':
    model = fit(np.arange(n_all))
    LP, ENT = extract(model, ids)
    cols = {}
    for j, c in enumerate(FEATS):
        cols[f'den_lp_{c}'] = LP[:, j]
        cols[f'den_ent_{c}'] = ENT[:, j]
    D = pd.DataFrame(cols)
    D['den_pll_sum'] = np.nansum(LP, 1)
    D['den_pll_mean'] = np.nanmean(LP, 1)
    D.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / f'{args.name}_tr.parquet')
    D.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / f'{args.name}_te.parquet')
    torch.save(model.state_dict(), CACHE / args.checkpoint)
    m = ~np.isnan(D['den_pll_sum'].values[:n_tr])
    a = auc(y_all[m], D['den_pll_sum'].values[:n_tr][m])
    print(f'{args.name} done. pll_sum oriented AUC {max(a, 1 - a):.5f} ({time.time() - t0:.0f}s)')
else:
    from sklearn.model_selection import StratifiedKFold
    folds = get_folds(y_all)
    y_ctx = np.zeros(n_all); y_ctx[:n_tr] = y_all
    for FOLD in fold_list(args.folds):
        torch.manual_seed(args.seed)   # every fold starts from the same state (one process per fold in our runs)
        trn_rows = np.where(folds != FOLD)[0]
        val_rows = np.where(folds == FOLD)[0]
        LLR = np.full((n_all, NCOL), np.nan, np.float32)
        inner = StratifiedKFold(5, shuffle=True, random_state=777)
        for k, (i_fit, i_app) in enumerate(inner.split(np.zeros(len(trn_rows)), y_all[trn_rows])):
            m_in = fit(trn_rows[i_fit], y_ctx=y_ctx)
            app = trn_rows[i_app]
            LP1, _ = extract(m_in, ids, y_ctx_val=1, rows=app)
            LP0, _ = extract(m_in, ids, y_ctx_val=0, rows=app)
            LLR[app] = LP1 - LP0
            print(f'  inner {k} done ({time.time() - t0:.0f}s)', flush=True)
            del m_in; torch.cuda.empty_cache()
        model = fit(trn_rows, y_ctx=y_ctx)
        qr = np.concatenate([val_rows, np.arange(n_tr, n_all)])
        LP1, _ = extract(model, ids, y_ctx_val=1, rows=qr)
        LP0, _ = extract(model, ids, y_ctx_val=0, rows=qr)
        LLR[qr] = LP1 - LP0
        D = pd.DataFrame({f'cllr_{c}': LLR[:, j] for j, c in enumerate(FEATS)})
        D['cllr_sum'] = np.nansum(LLR, 1)
        D.to_parquet(CACHE / f'cllr_f{FOLD}.parquet')
        a_val = auc(y_all[val_rows], D['cllr_sum'].values[val_rows])
        a_trn = auc(y_all[trn_rows], D['cllr_sum'].values[trn_rows])
        print(f'cond fold{FOLD} done. cllr_sum val {a_val:.5f} | inner-OOF train {a_trn:.5f} ({time.time() - t0:.0f}s)')
        del model; torch.cuda.empty_cache()
