"""Completion-response block -> <name>_f<fold>_{tr,te}.parquet for the 5 outer folds (23 columns).

For a missing cell, the MLM (label-free) gives P(v | rest of the row) over the column's exact tokens; the completion
response is the label rate of the token (fold-wise response tables of density_lib/completion.py, smoothing 20 and 3)
averaged over that distribution, on the logit scale. Observed cells get the response of their own token ("context").
Per row and fold: mean / sd / min / max of the completion logits (smoothing 20) over the missing cells, mean / sd over the
missing screen cells, mean / sd of the context logits, mean completion confidence 1 - entropy / log(V - 2), the evidence
sum (context + completion logits), mean / min / max at smoothing 3, and the per-column completion logits (smoothing 20:
cmp_c_<col>, smoothing 3: cmp_cs_<col>) of the 5 screen columns (0 where observed).

  cmp3: python build_cmp.py --checkpoint den_uncond384v3.pt --name cmp3   (d384 / 8 layers / 30 epochs)
  cmp4: python build_cmp.py --checkpoint den_uncond_v4.pt --name cmp4     (d384 / 8 layers / 60 epochs)
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
from density_lib.base import CACHE, FEATS, TARGET, get_folds, load_comp  # noqa: E402
from density_lib.completion import inner_groups, lgt, response_table, row_tables  # noqa: E402
from density_lib.mlm import DEVICE, NCOL, cell_logits, load_mlm, load_vocab  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', default='den_uncond384v3.pt')
ap.add_argument('--dmodel', type=int, default=384)
ap.add_argument('--layers', type=int, default=8)
ap.add_argument('--name', default='cmp3')
args = ap.parse_args()

t0 = time.time()
train, test = load_comp()
y = train[TARGET].values
n_tr, n_te = len(train), len(test)
n_all = n_tr + n_te
folds = get_folds(y)
_, _, vocabs, IDS = load_vocab()
model = load_mlm(args.checkpoint, vocabs, args.dmodel, args.layers)

grp, tab_tr = inner_groups(y, folds)
# response tables per column: (V, 5 folds, 6 tables, 2 smoothings)
resp = [response_table(IDS[:n_tr, j], vocabs[c], y, folds, grp) for j, c in enumerate(FEATS)]
print(f'tables built ({time.time() - t0:.0f}s)', flush=True)

# completion logits / confidence of the MISSING cells: (n_all, NCOL, 5 folds)
comp_logit = np.full((n_all, NCOL, 5), np.nan, np.float32)
comp_logit_s = np.full((n_all, NCOL, 5), np.nan, np.float32)
comp_conf = np.full((n_all, NCOL), np.nan, np.float32)
X_all = torch.from_numpy(IDS)
BS = 16384
for j, c in enumerate(FEATS):
    rows = np.where(IDS[:, j] == 0)[0]
    if len(rows) == 0: continue
    V = vocabs[c]
    logV = np.log(max(V - 2, 2))
    Rj = torch.from_numpy(resp[j].reshape(V, 60)).to(DEVICE)
    for i in range(0, len(rows), BS):
        r = rows[i:i + BS]
        logits = cell_logits(model, X_all, r, j)
        p = torch.softmax(logits, -1)
        marg = (p @ Rj).cpu().numpy().reshape(len(r), 5, 6, 2)
        ls = torch.log_softmax(logits, -1)
        comp_conf[r, j] = 1.0 - (-(p * ls.clamp_min(-30)).sum(1)).cpu().numpy() / logV
        sel = row_tables(r, tab_tr, n_tr)
        ar = np.arange(len(r))
        for f in range(5):
            comp_logit[r, j, f] = lgt(marg[ar, f, sel[:, f], 0])
            comp_logit_s[r, j, f] = lgt(marg[ar, f, sel[:, f], 1])
    print(f'  {c}: {len(rows)} cells ({time.time() - t0:.0f}s)', flush=True)

# context logits of the OBSERVED cells: (n_all, NCOL, 5)
ctx_logit = np.full((n_all, NCOL, 5), np.nan, np.float32)
for j, c in enumerate(FEATS):
    idcol = IDS[:, j]
    obs = np.where(idcol != 0)[0]
    sel = row_tables(obs, tab_tr, n_tr)
    for f in range(5):
        ctx_logit[obs, j, f] = lgt(resp[j][idcol[obs], f, sel[:, f], 0])

SCRN = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'weekend_screen_time']
SCR = [FEATS.index(c) for c in SCRN]
for f in range(5):
    CL, CX = comp_logit[:, :, f], ctx_logit[:, :, f]
    CLS = comp_logit_s[:, :, f]
    with np.errstate(all='ignore'):
        D = pd.DataFrame({
            'cmp_mean': np.nanmean(CL, 1), 'cmp_sd': np.nanstd(CL, 1),
            'cmp_min': np.nanmin(CL, 1), 'cmp_max': np.nanmax(CL, 1),
            'cmp_scr_mean': np.nanmean(CL[:, SCR], 1), 'cmp_scr_sd': np.nanstd(CL[:, SCR], 1),
            'cmp_ctx_mean': np.nanmean(CX, 1), 'cmp_ctx_sd': np.nanstd(CX, 1),
            'cmp_conf_mean': np.nanmean(comp_conf, 1),
            'cmp_evsum': np.nansum(CX, 1) + np.nansum(CL, 1),   # total evidence incl. the completed cells
            'cmp_s_mean': np.nanmean(CLS, 1), 'cmp_s_min': np.nanmin(CLS, 1),
            'cmp_s_max': np.nanmax(CLS, 1),
            **{f'cmp_c_{c}': np.nan_to_num(CL[:, FEATS.index(c)], nan=0.0) for c in SCRN},
            **{f'cmp_cs_{c}': np.nan_to_num(CLS[:, FEATS.index(c)], nan=0.0) for c in SCRN},
        })
    D = D.fillna(0.0).astype(np.float32)
    D.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / f'{args.name}_f{f}_tr.parquet')
    D.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / f'{args.name}_f{f}_te.parquet')
    print(f'fold {f} saved ({time.time() - t0:.0f}s)', flush=True)
print('done')
