#!/usr/bin/env python
"""Lookup-transformer: a transformer over per-column tokens that carry each column's exact-value embedding plus its
fold-safe evidence, initialized from the masked-lattice MLM. One runner for the 2 pairs of this folder.

Per row: CLS + 12 column tokens + group tokens + 1 missing-bit token.
  column token_c = Emb_c[exact value] + (PLR_c(rank-gauss value) + W_c @ evidence_c) * observed_c
    Emb_c uses the MLM vocabulary (0 = real missing, 1 = mask, 2.. = sorted exact values), so Emb_c[0] is a learned
    missing state; PLR = periodic embedding (24 frequencies, init sd 0.4) of the rank-gauss value.
  evidence_c (each of the evidence types standardized with the mean / sd over train rows and columns):
    v1, 7 dims: TE logit (s 20), sharp TE logit (s 4), original LLR, log1p count, amplification, MLM log p, MLM entropy
    v2, 11 dims: + completion logit (cmp_c_), reconstruction z-residual (rrz_), budget-posterior fill and sd (bp2_)
  group tokens (MLP-embedded inputs standardized with train statistics, NaN -> 0; a learned NA embedding replaces the
    token when any input is NaN): budget geometry (11), region posterior / fills (9), kNN responses (4), MC posterior
    (11), pair / region / geometry TE logits (16, no NA state), v2 only: fill geometry (12: MLM-fill geometry,
    completion aggregates, budget-posterior repair probability); then a linear token of the 12 missing bits.
Encoder: reps (6) parallel transformer encoders (d 256, 8 layers, 8 heads, norm-first, GELU, dropout 0.1) on the same
tokens, each with its own LayerNorm and MLP head on the CLS output; logit = mean over the replicas, loss = mean of the
replicas' BCE. Init: column embeddings and the first 6 layers of every replica from den_uncond256.pt (d256/L6 MLM of
../density_features).
Training: 92% of outer-train (stratified split, seed 99) for fitting, AdamW lr 2e-3 (weight decay 1e-5, 1e-4 on the
embeddings), OneCycle (10% warm-up), batch 1536, up to 40 epochs, bf16 autocast, grad clip 1, EMA 0.999 of the weights;
from epoch 3 the EMA weights are scored on the other 8% after every epoch, stop after 6 epochs without gain; the best
EMA weights predict the fold's validation rows and the test rows. Seed 42.

Usage: python train.py <pair> [--folds 0,1,2,3,4]
  runs the missing folds (finished folds are kept in parts/), then writes preds/<pair>_{oof,test}.npy"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / 'realmlp_density'))   # shared lib/
from common.data import folds, target  # noqa: E402
from common.pairs import assemble as assemble_pair, fold_list, have_part, save_part  # noqa: E402
from lib.assemble import BASE_CFG, FEATS, assemble, density_file  # noqa: E402

# pair: (version, MLM density files of the den channels: den384 = d384 / 6-layer / 24-epoch MLM,
#        den = d320 / 7-layer / 25-epoch MLM; as in the original runs)
PAIRS = {'lookup_transformer_evidence7': (1, 'den384'), 'lookup_transformer_evidence11_fills': (2, 'den')}
SEED = 42
DEV = 'cuda'
D_MODEL, LAYERS, HEADS, EPOCHS, BS, LR = 256, 8, 8, 40, 1536, 2e-3
DROP, EMB_WD, VAL_FRAC, PATIENCE, REPS = 0.1, 1e-4, 0.08, 6, 6
NCOL = len(FEATS)
GEO_COLS = ['sgw', 'slack', 'slack_frac', 'share_social_media_hours', 'share_gaming_hours', 'share_work_study_hours',
            'wk_minus_d', 'wk_minus_sgw', 'd_m6', 'd_m8', 's_m4']
RP_COLS = ['p_reg_lo', 'p_reg_mid', 'p_reg_hi', 'reg_exp', 'fill_daily_screen_time_hours', 'fill_social_media_hours',
           'fill_slack', 'fill_d_m8', 'fill_s_m4']
KN_COLS = ['knn_r16', 'knn_r64', 'knn_rw', 'knn_dist']
MC_COLS = ['mc_p_hi', 'mc_p_lo', 'mc_p_mid', 'mc_e_d8', 'mc_e_s4', 'mc_e_slack', 'mc_v_slack', 'mc_p_slack_low',
           'mc_e_wkd', 'mc_sd_d', 'mc_sd_s']
MF_COLS = ['mf_slack', 'mf_share_s', 'mf_share_g', 'mf_share_w', 'mf_wk_minus_d', 'mf_d_m8', 'mf_s_m4', 'mf_region',
           'cmp5_mean', 'cmp5_s_mean', 'cmp5_pair20', 'bp2_pi_d']
V2_FILES = ['mlmfill', 'rrd', 'cmp3_f{fold}', 'bp2_f{fold}']   # density_features files added by v2


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).astype(np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def token_ids():
    """Exact-value ids per column as in the MLM: 0 = missing, 1 = mask, 2.. = sorted values; vocabulary sizes."""
    tokens = pd.concat([pd.read_parquet(density_file('tok_tr.parquet')),
                        pd.read_parquet(density_file('tok_te.parquet'))], ignore_index=True)
    vocabs, ids = {}, np.zeros((len(tokens), NCOL), np.int64)
    for j, c in enumerate(FEATS):
        vals = sorted(v for v in tokens[c].unique() if v != '__NA__')
        mp = {v: i + 2 for i, v in enumerate(vals)}
        vocabs[c] = len(vals) + 2
        ids[:, j] = tokens[c].map(mp).fillna(0).astype(np.int64).values
    return vocabs, ids


def std_block(AB, cols, n_tr):
    M = AB[cols].values.astype(np.float64)
    mu, sd = np.nanmean(M[:n_tr], 0), np.nanstd(M[:n_tr], 0) + 1e-6
    Z = (M - mu) / sd
    return np.nan_to_num(Z, nan=0.0).astype(np.float32), np.isnan(Z).any(1).astype(np.float32)


def build_inputs(fold, version, den):
    D = assemble(fold, BASE_CFG, den=den)
    A, B = D['A'], D['B']
    if version == 2:
        for stem in V2_FILES:
            stem = stem.format(fold=fold)
            A = pd.concat([A, pd.read_parquet(density_file(f'{stem}_tr.parquet'))], axis=1)
            B = pd.concat([B, pd.read_parquet(density_file(f'{stem}_te.parquet'))], axis=1)
    AB = pd.concat([A, B], ignore_index=True)
    n_tr = len(A)

    def col(name, default):
        return AB[name].values if name in AB.columns else np.full(len(AB), default)
    nev = 7 if version == 1 else 11
    EV = np.zeros((len(AB), NCOL, nev), np.float32)
    for j, c in enumerate(FEATS):
        ev = [logit(col(f'te_{c}', 0.7)), logit(col(f'tes_{c}', 0.7)),
              np.nan_to_num(col(f'ollr_{c}', 0), nan=0.0), np.nan_to_num(np.log1p(col(f'cnt_{c}', 0)), nan=0.0),
              np.nan_to_num(col(f'amp_{c}', 0), nan=0.0), np.nan_to_num(col(f'den_lp_{c}', 0), nan=0.0),
              np.nan_to_num(col(f'den_ent_{c}', 0), nan=0.0)]
        if version == 2:
            ev += [np.nan_to_num(col(f'{p}{c}', 0), nan=0.0) for p in ('cmp_c_', 'rrz_', 'bp2_fill_', 'bp2_sd_')]
        EV[:, j, :] = np.stack(ev, 1).astype(np.float32)
    mu = EV[:n_tr].reshape(-1, nev).mean(0)
    sd = EV[:n_tr].reshape(-1, nev).std(0) + 1e-6
    EV = (EV - mu) / sd

    RG = np.zeros((len(AB), NCOL), np.float32)   # rank-gauss of each observed value (train + test)
    for j, c in enumerate(FEATS):
        v = AB[c].values.astype(np.float64)
        o = ~np.isnan(v)
        if o.sum() > 10:
            qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=400000, random_state=0)
            RG[o, j] = qt.fit_transform(v[o].reshape(-1, 1)).ravel().astype(np.float32)

    groups = [std_block(AB, cols, n_tr) for cols in (GEO_COLS, RP_COLS, KN_COLS, MC_COLS)]
    rs_raw = AB[[c for c in AB.columns if c.startswith(('te2_', 'ter_', 'teg_'))]].values.astype(np.float64)
    rs_raw = np.log(np.clip(rs_raw, 1e-4, 1 - 1e-4) / (1 - np.clip(rs_raw, 1e-4, 1 - 1e-4)))
    rs_mu, rs_sd = np.nanmean(rs_raw[:n_tr], 0), np.nanstd(rs_raw[:n_tr], 0) + 1e-6
    groups.append((np.nan_to_num((rs_raw - rs_mu) / rs_sd, nan=0.0).astype(np.float32), np.zeros(len(AB), np.float32)))
    if version == 2:
        groups.append(std_block(AB, MF_COLS, n_tr))
    T = dict(rg=torch.from_numpy(RG), ev=torch.from_numpy(EV),
             nb=torch.from_numpy(AB[[f'na_{c}' for c in FEATS]].values.astype(np.float32)))
    for i, (x, na) in enumerate(groups):
        T[f'g{i}'], T[f'g{i}_na'] = torch.from_numpy(x), torch.from_numpy(na)
    return T, [x.shape[1] for x, _ in groups], nev, D


class PLR(nn.Module):
    def __init__(self, nfeat, k, d, sigma=0.4):
        super().__init__()
        self.f = nn.Parameter(torch.randn(nfeat, k) * sigma)
        self.w = nn.Parameter(torch.randn(nfeat, 2 * k, d) / math.sqrt(2 * k))
        self.b = nn.Parameter(torch.zeros(nfeat, d))

    def forward(self, x):
        z = 2 * math.pi * x.unsqueeze(-1) * self.f.unsqueeze(0)
        z = torch.cat([torch.sin(z), torch.cos(z)], -1)
        return torch.einsum('bfk,fkd->bfd', z, self.w) + self.b


class Group(nn.Module):
    def __init__(self, dim, d):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(dim, d), nn.GELU(), nn.Linear(d, d))
        self.na = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, x, na):
        return torch.where(na.unsqueeze(-1) > 0.5, self.na.expand(len(x), -1), self.lin(x))


class LKT(nn.Module):
    def __init__(self, vocabs, nev, group_dims):
        super().__init__()
        d = D_MODEL
        self.emb = nn.ModuleList([nn.Embedding(vocabs[c], d) for c in FEATS])
        for e in self.emb:
            nn.init.normal_(e.weight, std=0.02)
        self.plr = PLR(NCOL, 24, d)
        self.evl = nn.ModuleList([nn.Linear(nev, d) for _ in FEATS])
        self.groups = nn.ModuleList([Group(g, d) for g in group_dims])
        self.g_nb = nn.Linear(NCOL, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 1 + NCOL + len(group_dims) + 1, d) * 0.02)

        def encoder():
            layer = nn.TransformerEncoderLayer(d, HEADS, d * 4, DROP, activation='gelu', batch_first=True,
                                               norm_first=True)
            return nn.TransformerEncoder(layer, LAYERS)
        self.trs = nn.ModuleList([encoder() for _ in range(REPS)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(REPS)])
        self.heads_out = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(DROP), nn.Linear(d, 1))
                                        for _ in range(REPS)])

    def forward(self, b):
        obs = (b['ids'] != 0).float().unsqueeze(-1)
        tok = torch.stack([self.emb[j](b['ids'][:, j]) for j in range(NCOL)], 1)
        tok = tok + self.plr(b['rg']) * obs
        tok = tok + torch.stack([self.evl[j](b['ev'][:, j]) for j in range(NCOL)], 1) * obs
        g = torch.stack([grp(b[f'g{i}'], b[f'g{i}_na']) for i, grp in enumerate(self.groups)] + [self.g_nb(b['nb'])], 1)
        x = torch.cat([self.cls.expand(len(tok), -1, -1), tok, g], 1) + self.pos
        outs = torch.stack([self.heads_out[r](self.norms[r](self.trs[r](x))[:, 0]).squeeze(-1) for r in range(REPS)], 0)
        return outs.mean(0), outs


def load_pretrained(model):
    """Column embeddings and encoder layers of the d256/L6 MLM into every replica (shape-matched tensors)."""
    sd_pre = torch.load(density_file('den_uncond256.pt'), map_location=DEV)
    own = model.state_dict()
    loaded = 0
    for k, v in sd_pre.items():
        if k.startswith('emb.') and k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
        if k.startswith('tr.'):
            for r in range(REPS):
                k2 = k.replace('tr.', f'trs.{r}.', 1)
                if k2 in own and own[k2].shape == v.shape:
                    own[k2] = v
                    loaded += 1
    model.load_state_dict(own)
    return loaded


def run_fold(pair, fold):
    t0 = time.time()
    version, den = PAIRS[pair]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    vocabs, ids = token_ids()
    T, group_dims, nev, D = build_inputs(fold, version, den)
    T['ids'] = torch.from_numpy(ids)
    y_all = D['y'].astype(np.float32)
    trn_idx, val_idx = D['trn_idx'], D['val_idx']
    n_tr, n_te = len(D['train']), len(D['test'])
    Y = torch.from_numpy(y_all)
    print(f'data ready ({time.time() - t0:.0f}s)', flush=True)

    model = LKT(vocabs, nev, group_dims).to(DEV)
    print(f'pretrained init: {load_pretrained(model)} tensors', flush=True)

    i_tr, i_es = train_test_split(np.arange(len(trn_idx)), test_size=VAL_FRAC, random_state=99,
                                  stratify=y_all[trn_idx])
    fit_rows, es_rows = trn_idx[i_tr], trn_idx[i_es]
    embp, rest = [], []
    for n_, p_ in model.named_parameters():
        (embp if n_.startswith('emb.') else rest).append(p_)
    opt = torch.optim.AdamW([{'params': rest, 'weight_decay': 1e-5}, {'params': embp, 'weight_decay': EMB_WD}], lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=math.ceil(len(fit_rows) / BS) * EPOCHS + 5,
                                                pct_start=0.1)
    params = list(model.parameters())
    ema = [p.detach().clone() for p in params]
    g = torch.Generator(device=DEV).manual_seed(SEED + fold)

    def gather(rows_t):
        return {k: v[rows_t].to(DEV, non_blocking=True) for k, v in T.items()}

    @torch.no_grad()
    def predict(rows, chunk=32768):
        model.eval()
        out = np.zeros(len(rows), np.float32)
        for i in range(0, len(rows), chunk):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lo, _ = model(gather(torch.from_numpy(rows[i:i + chunk])))
            out[i:i + chunk] = lo.float().cpu().numpy()
        return out

    best_auc, best_w, bad = 0.0, None, 0
    fit_t = torch.from_numpy(fit_rows)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(fit_rows), generator=g, device=DEV)
        for i in range(0, len(fit_rows), BS):
            rows_t = fit_t[perm[i:i + BS].cpu()]
            b = gather(rows_t)
            yy = Y[rows_t].to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, all_lo = model(b)
                loss = sum(nn.functional.binary_cross_entropy_with_logits(all_lo[r], yy)
                           for r in range(all_lo.shape[0])) / all_lo.shape[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                torch._foreach_mul_(ema, 0.999)
                torch._foreach_add_(ema, [p.detach() for p in params], alpha=0.001)
        if ep >= 3:   # score the EMA weights on the 8% holdout
            bak = [p.detach().clone() for p in params]
            with torch.no_grad():
                for p, e in zip(params, ema):
                    p.copy_(e)
            a = roc_auc_score(y_all[es_rows], predict(es_rows))
            if a > best_auc:
                best_auc, best_w, bad = a, [e.clone() for e in ema], 0
            else:
                bad += 1
            print(f'  ep{ep} innerAUC={a:.6f} best={best_auc:.6f} ({time.time() - t0:.0f}s)', flush=True)
            with torch.no_grad():
                for p, k2 in zip(params, bak):
                    p.copy_(k2)
            if bad >= PATIENCE:
                break

    with torch.no_grad():
        for p, e in zip(params, best_w):
            p.copy_(e)
    pv = predict(val_idx)
    pt = predict(np.arange(n_tr, n_tr + n_te))
    print(f'{pair} fold {fold}: inner={best_auc:.6f} VAL_AUC={roc_auc_score(y_all[val_idx], pv):.6f} '
          f'({time.time() - t0:.0f}s)', flush=True)
    return sigmoid(pv), sigmoid(pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(PAIRS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    for k in fold_list(a.folds):
        if not have_part(HERE, a.pair, k):
            save_part(HERE, a.pair, k, *run_fold(a.pair, k))
    y = target()
    assemble_pair(HERE, a.pair, folds(y), y)


if __name__ == '__main__':
    main()
