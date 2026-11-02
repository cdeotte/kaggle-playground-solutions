#!/usr/bin/env python
"""Matched MLP readouts of the stage 3-6 contrasts.

Each config is an input matrix (a baseline + label-free bank columns) and the columns it sets to 0. A real / zero pair
differs only in whether one block carries its values or is zeroed, so the seed-averaged logit difference real - zero is
the block's signal; the probit configs form a probit interaction contrast in the same way.
Inputs: categoricals one-hot (values 1-3), numerics as they are, plus the known-source offset as a column; standardized
with the fit rows (std floor 0.001, clip +-10). Model: 8 parallel members of a 256-128-64 GELU MLP (last layer
zero-initialized). Logit link: logit = source offset + member output, BCE. Probit link: the offset is mapped to
Phi^-1(sigmoid(offset)), loss = probit likelihood, prediction = sigmoid(mean over members of the probit log-odds).
AdamW lr 2e-3 (cosine to 2%), wd 0.01, 6 epochs, batch 2048, Gaussian input noise 0.08, dropout 0.15, gradient clip 1,
TF32 matmuls, no early stopping. Prediction = sigmoid(mean over the 8 members) for query = [fold k rows, test rows].
Writes runs/<config>_s<seed>_f<k>/predictions.npz (finished runs are skipped).
Usage: python contrast_mlp.py --configs n2_reg_real,n2_reg_zero [--seeds 42-57] [--folds 0,1,2,3,4]"""
import argparse
import time

import numpy as np
from scipy.special import expit, ndtri

from contrast_data import (EMB8, GEN_DELTA, all_bank, baseline, coarse_baseline, fold_bank, ints, labels, rich_baseline,
                           run_dir, source_margin)

NN = dict(members=8, widths=[256, 128, 64], epochs=6, batch=2048, lr=.002, wd=.01, dropout=.15, noise=.08)
CONTEXTS13 = ['Age', 'Daily_Commute_km', 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
              'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Subsidy_Available',
              'Range_Anxiety_Level', 'Environmental_Concern_Level', 'sac']
COARSE27 = ([f'parent_entropy_{c}' for c in CONTEXTS13] + [f'parent_entropy_count_{c}' for c in CONTEXTS13]
            + ['value_count_uncertainty'])
MASKED_FIELDS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Gender', 'City_Type', 'Current_Car_Type',
                 'Home_Charging_Possible', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
                 'Subsidy_Available', 'Range_Anxiety_Level', 'Environmental_Concern_Level']
MASKED21 = ([f'conditional_lift_{f}' for f in MASKED_FIELDS]
            + [f'{s}_{f}' for f in ('Annual_Income_USD', 'Subsidy_Available', 'Range_Anxiety_Level', 'Environmental_Concern_Level')
               for s in ('source_expectation', 'source_expectation_shift')])
BASELINES = dict(base=baseline, rich=rich_baseline, coarse=coarse_baseline)
FOLD_BANKS = ('embedding_income', 'generative')                      # per-fold banks (stage3 / stage4)
ALL_BANKS = dict(core_income=5, df8_a5=5, coarse_entropy=5,          # all-row banks: name -> stage
                 dist_p100bpe=6, dist_p5000bpe=6, masked_density=6)


def pair(name, base, bank, columns):
    """name_real: the bank columns as they are; name_zero: the same columns set to 0."""
    return {f'{name}_real': dict(baseline=base, banks=[(bank, columns)]),
            f'{name}_zero': dict(baseline=base, banks=[(bank, columns)], zero_banks=True)}


CONFIGS = {
    # stage 3 (income_mixture_contrast): with / without the 4 EM income-mixture columns
    'n2_reg_real': dict(baseline='base', clip_saved=False),
    'n2_reg_zero': dict(baseline='base', zero_prefixes=['mixture_'], clip_saved=False),
    # stage 4 (income_context_svd_contrast)
    **pair('emb8', 'base', 'embedding_income', EMB8),
    **pair('gen_difference', 'base', 'generative', GEN_DELTA),
    # stage 5 (fine_context_contrast); the rich baseline adds the stage 4 embedding and generator columns
    **pair('core_income8', 'rich', 'core_income', [f'real_{j}' for j in range(8)]),
    **pair('df8_a5_entropy', 'rich', 'df8_a5', [f'real_entropy_{c}' for c in CONTEXTS13]),
    'link_logit_embedding_real': dict(baseline='rich'),
    'link_logit_embedding_zero': dict(baseline='rich', zero_prefixes=['embed_real_income_']),
    'cube_probit_6': dict(baseline='rich', zero_prefixes=['mixture_'], link='probit'),
    'cube_probit_2': dict(baseline='rich', zero_prefixes=['mixture_', 'generator_context_delta_'], link='probit'),
    'cube_probit_4': dict(baseline='rich', zero_prefixes=['mixture_', 'embed_real_income_'], link='probit'),
    'link_probit_all_zero': dict(baseline='rich', zero_prefixes=['mixture_', 'embed_real_income_', 'generator_context_delta_'],
                                 link='probit'),
    # stage 5 coarse (coarse_context_contrast)
    **pair('coarse_entropy', 'rich', 'coarse_entropy', COARSE27),
    # stage 6 (masked_density_contrast); the coarse baseline adds the 27 coarse entropy columns
    **pair('surprise_lift', 'coarse', 'dist_p100bpe', [f'loo_lift_{c}' for c in CONTEXTS13]),
    **pair('dist_p5000bpe_profile', 'coarse', 'dist_p5000bpe', [f'profile_{j}' for j in range(16)]),
    **pair('masked_lift_completion', 'coarse', 'masked_density', MASKED21),
}


def inputs(k, cfg):
    X, Q, names, types, fit, query = BASELINES[cfg['baseline']](k)
    for bank, columns in cfg.get('banks', []):
        if bank in FOLD_BANKS:
            d, nm = fold_bank(bank, k)
            ids = [nm.index(c) for c in columns]
            x, q = d['train_features'][:, ids].copy(), d['query_features'][:, ids].copy()
        else:
            A, nm = all_bank(bank, ALL_BANKS[bank])
            ids = [nm.index(c) for c in columns]
            x, q = A[fit][:, ids].copy(), A[query][:, ids].copy()
        if cfg.get('zero_banks'):
            x[:] = 0
            q[:] = 0
        X, Q = np.column_stack([X, x]), np.column_stack([Q, q])
        names, types = names + [f'{bank}_{c}' for c in columns], types + ['q'] * len(columns)
    for prefix in cfg.get('zero_prefixes', []):
        ids = [j for j, n in enumerate(names) if n.startswith(prefix)]
        X[:, ids] = 0
        Q[:, ids] = 0
    assert np.isfinite(X).all() and np.isfinite(Q).all()
    return X, Q, types, fit, query


def dense(A, types):
    columns = []
    for j, t in enumerate(types):
        if t == 'c':
            for val in range(1, 4):
                columns.append((A[:, j] == val).astype('float32'))
        else:
            columns.append(A[:, j])
    return np.column_stack(columns).astype('float32')


def make_model(dim, cfg, device):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Ensemble(nn.Module):
        def __init__(self):
            super().__init__()
            members = cfg['members']
            dims = [dim] + cfg['widths'] + [1]
            self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
                w = torch.empty(members, a, b, device=device).normal_(0, np.sqrt(2 / a))
                if i == len(dims) - 2:
                    w.zero_()
                self.weights.append(nn.Parameter(w))
                self.biases.append(nn.Parameter(torch.zeros(members, b, device=device)))

        def forward(self, x):
            x = x[:, None, :].expand(-1, cfg['members'], -1)
            for j, (w, b) in enumerate(zip(self.weights, self.biases)):
                x = torch.einsum('bki,kio->bko', x, w) + b
                if j < len(self.weights) - 1:
                    x = F.gelu(x)
                    if self.training and cfg['dropout']:
                        x = F.dropout(x, cfg['dropout'], True)
            return x.squeeze(-1)
    return Ensemble()


def fit_predict(X, Q, types, fit_y, mf, mq, cfg, seed):
    """Trains on (X, fit_y) with offset mf; returns probabilities for Q (offset mq). Only fit_y holds labels."""
    import torch
    from torch.nn import functional as F
    torch.backends.cuda.matmul.allow_tf32 = True
    probit = cfg.get('link') == 'probit'
    X = np.column_stack([dense(X, types), mf])
    Q = np.column_stack([dense(Q, types), mq])
    mean = X.mean(axis=0, dtype='float64')
    scale = np.maximum(X.std(axis=0, dtype='float64'), .001)
    X = np.clip((X - mean) / scale, -10, 10).astype('float32')
    Q = np.clip((Q - mean) / scale, -10, 10).astype('float32')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dev = 'cuda'
    model = make_model(X.shape[1], cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    if probit:
        mf = ndtri(expit(mf.astype('float64'))).astype('float32')
        mq = ndtri(expit(mq.astype('float64'))).astype('float32')
    xt = torch.tensor(X, device=dev)
    yt = torch.tensor(fit_y, dtype=torch.float32, device=dev)
    mt = torch.tensor(mf, device=dev)
    qt = torch.tensor(Q, device=dev)
    qm = torch.tensor(mq, device=dev)
    batch, epochs = cfg['batch'], cfg['epochs']
    steps = epochs * int(np.ceil(len(X) / batch))
    step = 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X), device=dev)
        for j in range(0, len(X), batch):
            ix = order[j:j + batch]
            a = xt[ix]
            if cfg['noise']:
                a = a + cfg['noise'] * torch.randn_like(a)
            lr = cfg['lr'] * (.02 + .98 * .5 * (1 + np.cos(np.pi * step / max(steps - 1, 1))))
            for g in opt.param_groups:
                g['lr'] = lr
            t = mt[ix, None] + model(a)
            yy = yt[ix, None]
            if probit:
                loss = -(yy * torch.special.log_ndtr(t) + (1 - yy) * torch.special.log_ndtr(-t)).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(t, yy.expand_as(t), reduction='none').mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            step += 1
    model.eval()
    pred = []
    with torch.no_grad():
        for j in range(0, len(Q), 8192):
            t = qm[j:j + 8192, None] + model(qt[j:j + 8192])
            zz = torch.special.log_ndtr(t) - torch.special.log_ndtr(-t) if probit else t
            pred.append(torch.sigmoid(zz.mean(dim=1)).cpu().numpy())
    return np.concatenate(pred)


def run_config(name, k, seeds, margin):
    """All missing seeds of one config on fold k (the inputs are built once)."""
    seeds = [s for s in seeds if not (run_dir(name, s, k) / 'predictions.npz').exists()]
    if not seeds:
        return
    from sklearn.metrics import roc_auc_score
    cfg = CONFIGS[name]
    X, Q, types, fit, query = inputs(k, cfg)
    y = labels()
    nv = int((query < len(y)).sum())
    for seed in seeds:
        t = time.time()
        pred = fit_predict(X, Q, types, y[fit].copy(), margin[fit], margin[query], {**NN, 'link': cfg.get('link', 'logit')}, seed)
        assert np.isfinite(pred).all()
        if cfg.get('clip_saved', True):
            pred = pred.clip(1e-7, 1 - 1e-7)
        run = run_dir(name, seed, k)
        run.mkdir(parents=True, exist_ok=True)
        np.savez(run / 'predictions.npz', index=query, prediction=pred)
        print(f'{run.name}: {X.shape}, fold AUC {roc_auc_score(y[query[:nv]], pred[:nv]):.6f} ({time.time() - t:.0f}s)',
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True, help='comma-separated config names (see CONFIGS)')
    ap.add_argument('--seeds', default='42-57')
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    names = a.configs.split(',')
    assert all(n in CONFIGS for n in names), [n for n in names if n not in CONFIGS]
    margin = source_margin()
    for k in ints(a.folds):
        for name in names:
            run_config(name, k, ints(a.seeds), margin)


if __name__ == '__main__':
    main()
