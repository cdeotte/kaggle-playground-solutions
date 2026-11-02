#!/usr/bin/env python
"""Income-group context distributions relative to the original data (banks dist_p100bpe, dist_p5000bpe of
masked_density_contrast). No labels are read.

Rows are grouped by (income // 100, leading BPE piece) [p100bpe, alpha 20] or (income // 5000, leading BPE piece)
[p5000bpe, alpha 0.5]. For each of 13 contexts, the group's context distribution P is smoothed toward the original data's
marginal G with alpha pseudo-rows. Per row: entropy of P (also times 1 / sqrt(count of the exact income + 20)), Renyi-2 and
min entropy, surprisal variance, KL and JS to G, singleton mass, log group count, a 16-dim profile (randomized SVD, seed
42, of (sqrt(P) - sqrt(G)) over all contexts, groups weighted by sqrt(min(count, 20000))), and the leave-one-out log lift
of the row's own context value over G (loo_lift_<context>) and its standardized surprise.
Writes cache/stage6/dist_<recipe>_all.npy (+ .json names, _state.npz).
Usage: python distribution_bank.py"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

from contrast_data import STAGE, STAGE1, write_names

FIELDS = ['Age', 'Daily_Commute_km', 'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Number_of_Cars_Owned',
          'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Subsidy_Available', 'Range_Anxiety_Level',
          'Environmental_Concern_Level']
RECIPES = {
    'p100bpe': dict(keys=['inc100', 'inc_bpe0'], alpha=20.),
    'p5000bpe': dict(keys=['inc5000', 'inc_bpe0'], alpha=.5),
}


def data():
    key = pd.read_parquet(STAGE1 / 'keys.parquet')
    source = pd.read_parquet(STAGE1 / 'source_keys.parquet')
    meta = json.loads((STAGE1 / 'manifest.json').read_text())
    contexts, orig, sizes = {}, {}, {}
    for field in FIELDS:
        contexts[field] = key[field].to_numpy()
        orig[field] = source[field].to_numpy()
        sizes[field] = meta['key_sizes'][field]
    cards = [sizes[f] for f in ['Subsidy_Available', 'Range_Anxiety_Level', 'Environmental_Concern_Level']]

    def sac(d):
        return ((d['Subsidy_Available'].to_numpy() * cards[1] + d['Range_Anxiety_Level'].to_numpy()) * cards[2]
                + d['Environmental_Concern_Level'].to_numpy())
    contexts['sac'], orig['sac'], sizes['sac'] = sac(key), sac(source), int(np.prod(cards))
    return key, source, contexts, orig, sizes


def compute(name):
    start = time.time()
    recipe = RECIPES[name]
    key, source, contexts, orig, sizes = data()
    group = pd.factorize(pd.MultiIndex.from_arrays([key[c] for c in recipe['keys']]), sort=True)[0].astype('int32')
    ng = group.max() + 1
    count = np.bincount(group, minlength=ng)
    income = key.Annual_Income_USD.to_numpy()
    ni = np.bincount(income)
    uncertainty = 1 / np.sqrt(ni[income] + 20)
    entropy, shapes, surprises, shape_names, surprise_names, profiles, states = [], [], [], [], [], [], {}
    alpha = recipe['alpha']
    for field, c in contexts.items():
        ns = sizes[field]
        C = np.bincount(group * ns + c, minlength=ng * ns).reshape(ng, ns).astype('float64')
        S = np.bincount(orig[field], minlength=ns).astype('float64')
        G = (S + .5 / ns) / (len(source) + .5)
        P = (C + alpha * G) / (count[:, None] + alpha)
        lp = np.log(P)
        H = -(P * lp).sum(axis=1)
        H2 = -np.log(np.sum(P * P, axis=1))
        Hinf = -np.log(P.max(axis=1))
        var = np.sum(P * (lp + H[:, None]) ** 2, axis=1)
        kl = np.sum(P * (lp - np.log(G)), axis=1)
        M = .5 * (P + G)
        js = .5 * np.sum(P * np.log(P / M) + G * np.log(G / M), axis=1)
        gt = (C == 1).sum(axis=1) / count
        entropy.append(H[group])
        for label, value in zip(['renyi2', 'min_entropy', 'surprisal_variance', 'kl_original', 'js_original', 'singleton_mass'],
                                [H2, Hinf, var, kl, js, gt]):
            shapes.append(value[group])
            shape_names.append(f'{label}_{field}')
        local = (C[group, c] - 1 + alpha * G[c]) / (count[group] - 1 + alpha)
        assert np.all(local > 0)
        surprises += [np.log(local / G[c]), (-np.log(local) - H[group]) / np.sqrt(var[group] + .01)]
        surprise_names += [f'loo_lift_{field}', f'loo_standard_surprise_{field}']
        profiles.append((np.sqrt(P) - np.sqrt(G)) / np.sqrt(len(contexts)))
        states[f'prob_{field}'] = P.astype('float64')
        states[f'original_{field}'] = G
        states[f'count_{field}'] = C.astype('int32')
    R = np.column_stack(profiles)
    weighted = R * np.sqrt(np.minimum(count, 20000))[:, None]
    _, s, V = randomized_svd(weighted, n_components=16, n_iter=5, random_state=42)
    profile = (R @ V.T)[group]
    H = np.column_stack(entropy)
    entropy_names = [f'entropy_{field}' for field in contexts] + [f'entropy_count_{field}' for field in contexts]
    shape_names += ['parent_logcount']
    shapes.append(np.log1p(count[group]))
    A = np.column_stack([H, H * uncertainty[:, None], *shapes, profile, *surprises]).astype('float32')
    names = entropy_names + shape_names + [f'profile_{j}' for j in range(16)] + surprise_names
    assert A.shape[1] == len(names) and np.isfinite(A).all()
    STAGE[6].mkdir(parents=True, exist_ok=True)
    path = STAGE[6] / f'dist_{name}_all.npy'
    np.save(path, A)
    np.savez(STAGE[6] / f'dist_{name}_state.npz', group=group, count=count, basis=V.T, singular=s, **states)
    write_names(path, names, recipe=recipe)
    print(f'dist_{name} {A.shape}, {ng} groups ({time.time() - start:.0f}s)', flush=True)


if __name__ == '__main__':
    for name in RECIPES:
        if not (STAGE[6] / f'dist_{name}_all.npy').exists():
            compute(name)
