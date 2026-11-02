#!/usr/bin/env python
"""Builds the 7 pairs from the saved fold predictions. Every pair is formed per outer fold on its query rows
[fold k rows, test rows]: OOF = the fold-k part, test = mean of the 5 fold probabilities. Seed averages are means of
logit(p clipped to [1e-7, 1 - 1e-7]); z() = logit(p clipped to [1e-15, 1 - 1e-15]) clipped to +-30.
The stage 3-6 carriers are cumulative: each one is its predecessor's fold logit plus new contrasts.

  income_mixture_contrast      z3 = gen0 + 4 (n2_reg_real - n2_reg_zero)                         seeds 42, 43
  income_context_svd_contrast  z4 = z3 + 4 E + 4 G, E = emb8_real - emb8_zero,
                               G = gen_difference_real - gen_difference_zero
                               (E, G = mean of the seed-42..49 and the seed-50..57 differences)  -> cache/stage4/average16_fold<k>.npz
  fine_context_contrast        z5 = logit(p4) + 20 x 0.2 (C + F + I + N), C = core_income8_real - core_income8_zero,
                               F = df8_a5_entropy_real - df8_a5_entropy_zero,
                               I = link_logit_embedding_real - link_logit_embedding_zero,
                               N = - cube_probit_6 + cube_probit_2 + cube_probit_4 - link_probit_all_zero
                               (seeds 42-57)                                                     -> cache/stage5/export_all16_f<k>.npz
  coarse_context_contrast      z5c = z5 + 20 x 0.2 (coarse_entropy_real - coarse_entropy_zero)   -> cache/stage5/coarse_all16_f<k>.npz
  masked_density_contrast      z6 = 0.5 clip(logit(p5c), +-30) + 4 L + 2 P + 2 M, clipped to +-30,
                               L = surprise_lift, P = dist_p5000bpe_profile, M = masked_lift_completion (real - zero)
  copula_gam_contrast          2 z(C) - z(G), clipped to +-30; C = sigmoid(mean over seeds 42-49 of z(copula XGB)),
                               G = sigmoid(clip(0.5 z(sparse GAM) + 0.5 probit response margin, +-30))
  copula_gpt2_contrast         3 z(C) - 2 z(N), clipped to +-30; N = the GPT-2 residual
Pairs are written to preds/ as logits z(p).
Usage: python assemble.py <pair> [<pair> ...]"""
import argparse

import numpy as np
from scipy.special import expit, logit

from contrast_data import ALL16, CONTRAST, GPT_RESIDUAL, STAGE, fold_ids, mean_logit, save_logits, split_indices, z

FINE_TERMS = {
    'core_income8': [('core_income8_real', 1.), ('core_income8_zero', -1.)],
    'df8_a5_entropy': [('df8_a5_entropy_real', 1.), ('df8_a5_entropy_zero', -1.)],
    'link_logit_embedding': [('link_logit_embedding_real', 1.), ('link_logit_embedding_zero', -1.)],
    'negative_probit_eg': [('cube_probit_6', -1.), ('cube_probit_2', 1.), ('cube_probit_4', 1.), ('link_probit_all_zero', -1.)],
}
FINE_WEIGHT, SCALE = .2, 20
MASKED_MULTIPLIERS = {'surprise_lift': 4., 'dist_p5000bpe_profile': 2., 'masked_lift_completion': 2.}


def collect(fold_prediction):
    """fold_prediction(k) -> probabilities on the query rows of fold k; returns OOF and test (mean of the 5 folds)."""
    f = fold_ids()
    oof, tests = np.zeros(len(f)), []
    for k in range(5):
        valid = split_indices(k)[1]
        p = fold_prediction(k)
        assert np.isfinite(p).all() and p.min() > 0 and p.max() < 1
        oof[valid] = p[:len(valid)]
        tests.append(p[len(valid):])
    return oof, np.mean(tests, axis=0)


def stage3_logit(k):
    s = [42, 43]
    return mean_logit('gen0', k, s) + 4 * (mean_logit('n2_reg_real', k, s) - mean_logit('n2_reg_zero', k, s))


def income_mixture_contrast():
    return collect(lambda k: expit(stage3_logit(k)))


def income_context_svd_contrast():
    def fold(k):
        base = stage3_logit(k)
        deltas = []
        for family in ('emb8', 'gen_difference'):
            groups = [mean_logit(f'{family}_real', k, s) - mean_logit(f'{family}_zero', k, s)
                      for s in (range(42, 50), range(50, 58))]
            deltas.append((groups[0] + groups[1]) / 2)
        p = expit(base + 4 * deltas[0] + 4 * deltas[1])
        STAGE[4].mkdir(parents=True, exist_ok=True)
        np.savez(STAGE[4] / f'average16_fold{k}.npz', index=split_indices(k)[2], prediction=p, embedding=deltas[0],
                 generator=deltas[1], base_logit=base)
        return p
    return collect(fold)


def fine_context_contrast():
    def fold(k):
        query = split_indices(k)[2]
        zs = {c: mean_logit(c, k, ALL16) for terms in FINE_TERMS.values() for c, _ in terms}
        delta = np.zeros(len(query))
        for terms in FINE_TERMS.values():
            delta += FINE_WEIGHT * sum(w * zs[c] for c, w in terms)
        old = np.load(STAGE[4] / f'average16_fold{k}.npz')
        assert np.array_equal(old['index'], query)
        latent = logit(old['prediction'].clip(1e-15, 1 - 1e-15)) + SCALE * delta
        p = expit(latent)
        STAGE[5].mkdir(parents=True, exist_ok=True)
        np.savez(STAGE[5] / f'export_all16_f{k}.npz', index=query, prediction=p, latent=latent, correction=delta)
        return p
    return collect(fold)


def coarse_context_contrast():
    def fold(k):
        query = split_indices(k)[2]
        d = FINE_WEIGHT * (mean_logit('coarse_entropy_real', k, ALL16) - mean_logit('coarse_entropy_zero', k, ALL16))
        primary = np.load(STAGE[5] / f'export_all16_f{k}.npz')
        assert np.array_equal(primary['index'], query)
        p = expit(primary['latent'] + SCALE * d)
        np.savez(STAGE[5] / f'coarse_all16_f{k}.npz', index=query, prediction=p, correction=d)
        return p
    return collect(fold)


def masked_density_contrast():
    def fold(k):
        query = split_indices(k)[2]
        delta = np.zeros(len(query))
        for family, multiplier in MASKED_MULTIPLIERS.items():
            delta += multiplier * (mean_logit(f'{family}_real', k, ALL16) - mean_logit(f'{family}_zero', k, ALL16))
        old = np.load(STAGE[5] / f'coarse_all16_f{k}.npz')
        assert np.array_equal(old['index'], query)
        anchor = logit(old['prediction'].astype('float64').clip(1e-15, 1 - 1e-15)).clip(-30, 30)
        return expit((.5 * anchor + delta).clip(-30, 30))
    return collect(fold)


def copula(k):
    """C on the query rows of fold k: sigmoid of the mean z over the 8 seeds."""
    root = CONTRAST / 'copula' / 'models' / 'production_raw' / f'fold_{k}'
    return expit(np.mean([z(np.r_[np.load(root / f'seed_{s}' / 'valid.npy'), np.load(root / f'seed_{s}' / 'test.npy')])
                          for s in range(42, 50)], axis=0))


def copula_gam_contrast():
    def fold(k):
        root = CONTRAST / 'additive_runs' / 'logit_p10_cross10_converged' / f'fold_{k}'
        additive = np.r_[np.load(root / 'valid.npy'), np.load(root / 'test.npy')]
        margin = np.load(CONTRAST / 'margins' / 'response_probit' / f'fold_{k}' / 'query.npy')
        g = expit(np.clip(.5 * z(additive) + .5 * margin, -30, 30))
        r = 1.
        return expit(np.clip((1 + r) * z(copula(k)) - r * z(g), -30, 30))
    return collect(fold)


def copula_gpt2_contrast():
    def fold(k):
        ep = GPT_RESIDUAL / 'gpt_runs' / 'lr5e5' / f'fold_{k}' / 'epoch_1'
        n = np.r_[np.load(ep / 'valid.npy'), np.load(ep / 'test.npy')]
        return expit(np.clip(3 * z(copula(k)) - 2 * z(n), -30, 30))
    return collect(fold)


PAIRS = {f.__name__: f for f in (income_mixture_contrast, income_context_svd_contrast, fine_context_contrast,
                                 coarse_context_contrast, masked_density_contrast, copula_gam_contrast, copula_gpt2_contrast)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='+', choices=list(PAIRS))
    for name in ap.parse_args().pairs:
        oof, test = PAIRS[name]()
        save_logits(name, oof, test)


if __name__ == '__main__':
    main()
