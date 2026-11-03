"""Source-tempered exact-token responses and their compact residual contrasts ('ares').

For 6 fields (age, daily, social, notifications, app opens, weekend) and each class, a weighted power law
log1p(train count) ~ a + b log1p(original count) over the exact two-decimal tokens shared by the original and the
training rows maps the original's class counts to the competition's; the implied class probability of a token is a
generative prior. Per field: prior logit, exact-token posteriors (sum + alpha prior) / (count + alpha) for
alpha 5 / 20 / 80, posterior(20) minus the standard smoothed rate (m 20, global prior), log1p(count), plus 8 row
aggregates (44 columns). Outer-training rows are encoded from the other 4 splitmix64(id) groups, validation and test
rows from all outer-training rows.
The model uses 14 contrasts: per field posterior(5) - posterior(80) and posterior(5) - prior, and their row sums."""
import json

import numpy as np
import pandas as pd

from common.data import TARGET

from .external import xgb_cache

FIELDS = ("age", "daily_screen_time_hours", "social_media_hours", "notifications_per_day", "app_opens_per_day",
          "weekend_screen_time")
ALPHAS = (5.0, 20.0, 80.0)
CONTRASTS = ("sharp_minus_smooth", "sharp_minus_source", "source_posterior_minus_standard")


def tokens(series):
    if pd.api.types.is_numeric_dtype(series.dtype):
        # integral original and float competition values share one spelling (24 and 24.0 -> '24.0')
        numeric = pd.to_numeric(series, errors="coerce").astype(np.float64).round(2)
        return numeric.astype("string").fillna("__MISSING__")
    return series.astype("string").fillna("__MISSING__")


def logit(probability):
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def tempered_names():
    names = []
    for field in FIELDS:
        names += [f"source_tempered__{field}__{suffix}" for suffix in (
            "prior_logit", "posterior_a5_logit", "posterior_a20_logit", "posterior_a80_logit",
            "posterior_a20_minus_standard", "log_count")]
    names += ["source_tempered__aggregate__prior_logit_mean", "source_tempered__aggregate__prior_logit_std",
              "source_tempered__aggregate__posterior_a20_mean", "source_tempered__aggregate__posterior_a20_std",
              "source_tempered__aggregate__posterior_a20_min", "source_tempered__aggregate__posterior_a20_max",
              "source_tempered__aggregate__delta_mean", "source_tempered__aggregate__log_count_mean"]
    return names


def weighted_power_fit(source_count, synthetic_count):
    shared = source_count.index.intersection(synthetic_count.index)
    x = np.log1p(source_count.reindex(shared).to_numpy(np.float64))
    y = np.log1p(synthetic_count.reindex(shared).to_numpy(np.float64))
    weight = synthetic_count.reindex(shared).to_numpy(np.float64)
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    root_weight = np.sqrt(np.maximum(weight, 1.0))
    coefficient = np.linalg.lstsq(design * root_weight[:, None], y * root_weight, rcond=None)[0]
    return float(coefficient[0]), float(coefficient[1])


def fit_apply(fit, fit_y, apply, source):
    """The 44 source-tempered columns of the `apply` rows, fitted on the `fit` rows (and the original)."""
    global_prior = float(np.mean(fit_y))
    prior_blocks, posterior20_blocks, delta_blocks, count_blocks, blocks = [], [], [], [], []
    source_y = source[TARGET].to_numpy(np.int8)
    for field in FIELDS:
        fit_token, apply_token, source_token = tokens(fit[field]), tokens(apply[field]), tokens(source[field])
        source_counts, synthetic_counts, coefficients = {}, {}, {}
        for label in (0, 1):
            source_counts[label] = source_token[source_y == label].value_counts(sort=False)
            synthetic_counts[label] = fit_token[fit_y == label].value_counts(sort=False)
            coefficients[label] = weighted_power_fit(source_counts[label], synthetic_counts[label])
        union = source_counts[0].index.union(source_counts[1].index).union(
            synthetic_counts[0].index.union(synthetic_counts[1].index))
        predicted = {}
        for label in (0, 1):
            intercept, slope = coefficients[label]
            source_value = source_counts[label].reindex(union).fillna(0).to_numpy(np.float64)
            predicted[label] = np.maximum(np.expm1(intercept + slope * np.log1p(source_value)), 1e-3)
        generative_map = pd.Series(predicted[1] / (predicted[0] + predicted[1]), index=union)
        apply_prior = apply_token.map(generative_map).fillna(global_prior).to_numpy(np.float64)

        statistics = pd.DataFrame({"token": fit_token, "target": fit_y}).groupby(
            "token", sort=False, observed=True)["target"].agg(["sum", "count"])
        apply_sum = apply_token.map(statistics["sum"]).fillna(0).to_numpy(np.float64)
        apply_count = apply_token.map(statistics["count"]).fillna(0).to_numpy(np.float64)
        standard_probability = (apply_sum + 20.0 * global_prior) / (apply_count + 20.0)
        prior_logit = logit(apply_prior)
        posterior_logits = [logit((apply_sum + alpha * apply_prior) / (apply_count + alpha)) for alpha in ALPHAS]
        delta = posterior_logits[1] - logit(standard_probability)
        log_count = np.log1p(apply_count)
        blocks += [prior_logit, posterior_logits[0], posterior_logits[1], posterior_logits[2], delta, log_count]
        prior_blocks.append(prior_logit)
        posterior20_blocks.append(posterior_logits[1])
        delta_blocks.append(delta)
        count_blocks.append(log_count)
    prior_matrix, posterior_matrix = np.column_stack(prior_blocks), np.column_stack(posterior20_blocks)
    delta_matrix, count_matrix = np.column_stack(delta_blocks), np.column_stack(count_blocks)
    blocks += [prior_matrix.mean(axis=1), prior_matrix.std(axis=1), posterior_matrix.mean(axis=1),
               posterior_matrix.std(axis=1), posterior_matrix.min(axis=1), posterior_matrix.max(axis=1),
               delta_matrix.mean(axis=1), count_matrix.mean(axis=1)]
    return np.column_stack(blocks).astype(np.float32)


def ares_names():
    names = [f"ares__{field}__{contrast}" for field in FIELDS for contrast in CONTRASTS]
    return names + [f"ares__aggregate__{contrast}_sum" for contrast in CONTRASTS]


def reduce_features(values, upstream_names):
    """44 source-tempered columns -> 21 contrasts (3 per field + 3 row sums)."""
    values = np.asarray(values, np.float32)
    lookup = {name: index for index, name in enumerate(upstream_names)}
    per_field, by_contrast = [], [[] for _ in CONTRASTS]
    for field in FIELDS:
        prior = values[:, lookup[f"source_tempered__{field}__prior_logit"]]
        posterior5 = values[:, lookup[f"source_tempered__{field}__posterior_a5_logit"]]
        posterior80 = values[:, lookup[f"source_tempered__{field}__posterior_a80_logit"]]
        delta = values[:, lookup[f"source_tempered__{field}__posterior_a20_minus_standard"]]
        for index, contrast in enumerate((posterior5 - posterior80, posterior5 - prior, delta)):
            cast = np.asarray(contrast, np.float32)
            per_field.append(cast)
            by_contrast[index].append(cast)
    aggregates = [np.sum(np.column_stack(block), axis=1, dtype=np.float32) for block in by_contrast]
    return np.column_stack([*per_field, *aggregates]).astype(np.float32)


def load_ares(fold):
    """The 14 ares columns (source_posterior_minus_standard dropped) of fold `fold` from build_residual_ares.py:
    names, outer-training rows, validation rows, test rows."""
    root = xgb_cache() / "ares_production" / f"fold_{fold}"
    upstream_names = list(json.loads((root / "manifest.json").read_text())["feature_names"])
    names = ares_names()
    keep = [i for i, name in enumerate(names) if "source_posterior_minus_standard" not in name]
    out = [reduce_features(np.load(root / f"{role}_features.npy"), upstream_names)[:, keep]
           for role in ("train", "valid", "test")]
    return [names[i] for i in keep], out[0], out[1], out[2]
