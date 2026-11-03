"""Density-completion response channels (30; the RealMLP uses the 10 row aggregates).

For each of the 5 screen fields (daily, social, gaming, work/study, weekend) the field is masked and the density model
gives its full conditional distribution over the value lattice. That distribution is integrated against an exact-token
target-rate table (smoothing 20 toward the prior): context expectation, its std over the distribution and the
normalized entropy. Observed cells keep their exact-token rate ('observed_or_context'); missing cells use the
expectation. Rate tables are fit on the outer-training rows only. Outer-training rows are split into 5 groups by a
splitmix64 hash of their id; a training row reads the table fit without its own group (so it never sees its own label),
outer-validation and test rows read the table fit on all outer-training rows."""
import math

import numpy as np

from .lattice import FIELDS, SPECIAL_MASK

RESPONSE_FIELDS = ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                   'weekend_screen_time')
RESPONSE_FIELD_INDICES = tuple(FIELDS.index(f) for f in RESPONSE_FIELDS)
INNER_GROUPS = 5
SMOOTHING = np.float64(20.0)


def mix_group(ids):
    value = np.asarray(ids, dtype=np.uint64).copy()
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return np.asarray(value % np.uint64(INNER_GROUPS), dtype=np.int8)


def logit(probability):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def response_vector(tokens, target, cardinality):
    prior = float(np.mean(target))
    observed = tokens >= 3
    count = np.bincount(tokens[observed], minlength=cardinality).astype(np.float64)
    total = np.bincount(tokens[observed], weights=target[observed], minlength=cardinality).astype(np.float64)
    probability = (total + SMOOTHING * prior) / (count + SMOOTHING)
    probability[:3] = prior
    return probability, prior


def fit_response_tables(train_tokens, target, groups, cardinalities):
    grouped, full, priors = {}, {}, {}
    for i in RESPONSE_FIELD_INDICES:
        rows = []
        for g in range(INNER_GROUPS):
            active = groups != g
            rows.append(response_vector(train_tokens[active, i], target[active], cardinalities[i])[0])
        grouped[i] = np.stack(rows)
        full[i], priors[i] = response_vector(train_tokens[:, i], target, cardinalities[i])
    return grouped, full, priors


def feature_names():
    names = []
    for field in RESPONSE_FIELDS:
        names += [f'density_completion__{field}__context_expected_logit',
                  f'density_completion__{field}__observed_or_context_logit',
                  f'density_completion__{field}__context_response_std',
                  f'density_completion__{field}__context_entropy']
    names += ['density_completion__aggregate__completion_logit_mean',
              'density_completion__aggregate__completion_logit_std',
              'density_completion__aggregate__completion_logit_min',
              'density_completion__aggregate__completion_logit_max',
              'density_completion__aggregate__context_logit_mean',
              'density_completion__aggregate__context_logit_std',
              'density_completion__aggregate__missing_context_logit_mean',
              'density_completion__aggregate__missing_context_logit_std',
              'density_completion__aggregate__missing_field_count',
              'density_completion__aggregate__context_confidence_mean']
    return names


AGGREGATE_INDICES = [i for i, n in enumerate(feature_names()) if '__aggregate__' in n]


def single_model_conditional(torch, model):
    """Conditional distribution of one masked field under one density model (special tokens removed)."""
    def conditional(tokens, scalars, field_index):
        masked_tokens, masked_scalars = tokens.clone(), scalars.clone()
        masked_tokens[:, field_index] = SPECIAL_MASK
        masked_scalars[:, field_index] = 0.0
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, encoded = model(masked_tokens, masked_scalars)
            logits = model.heads[field_index](encoded[:, field_index]).float()
        probability = torch.softmax(logits, dim=1)
        probability[:, :3] = 0.0
        probability /= probability.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return probability
    return conditional


def score(torch, conditional, tokens_np, scalars_np, groups, grouped, full, priors, batch_size):
    """-> (rows x 30) float32. groups: hash group of each row (training rows) or None (valid / test rows)."""
    rows, n = len(tokens_np), len(RESPONSE_FIELD_INDICES)
    context_p = np.empty((rows, n), np.float32)
    completion_p = np.empty((rows, n), np.float32)
    context_std = np.empty((rows, n), np.float32)
    entropy = np.empty((rows, n), np.float32)
    missing_matrix = np.empty((rows, n), bool)
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            tokens = torch.from_numpy(tokens_np[start:stop]).cuda()
            scalars = torch.from_numpy(scalars_np[start:stop]).cuda()
            group_batch = groups[start:stop] if groups is not None else None
            for out, i in enumerate(RESPONSE_FIELD_INDICES):
                probability = conditional(tokens, scalars, i)
                if group_batch is None:
                    response_np = np.broadcast_to(full[i], (stop - start, probability.shape[1]))
                else:
                    response_np = grouped[i][group_batch]
                response = torch.from_numpy(np.asarray(response_np, dtype=np.float32)).cuda()
                expected = (probability * response).sum(dim=1)
                variance = (probability * (response - expected[:, None]).square()).sum(dim=1)
                real_cardinality = max(int(probability.shape[1]) - 3, 2)
                normalized_entropy = -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(dim=1) \
                    / math.log(real_cardinality)
                token_np = tokens_np[start:stop, i]
                exact = full[i][token_np] if group_batch is None else response_np[np.arange(stop - start), token_np]
                missing = token_np < 3
                expected_np = expected.cpu().numpy().astype(np.float64)
                context_p[start:stop, out] = expected_np
                completion_p[start:stop, out] = np.where(missing, expected_np, exact)
                context_std[start:stop, out] = variance.sqrt().cpu().numpy()
                entropy[start:stop, out] = normalized_entropy.cpu().numpy()
                missing_matrix[start:stop, out] = missing
                del probability, response
            del tokens, scalars

    context_logit, completion_logit = logit(context_p), logit(completion_p)
    blocks = []
    for k in range(n):
        blocks += [context_logit[:, k], completion_logit[:, k], context_std[:, k], entropy[:, k]]
    missing_context = np.where(missing_matrix, context_logit, np.nan)
    with np.errstate(all='ignore'):
        missing_mean = np.nanmean(missing_context, axis=1)
        missing_std = np.nanstd(missing_context, axis=1)
    blocks += [completion_logit.mean(axis=1), completion_logit.std(axis=1), completion_logit.min(axis=1),
               completion_logit.max(axis=1), context_logit.mean(axis=1), context_logit.std(axis=1),
               np.nan_to_num(missing_mean, nan=0.0), np.nan_to_num(missing_std, nan=0.0),
               missing_matrix.sum(axis=1).astype(np.float64), (1.0 - entropy).mean(axis=1)]
    matrix = np.column_stack(blocks).astype(np.float32)
    assert matrix.shape == (rows, len(feature_names())) and np.isfinite(matrix).all()
    return matrix


def build_fold(torch, conditional, encoder, train_frame, y, fold_train, fold_valid, test_frame, batch_size):
    """30 completion channels for the outer-train rows (hash-group complement tables), the outer-valid rows and the
    test rows (tables fit on all outer-train rows). train_frame must contain the 'id' column."""
    fit = train_frame.iloc[fold_train].reset_index(drop=True)
    fit_tokens, fit_scalars = encoder.transform(fit)
    groups = mix_group(fit['id'].to_numpy(np.int64))
    grouped, full, priors = fit_response_tables(fit_tokens, y[fold_train], groups, encoder.cardinalities)
    out = {'train': score(torch, conditional, fit_tokens, fit_scalars, groups, grouped, full, priors, batch_size)}
    valid_tokens, valid_scalars = encoder.transform(train_frame.iloc[fold_valid].reset_index(drop=True))
    out['valid'] = score(torch, conditional, valid_tokens, valid_scalars, None, grouped, full, priors, batch_size)
    if test_frame is not None:
        test_tokens, test_scalars = encoder.transform(test_frame)
        out['test'] = score(torch, conditional, test_tokens, test_scalars, None, grouped, full, priors, batch_size)
    return out
