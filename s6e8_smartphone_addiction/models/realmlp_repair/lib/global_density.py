"""Global (train + test predictors) masked density mixture and its channels.

Four target-free density transformers share one LatticeEncoder fit on the train + test + original predictors:
  g4           width 384, 8 layers, 60 epochs, Bernoulli-0.15 masks, rows = train + test predictors
  g5           width 448, 8 layers, 80 epochs, lr 0.0009
  g4orig8      g4 with the 7,500 original rows appended 8 times
  g9unitarget  g4 architecture, half of the rows hide exactly one observed field (60 epochs)
The conditional distribution of a masked field is the equal (0.25 each) mixture of the four softmaxes, raised to
1 / T_field and renormalized (field temperatures below, chosen on held-out masked predictor cells).

Channels per row (computed once for every train and test row; they read no target):
  gden    for each of the 12 fields: surprisal of the observed token (-log p / log V) and normalized entropy (24)
  mfill   the 9 numerics with missing cells filled by the conditional mean, the observed-cell residual z-scores
          (value - conditional mean) / conditional std clipped to +-12, and 9 screen-geometry values recomputed on
          the filled row (27)"""
import math

import numpy as np

from .lattice import FIELDS, NUMERIC_COUNT, SPECIAL_MASK, LatticeEncoder, concat_predictors

_G4 = dict(embedding_dim=384, layers=8, heads=8, ff_multiplier=3, dropout=0.08, learning_rate=0.001,
           weight_decay=0.00002, batch_size=2048, epochs=60, mask_probability=0.15, projection_dim=32)
COMPONENTS = {
    'g4': dict(config=_G4, seed=90142, original_repeat=0, scheme='masked'),
    'g4orig8': dict(config=_G4, seed=92142, original_repeat=8, scheme='masked'),
    'g5': dict(config=dict(_G4, embedding_dim=448, learning_rate=0.0009, epochs=80), seed=91142, original_repeat=0,
               scheme='masked'),
    'g9unitarget': dict(config=_G4, seed=20260851, original_repeat=0, scheme='unitarget'),
}
WEIGHTS = {'g4': 0.25, 'g4orig8': 0.25, 'g5': 0.25, 'g9unitarget': 0.25}
FIELD_TEMPERATURES = {'age': 1.0, 'daily_screen_time_hours': 1.1, 'social_media_hours': 1.1, 'gaming_hours': 1.0,
                      'work_study_hours': 1.1, 'sleep_hours': 1.1, 'notifications_per_day': 1.0,
                      'app_opens_per_day': 1.0, 'weekend_screen_time': 1.1, 'gender': 1.0, 'stress_level': 1.0,
                      'academic_work_impact': 1.2}
BATCH_SIZE = 2048
MFILL_NAMES = ([f'mfillvx__{f}__filled' for f in FIELDS[:NUMERIC_COUNT]]
               + [f'mfillvx__{f}__observed_residual_z' for f in FIELDS[:NUMERIC_COUNT]]
               + ['mfillvx__geometry__parts_sum', 'mfillvx__geometry__slack', 'mfillvx__geometry__utilization',
                  'mfillvx__geometry__slack_share', 'mfillvx__geometry__social_share',
                  'mfillvx__geometry__gaming_share', 'mfillvx__geometry__work_share',
                  'mfillvx__geometry__weekend_minus_daily', 'mfillvx__geometry__weekend_over_daily'])
GDEN_NAMES = [f'gden__{f}__{m}' for f in FIELDS for m in ('surprisal', 'entropy')]


def global_encoder(train, test, original):
    return LatticeEncoder(concat_predictors(train, test, original))


def load_models(torch, encoder, state_path):
    """state_path(component) -> path of the saved state dict."""
    from .lattice import load_model
    return {c: load_model(torch, state_path(c), encoder.cardinalities, COMPONENTS[c]['config']) for c in COMPONENTS}


def tempered_mixture(torch, models, tokens, scalars, field_index, drop_specials):
    field = FIELDS[field_index]
    masked_tokens, masked_scalars = tokens.clone(), scalars.clone()
    masked_tokens[:, field_index] = SPECIAL_MASK
    masked_scalars[:, field_index] = 0.0
    mixture = None
    for component, model in models.items():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, encoded = model(masked_tokens, masked_scalars)
            logits = model.heads[field_index](encoded[:, field_index]).float()
        weighted = torch.softmax(logits, dim=1) * float(WEIGHTS[component])
        mixture = weighted if mixture is None else mixture + weighted
    mixture = mixture.clamp_min(1e-30).pow(1.0 / float(FIELD_TEMPERATURES[field]))
    mixture /= mixture.sum(dim=1, keepdim=True).clamp_min(1e-30)
    if drop_specials:
        mixture[:, :3] = 0.0
        mixture /= mixture.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return mixture


def mixture_conditional(torch, models):
    """Conditional of one masked field for the completion channels (special tokens removed)."""
    def conditional(tokens, scalars, field_index):
        return tempered_mixture(torch, models, tokens, scalars, field_index, drop_specials=True)
    return conditional


def numeric_values(torch, encoder, field_index, width):
    raw = np.zeros(width, dtype=np.float32)
    vocabulary = np.asarray(encoder.vocabulary_values[field_index], dtype=np.float32)
    raw[3:3 + len(vocabulary)] = vocabulary
    return torch.from_numpy(raw).cuda()


def score_channels(torch, models, encoder, tokens_np, scalars_np, batch_size=BATCH_SIZE):
    """-> (gden rows x 24, conditional means rows x 9, conditional stds rows x 9)."""
    rows = len(tokens_np)
    metrics = np.full((rows, len(FIELDS), 2), np.nan, dtype=np.float32)
    means = np.empty((rows, NUMERIC_COUNT), dtype=np.float32)
    stds = np.empty_like(means)
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            tokens = torch.from_numpy(tokens_np[start:stop]).cuda()
            scalars = torch.from_numpy(scalars_np[start:stop]).cuda()
            for i in range(len(FIELDS)):
                mixture = tempered_mixture(torch, models, tokens, scalars, i, drop_specials=False)
                observed = tokens[:, i] >= 3
                if observed.any():
                    actual = mixture[observed].gather(1, tokens[observed, i][:, None]).squeeze(1)
                    ent = -(mixture[observed].clamp_min(1e-12) * mixture[observed].clamp_min(1e-12).log()).sum(dim=1) \
                        / math.log(mixture.shape[1])
                    active = np.flatnonzero(observed.cpu().numpy()) + start
                    metrics[active, i, 0] = (-actual.clamp_min(1e-30).log() / math.log(mixture.shape[1])).cpu().numpy()
                    metrics[active, i, 1] = ent.cpu().numpy()
                if i < NUMERIC_COUNT:
                    p = mixture.clone()
                    p[:, :3] = 0.0
                    p /= p.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    values = numeric_values(torch, encoder, i, mixture.shape[1])
                    mean = (p * values[None, :]).sum(dim=1)
                    variance = (p * (values[None, :] - mean[:, None]).square()).sum(dim=1)
                    means[start:stop, i] = mean.cpu().numpy()
                    stds[start:stop, i] = variance.clamp_min(1e-8).sqrt().cpu().numpy()
                del mixture
            del tokens, scalars
    gden = np.column_stack([metrics[:, i, m] for i in range(len(FIELDS)) for m in (0, 1)]).astype(np.float32)
    return gden, means, stds


def raw_numeric(tokens, encoder):
    raw = np.empty((len(tokens), NUMERIC_COUNT), dtype=np.float32)
    for i in range(NUMERIC_COUNT):
        values = np.zeros(len(encoder.vocabulary_values[i]) + 3, dtype=np.float32)
        values[3:] = np.asarray(encoder.vocabulary_values[i], dtype=np.float32)
        raw[:, i] = values[tokens[:, i]]
    return raw


def make_mfill(tokens, encoder, conditional_mean, conditional_std):
    raw = raw_numeric(tokens, encoder)
    missing = tokens[:, :NUMERIC_COUNT] < 3
    filled = np.where(missing, conditional_mean, raw).astype(np.float32)
    residual_z = np.where(missing, 0.0, (raw - conditional_mean) / np.maximum(conditional_std, np.float32(1e-4)))
    residual_z = np.clip(residual_z, -12.0, 12.0).astype(np.float32)
    daily, social, gaming, work, weekend = (filled[:, k].astype(np.float64) for k in (1, 2, 3, 4, 8))
    parts = social + gaming + work
    denominator = np.maximum(np.abs(daily), 1e-4)
    geometry = np.column_stack([parts, daily - parts, parts / denominator, (daily - parts) / denominator,
                                social / denominator, gaming / denominator, work / denominator, weekend - daily,
                                weekend / denominator]).astype(np.float32)
    matrix = np.column_stack([filled, residual_z, geometry]).astype(np.float32)
    assert matrix.shape[1] == len(MFILL_NAMES) and np.isfinite(matrix).all()
    return matrix
