"""Repair-posterior channels (20, target-free).

Rows whose four screen fields are complete concentrate at (social + gaming + work) / daily in [0.85, 0.95): the
generator appears to repair implausible screen compositions into that band. With the global density mixture (each
field masked, tempered, special tokens removed) and the row filled by conditional means where cells are missing:
  posterior_mass (4)         mass of the conditional of daily / social / gaming / work that lies inside the value
                             interval that would put the row's utilization inside the band
  conditional_delta_z (4)    (band-conditioned mean - unconditional mean) / conditional std, clipped to +-12
  mass mean / min / max / std of the four masses
  coupled (8)                utilization of the filled row, its delta-method std from the conditional stds of the missing
                             parts, normal probability of the band, distance to the band in z units, L2 size of the
                             projection of the parts onto 0.90 x daily (in field std units), number of missing screen
                             fields, observed band indicator for complete rows (-1 otherwise), probability that
                             daily + sleep >= 19.995 (the 24-hour cap)"""
import numpy as np

from .global_density import BATCH_SIZE, numeric_values, raw_numeric, tempered_mixture
from .lattice import FIELDS, NUMERIC_COUNT

SCREEN_INDICES = (1, 2, 3, 4)
REPAIR_NAMES = tuple(f'repair__{FIELDS[i]}__posterior_mass' for i in SCREEN_INDICES) + tuple(
    f'repair__{FIELDS[i]}__conditional_delta_z' for i in SCREEN_INDICES) + (
    'repair__aggregate__mass_mean', 'repair__aggregate__mass_min', 'repair__aggregate__mass_max',
    'repair__aggregate__mass_std', 'repair__coupled__filled_utilization', 'repair__coupled__utilization_std',
    'repair__coupled__normal_band_probability', 'repair__coupled__distance_to_band_z',
    'repair__coupled__component_projection_l2', 'repair__coupled__screen_missing_count',
    'repair__coupled__complete_observed_branch', 'repair__coupled__sleep_daily_cap_probability')


def _normal_interval_probability(mean, std, low, high):
    from scipy.special import ndtr
    safe = np.maximum(std, 1e-7)
    probability = ndtr((high - mean) / safe) - ndtr((low - mean) / safe)
    deterministic = std < 1e-7
    probability[deterministic] = ((mean[deterministic] >= low) & (mean[deterministic] < high)).astype(np.float64)
    return np.clip(probability, 0.0, 1.0)


def score(torch, models, encoder, tokens_np, scalars_np, batch_size=BATCH_SIZE):
    rows = len(tokens_np)
    means = np.empty((rows, NUMERIC_COUNT), dtype=np.float32)
    stds = np.empty_like(means)
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            tokens = torch.from_numpy(tokens_np[start:stop]).cuda()
            scalars = torch.from_numpy(scalars_np[start:stop]).cuda()
            for i in range(NUMERIC_COUNT):
                p = tempered_mixture(torch, models, tokens, scalars, i, drop_specials=True)
                values = numeric_values(torch, encoder, i, p.shape[1])
                mean = (p * values[None, :]).sum(dim=1)
                variance = (p * (values[None, :] - mean[:, None]).square()).sum(dim=1)
                means[start:stop, i] = mean.cpu().numpy()
                stds[start:stop, i] = variance.clamp_min(1e-8).sqrt().cpu().numpy()
            del tokens, scalars

    raw = raw_numeric(tokens_np, encoder)
    missing = tokens_np[:, :NUMERIC_COUNT] < 3
    filled = np.where(missing, means, raw).astype(np.float32)
    daily, social, gaming, work = (filled[:, k].astype(np.float64) for k in (1, 2, 3, 4))
    parts = social + gaming + work
    # value interval of each field that puts (social + gaming + work) / daily inside [0.85, 0.95)
    interval_low = {1: parts / 0.95, 2: 0.85 * daily - gaming - work, 3: 0.85 * daily - social - work,
                    4: 0.85 * daily - social - gaming}
    interval_high = {1: parts / 0.85, 2: 0.95 * daily - gaming - work, 3: 0.95 * daily - social - work,
                     4: 0.95 * daily - social - gaming}
    repair_mass = np.empty((rows, len(SCREEN_INDICES)), dtype=np.float32)
    repair_delta = np.empty_like(repair_mass)
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            tokens = torch.from_numpy(tokens_np[start:stop]).cuda()
            scalars = torch.from_numpy(scalars_np[start:stop]).cuda()
            for out, i in enumerate(SCREEN_INDICES):
                p = tempered_mixture(torch, models, tokens, scalars, i, drop_specials=True)
                values = numeric_values(torch, encoder, i, p.shape[1])
                low = torch.from_numpy(interval_low[i][start:stop].astype(np.float32)).cuda()
                high = torch.from_numpy(interval_high[i][start:stop].astype(np.float32)).cuda()
                inside = (values[None, :] >= low[:, None]) & (values[None, :] < high[:, None])
                mass = (p * inside).sum(dim=1)
                conditional_sum = (p * inside * values[None, :]).sum(dim=1)
                unconditional = torch.from_numpy(means[start:stop, i]).cuda()
                conditional = torch.where(mass > 1e-7, conditional_sum / mass.clamp_min(1e-7), unconditional)
                scale = torch.from_numpy(stds[start:stop, i]).cuda().clamp_min(1e-4)
                repair_mass[start:stop, out] = mass.cpu().numpy()
                repair_delta[start:stop, out] = ((conditional - unconditional) / scale).clamp(-12.0, 12.0).cpu().numpy()
            del tokens, scalars

    safe_daily = np.maximum(np.abs(daily), 1e-4)
    utilization = parts / safe_daily
    parts_variance = sum(np.where(missing[:, k], stds[:, k].astype(np.float64) ** 2, 0.0) for k in (2, 3, 4))
    daily_variance = np.where(missing[:, 1], stds[:, 1].astype(np.float64) ** 2, 0.0)
    utilization_std = np.sqrt(np.maximum(parts_variance / safe_daily ** 2 + parts ** 2 * daily_variance / safe_daily ** 4,
                                         0.0))
    normal_band_probability = _normal_interval_probability(utilization, utilization_std, 0.85, 0.95)
    distance = np.maximum.reduce((0.85 - utilization, utilization - 0.95, np.zeros(rows)))
    distance_z = distance / np.sqrt(utilization_std ** 2 + 0.01 ** 2)
    component_scale = np.divide(0.90 * daily, np.maximum(parts, 1e-4), out=np.ones_like(parts),
                                where=np.abs(parts) > 1e-4)
    current = np.column_stack((social, gaming, work))
    projected = current * component_scale[:, None]
    field_scales = np.asarray(encoder.stds[2:5], dtype=np.float64)
    projection_l2 = np.sqrt(np.square((projected - current) / np.maximum(field_scales, 1e-4)).sum(axis=1))
    screen_missing_count = missing[:, 1:5].sum(axis=1).astype(np.float64)
    complete_screen = screen_missing_count == 0
    observed_branch = np.full(rows, -1.0, dtype=np.float64)
    observed_branch[complete_screen] = ((utilization[complete_screen] >= 0.85)
                                        & (utilization[complete_screen] < 0.95)).astype(np.float64)
    cap_mean = daily + filled[:, 5].astype(np.float64)
    cap_std = np.sqrt(np.where(missing[:, 1], stds[:, 1].astype(np.float64) ** 2, 0.0)
                      + np.where(missing[:, 5], stds[:, 5].astype(np.float64) ** 2, 0.0))
    cap_probability = _normal_interval_probability(cap_mean, cap_std, 19.995, 1e9)

    columns = [repair_mass[:, k] for k in range(4)] + [repair_delta[:, k] for k in range(4)]
    columns += [repair_mass.mean(axis=1), repair_mass.min(axis=1), repair_mass.max(axis=1), repair_mass.std(axis=1),
                utilization, utilization_std, normal_band_probability, distance_z, projection_l2,
                screen_missing_count, observed_branch, cap_probability]
    matrix = np.column_stack(columns).astype(np.float32)
    assert matrix.shape == (rows, len(REPAIR_NAMES)) and np.isfinite(matrix).all()
    return matrix
