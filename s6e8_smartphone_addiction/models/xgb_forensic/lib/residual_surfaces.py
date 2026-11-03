"""Daily x social residual surfaces on the exact 0.01-hour grid, complement-cross-fitted.

The caller supplies a target residual of the training rows and a label-free owner group per training row
(splitmix_group of the row id). The residual is accumulated on the (daily, social) grid and smoothed with Gaussian
filters (sigma 1, 3, 8 grid ticks); a training row reads the smoothed sums of the other 4 owner groups, an application
row the sums of all training rows. Per sigma: smoothed residual / (smoothed count + 20) and log1p(smoothed count)."""
import numpy as np
from scipy.ndimage import gaussian_filter

GRID_STEP = 0.01
SIGMAS = (1.0, 3.0, 8.0)
SHRINKAGE = 20.0


def splitmix_group(ids, groups=5):
    """Label-free group of each row: splitmix64 finalizer of the id, modulo `groups`."""
    value = np.asarray(ids, dtype=np.uint64).copy()
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return np.asarray(value % np.uint64(groups), dtype=np.int8)


def _ticks(values):
    numeric = np.asarray(values, np.float64)
    finite = np.isfinite(numeric)
    ticks = np.zeros(len(numeric), dtype=np.int64)
    ticks[finite] = np.rint(numeric[finite] / GRID_STEP).astype(np.int64)
    return ticks, finite


def build_daily_social_surfaces(fit_daily, fit_social, apply_daily, apply_social, fit_residual, owner):
    """(training-row matrix, application-row matrix, names); 6 columns."""
    fit_daily, fit_social = np.asarray(fit_daily, np.float64), np.asarray(fit_social, np.float64)
    apply_daily, apply_social = np.asarray(apply_daily, np.float64), np.asarray(apply_social, np.float64)
    residual = np.asarray(fit_residual, np.float64)
    owner = np.asarray(owner, np.int8)

    fit_x_raw, fit_x_finite = _ticks(fit_daily)
    fit_y_raw, fit_y_finite = _ticks(fit_social)
    observed = fit_x_finite & fit_y_finite
    margin = int(np.ceil(4.0 * max(SIGMAS)))
    minimum_x = int(fit_x_raw[observed].min()) - margin
    maximum_x = int(fit_x_raw[observed].max()) + margin
    minimum_y = int(fit_y_raw[observed].min()) - margin
    maximum_y = int(fit_y_raw[observed].max()) + margin
    shape = (maximum_x - minimum_x + 1, maximum_y - minimum_y + 1)

    def coordinates(daily, social):
        x_raw, x_finite = _ticks(daily)
        y_raw, y_finite = _ticks(social)
        x, y = x_raw - minimum_x, y_raw - minimum_y
        valid = x_finite & y_finite & (x >= 0) & (x < shape[0]) & (y >= 0) & (y < shape[1])
        return np.clip(x, 0, shape[0] - 1), np.clip(y, 0, shape[1] - 1), valid

    fit_x, fit_y, fit_valid = coordinates(fit_daily, fit_social)
    apply_x, apply_y, apply_valid = coordinates(apply_daily, apply_social)
    fit_flat = fit_x * shape[1] + fit_y
    size = shape[0] * shape[1]
    full_count = np.bincount(fit_flat[fit_valid], minlength=size).reshape(shape).astype(np.float64)
    full_total = np.bincount(fit_flat[fit_valid], weights=residual[fit_valid], minlength=size).reshape(shape)
    group_counts, group_totals = [], []
    for held in range(5):
        chosen = fit_valid & (owner == held)
        group_counts.append(np.bincount(fit_flat[chosen], minlength=size).reshape(shape).astype(np.float64))
        group_totals.append(np.bincount(fit_flat[chosen], weights=residual[chosen], minlength=size).reshape(shape))

    fit_parts, apply_parts, names = [], [], []
    for sigma in SIGMAS:
        full_smoothed_count = gaussian_filter(full_count, sigma=sigma, mode="constant", cval=0.0)
        full_smoothed_total = gaussian_filter(full_total, sigma=sigma, mode="constant", cval=0.0)
        fit_response = np.zeros(len(fit_daily), dtype=np.float64)
        fit_log_count = np.zeros(len(fit_daily), dtype=np.float64)
        for held in range(5):
            held_count = gaussian_filter(group_counts[held], sigma=sigma, mode="constant", cval=0.0)
            held_total = gaussian_filter(group_totals[held], sigma=sigma, mode="constant", cval=0.0)
            complement_count = np.maximum(full_smoothed_count - held_count, 0.0)
            complement_total = full_smoothed_total - held_total
            query = (owner == held) & fit_valid
            local_count = complement_count[fit_x[query], fit_y[query]]
            fit_response[query] = complement_total[fit_x[query], fit_y[query]] / (local_count + SHRINKAGE)
            fit_log_count[query] = np.log1p(local_count)
        apply_response = np.zeros(len(apply_daily), dtype=np.float64)
        apply_log_count = np.zeros(len(apply_daily), dtype=np.float64)
        local_count = full_smoothed_count[apply_x[apply_valid], apply_y[apply_valid]]
        apply_response[apply_valid] = full_smoothed_total[apply_x[apply_valid], apply_y[apply_valid]] / (
            local_count + SHRINKAGE)
        apply_log_count[apply_valid] = np.log1p(local_count)
        fit_parts += [fit_response, fit_log_count]
        apply_parts += [apply_response, apply_log_count]
        names += [f"dsx__base_residual__sigma{sigma:g}", f"dsx__log_effective_count__sigma{sigma:g}"]
    return np.column_stack(fit_parts).astype(np.float32), np.column_stack(apply_parts).astype(np.float32), names
