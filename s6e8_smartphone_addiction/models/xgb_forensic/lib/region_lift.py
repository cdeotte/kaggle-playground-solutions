"""Strict region x exact-value interaction lifts (9 columns).

Region of a row from the original rule on the raw daily / social hours: 0 = daily <= 6 and social <= 4, 2 = daily > 8
or social > 4, 1 = otherwise, 3 = daily or social missing. For each numeric field: smoothed (m 20) target rates of the
(exact value, region) pair, of the exact value and of the region; lift = logit(pair) - logit(value) - logit(region)
+ logit(prior). Training rows are complement-cross-fitted over splitmix64(id) mod 5 owner groups (their prior from
the other 4 groups); application rows use all training rows. Rows of region 3 get NaN."""
import numpy as np

from .residual_surfaces import splitmix_group

FIELDS = ("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
          "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
SMOOTHING = 20.0


def token(value):
    """Exact float64 bit pattern of each value (all NaNs share one token)."""
    canonical = np.asarray(value, np.float64).copy()
    canonical[~np.isfinite(canonical)] = np.nan
    return canonical.view(np.int64).copy()


def region(frame):
    daily = frame["daily_screen_time_hours"].to_numpy(np.float64)
    social = frame["social_media_hours"].to_numpy(np.float64)
    observed = np.isfinite(daily) & np.isfinite(social)
    low = (daily <= 6.0) & (social <= 4.0)
    high = (daily > 8.0) | (social > 4.0)
    out = np.full(len(frame), 3, dtype=np.int64)
    out[observed & low] = 0
    out[observed & ~low & ~high] = 1
    out[observed & high] = 2
    return out


def posterior(fit_key, fit_target, apply_key, prior):
    def packed(value):
        array = np.ascontiguousarray(value, dtype=np.int64)
        if array.ndim == 1:
            return array
        return array.view(np.dtype((np.void, array.dtype.itemsize * array.shape[1]))).ravel()

    fit_packed = packed(fit_key)
    unique, inverse = np.unique(fit_packed, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    total = np.bincount(inverse, weights=np.asarray(fit_target, np.float64))
    rate = (total + SMOOTHING * prior) / (count + SMOOTHING)
    query = packed(apply_key)
    location = np.searchsorted(unique, query)
    found = location < len(unique)
    found[found] &= unique[location[found]] == query[found]
    output = np.full(len(query), prior, dtype=np.float64)
    output[found] = rate[location[found]]
    return output


def logit(probability):
    value = np.clip(np.asarray(probability, np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(value / (1.0 - value))


def build_strict_lift9(fit_frame, apply_frame, fit_target, fit_ids):
    """(training-row lifts, application-row lifts, names)."""
    target = np.asarray(fit_target, np.float64)
    owner = splitmix_group(np.asarray(fit_ids, np.int64))
    prior = float(target.mean())
    fit_region, apply_region = region(fit_frame), region(apply_frame)
    fit_columns, apply_columns, names = [], [], []
    for field in FIELDS:
        fit_token, apply_token = token(fit_frame[field].to_numpy()), token(apply_frame[field].to_numpy())
        fit_pair = np.column_stack((fit_token, fit_region))
        apply_pair = np.column_stack((apply_token, apply_region))
        pair_train, field_train, region_train, owner_prior_logit = (np.empty(len(target), np.float64)
                                                                    for _ in range(4))
        for held in range(5):
            application, training = owner == held, owner != held
            local_prior = float(target[training].mean())
            owner_prior_logit[application] = logit(local_prior)
            pair_train[application] = posterior(fit_pair[training], target[training], fit_pair[application],
                                                local_prior)
            field_train[application] = posterior(fit_token[training], target[training], fit_token[application],
                                                 local_prior)
            region_train[application] = posterior(fit_region[training], target[training], fit_region[application],
                                                  local_prior)
        pair_apply = posterior(fit_pair, target, apply_pair, prior)
        field_apply = posterior(fit_token, target, apply_token, prior)
        region_apply = posterior(fit_region, target, apply_region, prior)
        fit_columns.append(logit(pair_train) - logit(field_train) - logit(region_train) + owner_prior_logit)
        apply_columns.append(logit(pair_apply) - logit(field_apply) - logit(region_apply) + logit(prior))
        names.append(f"region_lift__{field}__interaction_lift")
    fit = np.column_stack(fit_columns).astype(np.float32)
    apply = np.column_stack(apply_columns).astype(np.float32)
    fit[fit_region == 3] = np.nan
    apply[apply_region == 3] = np.nan
    return fit, apply, names
