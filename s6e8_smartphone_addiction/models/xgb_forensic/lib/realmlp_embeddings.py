"""RealMLP-style numeric coordinates and semantic interaction constraints for a pure XGBoost.

PeriodicEmbedding  for the 9 raw numerics: robust scaling (outer-training median / IQR, NaN -> median, clipped to
                   +-8), 2 random frequencies (normal, sd 0.94) and phases (uniform) per input from
                   default_rng(250314); sin and cos of 2 pi x scaled x frequency + phase (36 columns)
RawRBFBasis        for the 9 raw numerics: 16 Gaussian bumps at the (i + 0.5) / 16 quantiles of the scaled
                   outer-training values, bandwidth max(0.10, 1.25 x median positive center spacing) (144 columns)
semantic_constraints
                   XGBoost interaction constraints: the first 42 (parent) columns form a shared core that may interact
                   with every feature family (one clique core + family per family prefix below, plus any leftovers)."""
import numpy as np
import pandas as pd

RAW_NAMES = ("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
             "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
FAMILY_PREFIXES = (
    ("imputer_v2", ("imputer_v2_l2_",)),
    ("screen_response", ("screen_response_logit__",)),
    ("fixed_original_residual", ("binary_llr__", "pair_llr__", "pair_logcount__", "geometry__", "latent_logrel__")),
    ("pairlift_cbank", ("pairlift__", "cbank__")),
    ("temperature", ("temperature__",)),
    ("global_density", ("gden__",)),
    ("masked_fill", ("mfillvx__", "wbank__")),
    ("repair_posterior", ("repair__",)),
    ("density_completion", ("density_completion__",)),
    ("ares_dsx", ("ares__", "dsx__")),
    ("imputed_screen_response", ("imputed_screen_response_logit__",)),
    ("observed_pmi", ("observed_joint_pmi__",)),
    ("exact_copula", ("exact_copula_density_ratio18_unsupported__",)),
    ("unsupported_pair_response", ("unsupported_pair_response__",)),
)


def _robust_scale(values):
    median = np.nanmedian(values, axis=0)
    scale = np.nanquantile(values, 0.75, axis=0) - np.nanquantile(values, 0.25, axis=0)
    median[~np.isfinite(median)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return median, scale


def _scaled(frame, names, median, scale):
    values = frame[list(names)].to_numpy(np.float64, copy=True)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.broadcast_to(median, values.shape)[missing]
    return np.clip((values - median) / scale, -8.0, 8.0)


class PeriodicEmbedding:
    def __init__(self, names=RAW_NAMES, width=2):
        self.names, self.width = tuple(names), width

    def fit_transform(self, frame):
        self.median, self.scale = _robust_scale(frame[list(self.names)].to_numpy(np.float64, copy=True))
        rng = np.random.default_rng(250314)
        self.frequency = rng.normal(0.0, 0.94, size=(len(self.names), self.width))
        self.phase = rng.uniform(-np.pi, np.pi, size=(len(self.names), self.width))
        return self.transform(frame)

    def transform(self, frame):
        scaled = _scaled(frame, self.names, self.median, self.scale)
        angle = np.float64(2.0 * np.pi) * scaled[:, :, None] * self.frequency[None, :, :] + self.phase[None, :, :]
        parts, names = [], []
        for feature_index, name in enumerate(self.names):
            for frequency_index in range(self.width):
                parts += [np.sin(angle[:, feature_index, frequency_index]),
                          np.cos(angle[:, feature_index, frequency_index])]
                names += [f"periodic__{name}__f{frequency_index:02d}__{wave}" for wave in ("sin", "cos")]
        return pd.DataFrame(np.column_stack(parts).astype(np.float32), columns=names, dtype=np.float32)


class RawRBFBasis:
    def __init__(self, width=16):
        self.width = width
        self.output_names = [f"rbf__{name}__rbf_{index:02d}" for name in RAW_NAMES for index in range(width)]

    def fit_transform(self, frame):
        values = frame[list(RAW_NAMES)].to_numpy(np.float64, copy=True)
        self.median, self.scale = _robust_scale(values)
        finite = np.where(np.isfinite(values), values, np.broadcast_to(self.median, values.shape))
        scaled = np.clip((finite - self.median) / self.scale, -8.0, 8.0)
        levels = (np.arange(self.width, dtype=np.float64) + 0.5) / self.width
        self.centers = np.quantile(scaled, levels, axis=0).T
        spacing = np.diff(self.centers, axis=1)
        spacing[spacing <= 1e-10] = np.nan
        with np.errstate(all="ignore"):
            local = np.nanmedian(spacing, axis=1)
        local[~np.isfinite(local)] = 0.10
        self.bandwidth = np.maximum(0.10, 1.25 * local)
        return self.transform(frame)

    def transform(self, frame):
        scaled = _scaled(frame, RAW_NAMES, self.median, self.scale)
        standardized = (scaled[:, :, None] - self.centers[None, :, :]) / self.bandwidth[None, :, None]
        rbf = np.exp(-0.5 * np.square(standardized)).reshape(len(frame), -1)
        return pd.DataFrame(rbf.astype(np.float32), columns=self.output_names, dtype=np.float32)


def semantic_constraints(columns, core_count=42):
    """Interaction-constraint cliques over the 290 frontier columns; returns (groups, core names)."""
    core = list(columns[:core_count])
    assigned = set(core)
    groups = []
    for _, prefixes in FAMILY_PREFIXES:
        members = [name for name in columns[core_count:] if name.startswith(prefixes)]
        assigned.update(members)
        groups.append(core + members)
    leftovers = [name for name in columns if name not in assigned]
    if leftovers:
        groups.append(core + leftovers)
    return groups, core
