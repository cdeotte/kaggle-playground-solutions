"""Cross-fitted field reconstruction (target-free): 189 channels in three nested views (compact 71, detailed +52,
full +66); the RealMLP uses the compact view, other repo models use the wider views.

Two reconstruction mechanisms predict every one of the 12 predictors from the other 11:
  XGBoost experts   one GPU XGBoost per field (regressor for numerics, softprob classifier for categoricals) on a
                    conditional matrix built after the field is hidden (raw values, missing flags, one-hots, missing
                    counts, screen sums / slack / shares); 900 rounds, depth 8, lr 0.04, min_child_weight 20,
                    subsample / colsample 0.85, lambda 5, alpha 0.05, max_bin 512; at most 480,000 observed rows per
                    field; numeric predictions clipped to the 0.1% / 99.9% quantiles
  autoencoder       lib/autoencoder.py, 8 epochs, AdamW lr 8e-4 / wd 2e-4, batch 2048, 2 MC-dropout samples
Within an outer fold the outer-training rows are split into 3 groups by a splitmix64 hash of their id; each group is
reconstructed by experts / an autoencoder fit on the other two groups (with their own normalization / category state).
Outer-validation and test rows are reconstructed by a fit on all outer-training rows.

Compact channels (71): for missing numeric cells the hybrid fill (0.75 XGB + 0.25 AE for screen fields, 0.65 / 0.35
otherwise), |XGB - AE| / scale, AE total std / scale and the original-data class LLR of the fill (64 quantile bins of
the original, screen fields in the aligned geometry); for missing categoricals the XGB / AE disagreement, entropy and
expected original LLR of the 0.65 / 0.35 hybrid distribution; error / disagreement / uncertainty / LLR aggregates over
all, screen and non-screen fields; and the screen geometry of the completed row (parts, slack, shares, weekend - daily,
the original's screen rule probability).
Detailed (+52): the XGBoost and AE fills and their original LLRs for missing numeric cells, both mechanisms' class
probabilities for missing categoricals. Full (+66): standardized observed-cell residuals of both mechanisms, the
completed values of XGB / AE / hybrid, 5 summaries of the AE latent, the screen geometry of the XGB- and AE-completed
rows."""
import gc

import numpy as np
import pandas as pd

from . import autoencoder as mae

NUMERIC_FIELDS = ('age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                  'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time')
CATEGORICAL_FIELDS = ('gender', 'stress_level', 'academic_work_impact')
FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS
SCREEN_FIELDS = ('daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
                 'weekend_screen_time')
SETTINGS = dict(inner_parts=3, xgb_rounds=900, max_xgb_fit_rows=480000, ae_epochs=8, ae_batch_size=2048,
                ae_mc_samples=2)


def align_original_screen_geometry(source):
    """Map the original screen block into the generator's frame: the generator behaves as if its fourth allocation
    coordinate were weekend - daily; the four positive allocations are normalized and decoded against daily."""
    out = source.loc[:, list(FIELDS)].copy()
    daily = out['daily_screen_time_hours'].to_numpy(np.float64)
    social = out['social_media_hours'].to_numpy(np.float64)
    gaming = out['gaming_hours'].to_numpy(np.float64)
    work = out['work_study_hours'].to_numpy(np.float64)
    weekend = out['weekend_screen_time'].to_numpy(np.float64)
    fourth = np.maximum(weekend - daily, 1e-3)
    allocations = np.maximum(np.column_stack([social, gaming, work, fourth]), 1e-3)
    shares = allocations / allocations.sum(axis=1, keepdims=True)
    out['social_media_hours'] = shares[:, 0] * daily
    out['gaming_hours'] = shares[:, 1] * daily
    out['work_study_hours'] = shares[:, 2] * daily
    return out


def splitmix_group(ids, groups):
    values = np.asarray(ids, dtype=np.uint64).copy()
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return (values % np.uint64(groups)).astype(np.int8)


class PredictorSchema:
    def __init__(self, train, original=None):
        sources = [train] + ([original] if original is not None else [])
        self.category_levels = {f: tuple(sorted(set().union(*[set(s[f].dropna().astype(str).tolist()) for s in sources])))
                                for f in CATEGORICAL_FIELDS}
        self.numeric_medians, self.numeric_scales = {}, {}
        for f in NUMERIC_FIELDS:
            values = pd.to_numeric(train[f], errors='coerce').to_numpy(np.float64)
            finite = values[np.isfinite(values)]
            q25, q75 = np.quantile(finite, [0.25, 0.75])
            self.numeric_medians[f] = float(np.median(finite))
            self.numeric_scales[f] = float(max(q75 - q25, np.std(finite), 1e-3))


def _numeric(frame, field):
    return pd.to_numeric(frame[field], errors='coerce').to_numpy(np.float32)


def conditional_predictor_matrix(frame, schema, masked_fields=()):
    """XGBoost matrix after hiding masked_fields (derived geometry is computed after hiding)."""
    masked = set(masked_fields)
    n = len(frame)
    columns, numeric_values, missing_values = [], {}, {}
    for f in NUMERIC_FIELDS:
        values = _numeric(frame, f).copy()
        if f in masked:
            values[:] = np.nan
        numeric_values[f] = values
        missing = ~np.isfinite(values)
        missing_values[f] = missing
        columns += [values, missing.astype(np.float32)]
    for f in CATEGORICAL_FIELDS:
        raw = frame[f].astype('string')
        missing = raw.isna().to_numpy()
        if f in masked:
            missing = np.ones(n, dtype=bool)
            raw = pd.Series(pd.array([pd.NA] * n, dtype='string'))
        missing_values[f] = missing
        columns.append(missing.astype(np.float32))
        text = raw.fillna('__MISSING__').astype(str).to_numpy()
        columns += [(text == level).astype(np.float32) for level in schema.category_levels[f]]
    columns.append(np.column_stack([missing_values[f] for f in FIELDS]).sum(axis=1).astype(np.float32))
    columns.append(np.column_stack([missing_values[f] for f in SCREEN_FIELDS]).sum(axis=1).astype(np.float32))
    daily, social, gaming, work, weekend = (numeric_values[f] for f in (
        'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'weekend_screen_time'))
    parts = social + gaming + work
    slack = daily - parts
    safe_daily = np.where(daily > 1e-3, daily, np.nan)
    for values in (parts, slack, weekend - daily, social / safe_daily, gaming / safe_daily, work / safe_daily,
                   slack / safe_daily):
        values = np.asarray(values, dtype=np.float32)
        values[~np.isfinite(values)] = np.nan
        columns.append(values)
    return np.column_stack(columns).astype(np.float32, copy=False)


def encode_autoencoder(frame, schema):
    """-> normalized numerics (median / scale, 0 when missing), categorical codes (0 = missing), observed mask."""
    numeric = np.empty((len(frame), len(NUMERIC_FIELDS)), dtype=np.float32)
    categorical = np.empty((len(frame), len(CATEGORICAL_FIELDS)), dtype=np.int64)
    observed = np.empty((len(frame), len(FIELDS)), dtype=bool)
    for i, f in enumerate(NUMERIC_FIELDS):
        values = _numeric(frame, f)
        finite = np.isfinite(values)
        numeric[:, i] = np.where(finite, (values - schema.numeric_medians[f]) / schema.numeric_scales[f], 0.0)
        observed[:, i] = finite
    for k, f in enumerate(CATEGORICAL_FIELDS):
        levels = {v: j + 1 for j, v in enumerate(schema.category_levels[f])}
        codes = np.array([levels.get(str(v), 0) if not pd.isna(v) else 0 for v in frame[f].astype('string')],
                         dtype=np.int64)
        categorical[:, k] = codes
        observed[:, len(NUMERIC_FIELDS) + k] = codes != 0
    return numeric, categorical, observed


ARRAY_KEYS = ('numeric_mean', 'numeric_epistemic', 'numeric_aleatoric', 'latent_summary')


def empty_bundle(n_rows, schema):
    return {'numeric_mean': np.empty((n_rows, len(NUMERIC_FIELDS)), np.float32),
            'numeric_epistemic': np.zeros((n_rows, len(NUMERIC_FIELDS)), np.float32),
            'numeric_aleatoric': np.zeros((n_rows, len(NUMERIC_FIELDS)), np.float32),
            'latent_summary': np.zeros((n_rows, 5), np.float32),
            'categorical_probability': {f: np.empty((n_rows, len(schema.category_levels[f])), np.float32)
                                        for f in CATEGORICAL_FIELDS}}


def assign_bundle(destination, rows, source):
    for key in ARRAY_KEYS:
        destination[key][rows] = source[key]
    for f in CATEGORICAL_FIELDS:
        destination['categorical_probability'][f][rows] = source['categorical_probability'][f]


def slice_bundle(bundle, rows):
    out = {key: bundle[key][rows] for key in ARRAY_KEYS}
    out['categorical_probability'] = {f: v[rows] for f, v in bundle['categorical_probability'].items()}
    return out


def xgb_fit_predict(fit, apply, schema, rounds, max_fit_rows, seed, threads):
    from xgboost import XGBClassifier, XGBRegressor
    result = empty_bundle(len(apply), schema)  # XGBoost gives no uncertainty / latent: those stay 0
    for k, f in enumerate(FIELDS):
        observed_idx = np.flatnonzero(fit[f].notna().to_numpy())
        if len(observed_idx) > max_fit_rows:
            observed_idx = np.sort(np.random.default_rng(seed + k).choice(observed_idx, max_fit_rows, replace=False))
        fit_rows = fit.iloc[observed_idx].loc[:, list(FIELDS)]
        x_fit = conditional_predictor_matrix(fit_rows, schema, masked_fields=(f,))
        x_apply = conditional_predictor_matrix(apply.loc[:, list(FIELDS)], schema, masked_fields=(f,))
        common = dict(n_estimators=rounds, max_depth=8, learning_rate=0.04, min_child_weight=20, subsample=0.85,
                      colsample_bytree=0.85, reg_lambda=5.0, reg_alpha=0.05, max_bin=512, tree_method='hist',
                      device='cuda', n_jobs=threads, random_state=seed + k, verbosity=0)
        if f in NUMERIC_FIELDS:
            y_fit = pd.to_numeric(fit_rows[f], errors='raise').to_numpy(np.float32)
            model = XGBRegressor(objective='reg:squarederror', **common)
            model.fit(x_fit, y_fit, verbose=False)
            prediction = model.predict(x_apply).astype(np.float64)
            lower, upper = np.quantile(pd.to_numeric(fit_rows[f], errors='raise').to_numpy(np.float64), [0.001, 0.999])
            result['numeric_mean'][:, NUMERIC_FIELDS.index(f)] = np.clip(prediction, lower, upper).astype(np.float32)
        else:
            levels = schema.category_levels[f]
            y_fit = fit_rows[f].astype(str).map({v: j for j, v in enumerate(levels)}).to_numpy(np.int32)
            model = XGBClassifier(objective='multi:softprob', num_class=len(levels), eval_metric='mlogloss', **common)
            model.fit(x_fit, y_fit, verbose=False)
            probability = model.predict_proba(x_apply).astype(np.float64)
            probability /= probability.sum(axis=1, keepdims=True)
            result['categorical_probability'][f][:] = probability.astype(np.float32)
        del model, x_fit, x_apply
        gc.collect()
    return result


def train_autoencoder(fit, schema, epochs, batch_size, seed, device='cuda'):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    numeric, categorical, observed = encode_autoencoder(fit, schema)
    empirical_masks = ~observed
    cardinalities = [len(schema.category_levels[f]) for f in CATEGORICAL_FIELDS]
    model = mae.build_model(cardinalities, mae.AutoencoderConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(numeric), torch.from_numpy(categorical),
                                      torch.from_numpy(observed)),
                        batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed + 11))
    mask_generator = torch.Generator(device=device).manual_seed(seed + 29)
    empirical = torch.as_tensor(empirical_masks, device=device)
    n_numeric = len(NUMERIC_FIELDS)
    for epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch_numeric, batch_categorical, batch_observed in loader:
            batch_numeric = batch_numeric.to(device, non_blocking=True)
            batch_categorical = batch_categorical.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            artificial = mae.sample_artificial_mask(batch_observed, empirical, mask_generator, 0.08)
            input_observed = batch_observed & ~artificial
            input_numeric = batch_numeric.masked_fill(~input_observed[:, :n_numeric], 0.0)
            input_categorical = batch_categorical.masked_fill(~input_observed[:, n_numeric:], 0)
            means, log_scales, logits, _ = model(input_numeric, input_categorical, input_observed)
            loss = mae.reconstruction_loss(means, log_scales, logits, batch_numeric, batch_categorical, artificial)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch_numeric)
            count += len(batch_numeric)
        print(f'  autoencoder epoch {epoch + 1}: loss {total / count:.5f}', flush=True)
    return model


def autoencoder_predict(model, apply, schema, mc_samples, device='cuda'):
    numeric, categorical, observed = encode_autoencoder(apply, schema)
    bundle = empty_bundle(len(apply), schema)
    for k, f in enumerate(FIELDS):
        prediction = mae.predict_masked_field(model, numeric, categorical, observed, k, device, batch_size=8192,
                                              mc_samples=mc_samples)
        if f in NUMERIC_FIELDS:
            i = NUMERIC_FIELDS.index(f)
            bundle['numeric_mean'][:, i] = prediction['mean'] * schema.numeric_scales[f] + schema.numeric_medians[f]
            bundle['numeric_epistemic'][:, i] = prediction['epistemic_std'] * schema.numeric_scales[f]
            bundle['numeric_aleatoric'][:, i] = prediction['aleatoric_std'] * schema.numeric_scales[f]
        else:
            bundle['categorical_probability'][f][:] = prediction['probability']
    bundle['latent_summary'][:] = mae.predict_latent_summary(model, numeric, categorical, observed, device)
    return bundle


class SourceLikelihood:
    """Class log-likelihood ratios of the original data: 64 quantile bins per numeric (screen fields in the aligned
    geometry), per level for categoricals; add-2 smoothing."""

    def __init__(self, original, original_y):
        aligned = align_original_screen_geometry(original)
        y = np.asarray(original_y, dtype=np.int8)
        self.edges, self.llr, self.categorical = {}, {}, {}
        for f in NUMERIC_FIELDS:
            source = aligned if f in SCREEN_FIELDS else original
            values = pd.to_numeric(source[f], errors='raise').to_numpy(np.float64)
            edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 65)))
            codes = np.searchsorted(edges[1:-1], values, side='right')
            bins = len(edges) - 1
            count0 = np.bincount(codes[y == 0], minlength=bins).astype(np.float64)
            count1 = np.bincount(codes[y == 1], minlength=bins).astype(np.float64)
            p0 = (count0 + 2.0) / (count0.sum() + 2.0 * bins)
            p1 = (count1 + 2.0) / (count1.sum() + 2.0 * bins)
            self.edges[f], self.llr[f] = edges, np.log(p1 / p0)
        for f in CATEGORICAL_FIELDS:
            values = original[f].astype(str).to_numpy()
            levels = sorted(set(values))
            mapping = {}
            for level in levels:
                mask = values == level
                p0 = (float(np.sum(mask & (y == 0))) + 2.0) / (float(np.sum(y == 0)) + 2.0 * len(levels))
                p1 = (float(np.sum(mask & (y == 1))) + 2.0) / (float(np.sum(y == 1)) + 2.0 * len(levels))
                mapping[level] = float(np.log(p1 / p0))
            self.categorical[f] = mapping

    def numeric(self, field, values):
        codes = np.searchsorted(self.edges[field][1:-1], np.asarray(values, np.float64), side='right')
        return self.llr[field][np.clip(codes, 0, len(self.llr[field]) - 1)]


def _safe_mean(values, available):
    return np.where(available, values, 0.0).sum(axis=1) / np.maximum(available.sum(axis=1), 1)


def assemble_features(frame, schema, xgb_bundle, ae_bundle, source):
    """-> (rows x 189 matrix, names, view_counts); views are nested: compact = first 71, detailed = first 123."""
    n = len(frame)
    feats, names = [], []

    def add(name, values):
        values = np.asarray(values, dtype=np.float32)
        assert values.shape == (n,) and np.isfinite(values).all(), name
        names.append(name)
        feats.append(values)

    raw = np.column_stack([pd.to_numeric(frame[f], errors='coerce') for f in NUMERIC_FIELDS]).astype(np.float64)
    observed = np.isfinite(raw)
    xgb = np.asarray(xgb_bundle['numeric_mean'], np.float64)
    ae = np.asarray(ae_bundle['numeric_mean'], np.float64)
    ae_epistemic = np.asarray(ae_bundle['numeric_epistemic'], np.float64)
    ae_aleatoric = np.asarray(ae_bundle['numeric_aleatoric'], np.float64)
    # XGBoost dominates point recovery on the screen fields; the autoencoder adds a complementary view
    weights = np.array([0.75 if f in SCREEN_FIELDS else 0.65 for f in NUMERIC_FIELDS])
    hybrid = weights * xgb + (1.0 - weights) * ae
    xgb_complete = np.where(observed, raw, xgb)
    ae_complete = np.where(observed, raw, ae)
    hybrid_complete = np.where(observed, raw, hybrid)
    scales = np.array([schema.numeric_scales[f] for f in NUMERIC_FIELDS])
    missing = ~observed
    source_hybrid = np.column_stack([source.numeric(f, hybrid[:, i]) for i, f in enumerate(NUMERIC_FIELDS)])
    source_xgb = np.column_stack([source.numeric(f, xgb[:, i]) for i, f in enumerate(NUMERIC_FIELDS)])
    source_ae = np.column_stack([source.numeric(f, ae[:, i]) for i, f in enumerate(NUMERIC_FIELDS)])

    # compact: missing-cell evidence
    for i, f in enumerate(NUMERIC_FIELDS):
        m = missing[:, i]
        add(f'hybrid__if_missing__{f}', np.where(m, hybrid[:, i], 0.0))
        add(f'hybrid__disagreement_norm_if_missing__{f}', np.where(m, np.abs(xgb[:, i] - ae[:, i]) / scales[i], 0.0))
        total_std = np.sqrt(np.square(ae_epistemic[:, i]) + np.square(ae_aleatoric[:, i]))
        add(f'hybrid__ae_uncertainty_if_missing__{f}', np.where(m, total_std / scales[i], 0.0))
        add(f'hybrid__source_llr_if_missing__{f}', np.where(m, source_hybrid[:, i], 0.0))

    categorical_missing = {}
    cat_disagreement, cat_entropy, cat_source = [], [], []
    for f in CATEGORICAL_FIELDS:
        levels = schema.category_levels[f]
        px = np.asarray(xgb_bundle['categorical_probability'][f], np.float64)
        pa = np.asarray(ae_bundle['categorical_probability'][f], np.float64)
        ph = 0.65 * px + 0.35 * pa
        m = ~frame[f].notna().to_numpy()
        categorical_missing[f] = m
        disagreement = 0.5 * np.abs(px - pa).sum(axis=1)
        entropy = -np.sum(ph * np.log(np.clip(ph, 1e-8, 1.0)), axis=1)
        expected_llr = ph @ np.array([source.categorical[f].get(level, 0.0) for level in levels])
        cat_disagreement.append(np.where(m, disagreement, 0.0))
        cat_entropy.append(np.where(m, entropy, 0.0))
        cat_source.append(np.where(m, expected_llr, 0.0))
        add(f'hybrid__cat_disagreement_if_missing__{f}', cat_disagreement[-1])
        add(f'hybrid__cat_entropy_if_missing__{f}', cat_entropy[-1])
        add(f'hybrid__cat_source_llr_if_missing__{f}', cat_source[-1])

    screen_idx = [NUMERIC_FIELDS.index(f) for f in SCREEN_FIELDS]
    other_idx = [i for i in range(len(NUMERIC_FIELDS)) if i not in screen_idx]
    xgb_error, ae_error = (raw - xgb) / scales, (raw - ae) / scales
    disagreement = np.abs(xgb - ae) / scales
    uncertainty = np.sqrt(np.square(ae_epistemic) + np.square(ae_aleatoric)) / scales
    for label, idx in (('all', list(range(len(NUMERIC_FIELDS)))), ('screen', screen_idx), ('nonscreen', other_idx)):
        obs, mis = observed[:, idx], missing[:, idx]
        add(f'reconagg__xgb_abs_error_observed__{label}', _safe_mean(np.abs(xgb_error[:, idx]), obs))
        add(f'reconagg__ae_abs_error_observed__{label}', _safe_mean(np.abs(ae_error[:, idx]), obs))
        add(f'reconagg__disagreement_missing__{label}', _safe_mean(disagreement[:, idx], mis))
        add(f'reconagg__uncertainty_missing__{label}', _safe_mean(uncertainty[:, idx], mis))
        add(f'reconagg__source_llr_missing__{label}', np.where(mis, source_hybrid[:, idx], 0.0).sum(axis=1))
    add('reconagg__categorical_disagreement_missing', np.column_stack(cat_disagreement).mean(axis=1))
    add('reconagg__categorical_entropy_missing', np.column_stack(cat_entropy).mean(axis=1))
    add('reconagg__categorical_source_llr_missing', np.column_stack(cat_source).sum(axis=1))

    def completed_screen(prefix, complete):
        daily, social, gaming, work, weekend = (complete[:, NUMERIC_FIELDS.index(f)] for f in (
            'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours',
            'weekend_screen_time'))
        parts = social + gaming + work
        slack = daily - parts
        denom = np.maximum(daily, 1e-3)
        add(f'{prefix}__screen_parts_sum', parts)
        add(f'{prefix}__screen_slack', slack)
        add(f'{prefix}__screen_slack_share', slack / denom)
        add(f'{prefix}__screen_social_share', social / denom)
        add(f'{prefix}__screen_gaming_share', gaming / denom)
        add(f'{prefix}__screen_work_share', work / denom)
        add(f'{prefix}__weekend_minus_daily', weekend - daily)
        # the original data's labelling rule on screen time
        add(f'{prefix}__source_rule_probability',
            np.where((daily > 8.0) | (social > 4.0), 1.0, np.where((daily <= 6.0) & (social <= 4.0), 0.0, 0.4556097561)))

    completed_screen('hybridgeom', hybrid_complete)
    view_counts = {'compact': len(names)}

    # detailed: the fills of each mechanism for missing cells
    for i, f in enumerate(NUMERIC_FIELDS):
        m = missing[:, i]
        add(f'xgb__if_missing__{f}', np.where(m, xgb[:, i], 0.0))
        add(f'ae__if_missing__{f}', np.where(m, ae[:, i], 0.0))
        add(f'xgb__source_llr_if_missing__{f}', np.where(m, source_xgb[:, i], 0.0))
        add(f'ae__source_llr_if_missing__{f}', np.where(m, source_ae[:, i], 0.0))
    for f in CATEGORICAL_FIELDS:
        m = categorical_missing[f]
        for mechanism, bundle in (('xgb', xgb_bundle), ('ae', ae_bundle)):
            probability = np.asarray(bundle['categorical_probability'][f])
            for k, level in enumerate(schema.category_levels[f]):
                add(f'{mechanism}__cat_probability_if_missing__{f}__{level}', np.where(m, probability[:, k], 0.0))
    view_counts['detailed'] = len(names)

    # full: observed-cell residuals, completed rows and the autoencoder latent summaries
    for i, f in enumerate(NUMERIC_FIELDS):
        o = observed[:, i]
        add(f'xgbfull__residual_norm_observed__{f}', np.where(o, xgb_error[:, i], 0.0))
        add(f'aefull__residual_norm_observed__{f}', np.where(o, ae_error[:, i], 0.0))
        add(f'xgbfull__completed__{f}', xgb_complete[:, i])
        add(f'aefull__completed__{f}', ae_complete[:, i])
        add(f'hybridfull__completed__{f}', hybrid_complete[:, i])
    latent = np.asarray(ae_bundle['latent_summary'], dtype=np.float64)
    for k, label in enumerate(('norm', 'mean', 'std', 'abs_mean', 'abs_max')):
        add(f'aefull__latent__{label}', latent[:, k])
    completed_screen('xgbfullgeom', xgb_complete)
    completed_screen('aefullgeom', ae_complete)
    view_counts['full'] = len(names)
    return np.column_stack(feats).astype(np.float32), names, view_counts


def build_fold(fold, train, fold_train, fold_valid, test, original, original_y, threads=12):
    """train: train.csv (with id). -> ({'train', 'valid', 'test'}: rows x 189 matrices, names, view_counts)."""
    import torch
    s = SETTINGS
    outer_train = train.iloc[fold_train].loc[:, list(FIELDS)].reset_index(drop=True)
    outer_valid = train.iloc[fold_valid].loc[:, list(FIELDS)].reset_index(drop=True)
    source_raw = original.loc[:, list(FIELDS)].copy()
    source_aligned = align_original_screen_geometry(source_raw)
    schema = PredictorSchema(outer_train, source_aligned)
    groups = splitmix_group(train['id'].to_numpy(np.int64)[fold_train], s['inner_parts'])
    train_xgb, train_ae = empty_bundle(len(outer_train), schema), empty_bundle(len(outer_train), schema)

    for g in range(s['inner_parts']):
        fit = outer_train.loc[groups != g].reset_index(drop=True)
        apply_rows = np.flatnonzero(groups == g)
        apply = outer_train.iloc[apply_rows].reset_index(drop=True)
        inner_schema = PredictorSchema(fit, source_aligned)
        assign_bundle(train_xgb, apply_rows, xgb_fit_predict(fit, apply, inner_schema, s['xgb_rounds'],
                                                             s['max_xgb_fit_rows'], 12000 + 1000 * fold + 100 * g,
                                                             threads))
        seed = 22000 + 1000 * fold + 100 * g
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = train_autoencoder(fit, inner_schema, s['ae_epochs'], s['ae_batch_size'], seed)
        assign_bundle(train_ae, apply_rows, autoencoder_predict(model, apply, inner_schema, s['ae_mc_samples']))
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f'fold {fold}: inner part {g} reconstructed', flush=True)

    application = pd.concat([outer_valid, test.loc[:, list(FIELDS)]], ignore_index=True) if test is not None \
        else outer_valid
    full_xgb = xgb_fit_predict(outer_train, application, schema, s['xgb_rounds'], s['max_xgb_fit_rows'],
                               32000 + 1000 * fold, threads)
    torch.manual_seed(42000 + fold)
    torch.cuda.manual_seed_all(42000 + fold)
    model = train_autoencoder(outer_train, schema, s['ae_epochs'], s['ae_batch_size'], 42000 + fold)
    full_ae = autoencoder_predict(model, application, schema, s['ae_mc_samples'])
    del model
    torch.cuda.empty_cache()

    source = SourceLikelihood(source_raw, original_y)
    n_valid = len(outer_valid)
    out = {}
    out['train'], names, view_counts = assemble_features(outer_train, schema, train_xgb, train_ae, source)
    out['valid'] = assemble_features(outer_valid, schema, slice_bundle(full_xgb, slice(0, n_valid)),
                                     slice_bundle(full_ae, slice(0, n_valid)), source)[0]
    if test is not None:
        test_frame = test.loc[:, list(FIELDS)].reset_index(drop=True)
        out['test'] = assemble_features(test_frame, schema, slice_bundle(full_xgb, slice(n_valid, None)),
                                        slice_bundle(full_ae, slice(n_valid, None)), source)[0]
    return out, names, view_counts
