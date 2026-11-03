"""The 42-column parent of the forensic XGBoost: raw columns, screen-budget features, combined frequencies, fractional
cents, nested target encodings with weight-of-evidence sums, and two partial-budget residuals.

prepare_data() -> 29 label-free columns (train and test):
  12 raw columns (categoricals as pandas categories, levels in order of first appearance in train + test)
  parts_sum (skip-NaN sum of social + gaming + work), social / gaming / work share of daily, weekend - daily,
  complete_screen_budget_remaining (daily - strict component sum)
  daily / weekend fractional cent: rint(100 x) mod 100
  <numeric>__frequency: count of the exact value over train + test (NaN is its own value)
add_repeated_inner_target_encoding() -> + 9 nested TEs of the exact numeric values (smooth 1) and 2 WOE sums:
  inside the outer fold, 3 inner StratifiedKFold(5) views with seeds 42 / 1042 / 2042 + (outer fold + 1) are encoded
  with cuML's TargetEncoder (split_method='customize') and averaged; validation / test rows use the full outer-training
  map. raw_te_woe_sum = sum over the 9 TEs of logit(TE) - logit(prior); screen_te_woe_sum = the same over the 5 screen
  columns.
add_partial_budget_features() -> + daily_minus_known_parts (daily - parts_sum) and scaled_complete_budget_remaining
  (rows with a missing component: daily - parts_sum scaled up by the median component / daily shares of complete rows
  of train + test)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.data import TARGET, read_test, read_train

SEED = 42
MISSING_SENTINEL = -1_000_000.0
RAW_NUMERIC = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
               "sleep_hours", "notifications_per_day", "app_opens_per_day", "weekend_screen_time"]
V5_FEATURES = ["parts_sum", "social_media_share", "gaming_share", "work_study_share", "weekend_minus_daily",
               "complete_screen_budget_remaining"]
DECIMAL_FEATURES = ["daily_screen_time_hours__fractional_cent", "weekend_screen_time__fractional_cent"]
FREQUENCY_FEATURES = [f"{column}__frequency" for column in RAW_NUMERIC]
TE_COLUMNS = [f"{column}__te" for column in RAW_NUMERIC]
WOE_COLUMNS = ["raw_te_woe_sum", "screen_te_woe_sum"]
SCREEN_INDICES = [1, 2, 3, 4, 8]
PRIMARY_SMOOTH = 1.0
TE_INTERNAL_FOLDS = 5
INNER_SEED_BASES = (SEED, 1042, 2042, 3042)
COMPONENT_COLUMNS = ["social_media_hours", "gaming_hours", "work_study_hours"]
PARTIAL_BUDGET_COLUMNS = ["daily_minus_known_parts", "scaled_complete_budget_remaining"]


def add_parent_features(frame):
    result = frame.copy()
    result["parts_sum"] = result[COMPONENT_COLUMNS].sum(axis=1)
    daily = result["daily_screen_time_hours"].replace(0, np.nan)
    result["social_media_share"] = result["social_media_hours"] / daily
    result["gaming_share"] = result["gaming_hours"] / daily
    result["work_study_share"] = result["work_study_hours"] / daily
    result["weekend_minus_daily"] = result["weekend_screen_time"] - result["daily_screen_time_hours"]
    strict_sum = result["social_media_hours"] + result["gaming_hours"] + result["work_study_hours"]
    result["complete_screen_budget_remaining"] = result["daily_screen_time_hours"] - strict_sum
    for column in ("daily_screen_time_hours", "weekend_screen_time"):
        result[f"{column}__fractional_cent"] = np.rint(result[column] * 100.0) % 100
    return result


def _clean_key(series):
    return series.astype(np.float64).fillna(MISSING_SENTINEL)


def append_combined_frequencies(train_frame, test_frame):
    train_frame, test_frame = train_frame.copy(), test_frame.copy()
    for column in RAW_NUMERIC:
        combined = pd.concat([_clean_key(train_frame[column]), _clean_key(test_frame[column])], ignore_index=True)
        counts = combined.value_counts(dropna=False)
        train_frame[f"{column}__frequency"] = _clean_key(train_frame[column]).map(counts).astype(np.float32)
        test_frame[f"{column}__frequency"] = _clean_key(test_frame[column]).map(counts).astype(np.float32)
    return train_frame, test_frame


def raw_features(test):
    return [column for column in test.columns if column not in ("id", TARGET)]


def pre_te_columns(raw):
    return list(raw) + V5_FEATURES + DECIMAL_FEATURES + FREQUENCY_FEATURES


def parent_columns(raw):
    return pre_te_columns(raw) + TE_COLUMNS + WOE_COLUMNS


def prepare_data():
    """train.csv, test.csv, the 29 pre-TE columns of train and test (canonical order) and the target."""
    train, test = read_train(), read_test()
    raw = raw_features(test)
    X = add_parent_features(train[raw])
    X_test = add_parent_features(test[raw])
    y = train[TARGET].astype(np.int8).copy()
    for column in X.select_dtypes(include=["object", "category"]).columns:
        levels = pd.Index(pd.concat([X[column], X_test[column]], ignore_index=True).dropna().unique())
        dtype = pd.CategoricalDtype(categories=levels)
        X[column] = X[column].astype(dtype)
        X_test[column] = X_test[column].astype(dtype)
    X, X_test = append_combined_frequencies(X, X_test)
    keep = pre_te_columns(raw)
    return train, test, X[keep], X_test[keep], y


def make_inner_fold_ids(target, seed):
    y = np.asarray(target).reshape(-1)
    fold_ids = np.full(len(y), -1, dtype=np.int32)
    splitter = StratifiedKFold(n_splits=TE_INTERNAL_FOLDS, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(np.zeros(len(y), dtype=np.int8), y)):
        fold_ids[valid_idx] = fold
    return fold_ids


def encode_view(train_keys, valid_keys, target, fold_ids, smooth, encoder_seed, encoder_class):
    """Encode the 9 exact numeric keys at one smoothing strength with shared inner fold ids."""
    train_encoded = np.empty((len(train_keys), len(RAW_NUMERIC)), dtype=np.float32)
    valid_encoded = np.empty((len(valid_keys), len(RAW_NUMERIC)), dtype=np.float32)
    for column_index, column in enumerate(RAW_NUMERIC):
        encoder = encoder_class(n_folds=TE_INTERNAL_FOLDS, smooth=smooth, seed=encoder_seed,
                                split_method="customize", output_type="numpy")
        train_encoded[:, column_index] = np.asarray(
            encoder.fit_transform(train_keys[column], target, fold_ids=fold_ids)).reshape(-1)
        valid_encoded[:, column_index] = np.asarray(encoder.transform(valid_keys[column])).reshape(-1)
        del encoder
    return train_encoded, valid_encoded


def cuml_target_encoder():
    from cuml.preprocessing import TargetEncoder
    return TargetEncoder


def add_repeated_inner_target_encoding(X_train, y_train, X_valid, *, outer_fold, encoder_class=None, repeats=3):
    """Average `repeats` inner-5-fold TE views (seeds base + outer_fold, outer_fold is 1-based) of the 9 exact numeric
    keys, then add the two WOE sums. Returns the 40-column parent for the training and application rows."""
    encoder_class = encoder_class or cuml_target_encoder()
    target = y_train.to_numpy(dtype=np.float32, copy=False)
    train_keys = X_train[RAW_NUMERIC].astype(np.float64).fillna(MISSING_SENTINEL)
    valid_keys = X_valid[RAW_NUMERIC].astype(np.float64).fillna(MISSING_SENTINEL)
    seeds = [seed + outer_fold for seed in INNER_SEED_BASES[:repeats]]
    train_views, valid_views = [], []
    for seed in seeds:
        fold_ids = make_inner_fold_ids(target, seed)
        train_encoded, valid_encoded = encode_view(train_keys, valid_keys, target, fold_ids, PRIMARY_SMOOTH, seed,
                                                   encoder_class)
        train_views.append(train_encoded)
        valid_views.append(valid_encoded)
    train_average = np.mean(np.stack(train_views).astype(np.float64), axis=0).astype(np.float32)
    valid_average = np.mean(np.stack(valid_views).astype(np.float64), axis=0).astype(np.float32)

    train_model, valid_model = X_train.copy(), X_valid.copy()
    train_model[TE_COLUMNS] = train_average
    valid_model[TE_COLUMNS] = valid_average
    epsilon = 1e-5
    prior = float(target.mean())
    prior_logit = np.log(prior / (1.0 - prior))
    for model, average in ((train_model, train_average), (valid_model, valid_average)):
        probability = np.clip(average, epsilon, 1.0 - epsilon)
        woe = np.log(probability / (1.0 - probability)) - prior_logit
        model[WOE_COLUMNS[0]] = woe.sum(axis=1).astype(np.float32)
        model[WOE_COLUMNS[1]] = woe[:, SCREEN_INDICES].sum(axis=1).astype(np.float32)
    return train_model, valid_model


def compute_component_share_priors(*frames):
    """Median component / daily shares over the complete (all 4 fields observed, daily != 0) unlabeled rows."""
    required = ["daily_screen_time_hours", *COMPONENT_COLUMNS]
    combined = pd.concat([frame[required] for frame in frames], axis=0, ignore_index=True)
    complete = combined[required].notna().all(axis=1) & (combined["daily_screen_time_hours"].abs() > 1e-12)
    ratios = combined.loc[complete, COMPONENT_COLUMNS].div(combined.loc[complete, "daily_screen_time_hours"], axis=0)
    return {column: float(ratios[column].median()) for column in COMPONENT_COLUMNS}


def add_partial_budget_features(train_model, valid_model, component_share_priors):
    prior = pd.Series(component_share_priors, dtype=np.float64)
    results = []
    for frame in (train_model, valid_model):
        result = frame.copy()
        result["daily_minus_known_parts"] = frame["daily_screen_time_hours"] - frame["parts_sum"]
        observed = frame[COMPONENT_COLUMNS].notna()
        observed_prior_mass = observed.mul(prior, axis=1).sum(axis=1)
        complete_components = observed.all(axis=1)
        has_observed_component = observed.any(axis=1)
        estimated_total_parts = pd.Series(np.nan, index=frame.index, dtype=np.float64)
        scaled_rows = (~complete_components) & has_observed_component
        estimated_total_parts.loc[scaled_rows] = (frame.loc[scaled_rows, "parts_sum"] * float(prior.sum())
                                                  / observed_prior_mass.loc[scaled_rows])
        no_parts_rows = (~complete_components) & (~has_observed_component)
        estimated_total_parts.loc[no_parts_rows] = frame.loc[no_parts_rows, "daily_screen_time_hours"] * float(prior.sum())
        result["scaled_complete_budget_remaining"] = (
            frame["daily_screen_time_hours"] - estimated_total_parts).where(~complete_components)
        results.append(result)
    return results[0], results[1]
