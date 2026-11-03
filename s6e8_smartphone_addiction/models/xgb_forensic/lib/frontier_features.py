"""The 290-column frontier matrix: the 102-column forensic matrix (lib/strong.py) with 188 extra columns inserted after
its screen-response block, the completion base margin and the XGBoost parameters of the frontier models.

Extra blocks, in model order (frontier_parts):
  residual24   24 fixed-original source channels of ../forensic_features (LLRs, pair LLRs / counts, CDF geometry)
  source134    caches of ../realmlp_repair: pair-lift bank (10), coarse lift bank (9), generator temperature (39),
               global-density surprisal / entropy (24) and conditional fills (27), derived-coordinate lift bank (5),
               repair posteriors (20)
  completion   the 10 global-density completion aggregates of ../realmlp_repair
  ares         14 source-tempered residual contrasts (build_residual_ares.py)
  dsx          6 daily x social residual surfaces of y - sigmoid(completion_logit_mean) (lib/residual_surfaces.py),
               owner groups splitmix64(id) mod 5
Base margin: logit(prior) + 0.55 raw_te_woe_sum + 0.40 (completion_logit_mean - logit(prior))."""
import numpy as np

from common.data import TARGET, cache_dir, read_test, read_train
from models.forensic_features.lib.blocks import source_channel

from .residual_ares import load_ares
from .residual_surfaces import build_daily_social_surfaces, splitmix_group

RESIDUAL24_NAMES = (
    "binary_llr__work_study_hours__round2", "binary_llr__gaming_hours__round2", "pair_llr__app_opens_per_day__gender",
    "pair_llr__app_opens_per_day__stress_level", "geometry__social_media_hours__cdf0",
    "geometry__social_media_hours__cdf_delta", "geometry__notifications_per_day__cdf_delta",
    "binary_llr__app_opens_per_day__q8", "binary_llr__sleep_hours__q32", "geometry__notifications_per_day__cdf0",
    "geometry__notifications_per_day__cdf1", "geometry__app_opens_per_day__cdf1", "binary_llr__app_opens_per_day__q16",
    "pair_llr__sleep_hours__stress_level", "latent_logrel__None__notifications_per_day__q16",
    "binary_llr__gaming_hours__q128", "binary_llr__sleep_hours__q128",
    "pair_logcount__sleep_hours__notifications_per_day",
    "pair_llr__sleep_hours__gender", "geometry__work_study_hours__logdistance_delta",
    "pair_llr__sleep_hours__academic_work_impact", "binary_llr__work_study_hours__q16",
    "pair_logcount__work_study_hours__sleep_hours", "pair_llr__work_study_hours__notifications_per_day",
)
COMPLETION_RESPONSE = "density_completion__aggregate__completion_logit_mean"
FRONTIER_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "device": "cuda",
    "enable_categorical": True, "max_bin": 1024, "n_estimators": 5000, "learning_rate": 0.02, "max_depth": 6,
    "grow_policy": "depthwise", "max_leaves": 0, "min_child_weight": 20.0, "subsample": 0.7, "colsample_bytree": 0.7,
    "colsample_bynode": 0.6, "reg_alpha": 0.0, "reg_lambda": 4.0, "gamma": 0.0, "scale_pos_weight": 1.0,
    "max_delta_step": 0.0, "sampling_method": "uniform", "max_cat_threshold": 64, "random_state": 42, "n_jobs": 8,
    "verbosity": 0,
}


def residual24(train_idx, valid_idx):
    train_matrix, names = source_channel("train")
    test_matrix, _ = source_channel("test")
    cols = [names.index(name) for name in RESIDUAL24_NAMES]
    return (list(RESIDUAL24_NAMES), np.asarray(train_matrix[train_idx][:, cols], np.float32),
            np.asarray(train_matrix[valid_idx][:, cols], np.float32), np.asarray(test_matrix[:, cols], np.float32))


def _repair_lib():
    from models.realmlp_repair.lib import completion, global_density, lift, repair, temperature
    return completion, global_density, lift, repair, temperature


def _fold_arrays(directory, prefix="", columns=None):
    out = []
    for role in ("train", "valid", "test"):
        matrix = np.load(directory / f"{prefix}{role}.npy", mmap_mode="r")
        out.append(np.asarray(matrix if columns is None else matrix[:, columns], np.float32))
    return out


def _global_arrays(directory, train_idx, valid_idx):
    train_matrix = np.load(directory / "train.npy", mmap_mode="r")
    return [np.asarray(train_matrix[train_idx], np.float32), np.asarray(train_matrix[valid_idx], np.float32),
            np.asarray(np.load(directory / "test.npy", mmap_mode="r"), np.float32)]


def source134(fold, train_idx, valid_idx):
    """The 134-column lift / temperature / global-density / repair block from the ../realmlp_repair caches."""
    completion, global_density, lift, repair, temperature = _repair_lib()
    root = cache_dir("realmlp_repair")
    blocks = []
    for bank in ("pairlift", "cbank"):
        blocks.append((lift.BANKS[bank].names, _fold_arrays(root / "lift" / f"fold_{fold}", prefix=f"{bank}_")))
    temperature_names = [temperature.feature_names()[i] for i in temperature.STRONG39]
    blocks.append((temperature_names,
                   _fold_arrays(root / "temperature" / f"fold_{fold}", columns=temperature.STRONG39)))
    blocks.append((global_density.GDEN_NAMES + global_density.MFILL_NAMES,
                   _global_arrays(root / "global_channels", train_idx, valid_idx)))
    blocks.append((lift.BANKS["wbank"].names, _fold_arrays(root / "lift" / f"fold_{fold}", prefix="wbank_")))
    blocks.append((list(repair.REPAIR_NAMES), _global_arrays(root / "repair", train_idx, valid_idx)))
    names = [name for block_names, _ in blocks for name in block_names]
    train, valid, test = (np.concatenate([parts[role] for _, parts in blocks], axis=1) for role in range(3))
    return names, train, valid, test


def completion10(fold):
    completion = _repair_lib()[0]
    names = [completion.feature_names()[i] for i in completion.AGGREGATE_INDICES]
    arrays = _fold_arrays(cache_dir("realmlp_repair") / "global_completion" / f"fold_{fold}",
                          columns=completion.AGGREGATE_INDICES)
    return names, arrays[0], arrays[1], arrays[2]


def frontier_parts(fold, train_idx, valid_idx):
    """extra_parts callable of build_strong_fold: the 188 extra columns (names, train, valid, test) per block."""
    train, test = read_train(), read_test()
    target = train[TARGET].to_numpy(np.int8)
    fit_ids = train["id"].to_numpy(np.int64)[train_idx]
    completion_block = completion10(fold)
    response = completion_block[1][:, completion_block[0].index(COMPLETION_RESPONSE)].astype(np.float64)
    residual = target[train_idx].astype(np.float64) - 1.0 / (1.0 + np.exp(-response))
    apply_rows = np.concatenate([valid_idx, np.arange(len(test)) + len(train)])
    daily = np.concatenate([train["daily_screen_time_hours"].to_numpy(), test["daily_screen_time_hours"].to_numpy()])
    social = np.concatenate([train["social_media_hours"].to_numpy(), test["social_media_hours"].to_numpy()])
    dsx_train, dsx_apply, dsx_names = build_daily_social_surfaces(
        daily[train_idx], social[train_idx], daily[apply_rows], social[apply_rows], residual, splitmix_group(fit_ids))
    n_valid = len(valid_idx)
    return [residual24(train_idx, valid_idx), source134(fold, train_idx, valid_idx), completion_block,
            load_ares(fold), (dsx_names, dsx_train, dsx_apply[:n_valid], dsx_apply[n_valid:])]


def completion_margin(frame, prior):
    prior_logit = float(np.log(prior / (1.0 - prior)))
    raw = frame["raw_te_woe_sum"].to_numpy(np.float64)
    completion = frame[COMPLETION_RESPONSE].to_numpy(np.float64)
    return prior_logit + 0.55 * raw + 0.40 * (completion - prior_logit)
