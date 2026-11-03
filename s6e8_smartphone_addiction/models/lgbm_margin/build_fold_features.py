#!/usr/bin/env python
"""Build the per-fold feature cache of the forensic LightGBM (layout: lib/fold_cache.py).

For outer fold k:
  strong        the 102-column forensic matrix of ../xgb_forensic with its base margin and probability, as float32
                (../catboost_forensic/lib/strong_cat.build_strong_application); the nested target encodings use the
                CPU exact-key encoder of lib/target_encoder.py instead of cuML's
  fixed_source  the fixed-original univariate binary LLR and log-count channels (../forensic_features)
  cat_numeric   all numeric columns of the source-fragment cache (../catboost_forensic/build_source_fragments.py)
  categories    native source-fragment categories (raw_cat, mask, exact_value, semantic_state, the PMI_PAIRS18 pairs of
                recurrence_c2 and source_pair_row, source_nearest_value, source_residual_tick, fragment_summary) and the
                semantic interaction categories (source_qbins, source_q16_pairs, imputed_screen, source_rule), recoded
                over the outer-training values, plus their log-frequencies and 6 per-row summaries
Usage: python build_fold_features.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.pairs import fold_list  # noqa: E402
from models.catboost_forensic.lib.strong_cat import (build_strong_application, cat_semantic_categories,  # noqa: E402
                                                     fixed_source_features, fragment_manifest, fragment_root,
                                                     native_categories, source_test_features)
from models.lgbm_margin.lib.fold_cache import (SUMMARY_NAMES, category_frequency, category_summaries,  # noqa: E402
                                               contiguous_categories, fold_root)
from models.lgbm_margin.lib.target_encoder import CPUExactTargetEncoder  # noqa: E402

NATIVE_GROUPS = ["raw_cat", "mask", "exact_value", "semantic_state", "recurrence_pmi18_c2", "source_pair_pmi18",
                 "source_nearest_value", "source_residual_tick", "fragment_summary"]
SEMANTIC_GROUPS = ["source_qbins", "source_q16_pairs", "imputed_screen", "source_rule"]
FIXED_SOURCE_GROUPS = ["univariate_llr_logcount"]
CATEGORY_GROUP_PREFIXES = {
    "native_base": ("native__raw_cat__", "native__mask__", "native__exact_value__", "native__semantic_state__"),
    "native_pair": ("native__recurrence_c2__", "native__source_pair_row__"),
    "native_source_value": ("native__source_nearest_value__", "native__source_residual_tick__"),
    "native_fragment": ("native__fragment_summary__",),
    "semantic": ("semantic__",),
}


def build_fold(fold):
    (strong_train, strong_valid, strong_test, target, train_idx, valid_idx,
     _, _, _) = build_strong_application(fold, encoder_class=CPUExactTargetEncoder)
    test_idx = len(target) + np.arange(len(strong_test), dtype=np.int64)

    native_frames = [native_categories(rows, NATIVE_GROUPS)[0] for rows in (train_idx, valid_idx, test_idx)]
    category_names = native_frames[0].columns.tolist()
    semantic_train, semantic_apply, semantic_names = cat_semantic_categories(
        strong_train, pd.concat([strong_valid, strong_test], ignore_index=True), SEMANTIC_GROUPS)
    semantic_frames = [semantic_train, semantic_apply.iloc[:len(valid_idx)].reset_index(drop=True),
                       semantic_apply.iloc[len(valid_idx):].reset_index(drop=True)]
    category_names += semantic_names

    fixed_train = fixed_source_features(train_idx, FIXED_SOURCE_GROUPS)[0]
    fixed_valid = fixed_source_features(valid_idx, FIXED_SOURCE_GROUPS)[0]
    fixed_names = fixed_train.columns.tolist()
    fixed_test = source_test_features(fixed_names, [])[0]

    cache_numeric = np.load(fragment_root() / "numeric.npy", mmap_mode="r")
    cat_numeric = [np.asarray(cache_numeric[rows], dtype=np.float32) for rows in (train_idx, valid_idx, test_idx)]
    cat_numeric_names = [f"cat_numeric__{name}" for name in fragment_manifest()["numeric_names"]]

    strong_names = strong_train.columns.tolist()
    base_numeric_names = strong_names + fixed_names + cat_numeric_names
    numeric_frames, raw_category_frames = [], []
    for strong, fixed, native, semantic, cat_num in zip((strong_train, strong_valid, strong_test),
                                                         (fixed_train, fixed_valid, fixed_test), native_frames,
                                                         semantic_frames, cat_numeric):
        numeric_frames.append(np.column_stack([strong.to_numpy(np.float32), fixed.to_numpy(np.float32),
                                               cat_num]).astype(np.float32, copy=False))
        raw_category_frames.append(np.column_stack([native.to_numpy(np.int32),
                                                    semantic.to_numpy(np.int32)]).astype(np.int32, copy=False))

    cat_train, cat_valid, cat_test, cardinalities = contiguous_categories(*raw_category_frames)
    freq_train, freq_valid, freq_test = category_frequency(cat_train, cat_valid, cat_test, cardinalities)
    roles = {
        "train": (numeric_frames[0], cat_train, freq_train, category_summaries(cat_train, freq_train)),
        "valid": (numeric_frames[1], cat_valid, freq_valid, category_summaries(cat_valid, freq_valid)),
        "test": (numeric_frames[2], cat_test, freq_test, category_summaries(cat_test, freq_test)),
    }
    out = fold_root(fold)
    out.mkdir(parents=True, exist_ok=True)
    for role, arrays in roles.items():
        for kind, values in zip(("numeric", "categories", "category_frequency", "category_summary"), arrays):
            np.save(out / f"{role}_{kind}.npy", np.ascontiguousarray(values))
    np.save(out / "train_indices.npy", train_idx)
    np.save(out / "valid_indices.npy", valid_idx)

    n_strong, n_fixed = len(strong_names), len(fixed_names)
    numeric_groups = {
        "strong": list(range(0, n_strong)),
        "fixed_source": list(range(n_strong, n_strong + n_fixed)),
        "cat_numeric": list(range(n_strong + n_fixed, len(base_numeric_names))),
        "cat_numeric_source_alignment": [i for i, n in enumerate(base_numeric_names)
                                         if n.startswith("cat_numeric__source_alignment__")],
        "cat_numeric_fragment": [i for i, n in enumerate(base_numeric_names)
                                 if n.startswith("cat_numeric__fragment_summary__")],
        "cat_numeric_domain": [i for i, n in enumerate(base_numeric_names) if n.startswith("cat_numeric__domain__")],
    }
    category_groups = {group: [i for i, n in enumerate(category_names) if n.startswith(prefixes)]
                       for group, prefixes in CATEGORY_GROUP_PREFIXES.items()}
    manifest = {
        "fold": fold,
        "rows": {role: int(arrays[0].shape[0]) for role, arrays in roles.items()},
        "base_numeric_names": base_numeric_names,
        "numeric_groups": numeric_groups,
        "category_names": category_names,
        "category_groups": category_groups,
        "category_cardinalities": cardinalities,
        "category_frequency_names": [f"category_logcount__{name}" for name in category_names],
        "category_summary_names": SUMMARY_NAMES,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"fold {fold}: numeric {numeric_frames[0].shape[1]}, categories {cat_train.shape[1]}, rows "
          f"{manifest['rows']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    for fold in fold_list(args.folds):
        build_fold(fold)


if __name__ == "__main__":
    main()
