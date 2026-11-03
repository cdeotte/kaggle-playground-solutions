#!/usr/bin/env python
"""Nearest-neighbour geometry and imputations against the original dataset (lib/source_knn.py), for every train and
test row. The reference set is only the 7,500 original rows (their labels for the geometry block, their screen values
for the imputations); the competition labels are not read. Runs on the default GPU (cupy).
Writes cache/source_knn_v1/{train.npy, test.npy, manifest.json} (17 columns) and
cache/source_knn_impute_v1/{train.npy, test.npy, manifest.json} (15 columns).
Usage: python build_source_knn.py [--chunk-rows 6000]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import TARGET, read_original, read_test, read_train  # noqa: E402
from models.xgb_forensic.lib.external import xgb_cache  # noqa: E402
from models.xgb_forensic.lib.source_knn import (IMPUTE_FEATURE_NAMES, IMPUTED, KNN_FEATURE_NAMES, encode,  # noqa: E402
                                                knn_features, knn_imputations)


def save(name, train_output, test_output, names):
    out = xgb_cache() / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train.npy", train_output)
    np.save(out / "test.npy", test_output)
    (out / "manifest.json").write_text(json.dumps({"feature_names": names, "rows_train": len(train_output),
                                                   "rows_test": len(test_output)}, indent=1) + "\n")
    print(f"{name}: train {train_output.shape}, test {test_output.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-rows", type=int, default=6000)
    args = ap.parse_args()
    source, train, test = read_original(), read_train(), read_test()
    source_matrix, _, _ = encode(source, source)
    source_target = source[TARGET].to_numpy(np.int8)
    save("source_knn_v1", knn_features(train, source, source_matrix, source_target, args.chunk_rows),
         knn_features(test, source, source_matrix, source_target, args.chunk_rows), KNN_FEATURE_NAMES)
    source_values = source[IMPUTED].to_numpy(np.float32)
    save("source_knn_impute_v1", knn_imputations(train, source, source_matrix, source_values, args.chunk_rows),
         knn_imputations(test, source, source_matrix, source_values, args.chunk_rows), IMPUTE_FEATURE_NAMES)


if __name__ == "__main__":
    main()
