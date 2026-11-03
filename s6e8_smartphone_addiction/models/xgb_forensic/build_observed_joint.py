#!/usr/bin/env python
"""Label-free exact-pair recurrence over the combined train + test predictors.

For each of the 36 numeric pairs, over the train + test rows where both values are observed: log1p(count of the
exact value pair) and PMI = log(n_pair * n_complete / (n_left * n_right)); 0 for rows with a missing value.
Writes cache/observed_joint_cache_v1/{train_features.npy, test_features.npy} (train / test rows, 72 columns each:
logcount, pmi per pair) and manifest.json (columns, complete pair counts).
Usage: python build_observed_joint.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import read_test, read_train  # noqa: E402
from models.xgb_forensic.lib.external import xgb_cache  # noqa: E402
from models.xgb_forensic.lib.pair_evidence import build_observed_joint  # noqa: E402


def main():
    train_matrix, test_matrix, names, complete_counts = build_observed_joint(read_train(), read_test())
    out = xgb_cache() / "observed_joint_cache_v1"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_features.npy", train_matrix)
    np.save(out / "test_features.npy", test_matrix)
    manifest = {"columns": names, "shape": list(train_matrix.shape), "rows_test": len(test_matrix),
                "feature_audit": {"complete_pair_counts": complete_counts}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"observed joint: train {train_matrix.shape}, test {test_matrix.shape}", flush=True)


if __name__ == "__main__":
    main()
