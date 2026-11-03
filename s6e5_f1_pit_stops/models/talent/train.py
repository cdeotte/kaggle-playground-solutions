#!/usr/bin/env python
"""Nine TALENT deep tabular models and two local networks (SAINT-style, GrowNet-style) on one per-fold feature builder.

Per outer fold k (common.data.folds; fold seed 43 + k):
  - the outer-train rows are cut to `cap` rows by a stratified subsample (lib/fold_features.stratified_cap, fold seed)
    when `cap` is set;
  - lib/fold_features.FoldFeatureBuilder (categorical profile `low` or `rich`) is fitted on those rows, the original rows
    of the other 4 folds (common.data.orig_folds) and the test rows;
  - with `orig_rows`, those original rows (features from the same builder) are appended to the training rows.
TALENT models: TALENT's Method with the held-out fold as its validation set, quantile-normalized numerics and the
model's `cat_policy`, max_epoch epochs; the epoch with the best held-out AUC (auc_first_metric) predicts the held-out
fold and the test rows. SAINT-style / GrowNet-style networks: lib/saint_style.py, lib/grownet_style.py.
Predictions are probabilities clipped to [1e-7, 1 - 1e-7]; test = mean of the 5 fold models.

Usage: python train.py <pair> [--folds 0,1,2,3,4] [--force]
Folds can run in separate processes; the pair is written once all 5 folds are done. Checkpoints go to cache/<pair>/."""
import argparse
import gc
import os
import random
import sys
import warnings
from pathlib import Path
from types import MethodType, SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score  # noqa: E402
from TALENT.model.methods.base import check_softmax  # noqa: E402
from TALENT.model.utils import get_method  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, cache_dir, folds, orig_folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.fold_features import FoldFeatureBuilder, stratified_cap  # noqa: E402
from lib.grownet_style import fit_grownet_fold  # noqa: E402
from lib.saint_style import fit_saint_fold  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42

# model: TALENT model_type (or the local saint_style / grownet_style network); cat_profile: `low` (20 keys) or `rich`
# (34 keys); orig_rows: append the original rows of the other folds; cap: stratified cap on the outer-train rows;
# cat_policy: TALENT categorical encoding; config: TALENT config (model / training), or the local network's.
MODELS = {
    "modernnca_lowcard_ohe_cap160k": dict(
        model="modernNCA", cat_profile="low", orig_rows=False, cap=160000, cat_policy="tabr_ohe",
        batch_size=512, max_epoch=4,
        config={"model": {"dim": 96, "dropout": 0.08, "d_block": 384, "n_blocks": 1, "temperature": 0.75,
                          "num_embeddings": {"type": "PLREmbeddings", "n_frequencies": 32, "frequency_scale": 0.045,
                                             "d_embedding": 16, "lite": True},
                          "sample_rate": 0.05},
                "training": {"lr": 0.008, "weight_decay": 0.0002, "n_bins": 64},
                "general": {}}),
    "tabr_lowcard_ohe_cap220k": dict(
        model="tabr", cat_profile="low", orig_rows=False, cap=220000, cat_policy="tabr_ohe",
        batch_size=1024, max_epoch=6,
        config={"model": {"num_embeddings": {"type": "PLREmbeddings", "n_frequencies": 48, "frequency_scale": 0.045,
                                             "d_embedding": 18, "lite": True},
                          "d_main": 192, "context_dropout": 0.25, "d_multiplier": 2.0, "encoder_n_blocks": 0,
                          "predictor_n_blocks": 1, "mixer_normalization": "auto", "dropout0": 0.12, "dropout1": 0.0,
                          "normalization": "LayerNorm", "activation": "ReLU", "memory_efficient": True,
                          "candidate_encoding_batch_size": 65536},
                "training": {"lr": 0.00045, "weight_decay": 1e-06, "n_bins": 64},
                "general": {}}),
    "node_rich_binary_orig_rows": dict(
        model="node", cat_profile="rich", orig_rows=True, cap=None, cat_policy="binary",
        batch_size=4096, max_epoch=15,
        config={"model": {"num_layers": 2, "depth": 6, "layer_dim": 512, "tree_dim": 2, "choice_function": "sparsemax",
                          "bin_function": "sparsemoid"},
                "training": {"lr": 0.0012, "weight_decay": 1e-05, "n_bins": 64},
                "general": {}}),
    "excelformer_rich_binary_orig_rows": dict(
        model="excelformer", cat_profile="rich", orig_rows=True, cap=None, cat_policy="binary",
        batch_size=1024, max_epoch=12,
        config={"model": {"prenormalization": True, "kv_compression": None, "kv_compression_sharing": None,
                          "token_bias": True, "ffn_dropout": 0.05, "attention_dropout": 0.15, "residual_dropout": 0.02,
                          "n_layers": 3, "n_heads": 8, "d_token": 192, "init_scale": 0.01},
                "training": {"lr": 0.00018, "weight_decay": 1e-05, "mix_type": "none", "n_bins": 64},
                "general": {}}),
    "trompt_lowcard_indices_cap240k": dict(
        model="trompt", cat_profile="low", orig_rows=False, cap=240000, cat_policy="indices",
        batch_size=512, max_epoch=8,
        config={"model": {"P": 48, "d": 96, "n_cycles": 3},
                "training": {"lr": 0.00045, "weight_decay": 1e-05, "n_bins": 64},
                "general": {}}),
    "amformer_lowcard_indices_orig_rows": dict(
        model="amformer", cat_profile="low", orig_rows=True, cap=None, cat_policy="indices",
        batch_size=1024, max_epoch=10,
        config={"model": {"dim": 128, "depth": 2, "heads": 4, "attn_dropout": 0.12, "ff_dropout": 0.08,
                          "groups": [32, 32], "sum_num_per_group": [16, 8], "prod_num_per_group": [4, 4],
                          "cluster": True, "target_mode": "mix", "token_descent": False, "num_special_tokens": 2,
                          "use_prod": True, "use_cls_token": True},
                "training": {"lr": 0.00016, "weight_decay": 1e-05, "n_bins": 64},
                "general": {}}),
    "dcn2_rich_indices_orig_rows": dict(
        model="dcn2", cat_profile="rich", orig_rows=True, cap=None, cat_policy="indices",
        batch_size=4096, max_epoch=15,
        config={"model": {"d": 384, "n_hidden_layers": 2, "n_cross_layers": 4, "hidden_dropout": 0.08,
                          "cross_dropout": 0.08, "stacked": False, "d_embedding": 24},
                "training": {"lr": 0.0003, "weight_decay": 0.00015, "n_bins": 64},
                "general": {}}),
    "mlp_plr_lowcard_ohe_cap300k": dict(
        model="mlp_plr", cat_profile="low", orig_rows=False, cap=300000, cat_policy="tabr_ohe",
        batch_size=4096, max_epoch=18,
        config={"model": {"d_layers": [640, 320, 256], "dropout": 0.18,
                          "num_embeddings": {"type": "PLREmbeddings", "n_frequencies": 64, "frequency_scale": 0.045,
                                             "d_embedding": 24, "lite": True}},
                "training": {"lr": 0.0008, "weight_decay": 0.0002, "n_bins": 64},
                "general": {}}),
    "autoint_lowcard_indices_cap260k": dict(
        model="autoint", cat_profile="low", orig_rows=False, cap=260000, cat_policy="indices",
        batch_size=2048, max_epoch=12,
        config={"model": {"n_layers": 3, "d_token": 96, "n_heads": 8, "attention_dropout": 0.12,
                          "residual_dropout": 0.02, "activation": "relu", "prenormalization": False,
                          "initialization": "kaiming", "kv_compression": None, "kv_compression_sharing": None},
                "training": {"lr": 0.00018, "weight_decay": 5e-05, "n_bins": 64},
                "general": {}}),
    "saint_style_lowcard_orig_rows_cap300k": dict(
        model="saint_style", cat_profile="low", orig_rows=True, cap=300000,
        batch_size=2048, max_epoch=18,
        config={"model": {"d_model": 96, "num_tokens": 8, "depth": 3, "heads": 4, "dropout": 0.12, "mlp_hidden": 512,
                          "patience": 5},
                "training": {"lr": 0.0006, "weight_decay": 0.0001, "grad_clip": 5.0, "pos_weight_power": 0.5}}),
    "grownet_style_lowcard_orig_rows_cap300k": dict(
        model="grownet_style", cat_profile="low", orig_rows=True, cap=300000,
        batch_size=4096, max_epoch=24,
        config={"model": {"hidden_d": 256, "d_embedding": 16, "dropout": 0.05, "stage_lr": 1.0, "patience": 7},
                "training": {"lr": 0.0008, "weight_decay": 0.0001, "epochs_per_stage": 1, "grad_clip": 5.0}}),
}
LOCAL_FITS = {"saint_style": fit_saint_fold, "grownet_style": fit_grownet_fold}


def sigmoid_np(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def logits_to_prob(raw):
    """Probability of class 1 from TALENT's outputs (2 logits per row for a binary task)."""
    if torch.is_tensor(raw):
        arr = raw.detach().cpu().numpy()
    else:
        arr = np.asarray(raw)
    if arr.ndim == 1:
        return sigmoid_np(arr).astype("float64")
    if arr.ndim == 2 and arr.shape[1] == 1:
        return sigmoid_np(arr[:, 0]).astype("float64")
    probs = check_softmax(arr)
    return probs[:, 1].astype("float64")


def auc_first_metric(self, predictions, labels, y_info):
    """Replaces TALENT's Method.metric. TALENT saves the 'best-val' checkpoint at the epoch with the best first metric
    (accuracy for classification); with AUC first it keeps the epoch with the best held-out AUC."""
    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
    labels_np = labels_np.astype(int).reshape(-1)
    probs = logits_to_prob(predictions)
    auc = roc_auc_score(labels_np, probs)
    ll = log_loss(labels_np, np.column_stack([1.0 - probs, probs]), labels=[0, 1])
    acc = accuracy_score(labels_np, probs >= 0.5)
    return (auc, ll, acc), ("AUC", "LogLoss", "Accuracy")


def torch_load_weights_only_false():
    """GrowNet run: torch.load defaults to weights_only=False (PyTorch >= 2.6 defaults to True)."""
    torch_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return torch_load(*args, **kwargs)

    torch.load = load


def make_args(spec, save_path, seed):
    return SimpleNamespace(
        model_type=spec["model"],
        config=spec["config"],
        seed=seed,
        save_path=str(save_path),
        max_epoch=spec["max_epoch"],
        batch_size=spec["batch_size"],
        num_nan_policy="mean",
        cat_nan_policy="new",
        num_policy="none",
        cat_policy=spec["cat_policy"],
        normalization="quantile",
        use_float=True,
    )


def fit_talent_fold(spec, Xtr_num, Xtr_cat, y_tr, Xva_num, Xva_cat, y_va, Xte_num, Xte_cat, fold_dir, seed):
    """TALENT fit with the held-out fold as validation set; the best-AUC checkpoint predicts held-out and test rows."""
    info = {"task_type": "binclass", "n_num_features": int(Xtr_num.shape[1]), "n_cat_features": int(Xtr_cat.shape[1]),
            "n_classes": 2}
    data_train = (
        {"train": Xtr_num, "val": Xva_num},
        {"train": Xtr_cat, "val": Xva_cat},
        {"train": y_tr.astype("int64"), "val": y_va.astype("int64")},
    )
    method = get_method(spec["model"])(make_args(spec, fold_dir, seed), is_regression=False)
    method.metric = MethodType(auc_first_metric, method)
    method.fit(data_train, info, train=True, config=spec["config"])

    val_data = ({"test": Xva_num}, {"test": Xva_cat}, {"test": y_va.astype("int64")})
    _, _, _, val_logits = method.predict(val_data, info, "best-val")
    val_prob = np.clip(logits_to_prob(val_logits), 1e-7, 1 - 1e-7)

    test_data = ({"test": Xte_num}, {"test": Xte_cat}, {"test": np.zeros(len(Xte_num), dtype="int64")})
    _, _, _, test_logits = method.predict(test_data, info, "best-val")
    test_prob = np.clip(logits_to_prob(test_logits), 1e-7, 1 - 1e-7)
    return val_prob, test_prob


def run_fold(pair, k, train, test, orig, y, fold_ids, orig_fold_ids):
    spec = MODELS[pair]
    fold_seed = SEED + k + 1
    fold_dir = cache_dir("talent") / pair / f"fold{k}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tr_idx, va_idx = np.flatnonzero(fold_ids != k), np.flatnonzero(fold_ids == k)
    X_tr_df = train.iloc[tr_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    X_va_df = train.iloc[va_idx].reset_index(drop=True)
    y_va = y[va_idx]
    orig_tr_df = orig.iloc[np.flatnonzero(orig_fold_ids != k)].reset_index(drop=True)

    X_tr_fit_df, y_tr_fit = stratified_cap(X_tr_df, y_tr, spec["cap"], fold_seed)
    builder = FoldFeatureBuilder(cat_profile=spec["cat_profile"], use_original_priors=True)
    builder.fit(X_tr_fit_df, orig_tr_df, test)

    Xtr_num, Xtr_cat = builder.transform(X_tr_fit_df.drop(columns=[TARGET], errors="ignore"))
    if spec["orig_rows"]:
        Xorig_num, Xorig_cat = builder.transform(orig_tr_df.drop(columns=[TARGET], errors="ignore"))
        Xtr_num = np.vstack([Xtr_num, Xorig_num]).astype("float32")
        Xtr_cat = np.vstack([Xtr_cat, Xorig_cat])
        y_tr_fit = np.concatenate([y_tr_fit, orig_tr_df[TARGET].astype(int).values])

    Xva_num, Xva_cat = builder.transform(X_va_df.drop(columns=[TARGET], errors="ignore"))
    Xte_num, Xte_cat = builder.transform(test)
    print(f"{pair} fold {k}: {len(y_tr_fit)} training rows, {len(builder.num_cols)} numeric features, "
          f"{len(builder.cat_cols)} categorical keys", flush=True)

    if spec["model"] in LOCAL_FITS:
        val_prob, test_prob = LOCAL_FITS[spec["model"]](
            spec["config"], spec["batch_size"], spec["max_epoch"],
            Xtr_num, Xtr_cat, y_tr_fit.astype("float32"),
            Xva_num, Xva_cat, y_va.astype("float32"),
            Xte_num, Xte_cat, fold_dir, fold_seed,
        )
    else:
        val_prob, test_prob = fit_talent_fold(spec, Xtr_num, Xtr_cat, y_tr_fit, Xva_num, Xva_cat, y_va,
                                              Xte_num, Xte_cat, fold_dir, fold_seed)
    save_part(HERE, pair, k, val_prob, test_prob)
    print(f"{pair} fold {k}: AUC {roc_auc_score(y_va, val_prob):.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(MODELS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    if MODELS[args.pair]["model"] == "grownet_style":
        torch_load_weights_only_false()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].astype(int).values
    fold_ids, orig_fold_ids = folds(y), orig_folds(orig[TARGET].astype(int).values)
    for k in fold_list(args.folds):
        if have_part(HERE, args.pair, k) and not args.force:
            print(f"{args.pair} fold {k}: done", flush=True)
            continue
        run_fold(args.pair, k, train, test, orig, y, fold_ids, orig_fold_ids)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(HERE, args.pair, fold_ids, y)


if __name__ == "__main__":
    main()
