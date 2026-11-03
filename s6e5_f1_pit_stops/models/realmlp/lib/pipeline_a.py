"""Pipeline A features and the RealMLP fold loop (pytabkit RealMLP_TD_Classifier).

Pipeline A (pipeline_a_fe) is learned on a set of fitting rows and applied unchanged to every other frame:
- two ratios, _LP_RP = LapNumber / (RaceProgress + 1e-6) and _TL_LP = TyreLife / max(LapNumber, 1);
- every numeric column and the two ratios floored and coded as a category (`<col>_cat_`; codes in order of first
  appearance in the fitting rows, -1 for values the fitting rows do not have);
- counts in the fitting rows of Driver, Compound, Race and of the Year and PitStop codes;
- 200 quantile bins of RaceProgress;
- codes of the Race x Compound and Race x Year combinations (`Race_Compound_`, `Race_Year_`).
That is 37 columns. use_global_fe=True fits it once on all train rows; use_global_fe=False fits it on the training rows of
each fold (synthetic rows only) and applies it to the fold's validation rows, the test rows and the original rows.

Folds: StratifiedKFold(n_splits, shuffle=True, random_state=seed) of the train rows and, separately, of the original rows
(the defaults 5 / 42 give the folds of common.data). Training rows of fold k = the train rows outside fold k plus, with
original_mode='fold', the original rows outside original fold k. With target_encode_all_combos, the two combination
codes get an sklearn TargetEncoder (smooth 'auto', inner cv = n_splits, shuffle, seed + k) fitted on the training rows.
The fold's validation rows are passed to fit as the validation set (val metric 1-AUC; pytabkit keeps the best epoch).
Model seed = seed + k + model_seed_offset (+ each ensemble_seed_offset; their models are averaged). k is 1-based.
OOF = the held-out predictions; test = mean over the fold models; both clipped to [1e-7, 1 - 1e-7]."""
import gc
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

from common.data import ID, TARGET, read_original, read_test, read_train

warnings.filterwarnings("ignore")
EPS = 1e-6


@dataclass
class ExpConfig:
    seed: int = 42
    n_splits: int = 5
    use_global_fe: bool = True
    original_mode: str = "fold"  # 'fold': append the original rows outside the fold; 'none': train rows only
    target_encode_all_combos: bool = True
    model_seed_offset: int = 0
    ensemble_seed_offsets: List[int] = field(default_factory=list)
    params_override: Dict = field(default_factory=dict)


BASE_PARAMS = {
    "random_state": 42,
    "verbosity": 2,
    "val_metric_name": "1-auc_ovr",
    "n_ens": 8,
    "n_epochs": 4,
    "batch_size": 256,
    "use_early_stopping": False,
    "lr": 0.03,
    "wd": 0.018,
    "sq_mom": 0.98,
    "lr_sched": "lin_cos_log_15",
    "first_layer_lr_factor": 0.25,
    "embedding_size": 6,
    "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128],
    "act": "silu",
    "p_drop": 0.05,
    "p_drop_sched": "expm4t",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_act_name": "gelu",
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "ls_eps": 0.01,
    "ls_eps_sched": "sqrt_cos",
    "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def realmlp_params(cfg, fold: int, ensemble_seed_offset: int = 0, base_params: Optional[Dict] = None) -> Dict:
    params = dict(BASE_PARAMS if base_params is None else base_params)
    params.update(cfg.params_override)
    params["random_state"] = int(cfg.seed + fold + cfg.model_seed_offset + ensemble_seed_offset)
    return params


def pipeline_a_fe(df: pd.DataFrame, fit: bool = False, state: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict, List[str]]:
    df = df.copy()
    if state is None:
        state = {}

    cat_cols_base = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols_base = df.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()
    num_cols_base = [c for c in num_cols_base if c not in [ID, TARGET]]

    df["_LP_RP"] = (pd.to_numeric(df["LapNumber"], errors="coerce") / (pd.to_numeric(df["RaceProgress"], errors="coerce") + EPS)).astype("float32")
    df["_TL_LP"] = (pd.to_numeric(df["TyreLife"], errors="coerce") / pd.to_numeric(df["LapNumber"], errors="coerce").clip(lower=1)).astype("float32")

    for col in num_cols_base + ["_LP_RP", "_TL_LP"]:
        cat_name = f"{col}_cat_" if col in num_cols_base else f"{col[1:]}_cat_"
        x = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if fit:
            _, uniques = np.floor(x).factorize()
            state[col] = uniques
        code_map = {cat: i for i, cat in enumerate(state[col])}
        codes = np.floor(x).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes.astype(str)

    for col in cat_cols_base + ["Year_cat_", "PitStop_cat_"]:
        cnt_name = f"_{col}_count" if col in cat_cols_base else f"_{col[:-1]}_count"
        if fit:
            state[cnt_name] = df[col].astype(str).value_counts(dropna=False)
        df[cnt_name] = df[col].astype(str).map(state[cnt_name]).fillna(0).astype("int32")

    for col, nb in [("RaceProgress", 200)]:
        bn = f"{col}_{nb}_qbin_"
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
        if fit:
            kb = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy="quantile", subsample=None)
            kb.fit(x)
            state[bn] = kb
        df[bn] = state[bn].transform(x).ravel().astype("int32").astype(str)

    combo_names = []
    for cols in [("Race", "Compound"), ("Race", "Year")]:
        cn = "_".join(cols) + "_"
        combo_names.append(cn)
        s = df[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "_" + df[c].astype(str)
        if fit:
            _, uniques = pd.factorize(s, sort=False)
            state[cn] = uniques
        code_map = {cat: i for i, cat in enumerate(state[cn])}
        codes = s.map(code_map).fillna(-1).astype("int32")
        df[cn] = codes.astype(str)

    return df, state, combo_names


def build_features_global(train_raw: pd.DataFrame, test_raw: pd.DataFrame, orig_raw: pd.DataFrame):
    X = train_raw.drop(columns=[ID, TARGET]).copy()
    y = train_raw[TARGET].astype(int).reset_index(drop=True)
    Xt = test_raw.drop(columns=[ID]).copy()
    orig = orig_raw.drop(columns=[TARGET]).copy()
    y_orig = orig_raw[TARGET].astype(int).reset_index(drop=True)
    X, state, combo_names = pipeline_a_fe(X, fit=True)
    Xt, _, _ = pipeline_a_fe(Xt, state=state)
    orig, _, _ = pipeline_a_fe(orig, state=state)
    return X.reset_index(drop=True), y, Xt.reset_index(drop=True), orig.reset_index(drop=True), y_orig, combo_names


def build_features_fold_local(cfg, X_tr_base, X_val_base, X_test_base, X_orig_base):
    """Pipeline A fitted on the fold's training rows. Returns the four frames and the columns to target encode."""
    X_tr, state, combo_names = pipeline_a_fe(X_tr_base, fit=True)
    X_val, _, _ = pipeline_a_fe(X_val_base, state=state)
    X_test, _, _ = pipeline_a_fe(X_test_base, state=state)
    X_orig, _, _ = pipeline_a_fe(X_orig_base, state=state)
    return X_tr, X_val, X_test, X_orig, combo_names


def add_target_encoding(X_tr, y_tr, X_val, X_test, te_cols, cfg, fold: int):
    if not te_cols:
        return X_tr, X_val, X_test, []
    te = TargetEncoder(cv=cfg.n_splits, smooth="auto", shuffle=True, random_state=cfg.seed + fold)
    te_names = [f"TE_{c}" for c in te_cols]
    X_tr, X_val, X_test = X_tr.copy(), X_val.copy(), X_test.copy()
    X_tr[te_names] = te.fit_transform(X_tr[te_cols], y_tr)
    X_val[te_names] = te.transform(X_val[te_cols])
    X_test[te_names] = te.transform(X_test[te_cols])
    return X_tr, X_val, X_test, te_names


def stratified_splits(y, n_splits: int, seed: int):
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def run_experiment(cfg, base_params: Optional[Dict] = None, fold_features=build_features_fold_local):
    """Trains every fold and returns (oof, test) probabilities (float32, train.csv / test.csv order).
    fold_features(cfg, X_tr, X_val, X_test, X_orig) builds the fold-local features (use_global_fe=False);
    base_params replaces BASE_PARAMS (pipeline_a3.py passes its own)."""
    seed_everything(cfg.seed)
    train_raw, test_raw, orig_raw = read_train(), read_test(), read_original()
    y_comp = train_raw[TARGET].astype(int).reset_index(drop=True)
    y_orig_raw = orig_raw[TARGET].astype(int).reset_index(drop=True)
    train_folds = stratified_splits(y_comp, cfg.n_splits, cfg.seed)
    orig_folds = stratified_splits(y_orig_raw, cfg.n_splits, cfg.seed)

    if cfg.use_global_fe:
        X_all, y_all, X_test_all, X_orig_all, y_orig, global_te_cols = build_features_global(train_raw, test_raw, orig_raw)
    else:
        X_all = train_raw.drop(columns=[ID, TARGET]).reset_index(drop=True)
        y_all = y_comp
        X_test_all = test_raw.drop(columns=[ID]).reset_index(drop=True)
        X_orig_all = orig_raw.drop(columns=[TARGET]).reset_index(drop=True)
        y_orig = y_orig_raw

    oof_pred = np.full(len(train_raw), np.nan, dtype=np.float32)
    test_pred = np.zeros(len(test_raw), dtype=np.float32)
    offsets = list(cfg.ensemble_seed_offsets) if cfg.ensemble_seed_offsets else [0]

    for fold, ((tr_idx, val_idx), (orig_tr_idx, _)) in enumerate(zip(train_folds, orig_folds), start=1):
        seed_everything(cfg.seed + fold)
        if cfg.use_global_fe:
            X_tr_synth = X_all.iloc[tr_idx].copy()
            X_val = X_all.iloc[val_idx].copy()
            X_test = X_test_all.copy()
            X_orig_tr = X_orig_all.iloc[orig_tr_idx].copy()
            te_cols = list(global_te_cols)
        else:
            X_tr_synth, X_val, X_test, X_orig_tr, te_cols = fold_features(
                cfg, X_all.iloc[tr_idx].copy(), X_all.iloc[val_idx].copy(), X_test_all, X_orig_all.iloc[orig_tr_idx].copy())

        y_tr_synth = y_all.iloc[tr_idx].reset_index(drop=True)
        y_val = y_all.iloc[val_idx].reset_index(drop=True)
        y_orig_tr = y_orig.iloc[orig_tr_idx].reset_index(drop=True)
        if cfg.original_mode == "none":
            X_tr = X_tr_synth.reset_index(drop=True)
            y_tr = y_tr_synth
        else:
            X_tr = pd.concat([X_tr_synth, X_orig_tr], axis=0).reset_index(drop=True)
            y_tr = pd.concat([y_tr_synth, y_orig_tr], axis=0).reset_index(drop=True)

        if cfg.target_encode_all_combos:
            X_tr, X_val, X_test, _ = add_target_encoding(X_tr, y_tr, X_val, X_test, te_cols, cfg, fold)
        if fold == 1:
            print(f"{X_tr.shape[1]} features, target encoded: {te_cols if cfg.target_encode_all_combos else []}", flush=True)

        val_pred = np.zeros(len(X_val), dtype=np.float32)
        fold_test = np.zeros(len(X_test), dtype=np.float32)
        for offset in offsets:
            model = RealMLP_TD_Classifier(**realmlp_params(cfg, fold, offset, base_params))
            model.fit(X_tr, y_tr, X_val, y_val)
            val_pred += model.predict_proba(X_val)[:, 1].astype(np.float32) / len(offsets)
            fold_test += model.predict_proba(X_test)[:, 1].astype(np.float32) / len(offsets)
            del model
            torch.cuda.empty_cache()
            gc.collect()

        oof_pred[val_idx] = val_pred
        test_pred += fold_test / len(train_folds)
        print(f"fold {fold}/{cfg.n_splits}: train {X_tr.shape}, AUC {roc_auc_score(y_val, val_pred):.6f}", flush=True)
        del X_tr, X_val, X_test, X_tr_synth, X_orig_tr, y_tr, y_val, val_pred, fold_test
        torch.cuda.empty_cache()
        gc.collect()

    print(f"OOF AUC {roc_auc_score(y_all, oof_pred):.6f}", flush=True)
    return np.clip(oof_pred, 1e-7, 1 - 1e-7).astype(np.float32), np.clip(test_pred, 1e-7, 1 - 1e-7).astype(np.float32)
