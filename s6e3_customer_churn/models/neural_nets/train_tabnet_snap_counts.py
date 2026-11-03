#!/usr/bin/env python
"""TabNet on numeric transforms, crosses, train + test counts, original-data churn rates and snapped charges.

Needs ../snap_features (build.py) first: MonthlyCharges_snap / TotalCharges_snap (the charge replaced by the closest
original-data value when the original data does not contain it) and their *_snap_diff columns are read from
snap_features/cache/combo.csv.
Features (TotalCharges blank -> NaN; the original rows are a reference only, not training rows):
- the 19 columns and the 4 snap columns; |snap - value| of both charges, floor(MonthlyCharges_snap / 10) and
  floor(TotalCharges_snap / 50);
- tenure: zero flag, log1p, sqrt, 1 / (1 + x), floor(/ 6), floor(/ 12); MonthlyCharges: log1p, sqrt, floor(/ 10),
  rounded to 0 and 1 decimals; TotalCharges: log1p, sqrt, floor(/ 50), rounded to 0 and 1 decimals;
- charge relations: TotalCharges - tenure x MonthlyCharges and its absolute value, MonthlyCharges / tenure,
  TotalCharges / tenure, TotalCharges / MonthlyCharges (0 when undefined), tenure x MonthlyCharges, tenure +
  MonthlyCharges, MonthlyCharges + TotalCharges;
- Yes count over 8 service columns, Yes and No counts over 11 yes / no columns;
- 10 crosses: Contract x InternetService, Contract x PaymentMethod, InternetService x PaymentMethod, PaperlessBilling x
  PaymentMethod, OnlineSecurity x TechSupport, StreamingTV x StreamingMovies, Partner x Dependents, gender x
  SeniorCitizen, floor(tenure / 12) x Contract, floor(MonthlyCharges / 10) x InternetService;
- original-data row count and churn rate of Contract, InternetService, PaymentMethod, PaperlessBilling, Partner,
  Dependents, SeniorCitizen;
- count over train + test and a rare flag (count <= 25) of every string column, cross and floor bin;
- a missing flag for every column with missing values.
String columns and every column whose name contains '__' (the 8 named crosses and their counts) are ordinal-encoded on
train + test (+1) and embedded by TabNet (dim ceil(log2(cardinality)) clipped to 3-16); the other columns are numeric
(missing -> train + test median).
Model: pytorch-tabnet TabNetClassifier, n_d = n_a = 32, 5 steps, gamma 1.5, 2 independent + 2 shared GLU layers,
lambda_sparse 1e-4, momentum 0.02, gradient clip 2, entmax masks; Adam lr 2e-2, weight decay 1e-5, ReduceLROnPlateau on
the held-out AUC (factor 0.7, patience 8, min lr 1e-5); batch 8192, virtual batch 1024; up to 300 epochs with early
stopping on the held-out AUC (patience 30, best weights restored); TabNet seed 42 (python / numpy / torch seeded with
42 at the start). Test = mean of the 5 fold models.
Usage: python train_tabnet_snap_counts.py [--folds 0,1,2,3,4]"""
import argparse
import copy
import gc
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, cache_dir, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'tabnet_snap_counts_orig_priors'
MODEL_DIR = model_dir('neural_nets')
COMBO_PATH = cache_dir('snap_features') / 'combo.csv'
RANDOM_STATE = 42

# TabNet
N_D = 32
N_A = 32
N_STEPS = 5
GAMMA = 1.5
N_INDEPENDENT = 2
N_SHARED = 2
LAMBDA_SPARSE = 1e-4
MOMENTUM = 0.02
CLIP_VALUE = 2.0
CAT_EMB_DIM_DEFAULT = 3
CAT_EMB_DIM_MAX = 16

# training
MAX_EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 8192
VIRTUAL_BATCH_SIZE = 1024
LR = 2e-2
WEIGHT_DECAY = 1e-5

RARE_THRESH = 25


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_binary_y(df, target_col):
    y = df[target_col]
    if y.dtype == 'object' or str(y.dtype).startswith('string'):
        y = y.map({'No': 0, 'Yes': 1})
    y = pd.to_numeric(y, errors='coerce')
    if y.isna().any():
        raise ValueError(f"Could not convert target '{target_col}' to binary")
    return y.astype(np.int32).to_numpy()


def safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a / np.where(np.abs(b) < 1e-12, np.nan, b)


def clean_totalcharges(df):
    df['TotalCharges'] = (df['TotalCharges'].astype(str).str.strip()
                          .replace({'': np.nan, 'nan': np.nan, 'None': np.nan}))
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df


def add_missing_flags(df):
    new_cols = {}
    for c in df.columns:
        if df[c].isna().any():
            new_cols[f'{c}_was_missing'] = df[c].isna().astype('int8')
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def add_numeric_features(df):
    feats = {}
    tenure = pd.to_numeric(df['tenure'], errors='coerce').to_numpy(dtype=np.float64)
    feats['tenure_zero'] = (np.nan_to_num(tenure) == 0).astype(np.int8)
    feats['tenure_log1p'] = np.log1p(np.clip(np.nan_to_num(tenure), 0, None))
    feats['tenure_sqrt'] = np.sqrt(np.clip(np.nan_to_num(tenure), 0, None))
    feats['tenure_inv1p'] = 1.0 / (1.0 + np.clip(np.nan_to_num(tenure), 0, None))
    feats['tenure_bin6'] = np.floor(np.nan_to_num(tenure) / 6.0).astype(np.int16)
    feats['tenure_bin12'] = np.floor(np.nan_to_num(tenure) / 12.0).astype(np.int16)

    mc = pd.to_numeric(df['MonthlyCharges'], errors='coerce').to_numpy(dtype=np.float64)
    feats['MonthlyCharges_log1p'] = np.log1p(np.clip(np.nan_to_num(mc), 0, None))
    feats['MonthlyCharges_sqrt'] = np.sqrt(np.clip(np.nan_to_num(mc), 0, None))
    feats['MonthlyCharges_bin10'] = np.floor(np.nan_to_num(mc) / 10.0).astype(np.int16)
    feats['MonthlyCharges_round0'] = np.round(np.nan_to_num(mc), 0).astype(np.float32)
    feats['MonthlyCharges_round1'] = np.round(np.nan_to_num(mc), 1).astype(np.float32)

    tc = pd.to_numeric(df['TotalCharges'], errors='coerce').to_numpy(dtype=np.float64)
    feats['TotalCharges_log1p'] = np.log1p(np.clip(np.nan_to_num(tc), 0, None))
    feats['TotalCharges_sqrt'] = np.sqrt(np.clip(np.nan_to_num(tc), 0, None))
    feats['TotalCharges_bin50'] = np.floor(np.nan_to_num(tc) / 50.0).astype(np.int32)
    feats['TotalCharges_round0'] = np.round(np.nan_to_num(tc), 0).astype(np.float32)
    feats['TotalCharges_round1'] = np.round(np.nan_to_num(tc), 1).astype(np.float32)
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def add_charge_relationship_features(df):
    tenure = pd.to_numeric(df['tenure'], errors='coerce').to_numpy(dtype=np.float64)
    mc = pd.to_numeric(df['MonthlyCharges'], errors='coerce').to_numpy(dtype=np.float64)
    tc = pd.to_numeric(df['TotalCharges'], errors='coerce').to_numpy(dtype=np.float64)
    est_tc = np.nan_to_num(tenure) * np.nan_to_num(mc)
    feats = {
        'tc_minus_tenure_mc': np.nan_to_num(tc - est_tc, nan=0.0),
        'abs_tc_minus_tenure_mc': np.nan_to_num(np.abs(tc - est_tc), nan=0.0),
        'mc_over_tenure': np.nan_to_num(safe_div(mc, tenure), nan=0.0, posinf=0.0, neginf=0.0),
        'tc_over_tenure': np.nan_to_num(safe_div(tc, tenure), nan=0.0, posinf=0.0, neginf=0.0),
        'tc_over_mc': np.nan_to_num(safe_div(tc, mc), nan=0.0, posinf=0.0, neginf=0.0),
        'tenure_times_mc': np.nan_to_num(tenure * mc, nan=0.0),
        'tenure_plus_mc': np.nan_to_num(tenure + mc, nan=0.0),
        'mc_plus_tc': np.nan_to_num(mc + tc, nan=0.0),
    }
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def add_service_features(df):
    feats = {}
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    yes_mat = np.column_stack([(df[c].astype(str).fillna('missing') == 'Yes').astype(np.int8).to_numpy()
                               for c in service_cols])
    feats['svc_yes_count'] = yes_mat.sum(axis=1).astype(np.int16)

    yn_cols = ['Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    yes_mat = np.column_stack([(df[c].astype(str).fillna('missing') == 'Yes').astype(np.int8).to_numpy()
                               for c in yn_cols])
    no_mat = np.column_stack([(df[c].astype(str).fillna('missing') == 'No').astype(np.int8).to_numpy()
                              for c in yn_cols])
    feats['yn_yes_count'] = yes_mat.sum(axis=1).astype(np.int16)
    feats['yn_no_count'] = no_mat.sum(axis=1).astype(np.int16)
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def add_cross_features(df):
    pairs = [
        ('Contract', 'InternetService'),
        ('Contract', 'PaymentMethod'),
        ('InternetService', 'PaymentMethod'),
        ('PaperlessBilling', 'PaymentMethod'),
        ('OnlineSecurity', 'TechSupport'),
        ('StreamingTV', 'StreamingMovies'),
        ('Partner', 'Dependents'),
        ('gender', 'SeniorCitizen'),
    ]
    feats = {}
    for a, b in pairs:
        feats[f'{a}__{b}'] = df[a].astype(str).fillna('missing') + '__' + df[b].astype(str).fillna('missing')
    feats['tenurebin_contract'] = (df['tenure_bin12'].astype(str).fillna('missing') + '__'
                                   + df['Contract'].astype(str).fillna('missing'))
    feats['mcbin_internet'] = (df['MonthlyCharges_bin10'].astype(str).fillna('missing') + '__'
                               + df['InternetService'].astype(str).fillna('missing'))
    return pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)


def add_freq_features(train_df, test_df):
    """Count over train + test and rare flag of every string column, cross and floor bin."""
    combo = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    cat_like_cols = []
    for c in combo.columns:
        if (combo[c].dtype == 'object' or str(combo[c].dtype).startswith('string') or '__' in c
                or c.endswith('bin6') or c.endswith('bin12') or c.endswith('bin10') or c.endswith('bin50')):
            cat_like_cols.append(c)
    cat_like_cols = sorted(set(cat_like_cols))
    new_cols = {}
    for c in cat_like_cols:
        s = combo[c].astype(str).fillna('missing')
        cnt = s.map(s.value_counts(dropna=False)).astype(np.int32)
        new_cols[f'{c}_cnt_all'] = cnt.to_numpy()
        new_cols[f'{c}_rare_all'] = (cnt.to_numpy() <= RARE_THRESH).astype(np.int8)
    if new_cols:
        combo = pd.concat([combo, pd.DataFrame(new_cols, index=combo.index)], axis=1)
    return combo.iloc[:len(train_df)].copy(), combo.iloc[len(train_df):].copy().reset_index(drop=True)


def add_orig_features(train_df, test_df, orig_df):
    """Original-data row count and churn rate of 7 columns."""
    y_orig = orig_df[TARGET]
    if y_orig.dtype == 'object' or str(y_orig.dtype).startswith('string'):
        y_orig = y_orig.map({'No': 0, 'Yes': 1})
    y_orig = pd.to_numeric(y_orig, errors='coerce')
    keys = ['Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling', 'Partner', 'Dependents',
            'SeniorCitizen']
    global_mean = float(y_orig.mean())
    tr_new, te_new = {}, {}
    for k in keys:
        s = orig_df[k].astype(str).fillna('missing')
        vc = s.value_counts(dropna=False)
        mean_map = pd.DataFrame({k: s, TARGET: y_orig}).groupby(k)[TARGET].mean()
        tr_key = train_df[k].astype(str).fillna('missing')
        te_key = test_df[k].astype(str).fillna('missing')
        tr_new[f'orig_cnt_{k}'] = tr_key.map(vc).fillna(0).astype(np.float32).to_numpy()
        te_new[f'orig_cnt_{k}'] = te_key.map(vc).fillna(0).astype(np.float32).to_numpy()
        tr_new[f'orig_mean_{k}'] = tr_key.map(mean_map).fillna(global_mean).astype(np.float32).to_numpy()
        te_new[f'orig_mean_{k}'] = te_key.map(mean_map).fillna(global_mean).astype(np.float32).to_numpy()
    train_df = pd.concat([train_df, pd.DataFrame(tr_new, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(te_new, index=test_df.index)], axis=1)
    return train_df, test_df


def add_snap_from_combo(train_df, test_df):
    if not COMBO_PATH.exists():
        raise FileNotFoundError(f'{COMBO_PATH} not found: run ../snap_features/build.py first')
    combo = pd.read_csv(COMBO_PATH)
    wanted = ['MonthlyCharges_snap', 'TotalCharges_snap', 'MonthlyCharges_snap_diff', 'TotalCharges_snap_diff']
    assert len(combo) == len(train_df) + len(test_df), 'snap feature rows do not match train + test'
    tr = combo.iloc[:len(train_df)][wanted].reset_index(drop=True)
    te = combo.iloc[len(train_df):][wanted].reset_index(drop=True)
    train_df = pd.concat([train_df.reset_index(drop=True), tr], axis=1)
    test_df = pd.concat([test_df.reset_index(drop=True), te], axis=1)

    for df in (train_df, test_df):
        df['mc_snap_gap_abs'] = np.abs(
            pd.to_numeric(df['MonthlyCharges_snap'], errors='coerce').fillna(0).to_numpy()
            - pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0).to_numpy())
        df['MonthlyCharges_snap_bin10'] = np.floor(
            pd.to_numeric(df['MonthlyCharges_snap'], errors='coerce').fillna(0).to_numpy() / 10.0).astype(np.int16)
        df['tc_snap_gap_abs'] = np.abs(
            pd.to_numeric(df['TotalCharges_snap'], errors='coerce').fillna(0).to_numpy()
            - pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0).to_numpy())
        df['TotalCharges_snap_bin50'] = np.floor(
            pd.to_numeric(df['TotalCharges_snap'], errors='coerce').fillna(0).to_numpy() / 50.0).astype(np.int32)
    return train_df, test_df


def build_features(train_df, test_df, orig_df):
    train_df = clean_totalcharges(train_df.copy())
    test_df = clean_totalcharges(test_df.copy())
    orig_df = clean_totalcharges(orig_df.copy())
    for c in ['MonthlyCharges', 'SeniorCitizen']:
        train_df[c] = pd.to_numeric(train_df[c], errors='coerce')
        test_df[c] = pd.to_numeric(test_df[c], errors='coerce')
        orig_df[c] = pd.to_numeric(orig_df[c], errors='coerce')

    train_df, test_df = add_snap_from_combo(train_df, test_df)
    train_df = add_numeric_features(train_df)
    test_df = add_numeric_features(test_df)
    train_df = add_charge_relationship_features(train_df)
    test_df = add_charge_relationship_features(test_df)
    train_df = add_service_features(train_df)
    test_df = add_service_features(test_df)
    train_df = add_cross_features(train_df)
    test_df = add_cross_features(test_df)
    train_df, test_df = add_orig_features(train_df, test_df, orig_df)
    train_df, test_df = add_freq_features(train_df, test_df)
    train_df = add_missing_flags(train_df)
    test_df = add_missing_flags(test_df)
    return train_df, test_df


def prepare_tabnet_matrices(train_df, test_df):
    """Numeric columns first, then the ordinal-coded categoricals (TabNet embeds those)."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    cat_cols = []
    for c in train_df.columns:
        if c == ID:
            continue
        if train_df[c].dtype == 'object' or str(train_df[c].dtype).startswith('string') or '__' in c:
            if c in test_df.columns:
                cat_cols.append(c)
    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in train_df.columns if c not in cat_cols + [ID] and c in test_df.columns]

    for c in cat_cols:
        train_df[c] = train_df[c].astype('string').fillna('missing')
        test_df[c] = test_df[c].astype('string').fillna('missing')

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1,
                         dtype=np.float32)
    enc.fit(pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0, ignore_index=True))
    train_cat = enc.transform(train_df[cat_cols]).astype(np.float32) + 1.0
    test_cat = enc.transform(test_df[cat_cols]).astype(np.float32) + 1.0
    cat_dims = [int(max(train_cat[:, j].max(), test_cat[:, j].max())) + 1 for j in range(train_cat.shape[1])]
    cat_idxs = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    cat_emb_dim = [min(CAT_EMB_DIM_MAX, max(CAT_EMB_DIM_DEFAULT, int(np.ceil(np.log2(max(d, 2)))))) for d in cat_dims]

    X_num_train = train_df[num_cols].copy()
    X_num_test = test_df[num_cols].copy()
    for c in num_cols:
        X_num_train[c] = pd.to_numeric(X_num_train[c], errors='coerce')
        X_num_test[c] = pd.to_numeric(X_num_test[c], errors='coerce')
    med = pd.concat([X_num_train, X_num_test], axis=0, ignore_index=True).median(numeric_only=True)
    X_num_train = X_num_train.fillna(med).to_numpy(dtype=np.float32)
    X_num_test = X_num_test.fillna(med).to_numpy(dtype=np.float32)

    X_train = np.hstack([X_num_train, train_cat]).astype(np.float32)
    X_test = np.hstack([X_num_test, test_cat]).astype(np.float32)
    print(f'{X_train.shape[1]} inputs: {len(num_cols)} numeric, {len(cat_cols)} categorical', flush=True)
    return X_train, X_test, cat_idxs, cat_dims, cat_emb_dim


def make_model(cat_idxs, cat_dims, cat_emb_dim):
    return TabNetClassifier(
        n_d=N_D, n_a=N_A, n_steps=N_STEPS, gamma=GAMMA, n_independent=N_INDEPENDENT, n_shared=N_SHARED,
        lambda_sparse=LAMBDA_SPARSE, momentum=MOMENTUM, clip_value=CLIP_VALUE,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=LR, weight_decay=WEIGHT_DECAY),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params=dict(mode='max', factor=0.7, patience=8, min_lr=1e-5),
        mask_type='entmax', verbose=0, seed=RANDOM_STATE, device_name='auto',
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(RANDOM_STATE)
    train = read_train()
    test = read_test()
    orig = read_original()
    y = to_binary_y(train, TARGET)
    print(f'train {train.shape} | test {test.shape} | original {orig.shape}', flush=True)

    train_fe, test_fe = build_features(train.drop(columns=[TARGET]), test.copy(), orig.copy())
    X_train, X_test, cat_idxs, cat_dims, cat_emb_dim = prepare_tabnet_matrices(train_fe, test_fe)

    fold_ids = folds(y)
    for k in fold_list(args.folds):
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        X_tr, y_tr = X_train[tr_idx], y[tr_idx]
        X_va, y_va = X_train[va_idx], y[va_idx]
        clf = make_model(copy.deepcopy(cat_idxs), copy.deepcopy(cat_dims), copy.deepcopy(cat_emb_dim))
        clf.fit(X_train=X_tr, y_train=y_tr, eval_set=[(X_va, y_va)], eval_name=['valid'], eval_metric=['auc'],
                max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                virtual_batch_size=VIRTUAL_BATCH_SIZE, num_workers=0, drop_last=False)
        va_pred = clf.predict_proba(X_va)[:, 1]
        te_pred = clf.predict_proba(X_test)[:, 1]
        print(f'fold {k + 1}/5 | AUC={roc_auc_score(y_va, va_pred):.6f} | best_epoch={getattr(clf, "best_epoch", None)}',
              flush=True)
        save_part(MODEL_DIR, PAIR, k, va_pred, te_pred)
        del X_tr, y_tr, X_va, y_va, clf, va_pred, te_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
