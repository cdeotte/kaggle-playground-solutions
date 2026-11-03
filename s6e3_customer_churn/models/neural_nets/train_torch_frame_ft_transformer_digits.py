#!/usr/bin/env python
"""FT-Transformer (PyTorch Frame) on decimal-string digits of the charges, service flags, charge arithmetic, bins,
integer / step categoricals, pair categoricals and original-data mean / size mappings, with early stopping and bf16
autocast.

Features (MonthlyCharges and TotalCharges read as strings; strings stripped):
- decimal-string parts of the raw charges: 1st and 2nd decimal digits and the number of decimals of MonthlyCharges and
  TotalCharges (categoricals); then TotalCharges blanks -> tenure x MonthlyCharges;
- service flags (yes / no / no internet / no phone) of 9 service columns and their 4 counts;
- charge arithmetic: tenure x MonthlyCharges, TotalCharges / (tenure + 1), TotalCharges - tenure x MonthlyCharges (raw,
  absolute, relative), MonthlyCharges / (tenure + 1), TotalCharges / (MonthlyCharges + 1), MonthlyCharges - average
  monthly charge, MonthlyCharges + TotalCharges, log1p / sqrt / square of the 3 numerics, SeniorCitizen x charges;
- fixed bins of tenure, MonthlyCharges, TotalCharges and the average monthly charge; integer / step categoricals
  (SeniorCitizen, tenure, tenure // 3, 6, 12, round(MonthlyCharges), MonthlyCharges // 5, 10, TotalCharges // 50, 100,
  250, round(average monthly), average monthly // 5);
- 12 pair categoricals (8 column pairs and 4 bin pairs) and 4 decimal-digit crosses;
- original-data mappings (the same features built on the original rows): churn mean (unseen -> original churn rate) and
  row count of 35 single categoricals and of the 12 pairs.
54 categoricals and 158 numericals. Per fold: numerical NaN -> the fold's training median, categoricals as strings
(NaN -> '__MISSING__'), numericals min-max scaled on the fold's training rows.
Model: PyTorch Frame FTTransformer (128 channels, 4 layers, 2 output logits); EmbeddingEncoder for the categoricals,
LinearPeriodicEncoder for the numericals; column stats from the fold's training rows.
Training: cross-entropy with label smoothing 0.01, AdamW lr 5e-4, weight decay 5e-5, OneCycleLR (12% warm-up, no momentum
cycling) over 18 epochs, batch 1024, gradient clipping 1, bf16 autocast; early stopping on the held-out AUC (patience 4,
min delta 1e-6) and the best epoch predicts (softmax probability of churn). Python / numpy / torch are seeded with 42
(deterministic cuDNN) before the folds.
Usage: python train_torch_frame_ft_transformer_digits.py"""
import copy
import gc
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from torch_frame import stype
from torch_frame.data import DataFrameToTensorFrameConverter
from torch_frame.data.loader import DataLoader
from torch_frame.data.stats import compute_col_stats
from torch_frame.nn import EmbeddingEncoder, FTTransformer, LinearPeriodicEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ID, ORIGINAL_FILE, TARGET, model_dir  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'torch_frame_ft_transformer_decimal_digits_original_maps'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
MISSING_TOKEN = '__MISSING__'
TEST_FOLDS = 15


class CFG:
    FOLDS = 5
    EPOCHS = 18
    PATIENCE = 4
    LR = 5e-4
    WD = 5e-5
    WARMUP = 0.12
    BATCH = 1024
    LABEL_SMOOTH = 0.01
    GRAD_CLIP = 1.0
    USE_AMP = True
    CHANNELS = 128
    LAYERS = 4
    NUM_ENCODER = LinearPeriodicEncoder()


SERVICE_COLS = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
BASE_CAT_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                 'Contract', 'PaperlessBilling', 'PaymentMethod']
PAIR_COLS = [
    ('Contract', 'InternetService'),
    ('Contract', 'PaymentMethod'),
    ('InternetService', 'OnlineSecurity'),
    ('InternetService', 'TechSupport'),
    ('OnlineSecurity', 'TechSupport'),
    ('PaperlessBilling', 'PaymentMethod'),
    ('Partner', 'Dependents'),
    ('StreamingTV', 'StreamingMovies'),
    ('Contract', 'tenure_bin_'),
    ('InternetService', 'MonthlyCharges_bin_'),
    ('PaymentMethod', 'MonthlyCharges_bin_'),
    ('Contract', 'TotalCharges_bin_'),
]


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_safe_string_key(s):
    return s.astype('string').fillna(MISSING_TOKEN)


# ----------------------------------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------------------------------
def strip_object_cols(df):
    df = df.copy()
    for c in df.select_dtypes(include=['object', 'string']).columns.tolist():
        df[c] = df[c].astype('string').str.strip()
    return df


def add_charge_string_parts(df):
    df = df.copy()
    df['MonthlyCharges_str'] = df['MonthlyCharges'].astype('string')
    df['TotalCharges_str'] = df['TotalCharges'].astype('string')
    for c in ['MonthlyCharges_str', 'TotalCharges_str']:
        df[c] = df[c].fillna('missing').astype('string')
    df['mc_dec'] = df['MonthlyCharges_str'].str.split('.').str[1].fillna('')
    df['mc0_'] = df['mc_dec'].str[0].fillna('-1').replace('', '-1').astype('category')
    df['mc1_'] = df['mc_dec'].str[1].fillna('-1').replace('', '-1').astype('category')
    df['mc_len_'] = df['mc_dec'].str.len().fillna(0).astype('int32').astype('category')
    df['tc_dec'] = df['TotalCharges_str'].str.split('.').str[1].fillna('')
    df['tc0_'] = df['tc_dec'].str[0].fillna('-1').replace('', '-1').astype('category')
    df['tc1_'] = df['tc_dec'].str[1].fillna('-1').replace('', '-1').astype('category')
    df['tc_len_'] = df['tc_dec'].str.len().fillna(0).astype('int32').astype('category')
    return df


def clean_numeric_charges(df):
    df = df.copy()
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].replace('', pd.NA).replace(' ', pd.NA)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if df['TotalCharges'].isna().any():
        df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])
    return df


def load_data():
    dtypes = {'MonthlyCharges': 'string', 'TotalCharges': 'string'}
    train = pd.read_csv(DATA_DIR / 'train.csv', dtype=dtypes)
    test = pd.read_csv(DATA_DIR / 'test.csv', dtype=dtypes)
    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, dtype=dtypes)
    for df in (train, orig):
        if str(df[TARGET].dtype).startswith('string') or df[TARGET].dtype == 'O':
            df[TARGET] = df[TARGET].astype('string').str.strip().map({'Yes': 1, 'No': 0})
        df[TARGET] = df[TARGET].astype('int64')
    train, test, orig = [clean_numeric_charges(add_charge_string_parts(strip_object_cols(df)))
                         for df in (train, test, orig)]
    return train, test, orig


def add_service_features(df):
    df = df.copy()
    for c in SERVICE_COLS:
        if c not in df.columns:
            continue
        s = df[c].astype('string')
        df[f'{c}_yes'] = s.isin(['Yes', 'Fiber optic', 'DSL']).astype('float32')
        df[f'{c}_no'] = s.eq('No').astype('float32')
        df[f'{c}_no_internet'] = s.eq('No internet service').astype('float32')
        df[f'{c}_no_phone'] = s.eq('No phone service').astype('float32')
    yes_cols = [f'{c}_yes' for c in SERVICE_COLS if f'{c}_yes' in df.columns]
    no_cols = [f'{c}_no' for c in SERVICE_COLS if f'{c}_no' in df.columns]
    noi_cols = [f'{c}_no_internet' for c in SERVICE_COLS if f'{c}_no_internet' in df.columns]
    nop_cols = [f'{c}_no_phone' for c in SERVICE_COLS if f'{c}_no_phone' in df.columns]
    if yes_cols:
        df['_service_yes_count'] = df[yes_cols].sum(axis=1).astype('float32')
    if no_cols:
        df['_service_no_count'] = df[no_cols].sum(axis=1).astype('float32')
    if noi_cols:
        df['_no_internet_count'] = df[noi_cols].sum(axis=1).astype('float32')
    if nop_cols:
        df['_no_phone_count'] = df[nop_cols].sum(axis=1).astype('float32')
    return df


def add_charge_math_features(df):
    df = df.copy()
    t = df['tenure'].astype('float32')
    mc = df['MonthlyCharges'].astype('float32')
    tc = df['TotalCharges'].astype('float32')
    sc = df['SeniorCitizen'].astype('float32')
    df['_tenure_x_monthly'] = (t * mc).astype('float32')
    df['_avg_monthly_from_total'] = (tc / (t + 1.0)).astype('float32')
    df['_charge_gap'] = (tc - t * mc).astype('float32')
    df['_charge_gap_abs'] = np.abs(df['_charge_gap']).astype('float32')
    df['_charge_gap_rel'] = (df['_charge_gap'] / (tc + 1.0)).astype('float32')
    df['_monthly_per_tenure'] = (mc / (t + 1.0)).astype('float32')
    df['_total_per_monthly'] = (tc / (mc + 1.0)).astype('float32')
    df['_monthly_minus_avg'] = (mc - df['_avg_monthly_from_total']).astype('float32')
    df['_monthly_plus_total'] = (mc + tc).astype('float32')
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        x = df[col].astype('float32')
        df[f'_{col}_log1p'] = np.log1p(np.clip(x, 0, None)).astype('float32')
        df[f'_{col}_sqrt'] = np.sqrt(np.clip(x, 0, None)).astype('float32')
        df[f'_{col}_sq'] = np.square(x).astype('float32')
    df['_SeniorCitizen_x_MonthlyCharges'] = (sc * mc).astype('float32')
    df['_SeniorCitizen_x_TotalCharges'] = (sc * tc).astype('float32')
    return df


def add_deterministic_bins(df):
    df = df.copy()

    def cut(s, bins):
        return pd.cut(s, bins=bins, labels=False).fillna(-1).astype('int32').astype('category')

    charge_bins = [-np.inf, 20, 35, 50, 65, 80, 95, 110, np.inf]
    df['tenure_bin_'] = cut(df['tenure'], [-np.inf, 1, 6, 12, 24, 36, 48, 60, 72, np.inf])
    df['MonthlyCharges_bin_'] = cut(df['MonthlyCharges'], charge_bins)
    df['TotalCharges_bin_'] = cut(df['TotalCharges'], [-np.inf, 100, 500, 1000, 2000, 3000, 4500, 6000, np.inf])
    df['avg_monthly_bin_'] = cut(df['_avg_monthly_from_total'], charge_bins)
    return df


def add_numeric_categorical_copies(df):
    df = df.copy()
    df['SeniorCitizen_cat_'] = df['SeniorCitizen'].astype('int32').astype('category')
    df['tenure_int_'] = df['tenure'].round(0).astype('int32').astype('category')
    df['tenure_div3_'] = np.floor(df['tenure'] / 3).astype('int32').astype('category')
    df['tenure_div6_'] = np.floor(df['tenure'] / 6).astype('int32').astype('category')
    df['tenure_div12_'] = np.floor(df['tenure'] / 12).astype('int32').astype('category')
    df['MonthlyCharges_int_'] = df['MonthlyCharges'].round(0).astype('int32').astype('category')
    df['MonthlyCharges_div5_'] = np.floor(df['MonthlyCharges'] / 5).astype('int32').astype('category')
    df['MonthlyCharges_div10_'] = np.floor(df['MonthlyCharges'] / 10).astype('int32').astype('category')
    df['TotalCharges_div50_'] = np.floor(df['TotalCharges'] / 50).astype('int32').astype('category')
    df['TotalCharges_div100_'] = np.floor(df['TotalCharges'] / 100).astype('int32').astype('category')
    df['TotalCharges_div250_'] = np.floor(df['TotalCharges'] / 250).astype('int32').astype('category')
    df['avg_monthly_int_'] = df['_avg_monthly_from_total'].round(0).astype('int32').astype('category')
    df['avg_monthly_div5_'] = np.floor(df['_avg_monthly_from_total'] / 5).astype('int32').astype('category')
    return df


def add_pair_categoricals(df):
    df = df.copy()
    for a, b in PAIR_COLS:
        if a in df.columns and b in df.columns:
            df[f'{a}__{b}_'] = (to_safe_string_key(df[a]) + '|' + to_safe_string_key(df[b])).astype('category')
    df['mc0_tc0_'] = (to_safe_string_key(df['mc0_']) + '|' + to_safe_string_key(df['tc0_'])).astype('category')
    df['mc1_tc1_'] = (to_safe_string_key(df['mc1_']) + '|' + to_safe_string_key(df['tc1_'])).astype('category')
    df['Contract_mc0_'] = (to_safe_string_key(df['Contract']) + '|' + to_safe_string_key(df['mc0_'])).astype('category')
    df['InternetService_tc0_'] = (to_safe_string_key(df['InternetService']) + '|'
                                  + to_safe_string_key(df['tc0_'])).astype('category')
    return df


def add_orig_mappings(df, orig_df):
    """Original-data churn mean and row count of each single categorical and each pair (keys compared as strings)."""
    df = df.copy()
    global_mean = float(orig_df[TARGET].mean())
    map_cols = [c for c in (BASE_CAT_COLS
                            + ['mc0_', 'mc1_', 'mc_len_', 'tc0_', 'tc1_', 'tc_len_']
                            + ['tenure_bin_', 'MonthlyCharges_bin_', 'TotalCharges_bin_', 'avg_monthly_bin_']
                            + ['SeniorCitizen_cat_', 'tenure_div3_', 'tenure_div6_', 'tenure_div12_',
                               'MonthlyCharges_div5_', 'MonthlyCharges_div10_', 'TotalCharges_div50_',
                               'TotalCharges_div100_', 'TotalCharges_div250_', 'avg_monthly_div5_'])
                if c in df.columns and c in orig_df.columns]

    def add_mapping(name, df_key, orig_key):
        tmp = pd.DataFrame({'_key_': orig_key, TARGET: orig_df[TARGET].astype('float32').values})
        grp_mean = tmp.groupby('_key_', dropna=False, observed=False)[TARGET].mean()
        grp_size = tmp.groupby('_key_', dropna=False, observed=False).size()
        df[f'._orig_mean_{name}'] = pd.to_numeric(df_key.map(grp_mean), errors='coerce').fillna(global_mean).astype(
            'float32')
        df[f'._orig_size_{name}'] = pd.to_numeric(df_key.map(grp_size), errors='coerce').fillna(0).astype('float32')

    for c in map_cols:
        add_mapping(c, to_safe_string_key(df[c]), to_safe_string_key(orig_df[c]))
    for a, b in PAIR_COLS:
        if a in df.columns and b in df.columns and a in orig_df.columns and b in orig_df.columns:
            add_mapping(f'{a}__{b}', to_safe_string_key(df[a]) + '|' + to_safe_string_key(df[b]),
                        to_safe_string_key(orig_df[a]) + '|' + to_safe_string_key(orig_df[b]))
    return df


def full_feature_engineering_base(df):
    df = add_service_features(df.copy())
    df = add_charge_math_features(df)
    df = add_deterministic_bins(df)
    df = add_numeric_categorical_copies(df)
    return add_pair_categoricals(df)


def build_features():
    train, test, orig = load_data()
    X = train.drop([ID, TARGET], axis=1).copy()
    y = train[TARGET].astype('int64').values
    X_test = test.drop([ID], axis=1).copy()

    orig_map_df = full_feature_engineering_base(orig.copy())
    X = add_orig_mappings(full_feature_engineering_base(X), orig_map_df)
    X_test = add_orig_mappings(full_feature_engineering_base(X_test), orig_map_df)
    helpers = ['MonthlyCharges_str', 'TotalCharges_str', 'mc_dec', 'tc_dec']
    X = X.drop(columns=[c for c in helpers if c in X.columns])
    X_test = X_test.drop(columns=[c for c in helpers if c in X_test.columns])

    cat_cols, num_cols = [], []
    for c in X.columns:
        if str(X[c].dtype) == 'category' or str(X[c].dtype).startswith('string') or X[c].dtype == 'object':
            cat_cols.append(c)
        else:
            num_cols.append(c)
    print(f'{len(cat_cols)} categorical, {len(num_cols)} numerical columns', flush=True)

    col_to_stype = {c: stype.numerical for c in num_cols}
    col_to_stype.update({c: stype.categorical for c in cat_cols})
    col_to_stype[TARGET] = stype.numerical
    return X, y, X_test, cat_cols, num_cols, col_to_stype


# ----------------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda' and CFG.USE_AMP)):
            probs = torch.softmax(model(batch), dim=-1)[:, 1]
        out.append(probs.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def cross_validate(X, y, X_test, cat_cols, num_cols, col_to_stype, n_splits, device):
    seed_everything(SEED)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr = X.iloc[tr_idx].reset_index(drop=True).copy()
        X_va = X.iloc[val_idx].reset_index(drop=True).copy()
        y_tr = pd.Series(y[tr_idx], name=TARGET).reset_index(drop=True)
        y_va = pd.Series(y[val_idx], name=TARGET).reset_index(drop=True)
        X_te = X_test.reset_index(drop=True).copy()

        fold_num_cols = [c for c in num_cols if c in X_tr.columns]
        fold_cat_cols = [c for c in cat_cols if c in X_tr.columns]
        medians = X_tr[fold_num_cols].median()
        for df in (X_tr, X_va, X_te):
            df[fold_num_cols] = df[fold_num_cols].fillna(medians)
        for c in fold_cat_cols:
            for df in (X_tr, X_va, X_te):
                df[c] = to_safe_string_key(df[c]).astype('category')
        scaler = MinMaxScaler()
        X_tr[fold_num_cols] = scaler.fit_transform(X_tr[fold_num_cols]).astype(np.float32)
        X_va[fold_num_cols] = scaler.transform(X_va[fold_num_cols]).astype(np.float32)
        X_te[fold_num_cols] = scaler.transform(X_te[fold_num_cols]).astype(np.float32)
        train_df = pd.concat([X_tr, y_tr], axis=1)
        val_df = pd.concat([X_va, y_va], axis=1)

        stype_encoder_dict = {stype.categorical: EmbeddingEncoder(), stype.numerical: CFG.NUM_ENCODER}
        col_stats = {c: compute_col_stats(train_df[c], stype=col_to_stype[c]) for c in train_df.columns
                     if c in col_to_stype}
        converter = DataFrameToTensorFrameConverter(col_stats=col_stats, target_col=TARGET, col_to_stype=col_to_stype)
        tf_train, tf_val, tf_test = converter(train_df), converter(val_df), converter(X_te)
        train_loader = DataLoader(tf_train, batch_size=CFG.BATCH, shuffle=True, pin_memory=True)
        val_loader = DataLoader(tf_val, batch_size=CFG.BATCH, shuffle=False, pin_memory=True)
        test_loader = DataLoader(tf_test, batch_size=CFG.BATCH, shuffle=False, pin_memory=True)

        model = FTTransformer(channels=CFG.CHANNELS, out_channels=2, num_layers=CFG.LAYERS, col_stats=col_stats,
                              col_names_dict=converter.col_names_dict, stype_encoder_dict=stype_encoder_dict).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.LR,
                                                        total_steps=len(train_loader) * CFG.EPOCHS,
                                                        pct_start=CFG.WARMUP, cycle_momentum=False)

        best_auc, best_epoch, best_state, patience_left = -np.inf, 0, None, CFG.PATIENCE
        for epoch in range(1, CFG.EPOCHS + 1):
            model.train()
            total_loss, total_samples = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda' and CFG.USE_AMP)):
                    loss = F.cross_entropy(model(batch), batch.y.long(), label_smoothing=CFG.LABEL_SMOOTH)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * len(batch.y)
                total_samples += len(batch.y)

            val_auc = roc_auc_score(y[val_idx], predict(model, val_loader, device))
            if val_auc > best_auc + 1e-6:
                best_auc, best_epoch, patience_left = val_auc, epoch, CFG.PATIENCE
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_left -= 1
            print(f'Fold {fold} | Epoch {epoch:02d}/{CFG.EPOCHS} | train_loss={total_loss / (total_samples + 1e-12):.5f} '
                  f'| val_auc={val_auc:.6f} | best_auc={best_auc:.6f} | patience={patience_left}', flush=True)
            if patience_left <= 0:
                break

        model.load_state_dict(best_state)
        oof_preds[val_idx] = predict(model, val_loader, device)
        test_preds += predict(model, test_loader, device) / n_splits
        print(f'Fold {fold}/{n_splits} | best_epoch={best_epoch} | AUC = '
              f'{roc_auc_score(y[val_idx], oof_preds[val_idx]):.6f}', flush=True)

        del X_tr, X_va, X_te, y_tr, y_va, train_df, val_df, col_stats, converter, tf_train, tf_val, tf_test
        del train_loader, val_loader, test_loader, model, optimizer, scheduler, scaler, best_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'{n_splits}-fold OOF AUC = {roc_auc_score(y, oof_preds):.6f}', flush=True)
    return oof_preds, test_preds


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, X_test, cat_cols, num_cols, col_to_stype = build_features()
    oof, test_pred = cross_validate(X, y, X_test, cat_cols, num_cols, col_to_stype, CFG.FOLDS, device)
    if TEST_FOLDS != CFG.FOLDS:
        _, test_pred = cross_validate(X, y, X_test, cat_cols, num_cols, col_to_stype, TEST_FOLDS, device)
    save_pair(MODEL_DIR, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
