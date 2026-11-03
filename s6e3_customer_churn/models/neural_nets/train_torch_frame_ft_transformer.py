#!/usr/bin/env python
"""FT-Transformer (PyTorch Frame) on the raw columns, category copies of the numerics, original-data target means and
8 interaction categoricals.

Features (TotalCharges read as a string, blanks -> NaN -> tenure x MonthlyCharges; string columns stripped):
- categoricals (27): the 15 string columns; a category code of the value of each of the 4 numeric columns
  (SeniorCitizen, tenure, MonthlyCharges, TotalCharges; rounded to tens when the column has more than
  min(5000, 5% of rows) distinct values), with the codes of the train rows (test values unseen in train -> -1);
  8 interaction categoricals (code of the 'a_b' string): Contract x tenure, InternetService x MonthlyCharges,
  PaymentMethod x MonthlyCharges, OnlineSecurity x TechSupport, PaperlessBilling x PaymentMethod,
  SeniorCitizen x Contract, Partner x Dependents, tenure x MonthlyCharges;
- numericals (8): the 4 numeric columns min-max scaled on all train rows, and the original-data churn mean of each of
  their values (unseen -> the original churn rate), unscaled.
Model: PyTorch Frame FTTransformer (64 channels, 5 layers, 2 output logits); EmbeddingEncoder for the categoricals,
LinearPeriodicEncoder for the numericals; column stats from the fold's training rows.
Training: cross-entropy, AdamW lr 1e-3, weight decay 1e-4, OneCycleLR (15% warm-up, no momentum cycling), batch 512,
6 epochs; the last epoch predicts (softmax probability of churn). Python / numpy / torch are seeded with 42
(deterministic cuDNN) before the folds.
Usage: python train_torch_frame_ft_transformer.py"""
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

PAIR = 'torch_frame_ft_transformer_original_te_combos'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
TEST_FOLDS = 15


class CFG:
    FOLDS = 5
    EPOCHS = 6
    LR = 1e-3
    WD = 1e-4
    WARMUP = 0.15
    BATCH = 512
    CHANNELS = 64
    LAYERS = 5
    NUM_ENCODER = LinearPeriodicEncoder()


IMPORTANT_PAIRS = [
    ('Contract', 'tenure'),
    ('InternetService', 'MonthlyCharges'),
    ('PaymentMethod', 'MonthlyCharges'),
    ('OnlineSecurity', 'TechSupport'),
    ('PaperlessBilling', 'PaymentMethod'),
    ('SeniorCitizen', 'Contract'),
    ('Partner', 'Dependents'),
    ('tenure', 'MonthlyCharges'),
]


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean_totalcharges(df):
    df = df.copy()
    df['TotalCharges'] = df['TotalCharges'].astype('string').str.strip().replace('', pd.NA).replace(' ', pd.NA)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df


def strip_object_cols(df):
    df = df.copy()
    for c in df.select_dtypes(include=['object', 'string']).columns.tolist():
        df[c] = df[c].astype('string').str.strip()
    return df


def load_data():
    train = pd.read_csv(DATA_DIR / 'train.csv', dtype={'TotalCharges': 'string'})
    test = pd.read_csv(DATA_DIR / 'test.csv', dtype={'TotalCharges': 'string'})
    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, dtype={'TotalCharges': 'string'})
    for df in (train, orig):
        if df[TARGET].dtype == 'O' or str(df[TARGET].dtype).startswith('string'):
            df[TARGET] = df[TARGET].astype(str).str.strip().map({'Yes': 1, 'No': 0})
        df[TARGET] = df[TARGET].astype('int64')
    train, test, orig = clean_totalcharges(train), clean_totalcharges(test), clean_totalcharges(orig)
    for df in (train, test, orig):
        if df['TotalCharges'].isna().any():
            df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])
    return strip_object_cols(train), strip_object_cols(test), strip_object_cols(orig)


def feature_engineering(df, orig, num_cols, category_map, fit=False):
    """Category copies of the numerics, original-data target means of the numerics, interaction categoricals.
    New categoricals end with '_'; the target means start with '._' and stay unscaled."""
    df = df.copy()
    for col in num_cols:
        if fit:
            round_flag = df[col].nunique(dropna=True) > min(5000, len(df) * 0.05)
            series = df[col].round(-1) if round_flag else df[col]
            codes, uniques = pd.factorize(series)
            category_map[col] = {'uniques': pd.Index(uniques), 'round_flag': round_flag}
        else:
            round_flag = category_map[col]['round_flag']
            series = df[col].round(-1) if round_flag else df[col]
            code_map = {cat: i for i, cat in enumerate(category_map[col]['uniques'])}
            codes = series.map(code_map).fillna(-1).astype('int32')
        df[f'{col}_cat_'] = pd.Series(codes, index=df.index).astype('category')

    global_mean = float(orig[TARGET].mean())
    for col in num_cols:
        grp = orig.groupby(col, dropna=False)[TARGET].mean()
        df[f'._{col}_mean_target_orig'] = df[col].map(grp).fillna(global_mean).astype('float32')

    for col1, col2 in IMPORTANT_PAIRS:
        combo_name = f'{col1}_{col2}_'
        combo_series = df[col1].astype(str) + '_' + df[col2].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series)
            category_map[combo_name] = pd.Index(uniques)
        else:
            code_map = {cat: i for i, cat in enumerate(category_map[combo_name])}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = pd.Series(codes, index=df.index).astype('category')

    new_cat_cols = [c for c in df.columns if c.endswith('_')]
    new_num_cols = [c for c in df.columns if c.startswith('_')]
    new_enc_cols = [c for c in df.columns if c.startswith('.')]
    return df, new_cat_cols, new_num_cols, new_enc_cols


def build_features():
    train, test, orig = load_data()
    X = train.drop([ID, TARGET], axis=1).copy()
    y = train[TARGET].copy().values
    X_test = test.drop([ID], axis=1).copy()
    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    category_map = {}
    X, new_cat_cols, new_num_cols, new_enc_cols = feature_engineering(X, orig, num_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(X_test, orig, num_cols, category_map, fit=False)
    cat_cols = cat_cols + new_cat_cols
    num_cols_scaled = num_cols + new_num_cols
    num_cols_all = num_cols_scaled + new_enc_cols

    scaler = MinMaxScaler()
    X[num_cols_scaled] = scaler.fit_transform(X[num_cols_scaled]).astype(np.float32)
    X_test[num_cols_scaled] = scaler.transform(X_test[num_cols_scaled]).astype(np.float32)
    for c in new_enc_cols:
        X[c] = X[c].astype('float32')
        X_test[c] = X_test[c].astype('float32')
    for c in cat_cols:
        X[c] = X[c].astype('category')
        X_test[c] = X_test[c].astype('category')
    print(f'{len(cat_cols)} categorical, {len(num_cols_all)} numerical columns', flush=True)

    col_to_stype = {c: stype.numerical for c in num_cols_all}
    col_to_stype.update({c: stype.categorical for c in cat_cols})
    col_to_stype[TARGET] = stype.numerical
    return X, y, X_test, col_to_stype


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out.append(torch.softmax(model(batch), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def cross_validate(X, y, X_test, col_to_stype, n_splits, device):
    seed_everything(SEED)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        train_df = pd.concat([X.iloc[tr_idx].reset_index(drop=True),
                              pd.Series(y[tr_idx], name=TARGET).reset_index(drop=True)], axis=1)
        val_df = pd.concat([X.iloc[val_idx].reset_index(drop=True),
                            pd.Series(y[val_idx], name=TARGET).reset_index(drop=True)], axis=1)
        stype_encoder_dict = {stype.categorical: EmbeddingEncoder(), stype.numerical: CFG.NUM_ENCODER}
        col_stats = {c: compute_col_stats(train_df[c], stype=col_to_stype[c]) for c in train_df.columns
                     if c in col_to_stype}
        converter = DataFrameToTensorFrameConverter(col_stats=col_stats, target_col=TARGET, col_to_stype=col_to_stype)
        tf_train, tf_val, tf_test = converter(train_df), converter(val_df), converter(X_test)
        train_loader = DataLoader(tf_train, batch_size=CFG.BATCH, shuffle=True, pin_memory=True)
        val_loader = DataLoader(tf_val, batch_size=CFG.BATCH, shuffle=False, pin_memory=True)
        test_loader = DataLoader(tf_test, batch_size=CFG.BATCH, shuffle=False, pin_memory=True)

        model = FTTransformer(channels=CFG.CHANNELS, out_channels=2, num_layers=CFG.LAYERS, col_stats=col_stats,
                              col_names_dict=converter.col_names_dict, stype_encoder_dict=stype_encoder_dict).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.LR,
                                                        total_steps=len(train_loader) * CFG.EPOCHS,
                                                        pct_start=CFG.WARMUP, cycle_momentum=False)
        for epoch in range(1, CFG.EPOCHS + 1):
            model.train()
            total_loss, total_samples = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(batch), batch.y.long())
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * len(batch.y)
                total_samples += len(batch.y)
            val_auc = roc_auc_score(y[val_idx], predict(model, val_loader, device))
            print(f'Fold {fold} | Epoch {epoch:02d}/{CFG.EPOCHS} | train_loss={total_loss / (total_samples + 1e-12):.5f} '
                  f'| val_auc={val_auc:.6f}', flush=True)

        oof_preds[val_idx] = predict(model, val_loader, device)
        test_preds += predict(model, test_loader, device) / n_splits
        print(f'Fold {fold}/{n_splits} AUC = {roc_auc_score(y[val_idx], oof_preds[val_idx]):.6f}', flush=True)

        del train_df, val_df, col_stats, converter, tf_train, tf_val, tf_test, train_loader, val_loader, test_loader
        del model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'{n_splits}-fold OOF AUC = {roc_auc_score(y, oof_preds):.6f}', flush=True)
    return oof_preds, test_preds


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, X_test, col_to_stype = build_features()
    oof, test_pred = cross_validate(X, y, X_test, col_to_stype, CFG.FOLDS, device)
    if TEST_FOLDS != CFG.FOLDS:
        _, test_pred = cross_validate(X, y, X_test, col_to_stype, TEST_FOLDS, device)
    save_pair(MODEL_DIR, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
