#!/usr/bin/env python
"""Self-normalizing network (SELU) on character n-gram TF-IDF of the charge strings, Benford probabilities,
nearest-original-row churn votes and one-hot categoricals.

Cleaning: TotalCharges blank -> NaN -> tenure x MonthlyCharges (train, test and original rows); numerics as float32.
Features, built once on train + test (the original rows only shape features; they are not training rows):
- the 3 raw numerics;
- TF-IDF of character 2-4-grams (128 most frequent) of the text "MC<MonthlyCharges> TC<TotalCharges>" (float32 string
  form), vocabulary fitted on the original + train rows;
- Benford probability log10(1 + 1/d) of the first non-zero digit d of MonthlyCharges and TotalCharges (0 if none);
- the mean churn label and mean distance of the 3 nearest original rows (sklearn NearestNeighbors on the 3 numerics,
  standardized with original-data statistics);
- one-hot encodings of the 16 categoricals.
All columns are standardized on train + test.
Model: 4 SELU layers 256-128-64-32 with LeCun-normal init (N(0, 1 / fan_in), zero bias) and alpha dropout 0.05 ->
1 logit.
Training: BCE with logits on labels smoothed by 0.015 (y (1 - 0.015) + 0.0075), AdamW lr 1e-3, weight decay 1e-4,
ReduceLROnPlateau on the held-out AUC (factor 0.5, patience 3, min lr 1e-5), batch 1024 (last partial batch dropped),
up to 40 epochs; early stopping after 8 epochs without a gain in held-out AUC; the best epoch predicts the held-out and
test rows. Seed 42 (python / numpy / torch, deterministic cuDNN) once at the start, so the pair is reproduced only
when the 5 folds run in order in one process. Test = mean of the 5 fold models.
Usage: python train_snn_charge_ngrams.py [--folds 0,1,2,3,4]"""
import argparse
import gc
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, ID, NUMERICS, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'snn_charge_char_ngrams_benford_orig_knn'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
BATCH_SIZE = 1024
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
LABEL_SMOOTHING = 0.015

HIDDEN_SIZES = [256, 128, 64, 32]
ALPHA_DROPOUT_RATE = 0.05
TFIDF_MAX_FEATURES = 128  # character n-grams of the charge strings

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_CATS = list(CATEGORICALS)
BASE_NUMS = list(NUMERICS)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def smooth_labels(y, smoothing=LABEL_SMOOTHING):
    return y * (1.0 - smoothing) + 0.5 * smoothing


def get_leading_digit(series):
    """First non-zero digit of each value (0 when there is none)."""
    s_str = series.fillna(0).astype(str).str.replace(r'[^1-9]', '', regex=True)
    return s_str.str[0].fillna('0').astype(int)


def build_features():
    print('Loading train / test / original ...', flush=True)
    train = read_train().drop(columns=[ID])
    test = read_test().drop(columns=[ID])
    orig = read_original()

    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    y_train = train[TARGET].values
    train = train.drop(columns=[TARGET])
    y_orig = orig[TARGET].values

    for df in (train, test, orig):
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        for col in BASE_NUMS:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        miss = df['TotalCharges'].isna()
        if miss.any():
            df.loc[miss, 'TotalCharges'] = df.loc[miss, 'tenure'].fillna(0) * df.loc[miss, 'MonthlyCharges'].fillna(0)

    n_train = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # 1. TF-IDF of character n-grams of the charge strings
    combined['charge_text'] = ('MC' + combined['MonthlyCharges'].fillna(0).astype(str) + ' '
                               + 'TC' + combined['TotalCharges'].fillna(0).astype(str))
    orig['charge_text'] = ('MC' + orig['MonthlyCharges'].fillna(0).astype(str) + ' '
                           + 'TC' + orig['TotalCharges'].fillna(0).astype(str))
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=TFIDF_MAX_FEATURES)
    tfidf.fit(pd.concat([orig['charge_text'], combined.iloc[:n_train]['charge_text']]))
    tfidf_features = tfidf.transform(combined['charge_text']).toarray()
    for i in range(TFIDF_MAX_FEATURES):
        combined[f'tfidf_char_{i}'] = tfidf_features[:, i].astype('float32')

    # 2. Benford probability of the leading digit
    for col in ['MonthlyCharges', 'TotalCharges']:
        leading_d = get_leading_digit(combined[col])
        benford_p = np.log10(1 + 1 / np.clip(leading_d, 1, 9))
        combined[f'{col}_benford_prob'] = np.where(leading_d == 0, 0.0, benford_p).astype('float32')

    # 3. churn vote of the 3 nearest original rows
    scaler = StandardScaler().fit(orig[BASE_NUMS])
    orig_scaled = scaler.transform(orig[BASE_NUMS])
    comb_scaled = scaler.transform(combined[BASE_NUMS])
    knn = NearestNeighbors(n_neighbors=3, n_jobs=-1)
    knn.fit(orig_scaled)
    distances, indices = knn.kneighbors(comb_scaled)
    combined['orig_nn3_target_vote'] = y_orig[indices].mean(axis=1).astype('float32')
    combined['orig_nn3_avg_dist'] = distances.mean(axis=1).astype('float32')

    # 4. one-hot categoricals
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(combined[BASE_CATS].astype(str))
    cat_col_names = [f'cat_{i}' for i in range(cat_encoded.shape[1])]
    for i, col in enumerate(cat_col_names):
        combined[col] = cat_encoded[:, i].astype('float32')

    # 5. standardize everything
    features_to_scale = (BASE_NUMS + [f'tfidf_char_{i}' for i in range(TFIDF_MAX_FEATURES)]
                         + ['MonthlyCharges_benford_prob', 'TotalCharges_benford_prob', 'orig_nn3_target_vote',
                            'orig_nn3_avg_dist'] + cat_col_names)
    combined[features_to_scale] = StandardScaler().fit_transform(combined[features_to_scale]).astype('float32')
    print(f'{len(features_to_scale)} standardized features', flush=True)

    X_train = combined.iloc[:n_train][features_to_scale].values
    X_test = combined.iloc[n_train:][features_to_scale].values
    return X_train, X_test, y_train


class TabDataset(Dataset):
    def __init__(self, x_data, targets=None):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.x_data[idx], self.targets[idx]
        return self.x_data[idx]


class SNN(nn.Module):
    """Self-normalizing network: Linear (LeCun normal init) -> SELU -> AlphaDropout blocks."""

    def __init__(self, in_features, hidden_sizes, alpha_dropout_rate):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_sizes:
            linear = nn.Linear(prev_dim, h)
            nn.init.normal_(linear.weight, mean=0.0, std=np.sqrt(1.0 / prev_dim))
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.SELU(), nn.AlphaDropout(p=alpha_dropout_rate)]
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, 1)
        nn.init.normal_(self.out.weight, mean=0.0, std=np.sqrt(1.0 / prev_dim))
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.net(x)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    X_train, X_test, y_train = build_features()

    criterion = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(TabDataset(X_test), batch_size=BATCH_SIZE, shuffle=False)

    fold_ids = folds(y_train)
    for k in fold_list(args.folds):
        print(f'\n--- Fold {k + 1} ---', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        y_tr_smooth = smooth_labels(y_train[tr_idx])
        train_loader = DataLoader(TabDataset(X_train[tr_idx], y_tr_smooth), batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True)
        val_loader = DataLoader(TabDataset(X_train[va_idx], y_train[va_idx]), batch_size=BATCH_SIZE, shuffle=False)

        model = SNN(in_features=X_train.shape[1], hidden_sizes=HIDDEN_SIZES,
                    alpha_dropout_rate=ALPHA_DROPOUT_RATE).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                                               min_lr=1e-5)

        best_auc = 0
        patience_counter = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds_epoch = []
            with torch.no_grad():
                for x_batch, _ in val_loader:
                    val_preds_epoch.extend(torch.sigmoid(model(x_batch.to(DEVICE))).cpu().numpy())
            val_auc = roc_auc_score(y_train[va_idx], val_preds_epoch)
            scheduler.step(val_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                best_preds = np.array(val_preds_epoch)
                patience_counter = 0
                test_preds_fold = []
                with torch.no_grad():
                    for x_batch in test_loader:
                        test_preds_fold.extend(torch.sigmoid(model(x_batch.to(DEVICE))).cpu().numpy())
            else:
                patience_counter += 1
            print(f'  epoch {epoch:02d} | val AUC {val_auc:.6f} | best {best_auc:.6f}', flush=True)
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch}', flush=True)
                break

        print(f'Best Fold {k + 1} AUC: {best_auc:.6f}', flush=True)
        save_part(MODEL_DIR, PAIR, k, best_preds, np.array(test_preds_fold))
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assemble(MODEL_DIR, PAIR, fold_ids, y_train)


if __name__ == '__main__':
    main()
