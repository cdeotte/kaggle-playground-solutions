#!/usr/bin/env python
"""DeepFM on 41 token fields: categoricals, original-snapped numerics, original-quantile bins, original churn-rate
deciles and 3 crosses.

Cleaning: TotalCharges blank -> NaN -> tenure x MonthlyCharges (train, test and original rows); numerics as float32.
Token fields, built once on train + test (the original rows only shape the tokens; they are not training rows):
- the 16 categoricals;
- tenure / MonthlyCharges / TotalCharges snapped to the nearest distinct original value (as a token);
- 100-bin quantile bins of the 3 numerics (sklearn KBinsDiscretizer fitted on the original rows);
- floor(10 x original churn rate of the level) of every categorical (unseen level -> 0.22);
- 3 crosses: Contract x InternetService, PaymentMethod x PaperlessBilling, snapped MonthlyCharges x Contract.
Each field is factorized on train + test and offset into one shared vocabulary.
Model: DeepFM. Linear term (1-d embedding per token, Xavier init) + bias, FM second-order term of 16-d embeddings
(Xavier init), and an MLP on the flattened embeddings: 128-64-32 (BatchNorm, ReLU, dropout 0.3) -> 1; the three are
summed into one logit.
Training: BCE with logits, AdamW lr 3e-3, weight decay 1e-4, cosine annealing over 25 epochs (eta_min 1e-5), batch 1024
(last partial batch dropped), up to 25 epochs; early stopping after 6 epochs without a gain in held-out AUC; the best
epoch predicts the held-out and test rows. Seed 42 (python / numpy / torch, deterministic cuDNN) once at the start, so
the pair is reproduced only when the 5 folds run in order in one process. Test = mean of the 5 fold models.
Usage: python train_deepfm_snapped_tokens.py [--folds 0,1,2,3,4]"""
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, ID, NUMERICS, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'deepfm_snapped_quantile_prior_tokens'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
BATCH_SIZE = 1024
EPOCHS = 25
LR = 3e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6

EMBED_DIM = 16
MLP_HIDDEN = [128, 64, 32]
DROPOUT = 0.3

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


def get_nearest_val(vals, ref_vals):
    """Snap values to the closest distinct reference value."""
    ref_vals = np.sort(np.unique(ref_vals))
    idx = np.searchsorted(ref_vals, vals)
    idx = np.clip(idx, 1, len(ref_vals) - 1)
    left, right = ref_vals[idx - 1], ref_vals[idx]
    return np.where(np.abs(vals - left) <= np.abs(vals - right), left, right)


def build_tokens():
    print('Loading train / test / original ...', flush=True)
    train = read_train().drop(columns=[ID])
    test = read_test().drop(columns=[ID])
    orig = read_original()

    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    orig[TARGET] = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(np.int8)
    y_train = train[TARGET].values
    train = train.drop(columns=[TARGET])

    for df in (train, test, orig):
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        for col in BASE_NUMS:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        miss = df['TotalCharges'].isna()
        if miss.any():
            df.loc[miss, 'TotalCharges'] = df.loc[miss, 'tenure'].fillna(0) * df.loc[miss, 'MonthlyCharges'].fillna(0)

    n_train = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    categorical_features = BASE_CATS.copy()

    # 1. numerics snapped to the nearest original value, as tokens
    for col in BASE_NUMS:
        snapped = get_nearest_val(combined[col].fillna(0).values, orig[col].dropna().values)
        col_name = f'{col}_snapped_token'
        combined[col_name] = snapped.astype(str)
        categorical_features.append(col_name)

    # 2. 100-bin quantile bins fitted on the original rows
    for col in BASE_NUMS:
        kb = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')
        kb.fit(orig[[col]].dropna())
        col_name = f'{col}_quantile_token'
        combined[col_name] = kb.transform(combined[[col]].fillna(0)).flatten().astype(int).astype(str)
        categorical_features.append(col_name)

    # 3. original churn rate of every categorical level, as deciles
    for col in BASE_CATS:
        if col in orig.columns:
            m = orig.groupby(col)[TARGET].mean()
            proba = combined[col].map(m).fillna(0.22)
            col_name = f'orig_rate_decile_{col}'
            combined[col_name] = np.floor(proba * 10).astype(int).astype(str)
            categorical_features.append(col_name)

    # 4. crosses
    cross_pairs = [
        ('Contract', 'InternetService'),
        ('PaymentMethod', 'PaperlessBilling'),
        ('MonthlyCharges_snapped_token', 'Contract'),
    ]
    for a, b in cross_pairs:
        col_name = f'cross_{a}_{b}'
        combined[col_name] = combined[a].astype(str) + '_X_' + combined[b].astype(str)
        categorical_features.append(col_name)

    combined = combined[categorical_features]

    # one shared vocabulary: factorize every field and offset it by the sizes of the previous fields
    offsets = []
    current_offset = 0
    for col in categorical_features:
        combined[col] = pd.factorize(combined[col])[0]
        offsets.append(current_offset)
        current_offset += combined[col].nunique()
    for i, col in enumerate(categorical_features):
        combined[col] = combined[col] + offsets[i]
    print(f'{len(categorical_features)} fields, {current_offset} tokens', flush=True)

    X_train = combined.iloc[:n_train].values.astype(np.int64)
    X_test = combined.iloc[n_train:].values.astype(np.int64)
    return X_train, X_test, y_train, current_offset, len(categorical_features)


class TokenDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]


class DeepFM(nn.Module):
    def __init__(self, vocab_size, num_fields, embed_dim, mlp_hidden, dropout):
        super().__init__()
        self.num_fields = num_fields
        # first order
        self.linear_emb = nn.Embedding(vocab_size, 1)
        nn.init.xavier_uniform_(self.linear_emb.weight)
        self.bias = nn.Parameter(torch.zeros(1))
        # second order (FM)
        self.fm_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.fm_emb.weight)
        # deep part
        mlp_layers = []
        input_dim = num_fields * embed_dim
        for hidden in mlp_hidden:
            mlp_layers.extend([nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        linear_out = self.linear_emb(x).sum(dim=1).squeeze(-1) + self.bias
        emb = self.fm_emb(x)  # (batch, fields, embed_dim)
        sum_of_square = emb.sum(dim=1).pow(2)
        square_of_sum = emb.pow(2).sum(dim=1)
        fm_out = 0.5 * (sum_of_square - square_of_sum).sum(dim=1)
        deep_out = self.mlp(emb.reshape(x.size(0), -1)).squeeze(-1)
        return linear_out + fm_out + deep_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    X_train, X_test, y_train, vocab_size, n_fields = build_tokens()

    criterion = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(TokenDataset(X_test), batch_size=BATCH_SIZE, shuffle=False)

    fold_ids = folds(y_train)
    for k in fold_list(args.folds):
        print(f'\n--- Fold {k + 1} ---', flush=True)
        tr_idx, va_idx = np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]
        train_loader = DataLoader(TokenDataset(X_train[tr_idx], y_train[tr_idx]), batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True)
        val_loader = DataLoader(TokenDataset(X_train[va_idx], y_train[va_idx]), batch_size=BATCH_SIZE, shuffle=False)

        model = DeepFM(vocab_size=vocab_size, num_fields=n_fields, embed_dim=EMBED_DIM, mlp_hidden=MLP_HIDDEN,
                       dropout=DROPOUT).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

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
            scheduler.step()

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
