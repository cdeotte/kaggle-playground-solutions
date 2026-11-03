#!/usr/bin/env python
"""GPU XGBoost on features from a denoising autoencoder, nearest original rows and a real-vs-synthetic discriminator.

Label-free / original-data features, built once on the train, test and original rows:
- the 11 numerics; a rule score 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] +
  [Wind_Speed_kmh > 10] - 2 [stage Harvest] - 2 [stage Sowing] - [mulch Yes], with is_low (score <= 0) and is_high
  (score > 3);
- train + test frequency and log1p of every numeric;
- per class k: the original-data share of class k in every categorical level (24 columns; unseen -> original prior);
- 6 categorical bigrams (strings).
- DAE: the 11 numerics + 8 ordinal-coded categoricals (19 columns, standardized with original-data statistics); a
  denoising autoencoder (19 -> 96 -> 8 -> 96 -> 19, BatchNorm, ReLU, dropout 0.1; Gaussian noise 0.02 and 3% masking;
  MSE, Adam lr 1e-3, batch 512, 30 epochs, seed 42) trained on the original rows gives the 8 latent coordinates and the
  mean, mean square, mean absolute and max absolute reconstruction error (12 columns).
- KNN: the 5 nearest original rows in the same 19-column space (GPU torch.cdist; for the original rows themselves the
  5 nearest other rows): nearest / mean-of-3 / mean-of-5 distance, the nearest row's class and the class shares among
  3 and 5 neighbours (10 columns).
Per fold:
- target encodings of the 8 categoricals and the 6 bigrams, per class (42 columns; smoothing 20 toward the class
  rate): out-of-fold over 5 inner folds (StratifiedKFold seed 42) for the training rows, fitted on all training rows for
  the held-out, test and original rows;
- a discriminator (XGBoost, 300 trees, depth 5, lr 0.05) trained to tell the original rows (x6) from an equal number of
  the fold's training rows, on the numeric features above; its p(original) and logit are 2 more features.
Final model: XGBoost multi:softprob on the 126 numeric columns + the 14 categoricals / bigrams as pandas categoricals
(enable_categorical), depth 5, lr 0.03, min_child_weight 10, lambda 3, subsample 0.8, colsample_bytree 0.5, early
stopping 300 (mlogloss) on the held-out fold, trained on the fold's training rows only. Seed 42 (python / numpy /
torch) at the start and again before the DAE; the discriminator samples draw from the numpy stream fold after fold.
Test = mean of the 5 fold models.
Usage: python train_dae_xgb.py [--folds 0,1,2,3,4]"""
import argparse
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, save_part  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'dae_knn_discriminator_xgb_b'
MODEL_DIR = model_dir('neural_nets')
SEED = 42
INNER_FOLDS = 5
N_REP = 6  # copies of the original rows on the "real" side of the discriminator

DAE_LATENT = 8
DAE_HIDDEN = 96
DAE_EPOCHS = 30
DAE_LR = 1e-3
DAE_NOISE = 0.02
DAE_MASK = 0.03
DAE_BATCH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUMS = ['Soil_pH', 'Organic_Carbon', 'Electrical_Conductivity', 'Sunlight_Hours', 'Wind_Speed_kmh', 'Field_Area_hectare',
        'Temperature_C', 'Soil_Moisture', 'Humidity', 'Previous_Irrigation_mm', 'Rainfall_mm']
CATS = list(CATEGORICALS)
BIGRAMS = [('Crop_Type', 'Season'), ('Crop_Type', 'Region'), ('Season', 'Region'), ('Soil_Type', 'Crop_Type'),
           ('Crop_Growth_Stage', 'Mulching_Used'), ('Irrigation_Type', 'Water_Source')]
BIGRAM_COLS = [f'{a}__{b}' for a, b in BIGRAMS]
ALL_CATS_FE = CATS + BIGRAM_COLS  # 14
BASE_NUMS = (NUMS + [f'FREQ_{c}' for c in NUMS] + [f'LOG1P_{c}' for c in NUMS]
             + [f'ORIG_{c}_k{k}' for c in CATS for k in range(3)] + ['formula_score', 'is_low', 'is_high'])  # 60

XGB_DISC_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5, min_child_weight=5, reg_lambda=2.0, subsample=0.8,
    colsample_bytree=0.7, objective='binary:logistic', device='cuda', tree_method='hist', random_state=SEED,
    verbosity=0,
)
XGB_FINAL_PARAMS = dict(
    n_estimators=100_000, learning_rate=0.03, max_depth=5, min_child_weight=10, reg_lambda=3.0, subsample=0.8,
    colsample_bytree=0.5, objective='multi:softprob', num_class=3, eval_metric='mlogloss', device='cuda',
    tree_method='hist', enable_categorical=True, early_stopping_rounds=300, random_state=SEED, verbosity=0,
)


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def add_formula(df):
    score = (
        (df['Soil_Moisture'] < 25) * 2
        + (df['Rainfall_mm'] < 300) * 2
        + (df['Temperature_C'] > 30) * 1
        + (df['Wind_Speed_kmh'] > 10) * 1
        - (df['Crop_Growth_Stage'] == 'Harvest') * 2
        - (df['Crop_Growth_Stage'] == 'Sowing') * 2
        - (df['Mulching_Used'] == 'Yes') * 1
    ).astype(np.float32)
    df['formula_score'] = score
    df['is_low'] = (score <= 0).astype(np.float32)
    df['is_high'] = (score > 3).astype(np.float32)


class DAEModel(nn.Module):
    def __init__(self, d, h, z, noise_std, mask_prob):
        super().__init__()
        self.noise = noise_std
        self.mask = mask_prob
        self.enc = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, d))

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * self.noise
            x = x * (torch.rand_like(x) > self.mask).float()
        z = self.enc(x)
        return self.dec(z), z


def train_dae_and_extract(X_orig, query_list):
    """Train the DAE on the original rows; latent codes + reconstruction-error statistics for every query array."""
    in_dim = X_orig.shape[1]
    Xo = torch.tensor(X_orig, dtype=torch.float32).to(DEVICE)
    dl = DataLoader(TensorDataset(Xo), batch_size=DAE_BATCH, shuffle=True)
    model = DAEModel(in_dim, DAE_HIDDEN, DAE_LATENT, DAE_NOISE, DAE_MASK).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=DAE_LR)
    crit = nn.MSELoss()
    for _ in range(DAE_EPOCHS):
        model.train()
        for (xb,) in dl:
            opt.zero_grad()
            recon, _ = model(xb)
            crit(recon, xb).backward()
            opt.step()

    model.eval()
    results = []
    for Xq in query_list:
        dl2 = DataLoader(TensorDataset(torch.tensor(Xq, dtype=torch.float32).to(DEVICE)), batch_size=4096, shuffle=False)
        zs, rs = [], []
        with torch.no_grad():
            for (xb,) in dl2:
                r, z = model(xb)
                zs.append(z.cpu().numpy())
                rs.append(r.cpu().numpy())
        z_arr = np.vstack(zs).astype(np.float32)
        err = Xq - np.vstack(rs).astype(np.float32)
        results.append(np.hstack([
            z_arr,
            err.mean(axis=1, keepdims=True),
            (err ** 2).mean(axis=1, keepdims=True),
            np.abs(err).mean(axis=1, keepdims=True),
            np.abs(err).max(axis=1, keepdims=True),
        ]).astype(np.float32))
    return results


def gpu_knn(Xq_sc, Xo_sc, k=5, chunk=4096):
    Xq = torch.tensor(Xq_sc, dtype=torch.float32).to(DEVICE)
    Xo = torch.tensor(Xo_sc, dtype=torch.float32).to(DEVICE)
    n = len(Xq)
    dists = np.zeros((n, k), dtype=np.float32)
    idxs = np.zeros((n, k), dtype=np.int32)
    for i in range(0, n, chunk):
        tk = torch.topk(torch.cdist(Xq[i:i + chunk], Xo, p=2), k, dim=1, largest=False)
        dists[i:i + chunk] = tk.values.cpu().numpy()
        idxs[i:i + chunk] = tk.indices.cpu().numpy().astype(np.int32)
    return dists, idxs


def build_knn_feats(dists5, idxs5, y_ref):
    """dists5 / idxs5: (n, 5) -> (n, 10) features."""
    nn3_lbl = y_ref[idxs5[:, :3]]
    nn5_lbl = y_ref[idxs5]
    return np.column_stack([
        dists5[:, 0],
        dists5[:, :3].mean(axis=1),
        dists5.mean(axis=1),
        y_ref[idxs5[:, 0]].astype(np.float32),
        (nn3_lbl == 0).mean(axis=1),
        (nn3_lbl == 1).mean(axis=1),
        (nn3_lbl == 2).mean(axis=1),
        (nn5_lbl == 0).mean(axis=1),
        (nn5_lbl == 1).mean(axis=1),
        (nn5_lbl == 2).mean(axis=1),
    ]).astype(np.float32)


def make_disc_input(base_df, te_arr, dae_arr, knn_arr):
    return np.hstack([base_df[BASE_NUMS].astype(np.float32).values, te_arr, dae_arr, knn_arr]).astype(np.float32)


def get_disc_feats(disc_clf, X_base):
    p = disc_clf.predict_proba(X_base)[:, 1].astype(np.float32)
    logit = np.log(p.clip(1e-7, 1 - 1e-7) / (1 - p.clip(1e-7, 1 - 1e-7)))
    return np.column_stack([p, logit]).astype(np.float32)


def build_final_df(base_df, te_arr, dae_arr, knn_arr, disc_arr, all_num_cols):
    num_vals = np.hstack([base_df[BASE_NUMS].astype(np.float32).values, te_arr, dae_arr, knn_arr, disc_arr])
    df_out = pd.DataFrame(num_vals.astype(np.float32), columns=all_num_cols)
    for c in ALL_CATS_FE:
        df_out[c] = base_df[c].values
        df_out[c] = df_out[c].astype('category')
    return df_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()

    seed_everything(SEED)
    train, test, orig = read_train(), read_test(), read_original()
    y = train[TARGET].map(TARGET_MAP).astype(np.int32).values
    y_orig = orig[TARGET].map(TARGET_MAP).values
    print(f'train={train.shape}, test={test.shape}, orig={orig.shape}', flush=True)

    for df in [train, test, orig]:
        add_formula(df)
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    for c in NUMS:
        freq = combined[c].value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{c}'] = df[c].map(freq).fillna(0).astype(np.float32)
    for c in NUMS:
        for df in [train, test, orig]:
            df[f'LOG1P_{c}'] = np.log1p(df[c].clip(lower=0)).astype(np.float32)
    for c in CATS:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g: (g[TARGET].map(TARGET_MAP) == k).mean())
            gm = (y_orig == k).mean()
            for df in [train, test, orig]:
                df[f'ORIG_{c}_k{k}'] = df[c].map(m).fillna(gm).astype(np.float32)
    for a, b in BIGRAMS:
        for df in [train, test, orig]:
            df[f'{a}__{b}'] = df[a].astype(str) + '__' + df[b].astype(str)

    # numerics + ordinal-coded categoricals, standardized with original-data statistics (DAE / KNN space)
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(pd.concat([orig[CATS], train[CATS], test[CATS]], axis=0, ignore_index=True).astype(str))

    def to_raw(df):
        cat_enc = np.clip(oe.transform(df[CATS].astype(str)), 0, None).astype(np.float32)
        return np.hstack([df[NUMS].astype(np.float32).values, cat_enc])

    sc = StandardScaler()
    sc.fit(to_raw(orig))
    X_orig_sc = sc.transform(to_raw(orig)).astype(np.float32)
    X_train_sc = sc.transform(to_raw(train)).astype(np.float32)
    X_test_sc = sc.transform(to_raw(test)).astype(np.float32)

    print('Training the DAE on the original rows ...', flush=True)
    seed_everything(SEED)
    dae_train, dae_test, dae_orig = train_dae_and_extract(X_orig_sc, [X_train_sc, X_test_sc, X_orig_sc])

    print('KNN: train / test -> original (k = 5), original -> original (k = 6, skipping self) ...', flush=True)
    kd_tr, ki_tr = gpu_knn(X_train_sc, X_orig_sc, k=5)
    kd_te, ki_te = gpu_knn(X_test_sc, X_orig_sc, k=5)
    knn_tr = build_knn_feats(kd_tr, ki_tr, y_orig)
    knn_te = build_knn_feats(kd_te, ki_te, y_orig)
    kd_oo, ki_oo = gpu_knn(X_orig_sc, X_orig_sc, k=6)
    knn_oo = build_knn_feats(kd_oo[:, 1:], ki_oo[:, 1:], y_orig)

    n_te = len(ALL_CATS_FE) * 3  # 42
    all_num_cols = (BASE_NUMS + [f'TE_{c}_k{k}' for c in ALL_CATS_FE for k in range(3)]
                    + [f'dae_{i}' for i in range(dae_train.shape[1])] + [f'knn_{i}' for i in range(knn_tr.shape[1])]
                    + ['disc_p_real', 'disc_logit'])
    print(f'{len(all_num_cols)} numeric columns, {len(ALL_CATS_FE)} categorical columns', flush=True)
    test_r = test.reset_index(drop=True)

    fold_ids = folds(y)
    for fold in fold_list(args.folds):
        print(f'\n--- Fold {fold + 1}/5 ---', flush=True)
        tr_idx, val_idx = np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]
        X_tr = train.iloc[tr_idx].reset_index(drop=True)
        X_val = train.iloc[val_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        # nested per-class target encoding for the fold's training rows
        inner_skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        te_tr_arr = np.zeros((len(tr_idx), n_te), dtype=np.float32)
        col_i = 0
        for c in ALL_CATS_FE:
            for k in range(3):
                smooth = 20.0
                oof_te = np.full(len(tr_idx), (y_tr == k).mean(), dtype=np.float32)
                for inn_tr, inn_va in inner_skf.split(X_tr[CATS[:1]], y_tr):
                    ig = (y_tr[inn_tr] == k).mean()
                    stats = pd.Series(y_tr[inn_tr] == k).groupby(X_tr.iloc[inn_tr][c].values).agg(['sum', 'count'])
                    enc = (stats['sum'] + smooth * ig) / (stats['count'] + smooth)
                    oof_te[inn_va] = X_tr.iloc[inn_va][c].map(enc).fillna(ig).values.astype(np.float32)
                te_tr_arr[:, col_i] = oof_te
                col_i += 1

        # target encoding fitted on all training rows for the held-out / test / original rows
        te_val_arr = np.zeros((len(val_idx), n_te), dtype=np.float32)
        te_test_arr = np.zeros((len(test), n_te), dtype=np.float32)
        te_orig_arr = np.zeros((len(orig), n_te), dtype=np.float32)
        col_i = 0
        for c in ALL_CATS_FE:
            for k in range(3):
                gm = (y_tr == k).mean()
                smooth = 20.0
                stats = pd.Series(y_tr == k).groupby(X_tr[c].values).agg(['sum', 'count'])
                enc = (stats['sum'] + smooth * gm) / (stats['count'] + smooth)
                te_val_arr[:, col_i] = X_val[c].map(enc).fillna(gm).values.astype(np.float32)
                te_test_arr[:, col_i] = test[c].map(enc).fillna(gm).values.astype(np.float32)
                te_orig_arr[:, col_i] = orig[c].map(enc).fillna(gm).values.astype(np.float32)
                col_i += 1

        # discriminator: original rows (x N_REP) vs an equal-size sample of the fold's training rows
        X_real_disc = make_disc_input(orig, te_orig_arr, dae_orig, knn_oo)
        X_synth_disc = make_disc_input(X_tr, te_tr_arr, dae_train[tr_idx], knn_tr[tr_idx])
        n_real = len(X_real_disc) * N_REP
        ri = np.tile(np.arange(len(X_real_disc)), N_REP)
        si = np.random.choice(len(X_synth_disc), n_real, replace=False)
        X_disc = np.vstack([X_real_disc[ri], X_synth_disc[si]])
        y_disc = np.concatenate([np.ones(n_real), np.zeros(n_real)])
        perm = np.random.permutation(len(X_disc))
        disc_clf = XGBClassifier(**XGB_DISC_PARAMS)
        disc_clf.fit(X_disc[perm], y_disc[perm], verbose=False)

        disc_tr_f = get_disc_feats(disc_clf, X_synth_disc)
        disc_val_f = get_disc_feats(disc_clf, make_disc_input(X_val, te_val_arr, dae_train[val_idx], knn_tr[val_idx]))
        disc_te_f = get_disc_feats(disc_clf, make_disc_input(test_r, te_test_arr, dae_test, knn_te))

        X_tr_fin = build_final_df(X_tr, te_tr_arr, dae_train[tr_idx], knn_tr[tr_idx], disc_tr_f, all_num_cols)
        X_val_fin = build_final_df(X_val, te_val_arr, dae_train[val_idx], knn_tr[val_idx], disc_val_f, all_num_cols)
        X_te_fin = build_final_df(test_r, te_test_arr, dae_test, knn_te, disc_te_f, all_num_cols)

        final_clf = XGBClassifier(**XGB_FINAL_PARAMS)
        final_clf.fit(X_tr_fin, y_tr, eval_set=[(X_val_fin, y_val)], verbose=500)
        p_val = final_clf.predict_proba(X_val_fin).astype(np.float32)
        p_test = final_clf.predict_proba(X_te_fin).astype(np.float32)
        print(f'Fold {fold + 1}/5  BAC={balanced_accuracy_score(y_val, np.argmax(p_val, axis=1)):.5f}  '
              f'best_iter={final_clf.best_iteration}', flush=True)
        save_part(MODEL_DIR, PAIR, fold, p_val, p_test)
    assemble(MODEL_DIR, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
