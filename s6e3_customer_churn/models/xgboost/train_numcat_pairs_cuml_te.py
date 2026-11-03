#!/usr/bin/env python
"""GPU XGBoost with cuML target encodings of numeric x categorical pairs (pair xgb_numcat_pair_cuml_te_digits).

Features: the 16 categoricals (SeniorCitizen included) as native XGBoost categories, tenure / MonthlyCharges /
TotalCharges, and digit features (every integer digit position of the 3 numerics; the first 2 decimal digits of the
charges). Per outer fold: cuML TargetEncoder (smooth 10) means of the 48 numeric x categorical keys (exact numeric value
+ category) and of the 3 numerics as keys, out-of-fold over an inner StratifiedKFold(5, shuffle=True, random_state=42 +
fold number 1..5) for the training rows and fitted on all training rows for the validation / test rows; counts of the
3 numeric values on the training rows. Competition rows only.
Model: depth 3, lr 0.02, subsample 0.8, colsample_bytree 0.8, up to 20,000 rounds, early stopping 200 on the
validation fold, random_state 42 (default max_bin). Test = mean of the 5 fold models.
Usage: python train_numcat_pairs_cuml_te.py"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.simplefilter('ignore')

PAIR = 'xgb_numcat_pair_cuml_te_digits'
N_SPLITS = 5
INNER_TE_SPLITS = 5
SEED = 42
TE_SMOOTH = 10.0

CATS = ['gender', 'Partner', 'Dependents', 'SeniorCitizen', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod']
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
DIGIT_BASE_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

XGB_PARAMS = {
    'n_estimators': 20000,
    'learning_rate': 0.02,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': SEED,
    'early_stopping_rounds': 200,
    'device': 'cuda',
    'enable_categorical': True,
}


def log(msg):
    print(msg, flush=True)


def coerce_numeric(df, cols):
    for c in cols:
        if df[c].dtype == 'O' or str(df[c].dtype).startswith('string'):
            s = df[c].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
            df[c] = pd.to_numeric(s, errors='coerce')
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def add_digit_features(combined_df, base_cols, max_dec_digits=2):
    digit_cols = []
    out = combined_df.copy()
    for col in base_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce')
        col_vals = out[col].copy()
        fill_val = float(col_vals.median()) if col_vals.notna().any() else 0.0
        col_vals = col_vals.fillna(fill_val)
        max_val = float(col_vals.abs().max())
        max_int_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1

        new_cols = {}
        for i in range(max_int_digits):
            new_cols[f'{col}_digit_pos_{i + 1}'] = ((col_vals // (10 ** i)) % 10).astype('int16')
        if (col_vals % 1 != 0).any():
            for i in range(1, max_dec_digits + 1):
                new_cols[f'{col}_digit_dec_{i}'] = ((col_vals * (10 ** i)).round().astype(int) % 10).astype('int16')
        digit_cols += list(new_cols)
        if new_cols:
            out = pd.concat([out, pd.DataFrame(new_cols)], axis=1)
    return out, digit_cols


def numeric_to_key(s, decimals=6):
    x = pd.to_numeric(s, errors='coerce').astype(np.float64).fillna(-999999.123456)
    out = x.map(lambda v: f'{v:.{decimals}f}').str.rstrip('0').str.rstrip('.')
    return out.replace('', '0').astype(str)


def make_te_source_frames(train_df, test_df, cats, nums):
    tr_cat_str = {c: train_df[c].astype(str).fillna('__MISSING__') for c in cats}
    te_cat_str = {c: test_df[c].astype(str).fillna('__MISSING__') for c in cats}
    tr_num_key = {n: numeric_to_key(train_df[n]) for n in nums}
    te_num_key = {n: numeric_to_key(test_df[n]) for n in nums}

    tr_blocks, te_blocks, pair_cols, num_te_cols = {}, {}, [], []
    for n in nums:
        for c in cats:
            col = f'TEPAIR__{n}__X__{c}'
            tr_blocks[col] = (tr_num_key[n] + '__' + tr_cat_str[c]).astype(str)
            te_blocks[col] = (te_num_key[n] + '__' + te_cat_str[c]).astype(str)
            pair_cols.append(col)
    for n in nums:
        col = f'TENUM__{n}'
        tr_blocks[col] = tr_num_key[n].astype(str)
        te_blocks[col] = te_num_key[n].astype(str)
        num_te_cols.append(col)
    return pd.DataFrame(tr_blocks), pd.DataFrame(te_blocks), pair_cols, num_te_cols


def te_output_to_numpy(x, expected_len):
    if hasattr(x, 'to_pandas'):
        x = x.to_pandas()
    if isinstance(x, pd.DataFrame):
        arr = x.iloc[:, 0].to_numpy()
    elif isinstance(x, pd.Series):
        arr = x.to_numpy()
    elif hasattr(x, 'get'):
        arr = x.get()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(f'Unexpected TE output length {len(arr)} != expected {expected_len}')
    return arr.astype(np.float32)


def cuml_target_encode(train_col, y_train, other_cols, smooth=10.0):
    """Fits a cuML TargetEncoder on one string column and transforms each column of `other_cols`."""
    import cudf
    from cuml.preprocessing import TargetEncoder

    tr_gdf = cudf.from_pandas(pd.DataFrame({'x': train_col.astype(str).fillna('__MISSING__')}))
    enc = TargetEncoder(smooth=smooth)
    enc.fit(tr_gdf, cudf.Series(np.asarray(y_train).astype(np.float32)))
    out = []
    for col in other_cols:
        gdf = cudf.from_pandas(pd.DataFrame({'x': col.astype(str).fillna('__MISSING__')}))
        out.append(te_output_to_numpy(enc.transform(gdf), len(col)))
    return out


def make_leak_free_te_features(X_tr_raw, y_tr, X_va_raw, X_te_raw, outer_seed, inner_splits=5, smooth=10.0):
    """Training rows: inner out-of-fold encodings; validation / test rows: encoder fitted on all training rows."""
    cols = list(X_tr_raw.columns)
    tr_enc = pd.DataFrame(np.zeros((len(X_tr_raw), len(cols)), dtype=np.float32), columns=cols)
    va_enc = pd.DataFrame(np.zeros((len(X_va_raw), len(cols)), dtype=np.float32), columns=cols)
    te_enc = pd.DataFrame(np.zeros((len(X_te_raw), len(cols)), dtype=np.float32), columns=cols)
    inner_skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=outer_seed)

    for j, col in enumerate(cols):
        if (j + 1) % 10 == 0 or j == 0 or j == len(cols) - 1:
            log(f'    TE col {j + 1}/{len(cols)}: {col}')
        x_col_full = X_tr_raw[col].reset_index(drop=True)
        tr_col_enc = np.zeros(len(X_tr_raw), dtype=np.float32)
        for itr_idx, iva_idx in inner_skf.split(X_tr_raw, y_tr):
            tr_col_enc[iva_idx] = cuml_target_encode(x_col_full.iloc[itr_idx], y_tr[itr_idx],
                                                     [x_col_full.iloc[iva_idx]], smooth=smooth)[0]
        tr_enc[col] = tr_col_enc
        va_enc[col], te_enc[col] = cuml_target_encode(x_col_full, y_tr, [X_va_raw[col].reset_index(drop=True),
                                                                         X_te_raw[col].reset_index(drop=True)],
                                                      smooth=smooth)
        gc.collect()
    return tr_enc.astype(np.float32), va_enc.astype(np.float32), te_enc.astype(np.float32)


def make_count_encoding_features(train_df, valid_df, test_df, nums):
    tr_out, va_out, te_out = {}, {}, {}
    for n in nums:
        col = f'CNTNUM__{n}'
        tr_key, va_key, te_key = numeric_to_key(train_df[n]), numeric_to_key(valid_df[n]), numeric_to_key(test_df[n])
        cnt = tr_key.value_counts(dropna=False)
        tr_out[col] = tr_key.map(cnt).fillna(0).astype(np.float32)
        va_out[col] = va_key.map(cnt).fillna(0).astype(np.float32)
        te_out[col] = te_key.map(cnt).fillna(0).astype(np.float32)
    return (pd.DataFrame(tr_out, index=train_df.index).astype(np.float32),
            pd.DataFrame(va_out, index=valid_df.index).astype(np.float32),
            pd.DataFrame(te_out, index=test_df.index).astype(np.float32))


def build_features():
    train = read_train()
    test = read_test()
    train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype('int8')
    train = coerce_numeric(train, NUMS)
    test = coerce_numeric(test, NUMS)
    for c in NUMS:
        med = train[c].median() if train[c].notna().any() else 0.0
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

    combined_df = pd.concat([train, test], axis=0, ignore_index=True)
    for c in CATS:
        combined_df[c] = combined_df[c].astype(str).fillna('__MISSING__').astype('category')
    combined_df, digit_cols = add_digit_features(combined_df, DIGIT_BASE_COLS, max_dec_digits=2)
    log(f'{len(digit_cols)} digit features: {digit_cols}')

    train2 = combined_df.iloc[:len(train)].reset_index(drop=True).copy()
    test2 = combined_df.iloc[len(train):].reset_index(drop=True).copy()
    train_te_raw, test_te_raw, pair_raw_cols, num_te_raw_cols = make_te_source_frames(train2, test2, CATS, NUMS)
    te_names = [c.replace('TEPAIR__', 'TE__') for c in pair_raw_cols] + [c.replace('TENUM__', 'TE__')
                                                                          for c in num_te_raw_cols]
    base_features = CATS + NUMS + digit_cols
    y = train2[TARGET].values.astype(np.int8)
    log(f'{len(base_features)} base features, {len(te_names)} target-encoded keys, {len(NUMS)} counts')
    return train2, test2, train_te_raw, test_te_raw, te_names, base_features, digit_cols, y


def run_cv(train2, test2, train_te_raw, test_te_raw, te_names, base_features, digit_cols, y, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train2), dtype=np.float64)
    pred_test = np.zeros(len(test2), dtype=np.float64)
    cnt_cols = [f'CNTNUM__{n}' for n in NUMS]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train2[base_features], y), start=1):
        X_tr_base = train2.loc[tr_idx, base_features].reset_index(drop=True).copy()
        X_va_base = train2.loc[va_idx, base_features].reset_index(drop=True).copy()
        X_te_base = test2[base_features].reset_index(drop=True).copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        tr_te_enc, va_te_enc, te_te_enc = make_leak_free_te_features(
            train_te_raw.iloc[tr_idx].reset_index(drop=True), y_tr, train_te_raw.iloc[va_idx].reset_index(drop=True),
            test_te_raw.reset_index(drop=True), outer_seed=SEED + fold, inner_splits=INNER_TE_SPLITS, smooth=TE_SMOOTH)
        for enc in (tr_te_enc, va_te_enc, te_te_enc):
            enc.columns = te_names

        tr_cnt, va_cnt, te_cnt = make_count_encoding_features(train2.iloc[tr_idx].reset_index(drop=True),
                                                              train2.iloc[va_idx].reset_index(drop=True),
                                                              test2.reset_index(drop=True), NUMS)
        X_tr = pd.concat([X_tr_base, tr_te_enc, tr_cnt], axis=1)
        X_va = pd.concat([X_va_base, va_te_enc, va_cnt], axis=1)
        X_te = pd.concat([X_te_base, te_te_enc, te_cnt], axis=1)
        for c in CATS:
            for df in (X_tr, X_va, X_te):
                df[c] = df[c].astype('category')
        for c in te_names + cnt_cols + NUMS + digit_cols:
            for df in (X_tr, X_va, X_te):
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float32)

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
        va_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        pred_test += model.predict_proba(X_te)[:, 1] / n_splits
        log(f'{PAIR} fold {fold}: AUC {roc_auc_score(y_va, va_pred):.6f}, best_iteration '
            f'{getattr(model, "best_iteration", None)}')
        del X_tr_base, X_va_base, X_te_base, tr_te_enc, va_te_enc, te_te_enc, X_tr, X_va, X_te, model
        gc.collect()

    log(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}')
    return oof, pred_test


def main():
    data = build_features()
    oof, test_pred = run_cv(*data, N_SPLITS)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
