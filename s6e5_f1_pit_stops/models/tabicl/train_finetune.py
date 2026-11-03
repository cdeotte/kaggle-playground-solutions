#!/usr/bin/env python
"""Fine-tuned TabICL v2, eight members per fold (FinetunedTabICLClassifier of tabicl[finetune]).

Inputs: the 14 competition columns. Per outer fold, a ColumnTransformer fitted on the fold's training rows
ordinal-encodes the string columns (unknown / missing -> -1) and median-imputes the numeric ones (output: categoricals
first, then numerics, float32). Eight members (random_state 0..7) each fine-tune the TabICL v2 checkpoint
tabicl-classifier-v2-20260212.ckpt on the fold's training rows only (the library holds out its own 8% validation split
for early stopping on ROC AUC; the outer validation rows are never passed to fit) and predict the fold's validation rows
and the test rows. Member probabilities are clipped to [1e-7, 1 - 1e-7] and averaged. Test = mean of the 5 fold models.

Recipes (FINE_TUNE_PARAMS + overrides):
  tabicl_v2_finetune_8seed            raw columns
  tabicl_v2_finetune_lowcard_cats     Year, PitStop, LapNumber, Stint, TyreLife, Position, Position_Change cast to strings
                                      (ordinal codes like Driver, Compound, Race); the 4 continuous columns stay numeric
  tabicl_v2_finetune_original_append  raw columns; all 101,371 original rows appended to every fold's training rows
  tabicl_v2_finetune_bigger_context   max_data_size 14,000, 10 epochs, lr 8e-6, patience 4, no KV cache, batch size 1
Weights: downloaded from the Hugging Face model jingang/TabICL on first use.
Usage: python train_finetune.py <pair> [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import FEATURES, TARGET, folds, model_dir, read_original, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

FOLDER = 'tabicl'
N_MODELS = 8
RAW_CATS = ['Driver', 'Compound', 'Race']
LOW_CARD_CATS = ['Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position', 'Position_Change']
HIGH_CARD_NUMS = ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress']

FINE_TUNE_PARAMS = {
    'epochs': 8,
    'learning_rate': 1e-5,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'amp': True,
    'use_lr_scheduler': True,
    'warmup_proportion': 0.1,
    'n_estimators_finetune': 2,
    'n_estimators_validation': 2,
    'n_estimators_inference': 2,
    'max_data_size': 12_000,
    'finetune_ctx_query_ratio': 0.20,
    'validation_split_ratio': 0.08,
    'early_stopping': True,
    'patience': 3,
    'min_delta': 2e-5,
    'time_limit': None,
    'save_interval': 0,
    'norm_methods': None,
    'feat_shuffle_method': 'latin',
    'outlier_threshold': 4.0,
    'model_path': None,
    'allow_auto_download': True,
    'checkpoint_version': 'tabicl-classifier-v2-20260212.ckpt',
    'freeze_col': False,
    'freeze_row': False,
    'freeze_icl': False,
    'device': 'cuda',
    'verbose': True,
    'wandb_kwargs': None,
    'class_shuffle_method': 'shift',
    'softmax_temperature': 0.9,
    'average_logits': True,
    'support_many_classes': True,
    'eval_metric': 'roc_auc',
    'extra_classifier_kwargs': {
        'kv_cache': True,
        'batch_size': 2,
    },
}

RECIPES = {
    'tabicl_v2_finetune_8seed': dict(lowcard_cats=False, original=False, params={}),
    'tabicl_v2_finetune_lowcard_cats': dict(lowcard_cats=True, original=False, params={}),
    'tabicl_v2_finetune_original_append': dict(lowcard_cats=False, original=True, params={}),
    'tabicl_v2_finetune_bigger_context': dict(
        lowcard_cats=False, original=False, expandable_segments=True,
        params={'epochs': 10, 'learning_rate': 8e-6, 'max_data_size': 14_000, 'patience': 4,
                'extra_classifier_kwargs': {'kv_cache': False, 'batch_size': 1}}),
}


def cleanup_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def make_fold_encoder():
    cat_selector = make_column_selector(dtype_include=['object', 'string', 'category', 'bool', 'boolean'])
    num_selector = make_column_selector(dtype_include=np.number)
    cat_encoder = OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1,
                                 encoded_missing_value=-1)
    num_encoder = SimpleImputer(strategy='median')
    return ColumnTransformer(
        transformers=[('categorical', cat_encoder, cat_selector), ('continuous', num_encoder, num_selector)],
        remainder='drop', sparse_threshold=0.0, verbose_feature_names_out=False)


def to_float32_array(x):
    arr = np.asarray(x, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError('Non-finite values found after fold-local encoding')
    return arr


def positive_proba(model, proba):
    classes = list(model.classes_)
    return proba[:, classes.index(1) if 1 in classes else len(classes) - 1]


def lowcard_cat_frame(df):
    out = df[list(FEATURES)].copy()
    for c in RAW_CATS + LOW_CARD_CATS:
        out[c] = out[c].astype(str).fillna('__MISSING__')
    for c in HIGH_CARD_NUMS:
        out[c] = pd.to_numeric(out[c], errors='coerce').astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair', choices=list(RECIPES))
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    pair, r = args.pair, RECIPES[args.pair]
    os.environ.setdefault('WANDB_DISABLED', 'true')
    if r.get('expandable_segments'):
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    params_base = {**FINE_TUNE_PARAMS, **r['params']}

    out = model_dir(FOLDER)
    y = target().astype(int)
    fold_id = folds(y)
    todo = [k for k in fold_list(args.folds) if not have_part(out, pair, k)]
    if todo:
        from tabicl import FinetunedTabICLClassifier
        X = read_train()[list(FEATURES)]
        X_test = read_test()[list(FEATURES)]
        if r['original']:
            orig = read_original()
            X_orig, y_orig = orig[list(FEATURES)].copy(), orig[TARGET].astype(int).to_numpy()
            print(f'original rows appended to every fold: {len(X_orig):,}', flush=True)

    for k in todo:
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        X_tr_df, X_va_df, X_te_df = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy(), X_test.copy()
        y_fit, y_va = y[tr_idx], y[va_idx]
        if r['lowcard_cats']:
            X_tr_df, X_va_df, X_te_df = lowcard_cat_frame(X_tr_df), lowcard_cat_frame(X_va_df), lowcard_cat_frame(X_te_df)
        if r['original']:
            X_tr_df = pd.concat([X_tr_df, X_orig], axis=0, ignore_index=True)
            y_fit = np.concatenate([y_fit, y_orig])

        encoder = make_fold_encoder()
        X_tr = to_float32_array(encoder.fit_transform(X_tr_df))
        X_va = to_float32_array(encoder.transform(X_va_df))
        X_te = to_float32_array(encoder.transform(X_te_df))
        print(f'{pair} fold {k}: encoded shapes {X_tr.shape} {X_va.shape} {X_te.shape}', flush=True)

        va_pred = np.zeros(len(va_idx), dtype=np.float32)
        te_pred = np.zeros(len(X_te), dtype=np.float32)
        for member in range(N_MODELS):
            params = dict(params_base)
            params['random_state'] = member
            params['extra_classifier_kwargs'] = dict(params_base['extra_classifier_kwargs'])
            clf = FinetunedTabICLClassifier(**params)
            clf.fit(X_tr, y_fit)   # no outer validation rows: the fine-tuner splits its own validation set off X_tr
            p_va = np.clip(positive_proba(clf, clf.predict_proba(X_va)).astype(np.float32), 1e-7, 1 - 1e-7)
            p_te = np.clip(positive_proba(clf, clf.predict_proba(X_te)).astype(np.float32), 1e-7, 1 - 1e-7)
            va_pred += p_va / N_MODELS
            te_pred += p_te / N_MODELS
            print(f'{pair} fold {k} member {member}: AUC {roc_auc_score(y_va, p_va):.6f}', flush=True)
            del clf, p_va, p_te
            cleanup_cuda()
        save_part(out, pair, k, va_pred, te_pred)
        print(f'{pair} fold {k}: AUC {roc_auc_score(y_va, va_pred):.6f}', flush=True)
        del X_tr, X_va, X_te, encoder
        cleanup_cuda()
    assemble(out, pair, fold_id, y)


if __name__ == '__main__':
    main()
