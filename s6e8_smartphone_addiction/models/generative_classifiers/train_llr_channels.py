#!/usr/bin/env python
"""Generative Bayes channels (105 per-row log-likelihood ratios, pair lifts and context columns) read by a CatBoost
classifier.

Channels of a row, computed from a set of labelled fitting rows (n_y rows of class y, n rows in total):
  1. exact value tokens ('%.2f', raw strings, NaN -> '__NA__') of the 12 columns: log P(t | y=1) / P(t | y=0) with
     P(t | y) = (count_y(t) + s p(t)) / (n_y + s), p(t) = count(t) / n, s = 20, 5 and 1 (36 channels; unseen 0);
  2. 1-decimal tokens of the 8 numerics other than age, the same LLR with s = 50 and 20 (16);
  3. pair lifts of 16 column pairs (one pair twice, m 100 and m 20): base = sigmoid(logit p_A + logit p_B -
     logit prior), with p_A, p_B the single-column rates (count_1 + 20 prior) / (count + 20); pair rate
     (pair positives + m mean base) / (pair count + m); lift = pair rate - base (unseen pairs 0); plus the mean lift (17);
  4. exact-token LLR (s 20) of 6 budget quantities: slack, sleep + daily, utilization, weekend - daily,
     weekend - (social + gaming + work), notifications / (app opens + 0.25) (6);
  5. exact-token LLR (s 5) of the 12 columns in the original 7,500-row dataset (fixed external evidence, 12);
  6. the 9 numerics standardized with the fitting rows' mean / std (NaN -> 0) and their 9 missing flags (18).
Per outer fold: the outer-train rows get channels from a 5-fold inner cross-fit (StratifiedKFold seed 42), the held-out
and test rows from all outer-train rows. Model: CatBoostClassifier(iterations 800, learning_rate 0.04, depth 6,
l2_leaf_reg 5, random_seed 42, thread_count 8), fit on the outer-train channels, no early stopping. Test = mean of the
5 fold models.
Usage: python train_llr_channels.py [--folds 0,1,2,3,4]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds as fold_ids, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'bayes_llr_channels_catboost'

COLS_12 = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "age",
    "gender",
    "stress_level",
    "academic_work_impact"
]

PLIFT_PAIRS = [
    ("app_opens_per_day", "weekend_screen_time", 20.0),
    ("daily_screen_time_hours", "app_opens_per_day", 20.0),
    ("notifications_per_day", "weekend_screen_time", 20.0),
    ("daily_screen_time_hours", "weekend_screen_time", 20.0),
    ("daily_screen_time_hours", "notifications_per_day", 20.0),
    ("daily_screen_time_hours", "gaming_hours", 20.0),
    ("notifications_per_day", "app_opens_per_day", 100.0),
    ("notifications_per_day", "app_opens_per_day", 20.0),
    ("sleep_hours", "app_opens_per_day", 20.0),
    ("social_media_hours", "sleep_hours", 20.0),
    ("daily_screen_time_hours", "social_media_hours", 20.0),
    ("gaming_hours", "work_study_hours", 20.0),
    ("daily_screen_time_hours", "sleep_hours", 20.0),
    ("social_media_hours", "gaming_hours", 20.0),
    ("gender", "stress_level", 20.0),
    ("stress_level", "academic_work_impact", 20.0)
]

GEOMETRY = ["slack", "sleep_daily", "utilization", "weekend_minus_daily", "weekend_minus_sgw", "notif_div_app"]


def tok_exact(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.fillna("__NA__").astype(str).values
    v = s.values
    return np.array([f"{x:.2f}" if np.isfinite(x) else "__NA__" for x in v])


def tok_1dec(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.fillna("__NA__").astype(str).values
    v = s.values
    return np.array([f"{x:.1f}" if np.isfinite(x) else "__NA__" for x in v])


def _lgt(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def precompute_token_structures(df):
    tokens = {}
    for c in COLS_12:
        tokens[f"exact_{c}"] = tok_exact(df[c])
    for c in COLS_12[:8]:
        tokens[f"1dec_{c}"] = tok_1dec(df[c])
    for c1, c2, _ in PLIFT_PAIRS:
        t1 = tokens[f"exact_{c1}"]
        t2 = tokens[f"exact_{c2}"]
        tokens[f"pair_{c1}__{c2}"] = np.array([f"{a}___{b}" for a, b in zip(t1, t2)])

    d = df["daily_screen_time_hours"]
    s = df["social_media_hours"]
    g = df["gaming_hours"]
    w = df["work_study_hours"]
    sl = df["sleep_hours"]
    wk = df["weekend_screen_time"]
    notif = df["notifications_per_day"]
    app = df["app_opens_per_day"]
    geom = {
        "slack": d - (s + g + w),
        "sleep_daily": sl + d,
        "utilization": (s + g + w) / (d + 1e-5),
        "weekend_minus_daily": wk - d,
        "weekend_minus_sgw": wk - (s + g + w),
        "notif_div_app": notif / (app + 0.25)
    }
    for gname, gseries in geom.items():
        tokens[f"geom_exact_{gname}"] = tok_exact(gseries)
        tokens[f"geom_1dec_{gname}"] = tok_1dec(gseries)
    return tokens


def _llr_map(s_tr, y_trn, n1, n0, n_all, sm):
    """Token -> log P(t | y=1) / P(t | y=0) with Dirichlet smoothing s * p(t) toward the token's overall frequency."""
    c1 = s_tr[y_trn == 1].value_counts()
    c0 = s_tr[y_trn == 0].value_counts()
    cnt = s_tr.value_counts()
    mp = {}
    for t in set(s_tr.unique()):
        p_t = cnt.get(t, 1.0) / n_all
        prob1 = (c1.get(t, 0) + sm * p_t) / (n1 + sm)
        prob0 = (c0.get(t, 0) + sm * p_t) / (n0 + sm)
        mp[t] = np.log(prob1 / prob0)
    return mp


def extract_features_for_partition(tokens_tr_dict, tokens_eval_dict, trn_idx, eval_idx, y_trn, df_trn, df_eval,
                                   orig_df):
    """The 105 channels of the eval rows, computed from the fitting rows trn_idx (labels y_trn)."""
    feats = []
    n1 = (y_trn == 1).sum()
    n0 = (y_trn == 0).sum()
    n_all = len(trn_idx)
    prior = n1 / n_all
    lgt_prior = _lgt(prior)

    # 1. exact-token LLRs (s 20, 5, 1)
    p_exact_tr = {}
    p_exact_ev = {}
    for c in COLS_12:
        k = f"exact_{c}"
        s_tr = pd.Series(tokens_tr_dict[k][trn_idx])
        s_ev = pd.Series(tokens_eval_dict[k][eval_idx])
        for sm in [20.0, 5.0, 1.0]:
            feats.append(s_ev.map(_llr_map(s_tr, y_trn, n1, n0, n_all, sm)).fillna(0.0).values)
        c1 = s_tr[y_trn == 1].value_counts()
        cnt = s_tr.value_counts()
        p_map = {t: (c1.get(t, 0) + 20.0 * prior) / (cnt.get(t, 0) + 20.0) for t in set(s_tr.unique())}
        p_exact_tr[c] = s_tr.map(p_map).fillna(prior).values
        p_exact_ev[c] = s_ev.map(p_map).fillna(prior).values

    # 2. 1-decimal token LLRs (s 50, 20)
    for c in COLS_12[:8]:
        k = f"1dec_{c}"
        s_tr = pd.Series(tokens_tr_dict[k][trn_idx])
        s_ev = pd.Series(tokens_eval_dict[k][eval_idx])
        for sm in [50.0, 20.0]:
            feats.append(s_ev.map(_llr_map(s_tr, y_trn, n1, n0, n_all, sm)).fillna(0.0).values)

    # 3. pair lifts over the log-odds independence baseline
    lifts_ev = []
    for cA, cB, sm_pair in PLIFT_PAIRS:
        base_tr = _sig(_lgt(p_exact_tr[cA]) + _lgt(p_exact_tr[cB]) - lgt_prior)
        base_ev = _sig(_lgt(p_exact_ev[cA]) + _lgt(p_exact_ev[cB]) - lgt_prior)

        k_pair = f"pair_{cA}__{cB}"
        pair_tok_tr = pd.Series(tokens_tr_dict[k_pair][trn_idx])
        pair_tok_ev = pd.Series(tokens_eval_dict[k_pair][eval_idx])

        df_pair = pd.DataFrame({"pair": pair_tok_tr, "y": y_trn, "base": base_tr})
        g = df_pair.groupby("pair").agg({"y": "sum", "pair": "count", "base": "mean"})
        pair_p_map = ((g["y"] + sm_pair * g["base"]) / (g["pair"] + sm_pair)).to_dict()
        pair_p_raw = pair_tok_ev.map(pair_p_map).values
        pair_p_ev = np.where(np.isnan(pair_p_raw), base_ev, pair_p_raw)

        lift_ev = pair_p_ev - base_ev
        lifts_ev.append(lift_ev)
        feats.append(lift_ev)
    feats.append(np.mean(lifts_ev, axis=0))

    # 4. budget-geometry LLRs (s 20)
    for gname in GEOMETRY:
        k_ex = f"geom_exact_{gname}"
        s_tr = pd.Series(tokens_tr_dict[k_ex][trn_idx])
        s_ev = pd.Series(tokens_eval_dict[k_ex][eval_idx])
        feats.append(s_ev.map(_llr_map(s_tr, y_trn, n1, n0, n_all, 20.0)).fillna(0.0).values)

    # 5. original-data LLRs (s 5)
    for c in COLS_12:
        t_orig = pd.Series(tok_exact(orig_df[c]))
        s_ev = pd.Series(tokens_eval_dict[f"exact_{c}"][eval_idx])
        y_o = orig_df[TARGET].values
        mp_o = _llr_map(t_orig, y_o, (y_o == 1).sum(), (y_o == 0).sum(), len(orig_df), 5.0)
        feats.append(s_ev.map(mp_o).fillna(0.0).values)

    # 6. standardized numerics + missing flags
    for c in COLS_12[:9]:
        v_tr = df_trn[c].values
        v_ev = df_eval[c].values
        m = np.nanmean(v_tr)
        s = np.nanstd(v_tr) + 1e-6
        feats.append(np.where(np.isnan(v_ev), 0.0, (v_ev - m) / s))
        feats.append(np.isnan(v_ev).astype(float))

    return np.column_stack(feats).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    from catboost import CatBoostClassifier

    df_tr, df_te, df_orig = read_train(), read_test(), read_original()
    y = labels(df_tr[TARGET]).astype(np.int64)
    folds = fold_ids(y)
    n_test = len(df_te)
    print(f'train {len(df_tr)} rows, test {n_test} rows, original {len(df_orig)} rows', flush=True)

    print('token structures of train and test...', flush=True)
    tokens_tr = precompute_token_structures(df_tr)
    tokens_te = precompute_token_structures(df_te)

    for f in fold_list(args.folds):
        if have_part(HERE, NAME, f):
            continue
        t0 = time.time()
        trn_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        y_trn = y[trn_idx]

        # outer-train rows: 5-fold inner cross-fit
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_trn_inner = None
        for in_trn_rel, in_val_rel in skf_inner.split(np.zeros(len(trn_idx)), y_trn):
            in_trn = trn_idx[in_trn_rel]
            in_val = trn_idx[in_val_rel]
            block = extract_features_for_partition(tokens_tr, tokens_tr, in_trn, in_val, y[in_trn], df_tr.iloc[in_trn],
                                                   df_tr.iloc[in_val], df_orig)
            if X_trn_inner is None:
                X_trn_inner = np.zeros((len(trn_idx), block.shape[1]), dtype=np.float32)
            X_trn_inner[in_val_rel] = block

        # held-out and test rows: tables of all outer-train rows
        X_val_outer = extract_features_for_partition(tokens_tr, tokens_tr, trn_idx, val_idx, y_trn, df_tr.iloc[trn_idx],
                                                     df_tr.iloc[val_idx], df_orig)
        X_test_outer = extract_features_for_partition(tokens_tr, tokens_te, trn_idx, np.arange(n_test), y_trn,
                                                      df_tr.iloc[trn_idx], df_te, df_orig)
        print(f'fold {f}: {X_trn_inner.shape[1]} channels', flush=True)

        model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=5.0,
            thread_count=8,
            verbose=0,
            random_seed=42
        )
        model.fit(X_trn_inner, y_trn)
        val_probs = model.predict_proba(X_val_outer)[:, 1]
        save_part(HERE, NAME, f, val_probs, model.predict_proba(X_test_outer)[:, 1])
        print(f'fold {f}: val AUC {roc_auc_score(y[val_idx], val_probs):.6f} ({time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, NAME, folds, y)


if __name__ == '__main__':
    main()
