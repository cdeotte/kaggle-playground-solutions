#!/usr/bin/env python
"""The 99-column classical feature stack built with cuML / cuDF / CuPy, one file pair per outer fold:
cache/cf_f<k>_{tr,te}.parquet (train rows, test rows).

Every label-derived block is nested inside the outer fold: outer-train rows get inner out-of-fold values (5 inner folds,
seed 777), validation and test rows get the value of a fit on all outer-train rows. cuML's TargetEncoder does this
natively (fit_transform = out-of-fold, transform = full fit); the other label-derived blocks use an explicit inner
StratifiedKFold(5, shuffle=True, random_state=777). Target-free blocks use train + test.

Blocks (columns):
  raw 12        the 12 columns (categoricals as codes of their exact-value tokens)
  na 9          missingness flags of the numerics
  geo 8         sum of the 3 components, slack, 3 shares of daily, weekend - daily, sleep + daily, notifications / app opens
  cnt 12        log1p value counts of the exact-value tokens (train + test)
  te 24         TargetEncoder of each token, smoothing 1 and 20
  bank 10       TargetEncoder of 10 token pairs (combination mode, smoothing 5)
  woe 1         sum over the 12 tokens of nested WOE (smoothing 5)
  mrg_lr 1      nested LogisticRegression(C 1) decision function on the standardised raw matrix
  cte 2         TargetEncoder (smoothing 10) of KMeans clusters (k 64, 256) of the standardised raw matrix
  knn200 1      nested KNeighborsClassifier(200) probability on the standardised raw matrix
  sf2d 2        nested Gaussian-kernel (bandwidth 0.25) label smoother over the 100 nearest neighbours in the
                (daily, social) plane + the mean distance to the 10 nearest
  embpca 16     PCA of the standardised 384-d row embedding of the masked language model (row_emb384.npy, ../density_features)
  og_knn100 1   KNeighborsClassifier(100) fitted on the 7,500 original rows (standardised with original statistics)
Usage: python build_cuml_features.py [--folds 0,1,2,3,4]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CATEGORICALS, NUMERICS, TARGET, cache_dir, folds as fold_ids, read_original, read_test, read_train  # noqa: E402
from common.pairs import fold_list  # noqa: E402

NUMS, CATS = list(NUMERICS), list(CATEGORICALS)
FEATS = NUMS + CATS
CACHE = cache_dir('cuml_density_models')
TE_SM = [1.0, 20.0]
PAIRS = [('daily_screen_time_hours', 'social_media_hours'),
         ('daily_screen_time_hours', 'app_opens_per_day'),
         ('social_media_hours', 'weekend_screen_time'),
         ('daily_screen_time_hours', 'notifications_per_day'),
         ('notifications_per_day', 'app_opens_per_day'),
         ('social_media_hours', 'gaming_hours'),
         ('daily_screen_time_hours', 'sleep_hours'),
         ('weekend_screen_time', 'app_opens_per_day'),
         ('age', 'daily_screen_time_hours'),
         ('stress_level', 'daily_screen_time_hours')]
t0 = time.time()


def log(m):
    print(f'[{time.time() - t0:7.1f}s] {m}', flush=True)


def val_str(s):
    """Exact-value string tokens; NaN -> '__NA__'."""
    return s.astype(object).where(s.notna(), '__NA__').astype(str)


def timed(name, fn):
    t = time.time()
    out = fn()
    log(f'  {name}: {time.time() - t:.2f}s')
    return out


def build_fold(FOLD, train, test):
    import cudf
    import cupy as cp
    from cuml.cluster import KMeans
    from cuml.decomposition import PCA
    from cuml.linear_model import LogisticRegression
    from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
    from cuml.preprocessing import TargetEncoder

    y = train[TARGET].values.astype(np.float32)
    folds = fold_ids(y)
    trn_idx = np.where(folds != FOLD)[0]
    val_idx = np.where(folds == FOLD)[0]
    n_tr, n_te = len(train), len(test)
    log(f'fold {FOLD}: outer-train {len(trn_idx):,}  outer-val {len(val_idx):,}')

    # exact-value string tokens, factorised jointly over train + test (target-free)
    tok = {}
    for c in FEATS:
        s = pd.concat([val_str(train[c]), val_str(test[c])], ignore_index=True)
        tok[c] = pd.factorize(s)[0].astype(np.int32)

    OUT_TR, OUT_TE, NAMES = [], [], []

    def emit(name, arr_tr, arr_te):
        OUT_TR.append(np.asarray(arr_tr, np.float32).reshape(len(arr_tr), -1))
        OUT_TE.append(np.asarray(arr_te, np.float32).reshape(len(arr_te), -1))
        k = OUT_TR[-1].shape[1]
        NAMES.extend([name] if k == 1 else [f'{name}_{i}' for i in range(k)])

    gy_tr = cudf.Series(y[trn_idx])

    # ---- 1. raw + geometry
    X = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
    g = cudf.DataFrame({c: X[c].astype('float32') if c in NUMS else cudf.Series(tok[c]).astype('float32') for c in FEATS})
    for c in FEATS:
        emit(f'raw_{c}', g[c].to_numpy()[:n_tr], g[c].to_numpy()[n_tr:])
    for c in NUMS:
        na = X[c].isna().values.astype(np.float32)
        emit(f'na_{c}', na[:n_tr], na[n_tr:])
    d, s_, gm, w = X['daily_screen_time_hours'], X['social_media_hours'], X['gaming_hours'], X['work_study_hours']
    geo = {'sgw': (s_ + gm + w), 'slack': d - (s_ + gm + w),
           'share_s': s_ / d.clip(lower=.1), 'share_g': gm / d.clip(lower=.1), 'share_w': w / d.clip(lower=.1),
           'wk_minus_d': X['weekend_screen_time'] - d,
           'sleep_plus_d': X['sleep_hours'] + d,
           'notif_per_app': X['notifications_per_day'] / X['app_opens_per_day'].clip(lower=1)}
    for k, v in geo.items():
        v = v.values.astype(np.float32)
        emit(f'geo_{k}', v[:n_tr], v[n_tr:])
    log(f'raw + geometry: {len(NAMES)} cols')

    # ---- 2. value counts (target-free)
    def counts():
        for c in FEATS:
            vc = cudf.Series(tok[c]).value_counts()
            m = cudf.Series(np.arange(tok[c].max() + 1)).map(vc).fillna(0).to_numpy()
            v = np.log1p(m[tok[c]]).astype(np.float32)
            emit(f'cnt_{c}', v[:n_tr], v[n_tr:])
    timed('cnt', counts)

    # ---- 3. single-token target encodings (cuML TargetEncoder, smoothing 1 and 20)
    def te_single():
        for c in FEATS:
            gtr = cudf.DataFrame({c: tok[c][:n_tr][trn_idx]})
            gva = cudf.DataFrame({c: tok[c][:n_tr][val_idx]})
            gte = cudf.DataFrame({c: tok[c][n_tr:]})
            for sm in TE_SM:
                te = TargetEncoder(n_folds=5, smooth=sm, seed=777, split_method='random')
                oof = cp.asarray(te.fit_transform(gtr, gy_tr)).ravel()
                v_tr = np.zeros(n_tr, np.float32)
                v_tr[trn_idx] = cp.asnumpy(oof)
                v_tr[val_idx] = cp.asnumpy(cp.asarray(te.transform(gva)).ravel())
                emit(f'te{int(sm)}_{c}', v_tr, cp.asnumpy(cp.asarray(te.transform(gte)).ravel()))
    timed('te_single', te_single)

    # ---- 4. pair banks (TargetEncoder on the token combination)
    def te_pairs():
        for a, b in PAIRS:
            cols = {a: tok[a], b: tok[b]}
            gtr = cudf.DataFrame({k: v[:n_tr][trn_idx] for k, v in cols.items()})
            gva = cudf.DataFrame({k: v[:n_tr][val_idx] for k, v in cols.items()})
            gte = cudf.DataFrame({k: v[n_tr:] for k, v in cols.items()})
            te = TargetEncoder(n_folds=5, smooth=5.0, seed=777, split_method='random', multi_feature_mode='combination')
            oof = cp.asarray(te.fit_transform(gtr, gy_tr)).ravel()
            v_tr = np.zeros(n_tr, np.float32)
            v_tr[trn_idx] = cp.asnumpy(oof)
            v_tr[val_idx] = cp.asnumpy(cp.asarray(te.transform(gva)).ravel())
            emit(f'bank_{a[:6]}_{b[:6]}', v_tr, cp.asnumpy(cp.asarray(te.transform(gte)).ravel()))
    timed('te_pairs', te_pairs)

    # ---- 5. WOE sum (CuPy bincount)
    def woe():
        pri = float(y[trn_idx].mean())
        lp = np.log(pri / (1 - pri))
        acc_tr = np.zeros(n_tr, np.float32)
        acc_te = np.zeros(n_te, np.float32)
        for c in FEATS:
            K = tok[c].max() + 1
            code = tok[c][:n_tr]
            w_tr = np.zeros(n_tr, np.float32)
            for i_fit, i_app in StratifiedKFold(5, shuffle=True, random_state=777).split(np.zeros(len(trn_idx)), y[trn_idx]):
                fr, ar = trn_idx[i_fit], trn_idx[i_app]
                num = cp.bincount(cp.asarray(code[fr]), weights=cp.asarray(y[fr]), minlength=K) + 5.0 * pri
                den = cp.bincount(cp.asarray(code[fr]), minlength=K) + 5.0
                p = cp.clip(num / den, 1e-4, 1 - 1e-4)
                w_tr[ar] = cp.asnumpy(cp.log(p / (1 - p)) - lp)[code[ar]]
            num = cp.bincount(cp.asarray(code[trn_idx]), weights=cp.asarray(y[trn_idx]), minlength=K) + 5.0 * pri
            den = cp.bincount(cp.asarray(code[trn_idx]), minlength=K) + 5.0
            p = cp.clip(num / den, 1e-4, 1 - 1e-4)
            full = cp.asnumpy(cp.log(p / (1 - p)) - lp)
            w_tr[val_idx] = full[code[val_idx]]
            acc_tr += w_tr
            acc_te += full[tok[c][n_tr:]]
        emit('woe_sum', acc_tr, acc_te)
    timed('woe', woe)

    def nested(fit_predict, name, Xall):
        """fit_predict(Xf, yf) -> pred(X): inner 5-fold out-of-fold values for outer-train rows, a full outer-train fit
        for validation and test rows."""
        Xtr_all = Xall[:n_tr][trn_idx]
        ytr_all = y[trn_idx]
        parts = None
        for i_fit, i_app in StratifiedKFold(5, shuffle=True, random_state=777).split(np.zeros(len(trn_idx)), ytr_all):
            p = fit_predict(Xtr_all[i_fit], ytr_all[i_fit])
            v = p(Xtr_all[i_app])
            if parts is None:
                parts = np.zeros((n_tr, v.shape[1]), np.float32)
            parts[trn_idx[i_app]] = v
        pf = fit_predict(Xtr_all, ytr_all)
        parts[val_idx] = pf(Xall[:n_tr][val_idx])
        emit(name, parts, pf(Xall[n_tr:]))

    # standardised raw matrix (numerics median-imputed, categorical token codes) for the geometry sub-models
    RAWM = np.column_stack([np.nan_to_num(X[c].values.astype(np.float32), nan=float(np.nanmedian(X[c].values)))
                            for c in NUMS] + [tok[c].astype(np.float32) for c in CATS])
    RAWM = (RAWM - RAWM.mean(0)) / np.where(RAWM.std(0) > 1e-8, RAWM.std(0), 1.0)
    GRAW = cp.asarray(RAWM)

    # ---- 6. linear margin (cuML LogisticRegression)
    def mrg():
        def fp(Xf, yf):
            m = LogisticRegression(C=1.0, max_iter=2000)
            m.fit(cp.asarray(Xf), cp.asarray(yf))
            return lambda Z: cp.asnumpy(m.decision_function(cp.asarray(Z))).reshape(-1, 1).astype(np.float32)
        nested(fp, 'mrg_lr', RAWM)
    timed('mrg_lr', mrg)

    # ---- 7. cluster TE (cuML KMeans + TargetEncoder)
    def cte():
        for k in (64, 256):
            km = KMeans(n_clusters=k, random_state=42, n_init=1)
            lab = cp.asnumpy(cp.asarray(km.fit_predict(GRAW))).astype(np.int32)
            gtr = cudf.DataFrame({'c': lab[:n_tr][trn_idx]})
            gva = cudf.DataFrame({'c': lab[:n_tr][val_idx]})
            gte = cudf.DataFrame({'c': lab[n_tr:]})
            te = TargetEncoder(n_folds=5, smooth=10.0, seed=777, split_method='random')
            v_tr = np.zeros(n_tr, np.float32)
            v_tr[trn_idx] = cp.asnumpy(cp.asarray(te.fit_transform(gtr, gy_tr)).ravel())
            v_tr[val_idx] = cp.asnumpy(cp.asarray(te.transform(gva)).ravel())
            emit(f'cte{k}', v_tr, cp.asnumpy(cp.asarray(te.transform(gte)).ravel()))
    timed('cte_kmeans', cte)

    # ---- 8. kNN label response
    def knn():
        def fp(Xf, yf):
            m = KNeighborsClassifier(n_neighbors=200)
            m.fit(cp.asarray(Xf), cp.asarray(yf).astype(cp.int32))
            return lambda Z: cp.asnumpy(m.predict_proba(cp.asarray(Z)))[:, 1:2].astype(np.float32)
        nested(fp, 'knn200', RAWM)
    timed('knn200', knn)

    # ---- 9. smooth field: kernel-smoothed label response on the 2 strongest columns (cuML NearestNeighbors)
    def sf():
        S = RAWM[:, [NUMS.index('daily_screen_time_hours'), NUMS.index('social_media_hours')]].copy()

        def fp(Xf, yf):
            nn = NearestNeighbors(n_neighbors=100)
            nn.fit(cp.asarray(Xf))
            gyf = cp.asarray(yf)

            def pred(Z):
                o = np.empty((len(Z), 2), np.float32)
                for i in range(0, len(Z), 100000):
                    dist, idx = nn.kneighbors(cp.asarray(Z[i:i + 100000]))
                    dist, idx = cp.asarray(dist), cp.asarray(idx)
                    wgt = cp.exp(-(dist ** 2) / (2 * 0.25 ** 2)) + 1e-6
                    lab = gyf[idx]
                    o[i:i + 100000, 0] = cp.asnumpy((lab * wgt).sum(1) / wgt.sum(1))
                    o[i:i + 100000, 1] = cp.asnumpy(dist[:, :10].mean(1))
                return o
            return pred
        nested(fp, 'sf2d', S[:, :2])
    timed('sf2d', sf)

    # ---- 10. PCA of the masked-language-model row embedding (target-free)
    def embpca():
        E = np.load(cache_dir('density_features') / 'row_emb384.npy').astype(np.float32)
        E -= E.mean(0)
        sd = E.std(0)
        E /= np.where(sd > 1e-8, sd, 1.0)
        p = PCA(n_components=16, output_type='numpy')
        Z = p.fit_transform(cp.asarray(E))
        emit('embpca', Z[:n_tr], Z[n_tr:])
    timed('embpca', embpca)

    # ---- 11. kNN fitted on the original 7,500 rows (no competition labels: same for every fold)
    def origm():
        orig = read_original()
        for c in NUMS + [TARGET]:
            orig[c] = pd.to_numeric(orig[c], errors='coerce')
        cm = {c: {v: i for i, v in enumerate(sorted(set(orig[c].dropna().astype(str)) | set(train[c].dropna().astype(str))))}
              for c in CATS}

        def prep(df):
            M = [pd.to_numeric(df[c], errors='coerce') for c in NUMS] + [df[c].astype(str).map(cm[c]) for c in CATS]
            return pd.concat(M, axis=1).astype(np.float32)
        Xo = prep(orig)
        yo = pd.to_numeric(orig[TARGET], errors='coerce')
        ok = yo.notna() & Xo.notna().all(axis=1)
        Xo, yo = Xo[ok], yo[ok].astype(np.float32).values
        med = Xo.median()
        mu, sdv = Xo.mean(), Xo.std().replace(0, 1.0)
        Zo = ((Xo.fillna(med) - mu) / sdv).values.astype(np.float32)
        Xc = prep(pd.concat([train[FEATS], test[FEATS]], ignore_index=True))
        Zc = ((Xc.fillna(med) - mu) / sdv).values.astype(np.float32)
        kn = KNeighborsClassifier(n_neighbors=100)
        kn.fit(cp.asarray(Zo), cp.asarray(yo).astype(cp.int32))
        v = cp.asnumpy(kn.predict_proba(cp.asarray(Zc)))[:, 1].astype(np.float32)
        emit('og_knn100', v[:n_tr], v[n_tr:])
    timed('og_knn100', origm)

    A = pd.DataFrame(np.hstack(OUT_TR), columns=NAMES)
    B = pd.DataFrame(np.hstack(OUT_TE), columns=NAMES)
    assert not A.columns.duplicated().any(), 'duplicate names'
    A.to_parquet(CACHE / f'cf_f{FOLD}_tr.parquet')
    B.to_parquet(CACHE / f'cf_f{FOLD}_te.parquet')
    log(f'fold {FOLD}: {A.shape[1]} cuML-built columns -> {CACHE}/cf_f{FOLD}_{{tr,te}}.parquet')
    del GRAW
    cp.get_default_memory_pool().free_all_blocks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    train, test = read_train(), read_test()
    for k in fold_list(a.folds):
        build_fold(k, train, test)


if __name__ == '__main__':
    main()
