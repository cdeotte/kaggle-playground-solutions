#!/usr/bin/env python
"""Side features for Q-FT v2: every feature token also receives a few engineered numbers about its value, and two summary
tokens describe the whole row.

Built from the raw arrays of train, test and the original data (data.csv). The competition labels are never used; some blocks
use the ORIGINAL data's labels (fixed external data). For original rows (used in pretraining) those blocks are computed
out-of-fold within the original (5 folds, seed 2024), so an original row never sees its own label.
Per column c, side_lf[:, c, :] (zero-padded to K values):
  cnt_c, ocnt_c, amp_c              value counts over train+test, in the original, and their frequency ratio
  rp_mean_c, rp_sd_c, rp_omean_c    route profile: per Flight Distance value (a route id), the mean / sd of column c over
                                    train+test and the original's mean
  aux3_c (Class: 3 probabilities)   expected value of c given the other 20 columns + route-profile means (label-free XGB,
                                    5-fold out-of-fold over train+test)
  ares3_c                           actual - expected
side_ol[:, c, 0] = ollr_c: smoothed log-odds of the original's label for this exact value of c.
Global tokens: g_eng = 27 row aggregates + the number of NaNs (label-free); g_orig = the logit of an XGB trained only on the
original + 16 ancestor statistics (count and mean original label of the original rows that share a key, e.g. (Flight Distance, Age)).
Transforms: count-like -> log1p, amp -> log, then z-score with train+test statistics, NaN -> 0, clip +-8.
Writes features/side.npz with rows in the order train, test, original.
Usage: python side_features.py"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, labels, raw_arrays, read_original  # noqa: E402

RATINGS = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink',
           'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling',
           'Checkin service', 'Cleanliness']
FD, AGE, DEP, ARR = 'Flight Distance', 'Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
KEYS = {'FD_Age': [FD, AGE], 'FD_Age_Class_ToT': [FD, AGE, 'Class', 'Type of Travel'],
        'FD_Age_Class_ToT_Cust_Gender': [FD, AGE, 'Class', 'Type of Travel', 'Customer Type', 'Gender'],
        'FD_Age_Dep_Arr': [FD, AGE, DEP, ARR], 'ratings13': RATINGS, 'ratings13_Class_ToT': RATINGS + ['Class', 'Type of Travel'],
        'Age_ratings13': [AGE] + RATINGS, 'FD_Dep_Arr': [FD, DEP, ARR]}
PARTS = ('train', 'test', 'orig')


class Blocks:
    def __init__(self):
        Xtr, Xte, Xor, meta = raw_arrays(include_original=True)
        self.X = {'train': Xtr, 'test': Xte, 'orig': Xor}
        self.F = meta['features']
        self.IX = {f: i for i, f in enumerate(self.F)}
        self.NT, self.NE, self.NO = len(Xtr), len(Xte), len(Xor)
        self.yo = labels(read_original()[TARGET]).astype(np.float64)
        self.oof_orig = np.zeros(self.NO, np.int8)
        for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2024).split(np.zeros(self.NO), self.yo)):
            self.oof_orig[va] = k

    def codes(self, cols):
        """Integer code per row for the (multi-)column exact value over train | test | orig (NaN is its own value)."""
        A = np.concatenate([self.X[p][:, [self.IX[c] for c in cols]] for p in PARTS]).astype(np.float64)
        _, inv = np.unique(np.where(np.isnan(A), -999.0, A), axis=0, return_inverse=True)
        inv = inv.ravel()
        return inv[:self.NT], inv[self.NT:self.NT + self.NE], inv[self.NT + self.NE:], int(inv.max()) + 1

    def cnt(self):
        names, cols = [], {p: [] for p in PARTS}
        for f in self.F:
            ct, ce, co, K = self.codes([f])
            n_tt = np.bincount(np.concatenate([ct, ce]), minlength=K).astype(np.float64)
            n_o = np.bincount(co, minlength=K).astype(np.float64)
            amp = np.log((n_tt / (self.NT + self.NE) + 1e-6) / (n_o / self.NO + 1e-6))
            for p, c in zip(PARTS, (ct, ce, co)):
                cols[p] += [np.log1p(n_tt[c]), np.log1p(n_o[c]), amp[c]]
            names += [f'cnt_{f}', f'ocnt_{f}', f'amp_{f}']
        for p in PARTS:
            cols[p].append(np.isnan(self.X[p]).sum(1))
        return names + ['n_nan'], cols

    def ollr(self):
        def llr_map(c_fit, y_fit, K, s=5.0):
            prior = y_fit.mean()
            n = np.bincount(c_fit, minlength=K).astype(np.float64)
            pos = np.bincount(c_fit, weights=y_fit, minlength=K)
            p = (pos + s * prior) / (n + s)
            return np.log(p / (1 - p)) - np.log(prior / (1 - prior))
        names, cols = [], {p: [] for p in PARTS}
        for f in self.F:
            ct, ce, co, K = self.codes([f])
            m = llr_map(co, self.yo, K)
            o = np.zeros(self.NO)
            for k in range(5):
                fit, app = self.oof_orig != k, self.oof_orig == k
                o[app] = llr_map(co[fit], self.yo[fit], K)[co[app]]
            cols['train'].append(m[ct])
            cols['test'].append(m[ce])
            cols['orig'].append(o)
            names.append(f'ollr_{f}')
        return names, cols

    def anc(self):
        def feats(c_fit, y_fit, K, c_app):
            n = np.bincount(c_fit, minlength=K).astype(np.float64)[c_app]
            pos = np.bincount(c_fit, weights=y_fit, minlength=K)[c_app]
            return np.log1p(n), np.where(n > 0, pos / np.maximum(n, 1), np.nan)
        names, cols = [], {p: [] for p in PARTS}
        for kname, kc in KEYS.items():
            ct, ce, co, K = self.codes(kc)
            for p, c in (('train', ct), ('test', ce)):
                cols[p] += list(feats(co, self.yo, K, c))
            a, b = np.zeros(self.NO), np.zeros(self.NO)
            for k in range(5):
                fit, app = self.oof_orig != k, self.oof_orig == k
                a[app], b[app] = feats(co[fit], self.yo[fit], K, co[app])
            cols['orig'] += [a, b]
            names += [f'anc_n_{kname}', f'anc_y_{kname}']
        return names, cols

    def orgm(self):
        """Logit of an XGB trained only on the original (train/test: mean of its 5 fold models; original rows: its OOF)."""
        import xgboost as xgb
        p = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', max_depth=7, learning_rate=0.03,
                 min_child_weight=5, subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0, max_bin=512, seed=7)
        oof, tt, te = np.zeros(self.NO), np.zeros(self.NT), np.zeros(self.NE)
        dtr, dte = xgb.DMatrix(self.X['train']), xgb.DMatrix(self.X['test'])
        Xo = self.X['orig']
        for k in range(5):
            fit, app = self.oof_orig != k, self.oof_orig == k
            rest = np.where(fit)[0]
            es = np.random.default_rng(k).random(len(rest)) < 0.1      # early stopping on an inner split of the original
            d1, d2 = xgb.DMatrix(Xo[rest[~es]], self.yo[rest[~es]]), xgb.DMatrix(Xo[rest[es]], self.yo[rest[es]])
            b = xgb.train(p, d1, 5000, evals=[(d2, 'es')], early_stopping_rounds=200, verbose_eval=False)
            it = (0, b.best_iteration + 1)
            oof[app] = b.predict(xgb.DMatrix(Xo[app]), iteration_range=it, output_margin=True)
            tt += b.predict(dtr, iteration_range=it, output_margin=True) / 5
            te += b.predict(dte, iteration_range=it, output_margin=True) / 5
        return ['orgm_logit'], {'train': [tt], 'test': [te], 'orig': [oof]}

    def eng(self):
        """Label-free row aggregates: rating statistics, delay geometry, step points of the original, category combos."""
        inflight = ['Inflight wifi service', 'Food and drink', 'Seat comfort', 'Inflight entertainment', 'On-board service',
                    'Leg room service', 'Cleanliness']
        ground = ['Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Online boarding',
                  'Baggage handling', 'Checkin service']
        IX, names, cols = self.IX, None, {}
        for p in PARTS:
            A = self.X[p]
            r = A[:, [IX[c] for c in RATINGS]]
            dep, arr, age, fd = A[:, IX[DEP]], A[:, IX[ARR]], A[:, IX[AGE]], A[:, IX[FD]]
            cls, tot, cust, wifi = A[:, IX['Class']], A[:, IX['Type of Travel']], A[:, IX['Customer Type']], A[:, IX['Inflight wifi service']]
            f = dict(r_sum=r.sum(1), r_mean=r.mean(1), r_std=r.std(1), r_min=r.min(1), r_max=r.max(1), r_n0=(r == 0).sum(1),
                     r_n1=(r == 1).sum(1), r_n5=(r == 5).sum(1), r_n45=(r >= 4).sum(1), r_inflight=A[:, [IX[c] for c in inflight]].sum(1),
                     r_ground=A[:, [IX[c] for c in ground]].sum(1), d_diff=arr - dep, d_max=np.fmax(dep, arr),
                     d_arr_nan=np.isnan(arr).astype(np.float32), d_log_dep=np.log1p(dep), d_log_arr=np.log1p(arr),
                     d_dep_pos=(dep > 0).astype(np.float32), d_arr_pos=(arr > 0).astype(np.float32),
                     age_lt39=(age < 39).astype(np.float32), age_ge61=(age >= 61).astype(np.float32),
                     fd_ge1325=(fd >= 1325).astype(np.float32), wifi0=(wifi == 0).astype(np.float32),
                     wifi5=(wifi == 5).astype(np.float32), cls_tot=cls * 2 + tot, cust_tot=cust * 2 + tot,
                     cls_tot_cust=cls * 4 + tot * 2 + cust, fd_per_age=fd / np.maximum(age, 1))
            names = list(f)
            cols[p] = [f[n] for n in names]
        return names, cols

    def rprof(self):
        """Route profile (Flight Distance = route id): per distance, mean and sd of every other column over train+test, and the
        mean over the original rows (NaN-aware)."""
        ct, ce, co, K = self.codes([FD])
        cmap = {'train': ct, 'test': ce, 'orig': co}
        names, cols = [], {p: [] for p in PARTS}
        for i, col in enumerate(self.F):
            if col == FD:
                continue
            v = np.concatenate([self.X['train'][:, i], self.X['test'][:, i]]).astype(np.float64)
            c = np.concatenate([ct, ce])
            ok = ~np.isnan(v)
            n = np.bincount(c[ok], minlength=K)
            s1 = np.bincount(c[ok], weights=v[ok], minlength=K)
            s2 = np.bincount(c[ok], weights=v[ok] ** 2, minlength=K)
            mean = np.where(n > 0, s1 / np.maximum(n, 1), np.nan)
            sd = np.where(n > 1, np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0)), np.nan)
            vo = self.X['orig'][:, i].astype(np.float64)
            oko = ~np.isnan(vo)
            no = np.bincount(co[oko], minlength=K)
            omean = np.where(no > 0, np.bincount(co[oko], weights=vo[oko], minlength=K) / np.maximum(no, 1), np.nan)
            for p in PARTS:
                cols[p] += [mean[cmap[p]], sd[cmap[p]], omean[cmap[p]]]
            names += [f'rp_mean_{col}', f'rp_sd_{col}', f'rp_omean_{col}']
        return names, cols

    def aux3(self, rp_names, rp_cols):
        """For every column: an XGB (depth 9, eta 0.05, 800 rounds) predicts it from the other 20 raw columns + the route-profile
        means of the other columns. 5-fold KFold OOF over train+test (seed 0); original rows get the mean of the 5 fold models.
        Class (3 levels) -> softprob. ares3 = value - expectation. Label-free."""
        import xgboost as xgb
        RP = {p: np.stack(rp_cols[p], 1) for p in PARTS}
        names, cols, rnames, rcols = [], {p: [] for p in PARTS}, [], {p: [] for p in PARTS}
        for c, col in enumerate(self.F):
            keep = [i for i, n in enumerate(rp_names) if '_sd_' not in n and not n.endswith(f'_{col}')]

            def inputs(p):
                return np.c_[np.delete(self.X[p], c, axis=1), RP[p][:, keep]]
            A = np.concatenate([inputs('train'), inputs('test')])
            tgt = np.concatenate([self.X['train'][:, c], self.X['test'][:, c]]).astype(np.float64)
            fo = inputs('orig')
            multi = col == 'Class'
            prm = dict(tree_method='hist', device='cuda', max_depth=9, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       min_child_weight=10, max_bin=256, seed=0)
            prm.update(dict(objective='multi:softprob', num_class=3) if multi else dict(objective='reg:squarederror'))
            ko = 3 if multi else 1
            oof, po = np.zeros((len(A), ko)), np.zeros((len(fo), ko))
            do = xgb.DMatrix(fo, missing=np.nan)
            for a, b in KFold(5, shuffle=True, random_state=0).split(A):
                a = a[~np.isnan(tgt[a])]
                bst = xgb.train(prm, xgb.DMatrix(A[a], label=tgt[a], missing=np.nan), 800)
                oof[b] = bst.predict(xgb.DMatrix(A[b], missing=np.nan)).reshape(len(b), ko)
                po += bst.predict(do).reshape(len(fo), ko) / 5
            part = {'train': oof[:self.NT], 'test': oof[self.NT:], 'orig': po}
            for p in PARTS:
                cols[p] += [part[p][:, j] for j in range(ko)]
                if not multi:
                    rcols[p].append(self.X[p][:, c] - part[p][:, 0])
            names += [f'aux3_{col}'] if not multi else [f'aux3_{col}_p{j}' for j in range(ko)]
            if not multi:
                rnames.append(f'ares3_{col}')
            print(f'  aux3 {col} done', flush=True)
        return (names, cols), (rnames, rcols)


def transform(name, x, n_fit):
    """count-like -> log1p, amp -> log, then z-score with the statistics of the first n_fit rows (train+test), NaN -> 0, clip +-8."""
    x = np.asarray(x, np.float64)
    if name.startswith(('cnt_', 'ocnt_', 'n_nan', 'anc_n_')):
        x = np.log1p(np.maximum(x, 0))
    elif name.startswith('amp_'):
        x = np.log(np.maximum(x, 1e-3))
    m, s = np.nanmean(x[:n_fit]), np.nanstd(x[:n_fit])
    return np.clip(np.nan_to_num((x - m) / (s if s > 0 else 1.0), nan=0.0), -8, 8)


def main():
    B = Blocks()
    n_fit = B.NT + B.NE
    cat = lambda cols: [np.concatenate([cols[p][j] for p in PARTS]) for j in range(len(cols['train']))]  # noqa: E731
    blocks = {'cnt': B.cnt(), 'ollr': B.ollr(), 'anc': B.anc(), 'orgm': B.orgm(), 'eng': B.eng(), 'rprof': B.rprof()}
    blocks['aux3'], blocks['ares3'] = B.aux3(*blocks['rprof'])
    print('blocks built', flush=True)
    full = {b: dict(zip(names, cat(cols))) for b, (names, cols) in blocks.items()}
    cols = [c for c in read_original().columns if c != TARGET]            # token order = data.csv order (as in ../qft)
    per = {c: [] for c in cols}
    for b in ('cnt', 'rprof', 'aux3', 'ares3'):
        for n, v in full[b].items():
            hit = [c for c in cols if n.endswith('_' + c)] or (['Class'] if n.startswith('aux3_Class_p') else [])
            if hit:
                per[hit[0]].append(transform(n, v, n_fit))
    N = n_fit + B.NO
    K = max(len(v) for v in per.values())
    side_lf = np.zeros((N, len(cols), K), np.float16)
    for ci, c in enumerate(cols):
        for kk, v in enumerate(per[c]):
            side_lf[:, ci, kk] = v
    side_ol = np.zeros((N, len(cols), 1), np.float16)
    for n, v in full['ollr'].items():
        side_ol[:, cols.index(n[len('ollr_'):]), 0] = transform(n, v, n_fit)
    g_eng = np.stack([transform(n, v, n_fit) for n, v in full['eng'].items()] + [transform('n_nan', full['cnt']['n_nan'], n_fit)], 1)
    g_orig = np.stack([transform(n, v, n_fit) for b in ('orgm', 'anc') for n, v in full[b].items()], 1)
    out = HERE / 'features'
    out.mkdir(exist_ok=True)
    np.savez(out / 'side.npz', side_lf=side_lf, side_ol=side_ol, g_eng=g_eng.astype(np.float16), g_orig=g_orig.astype(np.float16))
    json.dump(dict(K=K, rows=dict(train=B.NT, test=B.NE, orig=B.NO)), open(out / 'side_meta.json', 'w'), indent=1)
    print(f'side_lf {side_lf.shape}, side_ol {side_ol.shape}, g_eng {g_eng.shape}, g_orig {g_orig.shape}')


if __name__ == '__main__':
    main()
