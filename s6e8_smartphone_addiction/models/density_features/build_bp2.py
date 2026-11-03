"""Budget-mixture posteriors -> bp2_f<fold>_{tr,te}.parquet for the 5 outer folds (26 columns).

A row's utilization u = (social + gaming + work) / daily is either "repaired" (u in [0.85, 0.95)) or organic. Three
label-free LightGBM classifiers (300 trees, lr 0.08, 63 leaves, seed 42) of the repaired state, fit on the rows of train +
test where u is defined: pi_d (context + social+gaming+work and shares, used when daily is missing), pi_1 (context + daily
and the components, used when one component is missing) and pi_c (context only, used by the 2-missing sampler).
Context = age, sleep, notifications, app opens, weekend, gender, stress, academic impact.

Exactly one of daily / social / gaming / work missing: the den_uncond384v3.pt MLM distribution over the missing column's
exact values is truncated to the values allowed by the budget (and daily <= 20.01 - sleep), then reweighted so that mass
pi lies on the repaired interval (u in [0.85, 0.95]) and 1 - pi outside it. Missing sleep (with daily observed): the MLM
distribution truncated to sleep <= 20.01 - daily. From the posterior: bp2_fill_<col> (mean), bp2_sd_<col> (sd) and the
fold-wise response logits (tables of density_lib/completion.py, smoothing 20 / 3) averaged over the posterior:
cmp5_c_<col> / cmp5_cs_<col>, and their row means cmp5_mean / cmp5_s_mean.
Exactly two of daily / social / gaming / work missing: a coupled sampler, 24 draws per row: u ~ U[0.85, 0.95) with
probability pi_c, else one of 25 quantiles of the organic u; with daily observed, the remaining budget u * daily - observed
components is split between the missing components by a Beta(2, 2) fraction; with daily missing, daily = components / u /
0.67; values snapped to the token lattice; cmp5_pair20 / cmp5_pair3 = the summed response logits of the two missing
fields averaged over the draws. Also bp2_pi_d and bp2_pi_1."""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from density_lib.base import CACHE, FEATS, TARGET, get_folds, load_comp  # noqa: E402
from density_lib.completion import inner_groups, lgt, response_table, row_tables  # noqa: E402
from density_lib.mlm import DEVICE, cell_logits, load_mlm, load_vocab, token_values  # noqa: E402

t0 = time.time()
train, test = load_comp()
y = train[TARGET].values
n_tr = len(train); n_all = n_tr + len(test)
folds = get_folds(y)

D = pd.concat([train, test], ignore_index=True)
dv = D['daily_screen_time_hours'].values.astype(np.float64)
sv = D['social_media_hours'].values.astype(np.float64)
gv = D['gaming_hours'].values.astype(np.float64)
wv = D['work_study_hours'].values.astype(np.float64)
slv = D['sleep_hours'].values.astype(np.float64)
COMP = {'daily_screen_time_hours': dv, 'social_media_hours': sv, 'gaming_hours': gv, 'work_study_hours': wv}
CN = list(COMP)
util = (sv + gv + wv) / dv
comp_ok = np.isfinite(util)
is_rep = ((util >= 0.85) & (util < 0.95)).astype(int)
u_organic = util[comp_ok & ((util < 0.85) | (util >= 0.95))]
u_org_quant = np.quantile(u_organic[np.isfinite(u_organic) & (u_organic > 0) & (u_organic <= 1.0)],
                          np.linspace(0.02, 0.98, 25))

# ---- repaired-state classifiers (label-free) ----
base_ctx_df = D[['age', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time']].copy()
for c in ['gender', 'stress_level', 'academic_work_impact']:
    base_ctx_df[c] = D[c].astype('category').cat.codes.replace(-1, np.nan)


def fit_pi(extra):
    X = pd.concat([base_ctx_df] + ([extra] if extra is not None else []), axis=1)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=63, verbose=-1, n_jobs=16, random_state=42)
    m.fit(X[comp_ok].values.astype(np.float32), is_rep[comp_ok])
    return m, X


sgw_all = sv + gv + wv
pi_d_m, pi_d_X = fit_pi(pd.DataFrame({'sgw': sgw_all, 'sh_s': sv / sgw_all, 'sh_g': gv / sgw_all}))
pi_1_m, pi_1_X = fit_pi(pd.DataFrame({'d': dv, 'c1': sv, 'c2': gv, 'c3': wv}))
pi_c_m, pi_c_X = fit_pi(None)
PI_D = pi_d_m.predict_proba(pi_d_X.values.astype(np.float32))[:, 1]
PI_1 = pi_1_m.predict_proba(pi_1_X.values.astype(np.float32))[:, 1]
PI_C = pi_c_m.predict_proba(pi_c_X.values.astype(np.float32))[:, 1]
print(f'pi models fit ({time.time() - t0:.0f}s)', flush=True)

# ---- MLM and response tables (daily, social, gaming, work, sleep) ----
tok_tr, tok_te, vocabs, IDS = load_vocab()
model = load_mlm('den_uncond384v3.pt', vocabs, 384, 8)
X_all = torch.from_numpy(IDS)
grp, tab_tr = inner_groups(y, folds)
RESP = {c: response_table(IDS[:n_tr, FEATS.index(c)], vocabs[c], y, folds, grp) for c in CN + ['sleep_hours']}
VV = {c: token_values(tok_tr, tok_te, c) for c in CN + ['sleep_hours']}
print(f'tables built ({time.time() - t0:.0f}s)', flush=True)

n_miss4 = sum(np.isnan(COMP[c]).astype(int) for c in CN)

# ================= exactly one of daily / social / gaming / work missing: MLM mixture posterior
BS = 16384
bp_fill = {c: np.zeros(n_all, np.float32) for c in CN + ['sleep_hours']}
bp_sd = {c: np.zeros(n_all, np.float32) for c in CN + ['sleep_hours']}
cmp5 = np.full((n_all, 5, 5, 2), np.nan, np.float32)   # (row, field, fold, smoothing)
FIELD_IDX = {c: i for i, c in enumerate(CN + ['sleep_hours'])}


def process_field(c, rows, lo, hi, feas_lo, feas_hi, pi_vec):
    """lo / hi: the repaired interval of the missing value (None: no mixture); feas_lo / feas_hi: its feasible range."""
    j = FEATS.index(c)
    vv = torch.tensor(VV[c], dtype=torch.float32, device=DEVICE)
    Rt = torch.from_numpy(RESP[c].reshape(len(VV[c]), 60)).to(DEVICE)
    ci = FIELD_IDX[c]
    for i in range(0, len(rows), BS):
        r = rows[i:i + BS]
        p = torch.softmax(cell_logits(model, X_all, r, j), -1)
        flo = torch.tensor(np.nan_to_num(feas_lo[r], nan=-1e9), dtype=torch.float32, device=DEVICE)
        fhi = torch.tensor(np.nan_to_num(feas_hi[r], nan=1e9), dtype=torch.float32, device=DEVICE)
        feas = ((vv.unsqueeze(0) >= flo.unsqueeze(1) - 1e-6) & (vv.unsqueeze(0) <= fhi.unsqueeze(1) + 1e-6)).float()
        feas[:, :2] = 0
        p = p * feas
        p = p / p.sum(1, keepdim=True).clamp_min(1e-9)
        if lo is not None:
            lo_r = torch.tensor(np.nan_to_num(lo[r], nan=-1e9), dtype=torch.float32, device=DEVICE)
            hi_r = torch.tensor(np.nan_to_num(hi[r], nan=1e9), dtype=torch.float32, device=DEVICE)
            inside = ((vv.unsqueeze(0) >= lo_r.unsqueeze(1)) & (vv.unsqueeze(0) <= hi_r.unsqueeze(1))).float() * feas
            m_in = (p * inside).sum(1)
            m_out = (p * (feas - inside).clamp_min(0)).sum(1).clamp_min(1e-9)
            pi_r = torch.tensor(pi_vec[r], dtype=torch.float32, device=DEVICE).clamp(0.02, 0.98)
            w_in = torch.where(m_in > 1e-6, pi_r, torch.zeros_like(pi_r))
            q = p * (inside * (w_in / m_in.clamp_min(1e-9)).unsqueeze(1) +
                     (feas - inside).clamp_min(0) * ((1 - w_in) / m_out).unsqueeze(1))
            q = q / q.sum(1, keepdim=True).clamp_min(1e-9)
        else:
            q = p
        e1 = (q @ vv.unsqueeze(1)).squeeze(1)
        e2 = (q @ (vv ** 2).unsqueeze(1)).squeeze(1)
        bp_fill[c][r] = e1.cpu().numpy()
        bp_sd[c][r] = torch.sqrt(torch.clamp(e2 - e1 ** 2, min=0)).cpu().numpy()
        marg = (q @ Rt).cpu().numpy().reshape(len(r), 5, 6, 2)
        sel = row_tables(r, tab_tr, n_tr)
        ar = np.arange(len(r))
        for f in range(5):
            cmp5[r, ci, f, 0] = lgt(marg[ar, f, sel[:, f], 0])
            cmp5[r, ci, f, 1] = lgt(marg[ar, f, sel[:, f], 1])


for c in CN:
    mrows = np.where(np.isnan(COMP[c]) & (n_miss4 == 1))[0]
    if c == 'daily_screen_time_hours':
        sgw_ = sv + gv + wv
        process_field(c, mrows, sgw_ / 0.95, sgw_ / 0.85, sgw_,
                      np.where(np.isfinite(slv), 20.01 - slv, np.inf), PI_D)
    else:
        others = sum(COMP[k] for k in CN if k not in (c, 'daily_screen_time_hours'))
        process_field(c, mrows, 0.85 * dv - others, 0.95 * dv - others, np.zeros(n_all), dv - others, PI_1)
    print(f'  single-missing {c}: {len(mrows)} ({time.time() - t0:.0f}s)', flush=True)
# sleep: cap truncation only (no repaired interval)
mrows = np.where(np.isnan(slv) & np.isfinite(dv))[0]
process_field('sleep_hours', mrows, None, None, np.zeros(n_all), np.where(np.isfinite(dv), 20.01 - dv, np.inf), None)
print(f'  sleep-missing: {len(mrows)} ({time.time() - t0:.0f}s)', flush=True)

# ================= exactly two missing: coupled sampler
rng = np.random.default_rng(4242)
K = 24
two_rows = np.where(n_miss4 == 2)[0]
print(f'two-missing rows: {len(two_rows)}', flush=True)
cmp5_pair = np.full((n_all, 5, 2), np.nan, np.float32)   # (row, fold, smoothing): sum of the two fields' response logits


def snap(vals_sorted, x):
    i = np.clip(np.searchsorted(vals_sorted, x), 1, len(vals_sorted) - 1)
    left = vals_sorted[i - 1]; right = vals_sorted[np.minimum(i, len(vals_sorted) - 1)]
    return np.where(np.abs(x - left) <= np.abs(right - x), left, right)


tokidx = {c: (VV[c], {round(val, 2): k for k, val in enumerate(VV[c])}) for c in CN}
CHUNK = 50000
for start in range(0, len(two_rows), CHUNK):
    r = two_rows[start:start + CHUNK]
    miss_mask = np.stack([np.isnan(COMP[c][r]) for c in CN], 1)
    pi_r = np.clip(PI_C[r], 0.05, 0.95)
    acc = np.zeros((len(r), 5, 2))
    for k_ in range(K):
        u = np.where(rng.random(len(r)) < pi_r, rng.uniform(0.85, 0.95, len(r)), rng.choice(u_org_quant, len(r)))
        obs_sum = np.nansum(np.stack([sv[r], gv[r], wv[r]], 1), 1)
        # case A: daily observed, two of social / gaming / work missing: split the remaining budget u * daily -
        # observed components between them with a Beta(2, 2) fraction
        caseA = ~np.isnan(dv[r]) & ~miss_mask[:, 0]
        T = u * dv[r]
        rem = np.clip(T - obs_sum, 0, None)
        frac = rng.beta(2, 2, len(r))
        comp_arr = {'social_media_hours': sv[r].copy(), 'gaming_hours': gv[r].copy(), 'work_study_hours': wv[r].copy()}
        missing_two = ['social_media_hours', 'gaming_hours', 'work_study_hours']
        filled = {c: comp_arr[c] for c in missing_two}
        fr_used = np.zeros(len(r))
        first = np.ones(len(r), bool)
        for c in missing_two:
            mm = np.isnan(comp_arr[c]) & caseA
            v_new = np.where(mm, np.clip(rem * np.where(first, frac, 1 - fr_used), 0, None), comp_arr[c])
            fr_used = np.where(first & mm, frac, fr_used)
            first = np.where(mm, False, first)
            filled[c] = v_new
        # case B: daily missing (+ one component): daily from the components and u
        caseB = np.isnan(dv[r])
        comp_obs2 = np.nansum(np.stack([filled['social_media_hours'], filled['gaming_hours'],
                                        filled['work_study_hours']], 1), 1)
        d_res = np.where(caseB, comp_obs2 / np.maximum(u, 0.5) / np.where(caseB, 1 - 0.33, 1), dv[r])
        # accumulate the response logits of the two missing fields
        for ci, c in enumerate(CN):
            mm = miss_mask[:, ci]
            if not mm.any(): continue
            x = d_res if c == 'daily_screen_time_hours' else filled.get(c, COMP[c][r])
            v_, vmap = tokidx[c]
            xs = snap(v_[2:], np.nan_to_num(x, nan=v_[2:].mean()))
            idxs = np.array([vmap.get(round(t, 2), 2) for t in xs])
            Rc = RESP[c]
            sel = row_tables(r, tab_tr, n_tr)
            for f in range(5):
                pr20 = Rc[idxs, f, sel[:, f], 0]; pr3 = Rc[idxs, f, sel[:, f], 1]
                acc[:, f, 0] += np.where(mm, lgt(pr20), 0)
                acc[:, f, 1] += np.where(mm, lgt(pr3), 0)
    cmp5_pair[r] = (acc / K).astype(np.float32)
    print(f'  pair chunk {start}: done ({time.time() - t0:.0f}s)', flush=True)

# ---- per-fold files ----
base_cols = {}
for c in CN + ['sleep_hours']:
    base_cols[f'bp2_fill_{c}'] = bp_fill[c]
    base_cols[f'bp2_sd_{c}'] = bp_sd[c]
base_cols['bp2_pi_d'] = PI_D.astype(np.float32)
base_cols['bp2_pi_1'] = PI_1.astype(np.float32)
for f in range(5):
    cols = dict(base_cols)
    CL = cmp5[:, :, f, 0]; CLS = cmp5[:, :, f, 1]
    with np.errstate(all='ignore'):
        cols['cmp5_mean'] = np.nan_to_num(np.nanmean(CL, 1), nan=0).astype(np.float32)
        cols['cmp5_s_mean'] = np.nan_to_num(np.nanmean(CLS, 1), nan=0).astype(np.float32)
    for c in CN + ['sleep_hours']:
        ci = FIELD_IDX[c]
        cols[f'cmp5_c_{c}'] = np.nan_to_num(CL[:, ci], nan=0).astype(np.float32)
        cols[f'cmp5_cs_{c}'] = np.nan_to_num(CLS[:, ci], nan=0).astype(np.float32)
    cols['cmp5_pair20'] = np.nan_to_num(cmp5_pair[:, f, 0], nan=0).astype(np.float32)
    cols['cmp5_pair3'] = np.nan_to_num(cmp5_pair[:, f, 1], nan=0).astype(np.float32)
    O = pd.DataFrame(cols)
    O.iloc[:n_tr].reset_index(drop=True).to_parquet(CACHE / f'bp2_f{f}_tr.parquet')
    O.iloc[n_tr:].reset_index(drop=True).to_parquet(CACHE / f'bp2_f{f}_te.parquet')
print(f'saved ({time.time() - t0:.0f}s)')
