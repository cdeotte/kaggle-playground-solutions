# Density features: base channels and masked-language-model channels

Shared feature steps of the density-model family (`../realmlp_density`, `../tabm_density`, `../lookup_transformer`,
`../xgb_density_margin`, `../catboost_density`, `../dcn`, `../ydf`, `../cuml_density_models`). This folder trains no
classifier and writes no prediction pairs: every script writes files to `cache/` that those folders read (table below).

Labels enter only through inner cross-fits: in outer fold k, the outer-train rows are encoded with maps fit on the other
inner groups of `StratifiedKFold(5, shuffle=True, random_state=777)`, the validation and test rows with maps fit on all
outer-train rows. Everything else is label-free and uses the predictors of train + test together (987,671 rows) for
counts, imputers and the language models; the 7,500 original rows enter only as fixed evidence (`orig` columns).

## Base channels
- **static** (`build_static.py`, `density_lib/features.py`, 120 columns): the 12 raw columns (categoricals as fixed
  ordinal codes); 12 missing flags + total / screen / behaviour missing counts; budget geometry (social + gaming + work,
  slack = daily - that sum, slack / daily, the three shares, weekend - daily, weekend - sum, weekend / daily, daily - 6,
  daily - 8, social - 4); the observed label-rule region (high if daily > 8 or social > 4, low if daily <= 6 and social
  <= 4, mid, NaN if undetermined); train + test counts of each exact value; original-data amplification
  log(p_synthetic(v) / p_original(v)); original label LLRs of each exact value (smoothing 5 toward the original prior) and
  of 0.1-rounded values for the six hour columns, + their sum; observed-only PMI of 18 exact value pairs. Plus XGBoost
  regressors (600 trees, lr 0.08, depth 8) for the 5 screen columns, each fit on the rows where it is observed from the
  other 11 raw columns (`imp_`, `fill_` + 5 geometry columns of the fills), and a 3-class XGBoost model of the region
  (700 trees) whose output is replaced by the one-hot region where the observed values determine it.
- **tok**: each column's exact value as a string token (`'%.2f'`, missing `'__NA__'`): the MLM vocabulary and the target
  encoding keys.
- **mc** (`build_mc.py`, 11 columns): 8-quantile XGBoost regressors of the 5 screen columns give conditional CDFs; 48
  inverse-CDF draws per row with the budget daily >= social + gaming + work enforced on the draws; region probabilities,
  expected excesses over the thresholds and slack moments.
- **impq** (`build_impq.py`, 9 columns): q10 / q90 XGBoost quantile regressors of daily and social, their width and the
  position of the thresholds 6 / 8 (daily) and 4 (social) inside the interval.

## Fold-wise assembly (library used by the model folders)
Import with `sys.path.insert(0, str(REPO / 'models' / 'density_features'))` (and the competition folder `REPO` on the
path), then `from density_lib.assemble import assemble`.
`density_lib/assemble.py`: `assemble(fold, cfg, par)` builds the train / test frames of one outer fold from the static
block and the cfg flags (docstring): counts (`cnt`), original evidence (`orig`), PMI (`pmi`), imputers (`imp`), budget
bounds (`bnd`), `mc`, `impq`, `knn`, `den`, decimal lattice (`lat`), nested exact-token target encodings (`te`, smoothing
`par te_smooth`; 12 single keys + daily x social and daily x weekend on the 0.1 lattice + notifications x app opens),
region x token keys (`rte`), 8 quantile-binned geometry keys (`geo`), the inner encodings averaged over 3 inner seeds
777 / 1778 / 2779 (`teavg`), a second smoothing of the 12 single keys (`te2s`), WOE sums (`woe`), the WOE base margin
(`mrg`, `par mrg_scale`) and `slim`. `density_lib/extras.py`: `add_smooth_fields` (`sf`: cross-fitted Gaussian-blurred
1-D and 2-D label-rate surfaces) and `add_embeddings` (`emb`: 48 PCs of `row_emb384.npy`).

The `den` flag reads `den<dtag>_{tr,te}.parquet` (`par dtag`, default `''`): realmlp, realmlp_battery and the first
lookup transformer read the d384 / 6-layer channels (`par {"dtag": "384"}` -> `den384_*`), the second lookup transformer
and the cuML models the d320 / 7-layer channels (`den_*`).

## Masked lattice language model (MLM)
`train_mlm.py`, `density_lib/mlm.py`. A transformer encoder over the 12 column tokens of a row: vocabulary per column =
`[NA]` (missing), `[MASK]`, and every exact token of train + test; per-column embedding + learned column vector,
pre-norm `TransformerEncoder` (8 heads, feed-forward 4d, dropout 0.1, GELU), final LayerNorm, one linear head per column.
Trained on all train + test rows without labels: each observed cell is masked with probability 0.30 (at least one per
row), cross-entropy on the masked cells only; AdamW lr 1e-3 / weight decay 1e-5, OneCycle (10% warm-up), batch 4096,
gradient clipping 1.0, bf16. After training every observed cell is masked in turn: `den_lp_<col>` = log p(x_col | rest),
`den_ent_<col>` = predictive entropy, `den_pll_sum` / `den_pll_mean` = row pseudo-log-likelihood (26 columns).

| Checkpoint | d / layers / epochs / seed | Final masked-token loss | Channels | Also used by |
|---|---|---:|---|---|
| `den_uncond256.pt` | 256 / 6 / 16 / 42 | 4.0056 | `den256` (not read) | initialization of `../lookup_transformer` |
| `den_uncond384.pt` | 384 / 6 / 24 / 42 | 3.9677 | `den384` | `row_emb384.npy` -> `knn`, `emb` |
| `den_uncond256v2.pt` | 256 / 6 / 20 / 42 | 4.0464 | `den256v2` | `mlmfill2` |
| `den_uncond384v3.pt` | 384 / 8 / 30 / 42 | 3.9821 | `den384v3` | `cmp3`, `mlmfill`, `mlmfill2`, `tokemb`, `rrd`, `bp2` |
| `den_uncond.pt` | 320 / 7 / 25 / 777 | 4.0057 | `den` | |
| `den_uncond_v4.pt` | 384 / 8 / 60 / 42 | 3.9069 | `den_v4` | `cmp4`, `mlmfill4`, `qfill` |
| `den_uncond_v5.pt` | 448 / 8 / 80 / 42 | 3.8605 | `den_v5` | |

Channels built from the trained models (label-free unless stated):
- **knn** (`build_knn.py`): row embedding = mean-pooled final hidden state of `den_uncond384.pt` (`row_emb384.npy`,
  float16); per fold, cosine kNN (cuML, k = 64) label responses: mean label of the 16 / 64 nearest, distance-weighted
  mean (w = 1 / (d + 1e-3)), mean distance; inner-cross-fitted like the target encodings.
- **cmp3 / cmp4** (`build_cmp.py`, 23 columns per fold): completion response. For a missing cell, the label rate of each
  exact token (fold-wise tables, smoothing 20 and 3, inner-cross-fitted) is averaged over the MLM's P(token | rest of
  row); row mean / sd / min / max of these logits, the screen-column means, the logits of the observed tokens
  ("context"), a confidence (1 - normalized entropy), the evidence sum and the per-screen-column completion logits.
- **mlmfill / mlmfill4** (`build_mlmfill.py`, 27 columns): E[v | rest] and its sd for every missing numeric cell, and the
  budget geometry and region recomputed from the filled values. **mlmfill2** (`build_mlmfill2.py`, 15 columns): the
  fills averaged over `den_uncond384v3.pt` and `den_uncond256v2.pt`.
- **tokemb** (`build_tokemb.py`, 36 columns): 3-component PCA of each column's token-embedding table.
- **rrd** (`build_rrd.py`, 19 columns): residual v - E[v | rest] and z-residual of every observed numeric cell.
- **bp2** (`build_bp2.py`, 26 columns per fold): budget-mixture posteriors. LightGBM classifiers (label-free) of the
  "repaired" utilization state 0.85 <= (social + gaming + work) / daily < 0.95; for rows missing one budget field, the
  MLM distribution of that field truncated to the budget-feasible range and reweighted between the repaired interval
  (mass pi) and the rest; for rows missing two, a coupled sampler (24 draws); posterior means / sds and fold-wise
  response logits (inner-cross-fitted tables).
- **qfill** (`build_qfill.py`, 14 columns): q10-q90 width and skew of the MLM posterior of every missing hour cell.

## Commands (run order)
```
python build_static.py
python build_mc.py
python build_impq.py

python train_mlm.py --dmodel 256 --layers 6 --epochs 16 --name den256   --checkpoint den_uncond256.pt
python train_mlm.py --dmodel 384 --layers 6 --epochs 24 --name den384   --checkpoint den_uncond384.pt
python train_mlm.py --dmodel 256 --layers 6 --epochs 20 --name den256v2 --checkpoint den_uncond256v2.pt
python train_mlm.py --dmodel 384 --layers 8 --epochs 30 --name den384v3 --checkpoint den_uncond384v3.pt
python train_mlm.py --dmodel 320 --layers 7 --epochs 25 --seed 777      # den_{tr,te}.parquet, den_uncond.pt
python train_mlm.py --dmodel 384 --layers 8 --epochs 60 --name den_v4   --checkpoint den_uncond_v4.pt
python train_mlm.py --dmodel 448 --layers 8 --epochs 80 --name den_v5   --checkpoint den_uncond_v5.pt

python build_knn.py emb --checkpoint den_uncond384.pt --dmodel 384 --layers 6 --emb row_emb384.npy
python build_knn.py knn --emb row_emb384.npy                          # --folds 0,1 runs a subset

python build_cmp.py --checkpoint den_uncond384v3.pt --name cmp3
python build_mlmfill.py --checkpoint den_uncond384v3.pt --name mlmfill
python build_mlmfill2.py
python build_tokemb.py
python build_rrd.py
python build_bp2.py

python build_cmp.py --checkpoint den_uncond_v4.pt --name cmp4
python build_mlmfill.py --checkpoint den_uncond_v4.pt --name mlmfill4
python build_qfill.py

# optional, not read by the final models: class-conditional MLM LLRs (flag cllr)
python train_mlm.py --mode cond --epochs 6
```
No other folder has to run first (only the data in `../../data`). One GPU: the MLMs take 6 to 80 minutes each (the 60-
and 80-epoch models the longest), the channel scripts seconds to a few minutes; `build_static.py`, `build_mc.py` and
`build_impq.py` train XGBoost on the GPU; `build_tokemb.py` runs on the CPU.

## Cache files (in run order)
| File | Built by | Contents |
|---|---|---|
| `static_tr.parquet`, `static_te.parquet` | `build_static.py` | the 120-column static block (above) |
| `tok_tr.parquet`, `tok_te.parquet` | `build_static.py` | exact-value tokens of the 12 columns |
| `mc_tr.parquet`, `mc_te.parquet` | `build_mc.py` | 11 Monte-Carlo posterior columns (flag `mc`) |
| `impq_tr.parquet`, `impq_te.parquet` | `build_impq.py` | 9 quantile-imputer columns (flag `impq`) |
| `den256_{tr,te}.parquet`, `den_uncond256.pt` | `train_mlm.py` d256 / 6 / 16 | density channels (not read); weights that initialize the lookup transformer |
| `den384_{tr,te}.parquet`, `den_uncond384.pt` | `train_mlm.py` d384 / 6 / 24 | 26 density channels: the `den` block (`dtag 384`) of realmlp, realmlp_battery and the first lookup transformer |
| `den256v2_{tr,te}.parquet`, `den_uncond256v2.pt` | `train_mlm.py` d256 / 6 / 20 | 26 density channels (flag `den256v2`); second model of mlmfill2 |
| `den384v3_{tr,te}.parquet`, `den_uncond384v3.pt` | `train_mlm.py` d384 / 8 / 30 | 26 density channels (flag `den384v3`); model of cmp3, mlmfill, mlmfill2, tokemb, rrd, bp2 |
| `den_{tr,te}.parquet`, `den_uncond.pt` | `train_mlm.py` d320 / 7 / 25, seed 777 | 26 density channels: the `den` block of the second lookup transformer and the cuML models |
| `den_v4_{tr,te}.parquet`, `den_uncond_v4.pt` | `train_mlm.py` d384 / 8 / 60 | 26 density channels (flag `denv4`); model of cmp4, mlmfill4, qfill |
| `den_v5_{tr,te}.parquet`, `den_uncond_v5.pt` | `train_mlm.py` d448 / 8 / 80 | 26 density channels (flag `denv5`) |
| `row_emb384.npy` | `build_knn.py emb` | 987,671 x 384 float16 row embeddings of `den_uncond384.pt` (flag `emb`, cuML embedding features) |
| `knn_f{0..4}.parquet` | `build_knn.py knn` | per fold, train rows then test rows: knn_r16, knn_r64, knn_rw, knn_dist (flag `knn`) |
| `cmp3_f{0..4}_{tr,te}.parquet` | `build_cmp.py` | 23 completion-response columns from `den_uncond384v3.pt` (flag `cmp3`) |
| `mlmfill_{tr,te}.parquet` | `build_mlmfill.py` | 27 fill + geometry columns `mf_*` from `den_uncond384v3.pt` (flag `mfill`) |
| `mlmfill2_{tr,te}.parquet` | `build_mlmfill2.py` | 15 two-model fill columns `mf2_*` (flag `mfill2`) |
| `tokemb_{tr,te}.parquet` | `build_tokemb.py` | 36 token-embedding PCA columns `tke{0,1,2}_<col>` (flag `tokemb`) |
| `rrd_{tr,te}.parquet` | `build_rrd.py` | 19 reconstruction-residual columns `rr_*`, `rrz_*` (flag `rrd`) |
| `bp2_f{0..4}_{tr,te}.parquet` | `build_bp2.py` | 26 budget-mixture posterior columns `bp2_*`, `cmp5_*` (flag `bp2`) |
| `cmp4_f{0..4}_{tr,te}.parquet` | `build_cmp.py` | 23 completion-response columns from `den_uncond_v4.pt` (runner flag `cmp4`, called `cmpv4` in the XGBoost / numeric-CatBoost runners) |
| `mlmfill4_{tr,te}.parquet` | `build_mlmfill.py` | 27 fill + geometry columns `mf_*` from `den_uncond_v4.pt` (flag `mfill4`) |
| `qfill_{tr,te}.parquet` | `build_qfill.py` | 14 posterior-shape columns `qf_*` (flag `qfill`) |
| `cllr_f{0..4}.parquet` | `train_mlm.py --mode cond` (optional) | class-conditional LLRs `cllr_*` (flag `cllr`, not read by the final models) |

`_tr` files hold the 691,369 train rows and `_te` files the 296,302 test rows, in train.csv / test.csv order; the
per-fold `_tr` files hold all train rows (inner-cross-fitted values for the outer-train rows of that fold).

## Requirements
numpy 2.2, pandas 2.3, pyarrow, scipy 1.16, scikit-learn 1.6, xgboost 3.2 (CUDA), torch 2.11 (CUDA), lightgbm 4.6
(`build_bp2.py`) and cuML 26.02 (`build_knn.py knn`).
