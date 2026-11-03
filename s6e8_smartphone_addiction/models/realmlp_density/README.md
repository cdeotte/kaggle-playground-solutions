# RealMLP on the density / completion channel matrix

Eight pytabkit `RealMLP_TD_Classifier` pairs on one fold-wise channel matrix (`lib/`, also used by `../tabm_density`
and `../lookup_transformer`). Everything label-derived is nested inside the outer fold: outer-train rows get encodings
fitted on the other 4 inner folds (`StratifiedKFold(5, shuffle=True, random_state=777)`), outer-validation and test rows
get the map fitted on all outer-train rows. Target-free channels use train + test predictors.

**Channels** (`lib/assemble.py`, `lib/fields.py`, `lib/matrix.py`):
- **base system** (first two pairs, 201 columns): the 12 raw columns (categoricals as fixed ordinal codes), missing
  flags + count, budget geometry (`slack = daily - (social + gaming + work)`, shares, weekend - daily, distances to the
  region boundaries daily 6 / 8 and social 4, observed region), exact-value counts over train + test, original-data
  evidence (smoothed per-token LLR of the 7,500 original rows, 0.1-grid LLR of the 6 screen / sleep columns,
  amplification `log(synthetic / original frequency)`), 18 exact pair PMIs, predictor-only XGBoost screen imputers +
  fills + 3-class region posterior, an 11-column Monte-Carlo posterior block, kNN label responses (16 / 64 neighbours,
  distance-weighted, mean distance) in the row-embedding space of the MLM, the MLM's per-cell `log p(x_c | rest)`,
  entropy and row pseudo-log-likelihood; nested target encodings (smoothing 20) of the 12 exact-value tokens, 3 pair
  keys (daily x social and daily x weekend on the 0.1 grid, notifications x app_opens), 5 region x token keys and 8
  quantile-binned geometry keys; a second TE of the 12 tokens at smoothing 4.
- **lean system** (the six `lean` pairs): the same minus counts, MC, kNN, the base density channels, the region keys,
  the per-column missing flags and the weak original columns; TE at smoothing 1 averaged over 3 inner seeds (777,
  1778, 2779), the second TE at smoothing 20, two WOE sums (9 numerics; daily / social / weekend) - 121 columns - plus
  12 **smooth fields** (`sf`: cross-fitted Gaussian-blurred response surfaces, 1-D per numeric and 2-D for daily x
  social, daily x weekend, notifications x app_opens) and the channel blocks of `../density_features` /
  `../lift_banks` listed in `lib/matrix.py`: MLM completion response (`cmp3` / `cmp4`), MLM density channels of other
  MLM builds (`den384v3`, `denv4`, `denv5`), MLM expected-value fills with recomputed geometry (`mfill`, `mfill2`,
  `mfill4`), reconstruction residuals of observed cells (`rrd`), posterior width / skew of missing cells (`qfill`),
  pair-lift banks (`plift`, `cbank`, `wbank`).
- Then response columns (`te_`, `tes_`, `te2_`, `ter_`, `teg_`, `knn_r*`) go to the logit scale (clip 1e-4), numeric
  NaN are filled with the outer-train median, and the 12 exact-value tokens are added as high-cardinality categorical
  columns (`cattok`), so the net learns its own per-value lookup embeddings.

**Nets** (`train.py`, seed 42, test = mean of the 5 fold models):
- `fast`: n_ens 24 (library-internal ensemble in one fit), 64 epochs, batch 1024, lr 0.04, `pblrd` embeddings with
  plr_sigma 0.3, best epoch by 1 - AUC on a random 4% of outer-train, no refit.
- `battery`: n_cv 5 + n_refit 5 + n_ens 16, 128 epochs, batch 512, lr 0.04, plr_sigma 0.3 (5 internal CV fits choose
  the epochs, 5 refits train on 100% of outer-train).
- `lean`: 16-epoch selector on 85% of outer-train (best epoch by 1 - AUC on the other 15%), then one refit on 100% of
  outer-train at that epoch; n_ens 48, batch 256, lr 0.0175 (`lin_cos_log_15`), plr_sigma 0.94, hidden 768/384/192,
  mish, p_drop 0.05, wd 0.017, label smoothing 0.005, front scale.

```
# needs ../density_features (and ../lift_banks for the plift / cbank / wbank pairs) built first
python train.py realmlp_density_knn_mc
python train.py realmlp_density_battery_sf_emb
python train.py realmlp_density_lean_cmp3
python train.py realmlp_density_lean_mfill_rrd
python train.py realmlp_density_lean_plift
python train.py realmlp_density_lean_cbank
python train.py realmlp_density_lean_mlm4_wbank
python train.py realmlp_density_lean_mlm4_qfill
python train.py realmlp_density_lean_cbank --folds 2   # one fold (e.g. one process per GPU); the pair is written
                                                       # once all 5 folds are in parts/
```
Per fold on one A100 80GB: about 15 min (`fast`), 2.5 h (`battery`), 20-28 min (`lean`).

| Pair | Model | OOF AUC |
|---|---|---:|
| `realmlp_density_knn_mc` | base system + cattok, `fast` net | 0.969246 |
| `realmlp_density_battery_sf_emb` | base system + smooth fields + 48 PCA components of the MLM row embeddings + cattok, `battery` net | 0.969521 |
| `realmlp_density_lean_cmp3` | lean system + cmp3 + den384v3, `lean` net | 0.970110 |
| `realmlp_density_lean_mfill_rrd` | + mfill, mfill2, rrd | 0.970255 |
| `realmlp_density_lean_plift` | + plift | 0.970329 |
| `realmlp_density_lean_cbank` | + cbank | 0.970390 |
| `realmlp_density_lean_mlm4_wbank` | v4-MLM channels (cmp4, denv4, mfill4) instead of cmp3 / den384v3 / mfill / mfill2, + wbank | 0.970490 |
| `realmlp_density_lean_mlm4_qfill` | + qfill and the v5-MLM density channels (denv5) | 0.970567 |

Original configuration of each run (channel flags of `lib/`, net parameters as logged; unlisted net parameters are the
regime values above):

| Pair | cfg | feat | par |
|---|---|---|---|
| `realmlp_density_knn_mc` | `cnt,den,geo,imp,knn,mc,orig,pmi,rte,te,te2s` | `cattok` | `{"n_epochs": 64, "bs": 1024, "plr_sigma": 0.3, "val_fraction": 0.04, "n_ens": 24, "num_emb_type": "pblrd", "val_metric": "1-auc_ovr"}` |
| `realmlp_density_battery_sf_emb` | `cnt,te,te2s,orig,pmi,imp,geo,rte,den,knn,mc` | `cattok,emb,sf` | `{"n_epochs": 128, "bs": 512, "n_ens": 16, "n_cv": 5, "n_refit": 5}` |
| `realmlp_density_lean_cmp3` | `cmp3,geo,imp,orig,pmi,slim,te,te2s,teavg,woe` | `cattok,den384v3,sf` | `{"te_smooth": 1.0, "te2s_smooth": 20.0, "n_epochs": 16, "n_ens": 48}` |
| `realmlp_density_lean_mfill_rrd` | same | `cattok,den384v3,mfill,mfill2,rrd,sf` | same |
| `realmlp_density_lean_plift` | same | `cattok,den384v3,mfill,mfill2,plift,rrd,sf` | same |
| `realmlp_density_lean_cbank` | same | `cattok,cbank,den384v3,mfill,mfill2,plift,rrd,sf` | same |
| `realmlp_density_lean_mlm4_wbank` | `cmp4,geo,imp,orig,pmi,slim,te,te2s,teavg,woe` | `cattok,cbank,denv4,mfill4,plift,rrd,sf,wbank` | same |
| `realmlp_density_lean_mlm4_qfill` | same | `cattok,cbank,denv4,denv5,mfill4,plift,qfill,rrd,sf,wbank` | same |

The first run logged its flags as `base,cattok,cnt,den,geo,imp,knn,mc,orig,pmi,rte,te,te2s,telogit` (`base` = base
columns and `telogit` = logit-scale responses are always on in `lib/`; `cattok` is a feat flag here). Matrix widths in the original runs (numeric + 12 categorical): 213, 273, 194, 255, 265, 274, 264, 304.

Inputs (all from `cache_dir('density_features')` unless noted): `static_*`, `tok_*` (every pair); `mc_*`, `knn_f{k}`,
`den384_*` (base system: density channels of the d384 / 6-layer / 24-epoch MLM, the model behind `row_emb384.npy` and
`knn_f{k}`); `row_emb384.npy` (battery); `cmp3_f{k}`, `den384v3_*`, `mlmfill_*`, `mlmfill2_*`, `rrd_*`, `mlmfill4_*`,
`den_v4_*`, `den_v5_*`, `qfill_*`, `cmp4_f{k}` (lean blocks); `plift_f{k}`, `cbank_f{k}`, `wbank_f{k}` from
`cache_dir('lift_banks')`.

Requirements: pytabkit (torch, CUDA), scikit-learn, scipy, pandas, pyarrow.
