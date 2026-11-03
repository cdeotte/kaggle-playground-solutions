# TabM on the lean density / completion channel matrix

Two pytabkit `TabM_D_Classifier` pairs on the lean fold-wise channel matrix of `../realmlp_density/lib` (see that
README for the channel recipe: nested exact-token TE at smoothing 1 averaged over 3 inner seeds + a second TE at
smoothing 20, pair / geometry-bin TEs, WOE sums, original-data LLR / amplification, pair PMIs, imputers / region
posterior, smooth response fields, MLM channel blocks, response columns on the logit scale, outer-train median fill,
the 12 exact-value tokens as categorical columns). Both pairs add `bp2`: budget-mixture posteriors of the screen cells
(repair-vs-organic mixture with a view-specific repair probability, sleep-cap truncation, a coupled sampler for rows
missing two budget components) and completion responses under them.

**Net**: `tabm-mini`, k = 32 members, piecewise-linear numeric embeddings (d_embedding 16), 128 epochs, lr 1e-3,
batch 512; the best epoch by 1 - AUC on a random 4% of outer-train, that checkpoint is kept (TabM has no refit); seed
42; test = mean of the 5 fold models.

**Per-fold logit standardization** (`standardize.py`, used by `tabm_density_mlm4_banks_std`): without a refit the 5 fold
models come out with different logit scales. Each fold model's logits are standardized with its own statistics (fold
identity and predictions only): OOF rows of fold k get `sigmoid((z - mean z) / sd z)` with z = the logits of fold model
k's validation predictions; test = `sigmoid(mean over k of (t_k - mean t_k) / sd t_k)` with t_k = the logits of fold
model k's test predictions (logit of float32 p clipped to [1e-7, 1 - 1e-7]). Raw pooled OOF of this stream: 0.969757.

```
# needs ../density_features and ../lift_banks built first
python train.py tabm_density_mfill_bp2
python train.py tabm_density_mlm4_banks_std     # runs the stream's folds, then standardize.py
python standardize.py                           # (re)writes the standardized pair from the 5 fold parts
python train.py tabm_density_mfill_bp2 --folds 3   # one fold (e.g. one process per GPU)
```
About 23 min per fold on one A100 80GB.

| Pair | Model | OOF AUC |
|---|---|---:|
| `tabm_density_mfill_bp2` | lean cfg + cmp3; sf, den384v3, mfill, mfill2, rrd, bp2 | 0.969672 |
| `tabm_density_mlm4_banks_std` | lean cfg + cmp4; sf, denv4, mfill4, rrd, plift, cbank, wbank, bp2; per-fold logit standardization | 0.970078 |

Original configuration of each run (matrix width incl. 12 categoricals: 281 and 290):

| Pair | cfg | feat | par |
|---|---|---|---|
| `tabm_density_mfill_bp2` | `cmp3,geo,imp,orig,pmi,slim,te,te2s,teavg,woe` | `bp2,cattok,den384v3,mfill,mfill2,rrd,sf` | `{"te_smooth": 1.0, "te2s_smooth": 20.0, "n_epochs": 128, "lr": 0.001}` |
| `tabm_density_mlm4_banks_std` | `cmp4,geo,imp,orig,pmi,slim,te,te2s,teavg,woe` | `bp2,cattok,cbank,denv4,mfill4,plift,rrd,sf,wbank` | `{"te_smooth": 1.0, "te2s_smooth": 20.0, "n_cv": 1, "n_refit": 0, "vf": 0.04, "n_epochs": 128, "bs": 512, "lr": 0.001, "tabm_k": 32, "num_emb": "pwl", "d_emb": 16}` |

Inputs: `static_*`, `tok_*`, `cmp3_f{k}` / `cmp4_f{k}`, `den384v3_*` / `den_v4_*`, `mlmfill_*`, `mlmfill2_*` /
`mlmfill4_*`, `rrd_*`, `bp2_f{k}` from `cache_dir('density_features')`; `plift_f{k}`, `cbank_f{k}`, `wbank_f{k}` from
`cache_dir('lift_banks')`.

Requirements: pytabkit (torch, CUDA), scikit-learn, scipy, pandas, pyarrow.
