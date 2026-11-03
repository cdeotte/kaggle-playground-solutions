# Deep & Cross Network (DCNv2) on the channel matrix and exact-value token embeddings

**Inputs** (`build_cache.py`, built per outer fold with `../xgb_density_margin/lib`): a base matrix of the 12 raw columns,
missing counts, budget geometry, original-data LLR / amplification, 18 pair PMIs, imputer fills and region posterior,
nested TEs of the 12 exact tokens, 3 pair keys and 8 geometry bins (smoothing 1, averaged over 3 inner seeds) and a second
TE of the tokens (smoothing 20), 2 WOE sums and the smooth response fields, plus
- `mlm3` (253 columns): completion response `cmp3_f*`, MLM density `den384v3`, MLM fills `mlmfill`, `mlmfill2`,
  reconstruction residuals `rrd` and the pair lifts `plift_f*`;
- `mlm4` (292 columns): `cmp4_f*` (completion block of the v4 MLM), `den_v4`, `den_v5`, `mlmfill4`, `rrd`, `plift_f*`,
  `cbank_f*`, `wbank_f*` and `qfill`.
Every label-derived channel is cross-fitted inside the outer-train rows (5 inner folds, seed 777). TE channels become
logits, missing values get the outer-train medians. The 12 columns also enter as exact-value token codes (vocabulary over
train + test; missing is a token).

**Model** (`dcn_model.py`, `train.py`): K = 16 members trained jointly in one fit (independent weights, prediction = mean of
their probabilities). Deep tower: 16-d token embeddings + all channels (learned per-channel front scale) + PLR embeddings
of every channel (16 learned frequencies, 4 outputs, ReLU) -> 512-512-256 SiLU MLP, dropout 0.05. Cross tower in field
space: the 12 token embeddings + one 16-d field per channel family (te_, tes_, te2_, teg_, woe_, cmp_, dn*_, mf_, mf2_,
plift, cb_, wb_, qf_, sf1_, sf2_, pmi_, ollr_, amp_, imp_, fill_, share_ and the rest), LayerNorm, 2 cross layers
`x + g * (x0 * (W x + b))` with zero-initialised gates g. Whole token embeddings are dropped with p 0.1 in training.
Channels are standardized on the outer-train rows with a log tail compression beyond 4 sd. AdamW (lr 2e-3, weight decay
0.01, embeddings 1e-4), batch 1024, 2% warm-up + cosine over n_epochs = 8, label smoothing 0.005, gradient clip 5; the
front scale trains at 6x the lr and the cross tower at `cross_lrf` x the lr. Seed 42.
**Epoch selection**: trained on 6/7 of the outer-train rows (StratifiedKFold(7, seed 777), first split) for the full
schedule, inner AUC after each epoch smoothed with a 3-point window, argmax (epoch 3 in every shipped fold); a fresh model
is then trained on all outer-train rows for that many epochs of the same schedule. One model per fold; test = mean of the 5
fold models (probabilities).

```
python build_cache.py mlm3       # cache/X_f{k}_*.npy, cols_f{k}.json, token codes (also read by ../ydf)
python build_cache.py mlm4       # cache/mlm4/
python train.py dcn_te_mlm3_plift
python train.py dcn_te_mlm4_banks
```
Run `../density_features` and `../lift_banks` first; `build_cache.py` imports `../xgb_density_margin/lib` (keep the folders
side by side). Building one fold of a matrix takes about a minute (CPU); training one fold about 2.5 minutes on one GPU.
`--folds 0,1` runs a subset; `train.py` writes `preds/<pair>_{oof,test}.npy` when the last fold is done.

| Pair | Model | OOF AUC |
|---|---|---:|
| dcn_te_mlm3_plift | mlm3 matrix (253 channels + 12 tokens), cross tower lr x 0.2 | 0.969464 |
| dcn_te_mlm4_banks | mlm4 matrix (292 channels + 12 tokens), cross tower lr x 0.1; fold 0 with the 12-epoch schedule (as shipped), folds 1-4 with 8 epochs | 0.969582 |

Requirements: torch (CUDA), numpy, pandas, pyarrow, scipy, scikit-learn.
