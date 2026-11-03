# Lookup-transformer initialized from the masked-lattice MLM

A custom PyTorch transformer that reads every row as one token per column. Each column token is the column's
exact-value embedding (the vocabulary of the masked-lattice MLM of `../density_features`: 0 = real missing, 1 = mask,
2.. = sorted exact values, so the missing state is a learned embedding) plus, gated by observedness, a periodic (PLR)
embedding of the rank-gauss value and a linear map of the column's **evidence vector** (standardized):
- `lookup_transformer_evidence7` (7 dims): nested TE logit (smoothing 20), sharp TE logit (smoothing 4), original-data
  LLR, log1p exact-value count, amplification, MLM `log p(x_c | rest)`, MLM entropy;
- `lookup_transformer_evidence11_fills` (11 dims): + the MLM completion logit (`cmp3`), the reconstruction z-residual
  (`rrd`), the budget-posterior fill and sd (`bp2`).

Group tokens (MLP-embedded standardized inputs; a learned NA embedding replaces the token when one of its inputs is
missing): budget geometry (11), region posterior / imputer fills (9), kNN responses in the MLM embedding space (4),
Monte-Carlo posterior (11), pair / region / geometry TE logits (16); evidence11_fills adds a fill-geometry token
(MLM-fill geometry, completion aggregates, repair probability; 12); then a token of the 12 missing bits, and a CLS
token. The fold-wise TEs, kNN responses etc. come from the base channel system
of `../realmlp_density/lib` (nested inside the outer fold).

**Model**: 6 parallel transformer encoders (d 256, 8 layers, 8 heads, norm-first, GELU, dropout 0.1) on the same
tokens, each with its own LayerNorm + MLP head on CLS; logit = mean of the 6 heads, loss = mean of their BCE. Column
embeddings and the first 6 encoder layers of every replica are initialized from `den_uncond256.pt` (the d256 / 6-layer
MLM trained on train + test predictors, no labels).
**Training**: 92% of outer-train (stratified split, seed 99) for fitting, AdamW lr 2e-3 (weight decay 1e-5, 1e-4 on
the embeddings), OneCycle with 10% warm-up, batch 1536, up to 40 epochs, bf16 autocast, gradient clip 1, EMA 0.999 of
the weights; from epoch 3 the EMA weights are scored on the remaining 8% after each epoch, patience 6; the best EMA
weights predict the validation fold and the test rows (sigmoid). No mask augmentation. Seed 42; test = mean of the 5
fold models.

```
# needs ../density_features built first (and ../realmlp_density/lib, imported)
python train.py lookup_transformer_evidence7
python train.py lookup_transformer_evidence11_fills
python train.py lookup_transformer_evidence7 --folds 4   # one fold (e.g. one process per GPU)
```
About 40-50 min per fold on one A100 80GB.

| Pair | Model | OOF AUC |
|---|---|---:|
| `lookup_transformer_evidence7` | 7-dim column evidence, 5 group tokens + missing bits | 0.968608 |
| `lookup_transformer_evidence11_fills` | 11-dim column evidence (+ completion, reconstruction residual, budget-posterior fill / sd) + fill-geometry token | 0.968977 |

Original configuration (both pairs): channel flags `cnt,te,te2s,orig,pmi,imp,geo,rte,den,knn,mc`; parameters
`{"d": 256, "layers": 8, "epochs": 40, "pre": 1, "aug": 0.0, "emb_wd": 0.0001, "patience": 6, "reps": 6, "bs": 1536}`
(heads 8, lr 2e-3, dropout 0.1, holdout 0.08).

Inputs from `cache_dir('density_features')`: `static_*`, `tok_*`, `mc_*`, `knn_f{k}`, `den_uncond256.pt`, and the MLM
density channels as in the original runs: `den384_*` (d384 / 6-layer / 24-epoch MLM) for evidence7, `den_*` (d320 /
7-layer / 25-epoch MLM, seed 777) for evidence11_fills, which also reads `mlmfill_*`, `rrd_*`, `cmp3_f{k}`, `bp2_f{k}`.

Requirements: torch (CUDA, bf16), scikit-learn, pandas, pyarrow.
