# Source-row structural models: donor attention, genealogy graph, transition likelihood

Three models built on one idea: competition rows look like mosaics of fragments of rows of the 7,500-row original
dataset, so each row is linked to its most compatible original rows ("donors") separately for six semantic field blocks.

**Donors** (`build_donors.py`, `lib/donors.py`, GPU). Blocks: screen core (daily, social, gaming, work/study hours),
screen + weekend (daily, social, weekend), engagement (daily, weekend, notifications, app opens), behavior (social,
gaming, notifications, app opens), lifestyle (age, daily, work/study, sleep), profile (age, sleep + gender, stress,
academic impact). Per block and row the 16 original rows with the lowest score are kept:
`score = mean robust-scaled squared distance over the row's observed block fields (categorical mismatch 0.75) - 0.75 x
exact-match fraction`, robust scales `(q75 - q25) / 1.349` of the original data. Retrieval reads no labels; original
labels and severities (`addiction_level`) enter the models only as attributes of the retrieved donors.

## Models

**source_attention_net** (`train_source_attention.py`, `lib/attention_model.py`, GPU): a neural network with
- a context branch on 404 fold features (the 333-column forensic matrix of `../lgbm_margin` + the first 71 compact
  reconstruction columns of `../realmlp_repair`, standardized on outer-train, NaN -> 0, clip ±10) and the 12 missing
  flags: LayerNorm, 416 -> 512 -> 256, residual block;
- donor cross-attention: one 64-d query per block attends over its 16 donor tokens (30 inputs each: donor numerics,
  deltas to the row's observed numerics, donor categoricals, match flags, original label, severity, retrieval score,
  distance kernel, exact fraction, rank); logits `q.k / 8 - softplus(a_b) score + c_b exact`; per block the attended
  token and 5 summaries (label, severity, score, exactness, entropy) feed a 414 -> 256 -> 192 branch;
- 4 experts on [context, donors], mixed by a router that sees only the 12 missing bits and the 6 block coverages;
- output `base_scale x strong__selected_base_margin + bias + routed residual` (base margin from the forensic matrix);
- auxiliary loss 0.015 x MSE of the row's observed robust-scaled numerics reconstructed from the attended tokens.

AdamW lr 2e-3, weight decay 2e-4, one-cycle (20% warm-up), batch 2048, bf16, label smoothing 0.002, gradient clip 2.
Per outer fold (seed 1701 + fold): the outer-train rows are split 80 / 20 by `splitmix64(id) mod 5`; 8 epochs on the
80% pick the epoch with the best AUC on the 20% (6, 6, 6, 6, 5 in our run); a fresh model is then trained on all
outer-train rows for that many epochs and predicts the held-out fold and test.

**source_graph_lr** (`train_graph_lr.py`, `lib/graph.py`, CPU): original rows are latent hub nodes of a graph over the
competition rows. 256 features: 24 raw (numerics, categorical codes, missing flags), 84 label-free structural columns
(per block: coverage, pattern, best score, best exact fraction, nearest donor id; per kernel k1 / k4 / k16: weight
entropy, donor-weighted original label and severity), 4 anchors from the `../lgbm_margin` forensic matrix (raw and
screen TE WoE sums, selected base margin and probability) and 144 label messages from the outer-train labels:
smoothed donor posteriors (m 4, 24, 128) + log support per block and kernel, and smoothed label means of the rows sharing
the same (nearest donor, pattern) per block (m 12, 64), nearest-donor pairs over all 15 block pairs (m 16, 96) and
nearest-donor triples over 3 block triples (m 32, 160), each with its log count. Kernels: k1 the nearest donor, k4 the 4
nearest with weights `exp(-1.25 Δscore)(0.35 + 0.65 exact)`, k16 all 16 with `exp(-0.60 Δscore)(0.25 + 0.75 exact)`,
normalized. Messages of an outer-train row exclude the row's own weight and label (and use a row-excluded prior);
held-out and test rows use all outer-train rows. Model: median fill, standardize, clip ±12, scikit-learn L2 logistic
regression C 0.08 (lbfgs, 300 iterations).

**source_transition_likelihood** (`train_transition.py`, `lib/transition.py`, CPU): a generative classifier with no
fitted discriminative model. `logit = base LLR + 0.5 x transition LLR`.
- Base LLR: per-class Dirichlet-smoothed histograms (outer-train rows, 32 quantile bins, pseudo-count 32) of the 9
  numerics (weight 0.8) and 13 derived contrasts (0.35), the 3 categoricals (0.5) and the 12-bit missingness pattern
  (0.8, pseudo-count 256); weighted mean of the log-likelihood ratios + prior log odds.
- Transition LLR: per block the top-4 donors with weights `exp(-1.25 Δscore)(0.25 + 0.75 exact)`; per donor events:
  standardized numeric residual bin (exact match separate) x donor-value quintile, donor category x observed category,
  retrieval score bin x exact bin (0.35), donor label x severity (0.5), donor id (0.35, pseudo-count x 8); per-class
  tables (pseudo-count 48) fitted on outer-train rows weighted by the donor weights; per block a logsumexp over the
  donors under each class; mean over covered blocks.

Test = mean of the 5 fold models for all three.

## Inputs from other folders
- `../lgbm_margin` (`source_attention_net`, `source_graph_lr`): `cache/features/fold_<k>/{train,valid,test}_numeric.npy`,
  the 333-column fold forensic matrix (rows: outer-train rows, held-out rows, test rows), names in
  `cache/features/fold_<k>/manifest.json` (`base_numeric_names`).
- `../realmlp_repair` (`source_attention_net`): `cache/reconstruction/fold_<k>/{train,valid,test}.npy`, the 71 compact
  reconstruction channels (`build_reconstruction.py`).

The paths are in `lib/inputs.py`. `source_transition_likelihood` needs only the donors.

## Steps
```
python build_donors.py                # cache/donors/{train,test}/   (GPU)
python train_source_attention.py      # -> preds/source_attention_net_{oof,test}.npy   (GPU, ~2 min per fold on an A100)
python train_graph_lr.py              # -> preds/source_graph_lr_{oof,test}.npy        (CPU, ~2.5 min per fold)
python train_transition.py            # -> preds/source_transition_likelihood_{oof,test}.npy   (CPU, ~10 s per fold)
```
`--folds 0,1` runs a subset; finished folds are kept in `parts/` and skipped; the pair is written once all 5 folds
exist. Variants: `train_source_attention.py --variant attention_noaux|context_control`, `train_graph_lr.py --variant
linear_control` (no label messages), `train_transition.py --variant control|k1_w010|...|k8_w050` (pair: `k4_w050`).

## Pairs

| Pair | Model | OOF AUC |
|---|---|---:|
| source_attention_net | mask-routed source-attention network, forensic + reconstruction context, auxiliary loss | 0.967932 |
| source_graph_lr | source-genealogy graph messages + structural features, L2 logistic regression | 0.959733 |
| source_transition_likelihood | class-conditional histogram + top-4 donor transition likelihood | 0.911932 |

Packages: torch (CUDA), numpy, pandas, scipy, scikit-learn.

## Gaps
- Our run also scored `source_graph_lr` on an inner 80 / 20 split of the outer-train rows for reporting; that score
  does not affect the predictions and is not ported.
- Fold predictions are kept as float32 in `parts/` (repo convention); our run kept float64.
