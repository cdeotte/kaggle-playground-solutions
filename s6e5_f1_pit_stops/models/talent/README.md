# TALENT deep tabular models, a SAINT-style transformer and a GrowNet-style network

Eleven neural models on one per-fold feature builder (`lib/fold_features.py`). Nine run through the
[TALENT](https://github.com/LAMDA-Tabular/TALENT) library (ModernNCA, TabR, NODE, ExcelFormer, Trompt, AMFormer, DCNv2,
MLP-PLR, AutoInt); two are local PyTorch networks with one logit (`lib/saint_style.py`, `lib/grownet_style.py`).
`train.py <pair>` selects the model from its config table.

**Features** (`lib/fold_features.py`, fitted inside each outer fold):
- **numerics**: the 11 raw numerics and 18 derived columns: tyre life / lap and stint / lap ratios, 1 - progress,
  lap - 100 x progress, degradation and lap-time delta per tyre lap, lap time per progress and per tyre lap,
  `1 / (position + 1) + 0.05 x position change`, progress x log1p(tyre life), degradation x delta, lap time -/+ delta,
  pit-window flag (0.18 < progress < 0.95) x tyre life / lap, |delta| x log1p(tyre life) + |degradation|,
  stint + progress, pit stops x lap, degradation x progress x log1p(tyre life);
- **categorical keys**, profile `low` (20): Driver, Compound, Race, the integer values of Year / PitStop / LapNumber /
  Stint / TyreLife / Position / Position_Change, bins (tyre life / 3, lap / 3, progress x 30, position / 2) and the crosses
  Race x Compound, Race x Year, Race x Stint, Compound x tyre-life bin, lap bin x Race, progress bin x Compound. Profile
  `rich` (34) adds lap time rounded to 0.1 / 0.01, delta to 0.01 / 0.001, degradation to 0.01 / 0.001, progress to 0.001,
  Driver x Compound, Driver x Race, Driver x Race x Stint, Race x Compound x tyre-life bin, Race x Compound x lap bin,
  Compound x delta (0.01) and Race x lap time (0.1);
- **frequencies**: one column per key, the normalized value counts over the fold's training rows + its original rows +
  the test rows;
- **original priors**: mean target and log1p count of 10 keys (Race, Compound, Driver, Race x Compound, Race x Year,
  Race x Stint, Compound x tyre-life bin, lap bin x Race, progress bin x Compound, Race x Compound x tyre-life bin) in the
  fold's original rows (unseen values: the original mean and 0).

That gives 69 numerics + 20 keys (`low`) or 83 numerics + 34 keys (`rich`).

**Rows**: outer fold k (`common.data.folds`) uses the fold seed 43 + k. With a `cap`, the 351k outer-train rows are cut to
a stratified random subsample of that size (fold seed); the builder is fitted on the kept rows. The fold's original rows
are the original rows of the other 4 folds (`common.data.orig_folds`, about 81k): they always feed the frequency and
prior columns, and the `orig_rows` models also append them to the training rows (after the cap). The held-out fold is
the validation set of every model.

**TALENT models**: TALENT's method for each `model_type`, numerics NaN -> mean and quantile-normalized (TALENT's
QuantileTransformer to a normal distribution), categorical keys as one-hot vectors (`tabr_ohe`: ModernNCA, TabR, MLP-PLR),
integer indices into the model's embeddings (`indices`) or binary-encoded columns appended to the numerics (`binary`).
Each model trains for `max_epoch` epochs (AdamW) and TALENT saves a `best-val` checkpoint at the epoch with the best first
value of its `metric()` (accuracy by default); `train.py` patches `metric()` to return AUC first (`auc_first_metric`), so
the epoch with the best held-out AUC predicts the held-out fold and the test rows (softmax of TALENT's 2 logits). Seed =
fold seed.

**Local networks** (`lib/tensor_data.py`): the numerics are standardized with the training rows' mean / std (clipped to
+-8) and every categorical key is indexed by its values in the training rows (0 = unseen value); seed = fold seed.
- **SAINT-style transformer** (`lib/saint_style.py`): a CLS token, one 96-d embedding per categorical key and 8 numeric
  tokens (BatchNorm + Linear of all numerics) go through 3 pre-norm transformer layers (4 heads, GELU, FFN 384, dropout
  0.12); the CLS output and a numeric MLP branch (512 -> 96) feed a LayerNorm + MLP head (512). BCE with
  pos_weight = sqrt(negatives / positives), AdamW lr 6e-4, weight decay 1e-4, cosine to 6e-5 over 18 epochs, gradient
  clip 5, batch 2048; the best held-out-AUC epoch predicts (patience 5).
- **GrowNet-style network** (`lib/grownet_style.py`): input = the numerics + a 16-d embedding of every categorical key;
  each stage is an MLP (256-256-1, BatchNorm, SiLU, dropout 0.05) and the logit is a learned scale x the sum of the
  stage outputs. Up to 24 stages; each new stage is followed by one epoch of BCE on the whole model (all stages and
  embeddings, fresh AdamW lr 8e-4, weight decay 1e-4, gradient clip 5, batch 4096); the model with the best held-out-AUC
  number of stages predicts (patience 7 stages).

Predictions are probabilities clipped to [1e-7, 1 - 1e-7]; test = mean of the 5 fold models.

```
python train.py modernnca_lowcard_ohe_cap160k
python train.py tabr_lowcard_ohe_cap220k
python train.py node_rich_binary_orig_rows
python train.py excelformer_rich_binary_orig_rows
python train.py trompt_lowcard_indices_cap240k
python train.py amformer_lowcard_indices_orig_rows
python train.py dcn2_rich_indices_orig_rows
python train.py mlp_plr_lowcard_ohe_cap300k
python train.py autoint_lowcard_indices_cap260k
python train.py saint_style_lowcard_orig_rows_cap300k
python train.py grownet_style_lowcard_orig_rows_cap300k
```
One GPU per run. `--folds 0,1` runs a subset of the folds (e.g. one process per GPU with `CUDA_VISIBLE_DEVICES`); finished
folds are kept in `parts/` and skipped (`--force` reruns them), and `preds/<pair>_{oof,test}.npy` is written when all 5
folds are done. Checkpoints go to `cache/<pair>/fold<k>/`.

No other repo folders are needed.

| Pair | Model | OOF AUC |
|---|---|---:|
| modernnca_lowcard_ohe_cap160k | ModernNCA; `low`, one-hot, cap 160k; dim 96, one block of 384, dropout 0.08, temperature 0.75, candidate sample rate 0.05, PLR-lite numeric embeddings (32 frequencies, scale 0.045, d 16); lr 8e-3, wd 2e-4, batch 512, 4 epochs | 0.941058 |
| tabr_lowcard_ohe_cap220k | TabR (FAISS retrieval); `low`, one-hot, cap 220k; d_main 192, context dropout 0.25, d_multiplier 2, 0 encoder / 1 predictor block, dropout0 0.12, LayerNorm, ReLU, PLR-lite (48, 0.045, 18); lr 4.5e-4, wd 1e-6, batch 1024, 6 epochs | 0.943445 |
| node_rich_binary_orig_rows | NODE; `rich`, binary encoding, all outer-train rows + original rows; 2 layers of 512 trees, depth 6, tree dim 2, sparsemax / sparsemoid; lr 1.2e-3, wd 1e-5, batch 4096, 15 epochs | 0.936121 |
| excelformer_rich_binary_orig_rows | ExcelFormer; `rich`, binary encoding, + original rows; 3 layers, 8 heads, d_token 192, pre-norm, token bias, dropout attention 0.15 / FFN 0.05 / residual 0.02, init scale 0.01, no feature / hidden mix; lr 1.8e-4, wd 1e-5, batch 1024, 12 epochs | 0.946912 |
| trompt_lowcard_indices_cap240k | Trompt; `low`, indices, cap 240k; 48 prompts, d 96, 3 cycles; lr 4.5e-4, wd 1e-5, batch 512, 8 epochs | 0.943453 |
| amformer_lowcard_indices_orig_rows | AMFormer; `low`, indices, + original rows; dim 128, depth 2, 4 heads, dropout attention 0.12 / FF 0.08, groups [32, 32], sum tokens [16, 8], product tokens [4, 4], clustering, target mode mix, CLS token; lr 1.6e-4, wd 1e-5, batch 1024, 10 epochs | 0.944542 |
| dcn2_rich_indices_orig_rows | DCNv2; `rich`, indices (embedding 24), + original rows; parallel 4 cross layers and 2 hidden layers of 384, dropout 0.08; lr 3e-4, wd 1.5e-4, batch 4096, 15 epochs | 0.939530 |
| mlp_plr_lowcard_ohe_cap300k | MLP-PLR; `low`, one-hot, cap 300k; layers 640-320-256, dropout 0.18, PLR-lite (64, 0.045, 24); lr 8e-4, wd 2e-4, batch 4096, 18 epochs | 0.943629 |
| autoint_lowcard_indices_cap260k | AutoInt; `low`, indices, cap 260k; 3 layers, d_token 96, 8 heads, dropout attention 0.12 / residual 0.02, post-norm, Kaiming init; lr 1.8e-4, wd 5e-5, batch 2048, 12 epochs | 0.942763 |
| saint_style_lowcard_orig_rows_cap300k | SAINT-style transformer (local); `low`, cap 300k + original rows | 0.941141 |
| grownet_style_lowcard_orig_rows_cap300k | GrowNet-style additive network (local); `low`, cap 300k + original rows | 0.942110 |

Packages: torch (CUDA), numpy, pandas, scikit-learn (1.6 or newer, see below), TALENT (`train.py` imports it for every
pair) with category_encoders and tqdm, delu and faiss (TabR), einops (AMFormer), requests (NODE).
TALENT (MIT license) is used unmodified, installed from commit `28ae777b19571e23525b67de4e1770d900d594b9`, e.g.
```
pip install --no-deps git+https://github.com/LAMDA-Tabular/TALENT.git@28ae777b19571e23525b67de4e1770d900d594b9
```
Its pinned dependency versions (torch 2.8.0, numpy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2, scipy 1.13.0) were not
used: our runs had torch 2.11, numpy 2.2, pandas 2.3, scikit-learn 1.6 and faiss 1.14. Use scikit-learn 1.6 or newer:
TALENT's `predict()` also scores the test rows (dummy all-zero labels) with the patched metric, and older versions raise
on a one-class AUC where 1.6 warns and returns NaN. The two runtime changes are in `train.py`: the `metric()` patch above
and, for the GrowNet-style pair, `torch.load` defaulting to `weights_only=False`.
