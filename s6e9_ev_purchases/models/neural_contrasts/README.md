# Neural and copula contrasts

Seven residual carriers for the stacker. A *contrast* trains the same model twice with the same seeds, once with a block
of label-free features (`real`) and once with that block set to 0 (`zero`); the seed-averaged logit difference
`real - zero`, times a fixed factor, is added to a base logit. Five carriers form a chain: each is the previous carrier's
fold logit plus new contrasts. The two copula carriers contrast a density-ratio XGB C with a smooth GLM / GAM blend G or
with a GPT-2 residual N. Every carrier is formed per outer fold on [held-out rows, test rows]; OOF = the held-out part,
test = mean of the 5 fold probabilities. Pairs are written to `preds/` as logits (`logit(clip(p, 1e-15))`, clipped to
±30), since these carriers come within 1e-10 of 0 and 1.

## Models

**income_mixture_contrast** = `gen0 + 4 (n2_reg_real - n2_reg_zero)`, seeds 42 and 43.
`gen0` (`income_generator.py`) is a semi-supervised generator of the exact income inside its parent (income // 1000,
leading BPE piece), one distribution per class, fit on the labelled outer-train rows plus every other row weighted by
an income-excluding context teacher; its prediction is the teacher logit plus the generator's class log-likelihood
ratio of the row's income. `n2_reg_*` (`contrast_mlp.py`) is an 8-member MLP on the nested response features of
`../income_mixture_xgb` plus its 4 EM income-mixture columns (`real`) or with those 4 columns at 0 (`zero`), on top of the
known-source offset.

**income_context_svd_contrast** = `income_mixture_contrast + 4 E + 4 G`, 16 seeds (42-57).
E: the MLP with vs without 8 columns of a label-free income-context SVD embedding (`income_embedding.py`).
G: the MLP with vs without 4 generator deltas (`income_generator.py --nested / --assemble`: rank-4 minus rank-0
generator log-likelihood ratio and entropies, inner-fold cross-fitted for the training rows).

**fine_context_contrast** = `logit(income_context_svd_contrast) + 4 (C + F + I + N)`, 16 seeds; all MLPs here also get the
8 embedding and 4 generator-delta columns. C: 8 core income-context factors (`context_bank.py`). F: 13 context entropies of
a label-free log-bilinear factorization (`density_factors.py`). I: with vs without the embedding columns.
N: a probit-link interaction, `- (embedding + generator) + (embedding only) + (generator only) - (neither)`, all four
without the EM mixture columns.

**coarse_context_contrast** = `fine_context_contrast + 4 (coarse_entropy_real - coarse_entropy_zero)`, 16 seeds;
27 coarse income-parent entropy columns (`coarse_entropy.py`).

**masked_density_contrast** = `0.5 clip(logit(coarse_context_contrast), ±30) + 4 L + 2 P + 2 M`, clipped to ±30,
16 seeds; the MLPs also get the 27 coarse columns. L: leave-one-out context lifts of (income // 100, piece) groups over the
original data's marginals; P: a 16-dim profile of (income // 5000, piece) groups (`distribution_bank.py`); M: 21 columns
of a cross-fitted masked-predictor density network (`masked_density.py`).

**copula_gam_contrast** = `2 z(C) - z(G)`, clipped to ±30.
C (`copula_xgb.py`, seeds 42-49): XGB on the information-aware EM base features of `../income_mixture_xgb`, its 4
inc_global EM columns and 6 class-conditional density-ratio expert columns (`copula_density.py`: XGB real vs
field-permuted rows per class, nested pools), started from the nested response-logit margin R (`response_margin.py`).
G = `sigmoid(0.5 z(sparse GAM) + 0.5 probit response margin)` (`sparse_gam.py`, `response_margin.py --link probit`).

**copula_gpt2_contrast** = `3 z(C) - 2 z(N)`, clipped to ±30. N (`gpt_residual.py`): GReaT replica e55 fine-tuned per fold
on the predictor text to learn a correction to R; the model after epoch 1 of a 2-epoch schedule, 2 field orders at
inference.

Each script's docstring has the details and every setting.

## Inputs from other folders
- `../income_mixture_xgb` (run it first): `cache/stage1` (prepared tables, `matched_f<k>/` fold matrices with
  `manifest.json` names / types, `semi_f<k>.npz`, `source_base.npy`), `cache/stage2/teachers/income_f<k>_*.npz`, and for
  the copula carriers `cache/em_info/{base,features_gpu}/fold_<k>/` (train / query packs + `manifest.json` names) and its
  `lib.py` (`Data`, `RAW`, `splits`, `train_noise`) and `compact.py` (`select`, `build`, `source_rule`).
- `../gpt2_replica`: `checkpoints/e55.pt`, loaded through `common/gpt_rows.py`.

## Steps
`contrast_mlp.py` runs every listed config for seeds 42-57 and folds 0-4 by default (`--seeds 42,43`, `--folds 0,1`
run subsets; finished runs are skipped, so several processes can share the work).

```
# income_mixture_contrast
python income_generator.py                     # gen0 + gen4_reg, outer folds, seeds 42, 43
python contrast_mlp.py --configs n2_reg_real,n2_reg_zero --seeds 42,43
python assemble.py income_mixture_contrast

# income_context_svd_contrast (builds on the runs above)
python income_generator.py --nested            # gen0 + gen4_reg on the 5 inner folds of every outer fold
python income_generator.py --assemble          # generator bank
python income_embedding.py
python contrast_mlp.py --configs emb8_real,emb8_zero,gen_difference_real,gen_difference_zero
python assemble.py income_context_svd_contrast # also writes its fold predictions to cache/stage4

# fine_context_contrast (anchored on the stage 4 fold predictions)
python context_bank.py
python density_factors.py
python contrast_mlp.py --configs core_income8_real,core_income8_zero,df8_a5_entropy_real,df8_a5_entropy_zero,link_logit_embedding_real,link_logit_embedding_zero,cube_probit_6,cube_probit_2,cube_probit_4,link_probit_all_zero
python assemble.py fine_context_contrast       # fold logits -> cache/stage5

# coarse_context_contrast (anchored on the fine_context_contrast fold logits)
python coarse_entropy.py
python contrast_mlp.py --configs coarse_entropy_real,coarse_entropy_zero
python assemble.py coarse_context_contrast     # fold predictions -> cache/stage5

# masked_density_contrast (anchored on the coarse_context_contrast fold predictions)
python distribution_bank.py
python masked_density.py --prepare
python masked_density.py --fit
python contrast_mlp.py --configs surprise_lift_real,surprise_lift_zero,dist_p5000bpe_profile_real,dist_p5000bpe_profile_zero,masked_lift_completion_real,masked_lift_completion_zero
python assemble.py masked_density_contrast

# copula_gam_contrast
python response_margin.py --link logit --name response_logit
python response_margin.py --link probit --name response_probit
python sparse_gam.py
python copula_density.py
python copula_xgb.py
python assemble.py copula_gam_contrast

# copula_gpt2_contrast (needs response_logit and copula_xgb.py)
python gpt_residual.py                         # replica environment, see below
python assemble.py copula_gpt2_contrast
```

Compute on one GPU: about 1,800 MLP fits of 15-25 s each, 120 generator fits of about 15 s, 5 masked-density fits of
about 1.5 min, 30 density-expert pools and 10 response-GLM folds of seconds each, 40 copula XGB fits of about 35 s,
5 GAM fits of about 3 min (CPU), and 5 GPT-2 fold runs of about 13 min plus inference.

## Caches
```
cache/stage3/  embedding_income_all.npy, embedding_income_f{0,4}.npz, generative_f{0,4}.npz (+ .json names)
cache/stage4/  embedding_income_f{1,2,3}.npz, generative_f{1,2,3}.npz, average16_fold<k>.npz
cache/stage5/  core_income_all.npy, df8_a5_{real,null,all}.npy, coarse_entropy_all.npy, export_all16_f<k>.npz,
               coarse_all16_f<k>.npz
cache/stage6/  dist_p100bpe_all.npy, dist_p5000bpe_all.npy, masked_inputs.npz, masked_density_f<k>.{pt,npz},
               masked_density_all.npy
cache/contrast/      margins/<name>/fold_<k>/, additive_runs/, copula/densities/, copula/models/
cache/gpt_residual/  gpt_runs/lr5e5/fold_<k>/epoch_1/
runs/<config>_s<seed>_f<k>/predictions.npz   MLP and generator fits
```
Per-fold banks (`*_f<k>.npz`) hold `fit` / `query` row indices and `train_features` / `query_features`; all-row banks
(`*_all.npy`) have one row per train + test row (train first). Other models read these banks by their file names.

## Pairs

| Pair | Model | OOF AUC |
|---|---|---:|
| income_mixture_contrast | income generator + 4 × EM-mixture MLP contrast | 0.942705 |
| income_context_svd_contrast | + 4 × income-context embedding + 4 × generator-delta contrasts | 0.941676 |
| fine_context_contrast | + 4 × (core factors + factorization entropy + embedding + probit interaction) | 0.937762 |
| coarse_context_contrast | + 4 × coarse parent-entropy contrast | 0.934736 |
| masked_density_contrast | 0.5 × previous + 4 × lift + 2 × profile + 2 × masked density | 0.932492 |
| copula_gam_contrast | 2 z(copula XGB) − z(sparse GAM + probit GLM) | 0.945777 |
| copula_gpt2_contrast | 3 z(copula XGB) − 2 z(GPT-2 residual) | 0.944802 |

## Notes
- **Environment.** `gpt_residual.py` ran in the replica environment (`../gpt2_replica/requirements.txt`,
  transformers 4.26.1): its attention patch replaces `GPT2Attention._attn`. Everything else runs in the base environment
  (numpy, pandas, scipy, scikit-learn, xgboost, torch).
- **Not bit-reproducible.** GPU training (TF32 matmuls, atomic adds) and the masked density in particular give close, not
  identical, numbers on a rerun.
- **Interfaces.** The scripts assume the `../income_mixture_xgb` names listed above (`lib.py`, `compact.py`, the cache file
  names and the `names` / `feature_names` / `feature_types` keys of its manifests). This code was not rerun after the port.
