# RealMLP forensic chain: density completion, reconstruction, pair lift, global density and repair posteriors

One RealMLP recipe grown in six cumulative steps; every step is a separate OOF / test pair. The last one,
`realmlp_forensic_repair20`, is our best single model. All steps share the same network, preprocessing, seeds and
5 outer folds; they differ only in the blocks appended to the base matrix.

**RealMLP** (`lib/realmlp.py`): pytabkit `RealMLP_TD_Classifier`, hidden [768, 384, 192], PBLRD numeric embeddings
(16, 8), Mish, dropout 0.05, weight decay 0.017, batch 256, label smoothing 0.005, 32 internal ensemble members,
9 epochs with a 15% inner validation split picking the epoch (8 in every fold) followed by a refit on all
outer-training rows; seed 142 + fold. Categoricals are strings ('__MISSING__' for NaN), numeric NaNs get the
outer-training median. Test = mean of the 5 fold models.

**Base matrix** (135 columns, 12 categorical): raw predictors, an exact-value categorical token for each numeric
(its own embedding per exact value), 19 smooth screen-budget geometry columns (`lib/rowwise.py`), plus 95 columns
selected by name from the forensic fold matrix of `../lgbm_margin`: nested (5-way inner cross-fitted) target encodings
of the 9 numerics, TE / WOE sums, the base margin, imputed screen values and missing indicators, screen-response logits,
18 observed joint-PMI pair channels, unsupported-pair response, original-data binary LLRs and original label priors.

## The chain

| Step | Pair | Adds | Columns |
|---|---|---|---:|
| 1 | `realmlp_forensic_base` | the base matrix | 135 |
| 2 | `realmlp_forensic_density_completion` | 10 density-completion aggregates from a per-fold masked density transformer | 145 |
| 3 | `realmlp_forensic_reconstruction` | 71 cross-fitted XGBoost / autoencoder reconstruction channels | 216 |
| 4 | `realmlp_forensic_pairlift` | 10 exact pair-lift residuals | 226 |
| 5 | `realmlp_forensic_global_density` | global train+test density mixture: rebuilt completion aggregates, 24 surprisal / entropy + 27 fill channels, coarse and derived-coordinate lift banks, 39 generator-temperature channels; reconstruction removed | 259 |
| 6 | `realmlp_forensic_repair20` | 20 repair-posterior channels | 279 |

1. **Base** - the matrix above.
2. **Density completion** (`build_fold_density.py`, `build_completion.py`, `lib/lattice.py`, `lib/completion.py`).
   Per outer fold a masked lattice density transformer is trained on the outer-training predictors only (every
   predictor is a token over its observed values; width 384, 4 layers, 10 epochs, cells hidden with probability 0.15).
   For the 5 screen fields (daily, social, gaming, work/study, weekend) the field is masked and its full conditional
   distribution is integrated against an exact-token target-rate table (smoothing 20): expectation, std, entropy;
   observed cells keep their exact rate. Rate tables use outer-training labels only, and a training row reads the
   table fit without its splitmix64(id) mod 5 group. The RealMLP gets the 10 row aggregates (completion logit
   mean / std / min / max, context logit mean / std, missing-field context mean / std, missing count, confidence).
   Missing screen values are represented by a whole conditional distribution instead of a point fill.
3. **Reconstruction** (`build_reconstruction.py`, `lib/reconstruction.py`, `lib/autoencoder.py`). Every predictor is
   reconstructed from the other 11 by a GPU XGBoost expert (900 rounds, depth 8, lr 0.04) and by a masked
   Transformer autoencoder (Gaussian / softmax heads, MC dropout), never reading the target. Outer-training rows are
   reconstructed 3-way cross-fitted (splitmix64 groups of the id); validation and test rows by a fit on all
   outer-training rows. 71 compact channels: hybrid fills of missing cells (0.75 XGB + 0.25 AE on screen fields),
   XGB / AE disagreement, AE uncertainty, original-data LLR of the fill, error / disagreement / uncertainty aggregates
   and the screen geometry of the completed row.
4. **Pair lift** (`build_lift_banks.py`, `lib/lift.py`). For 9 exact two-decimal pairs, the residual of the pair's
   target count against the log-odds-independence baseline of its two marginals, shrunk with s = 20 (100 for
   notifications x app opens), plus the row mean. Inner StratifiedKFold(5, seed 777) cross-fit inside the outer-training
   rows.
5. **Global density** (`train_global_density.py`, `build_global_channels.py`, `build_global_completion.py`,
   `build_temperature.py`, `build_lift_banks.py`, `lib/global_density.py`, `lib/temperature.py`). Four target-free
   density transformers are trained on the predictors of all train + test rows (shared lattice encoder over train,
   test and original): g4 (width 384, 8 layers, 60 epochs), g5 (width 448, 80 epochs), g4orig8 (g4 + the original rows
   repeated 8 times) and g9unitarget (half of the rows hide exactly one observed field). Their equal mixture, tempered
   per field (T = 1.1 for daily / social / work / sleep / weekend, 1.2 for academic impact, 1.0 otherwise), replaces
   the fold state in the completion aggregates and gives 24 surprisal / entropy channels and 27 conditional-mean
   fill / residual-z / screen-geometry channels. The 71 reconstruction channels are dropped (the 10 pair-lift
   channels stay). Added at the same time:
   the coarse-lattice lift bank (8 pairs on 0.25-hour / 0.1-hour roundings and exact counts, s = 1, + mean), the
   derived-coordinate lift bank (utilization = parts / daily rounded to 0.01 and weekend - daily rounded to 0.25,
   paired with exact screen values, s = 1, + mean) and 39 generator-temperature channels of the 5 screen fields
   (synthetic vs original exact-value log counts, their power-law residuals, row aggregates).
6. **Repair posteriors** (`build_repair.py`, `lib/repair.py`). Complete rows concentrate at
   (social + gaming + work) / daily in [0.85, 0.95), consistent with the generator repairing implausible screen
   compositions. From the global mixture: for daily / social / gaming / work, the conditional mass inside the value
   interval that puts the row in that band and the band-conditioned shift of the conditional mean (z units), mass
   aggregates, and coupled terms (filled utilization and its delta-method std, normal band probability, distance to
   the band, projection size onto 0.90 x daily, missing screen count, observed band indicator, daily + sleep cap
   probability). Target-free.

## Commands
```
# step 1 (needs the forensic fold matrices of ../lgbm_margin: cache/fold_<k>/, see lib/paths.py)
python train.py realmlp_forensic_base                 # ~10 min / fold on one GPU
# step 2
python build_fold_density.py                          # ~3 min / fold
python build_completion.py
python train.py realmlp_forensic_density_completion
# step 3
python build_reconstruction.py                        # ~15 min / fold
python train.py realmlp_forensic_reconstruction
# step 4
python build_lift_banks.py                            # pairlift, cbank, wbank (CPU)
python train.py realmlp_forensic_pairlift
# step 5
python build_temperature.py
python train_global_density.py --component g4         # 40-60 min each
python train_global_density.py --component g4orig8
python train_global_density.py --component g5
python train_global_density.py --component g9unitarget
python build_global_channels.py
python build_global_completion.py
python train.py realmlp_forensic_global_density
# step 6
python build_repair.py
python train.py realmlp_forensic_repair20
```
Every per-fold script takes `--folds 0,1,2,3,4` (folds can run in parallel on different GPUs). `train.py` writes
`parts/<pair>_f<k>_{valid,test}.npy` per fold and `preds/<pair>_{oof,test}.npy` (probabilities) once all 5 folds exist.

Cache layout (`train` = the fold's outer-training rows, `valid` = the fold's rows, both in train.csv order; the two
`*_features` fold caches are also read by `../xgb_forensic`, `../catboost_forensic`, `../tabm_forensic` and
`../pairlift_models`):

| Path under `cache/` | Step | Content |
|---|---|---|
| `fold_density/fold_<k>/model_state.pt` | 2 | per-fold density transformer |
| `density_completion_response_production_features/fold_<k>/` | 2 | `{train,valid,test}_features.npy` (30 columns), `train_indices.npy`, `valid_indices.npy`, `manifest.json` (feature_names) |
| `reconstruction-production-v1/fold_<k>/` | 3 | the same files, 189 columns (compact 71, detailed 52, full 66; manifest also has view_counts) |
| `lift/fold_<k>/{pairlift,cbank,wbank}_{train,valid,test}.npy` | 4-5 | lift banks (10 / 9 / 5 columns) |
| `temperature/fold_<k>/{train,valid,test}.npy` | 5 | 56 temperature channels (the RealMLP uses 39) |
| `global_density/<component>/model_state.pt` | 5 | g4, g4orig8, g5, g9unitarget |
| `global_channels/{train,test}.npy` | 5 | all train rows / test rows, 24 + 27 columns |
| `global_completion/fold_<k>/{train,valid,test}.npy` | 5 | 30 completion channels of the global mixture |
| `repair/{train,test}.npy` | 6 | all train rows / test rows, 20 columns |

| Pair | Model | OOF AUC |
|---|---|---:|
| realmlp_forensic_base | RealMLP, base matrix | 0.970055 |
| realmlp_forensic_density_completion | + per-fold density-completion aggregates | 0.970106 |
| realmlp_forensic_reconstruction | + XGBoost / autoencoder reconstruction channels | 0.970214 |
| realmlp_forensic_pairlift | + exact pair lift | 0.970299 |
| realmlp_forensic_global_density | global train+test density mixture channels, lift banks, temperature | 0.970662 |
| realmlp_forensic_repair20 | + repair-posterior channels | 0.970706 |

Requirements: pytabkit (RealMLP), torch (CUDA), xgboost (CUDA), scikit-learn, scipy, pandas, numpy.
