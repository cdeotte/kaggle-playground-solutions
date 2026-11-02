# Income-mixture XGBoost

Four XGBoost models on fold-local (nested) features. The common idea: rows that share an exact income value share a
class rate. For each outer fold, the rate of every exact income is estimated by EM from the labeled counts of the
label pool plus the rows outside it, whose posterior class probabilities combine the value's current rate with the
likelihood ratio of an income-blind "context teacher" (XGB on the other fields, calibrated per parent
= (income // 1000, leading GPT-2 BPE piece)). Every model starts boosting from a nested ridge-logistic margin and
averages seeds in logit space per fold; test = mean of the 5 fold models.

- **xgb_income_mixture**: 145 columns: raw and label-free count / original-data / source-law features, GPT-2 BPE
  keys of income and commute, target encodings (alpha 1 and 20) cross-fitted inside the outer-train rows,
  source-conditional ridge intercepts, and the 4 exact-income EM columns. Depth 7, eta 0.03, 400 rounds,
  4 parallel trees, seeds 42-49.
- **xgb_income_context_shape**: the same + 9 context-shape columns (`artifact_features.py`): how the observed
  subsidy / anxiety / concern frequencies of each income value differ from what its class mix predicts. Seeds 42-49.
- **xgb_full_context_contrasts**: 259 columns: the same + banks of `../neural_contrasts` (income embedding, generator
  context deltas, parent entropies, leave-one-out lifts, profiles, masked-density lifts), income digits and 4 income
  target-encoding contrasts. Seeds 42-57. Its runs are also the teacher of `../distillation`.
- **xgb_information_mixture**: a separate pipeline (`em_pipeline.py`, `lib.py`): static feature blocks of
  `../response_features`, nested encodings, source-rule intercepts, the exact-income EM columns and 14 observed-information
  columns (Newton solve of the mixture likelihood at unlabeled-evidence weights 0.25 / 1 / 4). Start margin =
  0.5 x the nested response margin of `../response_features` + 0.5 x the EM Bayes posterior margin. Seeds 42-49.

## Steps
```
python prepare.py                         # stage-1 tables (GPT-2 tokenizer for the BPE keys)
python matched_cache.py                   # nested LR margins + fold matrices (5 folds)
python semisupervised.py                  # context teacher + exact-income EM (5 folds)
python train_xgb.py --model mixture       # -> preds/xgb_income_mixture_{oof,test}.npy

python teachers.py                        # stage-2 income context teachers (5 folds x 6 pools)
python artifact_features.py               # context-shape artifacts
python train_xgb.py --model shape         # -> preds/xgb_income_context_shape_{oof,test}.npy

python extra_features.py                  # needs ../neural_contrasts first (which reads cache/stage1, cache/stage2)
python train_xgb.py --model digits_diffs  # -> preds/xgb_full_context_contrasts_{oof,test}.npy

python em_pipeline.py                     # needs ../response_features
python train_information.py               # -> preds/xgb_information_mixture_{oof,test}.npy
```
All scripts take `--folds 0,1,2,3,4`; finished folds / seeds are skipped. GPU: XGBoost (`device='cuda'`) and
PyTorch (nested margins). Needs xgboost, torch, transformers, pyarrow.

## Cache layout
Other folders read these caches; file names are the original ones.

| Folder | Content |
|---|---|
| `cache/stage1/` | `base.parquet`, `keys.parquet`, `source_keys.parquet`, `predictors.parquet`, `source_base.npy`, `source_y.npy`, `y.npy`, `train_ids.npy`, `test_ids.npy`, `fold_id.npy`, `splits.npz`, `manifest.json` (prepare.py); `nested_margin_f{k}.npz`; `semi_f{k}.npz/.json`; `matched_f{k}/{train,query,fit_index,query_index,train_margin,query_margin}.npy + manifest.json` |
| `cache/stage2/` | `teachers/income_f{k}_inner{j}.npz`, `teachers/income_f{k}_outer.npz` (fit, unlab, raw_logit) + .json; `artifacts_income_f{k}.npz/.json` |
| `cache/stage8/` | `digits_all.npy/.json` |
| `cache/tokens/` | not used: the GPT-2 tokenizer of transformers replaces the vendored vocab / merges files |
| `cache/em_info/` | `folds.npy`, `train_ids.npy`, `test_ids.npy`, `base/fold_{k}/`, `teachers_gpu/income/fold_{k}/pool_{j}/{fit,p}.npy`, `features_gpu/fold_{k}/`, `information_features/fold_{k}/`, `bayes_margins/fold_{k}/` (train.npy, query.npy, manifest.json with `names`) |
| `runs/` | `stage1/matched_semi1_s{seed}_f{k}/`, `stage2/shape_s{seed}_f{k}/`, `stage8/digits_diffs_s{seed}_f{k}/` (`predictions.npz`: index, prediction, margin); `em_info/production_information/fold_{k}/seed_{s}/{valid,test}.npy` |

Fold matrices and runs use query = fold-k rows, then all test rows (train rows first in all-row arrays).

Reads from other folders (paths in `layout.py` and `lib.py`):
`../neural_contrasts/cache/stage3/{embedding_income,generative}_f{k}.npz` (else `stage4/`),
`stage5/coarse_entropy_all.npy`, `stage6/{dist_p100bpe,dist_p5000bpe,masked_density}_all.npy` (+ .json names);
`../response_features/cache/static/{raw,smooth_focus,source_pair_core,lineage}_{train,test}.npy` + manifest,
`static/fold_{k}/nested_response_margin_{train,valid,test}.npy`,
`base/fold_0/source_full_global_{train,valid,test}.npy` + manifest, and optionally
`bpe/income_to_first_piece.csv` (else computed with `common/bpe.py`, same mapping).

## Pairs
| Pair | OOF AUC |
|---|---:|
| xgb_income_mixture | 0.946515 |
| xgb_income_context_shape | 0.946516 |
| xgb_full_context_contrasts | 0.946570 |
| xgb_information_mixture | 0.946387 |

## Gaps
Ported from the competition code and not re-run: expect small numerical differences (XGBoost version, GPU
non-determinism). The stage-1 source-decoder bank (`source_decoder.npy`) is not ported; none of these models uses it.
