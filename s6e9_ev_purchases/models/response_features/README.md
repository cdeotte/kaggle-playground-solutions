# Response features: RealMLP and XGBoost on fold-local target encodings

Five models on shared feature caches. The core features are target encodings ("responses") of the raw values: per-field
target rates, income and commute window residuals, the same encodings at many grouping resolutions, and a logistic margin
on them. Labels of the held-out fold never enter its features. Validation and test rows are encoded with all outer-train
labels of the fold. Training rows are encoded by leave-one-out (RealMLP) or inner cross-fits (XGBoost and the multires
families). Other inputs are label-free, apart from the published labels of the original dataset.

- **realmlp_response_geometry**: RealMLP (pytabkit, 16-member ensemble) on 209 inputs: the raw features, every numeric
  value as a categorical token, fixed-width bins, the original-data source bank (per-field value counts, original-label
  posteriors, distance to the nearest original value, local density ratios), leave-one-out target encodings of the 13
  fields, income and commute window residuals, a Gender x home-charging lift, and 30 fixed hinges of the encodings of 5 key
  fields. Target-derived inputs are scaled by fixed constants only.
- **realmlp_multires**: the same, plus 184 multiresolution encodings. These cover exact values, 16 "core" widths, and 38
  "wide" widths and log-ratios of income, commute, age and the station counts. Each uses alpha 1 and 20, with 1-seed or
  8-seed inner cross-fits, plus summaries.
- **xgb_nested_margin**: GPU XGBoost (900 rounds, depth 4) on 227 columns: static blocks (raw one-hot, smooth bases,
  original-data pair rates), part of the source bank, alpha-1 / alpha-20 encodings and the income / commute residuals.
  Boosting starts from a nested logistic-regression margin on the 26 encodings.
- **xgb_response_geometry**: adds 3 multires families and 127 differences between income resolutions. Training rows of all
  target-derived blocks get Gaussian noise (sd 0.04). Column-sampling weights: base encodings 0.6, multires 0.2,
  differences 0.05, everything else 1. Depth 5; 472 columns.
- **xgb_acceptance_entropy**: xgb_response_geometry plus two more blocks (526 columns). One is a target encoding of the
  income's leading GPT-2 BPE piece. The other is 52 entropy columns from the GReaT replicas e55 / e60: for each field, the
  entropy of the next-token distribution over that field's legal first tokens, with the row written label-first, once as
  "Yes" and once as "No".

Each script's docstring lists its exact features and settings.

## Steps
```
python base_cache.py         # cache/base      source bank, leave-one-out encodings, residuals (about 1 min per fold)
python static_cache.py       # cache/static    label-free static blocks (a few minutes)
python margin_cache.py       # cache/static/fold_<k>   alpha-1 / alpha-20 encodings + nested logistic margin
python multires_cache.py     # cache/multires  (about 1.5 min per fold)
python bpe_map.py            # cache/bpe
python acceptance_cache.py   # cache/acceptance: 26 replica maps (about 30 min each on one GPU), then the pack

python train_realmlp.py --pair realmlp_response_geometry   # needs base
python train_realmlp.py --pair realmlp_multires            # needs base, multires
python train_xgb.py --pair xgb_nested_margin               # needs base, static
python train_xgb.py --pair xgb_response_geometry           # needs base, static, multires
python train_xgb.py --pair xgb_acceptance_entropy          # needs base, static, multires, bpe, acceptance
```
The fold scripts take `--folds 0,1` to run a subset. The trainers keep finished folds in `parts/`, skip them, and write
`preds/<pair>_{oof,test}.npy` (probabilities; test = mean of the 5 fold models). XGBoost boosters go to `checkpoints/`.
`acceptance_cache.py --checkpoints e55 --held 0,1,2` scores a subset of the maps, skips maps that already exist, and builds
the pack once all 26 exist.

On one GPU, a RealMLP fold takes about 4 min (realmlp_response_geometry) or 7 min (realmlp_multires), and an XGBoost fold
takes under 2 min.

**Dependencies:** `common/` (data, pairs, bpe). `acceptance_cache.py` needs the GReaT replicas
`../gpt2_replica/checkpoints/{e55,e60}.pt`; we ran it in the replica environment (`../gpt2_replica/requirements.txt`).
Everything else used numpy 2.2.6, pandas 2.3.3, scipy 1.16.3 and scikit-learn 1.8.0, plus xgboost 3.1.2, pytabkit 1.7.3 and
torch 2.7.0, and transformers (GPT-2 tokenizer) for `bpe_map.py`.

## Cache layout (also read by other model folders)

`<block>_train` = the outer-train rows of fold k in train.csv order (`train_indices.npy`). `<block>_valid` = the rows of
fold k (`valid_indices.npy`). `<block>_test` = all test.csv rows. In `cache/static/` and `cache/acceptance/`, the train
arrays hold all train.csv rows. Each `manifest.json` holds column names.

| Folder | Files | manifest.json |
|---|---|---|
| `cache/base/fold_<k>/` | `{source_full_global, response_single_loo, response_local_loo, response_income_multiscale, response_commute_residual, response_target_gender_home}_{train,valid,test}.npy`, `{train,valid}_indices.npy` | `<block>_names`, `source_focus_names` |
| `cache/static/` | `{raw, smooth, semantic, source_pairs, lineage, smooth_focus, source_pair_core}_{train,test}.npy` | `block_names` |
| `cache/static/fold_<k>/` | `{response_alpha1, response_alpha20}_{train,valid,test}.npy` (float32), `nested_response_margin_{train,valid,test}.npy` (float64, [n, 1]), `{train,valid}_indices.npy` | `<block>_names` |
| `cache/multires/fold_<k>/` | `{exact_single, exact_avg8, core_single, core_avg8, wide_avg8, response_summaries}_{train,valid,test}.npy`, `{train,valid}_indices.npy` | `families.<family>.names` |
| `cache/bpe/` | `income_to_first_piece.csv` (columns income, first_piece) | |
| `cache/acceptance/` | `train.npy` [668,665 x 576], `test.npy` [286,571 x 576] (float32), `raw/accept_{e55,e60}_c{00..12}.npy` | `names`, `families`, `held_columns`, `numeric_columns` |

## Pairs

| Pair | OOF AUC |
|---|---:|
| realmlp_response_geometry | 0.945896 |
| realmlp_multires | 0.946056 |
| xgb_nested_margin | 0.946065 |
| xgb_response_geometry | 0.946242 |
| xgb_acceptance_entropy | 0.946444 |
