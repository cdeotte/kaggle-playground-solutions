# RealMLP and TabM on nested target encodings

The nets read the fold-specific target-encoding library of `../nested_te_xgb` (build it first): either 170 of its 203
channels (`MID`: without the `ares`, `w4` and `w7c` families) or all 203 (`Q8`), with the 6 categorical raw columns passed
as categoricals. Per fold and net: Gaussian noise (sd 0.1, logit units) on the label channels of the outer-train rows; the
epochs are selected on a random 15% of outer-train re-encoded from the labels of the other 85% (validation-style). RealMLP
(pytabkit) is refit after the selection, TabM keeps the selected checkpoint. Several nets per fold (new seed and noise
draw each) are averaged in logit space; test = mean of the 5 fold models. TabM outputs are standardised per fold (mean /
sd of the fold's validation logits, the same map on its test logits) before the blend.

| Pair | Streams (exact settings in `train.py`) | OOF AUC |
|---|---|---:|
| `realmlp_te` | 3 RealMLP streams with plr_sigma 1 and 2 nets per fold: MID with 16 ensemble members (seed 42), Q8 (seed 42), MID with 16 members (seed 7); equal logit weights | 0.945972 |
| `realmlp_tabm_te` | 2 short RealMLP streams (MID, 6 epochs, lr 0.025, batch 512, GELU, 4 nets per fold; seeds 42, 7) + 2 TabM streams (MID, seeds 42, 7); weight 1/4 each | 0.946023 |
| `realmlp_tabm_te_bpe` | the same RealMLP and TabM recipes with 2 more channels, the target encodings (s = 1, 20) of the income's GPT-2 BPE leading piece; seed 42; 50/50 | 0.946250 |

```
python ../nested_te_xgb/build_library.py    # once
python train.py all                         # or one pair, e.g. python train.py realmlp_tabm_te_bpe
python train.py --stream tabm_s42 --folds 0 # one stream and fold (to run folds in parallel)
```
Each stream keeps its finished folds in `runs/parts/` and its OOF / test pair in `runs/preds/`; the pairs go to `preds/`.
Needs pytabkit (torch); transformers for the BPE channels.
