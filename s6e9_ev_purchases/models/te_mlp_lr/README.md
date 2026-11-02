# MLPs on target encodings, and a logistic regression on pair target encodings

Seven models built on the features of `../lr_margin_xgb/lib` (champion keys, nested TE, residual TE, `p_rule`, artifact
bundle; see `../lr_margin_xgb/README.md`). They import that library, so keep the two folders side by side.

**TE MLP** (`nn_lib.py`, `train_te_mlp.py`): an MLP (Linear-BatchNorm-SiLU-Dropout 0.15 blocks, 512-256-128) on the
standardized raw / log-count / flag columns, logit/3 of 13 nested TEs (the 11 champion keys, income at m 10, income // 100 at
m 30) and the 4 income TE differences / 3. BCE, AdamW (lr 2e-3, wd 1e-5), OneCycle, batch 4096, bf16 autocast, 12 epochs;
predictions after the last epoch.
- `mlp_te_diffs`: as above, 5 seeds.
- `mlp_te_rule_rte`: + a linear skip from the inputs to the logit, + `logit(p_rule)` added to the output logit as a fixed
  offset, + the 3 residual income TEs as inputs; 20 seeds.
- `mlp_te_rule_art_rte`: `mlp_te_rule_rte` + the artifact bundle as inputs; 20 seeds.

**Pair TEs** (`pair_te.py`): nested TEs of 21 single keys (the 13 columns, income // 10, // 100, // 1000, // 5000, // 10000,
commute // 1, commute // 5, age // 5; m = 1) and of their 210 pairs (m = 10). Keys with more than 50,000 levels over
train + test are not used (215 of 231 are). Built once and cached in `cache/pair_te_pools2.pkl` (~4.4 GB; the scripts below
hold it in RAM).
- `lr_pair_te` (`train_pair_lr.py`): cuML logistic regression (L2, C = 0.1) on `[z, z^2, z^3]` of the 215 TE logits, the 6 inputs
  of the purchase formula and 11 residual TEs (`y - p_rule`) of the champion keys, standardized (662 columns).
- `mlp_pair_te` (`train_pair_mlp.py`): the MLP with a linear skip on the standardized raw / log-count / flag columns, the 215
  TE logits and the 11 residual TEs; 10 seeds.
- `mlp_pair_te_rule_wide`: + the `logit(p_rule)` offset, hidden 1024-512-256; 10 seeds.
- `mlp_pair_te_rule_wide_bpe`: `mlp_pair_te_rule_wide` + logits of nested TEs of the income's leading GPT-2 BPE piece
  (`common/bpe.py`; m 1 and 20) and of its pairs with 15 other keys (m 10), + a residual TE of the piece (m 5); 10 seeds.

```
python train_te_mlp.py mlp_te_diffs
python train_te_mlp.py mlp_te_rule_rte
python train_te_mlp.py mlp_te_rule_art_rte
python pair_te.py                                # builds cache/pair_te_pools2.pkl (the next scripts also build it if missing)
python train_pair_lr.py
python train_pair_mlp.py mlp_pair_te
python train_pair_mlp.py mlp_pair_te_rule_wide
python train_pair_mlp.py mlp_pair_te_rule_wide_bpe
```
Each script writes `preds/<pair>_{oof,test}.npy` (probabilities; test = mean of the 5 fold models, averaged over the seeds).

| Pair | Model | Seeds | OOF AUC |
|---|---|---:|---:|
| mlp_te_diffs | TE MLP | 5 | 0.945856 |
| mlp_te_rule_rte | TE MLP + skip + rule offset + rte | 20 | 0.945924 |
| mlp_te_rule_art_rte | TE MLP + skip + rule offset + art + rte | 20 | 0.945859 |
| lr_pair_te | cuML LR on single + pair TEs | - | 0.945957 |
| mlp_pair_te | MLP on single + pair TEs + rte | 10 | 0.946041 |
| mlp_pair_te_rule_wide | + rule offset, wider | 10 | 0.945912 |
| mlp_pair_te_rule_wide_bpe | + BPE-piece TEs and rte | 10 | 0.946181 |

Requirements: torch (CUDA), scikit-learn, scipy, pandas; cuML + cupy for `lr_pair_te`; transformers for the GPT-2 tokenizer of
the bpe model.
