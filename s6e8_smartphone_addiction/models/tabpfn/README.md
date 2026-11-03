# TabPFN v3 with context ensembling

The pretrained TabPFN v3 classifier (a tabular in-context transformer; no training, only a context) on 28 columns:
the 12 raw columns (categoricals as ordinal codes, NaN kept), nested exact-value target encodings of the 12 columns
(tokens `'%.2f'` / `'__NA__'`, smoothing 20; outer-train rows encoded by 5 inner folds with seed 777, validation and test
rows by the map of all outer-train rows) and 4 derived columns: slack (daily - social - gaming - work), sleep + daily,
utilization ((social + gaming + work) / daily) and the number of missing columns (`lib/tabpfn_features.py`).

Per outer fold, K = 8 label-stratified context subsets of 50,000 outer-train rows are drawn (numpy seed
777 + 100 x fold + k); TabPFN (`n_estimators=2`, CUDA) is fit on each context and predicts the fold's validation rows and
the test rows; the 8 probabilities are averaged. Test = mean of the 5 fold models.

```
python train.py                     # -> preds/tabpfn_v3_context_ensemble_{oof,test}.npy
python train.py --folds 0           # one fold (finished folds are kept in parts/)
python train.py --model-path <ckpt> # use a local copy of the checkpoint
```
No other repo folders are needed. Each fold makes 8 x (138k validation + 296k test) predictions on one GPU.

**Weights.** The checkpoint is not part of this repo. `train.py` downloads the default TabPFN v3 classifier,
`tabpfn-v3-classifier-v3_default.ckpt`, from the Hugging Face model
[Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3) on first use (through `huggingface_hub`, which the
tabpfn package installs; it is cached afterwards) and passes it to `TabPFNClassifier(model_path=...)`. The weights are
licensed by Prior Labs: you must accept their licence on the model page and be logged in to Hugging Face
(`hf auth login`) before the download works.

| Pair | Model | OOF AUC |
|---|---|---:|
| `tabpfn_v3_context_ensemble` | TabPFN v3, 8 contexts of 50,000 rows, raw + target encodings + 4 derived columns | 0.966141 |

Packages: tabpfn (a version that loads the v3 checkpoints; the run used 8.5.0), torch (CUDA), huggingface_hub, pandas,
scikit-learn.
