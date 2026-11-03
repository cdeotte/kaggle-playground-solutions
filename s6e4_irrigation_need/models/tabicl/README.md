# TabICL: zero-shot in-context classifier on the raw columns

One TabICL model on the 19 raw competition columns (no feature engineering, no original rows). TabICL is a pretrained
tabular in-context transformer: the fold's training rows are its context and no weights are trained. The pair belongs
to stack A of `stack/`.

**Model** (`train.py`): per fold (the 5 folds of `common/data.py`), eight `TabICLClassifier` models (`n_estimators=2`,
`kv_cache=True`, `support_many_classes=True`, `random_state` 0..7) get the fold's training rows as context (the 8
categoricals stay strings; TabICL encodes them itself) and predict the fold's validation rows and the test rows; the 8
probabilities are averaged. Inference is deterministic. Test = mean of the 5 fold models.

```
python train.py
```
Writes `preds/tabicl_zero_shot_8seed_a_{oof,test}.npy` (probabilities, class order Low, Medium, High). Only `data/` is
read; no other repo folder has to run first. One GPU.

| Pair | Stack | Model | OOF balanced acc. |
|---|---|---|---:|
| tabicl_zero_shot_8seed_a | A | zero-shot TabICLClassifier, 8 seeds x 2 estimators, raw columns | 0.961396 |

Requirements: tabicl (the run used 2.0.3 with its default classifier checkpoint `tabicl-classifier-v2-20260212.ckpt`,
which the package downloads from the Hugging Face model [jingang/TabICL](https://huggingface.co/jingang/TabICL) on
first use), torch (CUDA), scikit-learn, pandas, numpy.
