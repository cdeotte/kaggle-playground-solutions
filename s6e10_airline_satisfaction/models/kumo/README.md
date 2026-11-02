# Kumo-Tabular (small and large)

Zero-shot, full context, 8 estimators, raw features, fp16 autocast. The stock fit/predict builds the KV cache of all 8
estimators at once, which does not fit at full context; `kumo_stream()` does the same work one estimator at a time.

| Command | Pair | OOF AUC |
|---|---|---:|
| `python train.py --size large` | kumo_large | 0.960036 |
| `python train.py --size small` | kumo_small | 0.958484 |

Needs the `sdm` package (structured-data-models, from GitHub) and the Kumo-Tabular weights from Hugging Face.
