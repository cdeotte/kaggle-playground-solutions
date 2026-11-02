# TabFM 1.0

Zero-shot, full context, 8 estimators, raw features, bf16. Stock TabFM has no context cache, so every query batch re-processes
the whole training context. `tabfm_cache.py` adds an exact cache: training rows are processed once per member and the query
rows read the cached training states (same weights, same math, only the batching changes).

`python train.py` -> pair `tabfm`, OOF AUC 0.958871. Needs the `tabfm` package (1.0.1).
