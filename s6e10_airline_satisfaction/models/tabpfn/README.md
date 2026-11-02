# TabPFN-3.5

Zero-shot, full context (all 559,708 outer-train rows of the fold), 8 estimators, raw features (`common.data.raw_arrays`).

`python train.py` -> pair `tabpfn35`, OOF AUC 0.961145 (the best foundation model here).

Needs the `tabpfn` package and the TabPFN-3.5 classifier checkpoint (set `MODEL_PATH` in `train.py`).
Comparison of all the foundation models: https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363
