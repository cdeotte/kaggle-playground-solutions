# EXAONE-Tabular

Zero-shot, full context, 8 members, raw features. Two changes to the stock classifier: `support_row_limit` is raised to the
context size (the default silently subsamples the context to 100,000 rows), and the query chunk is set to 8,192 rows (at full
context its memory estimate picks one query row per forward, ~16 h per fold; the predictions do not depend on the chunk size).

`python train.py` -> pair `exaone`, OOF AUC 0.958557. Needs the `exaonetabular` package.
