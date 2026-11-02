# LimiX-2

Zero-shot, full context, raw features, with the official classification config limited to 8 light preprocessing pipelines
(0, 4, 8, 13, 20, 24, 27, 31). Stock LimiX re-runs the whole training context for every query batch when [train + queries]
does not fit in memory; `limix_cache.py` replaces the layer stack with an exact row-chunked version, so the context is processed
once per member.

`LIMIX_REPO=/path/to/LimiX python train.py` -> pair `limix2`, OOF AUC 0.958474.
Needs a clone of the LimiX GitHub repository (it provides `inference.predictor` and `model.v2_0`) and the LimiX-2 checkpoint
from Hugging Face.
