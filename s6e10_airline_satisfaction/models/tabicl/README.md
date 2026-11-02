# TabICLv2

Zero-shot, full context, 8 members, raw features. Precision fix: TabICL's default fp16 autocast is not accurate at full
context. Its column-embedding stage (inducing points that attend over all ~560k training rows of a column) loses precision,
and stock predictions even change with the number of query rows in the call. Running only that stage in fp32
(`inference_config={'COL_CONFIG': {'use_amp': False}}`) matches an all-fp32 run for a few seconds extra per member.

`python train.py` -> pair `tabicl2`, OOF AUC 0.958088. Needs `tabicl` (2.2.0) and the tabicl-classifier-v2 checkpoint.
