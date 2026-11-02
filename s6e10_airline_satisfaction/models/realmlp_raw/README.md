# RealMLP on the raw features

pytabkit RealMLP on the 21 raw features only, averaged over 8 seeds. A large `plr_sigma` (32) is the key setting: it lets the
periodic numeric embeddings resolve fine differences in Flight Distance (a route id).

`python train.py` -> pair `realmlp_raw`, OOF AUC 0.959311. Needs `pytabkit`.
