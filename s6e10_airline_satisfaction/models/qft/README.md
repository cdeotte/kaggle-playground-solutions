# Q-FT: a Qwen transformer that reads table rows as embeddings

Each of the 21 features becomes one embedding vector (column embedding + value embedding + a periodic embedding for numbers),
and the vectors go straight into Qwen3-0.6B-Base through `inputs_embeds` (no text). The rows are shuffled per row in training,
a CLS token at the end feeds a linear head, and the backbone is trained with LoRA (rank 16) for 3 epochs.

`python train.py` -> pair `qft`, OOF AUC 0.959795 (it did not improve our stack).
Discussion: https://www.kaggle.com/competitions/playground-series-s6e10/discussion/750009
