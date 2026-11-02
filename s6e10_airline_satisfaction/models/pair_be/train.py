#!/usr/bin/env python
"""pair_be: the source-pretrained BatchEnsemble MLP of ../field_be on the pair view (field view + embeddings of 154 exact
field-pair ids) with 8 members. Same code: ../field_be/be_transfer.py.
Needs the cache of ../transfer_features (python ../transfer_features/prepare.py).
Usage: python train.py --pretrain-only   (source pretraining -> pretrained/source.pt; also run by the next command if missing)
       python train.py [--folds 0,1,2,3,4]   -> preds/pair_be_transfer_aligned_{oof,test}.npy"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'field_be'))
from be_transfer import main  # noqa: E402  (../field_be/be_transfer.py)

if __name__ == '__main__':
    main(HERE, 'pair', 'pair_be_transfer_aligned')
