#!/usr/bin/env python
"""field_be: BatchEnsemble MLP (16 members) on the field view, pretrained on the labelled source rows (original data +
2 replica worlds), fine-tuned per fold, then aligned. The method is in be_transfer.py (shared with ../pair_be).
Needs the cache of ../transfer_features (python ../transfer_features/prepare.py).
Usage: python train.py --pretrain-only   (source pretraining -> pretrained/source.pt; also run by the next command if missing)
       python train.py [--folds 0,1,2,3,4]   -> preds/field_be_transfer_aligned_{oof,test}.npy"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from be_transfer import main  # noqa: E402

if __name__ == '__main__':
    main(HERE, 'field', 'field_be_transfer_aligned')
