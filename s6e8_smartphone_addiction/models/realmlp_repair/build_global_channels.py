#!/usr/bin/env python
"""Step 5: target-free channels of the global density mixture (lib/global_density.py) for every train and test row:
24 surprisal / entropy channels and 27 conditional-mean fill / residual / geometry channels. They read no target and
the mixture is fit on all predictors, so one pass serves every fold.
Writes cache/global_channels/{train,test}.npy (rows x 51).
Usage: python build_global_channels.py"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from lib.global_density import global_encoder, load_models, make_mfill, score_channels  # noqa: E402
from lib.paths import own_dir, read_original, read_test, read_train, save_arrays  # noqa: E402


def main():
    import torch
    train, test, original = read_train(), read_test(), read_original()
    encoder = global_encoder(train, test, original)
    models = load_models(torch, encoder, lambda c: own_dir('global_density', c) / 'model_state.pt')
    out = {}
    for role, frame in (('train', train), ('test', test)):
        tokens, scalars = encoder.transform(frame)
        gden, means, stds = score_channels(torch, models, encoder, tokens, scalars)
        out[role] = np.column_stack([gden, make_mfill(tokens, encoder, means, stds)]).astype(np.float32)
        print(f'{role}: global channels {out[role].shape}', flush=True)
    save_arrays(own_dir('global_channels'), **out)


if __name__ == '__main__':
    main()
