#!/usr/bin/env python
"""Step 6: the 20 repair-posterior channels (lib/repair.py) for every train and test row from the global density
mixture (target-free; one pass serves every fold).
Writes cache/repair/{train,test}.npy (rows x 20).
Usage: python build_repair.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from lib import repair  # noqa: E402
from lib.global_density import global_encoder, load_models  # noqa: E402
from lib.paths import own_dir, read_original, read_test, read_train, save_arrays  # noqa: E402


def main():
    import torch
    train, test, original = read_train(), read_test(), read_original()
    encoder = global_encoder(train, test, original)
    models = load_models(torch, encoder, lambda c: own_dir('global_density', c) / 'model_state.pt')
    out = {}
    for role, frame in (('train', train), ('test', test)):
        tokens, scalars = encoder.transform(frame)
        out[role] = repair.score(torch, models, encoder, tokens, scalars)
        print(f'{role}: repair channels {out[role].shape}', flush=True)
    save_arrays(own_dir('repair'), **out)


if __name__ == '__main__':
    main()
