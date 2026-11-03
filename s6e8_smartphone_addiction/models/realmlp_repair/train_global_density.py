#!/usr/bin/env python
"""Step 5: train one of the four global masked density transformers (lib/global_density.py) on the predictors of every
train and test row (no target): g4, g5, g4orig8 (+ the original rows x 8) and g9unitarget (half of the rows hide
exactly one observed field). The lattice encoder is fit on the train + test + original predictors.
Writes cache/global_density/<component>/model_state.pt.
Usage: python train_global_density.py --component g4|g4orig8|g5|g9unitarget"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from lib.global_density import COMPONENTS, global_encoder  # noqa: E402
from lib.lattice import concat_predictors, train_model  # noqa: E402
from lib.paths import own_dir, read_original, read_test, read_train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--component', required=True, choices=list(COMPONENTS))
    args = ap.parse_args()
    import torch
    spec = COMPONENTS[args.component]
    train, test, original = read_train(), read_test(), read_original()
    encoder = global_encoder(train, test, original)
    tokens, scalars = encoder.transform(concat_predictors(train, test))
    if spec['original_repeat']:
        original_tokens, original_scalars = encoder.transform(concat_predictors(original))
        tokens = np.concatenate([tokens, np.tile(original_tokens, (spec['original_repeat'], 1))], axis=0)
        scalars = np.concatenate([scalars, np.tile(original_scalars, (spec['original_repeat'], 1))], axis=0)
    print(f'{args.component}: {len(tokens)} training rows', flush=True)
    model = train_model(torch, torch.from_numpy(tokens).cuda(), torch.from_numpy(scalars).cuda(),
                        encoder.cardinalities, spec['config'], seed=spec['seed'], scheme=spec['scheme'])
    torch.save(model.state_dict(), own_dir('global_density', args.component) / 'model_state.pt')


if __name__ == '__main__':
    main()
