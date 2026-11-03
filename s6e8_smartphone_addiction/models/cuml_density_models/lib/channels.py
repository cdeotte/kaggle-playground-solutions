"""Channel files written by ../density_features and ../lift_banks, read from their cache folders.

CHANNELS: name -> (repo folder, file name); {f} = outer fold, {s} = 'tr' (train rows) or 'te' (test rows). Channels without
{f} are target-free and the same for every fold. Every channel's columns are renamed <name>_<column>."""
import pandas as pd

from common.data import cache_dir

CHANNELS = {
    # masked-language-model / density channels (../density_features)
    'cmp4': ('density_features', 'cmp4_f{f}_{s}.parquet'),      # completion-response block (per fold)
    'denv4': ('density_features', 'den_v4_{s}.parquet'),         # masked-density model v4: per-column log-probs / entropies
    'denv5': ('density_features', 'den_v5_{s}.parquet'),         # masked-density model v5
    'mfill4': ('density_features', 'mlmfill4_{s}.parquet'),      # MLM fills of the missing cells (mean / sd)
    'rrd': ('density_features', 'rrd_{s}.parquet'),              # reconstruction residuals
    'qfill': ('density_features', 'qfill_{s}.parquet'),          # quantile fills of the screen columns
    # nested lift / target-encoding banks (../lift_banks, per fold)
    'plift': ('lift_banks', 'plift_f{f}_{s}.parquet'),
    'cbank': ('lift_banks', 'cbank_f{f}_{s}.parquet'),
    'wbank': ('lift_banks', 'wbank_f{f}_{s}.parquet'),
    'wbank12': ('lift_banks', 'wbank12_f{f}_{s}.parquet'),
    'sbank': ('lift_banks', 'sbank_f{f}_{s}.parquet'),
    'ebank': ('lift_banks', 'ebank_f{f}_{s}.parquet'),
}


def read_channel(name, fold, split):
    """Channel `name` of outer fold `fold`; split 'tr' (train rows) or 'te' (test rows); columns <name>_<column>."""
    folder, pattern = CHANNELS[name]
    df = pd.read_parquet(cache_dir(folder) / pattern.format(f=fold, s=split))
    df.columns = [f'{name}_{c}' for c in df.columns]
    return df


def append_channels(A, B, names, fold, log=print):
    """Append the channels to the train frame A and the test frame B (columns already present are skipped)."""
    for nm in names:
        Ptr, Pte = read_channel(nm, fold, 'tr'), read_channel(nm, fold, 'te')
        dup = [c for c in Ptr.columns if c in A.columns]
        if dup:
            Ptr, Pte = Ptr.drop(columns=dup), Pte.drop(columns=dup)
        A, B = pd.concat([A, Ptr], axis=1), pd.concat([B, Pte], axis=1)
        log(f'  +{nm}: {Ptr.shape[1]} cols -> {A.shape}')
    return A, B
