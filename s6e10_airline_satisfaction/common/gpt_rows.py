"""Table rows as GReaT text, for the GPT-2 models.

A row is written exactly like the GReaT training text of the competition's generator: "<column> is <value>" fragments joined
by ", ", values as Python str() of the stored value (ints '36', floats '0.0', NaN 'None', the label 'True' / 'False').
Each fragment is tokenized once per distinct value and the token ids are concatenated per row. This is exact because GPT-2's
pre-tokenizer splits at these boundaries (check_tokenization verifies it on random rows)."""
import numpy as np
import pandas as pd
import torch

from common.data import DATA_DIR, ROOT, TARGET, read_test, read_train

HF_NAME = {'small': 'gpt2', 'medium': 'gpt2-medium', 'large': 'gpt2-large', 'neo': 'EleutherAI/gpt-neo-125M'}
REPLICA_DIR = ROOT / 'models' / 'gpt2_replica' / 'checkpoints'


def source_order():
    """The 21 features in data.csv column order."""
    return [c for c in pd.read_csv(DATA_DIR / 'data.csv', nrows=2).columns if c != TARGET]


def load_frame():
    """train rows then test rows, every feature as the exact string the replica saw in training. Returns (frame, n_train)."""
    cols = source_order()
    tr, te = read_train(), read_test()
    df = pd.concat([tr[cols], te[cols]], ignore_index=True)
    out = {}
    for c in cols:
        v = df[c]
        if v.dtype.kind == 'f':
            out[c] = np.array(['None' if pd.isna(x) else str(float(x)) for x in v], dtype=object)
        else:
            out[c] = v.astype(str).to_numpy()
    return pd.DataFrame(out), len(tr)


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')  # all GPT-2 sizes share this tokenizer; GPT-Neo uses the identical BPE
    tok.pad_token = tok.eos_token
    return tok


def load_replica(size, device='cuda'):
    """Causal LM (GPT-2, or GPT-Neo for size 'neo') initialized from the GReaT replica checkpoint (../models/gpt2_replica).
    Replica state dicts saved by transformers 4.26 may carry old attention-mask buffers; they are ignored."""
    from transformers import AutoConfig, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(HF_NAME[size]))
    state = torch.load(REPLICA_DIR / size / 'model.pt', map_location='cpu')
    res = model.load_state_dict(state, strict=False)
    assert not [k for k in res.missing_keys if k != 'lm_head.weight'], res.missing_keys
    assert not [k for k in res.unexpected_keys if not k.endswith(('attn.bias', 'masked_bias'))], res.unexpected_keys
    model.tie_weights()
    return model.to(device)


class GELUTanh(torch.nn.Module):
    """The same formula as GPT-2's gelu_new, as one fused kernel."""
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate='tanh')


def fast_gelu(model):
    for blk in model.transformer.h:
        blk.mlp.act = GELUTanh()
    return model


def fragment_tables(tok, frame, first_space):
    """Per feature: per-row codes into the column's distinct values, and the token ids of '{s}{c} is {v}' (the first field;
    s = ' ' right after a label prefix, '' at the start of the text) and ', {c} is {v}' (every later field)."""
    tabs = []
    for c in frame.columns:
        codes, uniq = pd.factorize(frame[c].to_numpy(), sort=False)
        first = tok([f'{first_space}{c} is {v}' for v in uniq], add_special_tokens=False)['input_ids']
        later = tok([f', {c} is {v}' for v in uniq], add_special_tokens=False)['input_ids']
        tabs.append((codes.astype(np.int64), first, later))
    return tabs


def pad_batch(seqs, eos):
    """Right-pad token lists with eos; returns ids [B, L] and a 0/1 mask of the real tokens."""
    L = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), L), eos, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for d, x in enumerate(seqs):
        ids[d, :len(x)] = torch.tensor(x)
        att[d, :len(x)] = 1
    return ids, att
