#!/usr/bin/env python
"""Kumo-Tabular (small or large), zero-shot, full context, on the 21 raw features.

Context = all outer-train rows of the fold (559,708 rows) with their labels; queries = the held-out fold + all test rows.
8 estimators. The run is wrapped in fp16 autocast (same accuracy as fp32 here, several times faster); sdm keeps its own
preprocessing in fp32.
Streaming: the stock fit() + predict() builds the KV cache of all 8 estimators at once, which does not fit in GPU memory at full
context. kumo_stream() does exactly the same work one estimator at a time (build its cache, predict every query chunk, free it).
GQA patch: grouped-query attention is computed by repeating the KV heads (exact) instead of SDPA's enable_gqa path.
Usage: python train.py --size small|large [--folds 0,1,2,3,4]"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

N_ESTIMATORS = 8
QUERY_CHUNK = 50_000


def patch_gqa():
    import torch.nn.functional as F
    sdpa = F.scaled_dot_product_attention

    def sdpa_expand(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        if enable_gqa and query.size(-3) != key.size(-3):
            rep = query.size(-3) // key.size(-3)
            key, value, enable_gqa = key.repeat_interleave(rep, dim=-3), value.repeat_interleave(rep, dim=-3), False
        return sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    F.scaled_dot_product_attention = sdpa_expand


def kumo_frame(X, y, cat_idx):
    import pandas as pd
    df = pd.DataFrame(X.astype(np.float32), columns=[f'f{j}' for j in range(X.shape[1])])
    for j in cat_idx:
        df[f'f{j}'] = df[f'f{j}'].round().astype(np.int64).astype(str)
    df['target'] = np.asarray(y).astype(np.int64).astype(str)
    return df


def positive_prob(sdm, out):
    names = list(out.columns[sdm.Stype.numerical]) if isinstance(out.columns, dict) else list(out.columns)
    return out.numerical[:, names.index('1')].float().cpu().numpy().astype(np.float64)


def kumo_stream(model, sdm, torch, table, n_ctx, chunks, n_est, seed):
    """== sdm fit() + predict(), one estimator at a time."""
    from sdm._inference import inference_mode
    from sdm.cache import Cache
    from sdm.models.base import _categorical_mask, _class_values, _stack_context
    from sdm.processing.execution import RecipeExecution
    x_ctx, y_ctx = table[:n_ctx].drop_columns('target'), table[:n_ctx, 'target']
    queries = [table[n_ctx + s:n_ctx + e].drop_columns('target') for s, e in chunks]

    def gen():
        return torch.Generator(device='cuda').manual_seed(seed)
    rex = RecipeExecution(copy.deepcopy(model.default_recipe()))
    with torch.amp.autocast('cuda', enabled=False), inference_mode('no_grad'):
        contexts = rex.fit_transform(x=x_ctx, y=y_ctx, related_tables=None, num_members=n_est, generator=gen())
    with torch.amp.autocast('cuda', enabled=False), inference_mode('inference'):
        q_members = [rex.transform(q, None) for q in queries]
    outs = [[] for _ in queries]
    for i, ctx in enumerate(contexts):
        members = [model._prepare_context(ctx, ())]
        with inference_mode('no_grad'):
            class_values = _class_values(members)
            context = _stack_context(members, class_values)
            cmask = _categorical_mask(members)
            cache = Cache(x_schemas=tuple(m.x.schema for m in members), y_schema=context.y.schema, related_tables_schema=None,
                          classes=context.y.categorical.categories[0] if context.y.categorical.size(-1) > 0 else None,
                          class_values=class_values, categorical_mask=cmask)
            model._forward(x_context=context.x, y_context=context.y, x_query=None, related_context_tables=None,
                           related_query_tables=None, cache=cache, generator=gen(), categorical_mask=cmask)
        cache.freeze()
        for c in range(len(queries)):
            qm = model._prepare_query(query=q_members[c][i], x_schema=cache['x_schemas'][0], related_tables_schema=None, callbacks=())
            outs[c] += model._forward_batch(contexts=None, queries=[qm], cache=cache, categorical_mask=cache['categorical_mask'],
                                            class_values=cache['class_values'], callbacks=(), requires_grad=False, generator=None)
        print(f'  estimator {i + 1}/{n_est} done', flush=True)
        del cache, members, context
        torch.cuda.empty_cache()
    res = []
    for c in range(len(queries)):
        with torch.amp.autocast('cuda', enabled=False), inference_mode('inference'):
            res.append(positive_prob(sdm, rex.transform_output(outs[c])))
    return np.concatenate(res)


def predict(Xc, yc, Xq, meta, fold, size):
    import pandas as pd
    import sdm
    import torch
    cat_idx = meta['cat_idx']
    df = pd.concat([kumo_frame(Xc, yc, cat_idx), kumo_frame(Xq, np.zeros(len(Xq)), cat_idx)], ignore_index=True)
    st = sdm.infer_stypes(df, overrides={'target': 'categorical', **{f'f{j}': 'categorical' for j in cat_idx}})
    table = sdm.TableTensor.from_pandas(df=df, stypes=st, device='cuda')
    del df
    chunks = [(s, min(s + QUERY_CHUNK, len(Xq))) for s in range(0, len(Xq), QUERY_CHUNK)]
    model = sdm.models.KumoTabular(task='classification', size=size, device='cuda')
    with torch.amp.autocast('cuda', dtype=torch.float16):
        return kumo_stream(model, sdm, torch, table, len(Xc), chunks, N_ESTIMATORS, 42 + fold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', required=True, choices=['small', 'large'])
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    import torch
    from sklearn.metrics import roc_auc_score
    torch.backends.cuda.matmul.allow_tf32 = True
    patch_gqa()
    name = f'kumo_{a.size}'
    X, Xt, meta = raw_arrays()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, name, k):
            continue
        tr, va = f != k, f == k
        p = predict(X[tr], y[tr], np.concatenate([X[va], Xt]), meta, k, a.size)
        nv = int(va.sum())
        save_part(HERE, name, k, p[:nv], p[nv:])
        print(f'fold {k}: AUC {roc_auc_score(y[va], p[:nv]):.6f}', flush=True)
    assemble(HERE, name, f, y)


if __name__ == '__main__':
    main()
