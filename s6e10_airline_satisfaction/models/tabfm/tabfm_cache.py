"""Exact context cache for TabFM 1.0 (PyTorch port, tabfm 1.0.1).

Why: stock TabFM has no context cache. TabFMClassifier.predict_proba runs, for every ensemble member and every call, ONE
forward over [all training rows + the query rows], so the whole training context is re-processed for every query batch, and
its 24-layer, d=2048 ICL attention runs over (train + query)^2 positions with a key mask.

Why a cache is exact: every stage that mixes rows only ever reads TRAINING rows.
  * cell_embedder: per row (training rows add their label embedding).
  * col_embedder / col_embedder_2 (3 induced-set blocks each): the 256 inducing points attend to training rows only
    (key mask), then every row attends to those 256 summaries. Cache: the per-block inducing summaries.
  * row_interactor / row_interactor_2: attention across the features of one row (rows independent).
  * icl_predictor (24 blocks): every row attends to training rows only (key mask); training rows never see query rows.
So training rows are processed once per member, and the query rows read the cached training states. The ICL stage runs
layer-major (one layer's training K/V at a time), which keeps memory small. Same weights, same bf16 compute, same
preprocessing / ensembling / temperature / averaging (only TabFMClassifier._batch_forward is replaced); differences vs stock
are floating-point accumulation order only."""
import math
import types

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _col_train(ce, emb, col_chunk):
    """ColEmbedding over training rows only; returns outputs and per-block inducing summaries [HC, num_inds, E]."""
    b, t, hc, e = emb.shape
    src = emb.permute(0, 2, 1, 3).reshape(b * hc, t, e)
    out = torch.empty_like(src)
    hid = [torch.empty(b * hc, blk.ind_vectors.shape[0], e, dtype=src.dtype, device=src.device) for blk in ce.tf_col.blocks]
    for s in range(0, b * hc, col_chunk):
        x = src[s:s + col_chunk]
        for bi, blk in enumerate(ce.tf_col.blocks):
            ind = blk.ind_vectors.unsqueeze(0).expand(x.shape[0], -1, -1)
            h = blk.mab1(ind, x, x, attn_mask=None)       # stock: key mask = training rows; here all keys ARE training rows
            hid[bi][s:s + col_chunk] = h
            x = blk.mab2(x, h, h)
        out[s:s + col_chunk] = ce.ln_w(ce.out_w(x))
    return out.reshape(b, hc, t, e).permute(0, 2, 1, 3), hid


@torch.no_grad()
def _col_query(ce, emb, hid):
    b, t, hc, e = emb.shape
    x = emb.permute(0, 2, 1, 3).reshape(b * hc, t, e)
    for bi, blk in enumerate(ce.tf_col.blocks):
        x = blk.mab2(x, hid[bi], hid[bi])                 # every row attends to the training-derived summaries
    return ce.ln_w(ce.out_w(x)).reshape(b, hc, t, e).permute(0, 2, 1, 3)


def _kv(attn, kv_in):
    b, tk, _ = kv_in.shape
    k = attn.key_ln(attn.k_proj(kv_in).view(b, tk, attn.nhead, attn.hd))
    v = attn.v_proj(kv_in).view(b, tk, attn.nhead, attn.hd)
    return k.transpose(1, 2), v.transpose(1, 2)


def _attend(attn, q_in, k, v):
    b, tq, d = q_in.shape
    q = attn.query_ln(attn.q_proj(q_in).view(b, tq, attn.nhead, attn.hd))
    scale = 1.442695041 / math.sqrt(attn.hd) * F.softplus(attn.per_dim_scale.float())
    q = (q * scale.to(q.dtype)).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, scale=1.0)
    return attn.out_proj(o.transpose(1, 2).reshape(b, tq, d))


@torch.no_grad()
def _icl_block(blk, xs, k, v, chunk):
    """MultiheadAttentionBlock.forward(q=xs, k=v=training states) with the training K/V precomputed."""
    out = torch.empty_like(xs)
    for s in range(0, xs.shape[1], chunk):
        x = xs[:, s:s + chunk]
        a = blk.post_attn_ln(_attend(blk.attn, blk.pre_attn_ln(x), k, v))
        x = x + a
        out[:, s:s + chunk] = x + blk._ff(x)
    return out


@torch.no_grad()
def member_forward(model, X_tr, y_tr, X_q, cat_mask, d, q_chunk=32768, col_chunk=8, icl_chunk=65536):
    """One ensemble member: logits [Q, max_classes] for the query rows (== stock forward on [train; query], query slice)."""
    dev, dt = model.cls_tokens.device, model.cls_tokens.dtype
    C, Q = X_tr.shape[0], X_q.shape[0]
    cm = None if cat_mask is None else torch.from_numpy(np.asarray(cat_mask))[None].to(dev)
    d_t = torch.tensor([int(d)], device=dev, dtype=torch.long)

    def prep(X):
        return torch.nan_to_num(torch.from_numpy(np.ascontiguousarray(X))[None].to(dev, torch.float32), nan=-100.0).to(dt)
    ytr = torch.from_numpy(np.asarray(y_tr))[None].to(dev)
    # ---- training rows: embedding stages (once) ----
    emb = model.cell_embedder(prep(X_tr), ytr, torch.tensor([C], device=dev), cm, d=d_t)
    emb, h1 = _col_train(model.col_embedder, emb, col_chunk)
    emb = torch.cat([model.cls_tokens.expand(1, C, -1, -1), emb], dim=2)
    emb = model.row_interactor(emb, d=d_t)
    emb, h2 = _col_train(model.col_embedder_2, emb, col_chunk)
    reps_tr = model.row_interactor_2(emb, d=d_t)
    del emb
    # ---- query rows: embedding stages (chunks, reading the cached column summaries) ----
    reps_q = torch.empty(1, Q, reps_tr.shape[-1], dtype=reps_tr.dtype, device=dev)
    zero = torch.zeros(1, dtype=torch.long, device=dev)
    for s in range(0, Q, q_chunk):
        xq = prep(X_q[s:s + q_chunk])
        n = xq.shape[1]
        yq = torch.full((1, n), -100, dtype=ytr.dtype, device=dev)
        e = model.cell_embedder(xq, yq, zero, cm, d=d_t)                       # train_size 0: no label embedding
        e = _col_query(model.col_embedder, e, h1)
        e = torch.cat([model.cls_tokens.expand(1, n, -1, -1), e], dim=2)
        e = model.row_interactor(e, d=d_t)
        e = _col_query(model.col_embedder_2, e, h2)
        reps_q[:, s:s + n] = model.row_interactor_2(e, d=d_t)
    # ---- ICL, layer-major: training rows attend to training rows; query rows attend to training rows ----
    icl = model.icl_predictor
    tm = torch.ones(1, C, dtype=torch.bool, device=dev)
    x_tr = reps_tr + icl.y_encoder(ytr) * tm[..., None]                        # == stock r = reps + y_enc * tm
    x_q = reps_q
    del reps_tr, reps_q
    for blk in icl.tf_icl.blocks:
        k, v = _kv(blk.attn, blk.pre_attn_ln(x_tr))
        new_q = _icl_block(blk, x_q, k, v, icl_chunk)
        x_tr = _icl_block(blk, x_tr, k, v, icl_chunk)
        x_q = new_q
        del k, v, new_q
    out = torch.empty(Q, icl.decoder.layers[-1].out_features, dtype=torch.float32, device=dev)
    for s in range(0, Q, icl_chunk):
        out[s:s + icl_chunk] = icl.decoder(icl.ln(x_q[0, s:s + icl_chunk])).float()
    return out.cpu().numpy()


def install(clf, **kw):
    """Replace clf._batch_forward (stock: one [train; query] forward per member) with the cached member_forward."""
    def _batch_forward(self, Xs, ys, cat_masks=None, ds=None):
        C = ys.shape[1]
        outs = []
        for e in range(Xs.shape[0]):
            outs.append(member_forward(self.model, Xs[e, :C], ys[e], Xs[e, C:], None if cat_masks is None else cat_masks[e],
                                       Xs.shape[-1] if ds is None else ds[e], **kw))
            print(f'  member {e + 1}/{Xs.shape[0]} done', flush=True)
        return np.stack(outs, axis=0)
    clf._batch_forward = types.MethodType(_batch_forward, clf)
    return clf
