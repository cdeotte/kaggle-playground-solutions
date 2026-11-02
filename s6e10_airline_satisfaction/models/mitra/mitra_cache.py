"""Exact, memory-bounded Mitra-v2 (AutoGluon Tab2D) inference at large context: a layer-by-layer, row-chunked re-implementation of
Tab2D.forward (non-flash path; batch 1, no padding) that calls Mitra's own submodules.

Why: stock Mitra (AutoGluon MitraClassifier) predicts in query chunks of max_samples_query (default 1,024) and re-runs the whole
support set through all 12 layers for every chunk (default support = a random 8,192 rows). At full context that is ~430 support
passes per fold.

Why this is exact (Layer.forward, non-flash path):
  * each layer = row attention (per column: every row attends to the SUPPORT rows of that column) -> MLP -> feature attention
    (within each row) -> MLP, each with pre-LayerNorm and a residual;
  * support rows never attend to query rows, so the support state of every layer does not depend on the queries;
  * row attention is independent per column (21 features + the label cell), and LayerNorm / MLP / feature attention act on each
    row separately.
So, per layer: for each column j, K/V are computed once from all support rows, then support rows and query rows are updated in
chunks (in place; K/V were computed from the pre-update support); then the per-row blocks run in row chunks. The support is
processed once per member, the queries in chunks, and memory stays bounded (fp32 states: support + queries ~45 GB at full context).
Under autocast, every op sees the same dtypes as in the stock forward (the residual stream is fp32 because the label embedding is
an nn.Embedding; q/k/v and MLP matmuls run in the autocast dtype)."""
import einops
import torch
import torch.nn.functional as F


def _attn(att, q_in, k, v, n_heads):
    """== MultiheadAttention.forward non-flash path with precomputed (projected, head-split) k and v."""
    q = einops.rearrange(att.q(q_in), 'b t (h d) -> b h t d', h=n_heads)
    o = F.scaled_dot_product_attention(q, k, v)
    return att.o(einops.rearrange(o, 'b h t d -> b t (h d)'))


def _row_blocks(L, X, a, e):
    """MLP 1 -> feature attention -> MLP 2 of Layer.forward for rows a:e of X (b, s, f+1, d), written back in place."""
    x = X[:, a:e]
    x = x + L.linear2(F.gelu(L.linear1(L.layer_norm2(x))))
    s = x.shape[1]
    t = einops.rearrange(L.layer_norm3(x), 'b s f d -> (b s) f d')
    x = x + einops.rearrange(L.attention2(t, t, t), '(b s) f d -> b s f d', s=s)
    x = x + L.linear4(F.gelu(L.linear3(L.layer_norm4(x))))
    X[:, a:e] = x


def layer_forward(L, S, Q, chunk):
    """== Layer.forward(support=S, query__=Q) (non-flash path), in place on S and Q."""
    h = L.attention1.n_heads
    for j in range(S.shape[2]):
        kv = L.layer_norm1(S[:, :, j])
        k = einops.rearrange(L.attention1.k(kv), 'b t (h d) -> b h t d', h=h)
        v = einops.rearrange(L.attention1.v(kv), 'b t (h d) -> b h t d', h=h)
        del kv
        for X in (S, Q):
            for a in range(0, X.shape[1], chunk):
                e = min(a + chunk, X.shape[1])
                X[:, a:e, j] += _attn(L.attention1, L.layer_norm1(X[:, a:e, j]), k, v, h)
        del k, v
    for X in (S, Q):
        for a in range(0, X.shape[1], chunk):
            _row_blocks(L, X, a, min(a + chunk, X.shape[1]))


def _embed(model, y_emb, x, chunk):
    """== einops.pack((y_emb, model.x_embedding(x)), 'b s * d') in row chunks."""
    out = None
    for a in range(0, x.shape[1], chunk):
        e = min(a + chunk, x.shape[1])
        part, _ = einops.pack((y_emb[:, a:e], model.x_embedding(x[:, a:e])), 'b s * d')
        if out is None:
            out = torch.empty((x.shape[0], x.shape[1]) + tuple(part.shape[2:]), dtype=part.dtype, device=part.device)
        out[:, a:e] = part
    return out


@torch.no_grad()
def forward(model, x_support, y_support, x_query, chunk=32768, dtype=torch.float16, final_fp32=True):
    """== model(x_support, y_support, x_query, no padding)[0] for a classification Tab2D: logits (n_query, dim_output), float32.
    dtype = autocast dtype (None = fp32, no autocast). final_fp32=True runs the final LayerNorm + Linear without autocast:
    under autocast the stock model emits its logits in the low-precision dtype (bf16 rounds them to steps of ~0.06 at
    |logit| ~ 8, which leaves many tied probabilities)."""
    dev = x_support.device
    b, n_s, f = x_support.shape
    n_q = x_query.shape[1]
    pad_s = torch.zeros((b, n_s), dtype=torch.bool, device=dev)
    pad_f = torch.zeros((b, f), dtype=torch.bool, device=dev)
    with torch.autocast(device_type='cuda', dtype=dtype or torch.bfloat16, enabled=dtype is not None):
        xs, xq = model.x_quantile(x_support, x_query, pad_s, pad_f)
        ys, yq = model.y_embedding(y_support, pad_s, n_q)
        S = _embed(model, ys, xs, chunk)
        Q = _embed(model, yq, xq, chunk)
        del xs, xq, ys, yq
        for L in model.layers:
            layer_forward(L, S, Q, chunk)
        del S
        out = torch.empty((n_q, model.dim_output), dtype=torch.float32, device=dev)
        with torch.autocast(device_type='cuda', dtype=dtype or torch.bfloat16, enabled=dtype is not None and not final_fp32):
            for a in range(0, n_q, chunk):
                e = min(a + chunk, n_q)
                out[a:e] = model.final_layer(model.final_layer_norm(Q[0, a:e, 0].float())).float()
    return out
