"""Exact, memory-bounded LimiX-2 (v2.0) inference at large context: a row-chunked, layer-by-layer replacement for
model.v2_0.layer.LayerStack.forward that calls LimiX's own submodules.

Why: stock LimiX runs every member as ONE forward over [all training rows + all query rows]. When that does not fit in GPU
memory, the predictor falls back to query batches of 16,384 (then 8,192) rows, and every batch re-runs the whole training
context: dozens of full-context passes per member.

Why the chunked version is exact (model/v2_0/layer.py):
  * each layer is a sequence of sublayers (F = feature attention, S = sample/row attention, M = MLP), each wrapped in
    pre-norm (RMSNorm per cell) + residual;
  * F, M and the norms act on each row (cell) independently -> evaluated in row chunks;
  * S computes queries from ALL rows and keys/values from TRAINING rows only (x_kv = x[:, :eval_pos]); y_type is unused
    by it -> each chunk of rows attends to the full set of training rows of that layer.
So every row's computation is identical to the stock forward; only the batching changes. All rows (train + queries) are
kept at each layer, and the training context is processed exactly once per member.
Install with install(); remove with uninstall(). Needs the LimiX repository on sys.path (import model.v2_0.layer)."""
import functools

import torch

import model.v2_0.layer as _L

_ORIG = _L.LayerStack.forward
CHUNK = 65536


def _call(sublayer, x, feature_mask, eval_pos, y_type):
    if isinstance(sublayer, functools.partial):
        out = sublayer(x, feature_mask, eval_pos, y_type=y_type)
    else:
        out = sublayer(x, y_type=y_type)
    return out[0] if isinstance(out, tuple) else out


def _rows(fn, n_rows, chunk):
    """Evaluate fn(s, e) for row chunks and concatenate along the row axis (dim 1)."""
    res = None
    for s in range(0, n_rows, chunk):
        part = fn(s, min(s + chunk, n_rows))
        if res is None:
            res = torch.empty((part.shape[0], n_rows) + tuple(part.shape[2:]), dtype=part.dtype, device=part.device)
        res[:, s:s + part.shape[1]] = part
        del part
    return res


def _sub(t, s, e):
    return None if t is None else t[:, s:e]


@torch.no_grad()
def layer_forward(layer, x, eval_pos, feature_mask, y_type, chunk):
    """== EncoderBaseLayer.forward(x, eval_pos, feature_mask=..., y_type=...) in inference mode, row-chunked."""
    n = x.shape[1]
    for idx, (sublayer, layer_norm) in enumerate(zip(layer.layer_steps, layer.layer_norms)):
        is_seq = isinstance(sublayer, functools.partial) and getattr(sublayer.func, '__name__', '') == 'call_sequence_attention'
        inp = _rows(lambda s, e: layer_norm(x[:, s:e]), n, chunk) if layer.pre_norm else x
        if is_seq:
            index = sublayer.keywords.get('index', 0)
            mod = layer.sequence_attentions[index * 2 if layer.seq_attn_isolated else index]
            kv = inp[:, :eval_pos].transpose(1, 2)
            out = _rows(lambda s, e: mod(x=inp[:, s:e].transpose(1, 2), x_kv=kv, calculate_sample_attention=False,
                                         y_type=_sub(y_type, s, e))[0].transpose(1, 2), n, chunk)
        else:
            out = _rows(lambda s, e: _call(sublayer, inp[:, s:e], _sub(feature_mask, s, e), eval_pos, _sub(y_type, s, e)), n, chunk)
        if layer.pre_norm:
            del inp
            if layer.all_norm:
                out = _rows(lambda s, e: layer.output_layer_norms[idx](out[:, s:e]), n, chunk)
            x = out.add_(x.to(dtype=out.dtype))
        else:
            out = out.add_(x.to(dtype=out.dtype))
            x = _rows(lambda s, e: layer_norm(out[:, s:e]), n, chunk)
        del out
    return x


@torch.no_grad()
def stack_forward(self, x, **kwargs):
    """== LayerStack.forward for inference without attention maps; falls back to stock otherwise."""
    if self.training or kwargs.get('calculate_sample_attention') or kwargs.get('calculate_feature_attention'):
        return _ORIG(self, x, **kwargs)
    eval_pos, feature_mask, y_type = kwargs['eval_pos'], kwargs.get('feature_mask'), kwargs.get('y_type')
    if y_type is None:
        y_type = torch.ones_like(x[:, :, 0, 0])
    nlayers = len(self.layers)
    feature_emb = reg_y_emb = cls_y_emb = None
    for idx, layer in enumerate(self.layers):
        if idx == self.cls_only_start_layer:
            x = x[:, :, -self.num_cls_tokens:, :].contiguous()
        x = layer_forward(layer, x, eval_pos, feature_mask, y_type, CHUNK)
        if idx == self.feature_emb_layer and idx < nlayers - 1:
            feature_emb = x.cpu()
        if idx == self.reg_y_emb_layer and idx < nlayers - 1:
            reg_y_emb = x[:, :, -self.num_cls_tokens:, :].clone()
        if idx == self.cls_y_emb_layer and idx < nlayers - 1:
            cls_y_emb = x[:, :, -self.num_cls_tokens:, :].clone()
    if feature_emb is not None:
        feature_emb = feature_emb.to(x.device)
    return x, None, None, feature_emb, reg_y_emb, cls_y_emb


def install(chunk=65536):
    global CHUNK
    CHUNK = chunk
    _L.LayerStack.forward = stack_forward


def uninstall():
    _L.LayerStack.forward = _ORIG
