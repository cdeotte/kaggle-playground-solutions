"""Native GPT-2 Yes-minus-No readout and the paired hard / soft update of the native GPT KD model."""
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F


def token_batch(ids, lengths, rows, device):
    lens = np.asarray(lengths[rows], dtype=np.int64)
    assert len(lens) and np.all(lens > 0)
    width = int((lens.max()+7)//8*8)
    tokens = np.asarray(ids[rows, :width], dtype=np.int64).copy()
    return torch.as_tensor(tokens, device=device), torch.as_tensor(lens, device=device)


def native_margin(model, ids, lengths, no_token, yes_token, amp=False):
    """Last real token's hidden state . (W[' Yes'] - W[' No']) with the tied output embedding, read out in fp32."""
    attention = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
    context = torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext()
    with context:
        h = model.transformer(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True).last_hidden_state
    last = h[torch.arange(len(ids), device=ids.device), lengths-1]
    weight = model.lm_head.weight
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    direction = weight[yes_token].to(dtype)-weight[no_token].to(dtype)
    return (last.to(dtype)*direction).sum(dim=-1)


def paired_objective(margin, true_target, soft_target, weight):
    n = len(true_target)
    assert margin.shape == (2*n,) and soft_target.shape == true_target.shape
    hard = F.binary_cross_entropy_with_logits(margin[:n], true_target)
    soft = F.binary_cross_entropy_with_logits(margin[n:], soft_target)
    return (hard+weight*soft)/(1+weight), hard, soft


def epoch_pairs(n_fit, n_query, seed, epoch):
    """Fit-row permutation and an equally long sequence of query rows (cycled permutations) for one epoch."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 424733]))
    fit_order = rng.permutation(n_fit)
    pieces = [rng.permutation(n_query) for _ in range((n_fit+n_query-1)//n_query)]
    query_order = np.concatenate(pieces)[:n_fit]
    return fit_order, query_order


def learning_rate(step, total_steps, peak):
    warmup = max(1, int(np.ceil(.05*total_steps)))
    if step < warmup:
        return peak*(step+1)/warmup
    return peak*max(0., (total_steps-step)/max(1, total_steps-warmup))


def update(model, optimizer, ids, lengths, true_rows, query_rows, true_target, soft_target, no_token, yes_token, cfg,
           device, amp=True):
    assert len(true_rows) == len(query_rows) == len(true_target) == len(soft_target)
    total = len(true_rows)
    optimizer.zero_grad(set_to_none=True)
    hard_total, soft_total = 0., 0.
    for start in range(0, total, cfg['microbatch']):
        end = min(start+cfg['microbatch'], total)
        rows = np.r_[true_rows[start:end], query_rows[start:end]]
        tokens, lens = token_batch(ids, lengths, rows, device)
        margin = native_margin(model, tokens, lens, no_token, yes_token, amp=amp)
        ty = torch.as_tensor(true_target[start:end], device=device, dtype=margin.dtype)
        sy = torch.as_tensor(soft_target[start:end], device=device, dtype=margin.dtype)
        loss, hard, soft = paired_objective(margin, ty, sy, cfg['teacher_weight'])
        assert torch.isfinite(loss)
        (loss*((end-start)/total)).backward()
        hard_total += float(hard.detach())*(end-start)
        soft_total += float(soft.detach())*(end-start)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
    assert torch.isfinite(norm)
    optimizer.step()
    return hard_total/total, soft_total/total, float(norm)
