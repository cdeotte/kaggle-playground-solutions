#!/usr/bin/env python
"""Next-token "acceptance" features of the GReaT replicas: cache/acceptance/{train,test}.npy + manifest.json.

1. Maps (cache/acceptance/raw/accept_<replica>_c<held>.npy, one per replica and held field, float32 [train + test rows, 36]).
   The GPT-2 replicas e55 and e60 of the data generator (../gpt2_replica, trained on the original data only) read every train
   and test row as GReaT text with the label first, once with "Will_Buy_EV is No" and once with "Will_Buy_EV is Yes":
       "Will_Buy_EV is Yes, Age is 35, Gender is Male, ..., <held field> is"
   (the 12 other fields in the original column order, then the held field). At the next token (temperature 0.7) we compute
   18 statistics per prompt: log-probabilities of the row's actual first token, of " None" (the GReaT spelling of a missing
   value) and of the "legal" tokens (the first tokens of the held field's values in the original data), the entropy of the
   legal distribution, margins and ranks of the actual token, and for numeric fields the expected / observed / spread of the
   legal value centres. No competition labels are read. About 30 minutes per map on one GPU (26 maps).
2. Pack (576 columns): for every held field and statistic, 4 combinations of the two prompts and the two replicas (mean of
   the Yes - No differences, e60 - e55 difference of those, mean of the prompt means, e60 - e55 difference of those), grouped
   in families; manifest.json has names, families, held_columns, numeric_columns. xgb_acceptance_entropy uses the 52 columns
   of family constrained_entropy (the legal-token entropy).
Maps that exist are skipped, so the maps can be split over several runs; the pack is built when all 26 maps exist.
Usage: python acceptance_cache.py [--checkpoints e55,e60] [--held 0,1,...,12] [--batch-rows 512]"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, TARGET, read_original  # noqa: E402
from common.pairs import fold_list  # noqa: E402

OUT = HERE / 'cache' / 'acceptance'
RAW = OUT / 'raw'
REPLICAS = HERE.parent / 'gpt2_replica' / 'checkpoints'
CHECKPOINTS = ('e55', 'e60')
# field order of the original data, in which the replicas were trained
SOURCE_ORDER = ('Age', 'Gender', 'Annual_Income_USD', 'City_Type', 'Daily_Commute_km', 'Number_of_Cars_Owned',
                'Current_Car_Type', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Home_Charging_Possible',
                'Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level')
TEMPERATURE = 0.70
TOP_K = 50
STAT_NAMES = ('actual_logp_full', 'none_logp_full', 'legal_mass_logp_full', 'accept_mass_logp_full', 'actual_logp_legal',
              'actual_logp_accept', 'none_logp_accept', 'survive_logp_accept', 'legal_entropy_normalized',
              'best_legal_logp_full', 'actual_margin_best_legal', 'none_margin_best_legal', 'actual_rank_legal_fraction',
              'actual_top50_margin', 'none_top50_margin', 'expected_legal_center_normalized',
              'observed_minus_expected_center_normalized', 'legal_std_normalized')

NULLABLE = {'Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level'}
CONSTRAINED_STATS = ('actual_logp_full', 'actual_logp_legal', 'actual_margin_best_legal', 'actual_rank_legal_fraction',
                     'actual_top50_margin', 'legal_entropy_normalized')
ACCEPTANCE_STATS = ('none_logp_full', 'none_logp_accept', 'survive_logp_accept', 'legal_mass_logp_full',
                    'accept_mass_logp_full', 'none_margin_best_legal', 'none_top50_margin')
GEOMETRY_STATS = ('expected_legal_center_normalized', 'observed_minus_expected_center_normalized', 'legal_std_normalized')
TRANSFORMS = ('prompt_delta_mean', 'checkpoint_delta_prompt_delta', 'prompt_mean', 'checkpoint_delta_prompt_mean')


def raw_path(checkpoint, held_index):
    return RAW / f'accept_{checkpoint}_c{held_index:02d}.npy'


# ----------------------------------------------------------------------------------------------------------------- maps
def first_token_ids(tokenizer, values):
    return np.asarray([tokenizer.encode(' ' + value, add_special_tokens=False)[0] for value in values], dtype=np.int64)


def legal_prefix_spec(tokenizer, held):
    """Legal first tokens of the held field (from its original-data values) and, for numeric fields, the normalized mean
    value behind each token."""
    source = read_original()[held].dropna().drop_duplicates()
    token_ids = first_token_ids(tokenizer, source.astype(str).tolist())
    unique_ids = np.unique(token_ids)
    numeric = bool(pd.api.types.is_numeric_dtype(source.dtype))
    centers = np.zeros(len(unique_ids), dtype=np.float32)
    lower, upper = 0.0, 1.0
    if numeric:
        values = pd.to_numeric(source, errors='raise').to_numpy(np.float64)
        lower, upper = float(np.min(values)), float(np.max(values))
        scale = max(upper - lower, 1.0)
        for index, token_id in enumerate(unique_ids):
            centers[index] = np.float32((float(np.mean(values[token_ids == token_id])) - lower) / scale)
    return unique_ids, centers, numeric, lower, upper


def build_prefixes(frame, context_order, held):
    contexts = [', '.join(f'{column} is {value}' for column, value in zip(context_order, row))
                for row in frame[context_order].astype(str).itertuples(index=False, name=None)]
    return [f'{TARGET} is {label}, {context}, {held} is' for label in ('No', 'Yes') for context in contexts]


def prompt_statistics(scaled_logits, actual_ids, none_id, legal_ids, legal_centers, numeric_observed):
    import torch
    full_lse = torch.logsumexp(scaled_logits, dim=1)
    rows = torch.arange(len(scaled_logits), device=scaled_logits.device)
    actual = scaled_logits[rows, actual_ids]
    none = scaled_logits[:, none_id]
    legal = scaled_logits.index_select(1, legal_ids)
    legal_lse = torch.logsumexp(legal, dim=1)
    accept_lse = torch.logaddexp(legal_lse, none)
    legal_prob = torch.softmax(legal, dim=1)
    legal_entropy = -(legal_prob * torch.log_softmax(legal, dim=1)).sum(1) / float(np.log(max(2, legal.shape[1])))
    best_legal = legal.max(1).values
    actual_rank = (legal > actual[:, None]).sum(1).float() / float(legal.shape[1])
    threshold = torch.topk(scaled_logits, k=TOP_K, dim=1, sorted=False).values.min(1).values
    if numeric_observed is None:
        expected_center = center_residual = center_std = torch.zeros_like(actual)
    else:
        expected_center = (legal_prob * legal_centers[None, :]).sum(1)
        variance = (legal_prob * (legal_centers[None, :] - expected_center[:, None]).square()).sum(1)
        center_residual = numeric_observed - expected_center
        center_std = torch.sqrt(torch.clamp_min(variance, 0.0))
    return torch.column_stack([
        actual - full_lse, none - full_lse, legal_lse - full_lse, accept_lse - full_lse, actual - legal_lse,
        actual - accept_lse, none - accept_lse, legal_lse - accept_lse, legal_entropy, best_legal - full_lse,
        actual - best_legal, none - best_legal, actual_rank, actual - threshold, none - threshold, expected_center,
        center_residual, center_std])


def score_batch(model, tokenizer, batch, context_order, held, spec, none_id, device):
    import torch
    legal_ids_np, legal_centers_np, numeric, lower, upper = spec
    token = tokenizer(build_prefixes(batch, context_order, held), add_special_tokens=True, padding=True, truncation=True,
                      max_length=256, return_tensors='pt')
    ids = token['input_ids'].to(device, non_blocking=True)
    attention = token['attention_mask'].to(device, non_blocking=True)
    assert int(attention.sum(1).max()) < 256, 'sequence truncated'
    n = len(batch)
    actual_ids_np = first_token_ids(tokenizer, batch[held].astype(str).tolist())
    assert set(actual_ids_np.tolist()) <= set(legal_ids_np.tolist()), 'a leading token is not in the original support'
    actual_ids = torch.as_tensor(np.concatenate([actual_ids_np, actual_ids_np]), dtype=torch.long, device=device)
    numeric_observed = None
    if numeric:
        observed = (pd.to_numeric(batch[held], errors='raise').to_numpy(np.float32) - lower) / max(upper - lower, 1.0)
        numeric_observed = torch.as_tensor(np.concatenate([observed, observed]), dtype=torch.float32, device=device)
    legal_ids = torch.as_tensor(legal_ids_np, dtype=torch.long, device=device)
    legal_centers = torch.as_tensor(legal_centers_np, dtype=torch.float32, device=device)
    with torch.inference_mode():  # fp32 activations (TF32 matmuls)
        hidden = model.transformer(input_ids=ids, attention_mask=attention).last_hidden_state
        last = hidden[torch.arange(len(ids), device=device), attention.sum(1) - 1]
        logits = model.lm_head(last)
    stats = prompt_statistics(logits.float() / TEMPERATURE, actual_ids, none_id, legal_ids, legal_centers, numeric_observed)
    return torch.column_stack([stats[:n], stats[n:]]).cpu().numpy().astype(np.float32)  # 18 "No" + 18 "Yes" columns


def score_maps(checkpoints, held_indices, batch_rows):
    if all(raw_path(c, h).exists() for c in checkpoints for h in held_indices):
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    cols = list(SOURCE_ORDER)
    frame = pd.concat([pd.read_csv(DATA_DIR / 'train.csv', usecols=cols), pd.read_csv(DATA_DIR / 'test.csv', usecols=cols)],
                      ignore_index=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    none_id = int(tokenizer.encode(' None', add_special_tokens=False)[0])
    device = torch.device('cuda')
    RAW.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        todo = [h for h in held_indices if not raw_path(checkpoint, h).exists()]
        if not todo:
            continue
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        model.load_state_dict(torch.load(REPLICAS / f'{checkpoint}.pt', map_location='cpu'), strict=True)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.to(device=device).eval()
        for held_index in todo:
            held = SOURCE_ORDER[held_index]
            context_order = [c for c in SOURCE_ORDER if c != held]
            spec = legal_prefix_spec(tokenizer, held)
            result = np.empty((len(frame), 2 * len(STAT_NAMES)), dtype=np.float32)
            started = time.time()
            for begin in range(0, len(frame), batch_rows):
                end = min(begin + batch_rows, len(frame))
                result[begin:end] = score_batch(model, tokenizer, frame.iloc[begin:end], context_order, held, spec, none_id,
                                                device)
            assert np.isfinite(result).all()
            np.save(raw_path(checkpoint, held_index), result)
            print(f'{checkpoint} {held}: {len(frame)} rows in {time.time() - started:.0f}s', flush=True)
        del model
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------- pack
def transformed(e55, e60, stat_index):
    width = e55.shape[1] // 2
    no55, yes55 = np.asarray(e55[:, stat_index], np.float32), np.asarray(e55[:, width + stat_index], np.float32)
    no60, yes60 = np.asarray(e60[:, stat_index], np.float32), np.asarray(e60[:, width + stat_index], np.float32)
    delta55, delta60 = yes55 - no55, yes60 - no60
    mean55, mean60 = (yes55 + no55) * np.float32(0.5), (yes60 + no60) * np.float32(0.5)
    return (delta55 + delta60) * np.float32(0.5), delta60 - delta55, (mean55 + mean60) * np.float32(0.5), mean60 - mean55


def build_pack():
    n_train = len(pd.read_csv(DATA_DIR / 'train.csv', usecols=['id']))
    original = read_original()
    held_columns = list(SOURCE_ORDER)
    numeric_columns = {held for held in held_columns if pd.api.types.is_numeric_dtype(original[held].dtype)}
    loaded = {held: {c: np.load(raw_path(c, i), mmap_mode='r') for c in CHECKPOINTS} for i, held in enumerate(held_columns)}
    n_test = len(loaded[held_columns[0]]['e55']) - n_train
    stat_index = {name: i for i, name in enumerate(STAT_NAMES)}

    plan = []
    families = {'constrained_all': [], 'acceptance_nullable': [], 'numeric_geometry': [], 'global_summaries': []}

    def declare(family, held, statistic, transform):
        families[family].append(len(plan))
        plan.append((family, held, statistic, transform))

    for held in held_columns:
        for statistic in CONSTRAINED_STATS:
            for transform in TRANSFORMS:
                declare('constrained_all', held, statistic, transform)
        if held in NULLABLE:
            for statistic in ACCEPTANCE_STATS:
                for transform in TRANSFORMS:
                    declare('acceptance_nullable', held, statistic, transform)
        if held in numeric_columns:
            for statistic in GEOMETRY_STATS:
                for transform in TRANSFORMS:
                    declare('numeric_geometry', held, statistic, transform)
    # mean / std / min / max over the 13 fields of every constrained statistic
    for statistic in CONSTRAINED_STATS:
        for transform in TRANSFORMS:
            for aggregation in ('mean', 'std', 'min', 'max'):
                declare('global_summaries', aggregation, statistic, transform)

    names = [''] * len(plan)
    OUT.mkdir(parents=True, exist_ok=True)
    train_out = np.lib.format.open_memmap(OUT / 'train.npy', mode='w+', dtype=np.float32, shape=(n_train, len(plan)))
    test_out = np.lib.format.open_memmap(OUT / 'test.npy', mode='w+', dtype=np.float32, shape=(n_test, len(plan)))
    for column, (family, held, statistic, transform) in enumerate(plan):
        if family == 'global_summaries':
            continue
        values = transformed(loaded[held]['e55'], loaded[held]['e60'], stat_index[statistic])[TRANSFORMS.index(transform)]
        train_out[:, column] = values[:n_train]
        test_out[:, column] = values[n_train:]
        names[column] = f'gpt_accept::{held}::{statistic}::{transform}'
    lookup = {(held, statistic, transform): i for i, (family, held, statistic, transform) in enumerate(plan)
              if family == 'constrained_all'}
    for column, (family, aggregation, statistic, transform) in enumerate(plan):
        if family != 'global_summaries':
            continue
        indices = [lookup[(held, statistic, transform)] for held in held_columns]
        for output, rows in ((train_out, n_train), (test_out, n_test)):
            for begin in range(0, rows, 32_768):
                end = min(begin + 32_768, rows)
                block = np.asarray(output[begin:end, indices], dtype=np.float32)
                output[begin:end, column] = getattr(block, aggregation)(1)
        names[column] = f'gpt_accept::GLOBAL_{aggregation}::{statistic}::{transform}'
    train_out.flush()
    test_out.flush()
    del train_out, test_out

    def semantic(name, predicate):
        families[name] = [i for i, spec in enumerate(plan) if predicate(*spec)]

    semantic('constrained_actual', lambda fam, held, stat, tr: fam == 'constrained_all'
             and stat in {'actual_logp_full', 'actual_logp_legal'})
    semantic('constrained_relative', lambda fam, held, stat, tr: fam == 'constrained_all'
             and stat in {'actual_margin_best_legal', 'actual_rank_legal_fraction', 'actual_top50_margin'})
    semantic('constrained_entropy', lambda fam, held, stat, tr: fam == 'constrained_all' and stat == 'legal_entropy_normalized')
    semantic('acceptance_probability', lambda fam, held, stat, tr: fam == 'acceptance_nullable' and stat in {
        'none_logp_full', 'none_logp_accept', 'survive_logp_accept', 'legal_mass_logp_full', 'accept_mass_logp_full'})
    semantic('acceptance_margin', lambda fam, held, stat, tr: fam == 'acceptance_nullable'
             and stat in {'none_margin_best_legal', 'none_top50_margin'})
    semantic('numeric_center', lambda fam, held, stat, tr: fam == 'numeric_geometry'
             and stat in {'expected_legal_center_normalized', 'observed_minus_expected_center_normalized'})
    semantic('numeric_spread', lambda fam, held, stat, tr: fam == 'numeric_geometry' and stat == 'legal_std_normalized')
    semantic('counterfactual_delta_all', lambda fam, held, stat, tr: tr in {'prompt_delta_mean', 'checkpoint_delta_prompt_delta'})
    semantic('prompt_mean_all', lambda fam, held, stat, tr: tr in {'prompt_mean', 'checkpoint_delta_prompt_mean'})
    families['acceptance_plus_constrained'] = sorted(set(families['acceptance_nullable'] + families['constrained_all']))
    families['all_compact'] = sorted(set(families['acceptance_nullable'] + families['constrained_all']
                                         + families['numeric_geometry'] + families['global_summaries']))
    manifest = {'names': names, 'families': families, 'held_columns': held_columns, 'numeric_columns': sorted(numeric_columns),
                'nullable_original_columns': sorted(NULLABLE), 'checkpoints': list(CHECKPOINTS), 'temperature': TEMPERATURE,
                'shape_train': [n_train, len(plan)], 'shape_test': [n_test, len(plan)], 'dtype': 'float32'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'pack: {len(plan)} columns -> {OUT}; ' + ', '.join(f'{k} {len(v)}' for k, v in families.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', default='e55,e60')
    ap.add_argument('--held', default=','.join(str(i) for i in range(len(SOURCE_ORDER))))
    ap.add_argument('--batch-rows', type=int, default=512)
    a = ap.parse_args()
    score_maps([c for c in a.checkpoints.split(',') if c], fold_list(a.held), a.batch_rows)
    if all(raw_path(c, i).exists() for c in CHECKPOINTS for i in range(len(SOURCE_ORDER))):
        build_pack()
    else:
        print('pack: not all maps are done yet', flush=True)


if __name__ == '__main__':
    main()
