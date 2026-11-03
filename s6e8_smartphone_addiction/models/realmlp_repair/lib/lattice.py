"""Masked lattice density transformer (target-free).

Every one of the 12 predictors is a token over the sorted values observed in the fit frame (ids 3..; 0 = unknown value,
1 = missing, 2 = mask); numeric fields also get a standardized scalar channel (clipped to +-8). The transformer embeds
each field (+ a field embedding + scalar * learned scale), prepends a CLS token, runs a pre-norm Transformer encoder and
predicts every field with its own softmax head. Training hides observed cells and minimizes their cross-entropy, so the
model learns the conditional distribution of a field given the other observed fields.

Masking schemes:
  train_masked     every observed cell is hidden with probability 0.15 (at least one per row)
  train_unitarget  half of the rows use that mask, the other half hide exactly one uniformly chosen observed field
"""
import json
import math
import random

import numpy as np
import pandas as pd

FIELDS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
          'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time', 'gender', 'stress_level',
          'academic_work_impact']
NUMERIC_COUNT = 9
SPECIAL_UNKNOWN, SPECIAL_MISSING, SPECIAL_MASK = 0, 1, 2

# per-fold state of the density-completion step (fit on the outer-training predictors of each fold)
FOLD_CONFIG = dict(embedding_dim=384, layers=4, heads=8, ff_multiplier=3, dropout=0.08, learning_rate=0.001,
                   weight_decay=0.00002, batch_size=2048, epochs=10, mask_probability=0.15, projection_dim=32)
FOLD_SEED = 60142  # + fold


class LatticeEncoder:
    def __init__(self, frame):
        self.vocabulary_values, self.vocabularies, self.means, self.stds = [], [], [], []
        for i, field in enumerate(FIELDS):
            observed = frame[field].dropna()
            if i < NUMERIC_COUNT:
                values = sorted(float(v) for v in observed.unique())
                raw = observed.to_numpy(np.float64)
                mean, std = float(raw.mean()), float(raw.std())
                if std < 1e-8:
                    std = 1.0
            else:
                values = sorted(str(v) for v in observed.astype(str).unique())
                mean, std = 0.0, 1.0
            self.vocabulary_values.append(values)
            self.vocabularies.append({v: k + 3 for k, v in enumerate(values)})
            self.means.append(mean)
            self.stds.append(std)

    @property
    def cardinalities(self):
        return [len(v) + 3 for v in self.vocabulary_values]

    def transform(self, frame):
        tokens = np.zeros((len(frame), len(FIELDS)), dtype=np.int64)
        scalars = np.zeros((len(frame), len(FIELDS)), dtype=np.float32)
        for i, field in enumerate(FIELDS):
            series = frame[field]
            missing = series.isna().to_numpy()
            if i < NUMERIC_COUNT:
                mapped = series.map(self.vocabularies[i]).fillna(0).to_numpy(np.int64)
                normalized = (series.to_numpy(np.float64) - self.means[i]) / self.stds[i]
                scalars[:, i] = np.clip(np.nan_to_num(normalized, nan=0.0), -8.0, 8.0).astype(np.float32)
            else:
                mapped = series.astype('string').map(self.vocabularies[i]).fillna(0).to_numpy(np.int64)
            mapped[missing] = SPECIAL_MISSING
            tokens[:, i] = mapped
        return tokens, scalars


def build_model(torch, cardinalities, config):
    nn = torch.nn
    dim = int(config['embedding_dim'])
    n_fields = len(FIELDS)

    class DensityTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleList([nn.Embedding(c, dim) for c in cardinalities])
            self.field_embedding = nn.Parameter(torch.zeros(1, n_fields, dim))
            self.scalar_scale = nn.Parameter(torch.empty(1, n_fields, dim))
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            layer = nn.TransformerEncoderLayer(d_model=dim, nhead=int(config['heads']),
                                               dim_feedforward=dim * int(config['ff_multiplier']),
                                               dropout=float(config['dropout']), activation='gelu', batch_first=True,
                                               norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=int(config['layers']), enable_nested_tensor=False)
            self.heads = nn.ModuleList([nn.Linear(dim, c) for c in cardinalities])
            generator = torch.Generator(device='cpu')
            generator.manual_seed(910247)
            projection = torch.randn(dim * 2, int(config['projection_dim']), generator=generator) / math.sqrt(dim * 2)
            self.register_buffer('projection', projection)
            nn.init.normal_(self.field_embedding, std=0.02)
            nn.init.normal_(self.scalar_scale, std=0.02)
            nn.init.normal_(self.cls, std=0.02)

        def forward(self, tokens, scalars):
            embedded = torch.stack([emb(tokens[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
            embedded = embedded + self.field_embedding + scalars.unsqueeze(-1) * self.scalar_scale
            cls = self.cls.expand(tokens.shape[0], -1, -1)
            encoded = self.transformer(torch.cat([cls, embedded], dim=1))
            return encoded[:, 0], encoded[:, 1:]

    return DensityTransformer()


def seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')


def bernoulli_mask(torch, observed, generator, mask_probability):
    mask = (torch.rand(observed.shape, generator=generator, device='cuda') < mask_probability) & observed
    empty = ~mask.any(dim=1)
    if empty.any():  # at least one hidden observed cell per row
        chosen = torch.multinomial(observed[empty].float(), num_samples=1).squeeze(1)
        mask[empty, chosen] = True
    return mask


def unitarget_mask(torch, observed, generator, mask_probability):
    standard_rows = torch.rand(len(observed), generator=generator, device=observed.device) < 0.5
    unitarget_rows = ~standard_rows
    mask = (torch.rand(observed.shape, generator=generator, device=observed.device) < mask_probability) \
        & observed & standard_rows[:, None]
    empty_standard = standard_rows & ~mask.any(dim=1)
    if empty_standard.any():
        chosen = torch.multinomial(observed[empty_standard].float(), num_samples=1, generator=generator).squeeze(1)
        mask[empty_standard, chosen] = True
    if unitarget_rows.any():
        chosen = torch.multinomial(observed[unitarget_rows].float(), num_samples=1, generator=generator).squeeze(1)
        mask[unitarget_rows, chosen] = True
    return mask


def train_model(torch, tokens, scalars, cardinalities, config, seed, scheme='masked'):
    """tokens / scalars: CUDA tensors of the fit rows. scheme 'masked' (Bernoulli) or 'unitarget' (hybrid)."""
    seed_everything(torch, seed)
    model = build_model(torch, cardinalities, config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']),
                                  weight_decay=float(config['weight_decay']))
    batch_size = int(config['batch_size'])
    p = float(config['mask_probability'])
    for epoch in range(1, int(config['epochs']) + 1):
        model.train()
        generator = torch.Generator(device='cuda')
        generator.manual_seed(seed * 100 + epoch)
        permutation = torch.randperm(len(tokens), generator=generator, device='cuda')
        epoch_loss, epoch_masked = 0.0, 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start:start + batch_size]
            original_tokens, original_scalars = tokens[index], scalars[index]
            observed = original_tokens.ge(3)
            if scheme == 'masked':
                mask = bernoulli_mask(torch, observed, generator, p)
            else:
                mask = unitarget_mask(torch, observed, generator, p)
            masked_tokens = original_tokens.masked_fill(mask, SPECIAL_MASK)
            masked_scalars = original_scalars.masked_fill(mask, 0.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, encoded_fields = model(masked_tokens, masked_scalars)
                loss_sum = torch.zeros((), device='cuda')
                count = 0
                for i, head in enumerate(model.heads):
                    active = mask[:, i]
                    if active.any():
                        logits = head(encoded_fields[active, i])
                        loss_sum = loss_sum + torch.nn.functional.cross_entropy(
                            logits, original_tokens[active, i], reduction='sum')
                        count += int(active.sum().item())
                loss = loss_sum / count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss += float(loss_sum.detach().cpu())
            epoch_masked += count
        print(json.dumps({'epoch': epoch, 'masked_cross_entropy': epoch_loss / epoch_masked}), flush=True)
    return model


def load_model(torch, path, cardinalities, config):
    model = build_model(torch, cardinalities, config).cuda()
    model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True), strict=True)
    model.eval()
    return model


def predictors(frame):
    return frame[FIELDS]


def concat_predictors(*frames):
    return pd.concat([f[FIELDS] for f in frames], ignore_index=True, copy=False)
