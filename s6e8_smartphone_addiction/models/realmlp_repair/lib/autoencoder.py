"""Masked mixed-type Transformer autoencoder used by the reconstruction step.

Tokens: numeric value x learned weight + bias + field embedding, categorical embeddings + field embedding, plus a
missing embedding; CLS token; 4-layer pre-norm Transformer (d 128, 8 heads, ff 384, dropout 0.10). Heads: a Gaussian
(mean, log scale clamped to [-4, 2]) per numeric field and a softmax per categorical field.
Training hides an empirical missingness pattern of a random training row (OR an independent 8% mask) on the observed
cells and minimizes the Gaussian NLL + categorical cross-entropy of the hidden cells.
Prediction hides one field at a time; with mc_samples > 1 dropout stays on (MC dropout) and the spread of the means is
the epistemic std, the mean predicted scale the aleatoric std."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AutoencoderConfig:
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 384
    dropout: float = 0.10
    latent_dim: int = 32


def build_model(category_cardinalities, config):
    import torch
    from torch import nn

    class MaskedTabularAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_numeric = 9
            self.n_categorical = len(category_cardinalities)
            self.n_fields = self.n_numeric + self.n_categorical
            self.field_embedding = nn.Parameter(torch.empty(self.n_fields, config.d_model))
            self.numeric_weight = nn.Parameter(torch.empty(self.n_numeric, config.d_model))
            self.numeric_bias = nn.Parameter(torch.zeros(self.n_numeric, config.d_model))
            self.category_embeddings = nn.ModuleList(
                [nn.Embedding(c + 1, config.d_model, padding_idx=0) for c in category_cardinalities])
            self.missing_embedding = nn.Embedding(2, config.d_model)
            self.cls = nn.Parameter(torch.empty(1, 1, config.d_model))
            layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads,
                                               dim_feedforward=config.d_ff, dropout=config.dropout, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(config.d_model)
            self.numeric_heads = nn.ModuleList([nn.Linear(config.d_model, 2) for _ in range(self.n_numeric)])
            self.category_heads = nn.ModuleList([nn.Linear(config.d_model, c) for c in category_cardinalities])
            self.latent_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.latent_dim),
                                             nn.Tanh())
            nn.init.normal_(self.field_embedding, std=0.02)
            nn.init.normal_(self.numeric_weight, std=0.02)
            nn.init.normal_(self.cls, std=0.02)

        def forward(self, numeric, categorical, observed):
            batch = numeric.shape[0]
            numeric_tokens = (numeric[:, :, None] * self.numeric_weight[None, :, :] + self.numeric_bias[None, :, :]
                              + self.field_embedding[None, :self.n_numeric, :])
            category_tokens = [emb(categorical[:, i]) + self.field_embedding[self.n_numeric + i][None, :]
                               for i, emb in enumerate(self.category_embeddings)]
            tokens = torch.cat([numeric_tokens, torch.stack(category_tokens, dim=1)], dim=1)
            tokens = tokens + self.missing_embedding((~observed).long())
            cls = self.cls.expand(batch, -1, -1)
            encoded = self.norm(self.encoder(torch.cat([cls, tokens], dim=1)))
            field_state = encoded[:, 1:, :]
            numeric_output = torch.stack([head(field_state[:, i, :]) for i, head in enumerate(self.numeric_heads)],
                                         dim=1)
            means = numeric_output[:, :, 0]
            log_scales = numeric_output[:, :, 1].clamp(-4.0, 2.0)
            category_logits = [head(field_state[:, self.n_numeric + i, :])
                               for i, head in enumerate(self.category_heads)]
            latent = self.latent_head(encoded[:, 0, :])
            return means, log_scales, category_logits, latent

    return MaskedTabularAutoencoder()


def sample_artificial_mask(observed, empirical_masks, generator, independent_probability):
    """Real missingness pattern of a random row OR independent masks, restricted to observed cells."""
    import torch
    batch = observed.shape[0]
    choice = torch.randint(0, empirical_masks.shape[0], (batch,), generator=generator, device=observed.device)
    pattern = empirical_masks[choice]
    independent = torch.rand(observed.shape, generator=generator, device=observed.device) < independent_probability
    artificial = observed & (pattern | independent)
    empty = ~artificial.any(dim=1)
    if bool(empty.any()):
        rows = torch.nonzero(empty, as_tuple=False).flatten()
        sampled = torch.multinomial(observed[rows].float(), 1, replacement=True, generator=generator)
        artificial[rows, sampled[:, 0]] = True
    return artificial


def reconstruction_loss(means, log_scales, category_logits, numeric_target, categorical_target, artificial_mask):
    import torch
    import torch.nn.functional as F
    numeric_mask = artificial_mask[:, :means.shape[1]]
    gaussian = 0.5 * torch.square((numeric_target - means) / torch.exp(log_scales)) + log_scales
    numeric_loss = (gaussian * numeric_mask).sum() / numeric_mask.sum().clamp_min(1)
    losses = []
    offset = means.shape[1]
    for i, logits in enumerate(category_logits):
        mask = artificial_mask[:, offset + i]
        if bool(mask.any()):
            losses.append(F.cross_entropy(logits[mask], categorical_target[mask, i] - 1))
    categorical_loss = torch.stack(losses).mean() if losses else numeric_loss.new_zeros(())
    return numeric_loss + categorical_loss


def predict_masked_field(model, numeric, categorical, observed, field_index, device, batch_size=8192, mc_samples=1):
    import torch
    n_rows = len(numeric)
    is_numeric = field_index < numeric.shape[1]
    means = np.empty((mc_samples, n_rows), dtype=np.float32)
    aleatoric = np.empty((mc_samples, n_rows), dtype=np.float32)
    probabilities = []
    original_mode = model.training
    for sample in range(mc_samples):
        model.train(mc_samples > 1)
        if not is_numeric and sample == 0:
            cardinality = model.category_heads[field_index - numeric.shape[1]].out_features
            probabilities = [np.empty((n_rows, cardinality), dtype=np.float32) for _ in range(mc_samples)]
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            num = torch.as_tensor(numeric[start:stop], device=device)
            cat = torch.as_tensor(categorical[start:stop], device=device)
            obs = torch.as_tensor(observed[start:stop], device=device).clone()
            obs[:, field_index] = False
            if is_numeric:
                num[:, field_index] = 0.0
            else:
                cat[:, field_index - numeric.shape[1]] = 0
            with torch.inference_mode():
                pred_mean, pred_log_scale, pred_logits, _ = model(num, cat, obs)
            if is_numeric:
                means[sample, start:stop] = pred_mean[:, field_index].cpu().numpy()
                aleatoric[sample, start:stop] = torch.exp(pred_log_scale[:, field_index]).cpu().numpy()
            else:
                probabilities[sample][start:stop] = torch.softmax(
                    pred_logits[field_index - numeric.shape[1]], dim=1).cpu().numpy()
    model.train(original_mode)
    if is_numeric:
        return {'mean': means.mean(axis=0), 'epistemic_std': means.std(axis=0), 'aleatoric_std': aleatoric.mean(axis=0)}
    return {'probability': np.stack(probabilities, axis=0).mean(axis=0)}


def predict_latent_summary(model, numeric, categorical, observed, device, batch_size=8192):
    """Permutation-stable summaries of the CLS latent with the row's real mask: norm, mean, std, mean |z|, max |z|."""
    import torch
    output = np.empty((len(numeric), 5), dtype=np.float32)
    original_mode = model.training
    model.eval()
    for start in range(0, len(numeric), batch_size):
        stop = min(start + batch_size, len(numeric))
        num = torch.as_tensor(numeric[start:stop], device=device)
        cat = torch.as_tensor(categorical[start:stop], device=device)
        obs = torch.as_tensor(observed[start:stop], device=device)
        with torch.inference_mode():
            _, _, _, latent = model(num, cat, obs)
        latent = latent.float()
        output[start:stop, 0] = torch.linalg.vector_norm(latent, dim=1).cpu().numpy()
        output[start:stop, 1] = latent.mean(dim=1).cpu().numpy()
        output[start:stop, 2] = latent.std(dim=1).cpu().numpy()
        output[start:stop, 3] = latent.abs().mean(dim=1).cpu().numpy()
        output[start:stop, 4] = latent.abs().max(dim=1).values.cpu().numpy()
    model.train(original_mode)
    return output
