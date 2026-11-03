"""Mask-routed source-attention network.

context [404 fold features + 12 missing bits] -> context encoder (LayerNorm, 416 -> 512 -> 256, residual block)
donor tokens: for each of the 6 field blocks, 16 original-row donors, each a 30-d token (9 donor numerics, 9 deltas to
    the row's observed numerics, 3 donor categoricals, 3 categorical-match flags, original label, severity, retrieval
    score, distance kernel exp(-clip(score + 0.75, 0, 12)), exact fraction, rank) -> donor encoder (30 -> 128 -> 64)
attention: one 64-d query per block from the context vector; logits = q.k / 8 - softplus(distance_scale) * score
    + exact_scale * exact; per block the attended token + 5 attention summaries (label, severity, score, exactness,
    entropy) -> donor branch (414 -> 256 -> 192)
experts: 4 heads on [context 256, donor 192], mixed by a router that sees only the 12 missing bits and the 6 normalized
    block coverages
output logit = base_scale * base_margin + base_bias + routed residual (base_scale starts at 0.35)
auxiliary head: reconstruct the row's observed robust-scaled numerics from the 6 attended tokens."""
import math

import torch
from torch import nn

VARIANTS = {
    'context_control': {'use_attention': False, 'auxiliary_weight': 0.0},
    'attention_noaux': {'use_attention': True, 'auxiliary_weight': 0.0},
    'attention_aux': {'use_attention': True, 'auxiliary_weight': 0.015},
}
COVERAGE_MAX = (4.0, 3.0, 4.0, 4.0, 4.0, 5.0)  # fields per block


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width * 2, width))

    def forward(self, value):
        return value + self.net(self.norm(value))


class SourceAttentionModel(nn.Module):
    def __init__(self, context_dim, source_numeric, source_categorical_scaled, source_target, source_severity, variant,
                 top_k=16):
        super().__init__()
        config = VARIANTS[variant]
        self.use_attention = bool(config['use_attention'])
        self.auxiliary_weight = float(config['auxiliary_weight'])
        self.top_k = int(top_k)
        self.register_buffer('source_numeric', source_numeric.to(torch.float32))
        self.register_buffer('source_categorical_scaled', source_categorical_scaled.to(torch.float32))
        self.register_buffer('source_target', source_target.to(torch.float32))
        self.register_buffer('source_severity', source_severity.to(torch.float32))
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim + 12), nn.Linear(context_dim + 12, 512), nn.SiLU(), nn.Dropout(0.06),
            nn.Linear(512, 256), nn.SiLU(), ResidualBlock(256, 0.05))
        self.block_embedding = nn.Parameter(torch.empty(6, 64))
        nn.init.normal_(self.block_embedding, std=0.03)
        self.query_projection = nn.Linear(256, 6 * 64)
        self.donor_encoder = nn.Sequential(nn.LayerNorm(30), nn.Linear(30, 128), nn.SiLU(), nn.Linear(128, 64))
        self.distance_scale = nn.Parameter(torch.ones(6))
        self.exact_scale = nn.Parameter(torch.ones(6) * 0.5)
        # each block: 64-d attended token + label, severity, score, exactness and entropy summaries
        self.donor_branch = nn.Sequential(
            nn.LayerNorm(6 * 69), nn.Linear(6 * 69, 256), nn.SiLU(), nn.Dropout(0.05), nn.Linear(256, 192), nn.SiLU())
        self.router = nn.Sequential(nn.Linear(18, 48), nn.SiLU(), nn.Linear(48, 4))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(448), nn.Linear(448, 160), nn.SiLU(), nn.Dropout(0.06), nn.Linear(160, 1))
            for _ in range(4)])
        self.base_scale = nn.Parameter(torch.tensor(0.35))
        self.base_bias = nn.Parameter(torch.tensor(0.0))
        self.reconstruction_head = nn.Sequential(nn.LayerNorm(6 * 64), nn.Linear(6 * 64, 128), nn.SiLU(), nn.Linear(128, 9))

    def forward(self, context, base_margin, raw_numeric, raw_numeric_observed, raw_categorical_scaled,
                raw_categorical_observed, missing_mask, donor_indices, donor_score, donor_exact, coverage):
        context_vector = self.context_encoder(torch.cat([context, missing_mask], dim=1))
        block_available = coverage > 0
        coverage_normalized = coverage / torch.tensor(COVERAGE_MAX, dtype=coverage.dtype, device=coverage.device)[None, :]
        router_weight = torch.softmax(self.router(torch.cat([missing_mask, coverage_normalized], dim=1)), dim=1)
        reconstruction = None

        if self.use_attention:
            index = donor_indices.to(torch.long)
            source_numeric = self.source_numeric[index]
            source_categorical = self.source_categorical_scaled[index]
            source_target = self.source_target[index]
            source_severity = self.source_severity[index]
            numeric_delta = (source_numeric - raw_numeric[:, None, None, :]) * raw_numeric_observed[:, None, None, :]
            categorical_match = (torch.isclose(source_categorical, raw_categorical_scaled[:, None, None, :], rtol=0.0,
                                               atol=1e-6).to(context.dtype)
                                 * raw_categorical_observed[:, None, None, :])
            rank = torch.linspace(0.0, 1.0, self.top_k, dtype=context.dtype, device=context.device)[None, None, :, None]
            rank = rank.expand(len(context), 6, -1, -1)
            distance_kernel = torch.exp(-torch.clamp(donor_score + 0.75, min=0.0, max=12.0))
            token_input = torch.cat([
                source_numeric, numeric_delta, source_categorical, categorical_match, source_target[..., None],
                source_severity[..., None], donor_score[..., None], distance_kernel[..., None], donor_exact[..., None],
                rank], dim=-1)
            token = self.donor_encoder(token_input) + self.block_embedding[None, :, None, :]
            query = self.query_projection(context_vector).reshape(-1, 6, 64) + self.block_embedding[None, :, :]
            logits = (query[:, :, None, :] * token).sum(dim=-1) / math.sqrt(64.0)
            logits = logits - torch.nn.functional.softplus(self.distance_scale)[None, :, None] * donor_score
            logits = logits + self.exact_scale[None, :, None] * donor_exact
            attention = torch.softmax(logits, dim=-1)
            attention = attention * block_available[:, :, None].to(attention.dtype)
            attended = (attention[..., None] * token).sum(dim=2)
            entropy = -(attention * torch.log(torch.clamp(attention, min=1e-8))).sum(dim=2)
            explicit = torch.stack([
                (attention * source_target).sum(dim=2), (attention * source_severity).sum(dim=2),
                (attention * donor_score).sum(dim=2), (attention * donor_exact).sum(dim=2), entropy], dim=-1)
            block_summary = torch.cat([attended, explicit], dim=-1)
            donor_vector = self.donor_branch(block_summary.reshape(len(context), -1))
            reconstruction = self.reconstruction_head(attended.reshape(len(context), -1))
        else:
            donor_vector = torch.zeros((len(context), 192), dtype=context.dtype, device=context.device)

        shared = torch.cat([context_vector, donor_vector], dim=1)
        expert_logits = torch.cat([expert(shared) for expert in self.experts], dim=1)
        residual = (router_weight * expert_logits).sum(dim=1)
        output = self.base_scale * base_margin + self.base_bias + residual
        return output, reconstruction
