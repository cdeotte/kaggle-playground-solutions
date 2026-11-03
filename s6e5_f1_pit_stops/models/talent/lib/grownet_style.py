"""GrowNet-style additive network with one logit (a local PyTorch model, not TALENT's GrowNet).

Input: the standardized numerics concatenated with a d_embedding embedding of every categorical key (index 0 = value not
seen in training). Each stage is an MLP (Linear hidden_d, BatchNorm, SiLU, dropout, Linear hidden_d, SiLU, dropout,
Linear 1); the logit is stage_lr (a learned scalar, initialised to 1) x the sum of all stage outputs.

Training: up to max_epoch stages. Every stage adds a new MLP and trains the whole model (all stages and the embeddings,
with a fresh AdamW) for epochs_per_stage epochs of BCE with gradient-norm clipping. The held-out AUC is measured after
every stage; the model is rebuilt with the stages of the best-AUC checkpoint, which predicts, and growth stops
`patience` stages after the best stage.
"""
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .tensor_data import encode_fold, make_tensor_loader, predict_proba


class GrowNetStage(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BinaryGrowNet(torch.nn.Module):
    def __init__(self, n_num, cat_cardinalities, emb_dim=16, hidden_dim=256, dropout=0.05, stage_lr=1.0):
        super().__init__()
        self.n_num = int(n_num)
        self.cat_cardinalities = list(cat_cardinalities)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.stage_lr = torch.nn.Parameter(torch.tensor(float(stage_lr), dtype=torch.float32))
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(int(card), self.emb_dim) for card in self.cat_cardinalities
        ])
        self.stages = torch.nn.ModuleList()
        self.in_dim = self.n_num + len(self.cat_cardinalities) * self.emb_dim
        for emb in self.embeddings:
            torch.nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def add_stage(self):
        stage = GrowNetStage(self.in_dim, self.hidden_dim, self.dropout)
        self.stages.append(stage)
        return stage

    def features(self, x_num, x_cat):
        parts = []
        if self.n_num:
            parts.append(x_num)
        if len(self.embeddings):
            emb_parts = [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
            parts.append(torch.cat(emb_parts, dim=1))
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def forward(self, x_num, x_cat):
        feats = self.features(x_num, x_cat)
        if len(self.stages) == 0:
            return torch.zeros(feats.shape[0], dtype=feats.dtype, device=feats.device)
        logits = torch.zeros(feats.shape[0], dtype=feats.dtype, device=feats.device)
        for stage in self.stages:
            logits = logits + stage(feats)
        return self.stage_lr * logits


def fit_grownet_fold(config, batch_size, max_epoch, Xtr_num, Xtr_cat, y_tr, Xva_num, Xva_cat, y_va, Xte_num, Xte_cat,
                     fold_dir, seed):
    """Grows stages on the training rows, keeps the best held-out-AUC stage count; returns (held-out, test)
    probabilities."""
    cfg_model = config["model"]
    cfg_train = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    (Xtr_num_s, Xtr_cat_i), (Xva_num_s, Xva_cat_i), (Xte_num_s, Xte_cat_i), cat_cards = encode_fold(
        Xtr_num, Xtr_cat, Xva_num, Xva_cat, Xte_num, Xte_cat)

    def new_model():
        return BinaryGrowNet(
            n_num=Xtr_num_s.shape[1],
            cat_cardinalities=cat_cards,
            emb_dim=cfg_model["d_embedding"],
            hidden_dim=cfg_model["hidden_d"],
            dropout=cfg_model["dropout"],
            stage_lr=cfg_model["stage_lr"],
        ).to(device)

    model = new_model()
    train_loader = make_tensor_loader(Xtr_num_s, Xtr_cat_i, y_tr, batch_size, shuffle=True, seed=seed)
    best_auc = -np.inf
    best_stage = -1
    best_state = None
    criterion = torch.nn.BCEWithLogitsLoss()
    patience = int(cfg_model.get("patience", 7))

    for stage_idx in range(max_epoch):
        model.add_stage().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"])
        model.train()
        losses = []
        for _ in range(int(cfg_train.get("epochs_per_stage", 1))):
            for xb_num, xb_cat, yb in train_loader:
                xb_num = xb_num.to(device, non_blocking=True)
                xb_cat = xb_cat.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb_num, xb_cat)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_train.get("grad_clip", 5.0)))
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

        val_prob = predict_proba(model, Xva_num_s, Xva_cat_i, batch_size=max(batch_size, 8192))
        val_auc = roc_auc_score(y_va, val_prob)
        print(f"Stage {stage_idx + 1:02d}/{max_epoch} loss={np.mean(losses):.5f} val_auc={val_auc:.6f}", flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            best_stage = stage_idx + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, fold_dir / "grownet_style_best.pt")
        elif (stage_idx + 1) - best_stage >= patience:
            print(f"Early stopping at stage {stage_idx + 1}; best stage {best_stage} AUC {best_auc:.6f}", flush=True)
            break

    if best_state is None:
        best_stage = len(model.stages)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Rebuild with exactly the number of stages of the best checkpoint (a model with more stages cannot load it).
    best_model = new_model()
    for _ in range(best_stage):
        best_model.add_stage().to(device)
    best_model.load_state_dict(best_state, strict=True)

    val_prob = np.clip(predict_proba(best_model, Xva_num_s, Xva_cat_i, batch_size=max(batch_size, 8192)), 1e-7, 1 - 1e-7)
    test_prob = np.clip(predict_proba(best_model, Xte_num_s, Xte_cat_i, batch_size=max(batch_size, 8192)), 1e-7, 1 - 1e-7)
    return val_prob, test_prob
