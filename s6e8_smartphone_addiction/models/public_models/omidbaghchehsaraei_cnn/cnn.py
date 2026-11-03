# Code of the public notebook "CNN for Predicting Smartphone Addiction" (version 1), cells 1-9:
# https://www.kaggle.com/code/omidbaghchehsaraei/cnn-for-predicting-smartphone-addiction
# Importing this module seeds the RNGs (seed 42) and builds the notebook's label-free features of train and test.
# Changes for our folds: data come from common.data; compute_fold_encodings gives the training rows inner
# out-of-fold target encodings; the fold loop of cell 9 is the function train_fold (no other change).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data import read_test, read_train  # noqa: E402

# notebook cell 1
import gc
import math
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore") 
SEED = 42

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# notebook cell 3
train_df = read_train()
test_df = read_test()

TARGET = "addicted_label"
ID_COL = "id" if "id" in train_df.columns else None

initial_features = [c for c in train_df.columns if c not in [TARGET, ID_COL]]

# notebook cell 5
def engineer_features(df):
    
    df = df.copy()

    df["missing_count"] = df[initial_features].astype(str).isin(["nan", "None", "<NA>", "np.nan"]).sum(axis=1).astype(float)
    
    def to_float(col_name):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name].astype(str), errors="coerce")
        return pd.Series(np.nan, index=df.index)
    
    s_media = to_float("social_media_hours")
    gaming = to_float("gaming_hours")
    work = to_float("work_study_hours")
    daily_screen = to_float("daily_screen_time_hours")
    sleep = to_float("sleep_hours")
    weekend_screen = to_float("weekend_screen_time")
    app_opens = to_float("app_opens_per_day")
    notifications = to_float("notifications_per_day")

    df["total_breakdown_hours"] = s_media.fillna(0.0) + gaming.fillna(0.0) + work.fillna(0.0)
    df["social_ratio"] = s_media / (daily_screen + 1e-5)
    df["gaming_ratio"] = gaming / (daily_screen + 1e-5)
    df["work_ratio"] = work / (daily_screen + 1e-5)
    df["unaccounted_screen_time"] = daily_screen - df["total_breakdown_hours"]
    df["unaccounted_screen_ratio"] = df["unaccounted_screen_time"] / (daily_screen + 1e-5)
    df["screen_to_sleep_ratio"] = daily_screen / (sleep + 1e-5)
    df["weekend_vs_daily_ratio"] = weekend_screen / (daily_screen + 1e-5)
    df["app_opens_per_hour"] = app_opens / (daily_screen + 1e-5)
    df["notifications_per_hour"] = notifications / (daily_screen + 1e-5)
    
    df["leisure_hours"] = s_media.fillna(0.0) + gaming.fillna(0.0)
    df["productivity_ratio"] = work / (df["leisure_hours"] + 1e-5)
    df["entertainment_disruption_index"] = df["leisure_hours"] / (sleep + 1e-5)
    df["screen_intensity_per_unlock"] = daily_screen / (app_opens + 1e-5)
    df["notification_distraction_index"] = notifications / (daily_screen + 1e-5)
    df["opens_per_notification"] = app_opens / (notifications + 1e-5)
    df["sleep_deprivation_index"] = (daily_screen + 1e-5) * (24.0 - sleep.clip(0, 24) + 1e-5)
    df["weekly_screen_estimate"] = (daily_screen * 5.0) + (weekend_screen * 2.0)

    df["log_notifications"] = np.log1p(notifications.clip(lower=0).fillna(0.0))
    df["sqrt_notifications"] = np.sqrt(notifications.clip(lower=0).fillna(0.0))
    df["log_app_opens"] = np.log1p(app_opens.clip(lower=0).fillna(0.0))
    df["sqrt_app_opens"] = np.sqrt(app_opens.clip(lower=0).fillna(0.0))
    df["log_daily_screen"] = np.log1p(daily_screen.clip(lower=0).fillna(0.0))
    df["sqrt_daily_screen"] = np.sqrt(daily_screen.clip(lower=0).fillna(0.0))
    df["log_leisure"] = np.log1p(df["leisure_hours"].clip(lower=0))

    engineered_cont_cols = [
        "total_breakdown_hours", "social_ratio", "gaming_ratio", "work_ratio",
        "unaccounted_screen_time", "unaccounted_screen_ratio", "screen_to_sleep_ratio", 
        "weekend_vs_daily_ratio", "app_opens_per_hour", "notifications_per_hour", 
        "productivity_ratio", "entertainment_disruption_index", "screen_intensity_per_unlock",
        "notification_distraction_index", "opens_per_notification", "sleep_deprivation_index",
        "weekly_screen_estimate", "leisure_hours", "log_notifications", "sqrt_notifications",
        "log_app_opens", "sqrt_app_opens", "log_daily_screen", "sqrt_daily_screen", "log_leisure"
    ]
    
    df[engineered_cont_cols] = df[engineered_cont_cols].fillna(0.0)
    return df

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

cat_cols = [c for c in initial_features if c in train_df.columns]

def compute_fold_encodings(tr_df, va_df, te_df, cat_cols, target_col, smoothing=10):
    """Frequency and smoothed target encodings of one fold. Our run's version of the notebook's function: the training
    rows get 5-fold inner out-of-fold target encodings (the notebook encoded them with the map fitted on all training
    rows); validation and test rows use the map fitted on all training rows, as in the notebook. Frequencies are
    unchanged."""
    global_mean = float(tr_df[target_col].mean())
    tr_out = pd.DataFrame(index=tr_df.index)
    va_out = pd.DataFrame(index=va_df.index)
    te_out = pd.DataFrame(index=te_df.index)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for col in cat_cols:
        col_str_tr = tr_df[col].astype(str)
        col_str_va = va_df[col].astype(str)
        col_str_te = te_df[col].astype(str)
        counts = col_str_tr.value_counts()
        tr_out[f"{col}_freq"] = col_str_tr.map(counts).astype(float).fillna(0.0).astype(np.float32)
        va_out[f"{col}_freq"] = col_str_va.map(counts).astype(float).fillna(0.0).astype(np.float32)
        te_out[f"{col}_freq"] = col_str_te.map(counts).astype(float).fillna(0.0).astype(np.float32)
        encoded = np.full(len(tr_df), global_mean, dtype=np.float32)
        for fit_pos, hold_pos in inner.split(tr_df, tr_df[target_col]):
            fit_subset = tr_df.iloc[fit_pos]
            fit_target = fit_subset[target_col]
            fit_mean = float(fit_target.mean())
            fit_keys = fit_subset[col].astype(str)
            stats = pd.DataFrame({"key": fit_keys.to_numpy(), "target": fit_target.to_numpy()}).groupby(
                "key").target.agg(["count", "mean"])
            mapping = (stats["count"] * stats["mean"] + smoothing * fit_mean) / (stats["count"] + smoothing)
            encoded[hold_pos] = col_str_tr.iloc[hold_pos].map(mapping).fillna(fit_mean).to_numpy(dtype=np.float32)
        full_stats = pd.DataFrame({"key": col_str_tr.to_numpy(), "target": tr_df[target_col].to_numpy()}).groupby(
            "key").target.agg(["count", "mean"])
        full_mapping = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (
            full_stats["count"] + smoothing)
        tr_out[f"{col}_te"] = encoded
        va_out[f"{col}_te"] = col_str_va.map(full_mapping).fillna(global_mean).astype(np.float32)
        te_out[f"{col}_te"] = col_str_te.map(full_mapping).fillna(global_mean).astype(np.float32)
    return tr_out, va_out, te_out


# notebook cell 7
class PiecewiseLinearEmbedding(nn.Module):
    def __init__(self, num_features, num_bins=12):
        super(PiecewiseLinearEmbedding, self).__init__()
        self.num_features = num_features
        self.num_bins = num_bins
        self.weights = nn.Parameter(torch.randn(num_features, num_bins) * 0.1)
        self.biases = nn.Parameter(torch.zeros(num_features, num_bins))

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        out = x_exp * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        return torch.relu(out)

class PeriodicEmbedding(nn.Module):
    def __init__(self, num_features, emb_dim=12, sigma=2.33):
        super(PeriodicEmbedding, self).__init__()
        self.coefficients = nn.Parameter(torch.randn(num_features, emb_dim // 2) * sigma)
        self.phase = nn.Parameter(torch.zeros(num_features, emb_dim // 2))

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        angles = 2 * math.pi * x_expanded * self.coefficients.unsqueeze(0) + self.phase.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class MultiSampleDropoutHead(nn.Module):
    def __init__(self, in_features, out_features=1, num_samples=5, drop_rate=0.2):
        super(MultiSampleDropoutHead, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(num_samples)])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.mean(torch.stack([self.fc(drop(x)) for drop in self.dropouts], dim=0), dim=0)

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock1D, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.Mish(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        w = self.fc(x).view(b, c, 1)
        return x * w

class QuantilePLR_TabularCNN(nn.Module):
    def __init__(self, num_features, plr_bins=12, fourier_dim=12, dropout_rate=0.08):
        super(QuantilePLR_TabularCNN, self).__init__()
        
        self.plr_emb = PiecewiseLinearEmbedding(num_features, num_bins=plr_bins)
        self.periodic_emb = PeriodicEmbedding(num_features, emb_dim=fourier_dim)
        
        total_emb_dim = plr_bins + fourier_dim
        
        self.linear_proj = nn.Sequential(
            nn.Linear(total_emb_dim, 32),
            nn.Mish()
        )
        
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.se = SEBlock1D(128)
        
        self.act = nn.Mish()
        self.drop = nn.Dropout(dropout_rate)
        
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        
        self.backbone = nn.Sequential(
            nn.Linear(128 * 2 + num_features, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish()
        )
        
        self.msd_head = MultiSampleDropoutHead(in_features=64, out_features=1, num_samples=5, drop_rate=0.2)

    def forward(self, x_raw):
        plr = self.plr_emb(x_raw)
        fourier = self.periodic_emb(x_raw)
        
        combined_emb = torch.cat([plr, fourier], dim=-1)
        emb = self.linear_proj(combined_emb)
        
        x = emb.transpose(1, 2)
        
        x = self.act(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.se(x)
        
        avg_p = self.global_avg(x).squeeze(-1)
        max_p = self.global_max(x).squeeze(-1)
        
        cat_features = torch.cat([avg_p, max_p, x_raw], dim=1)
        
        feat_repr = self.backbone(cat_features)
        logits = self.msd_head(feat_repr)
        return logits.squeeze(-1)

def predict_in_batches(model, x_data, device, batch_size=512):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(x_data.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(batch_x)
                probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy())
            
    return np.array(preds)

# notebook cell 9 (the body of its fold loop is train_fold; train.py runs our 5 folds in order)
EPOCHS = 100
BATCH_SIZE = 128
PATIENCE = 15
LR = 0.002
WD = 0.015
LABEL_SMOOTHING = 0.01

y = train_df[TARGET].values.astype(np.float32)

num_base_cols = train_df.select_dtypes(include=[np.number]).columns.drop([ID_COL, TARGET], errors='ignore').tolist()


def train_fold(fold, train_idx, val_idx):
    """One fold of the notebook's training loop; returns the validation and test predictions."""
    print(f"\n" + "="*20 + f" FOLD {fold + 1} / 5 " + "="*20)

    tr_df, va_df = train_df.iloc[train_idx].copy(), train_df.iloc[val_idx].copy()
    te_df = test_df.copy()

    tr_enc, va_enc, te_enc = compute_fold_encodings(
        tr_df, va_df, te_df, cat_cols=cat_cols, target_col=TARGET
    )

    X_tr_df = pd.concat([tr_df[num_base_cols].reset_index(drop=True), tr_enc.reset_index(drop=True)], axis=1).fillna(0.0)
    X_va_df = pd.concat([va_df[num_base_cols].reset_index(drop=True), va_enc.reset_index(drop=True)], axis=1).fillna(0.0)
    X_te_df = pd.concat([te_df[num_base_cols].reset_index(drop=True), te_enc.reset_index(drop=True)], axis=1).fillna(0.0)

    X_tr = X_tr_df.values.astype(np.float32)
    X_va = X_va_df.values.astype(np.float32)
    X_te = X_te_df.values.astype(np.float32)
    
    y_tr = y[train_idx]
    y_va = y[val_idx]

    scaler = QuantileTransformer(output_distribution="normal", random_state=SEED, n_quantiles=1000)
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    y_tr_smooth = y_tr * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

    train_dataset = TensorDataset(
        torch.from_numpy(X_tr), 
        torch.from_numpy(y_tr_smooth),
        torch.from_numpy(y_tr)
    )
    val_dataset = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = QuantilePLR_TabularCNN(num_features=X_tr.shape[1], plr_bins=12, fourier_dim=12, dropout_rate=0.08).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=1, eta_min=1e-6
    )
    scaler_amp = torch.amp.GradScaler('cuda')

    best_val_auc = 0.0
    best_weights = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_preds_list = []
        train_targets_list = []

        for batch_x, batch_y_smooth, batch_y_raw in train_loader:
            batch_x = batch_x.to(device)
            batch_y_smooth = batch_y_smooth.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(batch_x)
                loss = criterion(logits, batch_y_smooth)
                
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_loss += loss.item() * batch_x.size(0)
            train_preds_list.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets_list.extend(batch_y_raw.numpy())

        train_loss = train_loss / len(train_loader.dataset)
        train_auc = roc_auc_score(train_targets_list, train_preds_list)

        model.eval()
        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                val_preds_list.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets_list.extend(batch_y.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets_list, val_preds_list)
        
        scheduler.step()

        print(f"Epoch {epoch:03d}/{EPOCHS:03d} | "
              f"Train Loss: {train_loss:.4f} - Train AUC: {train_auc:.5f} | "
              f"Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.5f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"--> Early stopping triggered at Epoch {epoch}. Best Val ROC-AUC: {best_val_auc:.5f}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    fold_val_preds = predict_in_batches(model, X_va, device, batch_size=512)
    fold_test_preds = predict_in_batches(model, X_te, device, batch_size=512)

    print(f"--> Fold {fold + 1} Best ROC-AUC: {roc_auc_score(y_va, fold_val_preds):.5f}")

    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    return fold_val_preds, fold_test_preds
