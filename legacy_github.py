import os
import json
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score


# =========================================================
# 1. Reproducibility
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    # data
    data_dir: str = "./data"
    save_dir: str = "./checkpoints"
    file_pattern: str = "well_{well_id}.xlsx"
    sheet_name: str = "Sheet1"
    well_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    # 14 drilling feature columns from your uploaded table header
    feature_cols: Tuple[str, ...] = (
        "井深(8004) m",
        "钻头位置(8005) m",
        "大钩高度(8008) m",
        "大钩负荷(8010) kN",
        "钻压(8011) kN",
        "转盘转速(8012) RPM",
        "扭矩(8013) kN.m",
        "立管压力(8018) MPa",
        "套管压力(8019) MPa",
        "钻时(8020) min/m",
        "入口密度(8104) g/cm3",
        "出口密度(8105) g/cm3",
        "入口流量(8110) L/s",
        "出口流量(百分)(8137) %",
    )
    risk_col: str = "井壁失稳概率"

    # window and label
    window_size: int = 500
    step_size: int = 2
    risk_threshold: float = 0.7

    # training
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.2
    seed: int = 42

    # method parameters
    num_classes: int = 2
    label_smoothing: float = 0.1
    warmup_epochs: int = 4
    low_conf_threshold: float = 0.60
    high_conf_threshold: float = 0.85
    ema_decay: float = 0.95

    # uncertainty parameters
    expert_sigma: float = 0.10
    boundary_beta: float = 0.5
    boundary_tau: float = 10.0
    recon_lambda: float = 0.1
    teacher_suppress_shift: float = 0.5
    teacher_suppress_boundary: float = 0.5
    sample_weight_min: float = 0.5
    sample_weight_max: float = 1.5

    # model
    conv_channels: int = 64
    lstm_hidden: int = 64
    lstm_layers: int = 1
    dropout: float = 0.3


# =========================================================
# 2. Dataset and window construction
# =========================================================

def read_well_file(path: str, cfg: Config) -> pd.DataFrame:
    """Read one well file. Supports .xlsx, .xls and .csv."""
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=cfg.sheet_name)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    required_cols = list(cfg.feature_cols) + [cfg.risk_col]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing columns in data file:\n"
            + "\n".join(missing)
            + "\n\nCurrent columns are:\n"
            + "\n".join(map(str, df.columns.tolist()))
        )

    # Remove blank rows in Excel.
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def make_windows_from_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    risk_col: str,
    window_size: int,
    step_size: int,
    risk_threshold: float,
) -> Dict[str, np.ndarray]:
    """
    Build sliding-window samples.
    Label rule: if any point in the window has risk_prob >= 0.7, the window label is 1; otherwise 0.
    """
    features = df[feature_cols].values.astype(np.float32)
    risks = df[risk_col].values.astype(np.float32)

    X, y, risk_windows, centers = [], [], [], []
    n = len(df)
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        x_win = features[start:end]
        r_win = risks[start:end]
        label = 1 if np.max(r_win) >= risk_threshold else 0
        X.append(x_win)
        y.append(label)
        risk_windows.append(r_win)
        centers.append(start + window_size // 2)

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "risk_windows": np.asarray(risk_windows, dtype=np.float32),
        "centers": np.asarray(centers, dtype=np.float32),
    }


def split_train_val_by_time(data: Dict[str, np.ndarray], val_ratio: float):
    """
    Temporal split before training/validation usage.
    It avoids random window leakage as much as possible.
    """
    n = len(data["X"])
    split_idx = int(n * (1 - val_ratio))
    train = {k: v[:split_idx] for k, v in data.items()}
    val = {k: v[split_idx:] for k, v in data.items()}
    return train, val


def fit_scaler_on_train_windows(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    n, t, f = X_train.shape
    scaler.fit(X_train.reshape(-1, f))
    return scaler


def transform_windows(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, f = X.shape
    X2 = scaler.transform(X.reshape(-1, f)).reshape(n, t, f)
    return X2.astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)
        self.risk_windows = torch.tensor(data["risk_windows"], dtype=torch.float32)
        self.centers = torch.tensor(data["centers"], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "risk_window": self.risk_windows[idx],
            "center": self.centers[idx],
        }


# =========================================================
# 3. CNN-LSTM with auxiliary reconstruction head
# =========================================================

class CNNLSTMRecon(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2, conv_channels: int = 64,
                 lstm_hidden: int = 64, lstm_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

        # lightweight reconstruction head: reconstruct full window from hidden state
        self.recon_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, input_dim)
        )

    def forward(self, x):
        # x: [B, T, F]
        z = x.transpose(1, 2)          # [B, F, T]
        z = self.conv(z)               # [B, C, T]
        z = z.transpose(1, 2)          # [B, T, C]
        out, _ = self.lstm(z)          # [B, T, H]
        h = out[:, -1, :]              # [B, H]
        h = self.dropout(h)
        logits = self.classifier(h)

        # reconstruct each time step using the LSTM outputs
        recon = self.recon_head(out)   # [B, T, F]
        return logits, recon, h


# =========================================================
# 4. EMA teacher
# =========================================================

@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float):
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)


def copy_model(student: nn.Module) -> nn.Module:
    import copy
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


# =========================================================
# 5. Soft-label and uncertainty functions
# =========================================================

def smooth_one_hot(y: torch.Tensor, num_classes: int, smoothing: float):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y_smooth = torch.full((y.size(0), num_classes), off_value, device=y.device)
    y_smooth.scatter_(1, y.unsqueeze(1), on_value)
    return y_smooth


def kl_soft_loss(logits: torch.Tensor, target_prob: torch.Tensor, reduction="none"):
    log_prob = F.log_softmax(logits, dim=1)
    loss = F.kl_div(log_prob, target_prob, reduction="none").sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    return loss


def compute_uncertainties(batch, x, y, logits, recon, cfg: Config):
    """
    Returns uncertainty components:
    u_exp: expert risk probability criticality
    u_win: window-level weak-label expansion uncertainty
    u_boundary: boundary transition uncertainty
    u_shift: reconstruction-enhanced distribution shift proxy

    This is a practical reconstruction of the formulas in the paper.
    """
    prob = F.softmax(logits.detach(), dim=1)
    confidence = prob.max(dim=1).values
    risk_window = batch["risk_window"].to(x.device)

    r_max = risk_window.max(dim=1).values
    r_mean = risk_window.mean(dim=1)
    high_ratio = (risk_window >= cfg.risk_threshold).float().mean(dim=1)

    # expert uncertainty: high when r_max is close to threshold 0.7
    u_exp = torch.exp(-torch.abs(r_max - cfg.risk_threshold) / cfg.expert_sigma)
    u_exp = torch.clamp(u_exp, 0.0, 1.0)

    # window weak-label expansion uncertainty
    u_win_pos = 1.0 - high_ratio
    u_win_neg = torch.clamp(1.0 - torch.abs(r_max - cfg.risk_threshold) / cfg.risk_threshold, 0.0, 1.0)
    u_win = torch.where(y == 1, u_win_pos, u_win_neg)
    u_win = torch.clamp(u_win, 0.0, 1.0)

    # boundary uncertainty: variance of risk probability; stronger around changing/border samples
    risk_var = torch.var(risk_window, dim=1)
    if torch.max(risk_var) > torch.min(risk_var):
        risk_var_norm = (risk_var - risk_var.min()) / (risk_var.max() - risk_var.min() + 1e-8)
    else:
        risk_var_norm = torch.zeros_like(risk_var)
    # no exact switch-point index is assumed here; use risk variance + threshold criticality proxy
    u_boundary = cfg.boundary_beta * risk_var_norm + (1 - cfg.boundary_beta) * u_exp
    u_boundary = torch.clamp(u_boundary, 0.0, 1.0)

    # distribution shift proxy: reconstruction error normalized within batch
    recon_err = ((recon.detach() - x) ** 2).mean(dim=(1, 2))
    if torch.max(recon_err) > torch.min(recon_err):
        u_shift = (recon_err - recon_err.min()) / (recon_err.max() - recon_err.min() + 1e-8)
    else:
        u_shift = torch.zeros_like(recon_err)
    u_shift = torch.clamp(u_shift, 0.0, 1.0)

    return {
        "confidence": confidence,
        "r_mean": r_mean,
        "u_exp": u_exp,
        "u_win": u_win,
        "u_boundary": u_boundary,
        "u_shift": u_shift,
    }


def build_expert_soft_target(r_mean: torch.Tensor):
    p1 = torch.clamp(r_mean, 0.0, 1.0)
    p0 = 1.0 - p1
    return torch.stack([p0, p1], dim=1)


def source_aware_loss(batch, x, y, student_logits, teacher_logits, recon, epoch: int, cfg: Config):
    """
    Source-aware soft supervision + auxiliary reconstruction loss.
    """
    y_smooth = smooth_one_hot(y, cfg.num_classes, cfg.label_smoothing)

    # warmup: only use smoothed original labels
    if epoch <= cfg.warmup_epochs:
        cls_loss_vec = kl_soft_loss(student_logits, y_smooth, reduction="none")
        recon_loss = F.mse_loss(recon, x)
        total_loss = cls_loss_vec.mean() + cfg.recon_lambda * recon_loss
        return total_loss, {
            "cls_loss": cls_loss_vec.mean().item(),
            "recon_loss": recon_loss.item(),
            "mean_conf": F.softmax(student_logits.detach(), dim=1).max(dim=1).values.mean().item(),
        }

    with torch.no_grad():
        teacher_prob = F.softmax(teacher_logits, dim=1)
        teacher_conf = teacher_prob.max(dim=1).values

    unc = compute_uncertainties(batch, x, y, student_logits, recon, cfg)
    confidence = unc["confidence"]
    u_exp = unc["u_exp"]
    u_win = unc["u_win"]
    u_boundary = unc["u_boundary"]
    u_shift = unc["u_shift"]
    expert_target = build_expert_soft_target(unc["r_mean"])

    # losses from different sources
    loss_raw = kl_soft_loss(student_logits, y_smooth, reduction="none")
    loss_expert = kl_soft_loss(student_logits, expert_target, reduction="none")
    loss_teacher = kl_soft_loss(student_logits, teacher_prob.detach(), reduction="none")

    # source-aware weights
    w_raw = (1.0 - u_exp) * (1.0 - u_win) * (1.0 - u_boundary)
    w_expert = u_exp + u_win + u_boundary
    w_teacher = teacher_conf * (1.0 - cfg.teacher_suppress_shift * u_shift) * (1.0 - cfg.teacher_suppress_boundary * u_boundary)
    w_teacher = torch.clamp(w_teacher, min=0.0)

    w_sum = w_raw + w_expert + w_teacher + 1e-8
    w_raw = w_raw / w_sum
    w_expert = w_expert / w_sum
    w_teacher = w_teacher / w_sum

    cls_loss_vec = w_raw * loss_raw + w_expert * loss_expert + w_teacher * loss_teacher

    # sample weight: suppress uncertain/noisy supervision softly, not remove samples
    sample_weight = (1.0 + confidence - u_exp - u_win + 0.3 * u_boundary)
    sample_weight = torch.clamp(sample_weight, cfg.sample_weight_min, cfg.sample_weight_max)
    sample_weight = sample_weight / (sample_weight.mean() + 1e-8)

    cls_loss = (sample_weight * cls_loss_vec).mean()
    recon_loss = F.mse_loss(recon, x)
    total_loss = cls_loss + cfg.recon_lambda * recon_loss

    return total_loss, {
        "cls_loss": cls_loss.item(),
        "recon_loss": recon_loss.item(),
        "mean_conf": confidence.mean().item(),
        "u_exp": u_exp.mean().item(),
        "u_win": u_win.mean().item(),
        "u_boundary": u_boundary.mean().item(),
        "u_shift": u_shift.mean().item(),
    }


# =========================================================
# 6. Train and evaluation
# =========================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        logits, _, _ = model(x)
        prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = (prob >= 0.5).astype(np.int64)
        ys.extend(y.tolist())
        preds.extend(pred.tolist())
        probs.extend(prob.tolist())

    ys = np.asarray(ys)
    preds = np.asarray(preds)
    probs = np.asarray(probs)

    out = {
        "accuracy": float(accuracy_score(ys, preds)),
        "recall": float(recall_score(ys, preds, zero_division=0)),
        "f1": float(f1_score(ys, preds, zero_division=0)),
        "precision": float(precision_score(ys, preds, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(ys, probs))
    except Exception:
        out["roc_auc"] = None
    return out


def train_one_round(train_well: int, cfg: Config, device: str):
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)

    # load training well
    train_file = os.path.join(cfg.data_dir, cfg.file_pattern.format(well_id=train_well))
    df_train_well = read_well_file(train_file, cfg)
    data_all = make_windows_from_dataframe(
        df_train_well, list(cfg.feature_cols), cfg.risk_col,
        cfg.window_size, cfg.step_size, cfg.risk_threshold
    )
    train_data, val_data = split_train_val_by_time(data_all, cfg.val_ratio)

    scaler = fit_scaler_on_train_windows(train_data["X"])
    train_data["X"] = transform_windows(train_data["X"], scaler)
    val_data["X"] = transform_windows(val_data["X"], scaler)

    train_loader = DataLoader(WindowDataset(train_data), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(WindowDataset(val_data), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    input_dim = len(cfg.feature_cols)
    student = CNNLSTMRecon(
        input_dim=input_dim,
        num_classes=cfg.num_classes,
        conv_channels=cfg.conv_channels,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
    ).to(device)
    teacher = copy_model(student).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_acc = -1.0
    best_path = os.path.join(cfg.save_dir, f"proposed_train_well_{train_well}_best.pth")
    log_rows = []

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        running = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            student_logits, recon, _ = student(x)
            with torch.no_grad():
                teacher_logits, _, _ = teacher(x)

            loss, info = source_aware_loss(batch, x, y, student_logits, teacher_logits, recon, epoch, cfg)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            optimizer.step()
            update_ema(student, teacher, cfg.ema_decay)

            running.append(info)

        scheduler.step()
        val_metrics = evaluate(student, val_loader, device)
        mean_info = {k: float(np.mean([r[k] for r in running if k in r])) for k in running[0].keys()}
        row = {"epoch": epoch, **mean_info, **{f"val_{k}": v for k, v in val_metrics.items()}}
        log_rows.append(row)

        print(f"[Train well {train_well}] Epoch {epoch:02d} | "
              f"loss={mean_info['cls_loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
              f"val_auc={val_metrics['roc_auc']}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            checkpoint = {
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "config": asdict(cfg),
                "feature_cols": list(cfg.feature_cols),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "best_val_metrics": val_metrics,
                "train_well": train_well,
            }
            torch.save(checkpoint, best_path)

    pd.DataFrame(log_rows).to_csv(os.path.join(cfg.save_dir, f"train_log_well_{train_well}.csv"), index=False)
    print(f"Best model saved to: {best_path}")
    return best_path


def test_cross_well(checkpoint_path: str, cfg: Config, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_well = ckpt["train_well"]

    # recover scaler
    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(cfg.feature_cols)

    model = CNNLSTMRecon(
        input_dim=len(cfg.feature_cols),
        num_classes=cfg.num_classes,
        conv_channels=cfg.conv_channels,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["student_state_dict"])
    model.eval()

    rows = []
    for well_id in cfg.well_ids:
        if well_id == train_well:
            continue
        test_file = os.path.join(cfg.data_dir, cfg.file_pattern.format(well_id=well_id))
        df_test = read_well_file(test_file, cfg)
        test_data = make_windows_from_dataframe(
            df_test, list(cfg.feature_cols), cfg.risk_col,
            cfg.window_size, cfg.step_size, cfg.risk_threshold
        )
        test_data["X"] = transform_windows(test_data["X"], scaler)
        test_loader = DataLoader(WindowDataset(test_data), batch_size=cfg.batch_size, shuffle=False)
        metrics = evaluate(model, test_loader, device)
        rows.append({"train_well": train_well, "test_well": well_id, **metrics})
        print(f"Train well {train_well} -> Test well {well_id}: {metrics}")

    df = pd.DataFrame(rows)
    mean_row = {"train_well": train_well, "test_well": "mean"}
    for col in ["accuracy", "roc_auc", "recall", "f1", "precision"]:
        mean_row[col] = df[col].dropna().mean()
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    out_path = os.path.join(cfg.save_dir, f"cross_well_results_train_well_{train_well}.csv")
    df.to_csv(out_path, index=False)
    print(f"Cross-well result saved to: {out_path}")
    return df


# =========================================================
# 7. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--train_well", type=int, default=1)
    parser.add_argument("--all_rounds", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    cfg = Config(data_dir=args.data_dir, save_dir=args.save_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    if args.test_only:
        if not args.checkpoint:
            raise ValueError("Please provide --checkpoint for test_only mode.")
        test_cross_well(args.checkpoint, cfg, device)
        return

    if args.all_rounds:
        all_means = []
        for train_well in cfg.well_ids:
            best_path = train_one_round(train_well, cfg, device)
            df_res = test_cross_well(best_path, cfg, device)
            all_means.append(df_res[df_res["test_well"] == "mean"].iloc[0].to_dict())
        pd.DataFrame(all_means).to_csv(os.path.join(cfg.save_dir, "all_rounds_mean_results.csv"), index=False)
    else:
        best_path = train_one_round(args.train_well, cfg, device)
        test_cross_well(best_path, cfg, device)


if __name__ == "__main__":
    main()
