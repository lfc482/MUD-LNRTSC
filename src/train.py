import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .config import Config, save_config
from .dataset import (
    load_leave_one_well_data,
    fit_scaler_on_train_windows,
    transform_windows,
    WindowDataset,
)
from .evaluate import evaluate
from .losses import source_aware_loss
from .model import CNNLSTMRecon, copy_model, update_ema
from .utils import ensure_dir, set_seed


def build_model(cfg: Config, device: str):
    return CNNLSTMRecon(
        input_dim=len(cfg.feature_cols),
        num_classes=cfg.num_classes,
        conv_channels=cfg.conv_channels,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
    ).to(device)


def train_leave_one_out(test_well: int, cfg: Config, device: str):
    """Train on all wells except `test_well`, then test on the held-out well."""
    ensure_dir(cfg.save_dir)
    ensure_dir(cfg.result_dir)
    set_seed(cfg.seed)

    train_data, val_data, test_data, train_wells = load_leave_one_well_data(test_well, cfg)

    scaler = fit_scaler_on_train_windows(train_data["X"])
    train_data["X"] = transform_windows(train_data["X"], scaler)
    val_data["X"] = transform_windows(val_data["X"], scaler)
    test_data["X"] = transform_windows(test_data["X"], scaler)

    train_loader = DataLoader(WindowDataset(train_data), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(WindowDataset(val_data), batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(WindowDataset(test_data), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    student = build_model(cfg, device)
    teacher = copy_model(student).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_score = -1.0
    best_path = os.path.join(cfg.save_dir, f"proposed_test_well_{test_well}_best.pth")
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
        val_metrics, _ = evaluate(student, val_loader, device)
        mean_info = {k: float(np.mean([r[k] for r in running if k in r])) for k in running[0].keys()}
        row = {"epoch": epoch, **mean_info, **{f"val_{k}": v for k, v in val_metrics.items()}}
        log_rows.append(row)

        print(
            f"[Test well {test_well}] Epoch {epoch:02d} | "
            f"loss={mean_info['cls_loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['roc_auc']}"
        )

        score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["accuracy"]
        if score > best_score:
            best_score = score
            checkpoint = {
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "config": asdict(cfg),
                "feature_cols": list(cfg.feature_cols),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "best_val_metrics": val_metrics,
                "train_wells": train_wells,
                "test_well": test_well,
            }
            torch.save(checkpoint, best_path)

    pd.DataFrame(log_rows).to_csv(os.path.join(cfg.result_dir, f"train_log_test_well_{test_well}.csv"), index=False)

    test_metrics = test_checkpoint(best_path, cfg, device)
    return best_path, test_metrics


def test_checkpoint(checkpoint_path: str, cfg: Config, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    test_well = ckpt["test_well"]

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(cfg.feature_cols)

    # Load held-out test well data again.
    _, _, test_data, _ = load_leave_one_well_data(test_well, cfg)
    test_data["X"] = transform_windows(test_data["X"], scaler)
    test_loader = DataLoader(WindowDataset(test_data), batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["student_state_dict"])
    model.eval()

    pred_path = os.path.join(cfg.result_dir, f"proposed_test_well_{test_well}_predictions.csv")
    metrics, _ = evaluate(model, test_loader, device, prediction_save_path=pred_path)
    metrics_row = {"test_well": test_well, **metrics}
    pd.DataFrame([metrics_row]).to_csv(os.path.join(cfg.result_dir, f"metrics_test_well_{test_well}.csv"), index=False)
    print(f"Test well {test_well}: {metrics}")
    print(f"Prediction probabilities saved to: {pred_path}")
    return metrics_row


def run_all_folds(cfg: Config, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(cfg.save_dir)
    ensure_dir(cfg.result_dir)
    save_config(cfg, os.path.join(cfg.save_dir, "config.json"))

    rows = []
    for test_well in cfg.well_ids:
        _, metrics_row = train_leave_one_out(test_well, cfg, device)
        rows.append(metrics_row)

    df = pd.DataFrame(rows)
    mean_row = {"test_well": "mean"}
    std_row = {"test_well": "std"}
    for col in ["accuracy", "roc_auc", "recall", "f1", "precision"]:
        mean_row[col] = df[col].dropna().mean()
        std_row[col] = df[col].dropna().std()
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    out_path = os.path.join(cfg.result_dir, "leave_one_well_out_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Summary saved to: {out_path}")
    return df
