import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from .config import Config
from .utils import concat_data_dicts


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

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def max_consecutive_true(mask: np.ndarray) -> int:
    """Return the maximum length of consecutive True values."""
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def make_windows_from_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    risk_col: str,
    window_size: int,
    step_size: int,
    risk_threshold: float,
    high_risk_ratio_threshold: float,
    min_consecutive_high_risk: int,
) -> Dict[str, np.ndarray]:
    """
    Build sliding-window samples.

    Window label rule:
    A window is labeled as unstable if either:
    1) the ratio of high-risk points reaches `high_risk_ratio_threshold`, or
    2) the consecutive high-risk segment length reaches `min_consecutive_high_risk`.

    This avoids over-expansion caused by a single transient high-risk point.
    """
    features = df[feature_cols].values.astype(np.float32)
    risks = df[risk_col].values.astype(np.float32)

    X, y, risk_windows, centers, high_ratios = [], [], [], [], []
    n = len(df)
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        x_win = features[start:end]
        r_win = risks[start:end]
        high_mask = r_win >= risk_threshold
        high_ratio = float(np.mean(high_mask))
        max_run = max_consecutive_true(high_mask)
        label = 1 if (high_ratio >= high_risk_ratio_threshold or max_run >= min_consecutive_high_risk) else 0

        X.append(x_win)
        y.append(label)
        risk_windows.append(r_win)
        centers.append(start + window_size // 2)
        high_ratios.append(high_ratio)

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "risk_windows": np.asarray(risk_windows, dtype=np.float32),
        "centers": np.asarray(centers, dtype=np.float32),
        "high_ratios": np.asarray(high_ratios, dtype=np.float32),
    }


def split_dataframe_by_time(df: pd.DataFrame, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw continuous time series before window construction."""
    split_idx = int(len(df) * (1 - val_ratio))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def build_train_val_for_well(df: pd.DataFrame, cfg: Config) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train_df, val_df = split_dataframe_by_time(df, cfg.val_ratio)
    train_data = make_windows_from_dataframe(
        train_df, cfg.feature_cols, cfg.risk_col, cfg.window_size, cfg.step_size,
        cfg.risk_threshold, cfg.high_risk_ratio_threshold, cfg.min_consecutive_high_risk,
    )
    val_data = make_windows_from_dataframe(
        val_df, cfg.feature_cols, cfg.risk_col, cfg.window_size, cfg.step_size,
        cfg.risk_threshold, cfg.high_risk_ratio_threshold, cfg.min_consecutive_high_risk,
    )
    return train_data, val_data


def load_leave_one_well_data(test_well: int, cfg: Config):
    """Load and window data for leave-one-well-out cross-validation."""
    train_dicts, val_dicts = [], []
    train_wells = [w for w in cfg.well_ids if w != test_well]

    for well_id in train_wells:
        file_path = os.path.join(cfg.data_dir, cfg.file_pattern.format(well_id=well_id))
        df = read_well_file(file_path, cfg)
        train_data, val_data = build_train_val_for_well(df, cfg)
        train_dicts.append(train_data)
        val_dicts.append(val_data)

    test_file = os.path.join(cfg.data_dir, cfg.file_pattern.format(well_id=test_well))
    df_test = read_well_file(test_file, cfg)
    test_data = make_windows_from_dataframe(
        df_test, cfg.feature_cols, cfg.risk_col, cfg.window_size, cfg.step_size,
        cfg.risk_threshold, cfg.high_risk_ratio_threshold, cfg.min_consecutive_high_risk,
    )

    train_all = concat_data_dicts(train_dicts)
    val_all = concat_data_dicts(val_dicts)
    return train_all, val_all, test_data, train_wells


def fit_scaler_on_train_windows(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    n, t, f = X_train.shape
    scaler.fit(X_train.reshape(-1, f))
    return scaler


def transform_windows(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, t, f).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)
        self.risk_windows = torch.tensor(data["risk_windows"], dtype=torch.float32)
        self.centers = torch.tensor(data["centers"], dtype=torch.float32)
        self.high_ratios = torch.tensor(data["high_ratios"], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "risk_window": self.risk_windows[idx],
            "center": self.centers[idx],
            "high_ratio": self.high_ratios[idx],
        }
