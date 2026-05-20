from dataclasses import dataclass, asdict, fields
from typing import List
import json
import yaml


@dataclass
class Config:
    # data
    data_dir: str = "./data"
    save_dir: str = "./checkpoints"
    result_dir: str = "./results"
    file_pattern: str = "well_{well_id}.xlsx"
    sheet_name: str = "Sheet1"
    well_ids: List[int] = None

    # columns
    feature_cols: List[str] = None
    risk_col: str = "井壁失稳概率"

    # window and label
    window_size: int = 500
    step_size: int = 2
    risk_threshold: float = 0.7
    high_risk_ratio_threshold: float = 0.02
    min_consecutive_high_risk: int = 5

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

    def __post_init__(self):
        if self.well_ids is None:
            self.well_ids = [1, 2, 3, 4, 5, 6]
        if self.feature_cols is None:
            self.feature_cols = [
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
            ]


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    valid_keys = {f.name for f in fields(Config)}
    data = {k: v for k, v in data.items() if k in valid_keys}
    return Config(**data)


def save_config(cfg: Config, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
