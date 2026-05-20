import os
import json
import random
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def concat_data_dicts(dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not dicts:
        raise ValueError("No data dictionaries to concatenate.")
    keys = dicts[0].keys()
    return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in keys}
