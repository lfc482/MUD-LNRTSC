from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score


@torch.no_grad()
def evaluate(model, loader, device: str, prediction_save_path: Optional[str] = None) -> Tuple[dict, pd.DataFrame]:
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

    metrics = {
        "accuracy": float(accuracy_score(ys, preds)),
        "recall": float(recall_score(ys, preds, zero_division=0)),
        "f1": float(f1_score(ys, preds, zero_division=0)),
        "precision": float(precision_score(ys, preds, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(ys, probs))
    except Exception:
        metrics["roc_auc"] = None

    pred_df = pd.DataFrame({"y_true": ys, "y_score": probs, "y_pred": preds})
    if prediction_save_path is not None:
        pred_df.to_csv(prediction_save_path, index=False)
    return metrics, pred_df
