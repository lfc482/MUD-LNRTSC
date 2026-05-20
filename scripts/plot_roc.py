import argparse
import os
import glob

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def infer_label(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    name = name.replace("_predictions", "")
    name = name.replace("proposed_", "MUD-LNRTSC_")
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="figures/Figure_4_1_ROC_curves.png")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.prediction_dir, "*.csv")))
    files = [f for f in files if "prediction" in os.path.basename(f).lower()]
    if not files:
        raise FileNotFoundError(
            "No prediction CSV files found. Each file should contain `y_true` and `y_score` columns."
        )

    plt.figure(figsize=(5.6, 4.4), dpi=300)
    for f in files:
        df = pd.read_csv(f)
        if "y_true" not in df.columns or "y_score" not in df.columns:
            continue
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_score"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=1.2, label=f"{infer_label(f)} ({roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.9, label="Random")
    plt.xlabel("False Positive Rate", fontsize=8)
    plt.ylabel("True Positive Rate", fontsize=8)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(loc="lower right", fontsize=5.8, frameon=True, borderpad=0.25,
               handlelength=1.3, handletextpad=0.4, labelspacing=0.25)
    plt.grid(True, linestyle="--", linewidth=0.35, alpha=0.45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.savefig(os.path.splitext(args.output)[0] + ".tiff", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
