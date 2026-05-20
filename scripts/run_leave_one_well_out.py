import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.config import load_config
from src.train import run_all_folds, train_leave_one_out
from src.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--test_well", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Run all leave-one-well-out folds.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    if args.result_dir is not None:
        cfg.result_dir = args.result_dir

    ensure_dir(cfg.save_dir)
    ensure_dir(cfg.result_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if args.all:
        run_all_folds(cfg, device)
    else:
        train_leave_one_out(args.test_well, cfg, device)


if __name__ == "__main__":
    main()
