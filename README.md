# MUD-LNRTSC

This repository provides an implementation template for the **multi-source uncertainty decomposition based hierarchical mixed-supervision LNRTSC** method for cross-well wellbore instability prediction.

The repository is prepared for paper review and reproducibility. The raw field drilling data are **not** included due to confidentiality restrictions.

## Repository contents

```text
MUD-LNRTSC/
├── configs/                 # Hyperparameter configuration
├── src/                     # Source code for model, dataset, loss, training and evaluation
├── scripts/                 # Runnable scripts
├── examples/                # Dummy data-format example only
├── checkpoints/             # Model checkpoints, not raw data
├── results/                 # Output metrics and prediction probabilities
└── figures/                 # Generated figures such as ROC curves
```

## Data availability

The field drilling data used in the paper are not publicly released due to confidentiality restrictions and engineering data protection requirements. Only the implementation code, configuration files and dummy-format examples are provided.

The file `examples/example_data_format.csv` contains **dummy values** and is used only to demonstrate the required column names.

## Installation

```bash
pip install -r requirements.txt
```

## Data format

Each well file should be stored in `data/` and named as:

```text
well_1.xlsx
well_2.xlsx
...
well_6.xlsx
```

CSV files are also supported by changing `file_pattern` in `configs/default.yaml`.

The required feature columns and risk-probability column are defined in `configs/default.yaml`.

## Run leave-one-well-out cross-validation

Run all six folds:

```bash
python scripts/run_leave_one_well_out.py --config configs/default.yaml --all
```

Run a single fold, e.g., hold out Well 1 as the test well:

```bash
python scripts/run_leave_one_well_out.py --config configs/default.yaml --test_well 1
```

## Plot ROC curves

After evaluation, prediction-probability CSV files are saved in `results/`. Each file contains `y_true` and `y_score` columns.

```bash
python scripts/plot_roc.py --prediction_dir results --output figures/Figure_4_1_ROC_curves.png
```

## Notes

- Raw drilling data should not be uploaded to this repository.
- Model checkpoints can be placed in `checkpoints/` after confidentiality approval.
- The implementation follows the paper setting of multi-well training and single-well testing under leave-one-well-out cross-validation.
