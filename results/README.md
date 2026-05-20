# Results

This folder is reserved for locally generated training logs, evaluation metrics and prediction-probability CSV files.

For confidentiality reasons, model output results are not included in this public repository.

When users run the code locally with their own data, the generated result files will be saved here automatically. Prediction files used for ROC plotting should contain at least the following columns:

```csv
y_true,y_score
0,0.132
1,0.846
