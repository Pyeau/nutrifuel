# FYP Evaluation Tools

This document explains how to use the automated evaluation scripts created for your Final Year Project. These scripts generate the necessary graphs and data for your report.

## 1. Model Evaluation Script (`evaluate_model.py`)
This script evaluates your trained Random Forest model (`meal_plan_model.joblib`) against the dataset (`Final_data.csv`).

### Usage:
Run the following command in your terminal:
```bash
python evaluate_model.py
```

### Outputs (saved in `fyp_evaluation_results/`):
*   `confusion_matrix.png`: A heatmap showing how well the model predicts each goal (Weight Loss, Build Muscle, etc.).
*   `feature_importance.png`: A bar chart showing which factors (BMI, Fat%, Age) affect the decision the most.
*   `classification_report.csv`: A table with precision, recall, and F1-scores.

---

## 2. Data Split Comparison Script (`compare_splits.py`)
This script proves the robustness of your model by retraining it on different data split ratios (80/20, 70/30, 60/40) and comparing the accuracy.

### Usage:
Run the following command in your terminal:
```bash
python compare_splits.py
```

### Outputs (saved in `fyp_evaluation_results/`):
*   `split_comparison.png`: A bar chart comparing accuracy across different splits.
*   `split_comparison_results.csv`: The raw accuracy numbers.

---

## 3. Results for Report
A draft of your "Results & Discussion" chapter based on these findings can be found in:
*   `Chapter4_Draft.md`

You can copy the text from that file directly into your Microsoft Word report.
