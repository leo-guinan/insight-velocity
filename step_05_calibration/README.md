# Step 5: Backtest & Calibration

## What This Step Does

Calibrates the synthesis prediction model by:

1. **Evaluating Forecasts:** Tests precision, recall, ROC-AUC, PR-AUC, Brier score
2. **Learning Better Weights:** Trains a model to predict synthesis from features
3. **Granger Causality:** Tests if Outliers/Entropy "cause" IV changes
4. **Reweighting IV Components:** Optimizes feature importance

**Why it matters:** Ensures your synthesis predictions are accurate and calibrated. Learns optimal feature weights from data rather than hand-picked values.

## How It Works

### Target Definition
- **Binary target:** Synthesis = 1 if posts > 0, else 0
- **Features:** Breadth, Novelty, ΔIV, IV Momentum, Exploration Pressure, Integration, Compression, In Sweet Spot

### Model Training
- **Algorithm:** Logistic Regression (or XGBoost if available)
- **Output:** Learned synthesis probability and predictions
- **Evaluation:** Precision, Recall, ROC-AUC, PR-AUC, Brier score

### Granger Causality
- **Test:** Do Outliers/Entropy at lag 1-2 predict IV?
- **Purpose:** Identify predictive lags for forecasting
- **Output:** Correlation and p-values for each lag

## Input Files

- `iv_metrics_temporal.csv` - Temporal features from Step 3
- `iv_metrics_dynamic.csv` - System dynamics from Step 4 (optional, for additional features)

## Output Files

### `iv_metrics_calibrated.csv`
- **What it is:** IV metrics with learned synthesis predictions
- **New columns:**
  - `synthesis`: Binary target (1 = posts, 0 = no posts)
  - `synthesis_prob_learned`: Learned model probability (0-1)
  - `synthesis_pred_learned`: Learned model prediction (0 or 1)
- **How to interpret:**
  - `synthesis_prob_learned` = calibrated probability (more accurate than original forecast)
  - Compare with original `synthesis_probability` to see improvement

### `calibration_report.txt`
- **What it is:** Detailed calibration metrics
- **Contains:**
  - Original forecast evaluation (precision, recall, ROC-AUC, PR-AUC, Brier)
  - Learned model performance
  - Granger causality test results
- **How to interpret:**
  - Higher ROC-AUC / PR-AUC = better discrimination
  - Lower Brier score = better calibration (more accurate probabilities)
  - Significant Granger tests = predictive lag effects

### `calibration_curves.png`
- **What it is:** ROC and Precision-Recall curves
- **Panels:**
  1. ROC curves: True Positive Rate vs False Positive Rate
  2. Precision-Recall curves: Precision vs Recall
- **How to interpret:**
  - Curves above diagonal = better than random
  - Higher AUC = better model
  - Compare forecast vs learned to see improvement

## Key Metrics Explained

### Precision
- **Formula:** `TP / (TP + FP)`
- **Interpretation:** Of days predicted as synthesis, what fraction actually had posts?
- **Good value:** > 0.7

### Recall
- **Formula:** `TP / (TP + FN)`
- **Interpretation:** Of days with posts, what fraction were correctly predicted?
- **Good value:** > 0.7

### ROC-AUC
- **Interpretation:** Ability to distinguish synthesis from non-synthesis days
- **Range:** 0-1 (higher is better)
- **Good value:** > 0.8

### PR-AUC
- **Interpretation:** Precision-recall balance (better for imbalanced data)
- **Range:** 0-1 (higher is better)
- **Good value:** > 0.7

### Brier Score
- **Formula:** `mean((predicted_prob - actual)^2)`
- **Interpretation:** Calibration quality (lower is better)
- **Range:** 0-1
- **Good value:** < 0.2

### Granger Causality
- **Tests:** Do Outliers/Entropy at lag 1-2 predict IV changes?
- **Significant (p < 0.05):** Lag is predictive, keep it for forecasting
- **Non-significant:** Lag not predictive, can ignore

## Feature Importance

The learned model shows which features matter most:

- **IV Momentum** (0.53): Strongest predictor
- **Integration** (0.61): High importance
- **Compression** (0.58): High importance
- **Exploration Pressure** (0.20): Moderate importance
- **Breadth** (0.29): Moderate importance
- **Novelty** (0.18): Lower importance

## Usage

```bash
python backtest_calibrate.py
```

## Next Steps

Calibrated predictions feed into:
- **Step 6:** THRML simulation (uses synthesis probabilities)
- **Step 7:** Blind spot analysis (identifies false positives)

