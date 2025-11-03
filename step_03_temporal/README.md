# Step 3: Temporal Modeling

## What This Step Does

Turns Insight Velocity (IV) into a predictive signal by modeling temporal dynamics:

1. **Derivatives & Momentum:** Calculates daily acceleration (ΔIV) and momentum trends
2. **Synthesis Forecasting:** Predicts when synthesis windows will occur
3. **Exploration Pressure:** Early warning signal for upcoming synthesis bursts

**Why it matters:** Understanding how IV changes over time helps predict when you're moving toward synthesis (creation of outputs like blog posts) versus exploration (generating new ideas).

## How It Works

### Derivatives & Momentum
- **ΔIV (Daily Acceleration):** `IV_t - IV_(t-1)`
  - Positive = accelerating creativity
  - Negative = decelerating
- **IV Momentum:** 3-day rolling mean of ΔIV
  - Trend of acceleration over time
- **Exploration Pressure:** `Novelty_t × Entropy_t`
  - High pressure = high novelty + high diversity
  - Early warning for synthesis bursts

### Synthesis Forecasting
- **Rules:**
  - Sweet spot: Entropy ∈ [0.85, 0.95] and Outliers ∈ [0.1, 0.25]
  - Eruption: Outliers > 0.30 → expect synthesis 1-2 days later
- **Output:** Next Synthesis Probability (0-1)

## Input Files

- `iv_metrics.csv` - IV metrics from previous steps
- `iv_metrics_temporal.csv` - (created by this step)

## Output Files

### `iv_metrics_temporal.csv`
- **What it is:** Enhanced IV metrics with temporal features
- **New columns:**
  - `delta_iv`: Daily acceleration (IV_t - IV_(t-1))
  - `iv_momentum`: 3-day rolling mean of ΔIV
  - `exploration_pressure`: Novelty × Entropy
  - `momentum_direction`: ↑ (accelerating), ↓ (decelerating), → (stable)
  - `synthesis_probability`: Predicted probability of synthesis
  - `in_sweet_spot`: Boolean (true if in optimal synthesis zone)
  - `eruption_detected`: Boolean (true if outliers > 0.30)
- **How to interpret:**
  - Positive `delta_iv` = IV is increasing (good momentum)
  - High `exploration_pressure` = high novelty + entropy (potential eruption)
  - `synthesis_probability` > 0.5 = likely synthesis window

### `temporal_modeling_report.txt`
- **What it is:** Detailed analysis of temporal patterns
- **Contains:**
  - Summary statistics (averages, trends)
  - Day-by-day momentum analysis
  - Pattern analysis (eruption → synthesis)
  - Synthesis forecasts

### `idea_momentum_plot.png`
- **What it is:** IV vs ΔIV quadrant analysis
- **How to interpret:**
  - **Quadrant 1 (high IV, positive ΔIV):** Accelerating with high IV = optimal
  - **Quadrant 2 (high IV, negative ΔIV):** Decelerating despite high IV = stabilizing
  - **Quadrant 3 (low IV, negative ΔIV):** Decelerating with low IV = concern
  - **Quadrant 4 (low IV, positive ΔIV):** Accelerating from low IV = recovery

### `momentum_timeline.png`
- **What it is:** 3-panel timeline showing:
  1. IV and Momentum trends
  2. Daily Acceleration & Exploration Pressure
  3. Synthesis Probability over time
- **How to interpret:**
  - Days with posts marked with stars
  - Eruption days marked with triangles
  - High synthesis probability should align with actual posts

## Key Patterns

### Eruption → Synthesis Pattern
- **Eruption day:** High outliers (>30%) → high exploration pressure
- **Synthesis days:** 1-2 days later, posts appear
- **Example:** Oct 29 eruption (36% outliers) → Oct 30-31 synthesis (6 posts)

### Sweet Spot Zone
- **Definition:** Entropy 0.85-0.95, Outliers 10-25%
- **Interpretation:** Optimal conditions for synthesis
- **Days in sweet spot:** Should correlate with synthesis days

### Momentum Trends
- **Positive momentum:** Building toward synthesis
- **Negative momentum:** Stabilizing or cooling off
- **Sustained acceleration:** Good sign of productive cycle

## Usage

```bash
python temporal_modeling.py
```

Reads from `iv_metrics.csv` (or equivalent) and generates temporal features.

## Next Steps

Temporal features feed into:
- **Step 4:** Dynamic system modeling (oscillator model)
- **Step 5:** Backtest & calibration (model evaluation)
- **Step 7:** Blind spot analysis (suppression index)

