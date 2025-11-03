# Insight Velocity Analysis System

A comprehensive pipeline for analyzing idea flow, creativity cycles, and insight generation from daily data (tweets, posts, notes).

## Overview

This system transforms raw idea items into quantitative metrics that track:
- **Breadth:** Diversity of idea exploration
- **Novelty:** Fraction of novel/experimental ideas
- **Integration:** How well ideas link together
- **Compression:** Readiness to synthesize
- **Entropy Reduction:** Quality of synthesis

And uses these to predict **Insight Velocity (IV)**—the rate at which exploration converts to synthesis (outputs like blog posts).

## System Architecture

The pipeline consists of 7 sequential steps:

```
Step 1: k-NN Graph          → Semantic similarity network
Step 2: Clustering          → Idea communities (graph-based + embedding-based)
Step 3: Temporal Modeling  → Momentum, acceleration, synthesis forecasting
Step 4: Dynamic System      → Oscillator model, thermodynamics
Step 5: Calibration        → Model evaluation and optimization
Step 6: THRML Simulation   → Idea domain formation simulation
Step 7: Blind Spot Analysis → Identification and correction of missed predictions
```

## Quick Start

### 1. Extract Daily Data

```bash
# Extract tweets for a date range
python extract_daily_tweets.py

# Extract blog posts
python extract_ghost_posts.py
```

### 2. Run Full Pipeline

```bash
# Step 1: Build k-NN graph
python knn_pipeline.py --items items.csv --k 5 --min_sim 0.12

# Step 2: Compare clustering methods
python clustering_comparison.py --nodes knn_nodes.csv --edges knn_edges.csv --items items.csv

# Step 3: Calculate IV metrics
python calculate_iv_metrics.py

# Step 4: Temporal modeling
python temporal_modeling.py

# Step 5: Dynamic system modeling
python dynamic_system_modeling.py

# Step 6: Calibration
python backtest_calibrate.py

# Step 7: THRML simulation
python thrml_prototype.py

# Step 8: Blind spot analysis
python blind_spot_analysis.py
```

### 3. View Results

Each step has its own directory with outputs and a README explaining how to interpret them:
- `step_01_knn_graph/`
- `step_02_clustering/`
- `step_03_temporal/`
- `step_04_dynamic/`
- `step_05_calibration/`
- `step_06_simulation/`
- `step_07_blind_spots/`

## Key Insights

### The Exploration → Synthesis Cycle

The system identifies a clear pattern:
1. **Exploration Phase:** High entropy, high novelty, high exploration pressure
2. **Eruption:** Peaks in novelty (outliers > 30%)
3. **Synthesis:** 1-2 days later, outputs appear (blog posts)

**Example from data:**
- Oct 29: Eruption (36% outliers, high exploration pressure)
- Oct 30-31: Synthesis (6 blog posts total)

### Sweet Spot Zone

Optimal conditions for synthesis:
- **Entropy:** 0.85-0.95
- **Outliers:** 10-25%

Days in this zone have highest synthesis probability.

### Energy Efficiency

Measures how well exploration pressure converts to IV gain:
- **High efficiency:** Productive exploration days
- **Low/negative efficiency:** Deceleration despite exploration

## File Structure

```
insight_velocity/
├── README.md (this file)
├── steps/                    # Step documentation
├── step_01_knn_graph/        # k-NN graph outputs
├── step_02_clustering/       # Clustering results
├── step_03_temporal/         # Temporal modeling
├── step_04_dynamic/          # System dynamics
├── step_05_calibration/      # Model calibration
├── step_06_simulation/       # THRML simulation
├── step_07_blind_spots/      # Blind spot analysis
├── weekly_analysis/          # Daily analysis subdirectories
└── scripts/                  # All Python scripts
```

## Outputs Summary

### Core Metrics
- `iv_metrics.csv`: Base IV metrics
- `iv_metrics_temporal.csv`: With temporal features
- `iv_metrics_dynamic.csv`: With system dynamics
- `iv_metrics_calibrated.csv`: With learned predictions
- `iv_metrics_blind_spots.csv`: With blind spot analysis

### Visualizations
- Phase plots (entropy vs novelty, entropy vs energy)
- Momentum plots (IV vs ΔIV)
- Timeline visualizations
- Calibration curves
- Simulation comparisons

### Reports
- Detailed analysis reports for each step
- Weekly summaries
- Calibration metrics

## Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn
- networkx (graph operations)
- matplotlib (visualizations)
- hdbscan (HDBSCAN clustering)
- python-louvain (Louvain communities)
- Optional: thrml, jax (for THRML simulation)

Install with:
```bash
pip install pandas numpy scikit-learn networkx matplotlib hdbscan python-louvain
```

## Next Steps

1. **Extend to longer time periods:** Run analysis for weeks/months
2. **Cross-participant analysis:** Compare multiple people's IV patterns
3. **Intervention testing:** Use simulation to test what-if scenarios
4. **Dashboard creation:** Build live visualization dashboard

## References

Each step directory contains a detailed README explaining:
- What the step does
- How it works
- How to interpret outputs
- Key metrics and patterns

Start with `step_01_knn_graph/README.md` and work through sequentially.

