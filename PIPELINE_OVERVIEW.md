# Insight Velocity Pipeline Overview

## System Flow

```
Raw Data
  ↓
Step 1: k-NN Graph (semantic similarity)
  ↓
Step 2: Clustering (community detection + HDBSCAN)
  ↓
Step 3: IV Metrics (Breadth, Novelty, Integration, Compression, ER)
  ↓
Step 4: Temporal Modeling (momentum, acceleration, synthesis forecast)
  ↓
Step 5: Dynamic System (oscillator model, thermodynamics)
  ↓
Step 6: Calibration (model evaluation, learned weights)
  ↓
Step 7: THRML Simulation (idea domain formation)
  ↓
Step 8: Blind Spot Analysis (missed predictions, suppression index)
```

## Step-by-Step Guide

### Step 1: k-NN Idea Graph
**Purpose:** Build semantic similarity network  
**Input:** `items.csv`  
**Output:** `knn_nodes.csv`, `knn_edges.csv`  
**Directory:** `step_01_knn_graph/`  
**See:** `step_01_knn_graph/README.md`

### Step 2: Clustering Comparison
**Purpose:** Compare graph-based vs embedding-based clustering  
**Input:** `knn_nodes.csv`, `knn_edges.csv`, `items.csv`  
**Output:** `community_labels.csv`, `hdbscan_labels.csv`, `clustering_metrics.json`  
**Directory:** `step_02_clustering/`  
**See:** `step_02_clustering/README.md`

### Step 3: Temporal Modeling
**Purpose:** Calculate derivatives, momentum, and synthesis forecasts  
**Input:** IV metrics from Step 2  
**Output:** `iv_metrics_temporal.csv`, momentum plots  
**Directory:** `step_03_temporal/`  
**See:** `step_03_temporal/README.md`

### Step 4: Dynamic System Modeling
**Purpose:** Model IV as oscillator and thermodynamics system  
**Input:** `iv_metrics_temporal.csv`  
**Output:** `iv_metrics_dynamic.csv`, thermodynamics plots  
**Directory:** `step_04_dynamic/`  
**See:** `step_04_dynamic/README.md`

### Step 5: Calibration
**Purpose:** Evaluate and optimize synthesis predictions  
**Input:** `iv_metrics_temporal.csv` or `iv_metrics_dynamic.csv`  
**Output:** `iv_metrics_calibrated.csv`, calibration curves  
**Directory:** `step_05_calibration/`  
**See:** `step_05_calibration/README.md`

### Step 6: THRML Simulation
**Purpose:** Simulate idea domain formation  
**Input:** Daily k-NN graphs, IV metrics  
**Output:** `thrml_simulation_results.json`, `sim_reality_merged.csv`  
**Directory:** `step_06_simulation/`  
**See:** `step_06_simulation/README.md`

### Step 7: Blind Spot Analysis
**Purpose:** Identify and correct missed synthesis predictions  
**Input:** `iv_metrics_calibrated.csv`, simulation results  
**Output:** `iv_metrics_blind_spots.csv`, blind spot visualizations  
**Directory:** `step_07_blind_spots/`  
**See:** `step_07_blind_spots/README.md`

## Quick Reference

### Key Metrics

- **IV (Insight Velocity):** Overall measure of creative output rate
- **Breadth:** Diversity of idea exploration (avg entropy)
- **Novelty:** Fraction of novel/experimental ideas (outlier %)
- **Integration:** How well ideas link together
- **Compression:** Readiness to synthesize
- **Entropy Reduction:** Quality of synthesis (H_in - H_out)
- **ΔIV:** Daily acceleration (IV_t - IV_(t-1))
- **Exploration Pressure:** Novelty × Entropy (early warning signal)

### Key Patterns

1. **Eruption → Synthesis:** High outliers → posts 1-2 days later
2. **Sweet Spot Zone:** Entropy 0.85-0.95, Outliers 10-25%
3. **Energy Efficiency:** IV gain per unit exploration pressure
4. **Blind Spots:** Predicted synthesis but no posts occurred

### File Locations

- **Scripts:** `scripts/`
- **Step outputs:** `step_XX_*/`
- **Daily analysis:** `weekly_analysis/[date]/`
- **Reports:** `step_XX_*/README.md` for each step

## Next Steps

1. Read `README.md` for system overview
2. Read each step's `README.md` for detailed explanations
3. Run the pipeline sequentially
4. Review outputs in each step directory
5. Use visualizations to understand patterns
6. Refine models based on blind spot analysis

