# Step 6: THRML Simulation - Idea Domain Formation

## What This Step Does

Simulates idea domain formation using Energy-Based Models (EBMs):

1. **Maps k-NN graph to Potts model:** Nodes = categorical variables, edges = coupling weights
2. **Runs Gibbs sampling:** Simulates how clusters form and evolve
3. **Compares to reality:** Maps simulation outputs to actual metrics

**Why it matters:** Provides a controllable simulator of your idea system. You can test interventions (e.g., "what if novelty stays high?") and see how domains crystallize from exploration → synthesis.

## How It Works

### Energy Function
```
E(s) = -γ * Σ w_ij * 1[s_i = s_j] - Σ <h_i, e_s_i>
```

**Variable Mapping:**
- **Nodes:** Categorical variables (theme labels 0-K)
- **Edges (w_ij):** Cosine similarity from k-NN graph
- **Temperature (β):** `1 / Entropy` (higher entropy → lower β → higher temperature)
- **External field (h_i):** `α * ExplorationPressure` (stronger field = more "pull" into themes)
- **Coupling scale (γ):** `IntegrationPotential` (tighter coupling on synthesis days)

### Gibbs Sampling
- **Process:** Iteratively updates node states based on energy
- **Convergence:** Domains form as similar nodes cluster together
- **Output:** Final state (which items are in which theme)

## Input Files

- Daily analysis from `weekly_analysis/[date]/` directories
  - `knn_nodes.csv`
  - `knn_edges.csv`
- `iv_metrics_calibrated.csv` - For temperature and field parameters

## Output Files

### `thrml_simulation_results.json`
- **What it is:** Simulation outputs for each day
- **Structure:** Array of daily results
- **Each result contains:**
  - `date`: Date string
  - `domain_frac`: Largest domain fraction (0-1)
  - `label_entropy`: Entropy of theme labels (higher = more diverse)
  - `mixing_time`: Steps until convergence (or max steps if not converged)
  - `stable`: Whether convergence was achieved
  - `n_nodes`: Number of items simulated
  - `states`: Array of theme assignments
- **How to interpret:**
  - **Domain fraction:** Fraction of items in largest theme (higher = more consolidation)
  - **Label entropy:** Diversity of themes (higher = more diverse, lower = more focused)
  - **Mixing time:** How long until stable (lower = faster convergence)

### `sim_reality_merged.csv`
- **What it is:** Merged simulation and reality data
- **Columns:**
  - All columns from `iv_metrics_calibrated.csv`
  - `domain_frac`: Simulation output
  - `label_entropy`: Simulation output
  - `mixing_time`: Simulation output
- **How to interpret:** Compare simulation outputs (domain_frac, label_entropy) with real metrics (entropy_reduction, posts, IV, ΔIV)

### `simulation_reality_comparison.png`
- **What it is:** 4-panel comparison of simulation vs reality
- **Panels:**
  1. Domain Fraction vs Entropy Reduction
  2. Label Entropy vs IV
  3. Domain Fraction vs Posts (scatter)
  4. Label Entropy vs Entropy Reduction (scatter)
- **How to interpret:**
  - Strong correlations = simulation captures reality
  - R² values shown = fit quality
  - Alignment = simulation is predictive

## Key Patterns

### Domain Formation Pattern
- **Exploration days:** Higher domain fraction (domains forming)
- **Synthesis days:** Lower domain fraction (consolidation)
- **Example:** Oct 29 (eruption) = 0.32 domain frac → Oct 30-31 (synthesis) = 0.27, 0.22

### Label Entropy
- **High entropy:** More diverse themes (exploration)
- **Low entropy:** More focused themes (synthesis)
- **On synthesis days:** Actually increases (more diversity before consolidation)

### Simulation ↔ Reality Fits

**Strong Fits (R² > 0.7):**
- **ΔIV:** R² = 0.95 (excellent!)
  - Domain frac coefficient: +1.19
  - Label entropy coefficient: +1.23
- **IV:** R² = 0.77
  - Domain frac coefficient: +0.38
  - Label entropy coefficient: +0.75

**Interpretation:**
- Higher domain fraction + higher label entropy → higher ΔIV
- Simulation predicts acceleration well

**Moderate Fits (R² > 0.7):**
- **Entropy Reduction / Posts:** R² = 0.72
  - Negative coefficients = more domains → less synthesis (counterintuitive but observed)

## Physical Interpretation

### Domain Formation (Exploration → Synthesis)
- **High temperature (entropy):** Nodes explore freely, domains form
- **Cooling (synthesis):** Domains consolidate, label entropy decreases
- **Metastable states:** High domain fraction but not collapsed = blind spots

### Energy Landscape
- **Eruption phase:** High energy, high temperature, domains forming
- **Synthesis phase:** Lower energy, lower temperature, domains consolidating
- **Transition:** Driven by exploration pressure and integration

## Usage

```bash
python thrml_prototype.py
```

Runs simulation for each day in `weekly_analysis/` directories.

## Next Steps

Simulation results feed into:
- **Step 7:** Blind spot analysis (identifies metastable states)
- Model refinement (adjust temperature/field parameters)

