# Step 4: Dynamic System Modeling

## What This Step Does

Models Insight Velocity as a dynamical system using physics-based analogies:

- **Oscillator Model:** Treats IV as a damped driven oscillator
- **Thermodynamics:** Maps entropy to temperature, IV to energy
- **Phase Transitions:** Identifies eruption → synthesis transitions

**Why it matters:** This provides a physical framework for understanding creative cycles, allowing you to predict system behavior and optimize interventions.

## How It Works

### Oscillator Model
The system is modeled as: `m*x'' + c*x' + k*x = F(t)`

**Variable Mapping:**
- `x = IV` (position in insight space)
- `x' = ΔIV` (velocity / creative acceleration)
- `F(t) = Exploration Pressure` (driving force)
- `c = (1 - Integration Potential)` (damping)
- `k = (1 - Compression Readiness)` (elasticity)

**Intuition:**
- High exploration pressure = large driving force
- High integration = low damping = sustained oscillation
- High compression readiness = low elasticity = quick stabilization

### Thermodynamics
- **Temperature:** Entropy (higher entropy = higher temperature)
- **Energy:** IV (higher IV = higher energy)
- **Phase Transitions:**
  - **Eruption:** High temperature + high energy (exploration phase)
  - **Synthesis:** Low temperature + stable energy (consolidation phase)

## Input Files

- `iv_metrics_temporal.csv` - Temporal features from Step 3

## Output Files

### `iv_metrics_dynamic.csv`
- **What it is:** Enhanced IV metrics with system dynamics
- **New columns:**
  - `iv_simulated`: Simulated IV from oscillator model
  - `delta_iv_simulated`: Simulated acceleration
  - `energy_efficiency`: IV gain per unit Exploration Pressure
  - `rolling_efficiency`: 3-day rolling energy efficiency
- **How to interpret:**
  - `energy_efficiency` = how well exploration pressure converts to IV gain
  - Higher efficiency = better conversion (more IV per unit exploration)

### `dynamic_system_report.txt`
- **What it is:** System analysis with phase transitions
- **Contains:**
  - Model summary and parameters
  - Simulation accuracy metrics
  - Phase transition analysis (eruption → synthesis)
  - Energy efficiency analysis

### `insight_thermodynamics.png`
- **What it is:** Phase plot showing Entropy vs Energy (IV)
- **Axes:**
  - X-axis: Entropy (Temperature β⁻¹)
  - Y-axis: IV (Energy)
  - Color: IV Momentum
- **How to interpret:**
  - **High temperature zone (right):** High entropy = exploration phase
  - **Low temperature zone (left):** Low entropy = synthesis phase
  - **Green stars:** Synthesis days (posts created)
  - **Red triangles:** Eruption days
  - Movement through phase space shows creative cycle progression

### `oscillator_simulation.png`
- **What it is:** Comparison of actual vs simulated IV
- **Panels:**
  1. IV comparison (actual vs simulated)
  2. ΔIV comparison (actual vs simulated acceleration)
- **How to interpret:**
  - Good fit = model captures system dynamics
  - Large gaps = model needs refinement
  - Trend alignment = model captures direction if not magnitude

## Key Metrics

### Energy Efficiency
- **Formula:** `ΔIV / Exploration Pressure`
- **Interpretation:**
  - Positive = efficient conversion of exploration to IV gain
  - Negative = deceleration despite exploration pressure
  - High values = productive exploration days

### Phase Transitions

**Eruption Phase:**
- High temperature (entropy > 0.95)
- High energy (IV > 0.45)
- High driving force (exploration pressure > 0.3)
- Low damping (1 - integration < 0.5)
- **Example:** Oct 29

**Synthesis Phase:**
- Lower temperature (entropy 0.85-0.95)
- Stable energy (IV stable or increasing)
- Lower driving force (exploration pressure < 0.25)
- Higher damping (stabilization)
- **Example:** Oct 30-31

## Usage

```bash
python dynamic_system_modeling.py
```

## Next Steps

System dynamics feed into:
- **Step 6:** THRML simulation (uses temperature and field parameters)
- **Step 7:** Blind spot analysis (identifies metastable states)

