# Step 7: Blind Spot Analysis

## What This Step Does

Identifies "blind spots"—days where synthesis was predicted but didn't occur:

1. **Blind Spot Detection:** Finds days where `P_t > threshold` but `S_t = 0`
2. **Blind Spot Scoring:** Quantifies intensity of missed predictions
3. **Suppression Index:** Calculates systemic suppression of synthesis
4. **Cause Diagnosis:** Identifies why synthesis didn't occur
5. **Model Update:** Adjusts predictions using suppression index

**Why it matters:** Helps the system "learn where it's deaf"—identifying patterns where predictions fail and adjusting accordingly. This enables continuous improvement of synthesis forecasts.

## How It Works

### Blind Spot Definition
- **Condition:** `P_t > θ_P` but `S_t = 0`
  - Where `P_t` = predicted synthesis probability
  - `S_t` = actual synthesis (1 = posts, 0 = none)
  - `θ_P` = threshold (default 0.5)

### Blind Spot Score
```
B_t = P_t × (1 - S_t) × f(EP_t, IV_t, ER_t)
```
Where `f(EP, IV, ER) = 0.5(EP + IV) + ER`

**Interpretation:** Higher score = stronger signal for synthesis, but no output occurred.

### Suppression Index
```
SI_t = B_t / P_t
```
**Interpretation:** How much synthesis was suppressed relative to prediction.

### Model Update
```
P_t' = P_t × (1 - SI_t)
```
**Purpose:** Penalize recurrent blind zones by reducing future predictions.

## Input Files

- `iv_metrics_calibrated.csv` - Calibrated predictions from Step 5
- `sim_reality_merged.csv` - Simulation results from Step 6 (optional, for domain_frac, label_entropy)

## Output Files

### `iv_metrics_blind_spots.csv`
- **What it is:** Enhanced metrics with blind spot analysis
- **New columns:**
  - `is_blind_spot`: Boolean (true if predicted but no synthesis)
  - `blind_spot_score`: Intensity score (0-1, higher = stronger blind spot)
  - `suppression_index`: Suppression index (0-1, higher = more suppressed)
  - `synthesis_prob_adjusted`: Adjusted probability after suppression penalty
  - `blind_spot_cluster`: Cluster ID (if clustering was performed)
- **How to interpret:**
  - `blind_spot_score` > 0 = predicted synthesis but none occurred
  - `suppression_index` > 0 = systemic suppression detected
  - `synthesis_prob_adjusted` = more conservative prediction

### `blind_spot_report.txt`
- **What it is:** Detailed blind spot analysis
- **Contains:**
  - Summary statistics (blind spot count, rate)
  - Metrics (precision, recall, drift)
  - Diagnoses (probable causes and interventions)
  - Cluster analysis (if applicable)
- **How to interpret:**
  - Blind spot rate = % of false positives
  - High drift = timing issues (correct phase, wrong timing)
  - Low drift = overconfidence (wrong predictions)

### `blind_spots_predicted_vs_actual.png`
- **What it is:** Predicted vs actual synthesis plot
- **Axes:**
  - X-axis: Date
  - Y-axis: Predicted synthesis probability
- **Points:**
  - Green = synthesis occurred (posts created)
  - Red = no synthesis (predicted but no posts)
  - Orange X = blind spots (high probability, no posts)
- **How to interpret:**
  - Red dots in high-probability region = blind spots
  - Green dots in high-probability region = correct predictions
  - Horizontal line at 0.5 = threshold

### `blind_spots_heatmap.png` (if blind spots exist)
- **What it is:** Blind spot heatmap
- **Axes:**
  - X-axis: Entropy (Temperature)
  - Y-axis: Novelty (Outlier Fraction)
  - Color: Blind Spot Score
- **How to interpret:**
  - Shows which thermodynamic zones most often fail to collapse into synthesis
  - Red regions = frequent blind spots
  - Green zone = sweet spot (should have few blind spots)

### `phase_drift_chart.png`
- **What it is:** Expected vs actual synthesis timeline
- **Shows:**
  - Expected synthesis (1-2 days after eruptions)
  - Actual synthesis (days with posts)
  - Blind spots (high probability, no posts)
- **How to interpret:**
  - Alignment = good timing predictions
  - Drift = temporal misalignment
  - Blind spots = missed opportunities

## Key Patterns & Diagnoses

### Pattern Matching

**1. Overload (High Entropy + High Novelty, No Synthesis)**
- **Cause:** Not enough damping/integration time
- **Intervention:** Reduce input entropy next cycle

**2. Overconstraint (Low Entropy + High IV, No Synthesis)**
- **Cause:** Ideas too cohesive, no fresh input
- **Intervention:** Increase exploration

**3. Domains Formed But Not Collapsed (High EP + High Domain Fraction, No Synthesis)**
- **Cause:** Idea domains formed but not collapsed
- **Intervention:** Add forcing function (publishing trigger, collaborator feedback)

**4. Hidden Synthesis (High Prob + High Label Entropy, No Posts)**
- **Cause:** Insights internalized but not externalized
- **Intervention:** Check for note-taking or unseen outputs

**5. Low Integration Potential**
- **Cause:** Low integration (ideas not linking well)
- **Intervention:** Increase integration (link ideas more tightly)

## Metrics Explained

### Blind Spot Rate
- **Formula:** `#false positives / #predicted synthesis`
- **Interpretation:** % of predicted synthesis days that didn't produce posts
- **Good value:** < 0.2 (20%)

### Predictive Drift
- **Formula:** `avg(Δt between predicted & actual synthesis)`
- **Interpretation:** Temporal misalignment (days)
- **Low drift:** Overconfidence (wrong predictions)
- **High drift:** Timing issues (correct phase, wrong timing)

### Suppression Index Distribution
- **Average:** Overall suppression level
- **Max:** Strongest suppression instance
- **High values:** Recurrent blind zones that need attention

## Probabilistic Interpretation (THRML Frame)

In the THRML energy-based model frame:

- **Blind spots = metastable domains**
- **Energy landscape suggests transition, but Gibbs sampling doesn't converge**
- **Fix by:**
  - Lowering β (increase temperature → exploration)
  - Increasing coupling γ (integration) locally
  - Testing if higher β or lower γ would cause collapse (latent insight potential)

## Usage

```bash
python blind_spot_analysis.py
```

## Next Steps

Blind spot analysis feeds back into:
- **Model refinement:** Update synthesis forecast model with suppression index
- **Intervention planning:** Target specific blind spot causes
- **System optimization:** Reduce blind spot rate over time

