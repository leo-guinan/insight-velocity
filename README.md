# Insight Velocity Analysis System

A comprehensive pipeline for analyzing idea flow, creativity cycles, and insight generation from daily data (tweets, posts, notes, AI conversations).

## Overview

This system transforms raw idea items into quantitative metrics that track:
- **Breadth:** Diversity of idea exploration
- **Novelty:** Fraction of novel/experimental ideas
- **Integration:** How well ideas link together
- **Compression:** Readiness to synthesize
- **Entropy Reduction:** Quality of synthesis

And uses these to predict **Insight Velocity (IV)**—the rate at which exploration converts to synthesis (outputs like blog posts).

## Quick Start

### 1. Import Your Data

#### Import AI Archives

```bash
# Import OpenAI/ChatGPT conversations
python scripts/import_openai_archive.py openai-export.zip ~/mathlete-data-room

# Import Anthropic Claude conversations
python scripts/import_anthropic_archive.py claude-export.zip ~/mathlete-data-room
```

**Getting exports:**
- **OpenAI**: Settings → Data Controls → Export Data
- **Anthropic**: Contact support for export (JSON format)

#### Data Room Structure

Your data room should follow this structure:

```
mathlete-data-room/
├── tweets/
│   ├── ideas/
│   │   └── YYYY/MM/DD/*.json
│   └── conversations/
│       └── YYYY/MM/DD/*.json
├── writing/
│   └── YYYY/MM/DD/*.json
├── ai_archives/          # Private pole (gitignored)
│   ├── claude/
│   │   └── YYYY/MM/DD/*.json
│   └── gpt/ (or openai/)
│       └── YYYY/MM/DD/*.json
└── videos/ (optional)
    └── YYYY/MM/DD/*.json
```

**AI Archive Format:**
Each JSON file should contain:
```json
{
  "id": "unique_id",
  "text": "Human: question\n\nAssistant: answer",
  "date": "2025-10-27",
  "source": "claude",
  "messages": [
    {
      "role": "human",
      "content": "Question text",
      "timestamp": "2025-10-27T14:30:00Z"
    },
    {
      "role": "assistant",
      "content": "Answer text",
      "timestamp": "2025-10-27T14:30:15Z"
    }
  ]
}
```

### 2. Run Analysis

#### Standard IV Pipeline

```bash
# Run full pipeline (Steps 1-2)
python scripts/run_full_iv_pipeline.py ~/mathlete-data-room \
  --date-range 2025-10-01 2025-10-31

# Run individual steps
python scripts/knn_pipeline.py --items items.csv
python scripts/clustering_comparison.py --nodes knn_nodes.csv
python scripts/calculate_iv_metrics.py
python scripts/temporal_modeling.py
python scripts/dynamic_system_modeling.py
python scripts/backtest_calibrate.py
python scripts/thrml_prototype.py
python scripts/blind_spot_analysis.py
```

#### Two-Pole Adversarial Model

Analyze influence between your public (Twitter) and private (AI conversations) creative poles:

```bash
# Basic run
python scripts/run_two_pole_pipeline.py ~/mathlete-data-room \
  --date-range 2025-10-01 2025-10-31

# Custom output location
python scripts/run_two_pole_pipeline.py ~/mathlete-data-room \
  --output ./results/two_pole_october

# Adjust cluster sensitivity
python scripts/run_two_pole_pipeline.py ~/mathlete-data-room \
  --min-cluster-size 10
```

#### Blind Spot Analysis

Find ideas that should have become public but didn't:

```bash
# Run after two-pole analysis
python scripts/analyze_blind_spots.py two_pole/YYYYMMDD_HHMMSS/ --top 10
```

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

### Output Structure

```
insight_velocity/
├── step_01_knn_graph/      # k-NN similarity results
├── step_02_clustering/     # Cluster assignments
├── step_03_temporal/       # Momentum & forecasting
├── step_04_dynamic/        # Oscillator & thermodynamics
├── step_05_calibration/    # Model evaluation
├── step_06_simulation/     # THRML simulation
├── step_07_blind_spots/    # Missed predictions
└── two_pole/               # Two-pole analysis results
```

Each step directory contains:
- Data files (CSV/JSON)
- Visualizations (PNG plots)
- `README.md` with interpretation guide

## Two-Pole Model Details

### What It Measures

- **Private → Public Flow**: Ideas explored privately that become public synthesis
- **Public → Private Flow**: Public discussions that trigger deeper private exploration
- **Alignment Strength**: How well private and public themes map (0-1)
- **Unmapped Clusters**: Ideas in one pole but not the other (blind spots)
- **Optimal Lag**: Days between private idea and public emergence
- **Directional Influence**: Which pole leads (net_influence ±)

### Advanced Temporal Analysis

The two-pole model includes:

1. **Lead-Lag Map**: Optimal time lag (τ*) and Directional Influence Index (DII) per cluster pair
2. **Learning Lag**: Per-topic weighted median lag and Inverse Learning Curve (ILC)
3. **Survival Analysis**: Time-to-externalization using Kaplan-Meier and Cox models
4. **Hawkes Point Process**: Causal strength (α) and learning speed (β)
5. **Time-Aware Alignment**: Combines semantic similarity × cross-correlation
6. **Temporal Blind Spots**: Detects where synthesis should have occurred but didn't

### Key Metrics

**Cluster Alignment:**
- `alignment_strength`: 0-1, how well poles match (high = coherent)
- `n_alignments`: Number of strong cluster pairs
- `unmapped_public/private`: Clusters with no counterpart

**Directional Influence:**
- `net_influence`: Positive = Private→Public, Negative = Public→Private
- `optimal_lag`: Days between private activation and public synthesis
- `correlation_at_lag`: Strength at optimal lag
- `te_priv_to_pub`: Transfer entropy from private to public
- `te_pub_to_priv`: Transfer entropy from public to private

**Blind Spot Types:**
- **Type A (Reactivation Gap)**: Historical public matches, no recent 2025 posts
- **Type B (Burst→No-Collapse)**: Multiple private files, predicted synthesis, no public output
- **Type C (Channel Mismatch)**: Research-dense private content, mapped to lightweight public buckets

### Reading Results

**Strong Private → Public:**
```
Cluster 3 → Cluster 7: Private→Public, strength=0.247, lag=2 days
```
Meaning: Ideas explored privately in cluster 3 lead to public synthesis in cluster 7 within ~2 days.

**Blind Spot Example:**
```
Date: 2025-06-17, Private Cluster: 1
Type: Reactivation Gap
Lag: N/A (no recent public match)
Max Similarity: 0.71 to Public Cluster 74
Recommendation: Publish updated take + link to historical tweets
```

## Key Insights & Patterns

### The Exploration → Synthesis Cycle

The system identifies a clear pattern:
1. **Exploration Phase:** High entropy, high novelty, high exploration pressure
2. **Eruption:** Peaks in novelty (outliers > 30%)
3. **Synthesis:** 1-2 days later, outputs appear (blog posts)

**Example:**
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

## CLI Reference

### Standard Pipeline

```bash
python scripts/run_full_iv_pipeline.py <data_room> [options]

Options:
  --date-range START END    Filter to date range (YYYY-MM-DD)
  --output OUTPUT_DIR       Custom output directory
  --k K                     k for k-NN graph (default: 5)
  --min-sim MIN_SIM         Minimum similarity (default: 0.12)
```

### Two-Pole Pipeline

```bash
python scripts/run_two_pole_pipeline.py <data_room> [options]

Options:
  --date-range START END    Filter to date range
  --output OUTPUT_DIR       Custom output directory
  --min-cluster-size N      HDBSCAN min cluster size (default: 10)
  --similarity-threshold T  Alignment threshold (default: 0.25)
```

### Blind Spot Analysis

```bash
python scripts/analyze_blind_spots.py <two_pole_output_dir> [options]

Options:
  --top N                   Show top N blind spots (default: 5)
```

## Core Metrics Explained

### Standard IV Metrics

- **IV (Insight Velocity)**: Overall creative output rate
- **Breadth**: Diversity of exploration (avg entropy)
- **Novelty**: Fraction of experimental ideas (outlier %)
- **Integration**: How well ideas link together
- **Compression**: Readiness to synthesize
- **Entropy Reduction**: Quality indicator (H_in - H_out)
- **ΔIV**: Daily acceleration
- **Exploration Pressure**: Novelty × Entropy (early warning signal)

### Two-Pole Metrics

- **Alignment Strength**: Semantic coherence between poles
- **Net Influence**: Direction of idea flow
- **Optimal Lag**: Time between private incubation and public synthesis
- **Inverse Learning Curve (ILC)**: Speed of externalization (higher = faster)
- **Transfer Entropy**: Causal influence strength

## Daily Workflow

1. **Run blind-spot detector** → Get top 5 candidates
2. **Review each blind spot** → See private content and aligned public clusters
3. **Auto-generate content** → Tweet hooks, thread outlines, blog skeletons
4. **Publish & tag** → Record publish date
5. **Watch metrics** → Monitor IV, ER uptick next 48-72h
6. **Learn** → Update time-to-externalization and ILC

## Privacy & Git

The `.gitignore` excludes private AI conversations:

```gitignore
# Private AI conversations (never commit)
ai_archives/
**/ai_archives/
```

**Verify nothing tracked:**
```bash
git status
git check-ignore ai_archives/
```

## Dependencies

### Required

```bash
pip install pandas numpy scikit-learn networkx matplotlib hdbscan
```

### Optional (recommended)

```bash
pip install pot              # Optimal transport (faster alignment)
pip install python-louvain   # Community detection
pip install umap-learn       # Dimensionality reduction
pip install lifelines        # Survival analysis
pip install xgboost          # Model calibration
```

### Optional (advanced)

```bash
pip install thrml jax        # THRML simulation
```

## Troubleshooting

### "No items found"
- Check data room structure matches expected format
- Verify date range includes data
- Check JSON files have required fields (`text`, `date`)

### "Weak alignment" (two-pole)
- Increase `--min-cluster-size` (try 15-20)
- Check you have enough items in both poles (min 20-30 each)
- Verify dates overlap between poles

### "AttributeError: 'TfidfVectorizer' object has no attribute 'vocabulary_'"
- This is fixed in latest version; update if you see this

### Import errors
- Ensure you're running from project root
- Install missing dependencies
- Check Python version (3.8+)

## Next Steps

1. **Extend time periods**: Run analysis for weeks/months
2. **Cross-participant analysis**: Compare multiple people's IV patterns
3. **Intervention testing**: Use simulation to test what-if scenarios
4. **Dashboard creation**: Build live visualization dashboard
5. **Content automation**: Auto-generate and schedule posts from blind spots

## File Structure

```
insight_velocity/
├── README.md (this file)
├── .gitignore
├── scripts/
│   ├── run_full_iv_pipeline.py      # Standard pipeline CLI
│   ├── run_two_pole_pipeline.py     # Two-pole pipeline CLI
│   ├── analyze_blind_spots.py       # Blind spot analysis
│   ├── import_openai_archive.py     # OpenAI import
│   ├── import_anthropic_archive.py  # Anthropic import
│   └── [other analysis scripts]
├── step_01_knn_graph/
├── step_02_clustering/
├── step_03_temporal/
├── step_04_dynamic/
├── step_05_calibration/
├── step_06_simulation/
├── step_07_blind_spots/
└── two_pole/                         # Two-pole results
```

## References

Each step directory contains a detailed `README.md` explaining:
- What the step does
- How it works
- How to interpret outputs
- Key metrics and patterns

Start with `step_01_knn_graph/README.md` and work through sequentially.

