# Scripts Directory

This directory contains all Python scripts for the Insight Velocity analysis pipeline.

## Main Pipeline Scripts

### Data Extraction
- `extract_daily_tweets.py` - Extract tweets from SQLite database for specific dates
- `extract_ghost_posts.py` - Extract blog posts from Ghost export JSON

### Core Analysis
- `knn_pipeline.py` - Build k-NN graph from items (Step 1)
- `clustering_comparison.py` - Compare clustering methods (Step 2)
- `calculate_iv_metrics.py` - Calculate Insight Velocity components
- `temporal_modeling.py` - Temporal derivatives and momentum (Step 3)
- `dynamic_system_modeling.py` - Oscillator model and thermodynamics (Step 4)
- `backtest_calibrate.py` - Model calibration and evaluation (Step 5)
- `thrml_prototype.py` - THRML simulation (Step 6)
- `blind_spot_analysis.py` - Blind spot identification (Step 7)

### Batch Processing
- `batch_weekly_analysis.py` - Process multiple days automatically
- `view_weekly_report.py` - Generate human-readable weekly report
- `iv_summary_report.py` - Comprehensive IV analysis report
- `visualize_iv.py` - Create IV visualizations

## Usage

Run scripts from the project root directory:

```bash
# Example: Run from project root
cd /path/to/insight_velocity
python scripts/knn_pipeline.py --items items.csv
```

Or add scripts to your PATH, or run directly:

```bash
python -m scripts.knn_pipeline --items items.csv
```

## Dependencies

See main `README.md` for full dependency list. All scripts require:
- Python 3.8+
- pandas, numpy, scikit-learn
- networkx, matplotlib

Optional dependencies for specific scripts:
- `hdbscan`, `python-louvain` (clustering)
- `thrml`, `jax` (THRML simulation)
- `xgboost` (calibration, if available)

## Script Documentation

Each script has inline documentation. See the step directories for detailed explanations:
- `step_01_knn_graph/README.md` - k-NN graph scripts
- `step_02_clustering/README.md` - Clustering scripts
- etc.

