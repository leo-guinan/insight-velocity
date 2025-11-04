# Scripts Directory

This directory contains all Python scripts for the Insight Velocity analysis pipeline.

## Main Pipeline Scripts

### CLI Entry Point
- **`run_full_iv_pipeline.py`** - **RECOMMENDED**: Complete pipeline for mathlete data rooms
  - Parses data room structure
  - Runs Steps 1-2 (k-NN graph + clustering)
  - Organizes outputs into step directories
  - Creates summary report

### Data Extraction
- `extract_daily_tweets.py` - Extract tweets from SQLite database for specific dates
- `extract_ghost_posts.py` - Extract blog posts from Ghost export JSON

### Core Analysis Scripts
- `knn_pipeline.py` - Build k-NN graph from items (Step 1)
- `clustering_comparison.py` - Compare clustering methods (Step 2)
- `calculate_iv_metrics.py` - Calculate Insight Velocity components
- `temporal_modeling.py` - Temporal derivatives and momentum (Step 3)
- `dynamic_system_modeling.py` - Oscillator model and thermodynamics (Step 4)
- `backtest_calibrate.py` - Model calibration and evaluation (Step 5)
- `thrml_prototype.py` - THRML simulation (Step 6)
- `blind_spot_analysis.py` - Blind spot identification (Step 7)

### Batch Processing & Reports
- `batch_weekly_analysis.py` - Process multiple days automatically
- `view_weekly_report.py` - Generate human-readable weekly report
- `iv_summary_report.py` - Comprehensive IV analysis report
- `visualize_iv.py` - Create IV visualizations

## Quick Start

### For Mathlete Data Rooms

```bash
# Run complete pipeline on a data room
python scripts/run_full_iv_pipeline.py ~/mathlete-data-room

# Run on specific date range
python scripts/run_full_iv_pipeline.py ~/mathlete-data-room \
  --date-range 2025-10-01 2025-10-31

# Specify output directory
python scripts/run_full_iv_pipeline.py ~/mathlete-data-room \
  --output ~/mathlete-data-room/iv_reports/my_analysis
```

### For Individual Steps

If you want to run steps individually after the initial pipeline:

```bash
# From project root
cd /path/to/insight_velocity

# Step 1: Build k-NN graph
python scripts/knn_pipeline.py --items items.csv --k 5 --min_sim 0.12

# Step 2: Clustering
python scripts/clustering_comparison.py \
  --nodes knn_nodes.csv \
  --edges knn_edges.csv \
  --items items.csv

# Step 3: Temporal modeling
python scripts/temporal_modeling.py

# Step 4: Dynamic system
python scripts/dynamic_system_modeling.py

# Step 5: Calibration
python scripts/backtest_calibrate.py

# Step 6: THRML simulation
python scripts/thrml_prototype.py

# Step 7: Blind spots
python scripts/blind_spot_analysis.py
```

## Output Structure

When run on a data room, the pipeline creates:

```
<data_room>/iv_reports/<datetime>/
├── step_01_knn_graph/
│   ├── README.md
│   ├── items.csv
│   ├── knn_nodes.csv
│   └── knn_edges.csv
├── step_02_clustering/
│   ├── README.md
│   ├── community_labels.csv
│   ├── hdbscan_labels.csv
│   └── clustering_metrics.json
├── working/
│   └── [intermediate files]
└── summary.json
```

## Dependencies

See main `README.md` for full dependency list. All scripts require:
- Python 3.8+
- pandas, numpy, scikit-learn
- networkx, matplotlib

Optional dependencies:
- `hdbscan`, `python-louvain` (clustering)
- `thrml`, `jax` (THRML simulation)
- `xgboost` (calibration, if available)

## Extending the Pipeline

To run Steps 3-7 after the initial pipeline:

1. Use the outputs from Steps 1-2 in the output directory
2. Change working directory to the output directory
3. Run subsequent steps, pointing to the correct input files
4. Outputs will be organized into corresponding step directories

## Troubleshooting

**Import errors:** Make sure you're running from the project root or have installed the package.

**Date parsing errors:** The pipeline extracts dates from JSON files and file paths. If dates aren't parsing correctly, check the data room structure matches expected format.

**Missing dependencies:** Install missing packages with `pip install <package-name>`
