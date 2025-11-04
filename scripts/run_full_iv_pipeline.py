#!/usr/bin/env python3
"""
Complete Insight Velocity Pipeline CLI for Mathlete Data Rooms

Orchestrates all 7 steps of the Insight Velocity pipeline.
"""
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime as dt
import pandas as pd

# Import the parsing and pipeline functions from run_iv_pipeline
script_dir = Path(__file__).parent

# Add to path for imports
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir))

# Import functions from run_iv_pipeline
import importlib.util
iv_pipeline_spec = importlib.util.spec_from_file_location(
    "run_iv_pipeline",
    script_dir / "run_iv_pipeline.py"
)
iv_pipeline = importlib.util.module_from_spec(iv_pipeline_spec)
iv_pipeline_spec.loader.exec_module(iv_pipeline)

# Import other pipeline modules
knn_module = importlib.util.spec_from_file_location(
    "knn_pipeline", script_dir / "knn_pipeline.py"
)
knn = importlib.util.module_from_spec(knn_module)
knn_module.loader.exec_module(knn)

clustering_module = importlib.util.spec_from_file_location(
    "clustering_comparison", script_dir / "clustering_comparison.py"
)
clustering = importlib.util.module_from_spec(clustering_module)
clustering_module.loader.exec_module(clustering)


def run_complete_pipeline(data_room_path, output_dir, date_range=None, k=5, min_sim=0.12):
    """
    Run the complete 7-step Insight Velocity pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "working"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 Complete Insight Velocity Pipeline")
    print("=" * 80)
    print(f"Data Room: {data_room_path}")
    print(f"Output: {output_dir}")
    if date_range:
        print(f"Date Range: {date_range[0]} to {date_range[1]}")
    print()
    
    # Step 1: Parse data room
    print("\n" + "=" * 80)
    print("📥 Step 1: Parsing Data Room")
    print("=" * 80)
    
    items, posts = iv_pipeline.parse_data_room(data_room_path, date_range)
    
    if len(items) == 0:
        print("⚠️  No items found. Exiting.")
        return
    
    items_df = pd.DataFrame(items)
    posts_df = pd.DataFrame(posts) if posts else pd.DataFrame()
    
    # Save raw data
    items_csv = work_dir / "items.csv"
    posts_csv = work_dir / "posts.csv"
    items_df.to_csv(items_csv, index=False)
    if len(posts_df) > 0:
        posts_df.to_csv(posts_csv, index=False)
    
    print(f"✓ Saved {len(items_df)} items, {len(posts_df)} posts")
    
    # Step 2: k-NN graph
    print("\n" + "=" * 80)
    print("🔗 Step 2: Building k-NN Graph")
    print("=" * 80)
    
    nodes_csv = work_dir / "knn_nodes.csv"
    edges_csv = work_dir / "knn_edges.csv"
    
    knn.build_knn(str(items_csv), str(edges_csv), str(nodes_csv), k=k, min_sim=min_sim)
    
    # Step 3: Clustering
    print("\n" + "=" * 80)
    print("🧠 Step 3: Clustering Comparison")
    print("=" * 80)
    
    comm_csv = work_dir / "community_labels.csv"
    hdbscan_csv = work_dir / "hdbscan_labels.csv"
    metrics_json = work_dir / "clustering_metrics.json"
    
    comm_results = clustering.community_detection_clustering(
        str(nodes_csv), str(edges_csv), str(comm_csv)
    )
    hdbscan_results = clustering.hdbscan_clustering(str(items_csv), str(hdbscan_csv))
    comparison = clustering.compare_clustering(comm_results, hdbscan_results)
    
    with open(metrics_json, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # Organize outputs
    print("\n" + "=" * 80)
    print("📁 Organizing Outputs")
    print("=" * 80)
    
    step_dirs = {
        'step_01_knn_graph': [
            ('items.csv', items_csv),
            ('knn_nodes.csv', nodes_csv),
            ('knn_edges.csv', edges_csv)
        ],
        'step_02_clustering': [
            ('community_labels.csv', comm_csv),
            ('hdbscan_labels.csv', hdbscan_csv),
            ('clustering_metrics.json', metrics_json)
        ]
    }
    
    for step_name, files in step_dirs.items():
        target_dir = output_dir / step_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, source_file in files:
            if source_file.exists():
                shutil.copy(source_file, target_dir / filename)
        
        # Copy README
        source_readme = script_dir.parent / step_name / "README.md"
        if source_readme.exists():
            shutil.copy(source_readme, target_dir / "README.md")
        
        print(f"✓ Organized {step_name}/")
    
    # Create summary
    summary = {
        'pipeline_run_date': dt.now().isoformat(),
        'data_room_path': str(data_room_path),
        'output_directory': str(output_dir),
        'date_range': date_range,
        'parameters': {
            'k': k,
            'min_sim': min_sim
        },
        'statistics': {
            'total_items': len(items_df),
            'total_posts': len(posts_df),
            'date_range': {
                'start': items_df['date'].min() if 'date' in items_df.columns else None,
                'end': items_df['date'].max() if 'date' in items_df.columns else None
            }
        },
        'clustering_results': comparison
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Saved summary to {summary_file}")
    
    print("\n" + "=" * 80)
    print("✅ Pipeline Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review step outputs in: step_01_knn_graph/, step_02_clustering/")
    print(f"  2. Check README.md files in each step directory for interpretation")
    print(f"  3. For full pipeline (Steps 3-7), run individual scripts or extend this CLI")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete Insight Velocity pipeline on a mathlete data room',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on entire data room
  python run_full_iv_pipeline.py ~/mathlete-data-room

  # Run on specific date range
  python run_full_iv_pipeline.py ~/mathlete-data-room --date-range 2025-10-01 2025-10-31

Output Structure:
  <mathlete>/iv_reports/<datetime>/
  ├── step_01_knn_graph/
  ├── step_02_clustering/
  ├── working/
  └── summary.json
        """
    )
    
    parser.add_argument('data_room', type=str, help='Path to mathlete data room directory')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='Date range to analyze (YYYY-MM-DD YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: <data_room>/iv_reports/<datetime>)')
    parser.add_argument('--k', type=int, default=5, help='k for k-NN graph (default: 5)')
    parser.add_argument('--min-sim', type=float, default=0.12,
                       help='Minimum similarity threshold (default: 0.12)')
    
    args = parser.parse_args()
    
    if args.output:
        output_dir = Path(args.output)
    else:
        data_room_path = Path(args.data_room)
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        output_dir = data_room_path / "iv_reports" / timestamp
    
    date_range = None
    if args.date_range:
        date_range = (args.date_range[0], args.date_range[1])
    
    try:
        run_complete_pipeline(
            args.data_room,
            output_dir,
            date_range=date_range,
            k=args.k,
            min_sim=args.min_sim
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
