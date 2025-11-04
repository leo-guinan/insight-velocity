#!/usr/bin/env python3
"""
Two-Pole Adversarial Pipeline: Public vs Private Influence Analysis

Runs the complete two-pole analysis including:
- Building dual-pole graphs with HDBSCAN
- Cluster alignment via optimal transport
- Directional influence metrics
- Advanced temporal analysis (lead-lag, survival, Hawkes)
- Time-aware alignment and blind spot detection
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime as dt
import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import two-pole modeling
import importlib.util
two_pole_spec = importlib.util.spec_from_file_location(
    "two_pole_modeling",
    Path(__file__).parent / "two_pole_modeling.py"
)
two_pole = importlib.util.module_from_spec(two_pole_spec)
two_pole_spec.loader.exec_module(two_pole)

# Import advanced temporal analysis
temporal_spec = importlib.util.spec_from_file_location(
    "two_pole_temporal_advanced",
    Path(__file__).parent / "two_pole_temporal_advanced.py"
)
temporal_advanced = importlib.util.module_from_spec(temporal_spec)
temporal_spec.loader.exec_module(temporal_advanced)

# Import data room parser
import run_iv_pipeline as iv_pipeline


def run_two_pole_analysis(data_room_path, output_dir, date_range=None, min_cluster_size=2):
    """
    Run complete two-pole analysis pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "working"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🔬 Two-Pole Adversarial Analysis")
    print("=" * 80)
    print(f"Data Room: {data_room_path}")
    print(f"Output: {output_dir}")
    if date_range:
        print(f"Date Range: {date_range[0]} to {date_range[1]}")
    print()
    
    # Parse data room
    print("\n" + "=" * 80)
    print("📥 Step 1: Parsing Data Room")
    print("=" * 80)
    
    items, posts, ai_archives = iv_pipeline.parse_data_room(data_room_path, date_range)
    
    if len(items) == 0:
        print("⚠️  No public tweets found. Need public items to create a pole.")
        return
    
    if len(ai_archives) == 0:
        print("⚠️  No AI archives found. Need private items to create a pole.")
        print("   See DATA_ROOM_101_AI_ARCHIVES.md for how to add AI archives.")
        return
    
    print(f"\n✓ Loaded {len(items)} public items (Twitter)")
    print(f"✓ Loaded {len(ai_archives)} private items (AI archives)")
    
    # Prepare items for two-pole model
    public_items = [
        {'id': item['id'], 'text': item['text'], 'date': item['date']}
        for item in items
    ]
    
    private_items = [
        {'id': ai['id'], 'text': ai['text'], 'date': ai['date']}
        for ai in ai_archives
    ]
    
    # Build two-pole architecture
    print("\n" + "=" * 80)
    print("🔨 Step 2: Building Two-Pole Architecture")
    print("=" * 80)
    
    # Warn if min_cluster_size is very small with large datasets
    if min_cluster_size < 3 and (len(public_items) > 1000 or len(private_items) > 100):
        print(f"\n   ⚠️  Warning: min_cluster_size={min_cluster_size} with large datasets may create")
        print(f"   too many small clusters. Consider increasing to 5-10 for better alignment.")
    
    builder = two_pole.TwoPoleBuilder(min_cluster_size=min_cluster_size)
    pole_results = builder.build_pole_graphs(public_items, private_items)
    
    # Report cluster counts
    pub_n_clusters = pole_results['public']['clusters']['n_clusters']
    priv_n_clusters = pole_results['private']['clusters']['n_clusters']
    print(f"\n   📊 Cluster Summary:")
    print(f"      Public: {pub_n_clusters} clusters from {len(public_items)} items")
    print(f"      Private: {priv_n_clusters} clusters from {len(private_items)} items")
    
    # Warn if too many clusters
    if pub_n_clusters > 100 or priv_n_clusters > 100:
        print(f"\n   ⚠️  Warning: Very high cluster count detected!")
        print(f"      This may cause alignment issues (mass distributed too thinly).")
        print(f"      Recommendation: Increase --min-cluster-size (try {max(5, min_cluster_size * 2)})")
    
    # Align clusters
    print("\n" + "=" * 80)
    print("🔗 Step 3: Aligning Clusters")
    print("=" * 80)
    
    aligner = two_pole.ClusterAligner()
    alignment = aligner.align_clusters(
        pole_results['public']['centroids'],
        pole_results['private']['centroids'],
        pole_results['public']['activations'],
        pole_results['private']['activations']
    )
    
    # Compute directional influence (basic)
    print("\n" + "=" * 80)
    print("📊 Step 4: Computing Directional Influence")
    print("=" * 80)
    
    if not alignment:
        print("   ⚠️  No alignment results available. Skipping directional influence calculation.")
        influence_df = pd.DataFrame()
    elif not alignment.get('alignments'):
        print(f"   ⚠️  Alignment has no alignments (empty list). Skipping directional influence calculation.")
        print(f"   This could mean:")
        print(f"     - No clusters were aligned (all similarity < 0.01 threshold)")
        print(f"     - Try adjusting cluster sensitivity or date range")
        influence_df = pd.DataFrame()
    else:
        n_alignments = len(alignment['alignments'])
        print(f"   Processing {n_alignments} alignments...")
        influence_calculator = two_pole.DirectionalInfluence()
        influence_df = influence_calculator.compute_influence_metrics(
            alignment['alignments'],
            pole_results['public']['activations'],
            pole_results['private']['activations']
        )
    
    # Advanced temporal analysis
    print("\n" + "=" * 80)
    print("⏱️  Step 5: Advanced Temporal Analysis")
    print("=" * 80)
    
    temporal_analyzer = temporal_advanced.AdvancedTemporalAnalyzer(
        similarity_threshold=0.25,
        max_lag=14,
        rho_min=0.2
    )
    
    # Build lead-lag map with enhanced cross-correlation
    lead_lag_map = pd.DataFrame()
    if (pole_results['public'].get('centroids') and 
        pole_results['private'].get('centroids') and
        len(pole_results['public']['centroids']) > 0 and
        len(pole_results['private']['centroids']) > 0):
        lead_lag_map = temporal_analyzer.compute_lead_lag_map(
            pole_results['public']['centroids'],
            pole_results['private']['centroids'],
            pole_results['public']['activations'],
            pole_results['private']['activations']
        )
    
    # Per-topic learning lag and ILC
    topic_ilc = pd.DataFrame()
    if not lead_lag_map.empty:
        topic_ilc = temporal_analyzer.compute_per_topic_learning_lag(lead_lag_map)
    
    # Time-aware alignment
    time_aware_results = None
    if (pole_results['public'].get('centroids') and 
        pole_results['private'].get('centroids') and
        len(pole_results['public']['centroids']) > 0 and
        len(pole_results['private']['centroids']) > 0):
        time_aware_results = temporal_analyzer.time_aware_alignment(
            pole_results['public']['centroids'],
            pole_results['private']['centroids'],
            pole_results['public']['activations'],
            pole_results['private']['activations'],
            gamma=0.5
        )
    
    # Survival analysis
    survival_results = None
    if not lead_lag_map.empty:
        survival_results = temporal_analyzer.survival_analysis(
            pole_results['private']['activations'],
            pole_results['public']['activations'],
            lead_lag_map
        )
    
    # Hawkes point process
    hawkes_results = pd.DataFrame()
    if not lead_lag_map.empty:
        hawkes_results = temporal_analyzer.hawkes_point_process(
            pole_results['private']['activations'],
            pole_results['public']['activations'],
            lead_lag_map
        )
    
    # Time-aware blind spot detection
    blind_spots = pd.DataFrame()
    if not lead_lag_map.empty and not topic_ilc.empty:
        blind_spots = temporal_analyzer.detect_time_aware_blind_spots(
            pole_results['private']['activations'],
            pole_results['public']['activations'],
            lead_lag_map,
            topic_ilc
        )
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("📊 Step 6: Generating Visualizations")
    print("=" * 80)
    
    visualizer = temporal_advanced.TemporalVisualizer()
    
    if not lead_lag_map.empty:
        visualizer.plot_lag_surface(
            lead_lag_map,
            top_n=20,
            save_path=output_dir / 'lag_surface.png'
        )
        print("✓ Saved lag_surface.png")
        
        visualizer.plot_latency_histogram(
            lead_lag_map,
            save_path=output_dir / 'latency_histogram.png'
        )
        print("✓ Saved latency_histogram.png")
        
        # Plot lead-lag braid for top pair
        if len(lead_lag_map) > 0:
            top_pair = lead_lag_map.iloc[0]
            visualizer.plot_lead_lag_braid(
                pole_results['private']['activations'],
                pole_results['public']['activations'],
                top_pair['private_cluster'],
                top_pair['public_cluster'],
                top_pair['optimal_lag'],
                save_path=output_dir / 'lead_lag_braid.png'
            )
            print("✓ Saved lead_lag_braid.png")
    
    if survival_results:
        visualizer.plot_survival_curves(
            survival_results,
            save_path=output_dir / 'survival_curves.png'
        )
        print("✓ Saved survival_curves.png")
    
    # Save outputs
    print("\n" + "=" * 80)
    print("💾 Saving Outputs")
    print("=" * 80)
    
    # Save cluster labels
    pd.DataFrame({
        'id': [i['id'] for i in public_items],
        'label': pole_results['public']['clusters']['labels'],
        'probability': pole_results['public']['clusters'].get('probabilities', [None] * len(public_items))
    }).to_csv(output_dir / 'public_clusters.csv', index=False)
    print("✓ Saved public_clusters.csv")
    
    pd.DataFrame({
        'id': [i['id'] for i in private_items],
        'label': pole_results['private']['clusters']['labels'],
        'probability': pole_results['private']['clusters'].get('probabilities', [None] * len(private_items))
    }).to_csv(output_dir / 'private_clusters.csv', index=False)
    print("✓ Saved private_clusters.csv")
    
    # Save influence map (basic)
    if not influence_df.empty:
        influence_df.to_csv(output_dir / 'influence_map.csv', index=False)
        print("✓ Saved influence_map.csv")
    
    # Save advanced temporal results
    if not lead_lag_map.empty:
        lead_lag_map.to_csv(output_dir / 'lead_lag_map.csv', index=False)
        print("✓ Saved lead_lag_map.csv")
    
    if not topic_ilc.empty:
        topic_ilc.to_csv(output_dir / 'topic_ilc.csv', index=False)
        print("✓ Saved topic_ilc.csv")
    
    if not hawkes_results.empty:
        hawkes_results.to_csv(output_dir / 'hawkes_results.csv', index=False)
        print("✓ Saved hawkes_results.csv")
    
    if not blind_spots.empty:
        blind_spots.to_csv(output_dir / 'time_aware_blind_spots.csv', index=False)
        print("✓ Saved time_aware_blind_spots.csv")
    
    if survival_results:
        survival_results['survival_data'].to_csv(output_dir / 'survival_data.csv', index=False)
        print("✓ Saved survival_data.csv")
    
    if time_aware_results:
        # Save time-aware alignment matrices
        np.savez(
            output_dir / 'time_aware_alignment.npz',
            combined=time_aware_results['combined_matrix'],
            semantic=time_aware_results['semantic_matrix'],
            temporal=time_aware_results['temporal_matrix']
        )
        print("✓ Saved time_aware_alignment.npz")
    
    # Save time series activations
    pole_results['public']['activations'].to_csv(output_dir / 'public_activations.csv')
    pole_results['private']['activations'].to_csv(output_dir / 'private_activations.csv')
    print("✓ Saved public_activations.csv")
    print("✓ Saved private_activations.csv")
    
    # Save graphs
    if pole_results['public']['graph']['edges']:
        pd.DataFrame(pole_results['public']['graph']['edges']).to_csv(
            output_dir / 'public_graph_edges.csv', index=False
        )
    
    if pole_results['private']['graph']['edges']:
        pd.DataFrame(pole_results['private']['graph']['edges']).to_csv(
            output_dir / 'private_graph_edges.csv', index=False
        )
    print("✓ Saved graph edges")
    
    # Save cluster summaries
    summaries = {
        'public_clusters': {
            str(k): {'size': v} 
            for k, v in pole_results['public']['clusters']['cluster_counts'].items()
        },
        'private_clusters': {
            str(k): {'size': v}
            for k, v in pole_results['private']['clusters']['cluster_counts'].items()
        }
    }
    
    with open(output_dir / 'cluster_summaries.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    print("✓ Saved cluster_summaries.json")
    
    # Create comprehensive summary
    summary = {
        'pipeline_run_date': dt.now().isoformat(),
        'data_room_path': str(data_room_path),
        'output_directory': str(output_dir),
        'date_range': date_range,
        'parameters': {
            'min_cluster_size': min_cluster_size
        },
        'public_pole': {
            'n_items': len(public_items),
            'n_clusters': pole_results['public']['clusters']['n_clusters'],
            'n_outliers': pole_results['public']['clusters']['n_outliers'],
            'graph_edges': pole_results['public']['graph']['n_edges']
        },
        'private_pole': {
            'n_items': len(private_items),
            'n_clusters': pole_results['private']['clusters']['n_clusters'],
            'n_outliers': pole_results['private']['clusters']['n_outliers'],
            'graph_edges': pole_results['private']['graph']['n_edges']
        },
        'alignment': {
            'alignment_strength': alignment['alignment_strength'] if alignment else None,
            'n_alignments': len(alignment['alignments']) if alignment else 0,
            'unmapped_public': len(alignment['unmapped_public']) if alignment else 0,
            'unmapped_private': len(alignment['unmapped_private']) if alignment else 0
        } if alignment else None,
        'influence': {
            'n_pairs': len(influence_df),
            'private_dominant': int((influence_df['net_influence'] > 0).sum()) if not influence_df.empty else 0,
            'public_dominant': int((influence_df['net_influence'] < 0).sum()) if not influence_df.empty else 0,
            'strong_influence': int((influence_df['net_influence'].abs() > 0.1).sum()) if not influence_df.empty else 0
        } if not influence_df.empty else None
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("✓ Saved summary.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ Two-Pole Analysis Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\n📊 Summary:")
    print(f"  Public Pole: {summary['public_pole']['n_clusters']} clusters, {summary['public_pole']['n_outliers']} outliers")
    print(f"  Private Pole: {summary['private_pole']['n_clusters']} clusters, {summary['private_pole']['n_outliers']} outliers")
    if summary['alignment']:
        print(f"  Alignment: {summary['alignment']['n_alignments']} strong alignments")
        print(f"  Alignment Strength: {summary['alignment']['alignment_strength']:.4f}")
    if summary['influence']:
        print(f"  Private → Public: {summary['influence']['private_dominant']} pairs")
        print(f"  Public → Private: {summary['influence']['public_dominant']} pairs")
        print(f"  Strong Influence: {summary['influence']['strong_influence']} pairs")
    
    print("\n📁 Output Files:")
    print("  - public_clusters.csv: Public pole cluster assignments")
    print("  - private_clusters.csv: Private pole cluster assignments")
    print("  - influence_map.csv: Directional influence between aligned pairs")
    print("  - public_activations.csv: Time series of public cluster activity")
    print("  - private_activations.csv: Time series of private cluster activity")
    print("  - summary.json: Complete analysis summary")
    
    if not influence_df.empty:
        print("\n🔝 Top Influential Alignments:")
        top_influence = influence_df.head(5)
        for _, row in top_influence.iterrows():
            direction = "Private→Public" if row['net_influence'] > 0 else "Public→Private"
            print(f"  • Cluster {int(row['private_cluster'])} → {int(row['public_cluster'])}: "
                  f"{direction}, strength={row['net_influence']:.4f}")
    
    if summary.get('temporal_analysis'):
        print("\n⏱️  Temporal Analysis Summary:")
        ta = summary['temporal_analysis']
        print(f"  Lead-Lag Pairs: {ta.get('lead_lag_pairs', 0)}")
        print(f"  Private Leads: {ta.get('private_leads', 0)} pairs")
        print(f"  Mean Optimal Lag: {ta.get('mean_optimal_lag', 0):.2f} days")
        if 'topic_ilc' in ta:
            print(f"  Mean Learning Lag: {ta['topic_ilc'].get('mean_learning_lag', 0):.2f} days")
            print(f"  Mean ILC (penalized): {ta['topic_ilc'].get('mean_ilc_penalized', 0):.4f}")
        if 'survival' in ta and ta['survival'].get('median_time_to_externalization'):
            print(f"  Median Time-to-Externalization: {ta['survival']['median_time_to_externalization']:.1f} days")
        if 'blind_spots' in ta:
            print(f"  Time-Aware Blind Spots: {ta['blind_spots'].get('n_detected', 0)}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Two-Pole Adversarial Analysis: Public vs Private Influence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on entire data room
  python run_two_pole_pipeline.py ~/mathlete-data-room

  # Run on specific date range
  python run_two_pole_pipeline.py ~/mathlete-data-room --date-range 2025-10-01 2025-10-31

  # Custom output directory
  python run_two_pole_pipeline.py ~/mathlete-data-room --output ./two_pole_results

  # Adjust clustering sensitivity
  python run_two_pole_pipeline.py ~/mathlete-data-room --min-cluster-size 5

Data Room Structure Required:
  - tweets/ideas/ (Public pole)
  - ai_archives/ (Private pole)
  
See DATA_ROOM_101_AI_ARCHIVES.md for AI archive format.
        """
    )
    
    parser.add_argument('data_room', type=str, 
                       help='Path to mathlete data room directory')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='Date range to analyze (YYYY-MM-DD YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: <data_room>/two_pole/<datetime>)')
    parser.add_argument('--min-cluster-size', type=int, default=2,
                       help='HDBSCAN min_cluster_size (default: 2)')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        data_room_path = Path(args.data_room)
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        output_dir = data_room_path / "two_pole" / timestamp
    
    # Parse date range
    date_range = None
    if args.date_range:
        date_range = (args.date_range[0], args.date_range[1])
    
    try:
        run_two_pole_analysis(
            args.data_room,
            output_dir,
            date_range=date_range,
            min_cluster_size=args.min_cluster_size
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

