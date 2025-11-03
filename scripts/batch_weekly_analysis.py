#!/usr/bin/env python3
"""
Batch process tweets day by day and for the full week.
Extracts tweets, builds k-NN graphs, and runs clustering comparison for each.
"""
import sys
import json
import sqlite3
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import subprocess


def extract_tweets(db_path, username, start_date, end_date, output_items_csv):
    """
    Extract tweets for a date range and create items.csv.
    
    Args:
        db_path: Path to SQLite database
        username: Username to filter (screen_name)
        start_date: Start date (datetime)
        end_date: End date (datetime, exclusive)
        output_items_csv: Path to output items CSV file
    """
    if not Path(db_path).exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        return None
    
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = '''
        SELECT tweet_id, full_text, created_at_datetime, raw_json
        FROM tweets 
        WHERE screen_name = ? 
          AND created_at_datetime >= ? 
          AND created_at_datetime < ?
        ORDER BY created_at_datetime ASC
    '''
    
    cursor.execute(query, (username, start_timestamp, end_timestamp))
    results = cursor.fetchall()
    conn.close()
    
    if len(results) == 0:
        return None
    
    items = []
    for row in results:
        tweet_id, full_text, created_at_datetime, raw_json = row
        tweet_date = datetime.fromtimestamp(created_at_datetime, tz=timezone.utc)
        
        items.append({
            'id': f"tweet_{tweet_id}",
            'text': full_text if full_text else "",
            'date': tweet_date.strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'tweet'
        })
    
    df = pd.DataFrame(items)
    df.to_csv(output_items_csv, index=False)
    
    return len(results)


def run_knn_pipeline(items_csv, edges_csv, nodes_csv, k=5, min_sim=0.12):
    """Run k-NN pipeline to create edges and nodes."""
    from knn_pipeline import build_knn
    try:
        build_knn(items_csv, edges_csv, nodes_csv, k=k, min_sim=min_sim)
        return True
    except Exception as e:
        print(f"Error running k-NN pipeline: {e}", file=sys.stderr)
        return False


def run_clustering_comparison(nodes_csv, edges_csv, items_csv, comm_out, hdbscan_out, metrics_out):
    """Run clustering comparison and return metrics."""
    from clustering_comparison import (
        community_detection_clustering,
        hdbscan_clustering,
        compare_clustering
    )
    
    try:
        comm_results = community_detection_clustering(nodes_csv, edges_csv, comm_out)
        hdbscan_results = hdbscan_clustering(items_csv, hdbscan_out)
        comparison = compare_clustering(comm_results, hdbscan_results)
        
        # Save metrics
        with open(metrics_out, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    except Exception as e:
        print(f"Error running clustering comparison: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def process_date_range(date_label, start_date, end_date, output_dir, db_path, username):
    """
    Process a date range: extract tweets, build k-NN graph, run clustering.
    
    Returns:
        dict with results or None if no tweets
    """
    print("\n" + "=" * 80)
    print(f"Processing: {date_label}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 80)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # File paths
    items_csv = Path(output_dir) / "items.csv"
    edges_csv = Path(output_dir) / "knn_edges.csv"
    nodes_csv = Path(output_dir) / "knn_nodes.csv"
    comm_csv = Path(output_dir) / "community_labels.csv"
    hdbscan_csv = Path(output_dir) / "hdbscan_labels.csv"
    metrics_json = Path(output_dir) / "clustering_metrics.json"
    
    # Extract tweets
    print(f"\n📥 Extracting tweets...")
    tweet_count = extract_tweets(db_path, username, start_date, end_date, str(items_csv))
    
    if tweet_count is None or tweet_count == 0:
        print(f"  ⚠️  No tweets found for {date_label}")
        return None
    
    print(f"  ✓ Extracted {tweet_count} tweets to {items_csv}")
    
    # Build k-NN graph
    print(f"\n🔗 Building k-NN graph...")
    if not run_knn_pipeline(str(items_csv), str(edges_csv), str(nodes_csv)):
        return None
    
    print(f"  ✓ Created graph: {nodes_csv.name}, {edges_csv.name}")
    
    # Run clustering comparison
    print(f"\n🧠 Running clustering comparison...")
    metrics = run_clustering_comparison(
        str(nodes_csv), str(edges_csv), str(items_csv),
        str(comm_csv), str(hdbscan_csv), str(metrics_json)
    )
    
    if metrics is None:
        return None
    
    print(f"  ✓ Clustering complete: {metrics_json.name}")
    
    # Add metadata
    result = {
        'date_label': date_label,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'tweet_count': tweet_count,
        'metrics': metrics
    }
    
    return result


def generate_summary_report(all_results, output_file):
    """Generate a summary report comparing all dates."""
    print("\n" + "=" * 80)
    print("📊 Generating Summary Report")
    print("=" * 80)
    
    report = {
        'overview': {
            'total_days': len([r for r in all_results if r is not None]),
            'dates_analyzed': [r['date_label'] for r in all_results if r is not None]
        },
        'daily_results': [],
        'weekly_summary': None
    }
    
    # Daily results
    for result in all_results:
        if result is None:
            continue
        
        daily_summary = {
            'date': result['date_label'],
            'tweet_count': result['tweet_count'],
            'community_detection': {
                'communities': result['metrics']['community_detection']['T_d'],
                'entropy': result['metrics']['community_detection']['H_in_comm'],
                'largest_community': max(result['metrics']['community_detection']['community_sizes'].values()) if result['metrics']['community_detection']['community_sizes'] else 0
            },
            'hdbscan': {
                'clusters': result['metrics']['hdbscan']['T_topics'],
                'entropy': result['metrics']['hdbscan']['H_in_hdb'],
                'outlier_fraction': result['metrics']['hdbscan']['outlier_fraction'],
                'outliers': result['metrics']['hdbscan']['noise_count']
            },
            'comparison': {
                'entropy_difference': result['metrics']['entropy_difference'],
                'cluster_count_difference': result['metrics']['cluster_count_difference']
            }
        }
        
        report['daily_results'].append(daily_summary)
    
    # Weekly summary (last result should be the full week)
    if all_results and all_results[-1] is not None:
        weekly = all_results[-1]
        report['weekly_summary'] = {
            'tweet_count': weekly['tweet_count'],
            'community_detection': {
                'communities': weekly['metrics']['community_detection']['T_d'],
                'entropy': weekly['metrics']['community_detection']['H_in_comm']
            },
            'hdbscan': {
                'clusters': weekly['metrics']['hdbscan']['T_topics'],
                'entropy': weekly['metrics']['hdbscan']['H_in_hdb'],
                'outlier_fraction': weekly['metrics']['hdbscan']['outlier_fraction']
            }
        }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ Saved summary report to: {output_file}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("📈 Daily Comparison Summary")
    print("=" * 80)
    print(f"\n{'Date':<12} {'Tweets':<8} {'Comm':<6} {'H_comm':<8} {'HDB':<6} {'H_hdb':<8} {'Outliers':<10}")
    print("-" * 80)
    
    for daily in report['daily_results']:
        print(f"{daily['date']:<12} {daily['tweet_count']:<8} {daily['community_detection']['communities']:<6} "
              f"{daily['community_detection']['entropy']:<8.4f} {daily['hdbscan']['clusters']:<6} "
              f"{daily['hdbscan']['entropy']:<8.4f} {daily['hdbscan']['outlier_fraction']:<10.2%}")
    
    if report['weekly_summary']:
        print("-" * 80)
        print(f"{'WEEK TOTAL':<12} {report['weekly_summary']['tweet_count']:<8} "
              f"{report['weekly_summary']['community_detection']['communities']:<6} "
              f"{report['weekly_summary']['community_detection']['entropy']:<8.4f} "
              f"{report['weekly_summary']['hdbscan']['clusters']:<6} "
              f"{report['weekly_summary']['hdbscan']['entropy']:<8.4f} "
              f"{report['weekly_summary']['hdbscan']['outlier_fraction']:<10.2%}")
    
    return report


def main():
    # Configuration
    db_file = "tweets.db"
    username = "leo_guinan"
    start_week = datetime(2025, 10, 24, tzinfo=timezone.utc)
    end_week = datetime(2025, 11, 1, tzinfo=timezone.utc)  # Nov 1 exclusive
    
    base_output_dir = Path("weekly_analysis")
    base_output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("📅 Weekly Tweet Analysis: Day-by-Day + Full Week")
    print("=" * 80)
    print(f"Date range: {start_week.strftime('%Y-%m-%d')} to {end_week.strftime('%Y-%m-%d')}")
    print(f"Username: @{username}")
    print()
    
    all_results = []
    
    # Process each day from Oct 24 to Oct 31
    current_date = start_week
    while current_date < end_week:
        next_date = current_date + timedelta(days=1)
        date_label = current_date.strftime('%Y-%m-%d')
        day_dir = base_output_dir / date_label
        
        result = process_date_range(
            date_label, current_date, next_date, 
            str(day_dir), db_file, username
        )
        
        if result:
            all_results.append(result)
        else:
            print(f"  ⏭️  Skipping {date_label} (no tweets)")
        
        current_date = next_date
    
    # Process full week
    print("\n" + "=" * 80)
    print("Processing: FULL WEEK (Oct 24-31)")
    print("=" * 80)
    
    week_dir = base_output_dir / "full_week"
    week_result = process_date_range(
        "full_week", start_week, end_week,
        str(week_dir), db_file, username
    )
    
    if week_result:
        all_results.append(week_result)
    
    # Generate summary report
    if all_results:
        summary_file = base_output_dir / "weekly_summary.json"
        generate_summary_report(all_results, str(summary_file))
        
        print("\n" + "=" * 80)
        print("✅ Weekly Analysis Complete!")
        print("=" * 80)
        print(f"\nResults saved to: {base_output_dir}/")
        print(f"  - Daily analysis in subdirectories (YYYY-MM-DD/)")
        print(f"  - Full week analysis in: full_week/")
        print(f"  - Summary report: weekly_summary.json")
    else:
        print("\n⚠️  No results to summarize (no tweets found)")


if __name__ == "__main__":
    main()

