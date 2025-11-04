#!/usr/bin/env python3
"""
Insight Velocity Pipeline CLI

Runs the complete Insight Velocity analysis pipeline on a mathlete data room.

Usage:
    python run_iv_pipeline.py <data_room_path> [--date-range START END] [--output OUTPUT_DIR]
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime as dt, timedelta
import pandas as pd
import sqlite3

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pipeline functions - handle imports gracefully
try:
    from knn_pipeline import build_knn
except ImportError:
    from scripts.knn_pipeline import build_knn

try:
    from clustering_comparison import (
        community_detection_clustering,
        hdbscan_clustering,
        compare_clustering
    )
except ImportError:
    from scripts.clustering_comparison import (
        community_detection_clustering,
        hdbscan_clustering,
        compare_clustering
    )


def parse_tweet_json(json_path):
    """Parse a tweet JSON file from data room."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text
        text = data.get('text', data.get('full_text', ''))
        
        # Extract date
        created_at = data.get('created_at', '')
        date_str = None
        
        if created_at:
            try:
                # Parse Twitter date format: "Thu Mar 03 15:09:05 +0000 2022"
                date_obj = dt.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                date_str = date_obj.strftime('%Y-%m-%d')
            except:
                # Fallback: try ISO format
                try:
                    date_obj = dt.fromisoformat(created_at.replace('Z', '+00:00'))
                    date_str = date_obj.strftime('%Y-%m-%d')
                except:
                    pass
        
        # Fallback: extract date from file path (tweets/ideas/YYYY/MM/DD/...)
        if not date_str:
            parts = json_path.parts
            try:
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 4:  # Year
                        if i + 2 < len(parts) and parts[i+1].isdigit() and parts[i+2].isdigit():
                            year, month, day = part, parts[i+1], parts[i+2]
                            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            break
            except:
                pass
        
        return {
            'text': text,
            'date': date_str,
            'id': data.get('id_str', data.get('id', json_path.stem)),
            'data': data
        }
    except Exception as e:
        return None


def parse_post_json(json_path):
    """Parse a post JSON file from data room."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text (try different fields)
        title = data.get('title', '')
        plaintext = data.get('plaintext', '')
        html = data.get('html', '')
        text = data.get('text', '')
        
        # Combine title and content
        if plaintext:
            full_text = f"{title}\n\n{plaintext}".strip() if title else plaintext
        elif html:
            full_text = f"{title}\n\n{html[:500]}".strip() if title else html[:500]
        elif text:
            full_text = f"{title}\n\n{text}".strip() if title else text
        else:
            full_text = title
        
        # Extract date
        published_at = data.get('published_at', data.get('created_at', ''))
        date_str = None
        
        if published_at:
            try:
                if isinstance(published_at, (int, float)):
                    # Timestamp (possibly milliseconds)
                    ts = published_at / 1000 if published_at > 1e10 else published_at
                    date_obj = dt.fromtimestamp(ts)
                    date_str = date_obj.strftime('%Y-%m-%d')
                else:
                    # String date
                    date_obj = dt.fromisoformat(str(published_at).replace('Z', '+00:00'))
                    date_str = date_obj.strftime('%Y-%m-%d')
            except:
                pass
        
        # Fallback: try to extract from path
        if not date_str:
            parts = json_path.parts
            try:
                # Look for YYYY/MM/DD pattern in path
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 4:  # Year
                        if i + 2 < len(parts) and parts[i+1].isdigit() and parts[i+2].isdigit():
                            year, month, day = part, parts[i+1], parts[i+2]
                            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            break
            except:
                pass
        
        return {
            'text': full_text,
            'date': date_str,
            'title': title,
            'id': data.get('id', data.get('slug', json_path.stem)),
            'data': data
        }
    except Exception as e:
        return None


def parse_ai_archive_json(json_path):
    """Parse an AI archive JSON file from data room."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text (required field)
        text = data.get('text', '')
        
        # If no text field, reconstruct from messages array
        if not text and 'messages' in data:
            text_parts = []
            for msg in data['messages']:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if role == 'human':
                    text_parts.append(f"Human: {content}")
                elif role == 'assistant':
                    text_parts.append(f"Assistant: {content}")
            text = '\n\n'.join(text_parts)
        
        if not text:
            return None
        
        # Extract date
        date_str = data.get('date', '')
        if not date_str:
            # Try to parse from timestamp
            created_at = data.get('created_at') or data.get('create_time')
            if created_at:
                try:
                    if isinstance(created_at, (int, float)):
                        ts = created_at / 1000 if created_at > 1e10 else created_at
                        date_obj = dt.fromtimestamp(ts)
                        date_str = date_obj.strftime('%Y-%m-%d')
                    else:
                        date_obj = dt.fromisoformat(str(created_at).replace('Z', '+00:00'))
                        date_str = date_obj.strftime('%Y-%m-%d')
                except:
                    pass
        
        # Fallback: extract from file path
        if not date_str:
            parts = json_path.parts
            try:
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 4:  # Year
                        if i + 2 < len(parts) and parts[i+1].isdigit() and parts[i+2].isdigit():
                            year, month, day = part, parts[i+1], parts[i+2]
                            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            break
            except:
                pass
        
        source = data.get('source', 'custom')
        
        return {
            'text': text,
            'date': date_str,
            'id': data.get('id', json_path.stem),
            'source': source,
            'data': data
        }
    except Exception as e:
        return None


def parse_data_room(data_room_path, date_range=None):
    """
    Parse mathlete data room structure and extract items.
    
    Structure can be:
    - tweets/ideas/YYYY/MM/DD/*.json
    - tweets/conversations/YYYY/MM/DD/*.json
    - ai_archives/*/YYYY/MM/DD/*.json
    - writing/YYYY/MM/DD/*.json
    """
    data_room = Path(data_room_path)
    
    if not data_room.exists():
        raise ValueError(f"Data room not found: {data_room_path}")
    
    items = []
    posts = []
    ai_archives = []
    
    print(f"\n📥 Parsing data room: {data_room}")
    
    # Parse tweets from ideas/ and conversations/
    tweets_base = data_room / "tweets"
    if tweets_base.exists():
        print("  Extracting tweets...")
        
        for tweet_type in ['ideas', 'conversations']:
            tweets_dir = tweets_base / tweet_type
            if not tweets_dir.exists():
                continue
            
            # Find all JSON files recursively
            tweet_files = list(tweets_dir.rglob("*.json"))
            print(f"    Found {len(tweet_files)} {tweet_type} tweets")
            
            for json_file in tweet_files:
                result = parse_tweet_json(json_file)
                if result and result['text'] and result['date']:
                    items.append({
                        'id': f"tweet_{result['id']}",
                        'text': result['text'],
                        'date': result['date'],
                        'type': 'tweet',
                        'category': tweet_type
                    })
    
    # Parse AI archives
    ai_base = data_room / "ai_archives"
    if ai_base.exists():
        print("  Extracting AI archives...")
        
        # Find all JSON files recursively in ai_archives
        ai_files = list(ai_base.rglob("*.json"))
        print(f"    Found {len(ai_files)} AI archive files")
        
        for json_file in ai_files:
            result = parse_ai_archive_json(json_file)
            if result and result['text'] and result['date']:
                ai_archives.append({
                    'id': f"ai_{result['source']}_{result['id']}",
                    'text': result['text'],
                    'date': result['date'],
                    'type': 'ai_archive',
                    'source': result.get('source', 'custom')
                })
    
    # Parse writing (posts)
    writing_dir = data_room / "writing"
    if writing_dir.exists():
        print("  Extracting posts...")
        
        # Find all JSON files recursively
        post_files = list(writing_dir.rglob("*.json"))
        print(f"    Found {len(post_files)} post files")
        
        for json_file in post_files:
            result = parse_post_json(json_file)
            if result and result['text'] and result['date']:
                posts.append({
                    'id': f"post_{result['id']}",
                    'text': result['text'],
                    'date': result['date'],
                    'type': 'blog_post',
                    'title': result.get('title', '')
                })
    
    print(f"\n  Total items: {len(items)}")
    print(f"  Total AI archives: {len(ai_archives)}")
    print(f"  Total posts: {len(posts)}")
    
    # Filter by date range if specified
    if date_range:
        start_date, end_date = date_range
        items_before_count = len(items)
        ai_before_count = len(ai_archives)
        posts_before_count = len(posts)
        
        items = [item for item in items if item.get('date') and start_date <= item['date'] <= end_date]
        ai_archives = [ai for ai in ai_archives if ai.get('date') and start_date <= ai['date'] <= end_date]
        posts = [post for post in posts if post.get('date') and start_date <= post['date'] <= end_date]
        
        print(f"\n  Filtered to date range {start_date} to {end_date}:")
        print(f"    Items: {items_before_count} → {len(items)}")
        print(f"    AI Archives: {ai_before_count} → {len(ai_archives)}")
        print(f"    Posts: {posts_before_count} → {len(posts)}")
    
    return items, posts, ai_archives


def run_pipeline_steps(output_dir, items_df, posts_df, k=5, min_sim=0.12):
    """
    Run the complete Insight Velocity pipeline steps.
    """
    output_dir = Path(output_dir)
    work_dir = output_dir / "working"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Step 1: k-NN graph
    print("\n🔗 Step 1: Building k-NN Graph")
    print("-" * 80)
    
    items_csv = work_dir / "items.csv"
    items_df.to_csv(items_csv, index=False)
    print(f"  Saved {len(items_df)} items to {items_csv}")
    
    nodes_csv = work_dir / "knn_nodes.csv"
    edges_csv = work_dir / "knn_edges.csv"
    
    build_knn(str(items_csv), str(edges_csv), str(nodes_csv), k=k, min_sim=min_sim)
    results['knn'] = {'nodes': nodes_csv, 'edges': edges_csv, 'items': items_csv}
    
    # Step 2: Clustering
    print("\n🧠 Step 2: Clustering Comparison")
    print("-" * 80)
    
    comm_csv = work_dir / "community_labels.csv"
    hdbscan_csv = work_dir / "hdbscan_labels.csv"
    metrics_json = work_dir / "clustering_metrics.json"
    
    comm_results = community_detection_clustering(
        str(nodes_csv), str(edges_csv), str(comm_csv)
    )
    hdbscan_results = hdbscan_clustering(str(items_csv), str(hdbscan_csv))
    comparison = compare_clustering(comm_results, hdbscan_results)
    
    with open(metrics_json, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    results['clustering'] = {
        'community': comm_csv,
        'hdbscan': hdbscan_csv,
        'metrics': metrics_json
    }
    
    # For now, we'll create a simplified report
    # Full pipeline would continue with Steps 3-7
    
    return results


def organize_outputs(output_dir, results):
    """Organize outputs into step directories."""
    output_dir = Path(output_dir)
    
    print("\n📁 Organizing Outputs")
    print("-" * 80)
    
    step_dirs = {
        'step_01_knn_graph': [
            ('knn_nodes.csv', results['knn']['nodes']),
            ('knn_edges.csv', results['knn']['edges']),
            ('items.csv', results['knn']['items'])
        ],
        'step_02_clustering': [
            ('community_labels.csv', results['clustering']['community']),
            ('hdbscan_labels.csv', results['clustering']['hdbscan']),
            ('clustering_metrics.json', results['clustering']['metrics'])
        ]
    }
    
    for step_dir, files in step_dirs.items():
        target_dir = output_dir / step_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, source_file in files:
            if source_file.exists():
                import shutil
                shutil.copy(source_file, target_dir / filename)
                print(f"  Copied {filename} to {step_dir}/")
    
    # Copy READMEs from main project
    for step_dir in step_dirs.keys():
        source_readme = Path(__file__).parent.parent / step_dir / "README.md"
        target_readme = output_dir / step_dir / "README.md"
        if source_readme.exists():
            import shutil
            shutil.copy(source_readme, target_readme)
            print(f"  Copied README.md to {step_dir}/")


def create_summary_report(output_dir, items_df, posts_df, results):
    """Create summary report."""
    output_dir = Path(output_dir)
    
    summary = {
        'pipeline_run_date': dt.now().isoformat(),
        'statistics': {
            'total_items': len(items_df),
            'total_posts': len(posts_df),
            'date_range': {
                'start': items_df['date'].min() if 'date' in items_df.columns else None,
                'end': items_df['date'].max() if 'date' in items_df.columns else None
            }
        },
        'clustering_results': None,
        'output_directory': str(output_dir)
    }
    
    # Load clustering metrics if available
    metrics_file = results['clustering']['metrics']
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            summary['clustering_results'] = json.load(f)
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Saved summary to {summary_file}")
    
    return summary


def run_full_pipeline(data_room_path, output_dir, date_range=None, k=5, min_sim=0.12):
    """
    Run the complete Insight Velocity pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 Insight Velocity Pipeline")
    print("=" * 80)
    print(f"Data Room: {data_room_path}")
    print(f"Output: {output_dir}")
    if date_range:
        print(f"Date Range: {date_range[0]} to {date_range[1]}")
    print()
    
    # Parse data room
    items, posts, ai_archives = parse_data_room(data_room_path, date_range)
    
    if len(items) == 0 and len(ai_archives) == 0:
        print("⚠️  No items found. Exiting.")
        return
    
    # Create DataFrames
    items_df = pd.DataFrame(items)
    posts_df = pd.DataFrame(posts) if posts else pd.DataFrame()
    ai_archives_df = pd.DataFrame(ai_archives) if ai_archives else pd.DataFrame()
    
    # Run pipeline steps
    results = run_pipeline_steps(output_dir, items_df, posts_df, k=k, min_sim=min_sim)
    
    # Organize outputs
    organize_outputs(output_dir, results)
    
    # Create summary
    summary = create_summary_report(output_dir, items_df, posts_df, results)
    
    print("\n" + "=" * 80)
    print("✅ Pipeline Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nSee {output_dir}/summary.json for full report")
    print(f"\nStep outputs:")
    print(f"  - step_01_knn_graph/ - k-NN graph results")
    print(f"  - step_02_clustering/ - Clustering comparison")


def main():
    parser = argparse.ArgumentParser(
        description='Run Insight Velocity pipeline on a mathlete data room',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on entire data room
  python run_iv_pipeline.py ~/mathlete-data-room

  # Run on specific date range
  python run_iv_pipeline.py ~/mathlete-data-room --date-range 2025-10-01 2025-10-31

  # Specify output directory
  python run_iv_pipeline.py ~/mathlete-data-room --output ./results/2025-10

Output Structure:
  <mathlete>/iv_reports/<datetime>/
  ├── step_01_knn_graph/
  ├── step_02_clustering/
  ├── working/
  └── summary.json
        """
    )
    
    parser.add_argument(
        'data_room',
        type=str,
        help='Path to mathlete data room directory'
    )
    
    parser.add_argument(
        '--date-range',
        nargs=2,
        metavar=('START', 'END'),
        help='Date range to analyze (YYYY-MM-DD YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: <data_room>/iv_reports/<datetime>)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='k for k-NN graph (default: 5)'
    )
    
    parser.add_argument(
        '--min-sim',
        type=float,
        default=0.12,
        help='Minimum similarity threshold (default: 0.12)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        data_room_path = Path(args.data_room)
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        output_dir = data_room_path / "iv_reports" / timestamp
    
    # Parse date range
    date_range = None
    if args.date_range:
        date_range = (args.date_range[0], args.date_range[1])
    
    try:
        run_full_pipeline(
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
