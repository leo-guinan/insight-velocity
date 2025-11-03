#!/usr/bin/env python3
"""
Generate a human-readable report from weekly analysis results.
"""
import json
from pathlib import Path


def print_daily_report(daily):
    """Print a formatted daily report."""
    print(f"\n📅 {daily['date']}")
    print(f"   Tweets: {daily['tweet_count']}")
    print(f"\n   🌐 Community Detection (Graph-based):")
    print(f"      Communities: {daily['community_detection']['communities']}")
    print(f"      Entropy (H_in_comm): {daily['community_detection']['entropy']:.4f}")
    print(f"      Largest community: {daily['community_detection']['largest_community']} nodes")
    
    print(f"\n   🔢 HDBSCAN (Embedding-based):")
    print(f"      Clusters: {daily['hdbscan']['clusters']}")
    print(f"      Entropy (H_in_hdb): {daily['hdbscan']['entropy']:.4f}")
    print(f"      Outlier fraction: {daily['hdbscan']['outlier_fraction']:.2%} ({daily['hdbscan']['outliers']} outliers)")
    
    print(f"\n   ⚖️  Comparison:")
    print(f"      Entropy difference: {daily['comparison']['entropy_difference']:.4f}")
    print(f"      Cluster count difference: {daily['comparison']['cluster_count_difference']}")


def main():
    summary_file = Path("weekly_analysis/weekly_summary.json")
    
    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        print("Please run batch_weekly_analysis.py first.")
        return
    
    with open(summary_file, 'r') as f:
        report = json.load(f)
    
    print("=" * 80)
    print("📊 Weekly Tweet Analysis Report: October 24-31, 2025")
    print("=" * 80)
    print(f"\nTotal days analyzed: {report['overview']['total_days']}")
    print(f"Dates: {', '.join(report['overview']['dates_analyzed'])}")
    
    # Daily reports
    print("\n" + "=" * 80)
    print("📅 DAILY BREAKDOWN")
    print("=" * 80)
    
    for daily in report['daily_results']:
        if daily['date'] != 'full_week':
            print_daily_report(daily)
    
    # Weekly summary
    if report['weekly_summary']:
        weekly = report['weekly_summary']
        print("\n" + "=" * 80)
        print("📊 FULL WEEK SUMMARY (Oct 24-31)")
        print("=" * 80)
        print(f"\nTotal tweets: {weekly['tweet_count']}")
        
        print(f"\n🌐 Community Detection (Graph-based):")
        print(f"   Communities: {weekly['community_detection']['communities']}")
        print(f"   Entropy (H_in_comm): {weekly['community_detection']['entropy']:.4f}")
        print(f"   → {'Diverse exploration' if weekly['community_detection']['entropy'] > 0.7 else 'Focused clusters'}")
        
        print(f"\n🔢 HDBSCAN (Embedding-based):")
        print(f"   Clusters: {weekly['hdbscan']['clusters']}")
        print(f"   Entropy (H_in_hdb): {weekly['hdbscan']['entropy']:.4f}")
        print(f"   Outlier fraction: {weekly['hdbscan']['outlier_fraction']:.2%}")
        
        if weekly['hdbscan']['outlier_fraction'] > 0.3:
            print(f"   → High outlier rate suggests chaotic exploration or potential new directions")
        elif weekly['hdbscan']['outlier_fraction'] > 0.1:
            print(f"   → Moderate outlier rate suggests some novel/experimental ideas")
        else:
            print(f"   → Low outlier rate suggests mostly coherent clusters")
    
    # Comparison table
    print("\n" + "=" * 80)
    print("📈 COMPARISON TABLE")
    print("=" * 80)
    print(f"\n{'Date':<12} {'Tweets':<8} {'Comm':<6} {'H_comm':<10} {'HDB':<6} {'H_hdb':<10} {'Outliers':<12}")
    print("-" * 80)
    
    for daily in report['daily_results']:
        if daily['date'] != 'full_week':
            print(f"{daily['date']:<12} {daily['tweet_count']:<8} "
                  f"{daily['community_detection']['communities']:<6} "
                  f"{daily['community_detection']['entropy']:<10.4f} "
                  f"{daily['hdbscan']['clusters']:<6} "
                  f"{daily['hdbscan']['entropy']:<10.4f} "
                  f"{daily['hdbscan']['outlier_fraction']:<12.2%}")
    
    if report['weekly_summary']:
        weekly = report['weekly_summary']
        print("-" * 80)
        print(f"{'WEEK TOTAL':<12} {weekly['tweet_count']:<8} "
              f"{weekly['community_detection']['communities']:<6} "
              f"{weekly['community_detection']['entropy']:<10.4f} "
              f"{weekly['hdbscan']['clusters']:<6} "
              f"{weekly['hdbscan']['entropy']:<10.4f} "
              f"{weekly['hdbscan']['outlier_fraction']:<12.2%}")
    
    print("\n" + "=" * 80)
    print("💡 INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
🌐 Community Detection (Graph-based):
   - Sees ideas as connected nodes in a similarity network
   - High communities + high entropy = diverse exploration across many topics
   - Few communities + low entropy = focused deep dive on fewer topics
   
🔢 HDBSCAN (Embedding-based):
   - Sees ideas as points in semantic embedding space
   - Many clusters + high entropy = diverse exploration
   - High outlier rate = novel/experimental ideas that don't fit patterns
   
⚖️  Comparing Both:
   - Large difference in cluster counts = different perspectives on same data
   - Similar entropy = consistent level of diversity regardless of method
   - High outliers in HDBSCAN = ideas that are semantically unique even if graph-connected
    """)
    
    print("\n📁 Detailed results available in:")
    print("   - weekly_analysis/[date]/ for daily analysis")
    print("   - weekly_analysis/full_week/ for week overview")
    print("   - weekly_analysis/weekly_summary.json for raw data")


if __name__ == "__main__":
    main()

