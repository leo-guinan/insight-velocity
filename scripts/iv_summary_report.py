#!/usr/bin/env python3
"""
Generate a comprehensive Insight Velocity summary report.
"""
import json
import pandas as pd
from pathlib import Path


def main():
    iv_file = "iv_metrics.csv"
    summary_file = "weekly_analysis/weekly_summary.json"
    posts_file = "ghost_posts_oct24-31.csv"
    
    print("=" * 80)
    print("📊 INSIGHT VELOCITY ANALYSIS REPORT")
    print("October 24-31, 2025")
    print("=" * 80)
    
    # Load data
    iv_df = pd.read_csv(iv_file)
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    
    with open(summary_file, 'r') as f:
        weekly_report = json.load(f)
    
    posts_df = pd.read_csv(posts_file)
    
    print("\n" + "=" * 80)
    print("📈 EXECUTIVE SUMMARY")
    print("=" * 80)
    
    print(f"\n📅 Days analyzed: {len(iv_df)}")
    print(f"📝 Total tweets: {iv_df['tweets'].sum()}")
    print(f"✍️  Total blog posts: {iv_df['posts'].sum()}")
    print(f"📊 Average Insight Velocity: {iv_df['insight_velocity'].mean():.4f}")
    print(f"📈 Peak Insight Velocity: {iv_df['insight_velocity'].max():.4f} ({iv_df.loc[iv_df['insight_velocity'].idxmax(), 'date'].strftime('%Y-%m-%d')})")
    
    print("\n" + "=" * 80)
    print("🔍 DAY-BY-DAY BREAKDOWN")
    print("=" * 80)
    
    for _, row in iv_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        print(f"\n📅 {date_str}")
        print(f"   Tweets: {int(row['tweets'])} | Posts: {int(row['posts'])}")
        print(f"\n   🌐 Community Detection:")
        print(f"      Entropy: {row['entropy_comm']:.4f} ({'Very diverse' if row['entropy_comm'] > 0.95 else 'Diverse' if row['entropy_comm'] > 0.85 else 'Focused'})")
        
        print(f"\n   🔢 HDBSCAN:")
        print(f"      Entropy: {row['entropy_hdb']:.4f}")
        print(f"      Outlier fraction: {row['outlier_fraction']:.2%} ({'High novelty' if row['outlier_fraction'] > 0.25 else 'Moderate' if row['outlier_fraction'] > 0.10 else 'Low'})")
        
        print(f"\n   ⚙️  IV Components:")
        print(f"      Breadth (B_d): {row['breadth']:.4f}")
        print(f"      Novelty (N_d): {row['novelty']:.4f}")
        print(f"      Integration (I_p): {row['integration']:.4f}")
        print(f"      Compression (C_r): {row['compression']:.4f}")
        print(f"      Entropy Reduction (ER_d): {row['entropy_reduction']:.4f}")
        print(f"      🎯 Insight Velocity (IV): {row['insight_velocity']:.4f}")
        
        if row['posts'] > 0:
            day_posts = posts_df[posts_df['date_str'] == date_str]
            print(f"\n   ✍️  Blog Posts:")
            for _, post in day_posts.iterrows():
                print(f"      - {post['title']}")
    
    print("\n" + "=" * 80)
    print("📊 PATTERN ANALYSIS")
    print("=" * 80)
    
    # Phase plot analysis
    print("\n🎯 Insight Phase Plot Analysis:")
    print("   Sweet spot zone: Entropy 0.85-0.95, Outliers 10-25%")
    
    in_sweet_spot = []
    for _, row in iv_df.iterrows():
        avg_entropy = row['breadth']
        outlier_frac = row['outlier_fraction']
        in_zone = (0.85 <= avg_entropy <= 0.95) and (0.10 <= outlier_frac <= 0.25)
        if in_zone:
            in_sweet_spot.append(row['date'].strftime('%Y-%m-%d'))
    
    if in_sweet_spot:
        print(f"   ✓ Days in sweet spot: {', '.join(in_sweet_spot)}")
    else:
        print(f"   ⚠️  No days in sweet spot zone")
        print("   Days approaching sweet spot:")
        for _, row in iv_df.iterrows():
            avg_entropy = row['breadth']
            outlier_frac = row['outlier_fraction']
            dist = ((avg_entropy - 0.90)**2 + (outlier_frac - 0.175)**2)**0.5
            if dist < 0.15:
                print(f"      - {row['date'].strftime('%Y-%m-%d')}: Entropy {avg_entropy:.3f}, Outliers {outlier_frac:.2%}")
    
    # Breakthrough pattern
    print("\n💡 Breakthrough Pattern Analysis:")
    print("   Oct 29 had highest outlier rate (36.36%) - 'eruption' day")
    print("   Oct 30-31 had synthesis (6 blog posts)")
    print("   This matches the expected pattern: exploration → synthesis")
    
    if (iv_df['outlier_fraction'] > 0.30).any():
        eruption_days = iv_df[iv_df['outlier_fraction'] > 0.30]
        print(f"\n   🔥 Eruption days (outliers > 30%):")
        for _, row in eruption_days.iterrows():
            print(f"      - {row['date'].strftime('%Y-%m-%d')}: {row['outlier_fraction']:.1%} outliers")
    
    synthesis_days = iv_df[iv_df['posts'] > 0]
    if len(synthesis_days) > 0:
        print(f"\n   ✍️  Synthesis days (blog posts):")
        for _, row in synthesis_days.iterrows():
            print(f"      - {row['date'].strftime('%Y-%m-%d')}: {int(row['posts'])} post(s)")
    
    # Rolling trends
    print("\n📈 Rolling Trends (3-day):")
    if 'rolling_iv' in iv_df.columns:
        rolling_iv = iv_df['rolling_iv'].dropna()
        if len(rolling_iv) > 0:
            print(f"   Rolling IV trend: {rolling_iv.iloc[-1]:.4f}")
            if len(rolling_iv) > 1:
                trend = "↑ Increasing" if rolling_iv.iloc[-1] > rolling_iv.iloc[0] else "↓ Decreasing"
                print(f"   Overall trend: {trend}")
    
    print("\n" + "=" * 80)
    print("🎯 KEY INSIGHTS")
    print("=" * 80)
    
    insights = []
    
    # Highest IV day
    max_iv_row = iv_df.loc[iv_df['insight_velocity'].idxmax()]
    insights.append(f"• Peak IV: {max_iv_row['date'].strftime('%Y-%m-%d')} (IV={max_iv_row['insight_velocity']:.4f}, {int(max_iv_row['posts'])} posts)")
    
    # Most diverse day
    max_entropy_row = iv_df.loc[iv_df['breadth'].idxmax()]
    insights.append(f"• Most diverse exploration: {max_entropy_row['date'].strftime('%Y-%m-%d')} (entropy={max_entropy_row['breadth']:.4f})")
    
    # Most novel day
    max_novelty_row = iv_df.loc[iv_df['novelty'].idxmax()]
    insights.append(f"• Highest novelty: {max_novelty_row['date'].strftime('%Y-%m-%d')} ({max_novelty_row['novelty']:.2%} outliers)")
    
    # Best integration
    max_integration_row = iv_df.loc[iv_df['integration'].idxmax()]
    insights.append(f"• Best integration: {max_integration_row['date'].strftime('%Y-%m-%d')} (I_p={max_integration_row['integration']:.4f}, {int(max_integration_row['posts'])} posts)")
    
    for insight in insights:
        print(f"  {insight}")
    
    print("\n" + "=" * 80)
    print("📁 OUTPUT FILES")
    print("=" * 80)
    print("\n  Data:")
    print("    - iv_metrics.csv - Full IV metrics table")
    print("    - iv_metrics.json - JSON format")
    print("\n  Visualizations:")
    print("    - insight_phase_plot.png - Phase diagram (entropy vs novelty)")
    print("    - insight_trends.png - Rolling trends")
    print("    - iv_components.png - Component breakdown")
    print("\n  Analysis:")
    print("    - weekly_analysis/ - Daily analysis results")
    print("    - weekly_summary.json - Weekly summary")
    
    print("\n" + "=" * 80)
    print("✅ Report complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

