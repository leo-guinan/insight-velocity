#!/usr/bin/env python3
"""
Visualize Insight Velocity metrics:
1. Phase plot: entropy vs outlier % (Insight Phase Plot)
2. Rolling trends: 3-day rolling means
3. Timeline with breakthrough days
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_phase_plot(iv_df, output_file="insight_phase_plot.png"):
    """
    Create Insight Phase Plot: entropy vs outlier %
    
    The sweet spot is moderate entropy (0.85-0.95), 10-25% outliers
    = maximal integrative synthesis zone
    """
    print("\n📊 Creating Insight Phase Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate average entropy (breadth)
    avg_entropy = (iv_df['entropy_comm'] + iv_df['entropy_hdb']) / 2.0
    outlier_fraction = iv_df['outlier_fraction']
    
    # Color by post count (synthesis indicator)
    colors = iv_df['posts'].apply(lambda x: 'darkgreen' if x > 0 else 'lightblue')
    sizes = iv_df['tweets'] * 10  # Scale tweet count for size
    
    # Plot points
    scatter = ax.scatter(avg_entropy, outlier_fraction, 
                        s=sizes, c=colors, alpha=0.7, 
                        edgecolors='black', linewidths=1.5)
    
    # Annotate points with dates
    for idx, row in iv_df.iterrows():
        date_short = row['date'].split('-')[-1]  # Just day
        ax.annotate(date_short, 
                   ((row['entropy_comm'] + row['entropy_hdb']) / 2.0,
                    row['outlier_fraction']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    # Mark sweet spot zone
    sweet_spot_x = [0.85, 0.95, 0.95, 0.85, 0.85]
    sweet_spot_y = [0.10, 0.10, 0.25, 0.25, 0.10]
    ax.plot(sweet_spot_x, sweet_spot_y, 'g--', linewidth=2, alpha=0.5, label='Sweet Spot Zone')
    ax.fill(sweet_spot_x, sweet_spot_y, alpha=0.1, color='green')
    
    # Labels and title
    ax.set_xlabel('Average Entropy (Breadth)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outlier Fraction (Novelty)', fontsize=12, fontweight='bold')
    ax.set_title('Insight Phase Plot\nEntropy vs Novelty', fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', edgecolor='black', label='Days with Posts (Synthesis)'),
        Patch(facecolor='lightblue', edgecolor='black', label='Days without Posts'),
        plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Sweet Spot Zone')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add interpretation text
    ax.text(0.02, 0.98, 
           'Sweet Spot: Moderate entropy (0.85-0.95)\n+ 10-25% outliers = Maximal Synthesis',
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved phase plot to: {output_file}")
    
    plt.close()


def create_trend_plots(iv_df, output_file="insight_trends.png"):
    """
    Create rolling trend plots:
    1. Entropy trends (3-day rolling mean)
    2. Outlier fraction trends
    3. IV trends
    """
    print("\n📈 Creating trend plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    dates = pd.to_datetime(iv_df['date'])
    
    # Plot 1: Entropy Trends
    ax1 = axes[0]
    ax1.plot(dates, iv_df['entropy_comm'], 'o-', label='Community Detection Entropy', linewidth=2, markersize=8)
    ax1.plot(dates, iv_df['entropy_hdb'], 's-', label='HDBSCAN Entropy', linewidth=2, markersize=8)
    if 'rolling_entropy_comm' in iv_df.columns:
        rolling_comm = iv_df['rolling_entropy_comm'].bfill()
        rolling_hdb = iv_df['rolling_entropy_comm'].bfill()  # Use same column since we only have avg
        rolling_mean = (rolling_comm + rolling_hdb) / 2.0
        ax1.plot(dates, rolling_mean, '--', label='3-Day Rolling Mean', linewidth=2, alpha=0.7)
    
    ax1.set_ylabel('Entropy (Breadth)', fontsize=11, fontweight='bold')
    ax1.set_title('Entropy Trends: Daily vs Rolling Mean', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Outlier Fraction Trends
    ax2 = axes[1]
    ax2.plot(dates, iv_df['outlier_fraction'], 'o-', color='orange', 
            label='Daily Outlier Fraction', linewidth=2, markersize=8)
    if 'rolling_outlier_fraction' in iv_df.columns:
        rolling_outliers = iv_df['rolling_outlier_fraction'].bfill()
        ax2.plot(dates, rolling_outliers, '--', color='red', 
                label='3-Day Rolling Mean', linewidth=2, alpha=0.7)
    
    # Mark sweet spot zone
    ax2.axhspan(0.10, 0.25, alpha=0.2, color='green', label='Sweet Spot Range')
    
    ax2.set_ylabel('Outlier Fraction (Novelty)', fontsize=11, fontweight='bold')
    ax2.set_title('Novelty Trends: Daily vs Rolling Mean', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: IV Trends
    ax3 = axes[2]
    ax3.plot(dates, iv_df['insight_velocity'], 'o-', color='purple', 
            label='Daily Insight Velocity', linewidth=2, markersize=8)
    if 'rolling_iv' in iv_df.columns:
        rolling_iv = iv_df['rolling_iv'].bfill()
        ax3.plot(dates, rolling_iv, '--', color='darkviolet', 
                label='3-Day Rolling Mean', linewidth=2, alpha=0.7)
    
    # Mark days with posts
    posts_mask = iv_df['posts'] > 0
    if posts_mask.any():
        ax3.scatter(dates[posts_mask], iv_df.loc[posts_mask, 'insight_velocity'],
                   s=200, c='darkgreen', marker='*', 
                   label='Days with Blog Posts', zorder=5, edgecolors='black', linewidths=1)
    
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Insight Velocity (IV)', fontsize=11, fontweight='bold')
    ax3.set_title('Insight Velocity Trends', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.FixedFormatter([d.strftime('%m/%d') for d in dates]))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved trend plots to: {output_file}")
    
    plt.close()


def create_component_breakdown(iv_df, output_file="iv_components.png"):
    """
    Create stacked bar chart showing IV component breakdown.
    """
    print("\n🔢 Creating component breakdown...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = iv_df['date']
    
    # Normalized components (for visualization)
    # Note: These are already normalized in IV calculation, but we'll show proportions
    components = {
        'Breadth': iv_df['breadth'] * 0.3,
        'Novelty': iv_df['novelty'] * 0.2,
        'Integration': iv_df['integration'] * 0.2,
        'Compression': iv_df['compression'] * 0.15,
        'Entropy Reduction': iv_df['entropy_reduction'].clip(lower=0) * 0.15,
    }
    
    bottom = np.zeros(len(dates))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (component, values) in enumerate(components.items()):
        ax.bar(dates, values, bottom=bottom, label=component, color=colors[i], alpha=0.8)
        bottom += values
    
    ax.set_ylabel('Contribution to IV', fontsize=11, fontweight='bold')
    ax.set_title('Insight Velocity Component Breakdown', fontsize=12, fontweight='bold')
    ax.set_xticklabels([d.split('-')[-1] for d in dates], rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved component breakdown to: {output_file}")
    
    plt.close()


def main():
    iv_file = "iv_metrics.csv"
    
    print("=" * 80)
    print("📊 Visualizing Insight Velocity Metrics")
    print("=" * 80)
    
    # Load IV metrics
    iv_df = pd.read_csv(iv_file)
    iv_df['date'] = iv_df['date'].astype(str)
    
    print(f"\nLoaded IV metrics for {len(iv_df)} days")
    
    # Create visualizations
    try:
        create_phase_plot(iv_df)
        create_trend_plots(iv_df)
        create_component_breakdown(iv_df)
        
        print("\n" + "=" * 80)
        print("✅ Visualizations complete!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  - insight_phase_plot.png - Entropy vs Novelty phase diagram")
        print("  - insight_trends.png - Rolling trends for entropy, novelty, and IV")
        print("  - iv_components.png - Component breakdown by day")
        
    except ImportError:
        print("\n⚠️  Matplotlib not installed. Skipping visualizations.")
        print("Install with: pip install matplotlib")
    except Exception as e:
        print(f"\n⚠️  Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

