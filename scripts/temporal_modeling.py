#!/usr/bin/env python3
"""
Step 3: Temporal Modeling - Turning IV into a Predictive Signal

Computes:
1. Derivatives & Momentum:
   - ΔIV = IV_t - IV_(t-1) (daily acceleration)
   - IV_momentum = rolling_mean(ΔIV, 3) (3-day velocity trend)
   - exploration_pressure = Novelty_t × Entropy_t (early warning signal)

2. Forecast Synthesis Windows:
   - Sweet spot detection
   - Synthesis burst prediction based on outlier thresholds
   - Next Synthesis Probability
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_derivatives_and_momentum(iv_df):
    """
    Calculate temporal derivatives and momentum metrics.
    
    Returns DataFrame with added columns:
    - delta_iv: Daily acceleration (IV_t - IV_(t-1))
    - iv_momentum: 3-day rolling mean of ΔIV
    - exploration_pressure: Novelty × Entropy
    """
    print("\n🔢 Calculating Derivatives & Momentum...")
    
    # Sort by date to ensure correct ordering
    iv_df = iv_df.sort_values('date').reset_index(drop=True)
    
    # Calculate daily acceleration (ΔIV)
    iv_df['delta_iv'] = iv_df['insight_velocity'].diff()
    iv_df['delta_iv'] = iv_df['delta_iv'].fillna(0.0)  # First day has no previous value
    
    # Calculate IV momentum (3-day rolling mean of ΔIV)
    iv_df['iv_momentum'] = iv_df['delta_iv'].rolling(window=3, min_periods=1).mean()
    
    # Calculate exploration pressure (Novelty × Entropy)
    # Using average entropy (breadth) as the entropy measure
    iv_df['exploration_pressure'] = iv_df['novelty'] * iv_df['breadth']
    
    # Calculate additional momentum metrics
    iv_df['iv_velocity'] = iv_df['insight_velocity']  # Current velocity
    iv_df['momentum_direction'] = iv_df['delta_iv'].apply(lambda x: '↑' if x > 0 else '↓' if x < 0 else '→')
    
    print(f"  ✓ Calculated ΔIV, IV momentum, and exploration pressure")
    
    return iv_df


def forecast_synthesis_windows(iv_df):
    """
    Forecast synthesis windows based on patterns.
    
    Rules:
    - Sweet spot: Entropy ∈ [0.85, 0.95] and Outliers ∈ [0.1, 0.25]
    - High outliers (>0.30) → expect synthesis burst ≈ 1–2 days later
    """
    print("\n🔮 Forecasting Synthesis Windows...")
    
    # Sort by date
    iv_df = iv_df.sort_values('date').reset_index(drop=True)
    
    # Initialize forecast columns
    iv_df['in_sweet_spot'] = False
    iv_df['synthesis_probability'] = 0.0
    iv_df['expected_synthesis_days'] = 0
    iv_df['eruption_detected'] = False
    iv_df['synthesis_forecast'] = ''
    
    for idx, row in iv_df.iterrows():
        entropy = row['breadth']
        outliers = row['outlier_fraction']
        has_posts = row['posts'] > 0
        
        # Check if in sweet spot zone
        in_sweet_spot = (0.85 <= entropy <= 0.95) and (0.10 <= outliers <= 0.25)
        iv_df.at[idx, 'in_sweet_spot'] = in_sweet_spot
        
        # Check for eruption (high outliers)
        is_eruption = outliers > 0.30
        iv_df.at[idx, 'eruption_detected'] = is_eruption
        
        # Calculate synthesis probability
        synthesis_prob = 0.0
        
        # Base probability from current state
        if in_sweet_spot:
            synthesis_prob += 0.4  # High probability if in sweet spot
        
        if has_posts:
            synthesis_prob += 0.3  # Currently synthesizing
        
        # Check for recent eruptions
        if idx > 0:
            # Look back 1-2 days for eruptions
            for lookback in [1, 2]:
                if idx >= lookback:
                    prev_outliers = iv_df.iloc[idx - lookback]['outlier_fraction']
                    if prev_outliers > 0.30:
                        # Eruption detected, synthesis likely
                        days_since_eruption = lookback
                        synthesis_prob += 0.4 / days_since_eruption  # Decay with time
                        iv_df.at[idx, 'expected_synthesis_days'] = days_since_eruption
        
        # Forward-looking: predict future synthesis
        if is_eruption and idx < len(iv_df) - 1:
            # Mark next 1-2 days as likely synthesis windows
            for forward in [1, 2]:
                if idx + forward < len(iv_df):
                    future_prob = 0.5 / forward  # Decay with distance
                    if pd.isna(iv_df.at[idx + forward, 'synthesis_probability']):
                        iv_df.at[idx + forward, 'synthesis_probability'] = 0.0
                    iv_df.at[idx + forward, 'synthesis_probability'] += future_prob
        
        # Clamp probability to [0, 1]
        synthesis_prob = min(1.0, max(0.0, synthesis_prob))
        iv_df.at[idx, 'synthesis_probability'] = synthesis_prob
        
        # Generate forecast text
        forecast_parts = []
        if in_sweet_spot:
            forecast_parts.append("Sweet spot")
        if is_eruption:
            forecast_parts.append("Eruption detected")
        if synthesis_prob > 0.5:
            forecast_parts.append(f"High synthesis likelihood ({synthesis_prob:.1%})")
        elif synthesis_prob > 0.3:
            forecast_parts.append(f"Moderate synthesis likelihood ({synthesis_prob:.1%})")
        
        iv_df.at[idx, 'synthesis_forecast'] = " | ".join(forecast_parts) if forecast_parts else "Normal"
    
    print(f"  ✓ Calculated synthesis forecasts for {len(iv_df)} days")
    
    return iv_df


def create_momentum_plots(iv_df, output_dir="."):
    """Create visualizations for momentum analysis."""
    print("\n📊 Creating Momentum Visualizations...")
    
    # Sort by date
    iv_df = iv_df.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(iv_df['date'])
    
    # Plot 1: IV vs ΔIV (Idea Momentum Plot)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by synthesis probability
    colors = iv_df['synthesis_probability'].apply(
        lambda x: 'darkgreen' if x > 0.5 else 'orange' if x > 0.3 else 'lightblue'
    )
    sizes = 100 + iv_df['posts'] * 50  # Size by post count
    
    scatter = ax.scatter(iv_df['insight_velocity'], iv_df['delta_iv'],
                        s=sizes, c=colors, alpha=0.7,
                        edgecolors='black', linewidths=1.5)
    
    # Annotate with dates
    for idx, row in iv_df.iterrows():
        date_short = row['date'].strftime('%m/%d') if isinstance(row['date'], pd.Timestamp) else row['date'].split('-')[-2:]
        date_short = '/'.join(row['date'].split('-')[-2:]) if isinstance(row['date'], str) else date_short
        ax.annotate(date_short,
                   (row['insight_velocity'], row['delta_iv']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    # Add quadrant lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=iv_df['insight_velocity'].mean(), color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add quadrants labels
    ax.text(0.99, 0.99, 'Accelerating\nHigh IV', transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax.text(0.99, 0.01, 'Decelerating\nHigh IV', transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    ax.set_xlabel('Insight Velocity (IV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ΔIV (Daily Acceleration)', fontsize=12, fontweight='bold')
    ax.set_title('Idea Momentum Plot\nIV vs Daily Acceleration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', edgecolor='black', label='High Synthesis Probability (>50%)'),
        Patch(facecolor='orange', edgecolor='black', label='Moderate Synthesis Probability (30-50%)'),
        Patch(facecolor='lightblue', edgecolor='black', label='Normal/Low Synthesis Probability'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    momentum_plot_file = Path(output_dir) / "idea_momentum_plot.png"
    plt.savefig(momentum_plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved momentum plot to: {momentum_plot_file}")
    plt.close()
    
    # Plot 2: Momentum Timeline
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # IV and Momentum
    ax1 = axes[0]
    ax1.plot(dates, iv_df['insight_velocity'], 'o-', label='Insight Velocity', linewidth=2, markersize=8)
    ax1.plot(dates, iv_df['iv_momentum'], 's--', label='IV Momentum (3-day rolling)', linewidth=2, markersize=6)
    
    # Mark synthesis days
    synthesis_mask = iv_df['posts'] > 0
    if synthesis_mask.any():
        ax1.scatter(dates[synthesis_mask], iv_df.loc[synthesis_mask, 'insight_velocity'],
                   s=200, c='darkgreen', marker='*', label='Synthesis Days (Posts)',
                   zorder=5, edgecolors='black', linewidths=1)
    
    ax1.set_ylabel('IV / Momentum', fontsize=11, fontweight='bold')
    ax1.set_title('Insight Velocity Momentum Timeline', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # ΔIV and Exploration Pressure
    ax2 = axes[1]
    ax2.plot(dates, iv_df['delta_iv'], 'o-', color='purple', label='ΔIV (Daily Acceleration)',
            linewidth=2, markersize=8)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(dates, iv_df['exploration_pressure'], 's-', color='orange',
                  label='Exploration Pressure', linewidth=2, markersize=6, alpha=0.7)
    
    ax2.set_ylabel('ΔIV', fontsize=11, fontweight='bold', color='purple')
    ax2_twin.set_ylabel('Exploration Pressure', fontsize=11, fontweight='bold', color='orange')
    ax2.set_title('Daily Acceleration & Exploration Pressure', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Synthesis Probability
    ax3 = axes[2]
    ax3.plot(dates, iv_df['synthesis_probability'], 'o-', color='darkgreen',
            label='Synthesis Probability', linewidth=2, markersize=8)
    ax3.axhspan(0.3, 0.5, alpha=0.2, color='yellow', label='Moderate Zone')
    ax3.axhspan(0.5, 1.0, alpha=0.2, color='green', label='High Probability Zone')
    
    # Mark eruption days
    eruption_mask = iv_df['eruption_detected']
    if eruption_mask.any():
        ax3.scatter(dates[eruption_mask], iv_df.loc[eruption_mask, 'synthesis_probability'],
                   s=200, c='red', marker='^', label='Eruption Days',
                   zorder=5, edgecolors='black', linewidths=1)
    
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Synthesis Probability', fontsize=11, fontweight='bold')
    ax3.set_title('Synthesis Window Forecast', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    timeline_file = Path(output_dir) / "momentum_timeline.png"
    plt.savefig(timeline_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved momentum timeline to: {timeline_file}")
    plt.close()


def generate_momentum_report(iv_df, output_file="temporal_modeling_report.txt"):
    """Generate a text report of temporal modeling insights."""
    print("\n📝 Generating Momentum Report...")
    
    iv_df = iv_df.sort_values('date').reset_index(drop=True)
    
    report = []
    report.append("=" * 80)
    report.append("🧭 TEMPORAL MODELING REPORT - Insight Velocity Momentum Analysis")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics
    report.append("📊 SUMMARY STATISTICS")
    report.append("-" * 80)
    report.append(f"Average IV: {iv_df['insight_velocity'].mean():.4f}")
    report.append(f"Average ΔIV: {iv_df['delta_iv'].mean():.4f}")
    report.append(f"Average IV Momentum: {iv_df['iv_momentum'].mean():.4f}")
    report.append(f"Average Exploration Pressure: {iv_df['exploration_pressure'].mean():.4f}")
    report.append(f"Average Synthesis Probability: {iv_df['synthesis_probability'].mean():.2%}")
    report.append("")
    
    # Day-by-day analysis
    report.append("=" * 80)
    report.append("📅 DAY-BY-DAY MOMENTUM ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    for idx, row in iv_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date']
        report.append(f"📅 {date_str}")
        report.append(f"   IV: {row['insight_velocity']:.4f} | ΔIV: {row['delta_iv']:+.4f} ({row['momentum_direction']})")
        report.append(f"   IV Momentum: {row['iv_momentum']:+.4f}")
        report.append(f"   Exploration Pressure: {row['exploration_pressure']:.4f}")
        report.append(f"   Synthesis Probability: {row['synthesis_probability']:.1%}")
        
        if row['eruption_detected']:
            report.append(f"   🔥 ERUPTION DETECTED (outliers: {row['outlier_fraction']:.1%})")
        
        if row['in_sweet_spot']:
            report.append(f"   ✓ In Sweet Spot Zone")
        
        if row['posts'] > 0:
            report.append(f"   ✍️  Synthesis: {int(row['posts'])} post(s)")
        
        if row['synthesis_forecast']:
            report.append(f"   📊 Forecast: {row['synthesis_forecast']}")
        
        report.append("")
    
    # Pattern analysis
    report.append("=" * 80)
    report.append("🔍 PATTERN ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    # Eruption → Synthesis pattern
    eruptions = iv_df[iv_df['eruption_detected']]
    if len(eruptions) > 0:
        report.append("🔥 Eruption → Synthesis Pattern:")
        for _, eruption in eruptions.iterrows():
            eruption_date = eruption['date'].strftime('%Y-%m-%d') if isinstance(eruption['date'], pd.Timestamp) else eruption['date']
            report.append(f"   {eruption_date}: {eruption['outlier_fraction']:.1%} outliers")
            
            # Check next 1-2 days for synthesis
            eruption_idx = iv_df[iv_df['date'] == eruption['date']].index[0]
            for forward in [1, 2]:
                if eruption_idx + forward < len(iv_df):
                    future = iv_df.iloc[eruption_idx + forward]
                    future_date = future['date'].strftime('%Y-%m-%d') if isinstance(future['date'], pd.Timestamp) else future['date']
                    if future['posts'] > 0:
                        report.append(f"      → {future_date}: {int(future['posts'])} post(s) ✓ SYNTHESIS CONFIRMED")
    
    report.append("")
    
    # Momentum insights
    report.append("📈 Momentum Insights:")
    accelerating_days = iv_df[iv_df['delta_iv'] > 0]
    if len(accelerating_days) > 0:
        report.append(f"   - Accelerating days: {len(accelerating_days)}/{len(iv_df)} ({len(accelerating_days)/len(iv_df):.1%})")
        avg_accel = accelerating_days['delta_iv'].mean()
        report.append(f"   - Average acceleration: {avg_accel:+.4f}")
    
    decelerating_days = iv_df[iv_df['delta_iv'] < 0]
    if len(decelerating_days) > 0:
        report.append(f"   - Decelerating days: {len(decelerating_days)}/{len(iv_df)} ({len(decelerating_days)/len(iv_df):.1%})")
        avg_decel = decelerating_days['delta_iv'].mean()
        report.append(f"   - Average deceleration: {avg_decel:+.4f}")
    
    report.append("")
    
    # Forecast summary
    report.append("=" * 80)
    report.append("🔮 SYNTHESIS FORECAST SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    high_prob_days = iv_df[iv_df['synthesis_probability'] > 0.5]
    if len(high_prob_days) > 0:
        report.append(f"High Synthesis Probability (>50%): {len(high_prob_days)} days")
        for _, day in high_prob_days.iterrows():
            date_str = day['date'].strftime('%Y-%m-%d') if isinstance(day['date'], pd.Timestamp) else day['date']
            report.append(f"   - {date_str}: {day['synthesis_probability']:.1%}")
    
    report.append("")
    report.append("=" * 80)
    
    # Write report
    report_text = "\n".join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved report to: {output_file}")
    
    return report_text


def main():
    iv_file = "iv_metrics.csv"
    output_dir = Path(".")
    
    print("=" * 80)
    print("🧭 Step 3: Temporal Modeling - Turning IV into a Predictive Signal")
    print("=" * 80)
    
    # Load IV metrics
    print("\n📥 Loading IV metrics...")
    iv_df = pd.read_csv(iv_file)
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    
    print(f"  Loaded metrics for {len(iv_df)} days")
    
    # Calculate derivatives and momentum
    iv_df = calculate_derivatives_and_momentum(iv_df)
    
    # Forecast synthesis windows
    iv_df = forecast_synthesis_windows(iv_df)
    
    # Save enhanced metrics
    enhanced_file = "iv_metrics_temporal.csv"
    iv_df.to_csv(enhanced_file, index=False)
    print(f"\n✓ Saved enhanced metrics to: {enhanced_file}")
    
    # Save JSON version
    enhanced_json = "iv_metrics_temporal.json"
    iv_df_dict = iv_df.to_dict(orient='records')
    with open(enhanced_json, 'w') as f:
        json.dump(iv_df_dict, f, indent=2, default=str)
    print(f"✓ Saved enhanced metrics (JSON) to: {enhanced_json}")
    
    # Create visualizations
    try:
        create_momentum_plots(iv_df, output_dir)
    except Exception as e:
        print(f"\n⚠️  Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate report
    report_text = generate_momentum_report(iv_df)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEMPORAL MODELING SUMMARY")
    print("=" * 80)
    print(f"\n{'Date':<12} {'IV':<8} {'ΔIV':<8} {'Momentum':<10} {'Synthesis Prob':<15}")
    print("-" * 80)
    
    for _, row in iv_df.sort_values('date').iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date']
        print(f"{date_str:<12} {row['insight_velocity']:<8.4f} {row['delta_iv']:<+8.4f} "
              f"{row['iv_momentum']:<+10.4f} {row['synthesis_probability']:<15.1%}")
    
    print("\n" + "=" * 80)
    print("✅ Temporal modeling complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {enhanced_file} - Enhanced IV metrics with temporal features")
    print(f"  - {enhanced_json} - JSON version")
    print(f"  - idea_momentum_plot.png - IV vs ΔIV visualization")
    print(f"  - momentum_timeline.png - Momentum timeline")
    print(f"  - temporal_modeling_report.txt - Detailed report")


if __name__ == "__main__":
    main()

