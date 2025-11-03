#!/usr/bin/env python3
"""
Step 7: Blind Spot Analysis - Expected Insights That Didn't Occur

Identifies days where synthesis was predicted but didn't happen,
and analyzes why the system failed to collapse into synthesis.
"""
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


def identify_blind_spots(iv_df, threshold=0.5):
    """
    Identify blind spots: days where P_t > threshold but S_t = 0
    
    Returns dataframe with blind spot flags and scores.
    """
    print("\n🔍 Identifying Blind Spots...")
    
    iv_df = iv_df.copy()
    
    # Use learned synthesis probability if available, otherwise use original
    if 'synthesis_prob_learned' in iv_df.columns:
        pred_prob = iv_df['synthesis_prob_learned']
    elif 'synthesis_probability' in iv_df.columns:
        pred_prob = iv_df['synthesis_probability']
    else:
        print("  ⚠️  No synthesis probability column found")
        return iv_df
    
    actual_synthesis = iv_df['synthesis'] if 'synthesis' in iv_df.columns else (iv_df['posts'] > 0).astype(int)
    
    # Identify blind spots
    iv_df['is_blind_spot'] = (pred_prob > threshold) & (actual_synthesis == 0)
    blind_spots = iv_df['is_blind_spot'].sum()
    
    print(f"  Threshold: {threshold}")
    print(f"  Blind spots found: {blind_spots}/{len(iv_df)} ({blind_spots/len(iv_df):.1%})")
    
    return iv_df


def calculate_blind_spot_scores(iv_df):
    """
    Calculate blind spot intensity score:
    B_t = P_t × (1 - S_t) × f(EP_t, IV_t, ER_t)
    
    Where f(EP, IV, ER) = 0.5(EP + IV) + ER
    """
    print("\n📊 Calculating Blind Spot Scores...")
    
    iv_df = iv_df.copy()
    
    # Get synthesis probability
    if 'synthesis_prob_learned' in iv_df.columns:
        pred_prob = iv_df['synthesis_prob_learned']
    elif 'synthesis_probability' in iv_df.columns:
        pred_prob = iv_df['synthesis_probability']
    else:
        print("  ⚠️  No synthesis probability found")
        return iv_df
    
    # Get actual synthesis
    actual_synthesis = iv_df['synthesis'] if 'synthesis' in iv_df.columns else (iv_df['posts'] > 0).astype(int)
    
    # Get features
    ep = iv_df['exploration_pressure'].values if 'exploration_pressure' in iv_df.columns else iv_df['novelty'].values * iv_df['breadth'].values
    iv = iv_df['insight_velocity'].values
    er = iv_df['entropy_reduction'].values if 'entropy_reduction' in iv_df.columns else np.zeros(len(iv_df))
    
    # Calculate f(EP, IV, ER)
    f_score = 0.5 * (ep + iv) + er
    
    # Calculate blind spot score
    blind_spot_score = pred_prob * (1 - actual_synthesis) * f_score
    
    iv_df['blind_spot_score'] = blind_spot_score
    iv_df['suppression_index'] = blind_spot_score / (pred_prob + 1e-10)  # SI_t = B_t / P_t
    
    # Identify high blind spots
    iv_df['high_blind_spot'] = blind_spot_score > np.percentile(blind_spot_score[blind_spot_score > 0], 50) if (blind_spot_score > 0).any() else False
    
    print(f"  Average blind spot score: {blind_spot_score.mean():.4f}")
    print(f"  Max blind spot score: {blind_spot_score.max():.4f}")
    
    return iv_df


def cluster_blind_spots(iv_df):
    """
    Cluster blind spots by cause using context variables.
    """
    print("\n🔬 Clustering Blind Spots by Cause...")
    
    blind_spots_df = iv_df[iv_df['is_blind_spot']].copy()
    
    if len(blind_spots_df) == 0:
        print("  ⚠️  No blind spots found to cluster")
        return iv_df, []
    
    print(f"  Clustering {len(blind_spots_df)} blind spots...")
    
    # Features for clustering
    features = ['breadth', 'novelty', 'exploration_pressure', 'insight_velocity', 
                'integration', 'compression', 'domain_frac', 'label_entropy']
    
    # Filter available features
    available_features = [f for f in features if f in blind_spots_df.columns]
    
    if len(available_features) < 2:
        print("  ⚠️  Not enough features for clustering")
        return iv_df, []
    
    X = blind_spots_df[available_features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster (try K=2 to K=min(4, n_blind_spots))
    n_clusters = min(4, max(2, len(blind_spots_df)))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    blind_spots_df['blind_spot_cluster'] = clusters
    
    # Add cluster labels back to main dataframe
    iv_df['blind_spot_cluster'] = -1
    for idx in blind_spots_df.index:
        iv_df.loc[idx, 'blind_spot_cluster'] = blind_spots_df.loc[idx, 'blind_spot_cluster']
    
    # Analyze clusters
    cluster_analysis = []
    for cluster_id in range(n_clusters):
        cluster_data = blind_spots_df[blind_spots_df['blind_spot_cluster'] == cluster_id]
        if len(cluster_data) == 0:
            continue
        
        cluster_analysis.append({
            'cluster': cluster_id,
            'count': len(cluster_data),
            'avg_entropy': cluster_data['breadth'].mean() if 'breadth' in cluster_data.columns else 0,
            'avg_novelty': cluster_data['novelty'].mean() if 'novelty' in cluster_data.columns else 0,
            'avg_ep': cluster_data['exploration_pressure'].mean() if 'exploration_pressure' in cluster_data.columns else 0,
            'avg_iv': cluster_data['insight_velocity'].mean() if 'insight_velocity' in cluster_data.columns else 0,
            'avg_domain_frac': cluster_data['domain_frac'].mean() if 'domain_frac' in cluster_data.columns else 0,
            'avg_label_entropy': cluster_data['label_entropy'].mean() if 'label_entropy' in cluster_data.columns else 0,
        })
        
        print(f"\n  Cluster {cluster_id}: {len(cluster_data)} blind spots")
        print(f"    Avg Entropy: {cluster_analysis[-1]['avg_entropy']:.4f}")
        print(f"    Avg Novelty: {cluster_analysis[-1]['avg_novelty']:.4f}")
        print(f"    Avg EP: {cluster_analysis[-1]['avg_ep']:.4f}")
        print(f"    Avg IV: {cluster_analysis[-1]['avg_iv']:.4f}")
    
    return iv_df, cluster_analysis


def diagnose_blind_spot_causes(iv_df):
    """
    Diagnose probable causes of blind spots based on patterns.
    """
    print("\n🔬 Diagnosing Blind Spot Causes...")
    
    blind_spots_df = iv_df[iv_df['is_blind_spot']].copy()
    
    if len(blind_spots_df) == 0:
        print("  ⚠️  No blind spots found")
        return []
    
    diagnoses = []
    
    for idx, row in blind_spots_df.iterrows():
        diagnosis = {
            'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date'],
            'probable_cause': None,
            'intervention': None,
            'blind_spot_score': row.get('blind_spot_score', 0),
        }
        
        entropy = row.get('breadth', 0)
        novelty = row.get('novelty', 0)
        ep = row.get('exploration_pressure', 0)
        iv_val = row.get('insight_velocity', 0)
        domain_frac = row.get('domain_frac', 0)
        label_entropy = row.get('label_entropy', 0)
        synthesis_prob = row.get('synthesis_prob_learned', row.get('synthesis_probability', 0))
        integration = row.get('integration', 0)
        
        # Pattern matching
        if entropy > 0.9 and novelty > 0.25:
            diagnosis['probable_cause'] = 'Overload - not enough damping/integration time'
            diagnosis['intervention'] = 'Reduce input entropy next cycle'
        elif entropy < 0.85 and iv_val > 0.45:
            diagnosis['probable_cause'] = 'Overconstraint - ideas too cohesive, no fresh input'
            diagnosis['intervention'] = 'Increase exploration'
        elif ep > 0.3 and domain_frac > 0.3:
            diagnosis['probable_cause'] = 'Idea domains formed but not collapsed'
            diagnosis['intervention'] = 'Add forcing function (publishing trigger, collaborator feedback)'
        elif synthesis_prob > 0.7 and label_entropy > 1.6:
            diagnosis['probable_cause'] = 'Hidden synthesis - insights internalized but not externalized'
            diagnosis['intervention'] = 'Check for note-taking or unseen outputs'
        elif integration < 0.5:
            diagnosis['probable_cause'] = 'Low integration potential'
            diagnosis['intervention'] = 'Increase integration (link ideas more tightly)'
        else:
            diagnosis['probable_cause'] = 'Unknown - multiple factors'
            diagnosis['intervention'] = 'Review all metrics'
        
        diagnoses.append(diagnosis)
        
        print(f"\n  {diagnosis['date']}:")
        print(f"    Cause: {diagnosis['probable_cause']}")
        print(f"    Intervention: {diagnosis['intervention']}")
        print(f"    Blind Spot Score: {diagnosis['blind_spot_score']:.4f}")
    
    return diagnoses


def calculate_blind_spot_metrics(iv_df):
    """
    Calculate blind spot metrics:
    - Blind Spot Rate: #false positives / #predicted synthesis
    - Predictive Precision: TP / (TP + FP)
    - Predictive Recall: TP / (TP + FN)
    - Predictive Drift: avg(Δt between predicted & actual synthesis)
    """
    print("\n📊 Calculating Blind Spot Metrics...")
    
    # Get predictions and actuals
    if 'synthesis_prob_learned' in iv_df.columns:
        pred_prob = iv_df['synthesis_prob_learned']
    elif 'synthesis_probability' in iv_df.columns:
        pred_prob = iv_df['synthesis_probability']
    else:
        return {}
    
    actual_synthesis = iv_df['synthesis'] if 'synthesis' in iv_df.columns else (iv_df['posts'] > 0).astype(int)
    pred_synthesis = (pred_prob > 0.5).astype(int)
    
    # Calculate confusion matrix
    tp = ((pred_synthesis == 1) & (actual_synthesis == 1)).sum()
    fp = ((pred_synthesis == 1) & (actual_synthesis == 0)).sum()
    fn = ((pred_synthesis == 0) & (actual_synthesis == 1)).sum()
    tn = ((pred_synthesis == 0) & (actual_synthesis == 0)).sum()
    
    # Metrics
    blind_spot_rate = fp / (pred_synthesis.sum() + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # Predictive drift: temporal misalignment
    pred_dates = iv_df[pred_synthesis == 1]['date'].values
    actual_dates = iv_df[actual_synthesis == 1]['date'].values
    
    if len(pred_dates) > 0 and len(actual_dates) > 0:
        # Simple drift: average difference in closest matches
        drifts = []
        for actual_date in actual_dates:
            closest_pred = pred_dates[np.argmin(np.abs(pred_dates - actual_date))]
            drift = np.abs((actual_date - closest_pred).days) if hasattr(actual_date - closest_pred, 'days') else 0
            drifts.append(drift)
        avg_drift = np.mean(drifts) if drifts else 0
    else:
        avg_drift = 0
    
    metrics = {
        'blind_spot_rate': blind_spot_rate,
        'precision': precision,
        'recall': recall,
        'predictive_drift': avg_drift,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }
    
    print(f"  Blind Spot Rate: {blind_spot_rate:.2%} (#false positives / #predicted synthesis)")
    print(f"  Precision: {precision:.4f} (TP / (TP + FP))")
    print(f"  Recall: {recall:.4f} (TP / (TP + FN))")
    print(f"  Predictive Drift: {avg_drift:.2f} days (avg temporal misalignment)")
    
    return metrics


def update_synthesis_model(iv_df):
    """
    Update synthesis forecast model with suppression index:
    P_t' = P_t × (1 - SI_t)
    """
    print("\n🔄 Updating Synthesis Model with Suppression Index...")
    
    iv_df = iv_df.copy()
    
    # Get original predictions
    if 'synthesis_prob_learned' in iv_df.columns:
        original_pred = iv_df['synthesis_prob_learned']
    elif 'synthesis_probability' in iv_df.columns:
        original_pred = iv_df['synthesis_probability']
    else:
        print("  ⚠️  No predictions to update")
        return iv_df
    
    # Get suppression index
    si = iv_df['suppression_index'].values if 'suppression_index' in iv_df.columns else np.zeros(len(iv_df))
    
    # Update predictions
    adjusted_pred = original_pred * (1 - si)
    iv_df['synthesis_prob_adjusted'] = np.clip(adjusted_pred, 0, 1)
    
    print(f"  Average suppression index: {si.mean():.4f}")
    print(f"  Max suppression index: {si.max():.4f}")
    print(f"  Average adjustment: {(adjusted_pred - original_pred).mean():.4f}")
    
    return iv_df


def create_blind_spot_visualizations(iv_df, output_dir="."):
    """Create blind spot visualization plots."""
    print("\n📊 Creating Blind Spot Visualizations...")
    
    iv_df = iv_df.copy()
    iv_df = iv_df.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(iv_df['date'])
    
    # Get synthesis probabilities
    if 'synthesis_prob_adjusted' in iv_df.columns:
        pred_prob = iv_df['synthesis_prob_adjusted']
    elif 'synthesis_prob_learned' in iv_df.columns:
        pred_prob = iv_df['synthesis_prob_learned']
    elif 'synthesis_probability' in iv_df.columns:
        pred_prob = iv_df['synthesis_probability']
    else:
        print("  ⚠️  No predictions to plot")
        return
    
    actual_synthesis = iv_df['synthesis'] if 'synthesis' in iv_df.columns else (iv_df['posts'] > 0).astype(int)
    
    # Plot 1: Predicted vs Actual Synthesis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color by actual output
    colors = ['green' if s else 'red' for s in actual_synthesis]
    sizes = 100 + iv_df['posts'] * 50 if 'posts' in iv_df.columns else 100
    
    scatter = ax.scatter(dates, pred_prob, s=sizes, c=colors, alpha=0.7,
                        edgecolors='black', linewidths=1.5)
    
    # Mark blind spots
    blind_spots = iv_df['is_blind_spot'] if 'is_blind_spot' in iv_df.columns else pd.Series([False] * len(iv_df))
    if blind_spots.any():
        ax.scatter(dates[blind_spots], pred_prob[blind_spots],
                  s=200, marker='X', c='orange', label='Blind Spots',
                  edgecolors='black', linewidths=2, zorder=5)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Synthesis Probability', fontsize=11, fontweight='bold')
    ax.set_title('Predicted vs Actual Synthesis\n(Red = No Output, Green = Output, X = Blind Spot)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot1_file = Path(output_dir) / "blind_spots_predicted_vs_actual.png"
    plt.savefig(plot1_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved predicted vs actual plot to: {plot1_file}")
    plt.close()
    
    # Plot 2: Blind Spot Heatmap (Entropy vs Novelty)
    if len(iv_df[iv_df['is_blind_spot']]) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        entropy = iv_df['breadth'].values if 'breadth' in iv_df.columns else iv_df['entropy_comm'].values
        novelty = iv_df['novelty'].values if 'novelty' in iv_df.columns else iv_df['outlier_fraction'].values
        blind_scores = iv_df['blind_spot_score'].values if 'blind_spot_score' in iv_df.columns else np.zeros(len(iv_df))
        
        scatter = ax.scatter(entropy, novelty, s=200, c=blind_scores,
                            cmap='Reds', alpha=0.7,
                            edgecolors='black', linewidths=1.5,
                            vmin=0, vmax=blind_scores.max() if blind_scores.max() > 0 else 1)
        
        # Mark blind spots
        if 'is_blind_spot' in iv_df.columns:
            blind_mask = iv_df['is_blind_spot']
            ax.scatter(entropy[blind_mask], novelty[blind_mask],
                      s=300, marker='X', c='orange', label='Blind Spots',
                      edgecolors='black', linewidths=2, zorder=5)
        
        # Sweet spot zone
        sweet_spot_x = [0.85, 0.95, 0.95, 0.85, 0.85]
        sweet_spot_y = [0.10, 0.10, 0.25, 0.25, 0.10]
        ax.plot(sweet_spot_x, sweet_spot_y, 'g--', linewidth=2, alpha=0.5, label='Sweet Spot Zone')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Blind Spot Score', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Entropy (Temperature)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Novelty (Outlier Fraction)', fontsize=11, fontweight='bold')
        ax.set_title('Blind Spot Heatmap\nEntropy vs Novelty (colored by Blind Spot Score)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plot2_file = Path(output_dir) / "blind_spots_heatmap.png"
        plt.savefig(plot2_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved heatmap to: {plot2_file}")
        plt.close()
    
    # Plot 3: Phase Drift Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Expected phase transitions (eruption → synthesis)
    eruption_dates = dates[iv_df['eruption_detected']] if 'eruption_detected' in iv_df.columns else pd.Series([])
    synthesis_dates = dates[actual_synthesis == 1]
    
    # Plot expected (eruption + 1-2 days)
    if len(eruption_dates) > 0:
        expected_synthesis = [d + pd.Timedelta(days=1) for d in eruption_dates] + \
                            [d + pd.Timedelta(days=2) for d in eruption_dates]
        expected_synthesis = pd.Series(expected_synthesis).unique()
        
        ax.scatter(expected_synthesis, [0.5] * len(expected_synthesis),
                  s=200, marker='^', c='blue', label='Expected Synthesis',
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # Plot actual synthesis
    if len(synthesis_dates) > 0:
        ax.scatter(synthesis_dates, [0.6] * len(synthesis_dates),
                  s=200, marker='*', c='green', label='Actual Synthesis',
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # Mark blind spots
    if 'is_blind_spot' in iv_df.columns and iv_df['is_blind_spot'].any():
        blind_spot_dates = dates[iv_df['is_blind_spot']]
        ax.scatter(blind_spot_dates, [0.4] * len(blind_spot_dates),
                  s=200, marker='X', c='red', label='Blind Spots',
                  edgecolors='black', linewidths=2, zorder=5)
    
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Phase', fontsize=11, fontweight='bold')
    ax.set_title('Phase Drift Chart\nExpected vs Actual Synthesis', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(0, 1)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot3_file = Path(output_dir) / "phase_drift_chart.png"
    plt.savefig(plot3_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved phase drift chart to: {plot3_file}")
    plt.close()


def generate_blind_spot_report(iv_df, metrics, diagnoses, cluster_analysis, output_file="blind_spot_report.txt"):
    """Generate comprehensive blind spot report."""
    print("\n📝 Generating Blind Spot Report...")
    
    report = []
    report.append("=" * 80)
    report.append("🧭 BLIND SPOT ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics
    report.append("=" * 80)
    report.append("📊 SUMMARY STATISTICS")
    report.append("=" * 80)
    blind_spots = iv_df['is_blind_spot'].sum() if 'is_blind_spot' in iv_df.columns else 0
    report.append(f"Total blind spots: {blind_spots}/{len(iv_df)} ({blind_spots/len(iv_df):.1%})")
    report.append("")
    
    if metrics:
        report.append("Metrics:")
        report.append(f"  Blind Spot Rate: {metrics['blind_spot_rate']:.2%}")
        report.append(f"  Precision: {metrics['precision']:.4f}")
        report.append(f"  Recall: {metrics['recall']:.4f}")
        report.append(f"  Predictive Drift: {metrics['predictive_drift']:.2f} days")
        report.append("")
    
    # Diagnoses
    if diagnoses:
        report.append("=" * 80)
        report.append("🔬 BLIND SPOT DIAGNOSES")
        report.append("=" * 80)
        for diag in diagnoses:
            report.append(f"\n{diag['date']}:")
            report.append(f"  Cause: {diag['probable_cause']}")
            report.append(f"  Intervention: {diag['intervention']}")
            report.append(f"  Blind Spot Score: {diag['blind_spot_score']:.4f}")
        report.append("")
    
    # Cluster analysis
    if cluster_analysis:
        report.append("=" * 80)
        report.append("🔬 CLUSTER ANALYSIS")
        report.append("=" * 80)
        for cluster in cluster_analysis:
            report.append(f"\nCluster {cluster['cluster']}: {cluster['count']} blind spots")
            report.append(f"  Avg Entropy: {cluster['avg_entropy']:.4f}")
            report.append(f"  Avg Novelty: {cluster['avg_novelty']:.4f}")
            report.append(f"  Avg Exploration Pressure: {cluster['avg_ep']:.4f}")
            report.append(f"  Avg IV: {cluster['avg_iv']:.4f}")
        report.append("")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved report to: {output_file}")
    
    return report_text


def main():
    iv_file = "iv_metrics_calibrated.csv"
    sim_file = "sim_reality_merged.csv"
    output_dir = Path(".")
    
    print("=" * 80)
    print("🧭 Step 7: Blind Spot Analysis - Expected Insights That Didn't Occur")
    print("=" * 80)
    
    # Load data
    print("\n📥 Loading data...")
    iv_df = pd.read_csv(iv_file)
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    
    # Try to merge simulation data
    if Path(sim_file).exists():
        sim_df = pd.read_csv(sim_file)
        sim_df['date'] = pd.to_datetime(sim_df['date'])
        sim_cols = ['domain_frac', 'label_entropy', 'mixing_time']
        for col in sim_cols:
            if col in sim_df.columns:
                merged = sim_df[['date', col]].merge(iv_df[['date']], on='date', how='right')
                iv_df[col] = merged[col].values
    
    print(f"  Loaded metrics for {len(iv_df)} days")
    
    # Identify blind spots
    iv_df = identify_blind_spots(iv_df, threshold=0.5)
    
    # Calculate blind spot scores
    iv_df = calculate_blind_spot_scores(iv_df)
    
    # Cluster blind spots
    iv_df, cluster_analysis = cluster_blind_spots(iv_df)
    
    # Diagnose causes
    diagnoses = diagnose_blind_spot_causes(iv_df)
    
    # Calculate metrics
    metrics = calculate_blind_spot_metrics(iv_df)
    
    # Update model
    iv_df = update_synthesis_model(iv_df)
    
    # Create visualizations
    try:
        create_blind_spot_visualizations(iv_df, output_dir)
    except Exception as e:
        print(f"\n⚠️  Error creating plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Save enhanced data
    enhanced_file = "iv_metrics_blind_spots.csv"
    iv_df.to_csv(enhanced_file, index=False)
    print(f"\n✓ Saved enhanced metrics to: {enhanced_file}")
    
    # Generate report
    report_text = generate_blind_spot_report(iv_df, metrics, diagnoses, cluster_analysis)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 BLIND SPOT SUMMARY")
    print("=" * 80)
    
    blind_spots = iv_df[iv_df['is_blind_spot']] if 'is_blind_spot' in iv_df.columns else pd.DataFrame()
    
    if len(blind_spots) > 0:
        print(f"\n{'Date':<12} {'Prob':<8} {'Actual':<8} {'Blind Score':<12} {'Cause'}")
        print("-" * 80)
        for _, row in blind_spots.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date']
            prob = row.get('synthesis_prob_learned', row.get('synthesis_probability', 0))
            actual = 'Yes' if row.get('synthesis', row.get('posts', 0) > 0) else 'No'
            score = row.get('blind_spot_score', 0)
            cause = diagnoses[0]['probable_cause'] if diagnoses else 'Unknown'
            if len(diagnoses) > 0:
                diag = next((d for d in diagnoses if d['date'] == date_str), None)
                cause = diag['probable_cause'] if diag else 'Unknown'
            print(f"{date_str:<12} {prob:<8.2%} {actual:<8} {score:<12.4f} {cause[:40]}")
    else:
        print("\n  ✓ No blind spots detected!")
    
    print("\n" + "=" * 80)
    print("✅ Blind spot analysis complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {enhanced_file} - Enhanced metrics with blind spot analysis")
    print(f"  - blind_spots_predicted_vs_actual.png - Predicted vs actual plot")
    print(f"  - blind_spots_heatmap.png - Blind spot heatmap")
    print(f"  - phase_drift_chart.png - Phase drift visualization")
    print(f"  - blind_spot_report.txt - Detailed analysis report")


if __name__ == "__main__":
    main()

