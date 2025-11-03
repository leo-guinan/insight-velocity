#!/usr/bin/env python3
"""
Calculate Insight Velocity (IV) metrics from clustering results and blog posts.

Components:
- Breadth B_d: avg(normalized entropy_comm, entropy_hdb)
- Novelty N_d: outlier % from HDBSCAN
- Integration Potential I_p: correlation between cluster labels and cross-post references
- Compression Readiness C_r: (1 - N_d) weighted by cluster size variance
- Entropy Reduction ER_d: H_in - H_out (after adding blog posts)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_daily_metrics(summary_file):
    """Load daily metrics from weekly summary."""
    with open(summary_file, 'r') as f:
        report = json.load(f)
    
    daily_metrics = {}
    for daily in report['daily_results']:
        if daily['date'] != 'full_week':
            date = daily['date']
            daily_metrics[date] = {
                'tweet_count': daily['tweet_count'],
                'community_detection': daily['community_detection'],
                'hdbscan': daily['hdbscan'],
            }
    
    return daily_metrics


def calculate_breadth(entropy_comm, entropy_hdb):
    """Breadth B_d = avg(normalized entropy_comm, entropy_hdb)"""
    return (entropy_comm + entropy_hdb) / 2.0


def calculate_novelty(outlier_fraction):
    """Novelty N_d = outlier % from HDBSCAN"""
    return outlier_fraction


def calculate_compression_readiness(novelty, cluster_sizes):
    """Compression Readiness C_r = (1 - N_d) weighted by cluster size variance"""
    if not cluster_sizes or len(cluster_sizes) == 0:
        return 0.0
    
    sizes = list(cluster_sizes.values())
    if len(sizes) == 1:
        variance = 0.0
    else:
        variance = np.var(sizes)
    
    # Normalize variance (max variance would be if all sizes were extreme)
    max_possible_variance = np.var([1] * (len(sizes) - 1) + [sum(sizes)])
    normalized_variance = variance / max_possible_variance if max_possible_variance > 0 else 0.0
    
    compression_readiness = (1.0 - novelty) * normalized_variance
    
    return compression_readiness


def calculate_integration_potential(tweets_df, posts_df, tweets_clusters, posts_clusters):
    """
    Integration Potential I_p = correlation between cluster labels and cross-post references.
    
    We'll measure:
    1. How well tweet clusters map to post topics
    2. Semantic similarity between tweets and posts
    """
    if len(tweets_df) == 0 or len(posts_df) == 0:
        return 0.0
    
    # Combine tweet and post texts
    all_texts = list(tweets_df['text']) + list(posts_df['text'])
    all_labels = list(tweets_clusters) + list(posts_clusters)
    
    if len(set(all_labels)) == 1:  # All same cluster
        return 1.0
    
    # Create TF-IDF embeddings
    try:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        embeddings = tfidf.fit_transform([str(t) for t in all_texts])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Average similarity within same cluster vs different clusters
        same_cluster_similarities = []
        diff_cluster_similarities = []
        
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                sim = similarity_matrix[i, j]
                if all_labels[i] == all_labels[j]:
                    same_cluster_similarities.append(sim)
                else:
                    diff_cluster_similarities.append(sim)
        
        if not same_cluster_similarities or not diff_cluster_similarities:
            return 0.5  # Neutral if no comparisons possible
        
        avg_same = np.mean(same_cluster_similarities)
        avg_diff = np.mean(diff_cluster_similarities)
        
        # Integration potential: how much better same-cluster similarity is
        integration = (avg_same - avg_diff) / (avg_same + avg_diff + 1e-10)
        
        # Normalize to [0, 1]
        integration = max(0.0, min(1.0, (integration + 1.0) / 2.0))
        
        return integration
    except Exception as e:
        print(f"Warning: Could not calculate integration potential: {e}")
        return 0.5  # Neutral default


def calculate_entropy_reduction(tweets_entropy, combined_entropy):
    """
    Entropy Reduction ER_d = H_in - H_out
    
    Where:
    - H_in = entropy before adding posts (just tweets)
    - H_out = entropy after adding posts (tweets + posts)
    """
    return tweets_entropy - combined_entropy


def calculate_insight_velocity(breadth, novelty, integration, compression, entropy_reduction,
                               w_B=0.3, w_N=0.2, w_I=0.2, w_C=0.15, w_E=0.15):
    """
    Calculate Insight Velocity: IV_d ≈ w_B*B_d + w_N*N_d + w_I*I_p + w_C*C_r + w_E*ER_d
    """
    # Normalize components to [0, 1] range
    iv = (w_B * breadth +
          w_N * novelty +
          w_I * integration +
          w_C * compression +
          w_E * max(0, entropy_reduction))  # ER can be negative, clamp for now
    
    return iv


def calculate_daily_iv_metrics(daily_metrics, posts_df, daily_analysis_dir):
    """Calculate IV metrics for each day."""
    iv_results = {}
    
    for date, metrics in daily_metrics.items():
        print(f"\n📅 Calculating IV metrics for {date}")
        
        # Load daily analysis data
        date_dir = Path(daily_analysis_dir) / date
        
        # Try to load items and cluster labels
        items_file = date_dir / "items.csv"
        hdbscan_labels_file = date_dir / "hdbscan_labels.csv"
        
        if not items_file.exists() or not hdbscan_labels_file.exists():
            print(f"  ⚠️  Missing files for {date}, skipping detailed calculations")
            continue
        
        tweets_df = pd.read_csv(items_file)
        hdbscan_df = pd.read_csv(hdbscan_labels_file)
        
        # Merge labels
        tweets_df = tweets_df.merge(
            hdbscan_df[['id', 'hdbscan_label']], 
            on='id', 
            how='left'
        )
        tweets_df['hdbscan_label'] = tweets_df['hdbscan_label'].fillna(-1).astype(int)
        
        # Get metrics
        entropy_comm = metrics['community_detection']['entropy']
        entropy_hdb = metrics['hdbscan']['entropy']
        outlier_fraction = metrics['hdbscan']['outlier_fraction']
        
        # Get cluster sizes from detailed metrics file if available
        metrics_file = date_dir / "clustering_metrics.json"
        cluster_sizes = {}
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    detailed_metrics = json.load(f)
                    cluster_sizes = detailed_metrics.get('hdbscan', {}).get('non_noise_sizes', {})
            except:
                pass
        
        # Fallback: calculate from hdbscan labels if available
        if not cluster_sizes and 'hdbscan_label' in hdbscan_df.columns:
            non_noise_labels = hdbscan_df[hdbscan_df['hdbscan_label'] != -1]['hdbscan_label']
            cluster_sizes = dict(Counter(non_noise_labels))
        
        # Calculate components
        breadth = calculate_breadth(entropy_comm, entropy_hdb)
        novelty = calculate_novelty(outlier_fraction)
        compression = calculate_compression_readiness(novelty, cluster_sizes)
        
        # Integration Potential: Match posts to this day
        day_posts = posts_df[posts_df['date_str'] == date] if 'date_str' in posts_df.columns else pd.DataFrame()
        
        integration = 0.5  # Default neutral
        entropy_reduction = 0.0  # Default
        
        if len(day_posts) > 0:
            print(f"  Found {len(day_posts)} posts for {date}")
            
            # Create post clusters (simplified: one per post for now)
            # In reality, we'd cluster posts too, but for now just match
            post_labels = list(range(-1, -len(day_posts) - 1, -1))  # Unique negative labels
            
            # Calculate integration
            integration = calculate_integration_potential(
                tweets_df,
                day_posts,
                tweets_df['hdbscan_label'].tolist(),
                post_labels
            )
            
            # For entropy reduction, we'd need to combine tweets + posts and recalculate
            # For now, use a simple heuristic: more posts = more synthesis
            if len(day_posts) > 0:
                entropy_reduction = min(0.1, len(day_posts) * 0.03)  # Small boost per post
        
        # Calculate IV
        iv = calculate_insight_velocity(
            breadth, novelty, integration, compression, entropy_reduction
        )
        
        iv_results[date] = {
            'date': date,
            'tweet_count': metrics['tweet_count'],
            'post_count': len(day_posts) if len(day_posts) > 0 else 0,
            'breadth': float(breadth),
            'novelty': float(novelty),
            'integration': float(integration),
            'compression': float(compression),
            'entropy_reduction': float(entropy_reduction),
            'insight_velocity': float(iv),
            'entropy_comm': float(entropy_comm),
            'entropy_hdb': float(entropy_hdb),
            'outlier_fraction': float(outlier_fraction),
        }
        
        print(f"  ✓ Breadth: {breadth:.4f}, Novelty: {novelty:.4f}, IV: {iv:.4f}")
    
    return iv_results


def calculate_rolling_means(iv_results, window=3):
    """Calculate 3-day rolling means for entropy and outliers."""
    dates = sorted(iv_results.keys())
    
    rolling_results = {}
    
    for i in range(len(dates)):
        window_start = max(0, i - window + 1)
        window_dates = dates[window_start:i + 1]
        
        if len(window_dates) == 0:
            continue
        
        window_data = [iv_results[d] for d in window_dates]
        
        rolling_results[dates[i]] = {
            'window_dates': window_dates,
            'rolling_entropy_comm': np.mean([d['entropy_comm'] for d in window_data]),
            'rolling_entropy_hdb': np.mean([d['entropy_hdb'] for d in window_data]),
            'rolling_outlier_fraction': np.mean([d['outlier_fraction'] for d in window_data]),
            'rolling_iv': np.mean([d['insight_velocity'] for d in window_data]),
        }
    
    return rolling_results


def main():
    summary_file = "weekly_analysis/weekly_summary.json"
    posts_file = "ghost_posts_oct24-31.csv"
    daily_analysis_dir = "weekly_analysis"
    
    print("=" * 80)
    print("⚙️  Calculating Insight Velocity (IV) Metrics")
    print("=" * 80)
    
    # Load data
    print("\n📥 Loading data...")
    daily_metrics = load_daily_metrics(summary_file)
    posts_df = pd.read_csv(posts_file)
    
    print(f"  Loaded metrics for {len(daily_metrics)} days")
    print(f"  Loaded {len(posts_df)} blog posts")
    
    # Calculate IV metrics
    print("\n🔢 Calculating IV components...")
    iv_results = calculate_daily_iv_metrics(daily_metrics, posts_df, daily_analysis_dir)
    
    # Calculate rolling means
    print("\n📈 Calculating rolling means (3-day window)...")
    rolling_results = calculate_rolling_means(iv_results, window=3)
    
    # Combine results
    for date in iv_results:
        if date in rolling_results:
            iv_results[date].update(rolling_results[date])
    
    # Save results
    output_file = "iv_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(iv_results, f, indent=2)
    
    print(f"\n✓ Saved IV metrics to: {output_file}")
    
    # Create summary DataFrame
    iv_df = pd.DataFrame([
        {
            'date': r['date'],
            'tweets': r['tweet_count'],
            'posts': r['post_count'],
            'breadth': r['breadth'],
            'novelty': r['novelty'],
            'integration': r['integration'],
            'compression': r['compression'],
            'entropy_reduction': r['entropy_reduction'],
            'insight_velocity': r['insight_velocity'],
            'entropy_comm': r['entropy_comm'],
            'entropy_hdb': r['entropy_hdb'],
            'outlier_fraction': r['outlier_fraction'],
            'rolling_entropy_comm': r.get('rolling_entropy_comm', np.nan),
            'rolling_outlier_fraction': r.get('rolling_outlier_fraction', np.nan),
            'rolling_iv': r.get('rolling_iv', np.nan),
        }
        for r in iv_results.values()
    ]).sort_values('date')
    
    csv_output = "iv_metrics.csv"
    iv_df.to_csv(csv_output, index=False)
    print(f"✓ Saved IV metrics table to: {csv_output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 IV Metrics Summary")
    print("=" * 80)
    print(f"\n{'Date':<12} {'Tweets':<8} {'Posts':<8} {'Breadth':<10} {'Novelty':<10} {'IV':<10}")
    print("-" * 80)
    
    for _, row in iv_df.iterrows():
        print(f"{row['date']:<12} {int(row['tweets']):<8} {int(row['posts']):<8} "
              f"{row['breadth']:<10.4f} {row['novelty']:<10.4f} {row['insight_velocity']:<10.4f}")
    
    print("\n" + "=" * 80)
    print("✅ IV calculation complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Visualize phase plot (entropy vs outlier %)")
    print("  2. Plot rolling trends")
    print("  3. Map breakthrough days")


if __name__ == "__main__":
    main()

