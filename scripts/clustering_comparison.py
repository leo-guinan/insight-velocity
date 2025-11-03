#!/usr/bin/env python3
"""
Step 2: Clustering Comparison - Community Detection vs HDBSCAN

Compares two clustering approaches:
1. Graph-based community detection (Louvain) - network structure
2. HDBSCAN clustering - embedding space density
"""
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import json
from pathlib import Path


def compute_entropy(counts, normalize=True):
    """
    Compute normalized entropy: H = -∑p_i * log(p_i) / log(T)
    
    Args:
        counts: dict or Counter of cluster sizes
        normalize: if True, normalize by log(T) where T is number of clusters
    """
    counts = Counter(counts) if not isinstance(counts, Counter) else counts
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    probs = [c / total for c in counts.values() if c > 0]
    if not probs:
        return 0.0
    
    entropy = -sum(p * np.log(p) for p in probs)
    
    if normalize and len(counts) > 1:
        max_entropy = np.log(len(counts))
        if max_entropy > 0:
            entropy = entropy / max_entropy
    
    return entropy


def community_detection_clustering(nodes_csv, edges_csv, output_csv=None):
    """
    Perform community detection using Louvain algorithm on k-NN graph.
    
    Returns:
        dict with labels, metrics, and community info
    """
    print("=" * 60)
    print("🌐 Community Detection (Network Perspective)")
    print("=" * 60)
    
    # Load nodes and edges
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    
    print(f"Loaded {len(nodes_df)} nodes and {len(edges_df)} edges")
    
    # Build graph
    G = nx.Graph()
    
    # Add nodes
    node_id_map = {node_id: idx for idx, node_id in enumerate(nodes_df['id'])}
    for node_id in nodes_df['id']:
        G.add_node(node_id)
    
    # Add edges with weights
    for _, row in edges_df.iterrows():
        source = row['source']
        target = row['target']
        weight = row['similarity']
        G.add_edge(source, target, weight=weight)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Run Louvain community detection
    # Try to use best_partition from community package, fallback to networkx
    try:
        import community.community_louvain as community_louvain
        partition = community_louvain.best_partition(G, weight='weight')
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
    except ImportError:
        # Fallback to networkx greedy_modularity_communities
        print("Note: python-louvain not found, using networkx greedy_modularity_communities")
        communities_raw = nx.community.greedy_modularity_communities(G, weight='weight')
        partition = {}
        communities = {}
        for comm_id, comm in enumerate(communities_raw):
            communities[comm_id] = list(comm)
            for node in comm:
                partition[node] = comm_id
    
    # Assign labels to nodes
    labels = [partition.get(node_id, -1) for node_id in nodes_df['id']]
    nodes_df['community_label'] = labels
    
    # Compute metrics
    community_sizes = Counter(labels)
    T_d = len(communities)  # Number of communities
    H_in_comm = compute_entropy(community_sizes)
    
    # Remove noise if any (label -1)
    non_noise_sizes = {k: v for k, v in community_sizes.items() if k != -1}
    
    metrics = {
        'T_d': int(T_d),
        'community_sizes': {int(k): int(v) for k, v in community_sizes.items()},
        'non_noise_sizes': {int(k): int(v) for k, v in non_noise_sizes.items()},
        'H_in_comm': float(H_in_comm),
        'num_nodes': int(len(nodes_df)),
        'num_edges': int(len(edges_df)),
    }
    
    print(f"\n📊 Community Detection Metrics:")
    print(f"  Number of communities (T_d): {T_d}")
    print(f"  Entropy (H_in_comm): {H_in_comm:.4f}")
    print(f"\n  Community sizes:")
    for comm_id, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"    Community {comm_id}: {size} nodes")
    
    if output_csv:
        nodes_df.to_csv(output_csv, index=False)
        print(f"\n  Saved labels to: {output_csv}")
    
    return {
        'labels': labels,
        'metrics': metrics,
        'communities': communities,
        'nodes_df': nodes_df
    }


def hdbscan_clustering(items_csv, output_csv=None, min_cluster_size=2):
    """
    Perform HDBSCAN clustering on TF-IDF embeddings.
    
    Returns:
        dict with labels, metrics, and cluster info
    """
    print("\n" + "=" * 60)
    print("🔢 HDBSCAN Clustering (Embedding Perspective)")
    print("=" * 60)
    
    try:
        import hdbscan
    except ImportError:
        raise ImportError(
            "HDBSCAN not installed. Install with: pip install hdbscan"
        )
    
    # Load items
    items_df = pd.read_csv(items_csv)
    print(f"Loaded {len(items_df)} items")
    
    if not {'id', 'text'}.issubset(items_df.columns):
        raise ValueError("Items CSV must contain 'id' and 'text' columns")
    
    # Create TF-IDF embeddings (same as in k-NN pipeline)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = tfidf.fit_transform(items_df['text'].astype(str).values)
    
    print(f"Created TF-IDF embeddings: {X.shape[0]} items, {X.shape[1]} features")
    
    # Run HDBSCAN
    # Use UMAP for dimensionality reduction if dataset is large
    if X.shape[1] > 50:
        print("  Applying UMAP dimensionality reduction...")
        try:
            import umap
            reducer = umap.UMAP(n_components=50, random_state=42)
            X_reduced = reducer.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
            print(f"  Reduced to {X_reduced.shape[1]} dimensions")
        except ImportError:
            print("  Warning: UMAP not installed, using raw TF-IDF (may be slow)")
            X_reduced = X.toarray() if hasattr(X, 'toarray') else X
    else:
        X_reduced = X.toarray() if hasattr(X, 'toarray') else X
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
    labels = clusterer.fit_predict(X_reduced)
    probabilities = clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
    
    items_df['hdbscan_label'] = labels
    if probabilities is not None:
        items_df['hdbscan_probability'] = probabilities
    
    # Compute metrics
    cluster_sizes = Counter(labels)
    noise_count = cluster_sizes.get(-1, 0)
    non_noise_clusters = {k: v for k, v in cluster_sizes.items() if k != -1}
    
    T_topics = len(non_noise_clusters)  # Number of non-noise clusters
    H_in_hdb = compute_entropy(non_noise_clusters)  # Entropy excluding noise
    outlier_fraction = noise_count / len(items_df) if len(items_df) > 0 else 0.0
    
    metrics = {
        'T_topics': int(T_topics),
        'cluster_sizes': {int(k): int(v) for k, v in cluster_sizes.items()},
        'non_noise_sizes': {int(k): int(v) for k, v in non_noise_clusters.items()},
        'H_in_hdb': float(H_in_hdb),
        'outlier_fraction': float(outlier_fraction),
        'noise_count': int(noise_count),
        'num_items': int(len(items_df)),
    }
    
    print(f"\n📊 HDBSCAN Metrics:")
    print(f"  Number of clusters (non-noise): {T_topics}")
    print(f"  Entropy (H_in_hdb): {H_in_hdb:.4f}")
    print(f"  Outlier fraction: {outlier_fraction:.4f} ({noise_count} outliers)")
    print(f"\n  Cluster sizes:")
    for cluster_id, size in sorted(non_noise_clusters.items(), key=lambda x: x[1], reverse=True):
        print(f"    Cluster {cluster_id}: {size} items")
    if noise_count > 0:
        print(f"    Noise: {noise_count} items")
    
    if output_csv:
        items_df.to_csv(output_csv, index=False)
        print(f"\n  Saved labels to: {output_csv}")
    
    return {
        'labels': labels,
        'probabilities': probabilities,
        'metrics': metrics,
        'items_df': items_df
    }


def compare_clustering(comm_results, hdbscan_results):
    """
    Compare the two clustering approaches and generate summary.
    """
    print("\n" + "=" * 60)
    print("⚖️  Clustering Comparison")
    print("=" * 60)
    
    comm_metrics = comm_results['metrics']
    hdb_metrics = hdbscan_results['metrics']
    
    print("\n📊 Comparison Summary:")
    print(f"\n  Community Detection (Graph-based):")
    print(f"    Communities: {comm_metrics['T_d']}")
    print(f"    Entropy: {comm_metrics['H_in_comm']:.4f}")
    print(f"    Basis: Graph edges (similarity network)")
    
    print(f"\n  HDBSCAN (Embedding-based):")
    print(f"    Clusters: {hdb_metrics['T_topics']}")
    print(f"    Entropy: {hdb_metrics['H_in_hdb']:.4f}")
    print(f"    Outlier fraction: {hdb_metrics['outlier_fraction']:.4f}")
    print(f"    Basis: Embedding space density")
    
    print(f"\n🔍 Interpretation:")
    
    # High entropy = diverse exploration
    comm_high_entropy = comm_metrics['H_in_comm'] > 0.7
    hdb_high_entropy = hdb_metrics['H_in_hdb'] > 0.7
    
    if comm_high_entropy:
        print(f"  → Community Detection: High entropy suggests diverse idea exploration")
    else:
        print(f"  → Community Detection: Lower entropy suggests focused clusters")
    
    if hdb_high_entropy:
        print(f"  → HDBSCAN: High entropy suggests diverse exploration")
    else:
        print(f"  → HDBSCAN: Lower entropy suggests tight thematic focus")
    
    if hdb_metrics['outlier_fraction'] > 0.3:
        print(f"  → HDBSCAN: High outlier rate ({hdb_metrics['outlier_fraction']:.1%}) suggests")
        print(f"    chaotic exploration or potential new directions")
    elif hdb_metrics['outlier_fraction'] > 0.1:
        print(f"  → HDBSCAN: Moderate outlier rate suggests some novel/experimental ideas")
    else:
        print(f"  → HDBSCAN: Low outlier rate suggests mostly coherent clusters")
    
    # Comparison metrics
    comparison = {
        'community_detection': comm_metrics,
        'hdbscan': hdb_metrics,
        'entropy_difference': float(abs(comm_metrics['H_in_comm'] - hdb_metrics['H_in_hdb'])),
        'cluster_count_difference': int(abs(comm_metrics['T_d'] - hdb_metrics['T_topics'])),
    }
    
    return comparison


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare clustering methods: Community Detection vs HDBSCAN')
    parser.add_argument('--nodes', default='knn_nodes.csv', help='Input nodes CSV')
    parser.add_argument('--edges', default='knn_edges.csv', help='Input edges CSV')
    parser.add_argument('--items', default='items.csv', help='Input items CSV (for HDBSCAN)')
    parser.add_argument('--comm_out', default='community_labels.csv', help='Output CSV for community labels')
    parser.add_argument('--hdbscan_out', default='hdbscan_labels.csv', help='Output CSV for HDBSCAN labels')
    parser.add_argument('--metrics_out', default='clustering_metrics.json', help='Output JSON for metrics')
    parser.add_argument('--min_cluster_size', type=int, default=2, help='HDBSCAN min_cluster_size')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 Step 2: Clustering Comparison")
    print("=" * 60)
    print()
    
    # Community Detection
    comm_results = community_detection_clustering(
        args.nodes, 
        args.edges, 
        output_csv=args.comm_out
    )
    
    # HDBSCAN
    hdbscan_results = hdbscan_clustering(
        args.items,
        output_csv=args.hdbscan_out,
        min_cluster_size=args.min_cluster_size
    )
    
    # Compare
    comparison = compare_clustering(comm_results, hdbscan_results)
    
    # Save metrics
    if args.metrics_out:
        with open(args.metrics_out, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n  Saved metrics to: {args.metrics_out}")
    
    print("\n" + "=" * 60)
    print("✅ Clustering comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

